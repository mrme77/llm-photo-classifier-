import os
import io
import re
import asyncio
import pandas as pd
from pathlib import Path
from PIL import Image
import pillow_heif
from ollama import AsyncClient
from tqdm import tqdm
import argparse

# Register HEIC support
pillow_heif.register_heif_opener()

PICTURES_DIR = "pictures"
OUTPUT_CSV = "photo_classifications_llava.csv"
MODEL = "llava:7b"
CATEGORIES = ["martial arts", "soccer", "family", "panorama", "other"]

CONCURRENT_REQUESTS = 8
MAX_RETRIES = 3

semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

CATEGORY_KEYWORDS = {
    "martial arts": [
        "judo", "jiu-jitsu", "jiujitsu", "bjj", "taekwondo", "karate", "aikido", "kung fu", "mma",
        "martial arts", "dojo", "gi", "belt ceremony", "sparring", "grappling", "combat sport",
        "martial arts tournament", "fighting", "wrestler", "wrestling mat", "martial artist"
    ],
    "soccer": [
        "soccer", "football match", "football game", "soccer field", "soccer match", "soccer player",
        "soccer ball", "soccer pitch", "football pitch", "goal", "goalkeeper", "goalie",
        "soccer uniform", "soccer jersey", "penalty kick", "corner kick", "football player",
        "soccer cleats", "soccer team"
    ],
    "family": [
        "family", "children", "baby", "babies", "parents", "grandparents", "siblings", "birthday party",
        "kids", "toddler", "toddlers", "infant", "relatives", "family gathering", "family photo",
        "mother and child", "father and child", "parent and child", "group of children",
        "family portrait", "playground", "birthday celebration", "family event", "family members"
    ],
    "panorama": [
        "panorama", "panoramic", "landscape", "scenery", "scenic view", "vista", "viewpoint",
        "mountain", "mountains", "ocean", "sea", "lake", "river", "sunset", "sunrise",
        "beach", "coastline", "shore", "nature", "horizon", "skyline", "city view",
        "waterfall", "forest", "woods", "canyon", "valley", "cliff", "peak",
        "natural landscape", "seascape", "countryside", "tourist spot", "landmark",
        "sky", "clouds", "scenic", "beautiful view", "natural beauty", "outdoor scene"
    ],
    "other": []  # fallback
}

print(f"🚀 Starting LLaVA 7B Photo Classification")
print(f"Model: {MODEL}")
print(f"Categories: {', '.join(CATEGORIES)}")
print(f"Concurrency: {CONCURRENT_REQUESTS} requests")


async def classify_with_llava(client, image_path, retry_count=0):
    """Use LLaVA to classify image and generate description."""
    async with semaphore:
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.thumbnail((672, 672))
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                image_bytes = img_byte_arr.getvalue()

            prompt_text = """Look at this image and classify it into ONE of these categories:

Categories:
- martial arts (if showing martial arts training, dojos, competitions, or martial artists)
- soccer (if showing soccer games, players, soccer balls, or soccer fields)
- family (if showing family moments, parents with children, or family gatherings)
- panorama (if showing scenic landscapes, nature views, mountains, oceans, or tourist locations)
- other (if it doesn't fit any category above)

Instructions:
1. Describe what you see in the image (1 sentence)
2. State which category it belongs to using ONLY the category name
3. Explain why in one sentence

Begin your response with "Category: " followed by the category name."""

            response = await client.chat(
                model=MODEL,
                messages=[{
                    'role': 'user',
                    'content': prompt_text,
                    'images': [image_bytes]
                }],
                options={
                    'temperature': 0.1,
                }
            )

            full_response = response['message']['content'].strip()
            response_lower = full_response.lower()
            category = "other"
            confidence = "medium"
            category_found = False

            if response_lower.startswith("category:"):
                category_line = response_lower.split('\n')[0]
                category_text = category_line.replace("category:", "").strip()

                for cat in CATEGORIES:
                    if cat in category_text:
                        category = cat
                        confidence = "high"
                        category_found = True
                        break

            if not category_found:
                response_start = response_lower[:150]
                for cat in CATEGORIES:
                    if cat == "other":
                        continue
                    if cat in response_start:
                        category = cat
                        confidence = "high"
                        category_found = True
                        break

            if not category_found:
                for cat, keywords in CATEGORY_KEYWORDS.items():
                    if cat == "other":
                        continue

                    for kw in keywords:
                        pattern = r'\b' + re.escape(kw) + r'\b'

                        if re.search(pattern, response_lower):
                            match = re.search(pattern, response_lower)
                            kw_index = match.start()
                            context_before = response_lower[max(0, kw_index-25):kw_index]
                            negations = ["no ", "not ", "without ", "doesn't ", "don't ", "isn't ", "aren't ", "lack of "]
                            is_negated = any(neg in context_before for neg in negations)

                            if not is_negated:
                                category = cat
                                confidence = "medium"
                                category_found = True
                                break

                    if category_found:
                        break

            if not category_found:
                confidence = "low"

            return {
                'llava_category': category,
                'llava_confidence': confidence,
                'llava_reasoning': full_response,
                'validated': False,
                'manual_label': '',
                'error': ''
            }

        except Exception as e:
            error_msg = str(e)
            if retry_count < MAX_RETRIES and ("EOF" in error_msg or "connection" in error_msg.lower()):
                await asyncio.sleep(2 ** retry_count)
                return await classify_with_llava(client, image_path, retry_count + 1)
            
            return {
                'llava_category': 'ERROR',
                'llava_confidence': '',
                'llava_reasoning': '',
                'validated': False,
                'manual_label': '',
                'error': error_msg
            }


def get_image_files(directory):
    """Get all image files from directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
    image_files = []
    
    for file in Path(directory).iterdir():
        if file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    return sorted(image_files)


async def main(limit=None):
    """Main async function to process images with LLaVA."""
    client = AsyncClient()

    print(f"\n📂 Scanning {PICTURES_DIR} for images...")
    image_files = get_image_files(PICTURES_DIR)

    if limit:
        image_files = image_files[:limit]
        print(f"⚠️  Limiting to first {limit} images for testing")

    print(f"Found {len(image_files)} images to classify")

    if os.path.exists(OUTPUT_CSV):
        print(f"\n📄 Found existing CSV, will resume from where we left off...")
        df = pd.read_csv(OUTPUT_CSV)
        processed_files = set(df['filename'].tolist())
    else:
        df = pd.DataFrame(columns=['filename', 'file_path', 'llava_category', 'llava_confidence',
                                   'llava_reasoning', 'validated', 'manual_label', 'error'])
        processed_files = set()

    files_to_process = [f for f in image_files if f.name not in processed_files]

    if len(files_to_process) == 0:
        print("\n✅ All images already classified!")
        return

    print(f"📊 {len(processed_files)} already processed, {len(files_to_process)} remaining")
    print(f"\n🚀 Starting classification with {CONCURRENT_REQUESTS} concurrent requests...\n")

    tasks_with_files = []
    for image_file in files_to_process:
        tasks_with_files.append((image_file, classify_with_llava(client, str(image_file))))

    results = []
    
    try:
        with tqdm(total=len(tasks_with_files), desc="Classifying") as pbar:
            chunk_size = 20
            for i in range(0, len(tasks_with_files), chunk_size):
                chunk = tasks_with_files[i:i + chunk_size]

                chunk_results = await asyncio.gather(*[task for _, task in chunk])

                for (image_file, _), result in zip(chunk, chunk_results):
                    row = {
                        'filename': image_file.name,
                        'file_path': str(image_file.absolute()),
                        **result
                    }
                    results.append(row)
                    pbar.update(1)

                chunk_df = pd.DataFrame(results)
                updated_df = pd.concat([df, chunk_df], ignore_index=True)
                updated_df.to_csv(OUTPUT_CSV, index=False)
                
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Saving progress...")
        if results:
            chunk_df = pd.DataFrame(results)
            updated_df = pd.concat([df, chunk_df], ignore_index=True)
            updated_df.to_csv(OUTPUT_CSV, index=False)
        return

    final_df = pd.DataFrame(results)
    updated_df = pd.concat([df, final_df], ignore_index=True)
    updated_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Classification complete!")
    print(f"📊 Results saved to: {OUTPUT_CSV}")

    category_counts = updated_df['llava_category'].value_counts()
    print(f"\n📈 Classification Summary:")
    for category, count in category_counts.items():
        percentage = (count / len(updated_df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")

    confidence_counts = updated_df['llava_confidence'].value_counts()
    print(f"\n🎯 Confidence Breakdown:")
    for conf, count in confidence_counts.items():
        if conf:
            percentage = (count / len(updated_df)) * 100
            print(f"  {conf}: {count} ({percentage:.1f}%)")

    print(f"\n👉 Next step: Run label_app.py to validate classifications")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify photos using LLaVA 7B')
    parser.add_argument('--limit', type=int, help='Limit number of images to process (for testing)')
    args = parser.parse_args()
    
    asyncio.run(main(limit=args.limit))
