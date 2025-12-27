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

# --- CONFIGURATION ---
PICTURES_DIR = "pictures"
OUTPUT_CSV = "photo_classifications_llava.csv"
MODEL = "llava:7b"
CATEGORIES = ["martial arts", "soccer", "family", "panorama", "other"]

# Concurrency settings (optimized for M4 Mac with Ollama)
CONCURRENT_REQUESTS = 8
MAX_RETRIES = 3

# Semaphore for rate limiting
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

# Category keywords for mapping descriptions to categories
# Keywords are checked in order - more specific categories first to avoid misclassification
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
            # Load and prepare image
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.thumbnail((672, 672))  # LLaVA native resolution
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                image_bytes = img_byte_arr.getvalue()

            # Natural language prompt - ask for classification in conversational style
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

            # Call LLaVA via Ollama (async)
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

            # Debug: Log raw response for troubleshooting
            # Uncomment the next line to see what LLaVA actually returns
            # print(f"\n[RESPONSE]\n{full_response}\n[END]\n")

            # Extract category from the response
            # LLaVA should start with "Category: X", but we'll use a hybrid approach:
            # 1. Try to extract category from "Category: " line
            # 2. Check if category name appears in first 150 chars
            # 3. Fall back to keyword matching if needed

            response_lower = full_response.lower()
            category = "other"
            confidence = "medium"
            category_found = False

            # First, try to extract from "Category: X" format
            if response_lower.startswith("category:"):
                # Extract text after "category:"
                category_line = response_lower.split('\n')[0]  # Get first line
                category_text = category_line.replace("category:", "").strip()

                # Check if it matches any of our categories
                for cat in CATEGORIES:
                    if cat in category_text:
                        category = cat
                        confidence = "high"
                        category_found = True
                        break

            # Second attempt: Check if category name appears early in response
            if not category_found:
                response_start = response_lower[:150]
                for cat in CATEGORIES:
                    if cat == "other":
                        continue  # Skip "other" in this check to prioritize specific categories
                    if cat in response_start:
                        category = cat
                        confidence = "high"
                        category_found = True
                        break

            # Fallback: Use keyword matching on the full response if no category found at start
            if not category_found:
                for cat, keywords in CATEGORY_KEYWORDS.items():
                    if cat == "other":
                        continue

                    for kw in keywords:
                        # Use word boundary matching to avoid false positives
                        pattern = r'\b' + re.escape(kw) + r'\b'

                        if re.search(pattern, response_lower):
                            # Find the position for context checking
                            match = re.search(pattern, response_lower)
                            kw_index = match.start()

                            # Check if it's in a negative context (within 25 chars before keyword)
                            context_before = response_lower[max(0, kw_index-25):kw_index]

                            # Negation patterns to check
                            negations = ["no ", "not ", "without ", "doesn't ", "don't ", "isn't ", "aren't ", "lack of "]
                            is_negated = any(neg in context_before for neg in negations)

                            if not is_negated:
                                # Found a positive match!
                                category = cat
                                confidence = "medium"  # Medium confidence for keyword fallback
                                category_found = True
                                break

                    if category_found:
                        break

            # If still no category matched, keep as "other" with low confidence
            if not category_found:
                confidence = "low"

            return {
                'llava_category': category,
                'llava_confidence': confidence,
                'llava_reasoning': full_response,  # Store the full response as reasoning
                'validated': False,
                'manual_label': '',
                'error': ''
            }

        except Exception as e:
            error_msg = str(e)
            # Retry on connection errors
            if retry_count < MAX_RETRIES and ("EOF" in error_msg or "connection" in error_msg.lower()):
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
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
    
    # Get all image files
    print(f"\n📂 Scanning {PICTURES_DIR} for images...")
    image_files = get_image_files(PICTURES_DIR)
    
    if limit:
        image_files = image_files[:limit]
        print(f"⚠️  Limiting to first {limit} images for testing")
    
    print(f"Found {len(image_files)} images to classify")
    
    # Load existing CSV if it exists (for resuming)
    if os.path.exists(OUTPUT_CSV):
        print(f"\n📄 Found existing CSV, will resume from where we left off...")
        df = pd.read_csv(OUTPUT_CSV)
        # Convert to dict for faster lookup
        processed_files = set(df['filename'].tolist())
    else:
        df = pd.DataFrame(columns=['filename', 'file_path', 'llava_category', 'llava_confidence', 
                                   'llava_reasoning', 'validated', 'manual_label', 'error'])
        processed_files = set()
    
    # Filter out already processed files
    files_to_process = [f for f in image_files if f.name not in processed_files]
    
    if len(files_to_process) == 0:
        print("\n✅ All images already classified!")
        return
    
    print(f"📊 {len(processed_files)} already processed, {len(files_to_process)} remaining")
    print(f"\n🚀 Starting classification with {CONCURRENT_REQUESTS} concurrent requests...\n")
    
    # Create tasks
    tasks_with_files = []
    for image_file in files_to_process:
        tasks_with_files.append((image_file, classify_with_llava(client, str(image_file))))
    
    # Process all tasks with progress bar
    results = []
    
    try:
        with tqdm(total=len(tasks_with_files), desc="Classifying") as pbar:
            # Process in chunks for better progress updates and checkpoint saving
            chunk_size = 20
            for i in range(0, len(tasks_with_files), chunk_size):
                chunk = tasks_with_files[i:i + chunk_size]
                
                # Gather results for this chunk
                chunk_results = await asyncio.gather(*[task for _, task in chunk])
                
                # Pair results with files
                for (image_file, _), result in zip(chunk, chunk_results):
                    row = {
                        'filename': image_file.name,
                        'file_path': str(image_file.absolute()),
                        **result
                    }
                    results.append(row)
                    pbar.update(1)
                
                # Checkpoint save after each chunk
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
    
    # Final save
    final_df = pd.DataFrame(results)
    updated_df = pd.concat([df, final_df], ignore_index=True)
    updated_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Classification complete!")
    print(f"📊 Results saved to: {OUTPUT_CSV}")
    
    # Summary statistics
    category_counts = updated_df['llava_category'].value_counts()
    print(f"\n📈 Classification Summary:")
    for category, count in category_counts.items():
        percentage = (count / len(updated_df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Confidence breakdown
    confidence_counts = updated_df['llava_confidence'].value_counts()
    print(f"\n🎯 Confidence Breakdown:")
    for conf, count in confidence_counts.items():
        if conf:  # Skip empty values
            percentage = (count / len(updated_df)) * 100
            print(f"  {conf}: {count} ({percentage:.1f}%)")
    
    print(f"\n👉 Next step: Run label_app.py to validate classifications")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify photos using LLaVA 7B')
    parser.add_argument('--limit', type=int, help='Limit number of images to process (for testing)')
    args = parser.parse_args()
    
    asyncio.run(main(limit=args.limit))
