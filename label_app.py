import gradio as gr
import pandas as pd
import os
from PIL import Image
import tempfile
import pillow_heif

# Register HEIC support for PIL
pillow_heif.register_heif_opener()

CSV_FILE = 'photo_classifications_llava.csv'  # Read from LLaVA classification file
OUTPUT_CSV = 'photo_classifications_llava.csv'  # Save back to same file

# Updated Categories
LABELS = ["martial arts", "soccer", "family", "panorama", "other"]
EVALUATION_OPTIONS = ["correct", "incorrect"]

def load_data():
    """Loads the CSV and ensures required columns exist."""
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=['filename', 'file_path', 'llava_category', 'llava_confidence',
                                      'llava_reasoning', 'validated', 'manual_label', 'error'])

    df = pd.read_csv(CSV_FILE)

    # Ensure proper column types to avoid dtype warnings
    if 'manual_label' not in df.columns:
        df['manual_label'] = ""
    else:
        # Convert to string type to avoid dtype incompatibility
        df['manual_label'] = df['manual_label'].fillna('').astype(str)

    if 'validated' not in df.columns:
        df['validated'] = False
    else:
        # Ensure boolean type
        df['validated'] = df['validated'].fillna(False).astype(bool)

    return df

def get_next_image_index(df, current_index=None):
    """Returns the next unvalidated image index, or next sequential if all validated."""
    if len(df) == 0:
        return None

    # If no current index, find first unvalidated or start at 0
    if current_index is None:
        unvalidated = df[df['validated'] == False]
        if len(unvalidated) > 0:
            return unvalidated.index[0]
        return 0

    # Find next unvalidated image after current index
    remaining = df.iloc[current_index + 1:]
    unvalidated = remaining[remaining['validated'] == False]
    
    if len(unvalidated) > 0:
        return unvalidated.index[0]
    
    # If no unvalidated remaining, check if we've reached the end
    if current_index + 1 >= len(df):
        return None
    
    # Otherwise return next sequential
    return current_index + 1

def save_label(index, manual_label, history):
    """Updates the dataframe, saves, pushes to history, and gets next image."""
    df = load_data()

    # Update row with manual label and mark as validated
    df.at[index, 'manual_label'] = str(manual_label)
    df.at[index, 'validated'] = True

    # Save to CSV
    df.to_csv(CSV_FILE, index=False)

    # Update history (push current index)
    new_history = history + [index]

    # Get next image (pass current index to get sequential next)
    next_idx = get_next_image_index(df, current_index=index)
    
    # Return all outputs including new history
    # Unpack tuple from get_image_data and append new states
    img_data = get_image_data(next_idx)
    return img_data + (next_idx, new_history)

def undo_last(history):
    """Goes back to the previous image in history."""
    if not history:
        # No history, do nothing but re-return current state or empty
        # For simplicity, just return current state if possible, but we need an index.
        # Let's just reload the data as if starting over if history is empty
        df = load_data()
        idx = get_next_image_index(df)
        img_data = get_image_data(idx)
        return img_data + (idx, history)
    
    # Pop last index
    prev_index = history[-1]
    new_history = history[:-1]
    
    # Reload that image
    img_data = get_image_data(prev_index)
    return img_data + (prev_index, new_history)

def convert_heic_if_needed(image_path):
    """Converts HEIC images to JPEG for browser compatibility."""
    if not image_path.lower().endswith('.heic'):
        return image_path
    
    try:
        # Open HEIC and convert to JPEG
        img = Image.open(image_path)
        
        # Create temp file with .jpg extension
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        
        # Convert and save as JPEG
        img.convert('RGB').save(temp_path, 'JPEG', quality=95)
        return temp_path
    except Exception as e:
        print(f"Error converting HEIC: {e}")
        return image_path

def get_image_data(index):
    """Returns image path and status text for a given index."""
    df = load_data()

    print(f"\n[DEBUG] get_image_data called with index: {index}")
    print(f"[DEBUG] DataFrame has {len(df)} rows")

    if index is None:
        print("[DEBUG] Index is None, returning completion message")
        return (None, "## All images validated!", "",
                gr.update(visible=False), gr.update(visible=False))

    row = df.iloc[index]
    image_path = row['file_path']
    filename = row['filename']
    llava_category = row.get('llava_category', 'unknown')
    llava_confidence = row.get('llava_confidence', 'N/A')
    llava_reasoning = row.get('llava_reasoning', '')
    validated = row.get('validated', False)
    error = row.get('error', '')
    
    print(f"[DEBUG] Row {index} data:")
    print(f"[DEBUG]   filename: {filename}")
    print(f"[DEBUG]   image_path: {image_path}")
    print(f"[DEBUG]   error value: '{error}' (type: {type(error)})")
    print(f"[DEBUG]   error is truthy: {bool(error)}")
    
    # Existing label (if we are revisiting/undoing)
    current_manual_label = row.get('manual_label', "")

    # Handle missing file or errors
    # Fix: pandas reads empty CSV cells as NaN, which is truthy but not a real error
    if pd.notna(error) and str(error).strip():
        print(f"[DEBUG] ERROR FOUND - returning None")
        return (None, f"## Error\n{error}", "",
                gr.update(value="other", visible=True),
                gr.update(visible=True))
    
    if not os.path.exists(image_path):
        print(f"[DEBUG] File not found: {image_path}")
        return (None, f"## Error: File not found\n{image_path}", "",
                gr.update(value="other", visible=True),
                gr.update(visible=True))

    print(f"[DEBUG] File exists: {image_path}")
    
    # Convert HEIC to JPEG if needed for browser display
    display_path = convert_heic_if_needed(image_path)
    print(f"[DEBUG] Display path: {display_path}")
    print(f"[DEBUG] Display path exists: {os.path.exists(display_path)}")
    
    # --- INFO TEXT (Left Column) ---
    validated_count = len(df[df['validated'] == True])
    info_md = f"### 📸 Image {index + 1} of {len(df)}\n\n"
    info_md += f"**Progress:** {validated_count}/{len(df)} validated ({validated_count/len(df)*100:.1f}%)\n\n"
    info_md += f"**File:** `{filename}`\n\n"
    info_md += f"**Path:** `{image_path}`\n\n"
    
    # --- LLAVA PREDICTION (Top Box with Colors) ---
    llava_status_md = ""
    
    if llava_category == 'ERROR':
        llava_status_md = "## <span style='color:red'>❌ Classification Error</span>"
        llava_status_md += f"\n\n{error}"
    else:
        # Color code by confidence
        if llava_confidence == 'high':
            color = 'green'
            icon = '✅'
        elif llava_confidence == 'medium':
            color = 'orange'
            icon = '⚠️'
        else:
            color = 'gray'
            icon = '❓'
        
        llava_status_md = f"## <span style='color:{color}'>{icon} LLaVA Prediction: {llava_category}</span>"
        llava_status_md += f"\n\n**Confidence:** {llava_confidence}"
        llava_status_md += f"\n\n**Reasoning:** {llava_reasoning}"
    
    # Determine default value for dropdown
    # If we have a saved manual label, use that (revisiting/undo logic)
    if pd.notna(current_manual_label) and current_manual_label != "":
        default_label_value = current_manual_label
    else:
        default_label_value = llava_category if llava_category in LABELS else "other"

    print(f"[DEBUG] Returning image path: {display_path}")
    print(f"[DEBUG] Default label: {default_label_value}")
    print(f"[DEBUG] ---")

    return (display_path, info_md, llava_status_md,
            gr.update(value=default_label_value, visible=True),
            gr.update(value="Save & Next", interactive=True, visible=True))

with gr.Blocks(title="Photo Classification Review") as demo:
    gr.Markdown("# Photo Classification Review Tool")
    
    # State tracking
    current_index_state = gr.State()
    history_state = gr.State([]) # Stack of visited indices

    with gr.Row():
        with gr.Column(scale=2):
            image_display = gr.Image(type="filepath", height=600)

        with gr.Column(scale=1):
            # Explanation of LLaVA
            with gr.Accordion("ℹ️ About LLaVA Classification", open=False):
                gr.Markdown("""
                **What is LLaVA?**
                LLaVA 7b is a vision-language model running locally via Ollama.
                
                **How does it work?**
                It analyzes each image and classifies it into one of these categories:
                - **Martial Arts**: Judo, karate, taekwondo, dojos, tournaments
                - **Soccer**: Games, players, balls, soccer fields
                - **Family**: Groups with children, family gatherings, parents with kids
                - **Panorama**: Scenic views, landscapes, nature, tourist attractions
                - **Other**: Anything that doesn't fit the above
                
                **Confidence levels:**
                - <span style='color:green'>✅ **High**</span>: LLaVA is very confident
                - <span style='color:orange'>⚠️ **Medium**</span>: LLaVA is somewhat confident
                - <span style='color:gray'>❓ **Low**</span>: LLaVA is uncertain
                
                **Your job:** Validate or correct LLaVA's predictions.
                """)
                
            # Two Markdown blocks: LLaVA Status (Top) and File Info (Below)
            llava_display = gr.Markdown(elem_id="llava_box")
            info_display = gr.Markdown()
            
            gr.Markdown("---")
            gr.Markdown("### Validate Classification")
            
            label_dropdown = gr.Dropdown(
                choices=LABELS,
                label="Correct Category",
                value="other",
                interactive=True,
                info="Accept LLaVA's prediction or choose the correct category"
            )
            
            with gr.Row():
                undo_btn = gr.Button("↩️ Undo / Previous", variant="secondary")
                submit_btn = gr.Button("Save & Next", variant="primary")

    # Initial Load
    def on_load():
        print("\n[DEBUG] ========== ON_LOAD CALLED ==========")
        df = load_data()
        print(f"[DEBUG] Loaded {len(df)} rows from CSV")
        idx = get_next_image_index(df)
        print(f"[DEBUG] Next image index: {idx}")
        # Return initial data + index + empty history
        data = get_image_data(idx)
        print(f"[DEBUG] Returning {len(data)} items from on_load")
        print(f"[DEBUG] First item (image path): {data[0]}")
        print("[DEBUG] ======================================\n")
        return data + (idx, [])

    demo.load(on_load, outputs=[image_display, info_display, llava_display, label_dropdown, 
                                  submit_btn, current_index_state, history_state])

    # Save Button Click
    submit_btn.click(
        fn=lambda: gr.update(value="Saving...", interactive=False),
        inputs=None,
        outputs=submit_btn
    ).then(
        save_label,
        inputs=[current_index_state, label_dropdown, history_state],
        outputs=[image_display, info_display, llava_display, label_dropdown, 
                 submit_btn, current_index_state, history_state]
    )
    
    # Undo Button Click
    undo_btn.click(
        fn=lambda: gr.update(value="Loading...", interactive=False),
        inputs=None,
        outputs=submit_btn
    ).then(
        undo_last,
        inputs=[history_state],
        outputs=[image_display, info_display, llava_display, label_dropdown, 
                 submit_btn, current_index_state, history_state]
    )

if __name__ == "__main__":
    print(f"Starting Photo Classification Validation Tool")
    print(f"Working with: {CSV_FILE}")
    demo.launch(server_name="127.0.0.1", server_port=5500)
