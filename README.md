# LLM Photo Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated photo classification system using LLaVA vision-language model to organize personal photo libraries into meaningful categories.

## Overview

This project uses the **LLaVA 7B** vision model (via Ollama) to automatically classify iPhone photos into predefined categories, with a human-in-the-loop validation workflow. It's designed to handle large photo libraries efficiently with async processing and resume capability.

## Categories

The classifier organizes photos into:

- **Martial Arts** - Training sessions, dojos, competitions, martial artists
- **Soccer** - Games, players, soccer fields, matches
- **Family** - Family moments, parents with children, gatherings
- **Panorama** - Scenic landscapes, nature views, tourist locations
- **Other** - Everything else (objects, documents, screenshots, etc.)

## Workflow

```
1. Classify Photos (LLaVA)
   └─> python classify_with_llava.py
       └─> Generates: photo_classifications_llava.csv

2. Human Validation (Gradio UI)
   └─> ./run_ui.sh (or: python label_app.py)
       └─> Updates: photo_classifications_llava.csv

3. Organize Into Folders
   └─> ./organize_photos.sh
       └─> Creates: classified_photos/
           ├── martial_arts/
           ├── soccer/
           ├── family/
           ├── panorama/
           └── other/
```

## Features

### Smart Classification
- **Hybrid approach**: Model generates natural language descriptions, then categories are extracted using:
  1. Direct category extraction from LLaVA response
  2. Word-boundary keyword matching (prevents false positives)
  3. Negation context detection (avoids "no beach" matching "panorama")
- **Confidence scoring**: High/medium/low confidence based on classification method
- **Resume capability**: Picks up where it left off if interrupted

### Async Processing
- 8 concurrent requests (optimized for M4 Mac with Ollama)
- Progress bar with tqdm
- Checkpoint saving every 20 images
- Automatic retry on connection errors (up to 3 retries)

### Human-in-the-Loop
- Gradio web UI for validation
- Shows LLaVA's reasoning for transparency
- Easy correction with dropdown menu
- Tracks validation status

## Installation

### Prerequisites

1. **Ollama** with LLaVA model:
   ```bash
   brew install ollama
   ollama pull llava:7b
   ollama serve
   ```

2. **uv** (Python package manager):
   ```bash
   brew install uv
   ```

3. **Python dependencies**:

   Using uv (recommended):
   ```bash
   uv pip install pandas pillow pillow-heif ollama gradio tqdm
   ```

   Or using pip:
   ```bash
   pip install pandas pillow pillow-heif ollama gradio tqdm
   ```

### Setup

1. Place your photos in the `pictures/` directory
2. Supported formats: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`

## Usage

### 1. Classify Photos

Run the classifier on all images:
```bash
python classify_with_llava.py
```

Or test on a limited set first:
```bash
python classify_with_llava.py --limit 20
```

**Output**: `photo_classifications_llava.csv` with columns:
- `filename`, `file_path`
- `llava_category`, `llava_confidence`
- `llava_reasoning` (model's explanation)
- `validated`, `manual_label`, `error`

### 2. Validate Classifications

Launch the validation UI:
```bash
./run_ui.sh
```

Then open http://127.0.0.1:5500 in your browser.

**Navigation**:
- Review LLaVA's prediction and reasoning
- Accept (✓) or correct the category
- Use arrow keys or buttons to navigate
- Progress is auto-saved

![Validation UI](Screenshot%202025-12-27%20at%209.03.09%20AM.png)

### 3. Organize Photos

Once validated, organize photos into folders:
```bash
./organize_photos.sh
```

This creates `classified_photos/` with subfolders for each category.

**Note**: Uses manual labels when available, falls back to LLaVA predictions for unvalidated images.

## File Structure

```
Iphone16Pictures/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── .gitignore                         # Git exclusions
├── classify_with_llava.py             # Main classification script
├── label_app.py                       # Gradio validation UI
├── organize_photos.sh                 # Bash script to organize by category
├── run_ui.sh                          # Helper to launch validation UI
├── test_image_paths.py                # Debug utility
├── pictures/                          # Input: your photos
├── photo_classifications_llava.csv    # Output: classification results
└── classified_photos/                 # Output: organized by category
    ├── martial_arts/
    ├── soccer/
    ├── family/
    ├── panorama/
    └── other/
```

## How It Works

### Classification Logic

1. **Prompt LLaVA** with category descriptions and instructions
2. **Extract category** using three-tier approach:
   - **Tier 1** (High confidence): "Category: X" at start of response
   - **Tier 2** (High confidence): Category name in first 150 chars
   - **Tier 3** (Medium confidence): Keyword matching with negation detection
3. **Store full response** as reasoning for human review

### Keyword Matching

Uses regex word boundaries (`\b`) to avoid false positives:
- ✅ "beach panorama" → matches "beach" → **panorama**
- ❌ "turtle shell" → does NOT match "tournament" (no substring matching)

### Negation Detection

Checks 25 characters before keywords for negations:
- ❌ "no soccer visible" → "soccer" is negated → **not soccer**
- ✅ "ocean view" → "ocean" not negated → **panorama**

## Troubleshooting

### Verify Setup
```bash
python test_image_paths.py
```
Shows first 3 entries with file existence checks.

### Enable Debug Logging

In `classify_with_llava.py`, uncomment line 89:
```python
print(f"\n[RESPONSE]\n{full_response}\n[END]\n")
```

### Common Issues

**Ollama not running**:
```bash
ollama serve
```

**Wrong classifications**: Check if keywords need expansion in `CATEGORY_KEYWORDS` (lines 30-57)

**CSV parsing errors in organize_photos.sh**: Ensure CSV has proper escaping for commas in reasoning text

## Performance

On M4 Mac with 48GB RAM:
- **Speed**: ~3-5 seconds per image (8 concurrent)
- **Throughput**: ~600-800 images per hour
- **Model**: LLaVA 7B (local, no API costs)

## Customization

### Add New Categories

1. Update `CATEGORIES` list (line 19)
2. Add keywords in `CATEGORY_KEYWORDS` (lines 30-57)
3. Update prompt in `prompt_text` (lines 56-70)
4. Add folder in `organize_photos.sh` (line 27-33)

### Adjust Concurrency

Change `CONCURRENT_REQUESTS` (line 22) based on your hardware:
- M1/M2 Mac: 4-6
- M3/M4 Mac: 8-12
- Cloud GPU: 16-32

## License

MIT License - see [LICENSE](LICENSE) file

## Author

**Pasquale Salomone**

## Acknowledgments

- **LLaVA** - Large Language and Vision Assistant
- **Ollama** - Local LLM runtime
- **Gradio** - Web UI framework
