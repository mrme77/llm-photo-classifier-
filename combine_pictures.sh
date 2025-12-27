#!/bin/bash

# Script to combine all pictures from classified_photos subfolders into a single pictures folder

# Configuration
SOURCE_DIR="classified_photos"
TARGET_DIR="pictures"
IMAGE_EXTENSIONS=("jpg" "jpeg" "png" "heic" "gif" "webp" "JPG" "JPEG" "PNG" "HEIC" "GIF" "WEBP")

# Create target directory if it doesn't exist
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating $TARGET_DIR directory..."
    mkdir -p "$TARGET_DIR"
fi

# Counter for copied files
total_copied=0
total_skipped=0

echo "Starting to combine pictures from $SOURCE_DIR into $TARGET_DIR..."
echo ""

# Find all image files in subdirectories and copy them
for ext in "${IMAGE_EXTENSIONS[@]}"; do
    while IFS= read -r -d '' file; do
        filename=$(basename "$file")
        target_file="$TARGET_DIR/$filename"

        # Check if file already exists in target
        if [ -f "$target_file" ]; then
            echo "⚠️  Skipping (already exists): $filename"
            ((total_skipped++))
        else
            cp "$file" "$target_file"
            echo "✓ Copied: $filename"
            ((total_copied++))
        fi
    done < <(find "$SOURCE_DIR" -type f -name "*.$ext" -print0)
done

echo ""
echo "=========================================="
echo "Summary:"
echo "✓ Total files copied: $total_copied"
echo "⚠️  Total files skipped: $total_skipped"
echo "📁 All pictures combined in: $TARGET_DIR"
echo "=========================================="
