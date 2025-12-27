#!/bin/bash

CSV_FILE="photo_classifications_llava.csv"
OUTPUT_BASE_DIR="classified_photos"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting photo organization...${NC}"

if [ ! -f "$CSV_FILE" ]; then
    echo -e "${RED}Error: $CSV_FILE not found!${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_BASE_DIR"

mkdir -p "$OUTPUT_BASE_DIR/martial_arts"
mkdir -p "$OUTPUT_BASE_DIR/soccer"
mkdir -p "$OUTPUT_BASE_DIR/family"
mkdir -p "$OUTPUT_BASE_DIR/panorama"
mkdir -p "$OUTPUT_BASE_DIR/uncertain"
mkdir -p "$OUTPUT_BASE_DIR/other"
mkdir -p "$OUTPUT_BASE_DIR/errors"

echo -e "${GREEN}Created category folders in $OUTPUT_BASE_DIR/${NC}"

tail -n +2 "$CSV_FILE" | while IFS=',' read -r filename file_path llava_category llava_confidence llava_reasoning validated manual_label error; do
    file_path=$(echo "$file_path" | sed 's/"//g')
    llava_category=$(echo "$llava_category" | sed 's/"//g')
    manual_label=$(echo "$manual_label" | sed 's/"//g')

    if [ "$validated" = "True" ] && [ -n "$manual_label" ] && [ "$manual_label" != '""' ]; then
        predicted_category="$manual_label"
    else
        predicted_category="$llava_category"
    fi

    if [ ! -f "$file_path" ]; then
        echo -e "${YELLOW}Warning: File not found - $file_path${NC}"
        continue
    fi

    case "$predicted_category" in
        "martial arts")
            dest_dir="$OUTPUT_BASE_DIR/martial_arts"
            ;;
        "soccer")
            dest_dir="$OUTPUT_BASE_DIR/soccer"
            ;;
        "family")
            dest_dir="$OUTPUT_BASE_DIR/family"
            ;;
        "panorama")
            dest_dir="$OUTPUT_BASE_DIR/panorama"
            ;;
        "uncertain")
            dest_dir="$OUTPUT_BASE_DIR/uncertain"
            ;;
        "ERROR"|"TIMEOUT")
            dest_dir="$OUTPUT_BASE_DIR/errors"
            ;;
        *)
            dest_dir="$OUTPUT_BASE_DIR/other"
            ;;
    esac

    cp "$file_path" "$dest_dir/"

    if [ $? -eq 0 ]; then
        echo -e "Copied: $(basename "$file_path") -> $dest_dir"
    else
        echo -e "${RED}Failed to copy: $file_path${NC}"
    fi
done

echo ""
echo -e "${GREEN}=== Organization Complete ===${NC}"
echo -e "${GREEN}Summary:${NC}"
echo "Martial Arts: $(ls -1 "$OUTPUT_BASE_DIR/martial_arts" | wc -l | xargs) files"
echo "Soccer:       $(ls -1 "$OUTPUT_BASE_DIR/soccer" | wc -l | xargs) files"
echo "Family:       $(ls -1 "$OUTPUT_BASE_DIR/family" | wc -l | xargs) files"
echo "Panorama:     $(ls -1 "$OUTPUT_BASE_DIR/panorama" | wc -l | xargs) files"
echo "Uncertain:    $(ls -1 "$OUTPUT_BASE_DIR/uncertain" | wc -l | xargs) files"
echo "Other:        $(ls -1 "$OUTPUT_BASE_DIR/other" | wc -l | xargs) files"
echo "Errors:       $(ls -1 "$OUTPUT_BASE_DIR/errors" | wc -l | xargs) files"
echo ""
echo -e "${GREEN}Photos organized in: $OUTPUT_BASE_DIR/${NC}"
