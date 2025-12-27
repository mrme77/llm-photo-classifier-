#!/usr/bin/env python3
"""
Quick test script to verify image loading works correctly
"""
import os
import pandas as pd

CSV_FILE = 'photo_classifications_llava.csv'

# Load the CSV
df = pd.read_csv(CSV_FILE)

print(f"Total images in CSV: {len(df)}")
print(f"\nFirst 3 entries:")

for i in range(min(3, len(df))):
    row = df.iloc[i]
    filename = row['filename']
    file_path = row['file_path']
    exists = os.path.exists(file_path)
    
    print(f"\n{i+1}. {filename}")
    print(f"   Path: {file_path}")
    print(f"   Exists: {exists}")
    print(f"   Category: {row['llava_category']}")
    print(f"   Confidence: {row['llava_confidence']}")
