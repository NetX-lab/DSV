#!/bin/bash

BASE_DIR="scripts/"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist"
    exit 1
fi

echo "Cleaning experimental results in $BASE_DIR"

for exp_dir in $(find "$BASE_DIR" -type d -name "exp*"); do
    echo "Processing: $exp_dir"
    
    find "$exp_dir" -maxdepth 1 -type f \( -name "*.log" -o -name "*.json" -o -name "*.pdf" \) | while read file; do
        filename=$(basename "$file")
        if [[ "$filename" == *"reference"* ]] || [[ "$filename" == *"ref"* ]]; then
            echo "  Preserving: $filename"
        else
            echo "  Removing: $filename"
            rm -f "$file"
        fi
    done
done

echo "Cleanup completed!"
