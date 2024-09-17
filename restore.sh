#!/bin/bash

# Usage: ./restore.sh

OUTPUT_FOLDER="restored_folder"
mkdir -p "$OUTPUT_FOLDER"

for TAR_FILE in training*.tar.gz; do
    tar -xzvf "$TAR_FILE" -C "$OUTPUT_FOLDER"
done

echo "All files restored to $OUTPUT_FOLDER."
