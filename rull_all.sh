#!/bin/bash

# Define the folder containing audio files
AUDIO_FOLDER="new_audios"

# Loop through all .wav files in the folder
for file in "$AUDIO_FOLDER"/*.mp3; do
    # Check if there are any .wav files
    [ -e "$file" ] || continue

    # Extract filename without extension
    filename_without_ext=$(basename "$file" .mp3)

    echo "Processing: $filename_without_ext"  # Optional logging

    # Call main.py with the filename as an argument
    python main.py "$filename_without_ext"
done