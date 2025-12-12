#!/bin/bash

# Script to concatenate all .txt files in /training subdirectory
# Output file: concat_training.txt

# Check if the training directory exists
if [ ! -d "./training" ]; then
    echo "Error: 'training' subdirectory not found"
    exit 1
fi

# Check if there are any .txt files in the training directory
if ! ls ./training/*.txt 1> /dev/null 2>&1; then
    echo "Error: No .txt files found in 'training' subdirectory"
    exit 1
fi

# Remove output file if it already exists
[ -f ./concat_training.txt ] && rm ../concat_training.txt

# Concatenate all .txt files with a newline between each
> ./concat_training.txt  # Create/clear the output file

for file in ./training/*.txt; do
    cat "$file" >> ./concat_training.txt
    echo "" >> ./concat_training.txt  # Add a newline after each file
done

echo "Successfully concatenated all .txt files from 'training/' into concat_training.txt"
echo "Total lines: $(wc -l < ./concat_training.txt)"