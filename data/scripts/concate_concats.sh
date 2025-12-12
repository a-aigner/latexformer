#!/bin/bash

# Script to concatenate concat_training.txt and concat_training_inverse.txt
# Output file: training.txt

# Check if both input files exist
if [ ! -f "./concat_training.txt" ]; then
    echo "Error: 'concat_training.txt' not found"
    exit 1
fi

if [ ! -f "./concat_training_inverse.txt" ]; then
    echo "Error: 'concat_training_inverse.txt' not found"
    exit 1
fi

# Concatenate the files with an empty line in between
cat ./concat_training.txt > ./training.txt
echo "" >> ./training.txt
cat ./concat_training_inverse.txt >> ./training.txt

echo "Successfully merged concat_training.txt and concat_training_inverse.txt into training.txt"
echo "Total lines: $(wc -l < ./training.txt)"