#!/bin/bash

# Script to swap Input/Output pairs in concat_training.txt
# Output file: concat_training_inverse.txt

# Check if the input file exists
if [ ! -f "./concat_training.txt" ]; then
    echo "Error: 'concat_training.txt' not found"
    exit 1
fi

# Clear/create output file
> ./concat_training_inverse.txt

# Read the file line by line
while IFS= read -r line1 && IFS= read -r line2 && IFS= read -r line3; do
    # Extract the text after "Input: " and "Output: "
    input_text="${line1#Input: }"
    output_text="${line2#Output: }"
    
    # Write swapped lines to output file
    echo "Input: $output_text" >> ./concat_training_inverse.txt
    echo "Output: $input_text" >> ./concat_training_inverse.txt
    echo "" >> ./concat_training_inverse.txt
done < ./concat_training.txt

echo "Successfully created concat_training_inverse.txt with swapped Input/Output pairs"
echo "Total lines: $(wc -l < ./concat_training_inverse.txt)"