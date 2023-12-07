#!/bin/bash

# Iterate over all .cpp files in the current directory
for file in *.cpp; do
    # Check if the file exists
    if [ -e "$file" ]; then
        # Copy the file and change the extension to .cu
        cp "$file" "${file}.cu"
        echo "Copied $file to ${file}.cu"
    fi
done

# Iterate over all .c files in the current directory
for file in *.c; do
    # Check if the file exists
    if [ -e "$file" ]; then
        # Copy the file and change the extension to .cu
        cp "$file" "${file}.cu"
        echo "Copied $file to ${file}.cu"
    fi
done
