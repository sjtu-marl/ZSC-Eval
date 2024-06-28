#!/bin/bash

# Check if directory path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Target directory
target_dir="$1"

# Check if target directory exists
if [ ! -d "$target_dir" ]; then
  echo "Directory $target_dir does not exist."
  exit 1
fi

# Count the number of .yml files
yml_count=$(find "$target_dir" -type f -name "*.yml" | wc -l)

# Delete the .yml files
find "$target_dir" -type f -name "*.yml" -exec rm -f {} \;

# Print the number of deleted files
echo "$yml_count .yml files in $target_dir have been deleted."
