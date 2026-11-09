#!/bin/bash
# Build wheel package

# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Remove old dist file, build files, and build wheel
rm -rf build dist
rm -rf *.egg-info
python3 setup.py bdist_wheel

# Return to original directory
cd "$original_dir"
