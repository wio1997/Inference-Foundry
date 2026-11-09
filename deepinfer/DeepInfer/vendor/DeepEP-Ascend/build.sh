#!/bin/bash
# Development build: compile the C++ extension and symlink .so into the package

# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Remove old dist file, build files, and build
rm -rf build dist
rm -rf *.egg-info
python setup.py bdist_wheel

# Return to users' original directory
cd "$original_dir"
