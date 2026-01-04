#!/bin/sh
set -e # exit on first error

# Run evaluation
python -u src/dtu_mlops_garu/evaluate.py "$@" # expands to passed positional arguments

python -u src/dtu_mlops_garu/visualize.py "$@"

echo "All tasks completed successfully"