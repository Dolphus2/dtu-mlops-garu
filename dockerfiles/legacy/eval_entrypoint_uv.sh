#!/bin/sh
set -e # exit on first error

# Run evaluation
uv run src/dtu_mlops_garu/evaluate.py "$@" # expands to passed positional arguments

uv run src/dtu_mlops_garu/visualize.py "$@"

echo "All tasks completed successfully"
