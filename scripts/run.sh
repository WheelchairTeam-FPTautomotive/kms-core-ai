#!/bin/bash
# Evaluator Contract Script for RAG Core

set -e

INPUT_DIR=""
OUTPUT_FILE=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input) INPUT_DIR="$2"; shift ;;
        --output) OUTPUT_FILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: ./scripts/run.sh --input <input_dir> --output <output_file>"
    exit 1
fi

echo "=========================================================="
echo "Starting Core AI Offline Evaluation"
echo "Input Directory: $INPUT_DIR"
echo "Output File: $OUTPUT_FILE"
echo "=========================================================="

export PYTHONPATH="src:${PYTHONPATH}"

# Execute Python pipeline CLI inside the uv-managed environment
uv run python -m pipelines.solve_problem --input "$INPUT_DIR" --output "$OUTPUT_FILE"

echo "Core AI Offline Evaluation completed successfully."
