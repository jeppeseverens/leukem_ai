#!/bin/bash

# Script to combine all NN array job results into final CSV files

cd /exports/me-lcco-aml-hpc/Jeppe2/leukem_ai
source venv/bin/activate

echo "Combining NN array job results..."
echo "=================================="

# Set directories
RESULTS_DIR="out/NN_array"
OUTPUT_DIR="out/NN_combined"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Looking for results in: $RESULTS_DIR"
echo "Saving combined results to: $OUTPUT_DIR"

# Combine CV results (standard multi-class strategy for NN)
echo "Combining CV results..."
python python/combine_array_results.py \
    --results_dir "$RESULTS_DIR" \
    --model_type "NN" \
    --fold_type "CV" \
    --multi_type "standard" \
    --output_dir "$OUTPUT_DIR"

# Combine LOSO results (standard multi-class strategy for NN)
echo "Combining LOSO results..."
python python/combine_array_results.py \
    --results_dir "$RESULTS_DIR" \
    --model_type "NN" \
    --fold_type "loso" \
    --multi_type "standard" \
    --output_dir "$OUTPUT_DIR"

echo "Results combination completed!"
echo "Check the output directory: $OUTPUT_DIR"

deactivate 