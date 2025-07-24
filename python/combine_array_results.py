#!/usr/bin/env python3
"""
Script to combine results from SLURM array jobs into final CSV files.
Each array job produces one CSV file per hyperparameter combination.
This script combines them all into the final result files.
"""

import pandas as pd
import glob
import os
import argparse
from pathlib import Path

def combine_array_results(results_dir, model_type, fold_type, multi_type, output_dir):
    """
    Combine individual array job results into a single CSV file.
    
    Args:
        results_dir: Directory containing individual result files
        model_type: Model type (NN, SVM, XGBOOST)
        fold_type: Fold type (CV, loso)
        multi_type: Multi-class strategy (standard, OvR, OvO)
        output_dir: Directory to save combined results
    """
    
    # Pattern to match individual result files
    pattern = f"{results_dir}/*/{model_type}_inner_cv_{fold_type}_{multi_type}_param_*.csv"
    if fold_type == "loso":
        pattern = f"{results_dir}/*/{model_type}_inner_cv_loso_{multi_type}_param_*.csv"
    
    print(f"Looking for files matching pattern: {pattern}")
    
    # Find all matching files
    result_files = glob.glob(pattern)
    print(f"Found {len(result_files)} result files")
    
    if not result_files:
        print(f"No result files found for {model_type} {fold_type} {multi_type}")
        return
    
    # Read and combine all CSV files
    combined_dfs = []
    for file_path in sorted(result_files):
        print(f"Reading {file_path}")
        try:
            df = pd.read_csv(file_path, index_col=0)  # Assuming first column is index
            combined_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not combined_dfs:
        print("No valid result files found")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    print(f"Combined dataframe shape: {combined_df.shape}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    import datetime
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    if fold_type == "loso":
        output_filename = f"{model_type}_inner_cv_loso_{multi_type}_combined_{time}.csv"
    else:
        output_filename = f"{model_type}_inner_cv_{multi_type}_combined_{time}.csv"
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Save combined results
    combined_df.to_csv(output_path)
    print(f"Combined results saved to: {output_path}")
    print(f"Total rows: {len(combined_df)}")

def main():
    parser = argparse.ArgumentParser(description="Combine SLURM array job results into final CSV files.")
    
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing array job result subdirectories')
    parser.add_argument('--model_type', type=str, required=True,
                       help='Model type (NN, SVM, XGBOOST)')
    parser.add_argument('--fold_type', type=str, required=True,
                       help='Fold type (CV, loso)')
    parser.add_argument('--multi_type', type=str, required=True,
                       help='Multi-class strategy (standard, OvR, OvO)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save combined results')
    
    args = parser.parse_args()
    
    combine_array_results(
        args.results_dir,
        args.model_type, 
        args.fold_type,
        args.multi_type,
        args.output_dir
    )

if __name__ == "__main__":
    main() 