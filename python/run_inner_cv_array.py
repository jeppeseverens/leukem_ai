import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_test, transformers, classifiers

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import datetime
import pandas as pd
import argparse
import random
import pickle
import json

from pathlib import Path

def main():
    # Parsing
    parser = argparse.ArgumentParser(description="Run cross-validation for a single hyperparameter combination (array job).")

    # Define argument configurations
    arg_configs = {
        'model_type': {
            'type': str,
            'help': 'Model type to use'
        },
        'param_index': {
            'type': int,
            'help': 'Index of the hyperparameter combination to process (0-based)'
        },
        'k_out': {
            'type': int,
            'default': 5,
            'help': 'Number of outer folds (default: 5)'
        },
        'k_in': {
            'type': int,
            'default': 5,
            'help': 'Number of inner folds (default: 5)'
        },
        'n_max_param': {
            'type': int,
            'default': 96,
            'help': 'Maximum number of parameter combinations to sample (default: 96)'
        },
        'fold_type': {
            'type': str,
            'default': 'CV',
            'help': 'Type of cross-validation fold to use (default: CV)'
        },
        'random_seed': {
            'type': int,
            'default': 42,
            'help': 'Random seed for reproducibility (default: 42)'
        },
        'run_name': {
            'type': str,
            'default': 'run',
            'help': 'Name of the run (default: run)'
        }
    }

    # Add arguments from configurations
    for arg_name, config in arg_configs.items():
        parser.add_argument(
            f'--{arg_name}',
            **config
        )

    args = parser.parse_args()
    
    # Get the number of inner and outer folds
    k_out = args.k_out
    k_in = args.k_in
    param_index = args.param_index

    print(f"Processing hyperparameter combination {param_index} for model {args.model_type}")
    print(f"Using {k_in} inner folds, {k_out} outer folds")

    # Get the current date and time in string format
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # Create the output directory if it doesn't exist
    output_dir = f"out/{args.model_type}_array/{args.run_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir is {output_dir}")

    # Load and prepare data
    print("Loading and preparing data")

    base_path = Path(__file__).resolve().parent
    data_path = base_path.parent / "data"
    X, y, study_labels = train_test.load_data(data_path)
    X, y, study_labels = train_test.filter_data(X, y, study_labels, min_n = 10)
    y, label_mapping = train_test.encode_labels(y)

    # Define the model and parameter grid (same as original)
    if args.model_type == "XGBOOST":
        model = classifiers.WeightedXGBClassifier
        param_grid = {
            'n_genes': [2000, 3000, 5000],
            'class_weight': [True],
            'max_depth': [2, 3, 5],
            'learning_rate': [0.05, 0.1, 0.01],
            'n_estimators': [100, 200, 500],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1.0]
        }
    elif args.model_type == "SVM":
        from sklearn.svm import SVC
        model = SVC
        param_grid = {
            'n_genes': [1000, 2000, 3000, 5000],
            'C': [0.1, 1, 10, 100, 1000],  
            'gamma': ['auto', 'scale', 0.00001, 0.0001, 0.001],  
            'class_weight': ["balanced", None],
            'probability': [True]
        }
    elif args.model_type == "NN":
        model = classifiers.NeuralNet
        param_grid = {
            "n_genes": [2000, 3000, 5000, 7500],
            "n_neurons": [
                [800, 400, 200],
                [400, 200, 100],
                [200, 100, 50],
                [800, 400],
                [400, 200]
            ],
            "use_batch_norm": [False],
            "dropout_rate": [0.3, 0.5],
            "batch_size": [32],
            "patience": [30],
            "l2_reg": [0.001, 0.01, 0],
            "class_weight": [True, False],
            "min_delta": [0.005],
            "learning_rate": [0.0001],
            "loss_function": ["focal"],
        }
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    # Generate full parameter list (same logic as original)
    full_param_list = list(ParameterGrid(param_grid))

    # Batch norm and dropout do not play nicely together, waste of compute
    if args.model_type == "NN":
        full_param_list = [
            params for params in full_param_list
            if not (params['use_batch_norm'] and params['dropout_rate'] > 0)
        ]

    # Downsample if needed (using same random seed for reproducibility)
    random.seed(args.random_seed)
    n_downsample = args.n_max_param
    if len(full_param_list) > n_downsample:
        param_list = random.sample(full_param_list, k=n_downsample)
    else:
        param_list = full_param_list

    # Validate param_index
    if param_index >= len(param_list):
        raise ValueError(f"param_index {param_index} is out of range. Only {len(param_list)} parameter combinations available.")

    # Get the specific parameter combination for this array job
    single_param = param_list[param_index]
    print(f"Processing parameter combination: {single_param}")

    # Define the pipeline
    pipe = Pipeline([
        ('DEseq2', transformers.DESeq2RatioNormalizer()),
        ('feature_selection', transformers.FeatureSelection()),
        ('scaler', StandardScaler())
    ])
    print("Pipeline set up")

    # Determine multiclass strategy
    if args.model_type == "NN":
        multi_types = ["standard"]
    else:
        multi_types = ["OvR"]

    # Start the inner cross-validation process for this single parameter combination
    print("Starting inner cross-validation process for single parameter combination.")
    
    if args.fold_type == "CV":
        for multi_type in multi_types:
            # Modified function call to process single parameter
            df = train_test.run_inner_cv_single_param(
                X, y, study_labels, model, single_param, pipe, 
                multi_type=multi_type, k_out=k_out, k_in=k_in,
                model_type=args.model_type
            )

            # Convert encoded labels back to original class names
            df = train_test.restore_labels(df, label_mapping)

            # Save results with parameter index in filename
            output_filename = f"{args.model_type}_inner_cv_{multi_type}_param_{param_index:03d}_{time}.csv"
            df.to_csv(f"{output_dir}/{output_filename}")
            print(f"Saved results to {output_filename}")
            
    elif args.fold_type == "loso":
        for multi_type in multi_types:
            # Modified function call to process single parameter
            df = train_test.run_inner_cv_loso_single_param(
                X, y, study_labels, model, single_param, pipe, 
                multi_type=multi_type,
                model_type=args.model_type
            )

            # Convert encoded labels back to original class names
            df = train_test.restore_labels(df, label_mapping)

            # Save results with parameter index in filename
            output_filename = f"{args.model_type}_inner_cv_loso_{multi_type}_param_{param_index:03d}_{time}.csv"
            df.to_csv(f"{output_dir}/{output_filename}")
            print(f"Saved results to {output_filename}")
    else:
        raise ValueError(f"Fold type {args.fold_type} not supported.")

    print("Cross-validation process finished for parameter combination.")

if __name__ == "__main__":
    main() 