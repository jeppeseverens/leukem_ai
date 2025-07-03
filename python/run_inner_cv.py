import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_test, transformers, classifiers

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, ParameterSampler
import datetime
import pandas as pd
import argparse
import random

from pathlib import Path

def main():
    # Parsing
    parser = argparse.ArgumentParser(description="Run cross-validation with adjustable n_jobs.")

    # Define argument configurations
    arg_configs = {
        'model_type': {
            'type': str,
            'help': 'Model type to use'
        },
        'n_jobs': {
            'type': int,
            'default': 12,
            'help': 'Number of jobs to run in parallel for cross-validation (default: 12)'
        },
        'k_out': {
            'type': int,
            'default': 5,
            'help': 'Number of inner folds (default: 5)'
        },
        'k_in': {
            'type': int,
            'default': 5,
            'help': 'Number of outer folds (default: 5)'
        },
        'n_max_param': {
            'type': int,
            'default': 2,
            'help': 'Maximum number of parameter combinations to sample (default: 2)'
        },
        'fold_type': {
            'type': str,
            'default': 'CV',
            'help': 'Type of cross-validation fold to use (default: cv)'
        }
    }

    # Add arguments from configurations
    for arg_name, config in arg_configs.items():
        parser.add_argument(
            f'--{arg_name}',
            **config
        )

    args = parser.parse_args() # Parse the command-line arguments
    n_jobs = args.n_jobs # Get the value of n_jobs from the parsed arguments
    # Get the number of inner and outer folds
    k_out = args.k_out
    k_in = args.k_in

    print(f"Using model {args.model_type} with {k_in} inner folds, {k_out} outer folds, and {n_jobs} cores")

    # Get the current date and time in string format
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # Create the output directory if it doesn't exist
    output_dir = f"out/{args.model_type}/{time}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir is {output_dir}")

    # Load and prepare data
    print("Loading and preparing data")

    base_path = Path(__file__).resolve().parent
    data_path = base_path.parent / "data"
    X, y, study_labels = train_test.load_data(data_path)
    X, y, study_labels = train_test.filter_data(X, y, study_labels, min_n = 10)
    y, label_mapping = train_test.encode_labels(y)

    # Define the model and parameter grid
    if args.model_type == "XGBOOST":
        model = classifiers.WeightedXGBClassifier
        param_grid = {
            'n_genes': [2000, 3000, 5000],
            'class_weight': [True],
            'max_depth': [2, 3, 5],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 200],
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
            "n_genes": [3000, 5000, 10000],
            "n_neurons": [
                [800, 400, 100],
                [400, 200, 50],
                [200, 100, 25],
                [800, 400],
                [400, 200],
                [200, 100],
            ],
            "use_batch_norm": [False],
            "dropout_rate": [0.2, 0.5],
            "batch_size": [32],
            "patience": [50],
            "l2_reg": [0.001, 0],
            "class_weight": [True, False],
            "min_delta": [0.001],
            "learning_rate": [0.0001],
            "loss_function": ["focal"],
        }
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    # If needed downsample param_list
    full_param_list = list(ParameterGrid(param_grid))

    # Batch norm and dropout do not play nicely together, waste of compute
    if args.model_type == "NN":
        full_param_list = [
            params for params in full_param_list
            if not (params['use_batch_norm'] and params['dropout_rate'] > 0)
        ]

    # Downsample if needed
    n_downsample = args.n_max_param
    if len(full_param_list) > n_downsample:
        param_list = random.sample(full_param_list, k=n_downsample)
    else:
        param_list = full_param_list

    # Define the pipeline
    pipe = Pipeline([
        ('DEseq2', transformers.DESeq2RatioNormalizer()),
        ('feature_selection', transformers.FeatureSelection2()),
        ('scaler', StandardScaler())
    ])
    print("Pipeline set up")

    # Start the inner cross-validation process
    print("Starting inner cross-validation process.")
    # Iterate through different multiclass classification strategies
    # standard: Uses the classifier's default multiclass handling
    # OvO: One-vs-One strategy - trains binary classifier between each pair of classes
    # OvR: One-vs-Rest strategy - trains binary classifier for each class against all others
    if args.model_type == "NN":
        multi_types = ["standard"]
    else:
        multi_types = ["OvR"]
    if args.fold_type == "CV":
        for multi_type in multi_types:
            df = train_test.run_inner_cv(
                X, y, study_labels, model, param_list, n_jobs, pipe, 
                multi_type=multi_type, k_out=k_out, k_in=k_in,
                model_type = args.model_type
                )

            # Convert encoded labels back to original class names
            df = train_test.restore_labels(df, label_mapping)

            # Save results to CSV file with model type, strategy and timestamp
            df.to_csv(f"{output_dir}/{args.model_type}_inner_cv_{multi_type}_{time}.csv")   
    elif args.fold_type == "loso":
        for multi_type in multi_types:
            df = train_test.run_inner_cv_loso(
                X, y, study_labels, model, param_list, n_jobs, pipe, 
                multi_type=multi_type,
                model_type = args.model_type
                )

            # Convert encoded labels back to original class names
            df = train_test.restore_labels(df, label_mapping)

            # Save results to CSV file with model type, strategy and timestamp
            df.to_csv(f"{output_dir}/{args.model_type}_inner_cv_loso_{multi_type}_{time}.csv")   
    else:
        raise ValueError(f"Fold type {args.fold_type} not supported.")

    print("Cross-validation process finished.")

if __name__ == "__main__":
    main()
