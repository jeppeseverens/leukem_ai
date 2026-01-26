import sys
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable optimizations that require AVX
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_test, transformers, classifiers

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import datetime
import pandas as pd
import argparse
import ast

from pathlib import Path

def main():
    # Parsing
    parser = argparse.ArgumentParser(description="Run outer cross-validation using best parameters from inner CV.")

    # Define argument configurations
    arg_configs = {
        'model_type': {
            'type': str,
            'help': 'Model type to use (XGBOOST, SVM, NN)'
        },
        'multi_type': {
            'type': str,
            'default': 'standard',
            'help': 'Multiclass strategy to use (standard, OvO, OvR)'
        },
        'fold_type': {
            'type': str,
            'default': 'CV',
            'help': 'Type of cross-validation fold to use (CV, loso)'
        },
        'best_params_path': {
            'type': str,
            'help': 'Path to the best parameters CSV file from inner CV'
        }
    }

    # Add arguments from configurations
    for arg_name, config in arg_configs.items():
        parser.add_argument(
            f'--{arg_name}',
            **config
        )

    args = parser.parse_args()
    
    print(f"Using model {args.model_type} with {args.multi_type} strategy and {args.fold_type} fold type", flush=True)
    print(f"Best parameters from: {args.best_params_path}", flush=True)
    
    # Get the current date and time in string format
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # Get project root directory
    base_path = Path(__file__).resolve().parent
    project_root = base_path.parent
    
    # Create the output directory if it doesn't exist (use absolute path)
    output_dir = project_root / "data" / "out" / "outer_cv" / f"{args.model_type}_n10"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir is {output_dir}", flush=True)

    # Load and prepare data
    print("Loading and preparing data", flush=True)

    data_path = project_root / "data"
    X, y, study_labels = train_test.load_data(data_path)
    X, y, study_labels = train_test.filter_data(X, y, study_labels, min_n = 10)
    y, label_mapping = train_test.encode_labels(y)

    # Define the model based on model type
    if args.model_type == "XGBOOST":
        model = classifiers.WeightedXGBClassifier
    elif args.model_type == "SVM":
        from sklearn.svm import SVC
        model = SVC
    elif args.model_type == "NN":
        model = classifiers.NeuralNet
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    # Define the pipeline
    pipe = Pipeline([
        ('DEseq2', transformers.DESeq2RatioNormalizer()),
        ('feature_selection', transformers.FeatureSelection2()),
        ('scaler', StandardScaler())
    ])
    print("Pipeline set up", flush=True)

    # Load best parameters
    print(f"Loading best parameters from {args.best_params_path}", flush=True)
    best_params = pd.read_csv(args.best_params_path)
    print(f"Loaded {len(best_params)} best parameter sets", flush=True)

    # Start the outer cross-validation process
    print("Starting outer cross-validation process.", flush=True)
    
    if args.fold_type == "CV":
        print("Calling run_outer_cv (CV fold type)...", flush=True)
        df = train_test.run_outer_cv(
            X, y, study_labels, model, pipe, best_params,
            multi_type=args.multi_type, model_type=args.model_type
        )
    elif args.fold_type == "loso":
        print("Calling run_outer_cv_loso (loso fold type)...", flush=True)
        df = train_test.run_outer_cv_loso(
            X, y, study_labels, model, pipe, best_params,
            multi_type=args.multi_type, model_type=args.model_type
        )
    else:
        raise ValueError(f"Fold type {args.fold_type} not supported.")

    # Convert encoded labels back to original class names
    df = train_test.restore_labels(df, label_mapping)
    
    # Save results to CSV file with model type, strategy and timestamp
    output_filename = f"{args.model_type}_outer_cv_{args.fold_type}_{args.multi_type}_{time}.csv"
    output_path = output_dir / output_filename
    df.to_csv(output_path)
    print(f"Results saved to {output_path}")

    print("Outer cross-validation process finished.")

if __name__ == "__main__":
    print("Entering main() function", flush=True)
    main() 