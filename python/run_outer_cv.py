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
    parser = argparse.ArgumentParser(description="Run outer cross-validation using best parameters from inner CV.")

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
        },
        'fs_method': {
            'type': str,
            'default': 'mad',
            'help': 'Feature selection method: "mad" (intersection MVGs, default) or "eta2" (eta2_subtype - eta2_study)'
        }
    }

    for arg_name, config in arg_configs.items():
        parser.add_argument(f'--{arg_name}', **config)

    parser.add_argument(
        '--include_leftout', action='store_true',
        help='Also predict on left-out class samples (rare/excluded subtypes) per fold. '
             'Saves a separate CSV for use in left-out-aware rejection analysis.'
    )

    args = parser.parse_args()
    
    print(f"Using model {args.model_type} with {args.multi_type} strategy and {args.fold_type} fold type", flush=True)
    print(f"Best parameters from: {args.best_params_path}", flush=True)
    print(f"Feature selection method: {args.fs_method}", flush=True)
    print(f"Include left-out predictions: {args.include_leftout}", flush=True)
    
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    base_path = Path(__file__).resolve().parent
    project_root = base_path.parent
    
    fs_method_lower = args.fs_method.lower()
    fs_suffix = "_fs_eta" if fs_method_lower == "eta2" else ""
    output_dir = project_root / "data" / "out" / "outer_cv" / f"{args.model_type}_n10{fs_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir is {output_dir}", flush=True)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("Loading and preparing data", flush=True)
    data_path = project_root / "data"
    X_all, y_all, study_all = train_test.load_data(data_path)

    # Extract left-out samples BEFORE filtering (needs string labels)
    if args.include_leftout:
        X_leftout, y_leftout, study_leftout, leftout_global_idx = (
            train_test.get_leftout_samples(X_all, y_all, study_all, min_n=10)
        )
        has_leftout = len(leftout_global_idx) > 0
        if not has_leftout:
            print("No left-out samples found, proceeding without leftout predictions.", flush=True)
    else:
        has_leftout = False

    # Filter and encode (standard path, unchanged)
    X, y, study_labels = train_test.filter_data(X_all, y_all, study_all, min_n=10)
    y, label_mapping = train_test.encode_labels(y)

    del X_all, y_all, study_all

    # -------------------------------------------------------------------------
    # Model and pipeline setup
    # -------------------------------------------------------------------------
    if args.model_type == "XGBOOST":
        model = classifiers.WeightedXGBClassifier
    elif args.model_type == "SVM":
        from sklearn.svm import SVC
        model = SVC
    elif args.model_type == "NN":
        model = classifiers.NeuralNet
    else:
        raise ValueError(f"Model type {args.model_type} not supported")
    
    if fs_method_lower == "eta2":
        feature_selector = transformers.FeatureSelectionEta()
        print("Using eta2-based feature selection (eta2_subtype - eta2_study).", flush=True)
    elif fs_method_lower == "mad":
        feature_selector = transformers.FeatureSelection2()
        print("Using MAD-based intersecting MVG feature selection (default).", flush=True)
    else:
        raise ValueError(f"Unknown fs_method '{args.fs_method}'. Use 'mad' or 'eta2'.")
    
    pipe = Pipeline([
        ('DEseq2', transformers.DESeq2RatioNormalizer()),
        ('feature_selection', feature_selector),
        ('scaler', StandardScaler())
    ])
    print("Pipeline set up", flush=True)

    # -------------------------------------------------------------------------
    # Load best parameters
    # -------------------------------------------------------------------------
    print(f"Loading best parameters from {args.best_params_path}", flush=True)
    best_params = pd.read_csv(args.best_params_path)
    print(f"Loaded {len(best_params)} best parameter sets", flush=True)

    # -------------------------------------------------------------------------
    # Run outer cross-validation (standard, unchanged)
    # -------------------------------------------------------------------------
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

    df = train_test.restore_labels(df, label_mapping)
    
    output_filename = f"{args.model_type}_outer_cv_{args.fold_type}_{args.multi_type}{fs_suffix}_{time}.csv"
    output_path = output_dir / output_filename
    df.to_csv(output_path)
    print(f"Results saved to {output_path}")

    # -------------------------------------------------------------------------
    # Left-out predictions (separate CSV)
    # -------------------------------------------------------------------------
    if has_leftout:
        print("\nStarting left-out sample predictions.", flush=True)

        if args.fold_type == "CV":
            leftout_fold_assignments = train_test.assign_leftout_to_cv_folds(
                y_leftout, study_leftout, n_folds=5
            )
            leftout_df = train_test.run_outer_cv_leftout(
                X, y, study_labels,
                X_leftout, y_leftout, leftout_global_idx, leftout_fold_assignments,
                model, pipe, best_params,
                multi_type=args.multi_type, model_type=args.model_type,
            )
        elif args.fold_type == "loso":
            leftout_df = train_test.run_outer_cv_loso_leftout(
                X, y, study_labels,
                X_leftout, y_leftout, study_leftout, leftout_global_idx,
                model, pipe, best_params,
                multi_type=args.multi_type, model_type=args.model_type,
            )

        leftout_df = train_test.restore_labels(leftout_df, label_mapping)

        leftout_filename = f"{args.model_type}_outer_cv_{args.fold_type}_{args.multi_type}_leftout{fs_suffix}_{time}.csv"
        leftout_path = output_dir / leftout_filename
        leftout_df.to_csv(leftout_path)
        print(f"Left-out results saved to {leftout_path}")

    print("Outer cross-validation process finished.")

if __name__ == "__main__":
    print("Entering main() function", flush=True)
    main()