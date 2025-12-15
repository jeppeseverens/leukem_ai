import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable optimizations that require AVX
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_test, transformers, classifiers

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import datetime
import pandas as pd
import argparse
import ast
import joblib
import json

from pathlib import Path

def main():
    # Parsing
    parser = argparse.ArgumentParser(description="Train final models using best parameters from train/test analysis.")

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
            'help': 'Path to the best parameters CSV file from train/test analysis'
        },
        'output_dir': {
            'type': str,
            'default': 'data/out/final_models',
            'help': 'Output directory for saving final models'
        }
    }

    # Add arguments from configurations
    for arg_name, config in arg_configs.items():
        parser.add_argument(
            f'--{arg_name}',
            **config
        )

    args = parser.parse_args()
    
    print(f"Training final {args.model_type} model with {args.multi_type} strategy using {args.fold_type} best parameters")
    print(f"Best parameters from: {args.best_params_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Get the current date and time in string format
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(args.output_dir, exist_ok=True)
    # Load and prepare data
    print("Loading and preparing data")

    base_path = Path(__file__).resolve().parent
    data_path = base_path.parent / "data"
    X, y, study_labels = train_test.load_data(data_path)
    X, y, study_labels = train_test.filter_data(X, y, study_labels, min_n=10)
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
    print("Pipeline set up")

    # Load best parameters
    print(f"Loading best parameters from {args.best_params_path}")
    best_params = pd.read_csv(args.best_params_path)
    print(f"Loaded {len(best_params)} best parameter sets")

    # Set up pipeline cache directory
    pipelines_cache_dir = train_test.get_pipeline_cache_dir()
    
    # Start the final model training process
    print("Starting final model training process.")
    print(f"Using pipeline cache directory: {pipelines_cache_dir}")
    
    trained_models = train_test.train_final_models(
        X, y, study_labels, model, pipe, best_params,
        multi_type=args.multi_type, model_type=args.model_type,
        pipelines_dir=pipelines_cache_dir
    )

    # Convert encoded labels back to original class names for saving metadata
    label_mapping_reversed = {v: k for k, v in label_mapping.items()}
    
    # Save models and metadata
    print("Saving trained models and metadata...")
    
    for i, (model_info, trained_model) in enumerate(trained_models):
        # Create model filename
        model_filename = f"{args.model_type}_final_{args.fold_type}_{args.multi_type}"
        
        def clean_filename(name):
            """Clean filename by removing/replacing invalid characters"""
            # Replace problematic characters with safe alternatives
            replacements = {
                '/': '_',
                '\\': '_',
                ':': '_',
                '*': '_',
                '?': '_',
                '"': '_',
                '<': '_',
                '>': '_',
                '|': '_',
                '(': '_',
                ')': '_'
            }
            for char, replacement in replacements.items():
                name = name.replace(char, replacement)
            return name
        
        if args.multi_type in ["OvR", "OvO"]:
            if "class" in model_info:
                class_label = label_mapping_reversed.get(model_info["class"], model_info["class"])
                clean_class_label = clean_filename(str(class_label))
                model_filename += f"_class_{clean_class_label}"
            elif "class_0" in model_info and "class_1" in model_info:
                class_0_label = label_mapping_reversed.get(model_info["class_0"], model_info["class_0"])
                class_1_label = label_mapping_reversed.get(model_info["class_1"], model_info["class_1"])
                clean_class_0_label = clean_filename(str(class_0_label))
                clean_class_1_label = clean_filename(str(class_1_label))
                model_filename += f"_class_{clean_class_0_label}_vs_{clean_class_1_label}"
        
        model_filename += f"_model_{i}.pkl"
        model_path = os.path.join(args.output_dir, model_filename)
        
        # Save the trained model
        joblib.dump(trained_model, model_path)
        print(f"  Saved model: {model_path}")
        
        # Save model metadata
        metadata_filename = model_filename.replace('.pkl', '_metadata.json')
        metadata_path = os.path.join(args.output_dir, metadata_filename)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__class__') and 'sklearn' in str(type(obj)):
                # Handle sklearn objects (like Pipeline) by converting to string representation
                return str(obj)
            else:
                return obj
        
        # Create a copy of model_info without the preprocessing_pipeline (saved separately)
        model_info_for_json = {k: v for k, v in model_info.items() if k != 'preprocessing_pipeline'}
        
        metadata = {
            "model_type": args.model_type,
            "multi_type": args.multi_type,
            "fold_type": args.fold_type,
            "training_date": time,
            "model_info": convert_numpy_types(model_info_for_json),
            "label_mapping": label_mapping,
            "data_shape": tuple(int(x) for x in X.shape),
            "n_classes": int(len(set(y))),
            "class_distribution": {label_mapping_reversed[k]: int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_path}")

    # Note: Pipelines are now automatically saved in the pipeline cache directory during training
    # No need to save them again here as they are already cached by n_genes value
    print(f"  Pipelines have been cached in: {pipelines_cache_dir}")

    # Save label mapping
    label_mapping_filename = f"label_mapping_{args.model_type}_{args.fold_type}_{args.multi_type}.json"
    label_mapping_path = os.path.join(args.output_dir, label_mapping_filename)
    with open(label_mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"  Saved label mapping: {label_mapping_path}")

    print("Final model training process finished.")
    print(f"Trained {len(trained_models)} models and saved to {args.output_dir}")

if __name__ == "__main__":
    import numpy as np
    main()
