#!/usr/bin/env python3
"""
Prediction script for new samples using trained final models.

This script loads the final trained models (NN, SVM, XGBOOST) and ensemble weights
to make predictions on new samples. It applies the same preprocessing pipeline
and cutoffs as used during training.

Usage:
    python predict_new_samples.py --input_file path/to/new_samples.csv --output_dir path/to/output/
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import joblib
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import required modules from the project
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import classifiers
import transformers
import train_test
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def standardize_class_names(class_names):
    """
    Standardize class names to ensure consistency across all models.
    Converts special characters to dots to match R's make.names behavior.
    
    R's make.names converts the following characters to dots:
    - Parentheses: ( and )
    - Forward slashes: /
    - Colons: : (single and double ::)
    - Semicolons: ;
    - Spaces
    - Commas
    - Underscores
    
    Multiple consecutive dots are collapsed to double dots (..)
    Leading/trailing dots are removed.
    
    Parameters:
    -----------
    class_names : list or pd.Index
        List of class names to standardize
        
    Returns:
    --------
    list : Standardized class names
    """
    import re
    
    if isinstance(class_names, str):
        class_names = [class_names]
    
    standardized = []
    for name in class_names:
        # Replace all special characters with dots to match R's make.names behavior
        # These are the characters that R's make.names converts to dots
        name = name.replace('(', '.')
        name = name.replace(')', '.')
        name = name.replace('/', '.')
        name = name.replace(':', '.')
        name = name.replace(';', '.')
        name = name.replace('_', '.')
        name = name.replace(' ', '.')
        name = name.replace(',', '.')
        
        # Collapse multiple consecutive dots into double dots (..)
        # This matches R's make.names behavior
        name = re.sub(r'\.{3,}', '..', name)
        
        # Remove leading/trailing dots
        name = name.strip('.')
        
        standardized.append(name)
    
    return standardized


def load_cached_pipeline(n_genes, pipelines_dir):
    """
    Load a cached pipeline for the given n_genes value.
    
    Parameters:
    -----------
    n_genes : int
        Number of genes for feature selection
    pipelines_dir : str
        Path to the pipelines cache directory
        
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
        Fitted preprocessing pipeline
    """
    pipeline_filename = f"pipeline_ngenes_{n_genes}.pkl"
    pipeline_path = os.path.join(pipelines_dir, pipeline_filename)
    
    if os.path.exists(pipeline_path):
        print(f"  Loading cached pipeline for n_genes={n_genes}: {pipeline_path}")
        return joblib.load(pipeline_path)
    else:
        raise FileNotFoundError(f"Cached pipeline not found: {pipeline_path}")


def load_training_gene_order():
    """
    Load the original training data gene order from counts file.
    
    Returns:
    --------
    training_genes : list
        List of gene names in the same order as training data
    """
    base_path = Path(__file__).resolve().parent.parent
    counts_file = base_path / "data" / "counts_20aug25.csv"
    
    print(f"Loading training gene order from {counts_file}")
    
    # Read just the first row to get gene names (column headers)
    # The training data has genes as rows, so we need the index
    df_header = pd.read_csv(counts_file, nrows=1)
    
    # Get the first column name (should be gene identifier column)
    gene_col = df_header.columns[0]
    
    # Now read just the gene column to get all gene names
    df_genes = pd.read_csv(counts_file, usecols=[gene_col])
    training_genes = df_genes[gene_col].tolist()
    
    print(f"Found {len(training_genes)} genes in training data")
    
    return training_genes


def load_new_samples(input_file):
    """
    Load new samples from CSV file and reorder genes to match training data.
    Expected format: samples on rows, genes (ENS...) on columns.
    
    Parameters:
    -----------
    input_file : str
        Path to the CSV file containing new samples
        
    Returns:
    --------
    X : np.ndarray
        Gene expression data (samples x genes) in training gene order
    sample_names : list
        Sample identifiers from row names
    """
    print(f"Loading new samples from {input_file}")
    
    # Load the CSV file
    df = pd.read_csv(input_file, index_col=0)
    
    # Extract sample names from index
    sample_names = df.index.tolist()
    
    # Get training gene order
    training_genes = load_training_gene_order()
    
    # Check which genes are available in new data
    available_genes = set(df.columns)
    training_genes_set = set(training_genes)
    
    missing_genes = training_genes_set - available_genes
    extra_genes = available_genes - training_genes_set
    
    print(f"Loaded {df.shape[0]} samples with {df.shape[1]} genes")
    print(f"Training data expects {len(training_genes)} genes")
    print(f"Missing genes: {len(missing_genes)}")
    print(f"Extra genes: {len(extra_genes)}")
    
    if missing_genes:
        print(f"WARNING: {len(missing_genes)} genes from training data are missing in new data")
        if len(missing_genes) <= 10:
            print(f"Missing genes: {list(missing_genes)[:10]}")
        else:
            print(f"First 10 missing genes: {list(missing_genes)[:10]}")
    
    # Reorder columns to match training data and fill missing genes with zeros
    X_reordered = np.zeros((df.shape[0], len(training_genes)), dtype=np.float32)
    
    for i, gene in enumerate(training_genes):
        if gene in df.columns:
            X_reordered[:, i] = df[gene].values.astype(np.float32)
        else:
            # Missing gene - fill with zeros (or could use median/mean)
            X_reordered[:, i] = 0.0
    
    print(f"Reordered data shape: {X_reordered.shape}")
    
    return X_reordered, sample_names


def load_models_and_metadata(models_dir, pipelines_dir=None):
    """
    Load all final models and their metadata.
    
    Parameters:
    -----------
    models_dir : str
        Path to the final_models directory
    pipelines_dir : str, optional
        Path to the pipelines cache directory. If provided, will load cached pipelines
        based on n_genes from model metadata. Otherwise, loads reference pipelines.
        
    Returns:
    --------
    models : dict
        Dictionary containing loaded models and metadata for each model type
    """
    models = {}
    
    # Global pipeline cache to avoid loading the same pipeline multiple times
    global_pipeline_cache = {}
    
    # Load NN model (standard multiclass)
    nn_dir = os.path.join(models_dir, "NN")
    if os.path.exists(nn_dir):
        print("Loading NN model...")
        
        # Load model
        model_path = os.path.join(nn_dir, "NN_final_CV_standard_model_0.pkl")
        with open(model_path, 'rb') as f:
            nn_model = joblib.load(f)
        
        # Load label mapping
        label_mapping_path = os.path.join(nn_dir, "label_mapping_NN_CV_standard.json")
        with open(label_mapping_path, 'r') as f:
            nn_label_mapping = json.load(f)
        
        # Load metadata
        metadata_path = os.path.join(nn_dir, "NN_final_CV_standard_model_0_metadata.json")
        with open(metadata_path, 'r') as f:
            nn_metadata = json.load(f)
        
        # Load pipeline based on n_genes from metadata
        if pipelines_dir is not None and 'model_info' in nn_metadata and 'n_genes' in nn_metadata['model_info']:
            n_genes = nn_metadata['model_info']['n_genes']
            if n_genes not in global_pipeline_cache:
                global_pipeline_cache[n_genes] = load_cached_pipeline(n_genes, pipelines_dir)
            nn_pipeline = global_pipeline_cache[n_genes]
        else:
            # Fallback to reference pipeline
            pipeline_path = os.path.join(nn_dir, "pipeline_NN_CV_standard.pkl")
            with open(pipeline_path, 'rb') as f:
                nn_pipeline = joblib.load(f)
        
        models['NN'] = {
            'model': nn_model,
            'pipeline': nn_pipeline,
            'label_mapping': nn_label_mapping,
            'metadata': nn_metadata,
            'multi_type': 'standard'
        }
    
    # Load SVM models (OvR multiclass)
    svm_dir = os.path.join(models_dir, "SVM")
    if os.path.exists(svm_dir):
        print("Loading SVM models...")
        
        # Load label mapping
        label_mapping_path = os.path.join(svm_dir, "label_mapping_SVM_CV_OvR.json")
        with open(label_mapping_path, 'r') as f:
            svm_label_mapping = json.load(f)
        
        # Load all class-specific models and their pipelines
        svm_models = {}
        svm_metadata = {}
        svm_pipelines = {}  # Store pipeline for each class
        
        for file in os.listdir(svm_dir):
            if file.endswith('.pkl') and 'class_' in file:
                # Extract class name from filename
                class_name = file.replace('SVM_final_CV_OvR_class_', '').replace('.pkl', '')
                class_name = class_name.split('_model_')[0]
                
                # Load model
                model_path = os.path.join(svm_dir, file)
                with open(model_path, 'rb') as f:
                    svm_models[class_name] = joblib.load(f)
                
                # Load corresponding metadata
                metadata_file = file.replace('.pkl', '_metadata.json')
                metadata_path = os.path.join(svm_dir, metadata_file)
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        svm_metadata[class_name] = json.load(f)
                    
                    # Load pipeline based on n_genes from this class's metadata
                    if pipelines_dir is not None and 'model_info' in svm_metadata[class_name] and 'n_genes' in svm_metadata[class_name]['model_info']:
                        n_genes = svm_metadata[class_name]['model_info']['n_genes']
                        if n_genes not in global_pipeline_cache:
                            global_pipeline_cache[n_genes] = load_cached_pipeline(n_genes, pipelines_dir)
                        svm_pipelines[class_name] = global_pipeline_cache[n_genes]
        
        # Fallback to reference pipeline if no cached pipelines loaded
        if not svm_pipelines and pipelines_dir is None:
            pipeline_path = os.path.join(svm_dir, "pipeline_SVM_CV_OvR.pkl")
            if os.path.exists(pipeline_path):
                with open(pipeline_path, 'rb') as f:
                    reference_pipeline = joblib.load(f)
                # Use the same pipeline for all classes as fallback
                for class_name in svm_models.keys():
                    svm_pipelines[class_name] = reference_pipeline
        
        models['SVM'] = {
            'models': svm_models,
            'pipelines': svm_pipelines,  # Changed from single pipeline to per-class pipelines
            'label_mapping': svm_label_mapping,
            'metadata': svm_metadata,
            'multi_type': 'ovr'
        }
    
    # Load XGBOOST models (OvR multiclass)
    xgb_dir = os.path.join(models_dir, "XGBOOST")
    if os.path.exists(xgb_dir):
        print("Loading XGBOOST models...")
        
        # Load label mapping
        label_mapping_path = os.path.join(xgb_dir, "label_mapping_XGBOOST_CV_OvR.json")
        with open(label_mapping_path, 'r') as f:
            xgb_label_mapping = json.load(f)
        
        # Load all class-specific models and their pipelines
        xgb_models = {}
        xgb_metadata = {}
        xgb_pipelines = {}  # Store pipeline for each class
        
        for file in os.listdir(xgb_dir):
            if file.endswith('.pkl') and 'class_' in file:
                # Extract class name from filename
                class_name = file.replace('XGBOOST_final_CV_OvR_class_', '').replace('.pkl', '')
                class_name = class_name.split('_model_')[0]
                
                # Load model
                model_path = os.path.join(xgb_dir, file)
                with open(model_path, 'rb') as f:
                    xgb_models[class_name] = joblib.load(f)
                
                # Load corresponding metadata
                metadata_file = file.replace('.pkl', '_metadata.json')
                metadata_path = os.path.join(xgb_dir, metadata_file)
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        xgb_metadata[class_name] = json.load(f)
                    
                    # Load pipeline based on n_genes from this class's metadata
                    if pipelines_dir is not None and 'model_info' in xgb_metadata[class_name] and 'n_genes' in xgb_metadata[class_name]['model_info']:
                        n_genes = xgb_metadata[class_name]['model_info']['n_genes']
                        if n_genes not in global_pipeline_cache:
                            global_pipeline_cache[n_genes] = load_cached_pipeline(n_genes, pipelines_dir)
                        xgb_pipelines[class_name] = global_pipeline_cache[n_genes]
        
        # Fallback to reference pipeline if no cached pipelines loaded
        if not xgb_pipelines and pipelines_dir is None:
            pipeline_path = os.path.join(xgb_dir, "pipeline_XGBOOST_CV_OvR.pkl")
            if os.path.exists(pipeline_path):
                with open(pipeline_path, 'rb') as f:
                    reference_pipeline = joblib.load(f)
                # Use the same pipeline for all classes as fallback
                for class_name in xgb_models.keys():
                    xgb_pipelines[class_name] = reference_pipeline
        
        models['XGBOOST'] = {
            'models': xgb_models,
            'pipelines': xgb_pipelines,  # Changed from single pipeline to per-class pipelines
            'label_mapping': xgb_label_mapping,
            'metadata': xgb_metadata,
            'multi_type': 'ovr'
        }
    
    # Print summary of loaded pipelines
    if global_pipeline_cache:
        print(f"\nLoaded {len(global_pipeline_cache)} unique pipelines for n_genes: {sorted(global_pipeline_cache.keys())}")
    
    return models


def load_ensemble_weights(weights_dir):
    """
    Load ensemble weights for both global and OvR ensemble methods.
    
    Parameters:
    -----------
    weights_dir : str
        Path to the ensemble weights directory
        
    Returns:
    --------
    ensemble_weights : dict
        Dictionary containing ensemble weights
    """
    ensemble_weights = {}
    
    # Load global ensemble weights
    global_weights_path = os.path.join(weights_dir, "cv", "global_ensemble_weights_used.csv")
    if os.path.exists(global_weights_path):
        global_weights = pd.read_csv(global_weights_path)
        ensemble_weights['global'] = global_weights
        print("Loaded global ensemble weights")
    
    # Load OvR ensemble weights
    ovr_weights_path = os.path.join(weights_dir, "cv", "ovr_ensemble_weights_used.csv")
    if os.path.exists(ovr_weights_path):
        ovr_weights = pd.read_csv(ovr_weights_path)
        ensemble_weights['ovr'] = ovr_weights
        print("Loaded OvR ensemble weights")
    
    return ensemble_weights


def load_cutoffs(cutoffs_path):
    """
    Load prediction cutoffs for CV source.
    
    Parameters:
    -----------
    cutoffs_path : str
        Path to the cutoffs CSV file
        
    Returns:
    --------
    cutoffs : dict
        Dictionary containing cutoffs for each model
    """
    cutoffs_df = pd.read_csv(cutoffs_path)
    
    # Filter for CV source only
    cv_cutoffs = cutoffs_df[cutoffs_df['source'] == 'cv'].copy()
    
    cutoffs = {}
    for _, row in cv_cutoffs.iterrows():
        cutoffs[row['model']] = row['prob_cutoff']
    
    print(f"Loaded cutoffs for {len(cutoffs)} models")
    return cutoffs


def predict_nn_standard(X, models, sample_names):
    """
    Make predictions using the NN model (standard multiclass).
    
    Parameters:
    -----------
    X : np.ndarray
        Input data (samples x genes)
    models : dict
        Dictionary containing NN model info
    sample_names : list
        Sample identifiers
        
    Returns:
    --------
    predictions_df : pd.DataFrame
        DataFrame with predictions, probabilities, and sample info
    prob_matrix_df : pd.DataFrame
        DataFrame with full probability matrix (samples x classes)
    """
    print("Making NN predictions...")
    
    nn_info = models['NN']
    pipeline = nn_info['pipeline']
    model = nn_info['model']
    label_mapping = nn_info['label_mapping']
    
    # Create reverse mapping (encoded -> original labels)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Preprocess data using the pipeline
    # Note: We need to provide dummy study labels for preprocessing
    dummy_studies = np.zeros(X.shape[0])  # Assuming all samples from same study
    X_processed = pipeline.transform(X)
    
    # Make predictions
    pred_probs = model.predict_proba(X_processed)
    pred_classes = np.argmax(pred_probs, axis=1)
    
    # Convert back to original labels
    pred_labels = [reverse_mapping[cls] for cls in pred_classes]
    
    # Get maximum probability for each prediction
    max_probs = np.max(pred_probs, axis=1)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'sample_name': sample_names,
        'sample_index': range(len(sample_names)),
        'prediction': pred_labels,
        'prediction_prob': max_probs,
        'prediction_passed_cutoff': False  # Will be filled later with cutoffs
    })
    
    # Create full probability matrix DataFrame
    # Get class names in order and standardize them
    class_names = [reverse_mapping[i] for i in range(len(reverse_mapping))]
    class_names_standardized = standardize_class_names(class_names)
    prob_matrix_df = pd.DataFrame(pred_probs, columns=class_names_standardized)
    prob_matrix_df.insert(0, 'sample_name', sample_names)
    
    return results_df, prob_matrix_df


def predict_single_class(class_name, model, X_processed):
    """
    Make predictions for a single class model.
    
    Parameters:
    -----------
    class_name : str
        Name of the class
    model : sklearn model
        Trained model for this class
    X_processed : np.ndarray
        Preprocessed input data
        
    Returns:
    --------
    tuple : (class_name, probabilities, predictions)
    """
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_processed)
        if probs.shape[1] == 2:  # Binary classification
            class_probs = probs[:, 1]  # Probability of positive class
            class_preds = (probs[:, 1] >= 0.5).astype(int)
        else:
            class_probs = np.max(probs, axis=1)
            class_preds = np.argmax(probs, axis=1)
    else:
        # For models that only support decision_function
        scores = model.decision_function(X_processed)
        class_probs = 1 / (1 + np.exp(-scores))
        class_preds = (scores >= 0).astype(int)
    
    return class_name, class_probs, class_preds


def predict_ovr_models(X, models, model_type, sample_names):
    """
    Make predictions using OvR models (SVM or XGBOOST).
    Each class may use a different pipeline based on its n_genes hyperparameter.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data (samples x genes)
    models : dict
        Dictionary containing model info
    model_type : str
        'SVM' or 'XGBOOST'
    sample_names : list
        Sample identifiers
        
    Returns:
    --------
    predictions_df : pd.DataFrame
        DataFrame with predictions, probabilities, and sample info
    prob_matrix_df : pd.DataFrame
        DataFrame with full probability matrix (samples x classes)
    """
    print(f"Making {model_type} predictions...")
    
    model_info = models[model_type]
    class_models = model_info['models']
    class_pipelines = model_info.get('pipelines', {})
    label_mapping = model_info['label_mapping']
    
    # Group classes by pipeline to minimize data processing
    pipeline_groups = {}
    for class_name, model in class_models.items():
        if class_name in class_pipelines:
            pipeline = class_pipelines[class_name]
        else:
            pipeline = next(iter(class_pipelines.values())) if class_pipelines else None
            if pipeline is None:
                raise ValueError(f"No pipeline available for class {class_name}")
        
        pipeline_id = id(pipeline)
        if pipeline_id not in pipeline_groups:
            pipeline_groups[pipeline_id] = {'pipeline': pipeline, 'classes': []}
        pipeline_groups[pipeline_id]['classes'].append((class_name, model))
    
    # Process data once per unique pipeline and make predictions for all classes using that pipeline
    class_probabilities = {}
    class_predictions = {}
    
    for pipeline_id, group_info in pipeline_groups.items():
        pipeline = group_info['pipeline']
        classes_with_models = group_info['classes']
        
        # Process data once for this pipeline
        dummy_studies = np.zeros(X.shape[0])  # Assuming all samples from same study
        X_processed = pipeline.transform(X)
        print(f"  Processed data for pipeline {pipeline_id} (used by {len(classes_with_models)} classes)")
        
        # Make predictions for all classes using this processed data
        for class_name, model in classes_with_models:
            class_name, class_probs, class_preds = predict_single_class(class_name, model, X_processed)
            class_probabilities[class_name] = class_probs
            class_predictions[class_name] = class_preds
    
    # Use vectorized operations for faster aggregation
    class_names = list(class_probabilities.keys())
    prob_matrix = np.column_stack([class_probabilities[class_name] for class_name in class_names])
    
    # Standardize class names for consistency
    class_names_standardized = standardize_class_names(class_names)
    
    # Find the class with highest probability for each sample using numpy (faster than pandas)
    max_prob_indices = np.argmax(prob_matrix, axis=1)
    max_probs = np.max(prob_matrix, axis=1)
    pred_classes = [class_names_standardized[idx] for idx in max_prob_indices]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'sample_name': sample_names,
        'sample_index': range(len(sample_names)),
        'prediction': pred_classes,
        'prediction_prob': max_probs,
        'prediction_passed_cutoff': False  # Will be filled later with cutoffs
    })
    
    # Create full probability matrix DataFrame
    prob_matrix_df = pd.DataFrame(prob_matrix, columns=class_names_standardized)
    prob_matrix_df.insert(0, 'sample_name', sample_names)
    
    return results_df, prob_matrix_df


def predict_ensemble_global(individual_predictions, individual_prob_matrices, ensemble_weights, sample_names):
    """
    Make predictions using global ensemble method.
    
    Parameters:
    -----------
    individual_predictions : dict
        Dictionary containing pre-computed individual model predictions
    individual_prob_matrices : dict
        Dictionary containing probability matrices from individual models
    ensemble_weights : dict
        Dictionary containing ensemble weights
    sample_names : list
        Sample identifiers
        
    Returns:
    --------
    predictions_df : pd.DataFrame
        DataFrame with ensemble predictions
    prob_matrix_df : pd.DataFrame
        DataFrame with full probability matrix (samples x classes)
    """
    print("Making Global Ensemble predictions...")
    
    weights = ensemble_weights['global'].iloc[0]  # Should be only one row
    print(f"  Using weights: NN={weights['nn_weight']:.3f}, SVM={weights['svm_weight']:.3f}, XGB={weights['xgb_weight']:.3f}")
    
    # Pre-compute standardized column mappings for each model
    # This maps: standardized_class_name -> original_column_name_in_that_model
    model_col_mappings = {}
    for model_name, prob_matrix in individual_prob_matrices.items():
        if prob_matrix is None:
            continue
        
        # Create a mapping of standardized column names to original column names
        col_mapping = {}
        for col in prob_matrix.columns:
            if col != 'sample_name':
                standardized_col = standardize_class_names([col])[0]
                col_mapping[standardized_col] = col
        model_col_mappings[model_name] = col_mapping
        
        print(f"  {model_name}: mapped {len(col_mapping)} classes")
    
    # Get all unique standardized class names across all models
    all_classes = set()
    for col_mapping in model_col_mappings.values():
        all_classes.update(col_mapping.keys())
    
    all_classes = sorted(list(all_classes))
    print(f"  Total unique classes: {len(all_classes)}")
    
    # Initialize ensemble probability matrix
    ensemble_prob_matrix = np.zeros((len(sample_names), len(all_classes)))
    
    # For each model, add weighted probabilities
    for model_name, prob_matrix in individual_prob_matrices.items():
        if prob_matrix is None:
            continue
        
        # Get weight for this model
        if model_name == 'NN' and weights['nn_weight'] > 0:
            weight = weights['nn_weight']
        elif model_name == 'SVM' and weights['svm_weight'] > 0:
            weight = weights['svm_weight']
        elif model_name == 'XGBOOST' and weights['xgb_weight'] > 0:
            weight = weights['xgb_weight']
        else:
            continue
        
        # Get the column mapping for this model
        col_mapping = model_col_mappings[model_name]
        
        # Add weighted probabilities for each class
        classes_matched = 0
        for j, class_name in enumerate(all_classes):
            # Look up the original column name in this model's probability matrix
            if class_name in col_mapping:
                original_col = col_mapping[class_name]
                ensemble_prob_matrix[:, j] += weight * prob_matrix[original_col].values
                classes_matched += 1
        
        print(f"  {model_name}: matched {classes_matched}/{len(all_classes)} classes for weighting")
    
    # Normalize probabilities to sum to 1 for each sample
    for i in range(ensemble_prob_matrix.shape[0]):
        row_sum = np.sum(ensemble_prob_matrix[i, :])
        if row_sum > 0:
            ensemble_prob_matrix[i, :] = ensemble_prob_matrix[i, :] / row_sum
        else:
            # If all values are 0, set equal probabilities
            ensemble_prob_matrix[i, :] = 1.0 / len(all_classes)
    
    # Find best prediction for each sample
    max_prob_indices = np.argmax(ensemble_prob_matrix, axis=1)
    max_probs = np.max(ensemble_prob_matrix, axis=1)
    pred_classes = [all_classes[idx] for idx in max_prob_indices]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'sample_name': sample_names,
        'sample_index': range(len(sample_names)),
        'prediction': pred_classes,
        'prediction_prob': max_probs,
        'prediction_passed_cutoff': False  # Will be filled later with cutoffs
    })
    
    # Create full probability matrix DataFrame
    prob_matrix_df = pd.DataFrame(ensemble_prob_matrix, columns=all_classes)
    prob_matrix_df.insert(0, 'sample_name', sample_names)
    
    return results_df, prob_matrix_df


def predict_ensemble_ovr(individual_predictions, individual_prob_matrices, ensemble_weights, sample_names):
    """
    Make predictions using OvR ensemble method.
    
    Parameters:
    -----------
    individual_predictions : dict
        Dictionary containing pre-computed individual model predictions
    individual_prob_matrices : dict
        Dictionary containing probability matrices from individual models
    ensemble_weights : dict
        Dictionary containing ensemble weights
    sample_names : list
        Sample identifiers
        
    Returns:
    --------
    predictions_df : pd.DataFrame
        DataFrame with ensemble predictions
    prob_matrix_df : pd.DataFrame
        DataFrame with full probability matrix (samples x classes)
    """
    print("Making OvR Ensemble predictions...")
    
    ovr_weights = ensemble_weights['ovr']
    
    # Get all unique classes from OvR weights (already R-standardized)
    # Use them directly as the final standardized class names
    all_classes = []
    for _, weight_row in ovr_weights.iterrows():
        class_name_r_standardized = weight_row['class']  # e.g., "AML.with.inv.16..t.16.16..CBFB..MYH11"
        all_classes.append(class_name_r_standardized)
    
    # Initialize ensemble probability matrix
    ensemble_prob_matrix = np.zeros((len(sample_names), len(all_classes)))
    
    # Pre-compute standardized column mappings for each model to avoid repeated computation
    model_col_mappings = {}
    for model_name, prob_matrix in individual_prob_matrices.items():
        if prob_matrix is None:
            continue
        
        # Create a mapping of standardized column names to original column names
        col_mapping = {}
        for col in prob_matrix.columns:
            if col != 'sample_name':
                standardized_col = standardize_class_names([col])[0]
                col_mapping[standardized_col] = col
        model_col_mappings[model_name] = col_mapping
        
        print(f"  {model_name}: mapped {len(col_mapping)} classes")
    
    # For each class, calculate weighted ensemble score across all samples
    for j, (_, weight_row) in enumerate(ovr_weights.iterrows()):
        class_name_r_standardized = weight_row['class']  # e.g., "AML.with.inv.16..t.16.16..CBFB..MYH11"
        
        # For each model, get probabilities for this class
        for model_name, prob_matrix in individual_prob_matrices.items():
            if prob_matrix is None:
                continue
            
            # Get weight for this model and class
            if model_name == 'NN':
                weight = weight_row['nn_weight']
            elif model_name == 'SVM':
                weight = weight_row['svm_weight']
            elif model_name == 'XGBOOST':
                weight = weight_row['xgb_weight']
            else:
                continue
            
            if weight == 0:
                continue
            
            # Get the column mapping for this model
            col_mapping = model_col_mappings.get(model_name, {})
            
            # Add weighted probabilities for this class
            if class_name_r_standardized in col_mapping:
                original_col = col_mapping[class_name_r_standardized]
                ensemble_prob_matrix[:, j] += weight * prob_matrix[original_col].values
            else:
                print(f"  WARNING: Class '{class_name_r_standardized}' not found in {model_name} probability matrix")
    
    # Normalize probabilities to sum to 1 for each sample
    for i in range(ensemble_prob_matrix.shape[0]):
        row_sum = np.sum(ensemble_prob_matrix[i, :])
        if row_sum > 0:
            ensemble_prob_matrix[i, :] = ensemble_prob_matrix[i, :] / row_sum
        else:
            # If all values are 0, set equal probabilities
            ensemble_prob_matrix[i, :] = 1.0 / len(all_classes)
    
    # Find best prediction for each sample
    max_prob_indices = np.argmax(ensemble_prob_matrix, axis=1)
    max_probs = np.max(ensemble_prob_matrix, axis=1)
    pred_classes = [all_classes[idx] for idx in max_prob_indices]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'sample_name': sample_names,
        'sample_index': range(len(sample_names)),
        'prediction': pred_classes,
        'prediction_prob': max_probs,
        'prediction_passed_cutoff': False  # Will be filled later with cutoffs
    })
    
    # Create full probability matrix DataFrame
    prob_matrix_df = pd.DataFrame(ensemble_prob_matrix, columns=all_classes)
    prob_matrix_df.insert(0, 'sample_name', sample_names)
    
    return results_df, prob_matrix_df


def merge_probability_classes(prob_matrix_df):
    """
    Merge specific classes in the probability matrix:
    1. Sum probabilities for all classes with 'MDS', 'TP53', or 'Mecom' in their name to "MDS.r"
    2. Sum all other KMT2A classes (excluding MLLT3 fusion) to "other.KMT2A"
    
    Parameters:
    -----------
    prob_matrix_df : pd.DataFrame
        Probability matrix DataFrame with 'sample_name' column and class probability columns
        
    Returns:
    --------
    prob_matrix_df : pd.DataFrame
        Modified probability matrix with merged classes
    """
    # Get all column names except 'sample_name'
    class_columns = [col for col in prob_matrix_df.columns if col != 'sample_name']
    
    # Identify classes to merge for MDS or MECOM
    mds_mecom_classes = []
    for col in class_columns:
        col_lower = col.lower()
        if 'mds' in col_lower or 'tp53' in col_lower or 'mecom' in col_lower:
            mds_mecom_classes.append(col)
    
    # Identify classes to merge for other KMT2A (excluding MLLT3)
    other_kmt2a_classes = []
    for col in class_columns:
        col_lower = col.lower()
        # Check if it contains KMT2A but not MLLT3
        if 'kmt2a' in col_lower and 'mllt3' not in col_lower:
            other_kmt2a_classes.append(col)
    
    if mds_mecom_classes:
        print(f"  Merging {len(mds_mecom_classes)} classes to MDS.r: {mds_mecom_classes}")
        # Sum probabilities for MDS/MECOM classes
        prob_matrix_df['MDS.r'] = prob_matrix_df[mds_mecom_classes].sum(axis=1)
        # Remove individual classes
        prob_matrix_df = prob_matrix_df.drop(columns=mds_mecom_classes)
    
    if other_kmt2a_classes:
        print(f"  Merging {len(other_kmt2a_classes)} classes to other.KMT2A: {other_kmt2a_classes}")
        # Sum probabilities for other KMT2A classes
        prob_matrix_df['other.KMT2A'] = prob_matrix_df[other_kmt2a_classes].sum(axis=1)
        # Remove individual classes
        prob_matrix_df = prob_matrix_df.drop(columns=other_kmt2a_classes)
    
    return prob_matrix_df


def apply_cutoffs(predictions_dict, cutoffs):
    """
    Apply probability cutoffs to predictions.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary of prediction DataFrames
    cutoffs : dict
        Dictionary of cutoffs for each model
        
    Returns:
    --------
    predictions_dict : dict
        Updated dictionary with cutoff information
    """
    print("Applying probability cutoffs...")
    
    # Map model names to cutoff keys
    cutoff_mapping = {
        'NN': 'neural_net',
        'SVM': 'svm',
        'XGBOOST': 'xgboost',
        'Global_Ensemble': 'Global_Optimized',
        'OvR_Ensemble': 'OvR_Ensemble'
    }
    
    for model_name, df in predictions_dict.items():
        cutoff_key = cutoff_mapping.get(model_name, model_name)
        
        if cutoff_key in cutoffs:
            cutoff_value = cutoffs[cutoff_key]
            df['prediction_passed_cutoff'] = df['prediction_prob'] >= cutoff_value
            print(f"Applied cutoff {cutoff_value:.2f} to {model_name}")
        else:
            print(f"No cutoff found for {model_name}")
            df['prediction_passed_cutoff'] = True  # Default to True if no cutoff
    
    return predictions_dict


def save_predictions(predictions_dict, prob_matrices_dict, output_dir, input_filename_prefix):
    """
    Save prediction DataFrames and probability matrices to CSV files.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary of prediction DataFrames
    prob_matrices_dict : dict
        Dictionary of probability matrix DataFrames
    output_dir : str
        Output directory path
    input_filename_prefix : str
        Prefix to prepend to output files (based on input filename)
    """
    print(f"Saving predictions to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prediction summaries
    for model_name, df in predictions_dict.items():
        filename = f"{input_filename_prefix}_{model_name}_predictions.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {model_name} predictions to {filename}")
    
    # Save full probability matrices
    for model_name, df in prob_matrices_dict.items():
        filename = f"{input_filename_prefix}_{model_name}_probability_matrix.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {model_name} probability matrix to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Make predictions on new samples using trained models")
    parser.add_argument("--input_file", required=True, help="Path to input CSV file with new samples")
    parser.add_argument("--output_dir", required=True, help="Output directory for prediction results")
    parser.add_argument("--models_dir", default=None, help="Path to final_models directory")
    parser.add_argument("--weights_dir", default=None, help="Path to ensemble_weights directory")
    parser.add_argument("--cutoffs_file", default=None, help="Path to cutoffs CSV file")
    parser.add_argument("--pipelines_dir", default=None, help="Path to pipelines cache directory")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.models_dir is None:
        base_path = Path(__file__).resolve().parent.parent
        args.models_dir = base_path / "data" / "out" / "final_models"
    
    if args.weights_dir is None:
        base_path = Path(__file__).resolve().parent.parent
        args.weights_dir = base_path / "data" / "out" / "final_train_test" / "ensemble_weights" / "ensemble_weights"
    
    if args.cutoffs_file is None:
        base_path = Path(__file__).resolve().parent.parent
        args.cutoffs_file = base_path / "data" / "out" / "final_train_test" / "cutoffs" / "train_test_cutoffs.csv"
    
    # Set up pipeline cache directory
    if args.pipelines_dir is None:
        pipelines_dir = train_test.get_pipeline_cache_dir()
    else:
        pipelines_dir = args.pipelines_dir
    
    # Extract input filename without extension for output naming
    input_path = Path(args.input_file)
    input_filename = input_path.stem  # Gets filename without extension
    
    # Create subdirectory based on input filename
    output_dir = os.path.join(args.output_dir, f"{input_filename}_predictions")
    
    print("=== Leukemia AI Prediction Pipeline ===")
    print(f"Input file: {args.input_file}")
    print(f"Input filename: {input_filename}")
    print(f"Output directory: {output_dir}")
    print(f"Models directory: {args.models_dir}")
    print(f"Weights directory: {args.weights_dir}")
    print(f"Cutoffs file: {args.cutoffs_file}")
    print(f"Pipelines directory: {pipelines_dir}")
    
    # Load new samples
    X, sample_names = load_new_samples(args.input_file)
    
    # Load models and metadata with pipeline cache
    models = load_models_and_metadata(args.models_dir, pipelines_dir)
    
    # Load ensemble weights
    ensemble_weights = load_ensemble_weights(args.weights_dir)
    
    # Load cutoffs
    cutoffs = load_cutoffs(args.cutoffs_file)
    
    # Make predictions with all models
    predictions = {}
    prob_matrices = {}
    
    # Individual model predictions
    if 'NN' in models:
        predictions['NN'], prob_matrices['NN'] = predict_nn_standard(X, models, sample_names)
    
    if 'SVM' in models:
        predictions['SVM'], prob_matrices['SVM'] = predict_ovr_models(X, models, 'SVM', sample_names)
    
    if 'XGBOOST' in models:
        predictions['XGBOOST'], prob_matrices['XGBOOST'] = predict_ovr_models(X, models, 'XGBOOST', sample_names)
    
    # Merge probability classes BEFORE ensemble methods
    # This ensures ensemble weights for merged classes (MDS.r, other.KMT2A) make sense
    print("\n=== Merging probability classes ===")
    for model_name in ['NN', 'SVM', 'XGBOOST']:
        if model_name in prob_matrices:
            print(f"\nMerging classes in {model_name} probability matrix:")
            prob_matrices[model_name] = merge_probability_classes(prob_matrices[model_name])
            print(f"  Final {model_name} classes: {len(prob_matrices[model_name].columns) - 1}")  # -1 for sample_name
    
    # Ensemble predictions (reuse individual predictions and prob matrices to avoid redundant computation)
    if 'global' in ensemble_weights:
        predictions['Global_Ensemble'], prob_matrices['Global_Ensemble'] = predict_ensemble_global(
            predictions, prob_matrices, ensemble_weights, sample_names
        )
    
    if 'ovr' in ensemble_weights:
        predictions['OvR_Ensemble'], prob_matrices['OvR_Ensemble'] = predict_ensemble_ovr(
            predictions, prob_matrices, ensemble_weights, sample_names
        )
    
    # Apply cutoffs to predictions
    predictions = apply_cutoffs(predictions, cutoffs)

    # Save predictions and probability matrices with input filename prefix
    save_predictions(predictions, prob_matrices, output_dir, input_filename)

    print("\nPrediction pipeline completed successfully!")


if __name__ == "__main__":
    main()
