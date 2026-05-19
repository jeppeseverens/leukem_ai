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
    Load ensemble weights for the global ensemble method.
    
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
    
    # Load global ensemble weights (CV only, not LOSO)
    global_weights_path = os.path.join(weights_dir, "cv", "global_ensemble_weights_used.csv")
    if os.path.exists(global_weights_path):
        global_weights = pd.read_csv(global_weights_path)
        ensemble_weights['global'] = global_weights
        print("Loaded global ensemble weights (CV)")
    else:
        print(f"WARNING: Global ensemble weights not found at {global_weights_path}")

    return ensemble_weights


def load_cutoffs(cutoffs_path, required=False):
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
    if not os.path.exists(cutoffs_path):
        msg = f"Cutoffs file not found at {cutoffs_path}"
        if required:
            raise FileNotFoundError(msg)
        print(f"WARNING: {msg}")
        return {}
    
    cutoffs_df = pd.read_csv(cutoffs_path)
    
    # Filter for CV source only
    cv_cutoffs = cutoffs_df[cutoffs_df['source'] == 'cv'].copy()
    
    cutoffs = {}
    for _, row in cv_cutoffs.iterrows():
        cutoffs[row['model']] = row['prob_cutoff']
    
    if required and len(cutoffs) == 0:
        raise ValueError(f"No CV cutoffs found in required file: {cutoffs_path}")
    print(f"Loaded cutoffs for {len(cutoffs)} models")
    return cutoffs


def load_risk_coverage(risk_cov_path, required=False):
    """
    Load risk–coverage curve data.

    Parameters
    ----------
    risk_cov_path : str
        Path to a risk-coverage CSV file.

    Returns
    -------
    pd.DataFrame or None
        Risk–coverage table.
    """
    if not os.path.exists(risk_cov_path):
        msg = f"Risk–coverage file not found at {risk_cov_path}"
        if required:
            raise FileNotFoundError(msg)
        print(f"WARNING: {msg}")
        return None

    df = pd.read_csv(risk_cov_path)
    if df.empty:
        msg = f"Risk–coverage file {risk_cov_path} is empty"
        if required:
            raise ValueError(msg)
        print(f"WARNING: {msg}")
        return None

    return df


def prepare_risk_curve_for_selection(risk_cov_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize risk-curve input to the selection schema expected by cutoff
    selection: model, prob_cutoff, mean_risk, mean_coverage (+ optional mean_kappa).

    Preferred deployable input is the pre-aggregated CSV exported by
    `R/export_deployable_risk_seen_coverage_curves.R`.
    """
    required_cols = {"model", "prob_cutoff"}
    if risk_cov_df is None or risk_cov_df.empty:
        return pd.DataFrame()
    if not required_cols.issubset(risk_cov_df.columns):
        missing = sorted(required_cols - set(risk_cov_df.columns))
        raise ValueError(
            "Outer-CV risk-coverage input is missing required columns: "
            f"{', '.join(missing)}"
        )

    # If the deployable aggregated schema is already present, use it directly.
    if {"mean_risk", "mean_coverage"}.issubset(risk_cov_df.columns):
        out_cols = ["model", "prob_cutoff", "mean_risk", "mean_coverage"]
        if "mean_kappa" in risk_cov_df.columns:
            out_cols.append("mean_kappa")
        return risk_cov_df[out_cols].copy()

    # Backward-compatible fallback for raw heldout rows.
    # Hybrid selection metric:
    #   mean_risk     := 1 - accuracy            (all accepted samples)
    #   mean_coverage := coverage_known          (seen/known samples only)
    if {"accuracy", "coverage_known"}.issubset(risk_cov_df.columns):
        work = risk_cov_df.dropna(subset=["accuracy", "coverage_known"]).copy()
        if work.empty:
            raise ValueError(
                "Outer-CV risk-coverage file has no finite hybrid metrics "
                "(accuracy / coverage_known)."
            )
        work["risk_for_selection"] = 1.0 - work["accuracy"].astype(float)
        work["coverage_for_selection"] = work["coverage_known"].astype(float)
    elif {"accuracy", "perc_rejected"}.issubset(risk_cov_df.columns):
        work = risk_cov_df.dropna(subset=["accuracy", "perc_rejected"]).copy()
        if work.empty:
            raise ValueError(
                "Outer-CV risk-coverage file has no finite fallback metrics "
                "(accuracy / perc_rejected)."
            )
        work["risk_for_selection"] = 1.0 - work["accuracy"].astype(float)
        # Fallback only when known-coverage column is unavailable.
        work["coverage_for_selection"] = 1.0 - work["perc_rejected"].astype(float)
    else:
        raise ValueError(
            "Outer-CV risk-coverage file must include hybrid columns "
            "(accuracy, coverage_known) or fallback overall columns "
            "(accuracy, perc_rejected)."
        )

    # Aggregate per model+cutoff over held-out folds to match nested-CV reporting.
    summary = (
        work.groupby(["model", "prob_cutoff"], as_index=False)
        .agg(
            mean_risk=("risk_for_selection", "mean"),
            mean_coverage=("coverage_for_selection", "mean"),
        )
    )
    if "kappa" in work.columns:
        kappa_summary = (
            work.groupby(["model", "prob_cutoff"], as_index=False)
            .agg(mean_kappa=("kappa", "mean"))
        )
        summary = summary.merge(kappa_summary, on=["model", "prob_cutoff"], how="left")
    return summary


def choose_cutoffs_from_risk(
    risk_cov_df: pd.DataFrame,
    max_accepted_risk: float
) -> dict:
    """
    Derive per-model cutoffs from a risk–coverage table for a desired risk.

    Matches existing logic: among cutoffs with mean_risk <= max_accepted_risk,
    choose the one with highest mean_coverage (then highest mean_kappa if present).

    Parameters
    ----------
    risk_cov_df : pd.DataFrame
        Data frame with at least columns: model, prob_cutoff, mean_risk,
        mean_coverage (typically aggregated from outer-CV heldout sweeps).
    max_accepted_risk : float
        Maximum accepted error rate on accepted predictions (0–1, e.g. 0.02 for 2%).

    Returns
    -------
    dict
        Mapping from R model name to chosen probability cutoff.
    """
    cutoffs = {}
    if risk_cov_df is None or risk_cov_df.empty:
        return cutoffs

    def _normalize_model_name(model_name: str) -> str:
        name = str(model_name)
        if name.startswith("Global_Optimized"):
            return "Global_Optimized"
        if name.startswith("Global_Product_Optimized"):
            return "Global_Optimized"
        return name

    for model_name, sub in risk_cov_df.groupby("model"):
        sub_ok = sub[sub["mean_risk"] <= max_accepted_risk].copy()
        if not sub_ok.empty:
            max_cov = sub_ok["mean_coverage"].max()
            sub_ok = sub_ok[sub_ok["mean_coverage"] == max_cov]
            if "mean_kappa" in sub_ok.columns:
                max_kappa = sub_ok["mean_kappa"].max()
                sub_ok = sub_ok[sub_ok["mean_kappa"] == max_kappa]
            chosen = sub_ok.iloc[0]
        else:
            min_risk = sub["mean_risk"].min()
            sub_low = sub[sub["mean_risk"] == min_risk].copy()
            max_cov = sub_low["mean_coverage"].max()
            sub_low = sub_low[sub_low["mean_coverage"] == max_cov]
            if "mean_kappa" in sub_low.columns:
                max_kappa = sub_low["mean_kappa"].max()
                sub_low = sub_low[sub_low["mean_kappa"] == max_kappa]
            chosen = sub_low.iloc[0]

        cutoffs[_normalize_model_name(model_name)] = float(chosen["prob_cutoff"])

    return cutoffs


def load_multivariate_params(multivariate_path):
    """
    Load multivariate calibration parameters for global ensemble confidence.

    Parameters
    ----------
    multivariate_path : str
        Path to the multivariate parameters CSV file.

    Returns
    -------
    dict
        Mapping term -> coefficient for Global_Optimized.
    """
    if not os.path.exists(multivariate_path):
        print(f"WARNING: Multivariate parameters file not found at {multivariate_path}")
        return {}

    df = pd.read_csv(multivariate_path)
    if df.empty:
        print(f"WARNING: Multivariate parameters file {multivariate_path} is empty")
        return {}

    df = df[df["model"] == "Global_Optimized"].copy()
    if df.empty:
        print("WARNING: No Global_Optimized rows found in multivariate params")
        return {}

    params = {str(row["term"]): float(row["estimate"]) for _, row in df.iterrows()}
    if "(Intercept)" not in params:
        print("WARNING: (Intercept) missing in multivariate params; defaulting intercept to 0.0")
    print(f"Loaded multivariate parameters with {len(params)} terms from {multivariate_path}")
    return params


def resolve_calibration_method(calibration_method: str) -> str:
    """
    Normalize calibration method aliases.
    """
    aliases = {
        "multivariate": "multivariate",
        "multivariate_two_head": "multivariate",
        "univariate": "univariate",
        "univariate_two_head": "univariate",
    }
    if calibration_method not in aliases:
        raise ValueError(
            "Unsupported calibration method. Use one of: "
            "multivariate, univariate (legacy aliases: *_two_head)."
        )
    return aliases[calibration_method]


def resolve_calibration_setting(calibration_setting: str) -> str:
    """
    Normalize calibration setting aliases.
    """
    aliases = {
        "two_head": "two_head",
        "two_head_postcal": "two_head_postcal",
        "known_only": "known_only",
        "known_only_logit": "known_only_logit",
        "ood_aware": "ood_aware",
        "ood_aware_logit": "ood_aware_logit",
    }
    if calibration_setting not in aliases:
        raise ValueError(
            "Unsupported calibration setting. Use one of: "
            "two_head, two_head_postcal, known_only, known_only_logit, ood_aware, ood_aware_logit."
        )
    return aliases[calibration_setting]


def resolve_ensemble_method(ensemble_method: str) -> str:
    """
    Normalize ensemble method aliases.
    """
    aliases = {
        "product": "product",
        "product_of_experts": "product",
        "poe": "product",
        "weighted": "weighted",
        "weighted_sum": "weighted",
        "sum": "weighted",
    }
    if ensemble_method not in aliases:
        raise ValueError(
            "Unsupported ensemble method. Use one of: "
            "product, weighted."
        )
    return aliases[ensemble_method]


def resolve_deployable_risk_curve_path(
    cutoffs_root: str,
    suffix: str,
    calibration_method: str,
    calibration_setting: str,
    cutoff_curve_split: str = "cv",
) -> str:
    """
    Resolve method-specific deployable risk/seen-coverage curve path.
    """
    method_name = resolve_calibration_method(calibration_method)
    setting_name = resolve_calibration_setting(calibration_setting)
    split_name = str(cutoff_curve_split).lower()
    if split_name not in {"cv", "loso"}:
        raise ValueError("cutoff_curve_split must be either 'cv' or 'loso'.")
    method_file_split = os.path.join(
        str(cutoffs_root),
        f"cutoffs_{suffix}",
        f"risk_seen_coverage_curve_outercv_{method_name}_{setting_name}_{suffix}_{split_name}.csv",
    )
    if os.path.exists(method_file_split):
        return method_file_split
    method_file = os.path.join(
        str(cutoffs_root),
        f"cutoffs_{suffix}",
        f"risk_seen_coverage_curve_outercv_{method_name}_{setting_name}_{suffix}.csv",
    )
    if os.path.exists(method_file):
        return method_file

    # Backward-compatible fallback for earlier exported filenames (univariate).
    if split_name == "cv" and method_name == "univariate" and setting_name == "two_head":
        legacy_file = os.path.join(
            str(cutoffs_root),
            f"cutoffs_{suffix}",
            f"risk_seen_coverage_curve_outercv_{suffix}.csv",
        )
        return legacy_file

    return method_file

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
    print("Making Global Ensemble predictions (product-of-experts)...")
    
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
    
    # Initialize PoE matrix with multiplicative identity.
    ensemble_prob_matrix = np.ones((len(sample_names), len(all_classes)), dtype=np.float64)
    
    # For each model, multiply p(class)^weight (product-of-experts).
    eps = 1e-12
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
        
        # Multiply weighted expert probabilities for each class
        classes_matched = 0
        for j, class_name in enumerate(all_classes):
            # Look up the original column name in this model's probability matrix
            if class_name in col_mapping:
                original_col = col_mapping[class_name]
                probs = np.clip(prob_matrix[original_col].values.astype(np.float64), eps, 1.0)
                ensemble_prob_matrix[:, j] *= np.power(probs, weight)
                classes_matched += 1
        
        print(f"  {model_name}: matched {classes_matched}/{len(all_classes)} classes for weighting")
    
    # Normalize probabilities to sum to 1 for each sample.
    row_sums = ensemble_prob_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    ensemble_prob_matrix = ensemble_prob_matrix / row_sums
    
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


def predict_ensemble_weighted_global(individual_predictions, individual_prob_matrices, ensemble_weights, sample_names):
    """
    Make predictions using weighted-sum global ensemble.
    """
    print("Making Global Ensemble predictions (weighted sum)...")

    weights = ensemble_weights['global'].iloc[0]
    print(f"  Using weights: NN={weights['nn_weight']:.3f}, SVM={weights['svm_weight']:.3f}, XGB={weights['xgb_weight']:.3f}")

    model_col_mappings = {}
    for model_name, prob_matrix in individual_prob_matrices.items():
        if prob_matrix is None:
            continue
        col_mapping = {}
        for col in prob_matrix.columns:
            if col != 'sample_name':
                standardized_col = standardize_class_names([col])[0]
                col_mapping[standardized_col] = col
        model_col_mappings[model_name] = col_mapping
        print(f"  {model_name}: mapped {len(col_mapping)} classes")

    all_classes = set()
    for col_mapping in model_col_mappings.values():
        all_classes.update(col_mapping.keys())
    all_classes = sorted(list(all_classes))
    print(f"  Total unique classes: {len(all_classes)}")

    ensemble_prob_matrix = np.zeros((len(sample_names), len(all_classes)), dtype=np.float64)

    for model_name, prob_matrix in individual_prob_matrices.items():
        if prob_matrix is None:
            continue

        if model_name == 'NN' and weights['nn_weight'] > 0:
            weight = weights['nn_weight']
        elif model_name == 'SVM' and weights['svm_weight'] > 0:
            weight = weights['svm_weight']
        elif model_name == 'XGBOOST' and weights['xgb_weight'] > 0:
            weight = weights['xgb_weight']
        else:
            continue

        col_mapping = model_col_mappings[model_name]
        classes_matched = 0
        for j, class_name in enumerate(all_classes):
            if class_name in col_mapping:
                original_col = col_mapping[class_name]
                probs = prob_matrix[original_col].values.astype(np.float64)
                ensemble_prob_matrix[:, j] += weight * probs
                classes_matched += 1
        print(f"  {model_name}: matched {classes_matched}/{len(all_classes)} classes for weighting")

    row_sums = ensemble_prob_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    ensemble_prob_matrix = ensemble_prob_matrix / row_sums

    max_prob_indices = np.argmax(ensemble_prob_matrix, axis=1)
    max_probs = np.max(ensemble_prob_matrix, axis=1)
    pred_classes = [all_classes[idx] for idx in max_prob_indices]

    results_df = pd.DataFrame({
        'sample_name': sample_names,
        'sample_index': range(len(sample_names)),
        'prediction': pred_classes,
        'prediction_prob': max_probs,
        'prediction_passed_cutoff': False
    })

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
    Merge specific classes in the probability matrix using summed probabilities:
    1. Sum probabilities for all classes with 'MDS' or 'TP53' in their name -> "MDS.r"
    2. Sum probabilities for all other KMT2A classes (excluding MLLT3 fusion) -> "other.KMT2A"
    3. Sum probabilities for MECOM-related classes (GATA2;MECOM, MECOM other) -> "MECOM"

    After merging, row probabilities are renormalized to sum to 1.
    Matches R logic in utility_functions.R (merge_prob_method = "sum").

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

    # Identify classes to merge for MDS/TP53
    mds_classes = [
        col for col in class_columns
        if 'mds' in col.lower() or 'tp53' in col.lower()
    ]

    # Identify classes to merge for other KMT2A (excluding MLLT3)
    other_kmt2a_classes = [
        col for col in class_columns
        if 'kmt2a' in col.lower() and 'mllt3' not in col.lower()
    ]

    # Identify classes to merge for MECOM (e.g. GATA2;MECOM, MECOM other)
    mecom_classes = [
        col for col in class_columns
        if 'mecom' in col.lower() and ('gata2' in col.lower() or 'other' in col.lower())
    ]

    if mds_classes:
        prob_matrix_df['MDS.r'] = prob_matrix_df[mds_classes].sum(axis=1)
        prob_matrix_df = prob_matrix_df.drop(columns=mds_classes)

    if other_kmt2a_classes:
        prob_matrix_df['other.KMT2A'] = prob_matrix_df[other_kmt2a_classes].sum(axis=1)
        prob_matrix_df = prob_matrix_df.drop(columns=other_kmt2a_classes)

    if mecom_classes:
        prob_matrix_df['MECOM'] = prob_matrix_df[mecom_classes].sum(axis=1)
        prob_matrix_df = prob_matrix_df.drop(columns=mecom_classes)

    # Renormalize so each row sums to 1 (class columns only)
    class_cols = [c for c in prob_matrix_df.columns if c != 'sample_name']
    if class_cols:
        row_sums = prob_matrix_df[class_cols].sum(axis=1)
        row_sums = row_sums.replace(0, 1)
        prob_matrix_df[class_cols] = prob_matrix_df[class_cols].div(row_sums, axis=0)

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
    
    # Final deployment path is ensemble-only.
    cutoff_mapping = {"Global_Ensemble": "Global_Optimized"}

    for model_name, df in predictions_dict.items():
        if model_name != "Global_Ensemble":
            continue
        cutoff_key = cutoff_mapping.get(model_name, model_name)

        # Enforce multivariate-calibrated confidence only for final predictor.
        score_col = "prediction_prob_calibrated"
        if score_col not in df.columns:
            raise ValueError(
                "Global_Ensemble requires calibrated confidence "
                "(prediction_prob_calibrated), but it is missing."
            )

        if cutoff_key in cutoffs:
            cutoff_value = cutoffs[cutoff_key]
            df['prediction_passed_cutoff'] = df[score_col] >= cutoff_value
            print(f"Applied cutoff {cutoff_value:.3f} to {model_name} using {score_col}")
        else:
            print(f"No cutoff found for {model_name}")
            df['prediction_passed_cutoff'] = True  # Default to True if no cutoff
    
    return predictions_dict


def _build_two_head_feature_map(global_prob_df, individual_prob_matrices):
    """
    Build multivariate rejection features for two-head confidence scoring.
    """
    class_cols = [c for c in global_prob_df.columns if c != "sample_name"]
    if not class_cols:
        return None

    prob_mat = global_prob_df[class_cols].to_numpy(dtype=np.float64)
    top1_idx = np.argmax(prob_mat, axis=1)
    top1_prob = prob_mat[np.arange(prob_mat.shape[0]), top1_idx]

    if prob_mat.shape[1] > 1:
        part = np.partition(prob_mat, -2, axis=1)
        top2_prob = part[:, -2]
    else:
        top2_prob = np.zeros(prob_mat.shape[0], dtype=np.float64)
    margin = top1_prob - top2_prob

    clipped = np.clip(prob_mat, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1) / np.log(max(prob_mat.shape[1], 2))
    entropy = np.clip(entropy, 0.0, 1.0)

    per_model_top1_prob = []
    per_model_top1_class = []
    for model_name in ("NN", "SVM", "XGBOOST"):
        df = individual_prob_matrices.get(model_name)
        if df is None:
            continue

        mapping = {standardize_class_names([c])[0]: c for c in df.columns if c != "sample_name"}
        model_class_cols = [c for c in df.columns if c != "sample_name"]
        model_top1_prob = np.zeros(prob_mat.shape[0], dtype=np.float64)
        model_top1_cls = np.array([""] * prob_mat.shape[0], dtype=object)

        for i, cls_idx in enumerate(top1_idx):
            cls_name = class_cols[cls_idx]
            orig_col = mapping.get(cls_name)
            if orig_col is not None:
                model_top1_prob[i] = float(df.iloc[i][orig_col])
            if model_class_cols:
                row_vals = df.iloc[i][model_class_cols].to_numpy(dtype=np.float64)
                top_local_idx = int(np.argmax(row_vals))
                model_top1_cls[i] = standardize_class_names([model_class_cols[top_local_idx]])[0]

        per_model_top1_prob.append(model_top1_prob)
        per_model_top1_class.append(model_top1_cls)

    if len(per_model_top1_prob) >= 2:
        top1_var = np.var(np.column_stack(per_model_top1_prob), axis=1)
    else:
        top1_var = np.zeros(prob_mat.shape[0], dtype=np.float64)

    if per_model_top1_class:
        ensemble_top1_class = np.array([class_cols[i] for i in top1_idx], dtype=object)
        agree_counts = np.zeros(prob_mat.shape[0], dtype=np.float64)
        for model_preds in per_model_top1_class:
            agree_counts += (model_preds == ensemble_top1_class).astype(np.float64)
        n_models_agree = agree_counts
    else:
        n_models_agree = np.zeros(prob_mat.shape[0], dtype=np.float64)

    top1_clipped = np.clip(top1_prob, 1e-6, 1.0 - 1e-6)
    return {
        "max_prob": top1_prob,
        "logit_max_prob": np.log(top1_clipped / (1.0 - top1_clipped)),
        "margin": margin,
        "entropy": entropy,
        "n_models_agree": n_models_agree,
        "top1_prob_variance_across_models": top1_var,
    }


def _score_logistic_head(feature_map, params):
    """
    Score a logistic regression head from saved R GLM coefficients.
    """
    if not params:
        return np.zeros(len(next(iter(feature_map.values()))), dtype=np.float64)
    linear = np.full(
        len(next(iter(feature_map.values()))),
        params.get("(Intercept)", 0.0),
        dtype=np.float64,
    )
    for term, values in feature_map.items():
        if term in params:
            linear += float(params[term]) * values
    return 1.0 / (1.0 + np.exp(-linear))


def apply_global_two_head_product_confidence(
    global_pred_df,
    global_prob_df,
    individual_prob_matrices,
    correctness_params,
    ood_head_params,
):
    """
    Apply multivariate two-head confidence: P(correct|ID) * P(ID).
    """
    feature_map = _build_two_head_feature_map(global_prob_df, individual_prob_matrices)
    if feature_map is None:
        return global_pred_df

    p_correct = _score_logistic_head(feature_map, correctness_params)
    p_id = _score_logistic_head(feature_map, ood_head_params)
    global_pred_df["prediction_prob_calibrated"] = p_correct * p_id
    global_pred_df["prediction_prob_correct_head"] = p_correct
    global_pred_df["prediction_prob_id_head"] = p_id
    return global_pred_df


def apply_global_single_head_confidence(
    global_pred_df,
    global_prob_df,
    individual_prob_matrices,
    single_head_params,
):
    """
    Apply single-head confidence: P(target | features),
    where target is either correctness (known_only*) or correctness-and-ID (ood_aware*).
    """
    feature_map = _build_two_head_feature_map(global_prob_df, individual_prob_matrices)
    if feature_map is None:
        return global_pred_df
    p_target = _score_logistic_head(feature_map, single_head_params)
    global_pred_df["prediction_prob_calibrated"] = p_target
    return global_pred_df


def apply_two_head_postcalibration(global_pred_df, postcal_params):
    """
    Apply a final logistic recalibration on top of two-head product score.
    """
    if not postcal_params:
        raise ValueError("two_head_postcal requires post-calibration parameters.")
    if "prediction_prob_calibrated" not in global_pred_df.columns:
        raise ValueError("two_head_postcal requires prediction_prob_calibrated from two_head score.")

    eps = 1e-6
    raw_score = np.clip(global_pred_df["prediction_prob_calibrated"].to_numpy(dtype=np.float64), eps, 1 - eps)
    logit_score = np.log(raw_score / (1.0 - raw_score))
    intercept = float(postcal_params.get("(Intercept)", 0.0))
    slope = float(postcal_params.get("logit_score", 1.0))
    linear = intercept + slope * logit_score
    calibrated = 1.0 / (1.0 + np.exp(-linear))
    global_pred_df["prediction_prob_two_head_raw"] = raw_score
    global_pred_df["prediction_prob_calibrated"] = calibrated
    return global_pred_df


def save_predictions(predictions_dict, prob_matrices_dict, output_dir, input_filename_prefix, merge_suffix=""):
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
    merge_suffix : str
        Suffix to append to filenames (e.g., "_merged" or "_unmerged")
    """
    print(f"Saving predictions to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prediction summaries
    for model_name, df in predictions_dict.items():
        filename = f"{input_filename_prefix}_{model_name}_predictions{merge_suffix}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {model_name} predictions to {filename}")
    
    # Save full probability matrices
    for model_name, df in prob_matrices_dict.items():
        filename = f"{input_filename_prefix}_{model_name}_probability_matrix{merge_suffix}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {model_name} probability matrix to {filename}")


def run_predictions(
    X,
    sample_names,
    models,
    ensemble_weights,
    cutoffs,
    multivariate_params=None,
    ood_head_params=None,
    postcal_params=None,
    calibration_method="multivariate",
    calibration_setting="two_head",
    ensemble_method="product",
    merge_classes=False,
):
    """
    Run prediction pipeline for a single version (merged or unmerged).
    
    Parameters:
    -----------
    X : np.ndarray
        Input data (samples x genes)
    sample_names : list
        Sample identifiers
    models : dict
        Dictionary containing loaded models
    ensemble_weights : dict
        Dictionary containing ensemble weights
    cutoffs : dict
        Dictionary containing cutoffs for each model
    multivariate_params : dict
        Coefficients for the correctness head, P(correct | ID, features)
    ood_head_params : dict
        Coefficients for the OOD head, P(ID | features)
    postcal_params : dict
        Coefficients for optional post-calibration on two-head product score
    calibration_method : str
        Confidence model family. Supported:
        - multivariate (default)
        - univariate
    calibration_setting : str
        Calibration target setting. Supported:
        - two_head
        - two_head_postcal
        - known_only
        - known_only_logit
        - ood_aware
        - ood_aware_logit
    ensemble_method : str
        Ensemble aggregation method. Supported:
        - product (product-of-experts)
        - weighted (weighted sum)
    merge_classes : bool
        Whether to merge classes in probability matrices
        
    Returns:
    --------
    predictions : dict
        Dictionary of prediction DataFrames
    prob_matrices : dict
        Dictionary of probability matrix DataFrames
    """
    print(f"\n{'='*60}")
    print(f"Running predictions ({'MERGED' if merge_classes else 'UNMERGED'} classes)")
    print(f"{'='*60}")
    
    # Build individual model probabilities only as ensemble experts.
    predictions = {}
    prob_matrices = {}
    
    # Individual model predictions
    if 'NN' in models:
        predictions['NN'], prob_matrices['NN'] = predict_nn_standard(X, models, sample_names)
    
    if 'SVM' in models:
        predictions['SVM'], prob_matrices['SVM'] = predict_ovr_models(X, models, 'SVM', sample_names)
    
    if 'XGBOOST' in models:
        predictions['XGBOOST'], prob_matrices['XGBOOST'] = predict_ovr_models(X, models, 'XGBOOST', sample_names)
    
    # Apply class merging to expert probability matrices if requested.
    if merge_classes:
        for model_name in prob_matrices.keys():
            prob_matrices[model_name] = merge_probability_classes(prob_matrices[model_name].copy())
    
    ensemble_method = resolve_ensemble_method(ensemble_method)

    # Ensemble predictions
    if 'global' in ensemble_weights:
        if ensemble_method == "product":
            predictions['Global_Ensemble'], prob_matrices['Global_Ensemble'] = predict_ensemble_global(
                predictions, prob_matrices, ensemble_weights, sample_names
            )
        else:
            predictions['Global_Ensemble'], prob_matrices['Global_Ensemble'] = predict_ensemble_weighted_global(
                predictions, prob_matrices, ensemble_weights, sample_names
            )
    else:
        raise ValueError("Global ensemble weights are required for prediction.")

    # Apply class merging to ensemble probability matrices if requested
    if merge_classes and 'Global_Ensemble' in prob_matrices:
        prob_matrices['Global_Ensemble'] = merge_probability_classes(prob_matrices['Global_Ensemble'].copy())
    
    calibration_method = resolve_calibration_method(calibration_method)
    calibration_setting = resolve_calibration_setting(calibration_setting)

    # Apply selected calibration to global ensemble only.
    if not multivariate_params:
        raise ValueError(
            "Calibration parameters are required for final "
            "ensemble predictions."
        )
    if calibration_setting in {"two_head", "two_head_postcal"}:
        if not ood_head_params:
            raise ValueError(
                f"OOD-head parameters are required for {calibration_setting} ensemble predictions."
            )
        print(
            f"Applying {calibration_method}/{calibration_setting} confidence to "
            f"{ensemble_method} global ensemble..."
        )
        predictions["Global_Ensemble"] = apply_global_two_head_product_confidence(
            predictions["Global_Ensemble"],
            prob_matrices["Global_Ensemble"],
            prob_matrices,
            multivariate_params,
            ood_head_params,
        )
        if calibration_setting == "two_head_postcal":
            predictions["Global_Ensemble"] = apply_two_head_postcalibration(
                predictions["Global_Ensemble"],
                postcal_params,
            )
    else:
        print(
            f"Applying {calibration_method}/{calibration_setting} confidence to "
            f"{ensemble_method} global ensemble..."
        )
        predictions["Global_Ensemble"] = apply_global_single_head_confidence(
            predictions["Global_Ensemble"],
            prob_matrices["Global_Ensemble"],
            prob_matrices,
            multivariate_params,
        )

    # Apply cutoffs to predictions (uses calibrated prob if present)
    predictions = apply_cutoffs(predictions, cutoffs)

    # Final deployment output is ensemble-only.
    ensemble_only_predictions = {"Global_Ensemble": predictions["Global_Ensemble"]}
    ensemble_only_prob_matrices = {"Global_Ensemble": prob_matrices["Global_Ensemble"]}
    return ensemble_only_predictions, ensemble_only_prob_matrices


def main():
    parser = argparse.ArgumentParser(description="Make predictions on new samples using trained models")
    parser.add_argument("--input_file", required=True, help="Path to input CSV file with new samples")
    parser.add_argument("--output_dir", required=True, help="Output directory for prediction results")
    parser.add_argument("--models_dir", default=None, help="Path to final_models directory")
    parser.add_argument("--weights_dir", default=None, help="Path to final_train_test directory (will look for ensemble_weights_merged_summed and ensemble_weights_unmerged_maxprob subdirs)")
    parser.add_argument("--cutoffs_file", default=None, help="Path to final_train_test directory (will look for cutoffs_merged_summed and cutoffs_unmerged_maxprob subdirs)")
    parser.add_argument("--pipelines_dir", default=None, help="Path to pipelines cache directory")
    parser.add_argument(
        "--max_accepted_risk_pct",
        type=float,
        default=None,
        help="Maximum accepted error rate (percentage) on accepted predictions; "
             "if set, cutoffs will be derived from deployable risk/seen-coverage "
             "curve CSVs in cutoffs_* (e.g. 5 for 5%%)."
    )
    parser.add_argument(
        "--cutoff_curve_split",
        default="cv",
        choices=["cv", "loso"],
        help="Which outer-CV split to use for deployable cutoffs: cv (default) or loso."
    )
    parser.add_argument(
        "--calibration_method",
        default="multivariate",
        help="Calibration method for confidence scoring. Supported: "
             "multivariate (default), univariate."
    )
    parser.add_argument(
        "--calibration_setting",
        default="two_head",
        help="Calibration setting. Supported: two_head (default), two_head_postcal, known_only, known_only_logit, ood_aware, ood_aware_logit."
    )
    parser.add_argument(
        "--ensemble_method",
        default="product",
        help="Ensemble method. Supported: product (default), weighted."
    )
    parser.add_argument("--merged_only", action="store_true", help="Only run merged predictions")
    parser.add_argument("--unmerged_only", action="store_true", help="Only run unmerged predictions")
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    base_path = Path(__file__).resolve().parent.parent
    
    if args.models_dir is None:
        args.models_dir = base_path / "data" / "out" / "final_models"
    
    if args.weights_dir is None:
        args.weights_dir = base_path / "data" / "out" / "final_train_test"
    
    if args.cutoffs_file is None:
        # Base directory for cutoffs (will append _merged or _unmerged)
        args.cutoffs_file = base_path / "data" / "out" / "final_train_test"

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
    print(f"Weights base directory: {args.weights_dir}")
    print(f"Cutoffs base directory: {args.cutoffs_file}")
    print(f"Pipelines directory: {pipelines_dir}")
    print(f"Calibration method: {resolve_calibration_method(args.calibration_method)}")
    print(f"Calibration setting: {resolve_calibration_setting(args.calibration_setting)}")
    print(f"Ensemble method: {resolve_ensemble_method(args.ensemble_method)}")
    if args.max_accepted_risk_pct is not None:
        print(f"Maximum accepted risk (on accepted predictions): {args.max_accepted_risk_pct:.2f}%")
        print(f"Cutoff curve split source: {args.cutoff_curve_split}")
    
    # Determine which versions to run
    run_merged = not args.unmerged_only
    run_unmerged = not args.merged_only
    calibration_method = resolve_calibration_method(args.calibration_method)
    calibration_setting = resolve_calibration_setting(args.calibration_setting)
    ensemble_method = resolve_ensemble_method(args.ensemble_method)
    
    if args.merged_only and args.unmerged_only:
        print("ERROR: Cannot specify both --merged_only and --unmerged_only")
        return
    
    # Load new samples (only need to load once)
    X, sample_names = load_new_samples(args.input_file)
    
    # Load models and metadata with pipeline cache (only need to load once)
    models = load_models_and_metadata(args.models_dir, pipelines_dir)
    
    # Run predictions for unmerged version
    if run_unmerged:
        print("\n" + "="*60)
        print("UNMERGED VERSION")
        print("="*60)
        
        # Load unmerged ensemble weights (matches R: ensemble_weights_unmerged_maxprob)
        # Structure: final_train_test/ensemble_weights_unmerged_maxprob/cv/
        weights_dir_unmerged = os.path.join(str(args.weights_dir), "ensemble_weights_unmerged_maxprob")
        print(f"\nLoading unmerged ensemble weights from: {weights_dir_unmerged}")
        ensemble_weights_unmerged = load_ensemble_weights(weights_dir_unmerged)
        
        # Deployable risk/seen-coverage curve exported from outer-CV sweeps.
        risk_cov_file_unmerged = resolve_deployable_risk_curve_path(
            args.cutoffs_file,
            "unmerged_maxprob",
            calibration_method,
            calibration_setting,
            cutoff_curve_split=args.cutoff_curve_split,
        )

        # Use cutoff only when user requests risk-based selection.
        cutoffs_unmerged = {}
        if args.max_accepted_risk_pct is not None:
            print(f"\nLoading unmerged deployable risk/seen-coverage curve from: {risk_cov_file_unmerged}")
            risk_cov_unmerged = load_risk_coverage(risk_cov_file_unmerged, required=True)
            if risk_cov_unmerged is not None:
                max_risk = args.max_accepted_risk_pct / 100.0
                risk_cov_unmerged_summary = prepare_risk_curve_for_selection(risk_cov_unmerged)
                rc_cutoffs = choose_cutoffs_from_risk(risk_cov_unmerged_summary, max_risk)
                if rc_cutoffs:
                    print("Using risk-based cutoffs (unmerged) derived from deployable risk/seen-coverage curve")
                    cutoffs_unmerged = rc_cutoffs
                else:
                    raise ValueError(
                        "Could not derive unmerged risk-based cutoffs from "
                        f"{risk_cov_file_unmerged}"
                    )
        else:
            print("\nNo --max_accepted_risk_pct provided: not applying unmerged cutoff.")

        if calibration_method == "multivariate":
            params_subdir_unmerged = "multivariate_params_unmerged_maxprob"
            if calibration_setting in {"two_head", "two_head_postcal"}:
                correctness_file_unmerged = "multivariate_params_unmerged_maxprob.csv"
                ood_file_unmerged = "ood_head_params_unmerged_maxprob.csv"
                postcal_file_unmerged = "two_head_postcal_params_unmerged_maxprob.csv"
            elif calibration_setting == "known_only":
                correctness_file_unmerged = "multivariate_params_known_only_unmerged_maxprob.csv"
                ood_file_unmerged = None
                postcal_file_unmerged = None
            elif calibration_setting == "known_only_logit":
                correctness_file_unmerged = "multivariate_params_known_only_logit_unmerged_maxprob.csv"
                ood_file_unmerged = None
                postcal_file_unmerged = None
            elif calibration_setting == "ood_aware":
                correctness_file_unmerged = "multivariate_params_ood_aware_unmerged_maxprob.csv"
                ood_file_unmerged = None
                postcal_file_unmerged = None
            elif calibration_setting == "ood_aware_logit":
                correctness_file_unmerged = "multivariate_params_ood_aware_logit_unmerged_maxprob.csv"
                ood_file_unmerged = None
                postcal_file_unmerged = None
            else:
                raise ValueError(f"Unsupported calibration setting for multivariate: {calibration_setting}")
        else:
            params_subdir_unmerged = "univariate_params_unmerged_maxprob"
            if calibration_setting in {"two_head", "two_head_postcal"}:
                correctness_file_unmerged = "univariate_params_unmerged_maxprob.csv"
                ood_file_unmerged = "ood_head_params_univariate_unmerged_maxprob.csv"
                postcal_file_unmerged = "two_head_postcal_params_unmerged_maxprob.csv"
            elif calibration_setting == "known_only":
                correctness_file_unmerged = "univariate_params_known_only_unmerged_maxprob.csv"
                ood_file_unmerged = None
                postcal_file_unmerged = None
            elif calibration_setting == "known_only_logit":
                correctness_file_unmerged = "univariate_params_known_only_logit_unmerged_maxprob.csv"
                ood_file_unmerged = None
                postcal_file_unmerged = None
            elif calibration_setting == "ood_aware":
                correctness_file_unmerged = "univariate_params_ood_aware_unmerged_maxprob.csv"
                ood_file_unmerged = None
                postcal_file_unmerged = None
            elif calibration_setting == "ood_aware_logit":
                correctness_file_unmerged = "univariate_params_ood_aware_logit_unmerged_maxprob.csv"
                ood_file_unmerged = None
                postcal_file_unmerged = None
            else:
                raise ValueError(f"Unsupported calibration setting for univariate: {calibration_setting}")

        # Load selected correctness-head parameters for global ensemble.
        multivariate_file_unmerged = os.path.join(
            str(args.cutoffs_file),
            params_subdir_unmerged,
            correctness_file_unmerged,
        )
        print(f"\nLoading unmerged correctness-head parameters from: {multivariate_file_unmerged}")
        multivariate_params_unmerged = load_multivariate_params(multivariate_file_unmerged)

        # Load selected OOD head parameters only for two-head calibration.
        ood_head_params_unmerged = {}
        if ood_file_unmerged is not None:
            ood_head_file_unmerged = os.path.join(
                str(args.cutoffs_file),
                params_subdir_unmerged,
                ood_file_unmerged,
            )
            print(f"Loading unmerged OOD head parameters from: {ood_head_file_unmerged}")
            ood_head_params_unmerged = load_multivariate_params(ood_head_file_unmerged)

        postcal_params_unmerged = {}
        if postcal_file_unmerged is not None and calibration_setting == "two_head_postcal":
            postcal_file_path_unmerged = os.path.join(
                str(args.cutoffs_file),
                params_subdir_unmerged,
                postcal_file_unmerged,
            )
            print(f"Loading unmerged two-head postcal parameters from: {postcal_file_path_unmerged}")
            postcal_params_unmerged = load_multivariate_params(postcal_file_path_unmerged)
        
        # Run predictions
        predictions_unmerged, prob_matrices_unmerged = run_predictions(
            X,
            sample_names,
            models,
            ensemble_weights_unmerged,
            cutoffs_unmerged,
            multivariate_params_unmerged,
            ood_head_params_unmerged,
            postcal_params_unmerged,
            calibration_method=calibration_method,
            calibration_setting=calibration_setting,
            ensemble_method=ensemble_method,
            merge_classes=False,
        )
        
        # Save unmerged predictions
        save_predictions(predictions_unmerged, prob_matrices_unmerged, output_dir, input_filename, merge_suffix="_unmerged")
    
    # Run predictions for merged version (summed method; matches R train_test_analysis)
    if run_merged:
        print("\n" + "="*60)
        print("MERGED VERSION (Summed Method)")
        print("="*60)
        
        # Load merged ensemble weights (summed method)
        # Structure: final_train_test/ensemble_weights_merged_summed/cv/
        weights_dir_merged = os.path.join(str(args.weights_dir), "ensemble_weights_merged_summed")
        print(f"\nLoading merged ensemble weights (summed) from: {weights_dir_merged}")
        ensemble_weights_merged = load_ensemble_weights(weights_dir_merged)
        
        # Deployable risk/seen-coverage curve exported from outer-CV sweeps.
        risk_cov_file_merged = resolve_deployable_risk_curve_path(
            args.cutoffs_file,
            "merged_summed",
            calibration_method,
            calibration_setting,
            cutoff_curve_split=args.cutoff_curve_split,
        )

        # Use cutoff only when user requests risk-based selection.
        cutoffs_merged = {}
        if args.max_accepted_risk_pct is not None:
            print(f"\nLoading merged deployable risk/seen-coverage curve (summed) from: {risk_cov_file_merged}")
            risk_cov_merged = load_risk_coverage(risk_cov_file_merged, required=True)
            if risk_cov_merged is not None:
                max_risk = args.max_accepted_risk_pct / 100.0
                risk_cov_merged_summary = prepare_risk_curve_for_selection(risk_cov_merged)
                rc_cutoffs = choose_cutoffs_from_risk(risk_cov_merged_summary, max_risk)
                if rc_cutoffs:
                    print("Using risk-based cutoffs (merged) derived from deployable risk/seen-coverage curve")
                    cutoffs_merged = rc_cutoffs
                else:
                    raise ValueError(
                        "Could not derive merged risk-based cutoffs from "
                        f"{risk_cov_file_merged}"
                    )
        else:
            print("\nNo --max_accepted_risk_pct provided: not applying merged cutoff.")

        if calibration_method == "multivariate":
            params_subdir_merged = "multivariate_params_merged_summed"
            if calibration_setting in {"two_head", "two_head_postcal"}:
                correctness_file_merged = "multivariate_params_merged_summed.csv"
                ood_file_merged = "ood_head_params_merged_summed.csv"
                postcal_file_merged = "two_head_postcal_params_merged_summed.csv"
            elif calibration_setting == "known_only":
                correctness_file_merged = "multivariate_params_known_only_merged_summed.csv"
                ood_file_merged = None
                postcal_file_merged = None
            elif calibration_setting == "known_only_logit":
                correctness_file_merged = "multivariate_params_known_only_logit_merged_summed.csv"
                ood_file_merged = None
                postcal_file_merged = None
            elif calibration_setting == "ood_aware":
                correctness_file_merged = "multivariate_params_ood_aware_merged_summed.csv"
                ood_file_merged = None
                postcal_file_merged = None
            elif calibration_setting == "ood_aware_logit":
                correctness_file_merged = "multivariate_params_ood_aware_logit_merged_summed.csv"
                ood_file_merged = None
                postcal_file_merged = None
            else:
                raise ValueError(f"Unsupported calibration setting for multivariate: {calibration_setting}")
        else:
            params_subdir_merged = "univariate_params_merged_summed"
            if calibration_setting in {"two_head", "two_head_postcal"}:
                correctness_file_merged = "univariate_params_merged_summed.csv"
                ood_file_merged = "ood_head_params_univariate_merged_summed.csv"
                postcal_file_merged = "two_head_postcal_params_merged_summed.csv"
            elif calibration_setting == "known_only":
                correctness_file_merged = "univariate_params_known_only_merged_summed.csv"
                ood_file_merged = None
                postcal_file_merged = None
            elif calibration_setting == "known_only_logit":
                correctness_file_merged = "univariate_params_known_only_logit_merged_summed.csv"
                ood_file_merged = None
                postcal_file_merged = None
            elif calibration_setting == "ood_aware":
                correctness_file_merged = "univariate_params_ood_aware_merged_summed.csv"
                ood_file_merged = None
                postcal_file_merged = None
            elif calibration_setting == "ood_aware_logit":
                correctness_file_merged = "univariate_params_ood_aware_logit_merged_summed.csv"
                ood_file_merged = None
                postcal_file_merged = None
            else:
                raise ValueError(f"Unsupported calibration setting for univariate: {calibration_setting}")

        # Load selected correctness-head parameters for global ensemble.
        multivariate_file_merged = os.path.join(
            str(args.cutoffs_file),
            params_subdir_merged,
            correctness_file_merged,
        )
        print(f"\nLoading merged correctness-head parameters (summed) from: {multivariate_file_merged}")
        multivariate_params_merged = load_multivariate_params(multivariate_file_merged)

        # Load selected OOD head parameters only for two-head calibration.
        ood_head_params_merged = {}
        if ood_file_merged is not None:
            ood_head_file_merged = os.path.join(
                str(args.cutoffs_file),
                params_subdir_merged,
                ood_file_merged,
            )
            print(f"Loading merged OOD head parameters (summed) from: {ood_head_file_merged}")
            ood_head_params_merged = load_multivariate_params(ood_head_file_merged)

        postcal_params_merged = {}
        if postcal_file_merged is not None and calibration_setting == "two_head_postcal":
            postcal_file_path_merged = os.path.join(
                str(args.cutoffs_file),
                params_subdir_merged,
                postcal_file_merged,
            )
            print(f"Loading merged two-head postcal parameters (summed) from: {postcal_file_path_merged}")
            postcal_params_merged = load_multivariate_params(postcal_file_path_merged)
        
        # Run predictions
        predictions_merged, prob_matrices_merged = run_predictions(
            X,
            sample_names,
            models,
            ensemble_weights_merged,
            cutoffs_merged,
            multivariate_params_merged,
            ood_head_params_merged,
            postcal_params_merged,
            calibration_method=calibration_method,
            calibration_setting=calibration_setting,
            ensemble_method=ensemble_method,
            merge_classes=True,
        )
        
        # Save merged predictions
        save_predictions(predictions_merged, prob_matrices_merged, output_dir, input_filename, merge_suffix="_merged_summed")

    print("\n" + "="*60)
    print("Prediction pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
