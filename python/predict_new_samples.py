#!/usr/bin/env python3
"""
Prediction script for new samples using trained final models.

Loads final SVM models and rejection models exported by
R/calibration_reject_models_final.R (single-head only):
  - svm_single_head: max_prob GLM
  - svm_ridge_in_model: elastic-net on in-model confidence features

Usage:
    python predict_new_samples.py --input_file path/to/new_samples.csv --output_dir path/to/output/
    python predict_new_samples.py --rejector_mode all ...
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
import train_test
from enet_rejector_scoring import enet_params_file_key, score_logistic_head

# Final deployment: SVM classifier + SVM rejector recipes (R/calibration_reject_deploy_config.R).
DEPLOY_BASE_MODEL = "svm"
DEPLOYMENT_PREDICTOR = "SVM"
# Final deployment uses LOSO-selected hyperparameters (model filenames).
FINAL_FOLD_TYPE = "loso"

# Rejector cutoff tracks (parallel calibration exports).
CUTOFF_SOURCE_SELECTION = "selection_loso"
CUTOFF_SOURCE_DEPLOY_LOSO = "deploy_loso"
VALID_CUTOFF_SOURCES = (CUTOFF_SOURCE_SELECTION, CUTOFF_SOURCE_DEPLOY_LOSO)

THRESHOLD_METHOD_JACKKNIFE = "jackknife_adjusted"
THRESHOLD_METHOD_POOLED = "pooled_oof"
THRESHOLD_METHOD_UCB95 = "ucb_95"
VALID_THRESHOLD_METHODS = (
    THRESHOLD_METHOD_JACKKNIFE,
    THRESHOLD_METHOD_POOLED,
    THRESHOLD_METHOD_UCB95,
)


def resolve_ensemble_method(ensemble_method: str) -> str:
    """Normalize ensemble method aliases."""
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


def resolve_cutoff_sources(cutoff_source_arg: str) -> list:
    if cutoff_source_arg == "both":
        return [CUTOFF_SOURCE_SELECTION, CUTOFF_SOURCE_DEPLOY_LOSO]
    if cutoff_source_arg in VALID_CUTOFF_SOURCES:
        return [cutoff_source_arg]
    raise ValueError(
        f"Invalid cutoff_source '{cutoff_source_arg}'. "
        f"Use one of: {', '.join(VALID_CUTOFF_SOURCES)}, both."
    )


def resolve_threshold_methods(threshold_method_arg: str) -> list:
    if threshold_method_arg == "both":
        return list(VALID_THRESHOLD_METHODS)
    if threshold_method_arg in VALID_THRESHOLD_METHODS:
        return [threshold_method_arg]
    raise ValueError(
        f"Invalid threshold_method '{threshold_method_arg}'. "
        f"Use one of: {', '.join(VALID_THRESHOLD_METHODS)}, both."
    )


def calibration_artifact_suffix(label_set_key: str, cutoff_source: str) -> str:
    base = final_merge_suffix(label_set_key)
    if cutoff_source == CUTOFF_SOURCE_DEPLOY_LOSO:
        return f"{base}_deploy_loso"
    return base


def prediction_output_tag(cutoff_source: str) -> str:
    return "_deploy_loso" if cutoff_source == CUTOFF_SOURCE_DEPLOY_LOSO else ""


def prediction_threshold_tag(threshold_method: str) -> str:
    """Filename suffix for non-default cutoff derivation (jackknife has no suffix)."""
    if threshold_method == THRESHOLD_METHOD_POOLED:
        return "_pooled_oof"
    if threshold_method == THRESHOLD_METHOD_JACKKNIFE:
        return ""
    if threshold_method == THRESHOLD_METHOD_UCB95:
        return "_ucb_95"
    raise ValueError(f"Unsupported threshold_method for output naming: {threshold_method}")


def prediction_risk_tag(max_accepted_risk_pct) -> str:
    """Filename suffix when cutoffs are chosen from a risk-coverage curve."""
    if max_accepted_risk_pct is None:
        return ""
    whole = int(max_accepted_risk_pct)
    frac = int(round((float(max_accepted_risk_pct) - whole) * 10))
    return f"_risk{whole}p{frac}"


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
        model_path = os.path.join(nn_dir, f"NN_final_{FINAL_FOLD_TYPE}_standard_model_0.pkl")
        with open(model_path, 'rb') as f:
            nn_model = joblib.load(f)
        
        # Load label mapping
        label_mapping_path = os.path.join(nn_dir, f"label_mapping_NN_{FINAL_FOLD_TYPE}_standard.json")
        with open(label_mapping_path, 'r') as f:
            nn_label_mapping = json.load(f)
        
        # Load metadata
        metadata_path = os.path.join(nn_dir, f"NN_final_{FINAL_FOLD_TYPE}_standard_model_0_metadata.json")
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
            pipeline_path = os.path.join(nn_dir, f"pipeline_NN_{FINAL_FOLD_TYPE}_standard.pkl")
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
        label_mapping_path = os.path.join(svm_dir, f"label_mapping_SVM_{FINAL_FOLD_TYPE}_OvR.json")
        with open(label_mapping_path, 'r') as f:
            svm_label_mapping = json.load(f)
        
        # Load all class-specific models and their pipelines
        svm_models = {}
        svm_metadata = {}
        svm_pipelines = {}  # Store pipeline for each class
        
        for file in os.listdir(svm_dir):
            if file.endswith('.pkl') and 'class_' in file:
                # Extract class name from filename
                class_name = file.replace(f'SVM_final_{FINAL_FOLD_TYPE}_OvR_class_', '').replace('.pkl', '')
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
            pipeline_path = os.path.join(svm_dir, f"pipeline_SVM_{FINAL_FOLD_TYPE}_OvR.pkl")
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
        label_mapping_path = os.path.join(xgb_dir, f"label_mapping_XGBOOST_{FINAL_FOLD_TYPE}_OvR.json")
        with open(label_mapping_path, 'r') as f:
            xgb_label_mapping = json.load(f)
        
        # Load all class-specific models and their pipelines
        xgb_models = {}
        xgb_metadata = {}
        xgb_pipelines = {}  # Store pipeline for each class
        
        for file in os.listdir(xgb_dir):
            if file.endswith('.pkl') and 'class_' in file:
                # Extract class name from filename
                class_name = file.replace(f'XGBOOST_final_{FINAL_FOLD_TYPE}_OvR_class_', '').replace('.pkl', '')
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
            pipeline_path = os.path.join(xgb_dir, f"pipeline_XGBOOST_{FINAL_FOLD_TYPE}_OvR.pkl")
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


def load_ensemble_weights(weights_dir, ensemble_method="product"):
    """
    Load ensemble weights for the global ensemble method.
    
    Parameters:
    -----------
    weights_dir : str
        Path to the ensemble weights directory
    ensemble_method : str
        "product" uses PoE weights; "weighted" uses simple linear-average weights.
        
    Returns:
    --------
    ensemble_weights : dict
        Dictionary containing ensemble weights
    """
    ensemble_method = resolve_ensemble_method(ensemble_method)
    weights_filename = (
        "global_simple_ensemble_weights_used.csv"
        if ensemble_method == "weighted"
        else "global_ensemble_weights_used.csv"
    )
    ensemble_weights = {}
    
    global_weights_path = os.path.join(weights_dir, FINAL_FOLD_TYPE, weights_filename)
    if not os.path.exists(global_weights_path):
        raise FileNotFoundError(
            f"Global ensemble weights not found at {global_weights_path}. "
            "Run R/train_test_analysis.R on final selection outputs first."
        )
    global_weights = pd.read_csv(global_weights_path)
    ensemble_weights["global"] = global_weights
    print(f"Loaded global ensemble weights ({FINAL_FOLD_TYPE}, {ensemble_method})")
    return ensemble_weights


def load_cutoffs(
    cutoffs_path,
    required=False,
    cutoff_source=CUTOFF_SOURCE_SELECTION,
    threshold_method=THRESHOLD_METHOD_JACKKNIFE,
    target_risk_pct=None,
):
    """
    Load prediction cutoffs for a calibration track (selection_loso or deploy_loso).

    The deploy_cutoffs CSV holds one pooled-OOF scalar per (source, threshold_method,
    requested_target_risk). When target_risk_pct is given, restrict to the row whose
    requested_target_risk matches it.
    """
    if not os.path.exists(cutoffs_path):
        msg = f"Cutoffs file not found at {cutoffs_path}"
        if required:
            raise FileNotFoundError(msg)
        print(f"WARNING: {msg}")
        return {}

    cutoffs_df = pd.read_csv(cutoffs_path)
    mask = (cutoffs_df["source"] == cutoff_source) & (
        cutoffs_df["threshold_method"] == threshold_method
    )
    if target_risk_pct is not None and "requested_target_risk" in cutoffs_df.columns:
        mask &= (
            cutoffs_df["requested_target_risk"].astype(float)
            - target_risk_pct / 100.0
        ).abs() < 1e-9
    deploy_cutoffs = cutoffs_df[mask].copy()

    cutoffs = {}
    for _, row in deploy_cutoffs.iterrows():
        cutoffs[row["model"]] = row["prob_cutoff"]

    if required and len(cutoffs) == 0:
        raise ValueError(
            f"No {cutoff_source} cutoffs with threshold_method={threshold_method} "
            f"found in required file: {cutoffs_path}"
        )
    print(
        f"Loaded cutoffs for {len(cutoffs)} models "
        f"(source={cutoff_source}, threshold_method={threshold_method})"
    )
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


def prepare_risk_curve_for_selection(
    risk_cov_df: pd.DataFrame, threshold_method: str = None
) -> pd.DataFrame:
    """
    Normalize risk-curve input to the selection schema expected by cutoff
    selection: model, prob_cutoff, mean_risk, mean_coverage (+ optional mean_kappa).
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

    work = risk_cov_df
    if threshold_method is not None and "threshold_method" in work.columns:
        work = work[work["threshold_method"] == threshold_method].copy()
        if work.empty:
            raise ValueError(
                f"No risk-coverage rows for threshold_method={threshold_method}"
            )

    # If the deployable aggregated schema is already present, use it directly.
    if {"mean_risk", "mean_coverage"}.issubset(work.columns):
        out_cols = ["model", "prob_cutoff", "mean_risk", "mean_coverage"]
        if "mean_kappa" in work.columns:
            out_cols.append("mean_kappa")
        return work[out_cols].copy()

    # Backward-compatible fallback for raw heldout rows.
    # Hybrid selection metric:
    #   mean_risk     := 1 - accuracy            (all accepted samples)
    #   mean_coverage := coverage_known          (seen/known samples only)
    if {"accuracy", "coverage_known"}.issubset(work.columns):
        work = work.dropna(subset=["accuracy", "coverage_known"]).copy()
        if work.empty:
            raise ValueError(
                "Outer-CV risk-coverage file has no finite hybrid metrics "
                "(accuracy / coverage_known)."
            )
        work["risk_for_selection"] = 1.0 - work["accuracy"].astype(float)
        work["coverage_for_selection"] = work["coverage_known"].astype(float)
    elif {"accuracy", "perc_rejected"}.issubset(work.columns):
        work = work.dropna(subset=["accuracy", "perc_rejected"]).copy()
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


MAXPROB_REJECTOR_KEYS = (
    "svm_single_head",
)
RIDGE_REJECTOR_KEYS = (
    "svm_ridge_in_model",
)
ALL_REJECTOR_MODES = MAXPROB_REJECTOR_KEYS + RIDGE_REJECTOR_KEYS

ELASTICNET_KNN_COLUMNS = ("knn10_mean_d", "knn10_min_d", "knn10_q90_d")
PROB_MATRIX_META_COLUMNS = frozenset({"sample_name", "sample_index"})


def is_maxprob_rejector_key(rejector_key: str) -> bool:
    return rejector_key in MAXPROB_REJECTOR_KEYS


def is_ridge_rejector_key(rejector_key: str) -> bool:
    return rejector_key in RIDGE_REJECTOR_KEYS


def rejector_needs_knn10(rejector_key: str) -> bool:
    return "knn10" in rejector_key


def is_maxprob_single_rejector(rejector_key: str) -> bool:
    return rejector_key == "svm_single_head"


def is_two_head_rejector_key(rejector_key: str) -> bool:
    return "two_head" in rejector_key


def load_glm_params(params_path, head=None):
    """
    Load pooled GLM / glmnet rejector coefficients exported by calibration_reject_models_final.R.

    Returns (params, feature_scales). feature_scales is None for max-prob GLM exports;
    elastic-net exports include mean_x/sd_x for glmnet standardization.
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"GLM parameters file not found at {params_path}. "
            "Run R/calibration_reject_models_final.R after train_test_analysis.R."
        )

    df = pd.read_csv(params_path)
    if df.empty:
        raise ValueError(f"GLM parameters file is empty: {params_path}")

    df = df[df["model"] == DEPLOY_BASE_MODEL].copy()
    if df.empty:
        raise ValueError(f"No {DEPLOY_BASE_MODEL} rows in GLM params: {params_path}")

    if head is not None:
        if "head" not in df.columns:
            raise ValueError(
                f"Requested head='{head}' but GLM params have no 'head' column: {params_path}"
            )
        df = df[df["head"] == head].copy()
        if df.empty:
            raise ValueError(f"No rows for head='{head}' in GLM params: {params_path}")

    params = {str(row["term"]): float(row["estimate"]) for _, row in df.iterrows()}
    if "(Intercept)" not in params:
        raise ValueError(f"(Intercept) missing in GLM params: {params_path}")

    feature_scales = None
    if "mean_x" in df.columns and "sd_x" in df.columns:
        feature_scales = {}
        for _, row in df.iterrows():
            term = str(row["term"])
            if term == "(Intercept)":
                continue
            if pd.isna(row["mean_x"]) or pd.isna(row["sd_x"]):
                raise ValueError(
                    f"Elastic-net params missing mean_x/sd_x for term '{term}' in {params_path}. "
                    "Re-run R/calibration_reject_models_final.R."
                )
            feature_scales[term] = (float(row["mean_x"]), float(row["sd_x"]))

    label = f"head={head}, " if head is not None else ""
    scale_note = f", glmnet scales for {len(feature_scales)} terms" if feature_scales else ""
    print(f"Loaded GLM parameters ({label}{len(params)} terms{scale_note}) from {params_path}")
    return params, feature_scales


def resolve_rejector_mode(rejector_mode: str) -> str:
    """Normalize deployment rejector selection."""
    aliases = {
        "svm_single_head": "svm_single_head",
        "svm_ridge_in_model": "svm_ridge_in_model",
        "all": "all",
    }
    if rejector_mode not in aliases:
        raise ValueError(
            "Unsupported rejector mode. Use one of: "
            f"{', '.join(sorted(set(aliases.keys())))}."
        )
    return aliases[rejector_mode]


def resolve_rejector_modes(rejector_mode: str) -> list[str]:
    """Expand grouped rejector mode aliases."""
    mode = resolve_rejector_mode(rejector_mode)
    if mode == "all":
        return list(ALL_REJECTOR_MODES)
    return [mode]


def final_merge_suffix(label_set_key: str) -> str:
    """Suffix used in final deployment artifact filenames."""
    return f"_{label_set_key}"


def resolve_final_cutoffs_dir(cutoffs_root, label_set_key: str) -> str:
    return os.path.join(str(cutoffs_root), f"cutoffs_{label_set_key}")


def resolve_final_glm_params_path(
    cutoffs_root, label_set_key: str, rejector_key: str, cutoff_source: str = CUTOFF_SOURCE_SELECTION
) -> str:
    """Path to rejector params exported by calibration rejector scripts."""
    if rejector_key not in ALL_REJECTOR_MODES:
        raise ValueError(
            f"resolve_final_glm_params_path requires a concrete rejector_key, got '{rejector_key}'."
        )
    merge_suffix = calibration_artifact_suffix(label_set_key, cutoff_source)
    params_dir = os.path.join(str(cutoffs_root), f"multivariate_params_{label_set_key}")
    params_key = enet_params_file_key(rejector_key)
    return os.path.join(params_dir, f"multivariate_params_{params_key}{merge_suffix}.csv")


def load_final_deployment_cutoffs(
    cutoffs_root,
    label_set_key: str,
    rejector_key: str,
    max_accepted_risk_pct=None,
    cutoff_source: str = CUTOFF_SOURCE_SELECTION,
    threshold_method: str = THRESHOLD_METHOD_JACKKNIFE,
) -> dict:
    """
    Load deployment cutoffs from final-model exports.

    The deployed cutoff is the pooled-OOF scalar in deploy_cutoffs_{rejector}_{suffix}.csv
    (one row per source/threshold_method/target risk). This is the value reported in the
    deploy tables and the leave-one-study-out analogue validated by the publication LOSO
    calibration curve. We use it whenever the requested operating point matches a target
    risk present in that file. Only risk levels the deploy-cutoff table does not cover fall
    back to the per-fold risk-coverage curve.
    """
    if rejector_key not in ALL_REJECTOR_MODES:
        raise ValueError(
            f"load_final_deployment_cutoffs requires a concrete rejector_key, got '{rejector_key}'."
        )
    cutoffs_dir = resolve_final_cutoffs_dir(cutoffs_root, label_set_key)
    merge_suffix = calibration_artifact_suffix(label_set_key, cutoff_source)
    deploy_cutoffs_path = os.path.join(
        cutoffs_dir, f"deploy_cutoffs_{rejector_key}{merge_suffix}.csv"
    )
    if max_accepted_risk_pct is not None:
        pooled = load_cutoffs(
            deploy_cutoffs_path,
            required=False,
            cutoff_source=cutoff_source,
            threshold_method=threshold_method,
            target_risk_pct=max_accepted_risk_pct,
        )
        if pooled:
            return pooled

        # Operating point not in the deploy-cutoff table: select from the risk curve.
        risk_curve_path = os.path.join(
            cutoffs_dir, f"deploy_risk_coverage_curve_{rejector_key}{merge_suffix}.csv"
        )
        risk_df = load_risk_coverage(risk_curve_path, required=True)
        if "threshold_method" not in risk_df.columns:
            raise ValueError(
                f"Risk-coverage curve {risk_curve_path} lacks a threshold_method column; "
                f"cannot safely select a {threshold_method} cutoff."
            )
        print(f"Loading deploy risk-coverage curve from: {risk_curve_path}")
        summary = prepare_risk_curve_for_selection(
            risk_df, threshold_method=threshold_method
        )
        cutoffs = choose_cutoffs_from_risk(summary, max_accepted_risk_pct / 100.0)
        if not cutoffs:
            raise ValueError(
                f"Could not derive cutoffs at max_accepted_risk_pct={max_accepted_risk_pct} "
                f"from {risk_curve_path}"
            )
        return cutoffs

    print(f"Loading deployment cutoffs from: {deploy_cutoffs_path}")
    return load_cutoffs(
        deploy_cutoffs_path,
        required=True,
        cutoff_source=cutoff_source,
        threshold_method=threshold_method,
    )


def load_final_rejector_params(params_path: str, rejector_key: str):
    """Load rejector coefficients (max-prob GLM or ridge glmnet export)."""
    if is_maxprob_single_rejector(rejector_key):
        coef, scales = load_glm_params(params_path, head="accept_combined")
        return coef, None, scales, None
    if is_maxprob_rejector_key(rejector_key) and is_two_head_rejector_key(rejector_key):
        correct_coef, correct_scales = load_glm_params(params_path, head="correct_given_id")
        id_coef, id_scales = load_glm_params(params_path, head="id")
        return correct_coef, id_coef, correct_scales, id_scales
    if is_ridge_rejector_key(rejector_key) and not is_two_head_rejector_key(rejector_key):
        coef, scales = load_glm_params(params_path, head="accept_combined")
        return coef, None, scales, None
    if is_ridge_rejector_key(rejector_key) and is_two_head_rejector_key(rejector_key):
        correct_coef, correct_scales = load_glm_params(params_path, head="correct_given_id")
        id_coef, id_scales = load_glm_params(params_path, head="id")
        return correct_coef, id_coef, correct_scales, id_scales
    raise ValueError(f"load_final_rejector_params does not support rejector_key='{rejector_key}'.")


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


def merge_probability_classes(prob_matrix_df, merge_method="sum"):
    """
    Merge subtype-family classes in the probability matrix into single columns:
    1. classes with 'MDS' or 'TP53' in their name -> "MDS.r"
    2. other KMT2A classes (excluding MLLT3 fusion) -> "other.KMT2A"
    3. MECOM-related classes (GATA2;MECOM, MECOM other) -> "MECOM"

    merge_method controls how member probabilities are combined:
    - "sum": marginal probability of the family (matches R merge_prob_method="sum").
    - "max": max member probability (matches R merge_prob_method="max").
    After merging, row probabilities are renormalized to sum to 1.

    Parameters:
    -----------
    prob_matrix_df : pd.DataFrame
        Probability matrix DataFrame with 'sample_name' column and class probability columns
    merge_method : str
        "sum" or "max".

    Returns:
    --------
    prob_matrix_df : pd.DataFrame
        Modified probability matrix with merged classes
    """
    if merge_method not in ("sum", "max"):
        raise ValueError(f"merge_method must be 'sum' or 'max', got '{merge_method}'")
    combine = (lambda cols: prob_matrix_df[cols].max(axis=1)) if merge_method == "max" \
        else (lambda cols: prob_matrix_df[cols].sum(axis=1))
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
        prob_matrix_df['MDS.r'] = combine(mds_classes)
        prob_matrix_df = prob_matrix_df.drop(columns=mds_classes)

    if other_kmt2a_classes:
        prob_matrix_df['other.KMT2A'] = combine(other_kmt2a_classes)
        prob_matrix_df = prob_matrix_df.drop(columns=other_kmt2a_classes)

    if mecom_classes:
        prob_matrix_df['MECOM'] = combine(mecom_classes)
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

    cutoff_mapping = {DEPLOYMENT_PREDICTOR: DEPLOY_BASE_MODEL}

    for model_name, df in predictions_dict.items():
        if model_name != DEPLOYMENT_PREDICTOR:
            continue
        cutoff_key = cutoff_mapping.get(model_name, model_name)

        score_col = "prediction_prob_calibrated"
        if score_col not in df.columns:
            raise ValueError(
                f"{DEPLOYMENT_PREDICTOR} requires calibrated confidence "
                f"({score_col}), but it is missing."
            )

        if cutoff_key not in cutoffs:
            raise ValueError(
                f"No deployment cutoff found for {model_name} (expected key '{cutoff_key}')."
            )
        cutoff_value = cutoffs[cutoff_key]
        df["prediction_passed_cutoff"] = df[score_col] >= cutoff_value
        print(f"Applied cutoff {cutoff_value:.3f} to {model_name} using {score_col}")
    
    return predictions_dict


def _score_logistic_head(feature_map, params, feature_scales=None):
    """Score a logistic GLM or glmnet head from saved R coefficients."""
    return score_logistic_head(feature_map, params, feature_scales=feature_scales)


def apply_maxprob_single_head_confidence(global_pred_df, accept_params):
    """
    Max-prob single-head: P(accept_combined | max_prob).
    Matches R maxprob_single_head on accept_combined target.
    """
    max_prob = global_pred_df["prediction_prob"].to_numpy(dtype=np.float64)
    feature_map = {"max_prob": max_prob}
    p_accept = _score_logistic_head(feature_map, accept_params)
    global_pred_df = global_pred_df.copy()
    global_pred_df["prediction_prob_calibrated"] = p_accept
    return global_pred_df


def apply_maxprob_two_head_product_confidence(
    global_pred_df,
    correctness_params,
    ood_head_params,
):
    """
    Max-prob two-head product: P(correct|max_prob) * P(ID|max_prob).
    Matches R maxprob_two_head_product + two_head_combine=product.
    """
    max_prob = global_pred_df["prediction_prob"].to_numpy(dtype=np.float64)
    feature_map = {"max_prob": max_prob}
    p_correct = _score_logistic_head(feature_map, correctness_params)
    p_id = _score_logistic_head(feature_map, ood_head_params)
    global_pred_df = global_pred_df.copy()
    global_pred_df["prediction_prob_calibrated"] = p_correct * p_id
    global_pred_df["prediction_prob_correct_head"] = p_correct
    global_pred_df["prediction_prob_id_head"] = p_id
    return global_pred_df


def apply_maxprob_two_head_min_confidence(
    global_pred_df,
    correctness_params,
    ood_head_params,
):
    """Max-prob two-head min: min(P(correct|max_prob), P(ID|max_prob))."""
    max_prob = global_pred_df["prediction_prob"].to_numpy(dtype=np.float64)
    feature_map = {"max_prob": max_prob}
    p_correct = _score_logistic_head(feature_map, correctness_params)
    p_id = _score_logistic_head(feature_map, ood_head_params)
    global_pred_df = global_pred_df.copy()
    global_pred_df["prediction_prob_calibrated"] = np.minimum(p_correct, p_id)
    global_pred_df["prediction_prob_correct_head"] = p_correct
    global_pred_df["prediction_prob_id_head"] = p_id
    return global_pred_df


def _class_probability_matrix(prob_df):
    """Extract class probability matrix from a prediction probability DataFrame."""
    prob_cols = [c for c in prob_df.columns if c not in PROB_MATRIX_META_COLUMNS]
    if not prob_cols:
        raise ValueError("Probability matrix has no class columns.")
    return prob_df[prob_cols].to_numpy(dtype=np.float64)


def build_rejection_feature_map(global_pred_df, global_prob_df, base_prob_matrices):
    """
    Rejection features for elastic-net heads (matches R get_rejection_features_from_matrix).
    """
    prob_mat = _class_probability_matrix(global_prob_df)
    n_rows, n_classes = prob_mat.shape
    pred_indices = np.argmax(prob_mat, axis=1)
    row_idx = np.arange(n_rows)
    max_prob = prob_mat[row_idx, pred_indices]

    prob_mod = prob_mat.copy()
    prob_mod[row_idx, pred_indices] = -np.inf
    second_prob = np.max(prob_mod, axis=1)
    margin = max_prob - second_prob

    prob_clipped = np.clip(prob_mat, 1e-12, None)
    if n_classes > 1:
        entropy = -np.sum(prob_clipped * np.log(prob_clipped), axis=1) / np.log(n_classes)
        entropy = np.clip(entropy, 0.0, 1.0)
    else:
        entropy = np.zeros(n_rows, dtype=np.float64)

    top1_probs = []
    for model_name in ("SVM", "XGBOOST", "NN"):
        prob_df = base_prob_matrices.get(model_name)
        if prob_df is None:
            continue
        top1_probs.append(np.max(_class_probability_matrix(prob_df), axis=1))
    if len(top1_probs) >= 2:
        top1_var = np.var(np.column_stack(top1_probs), axis=1)
        top1_var = np.maximum(0.0, top1_var)
        top1_var[~np.isfinite(top1_var)] = 0.0
    else:
        top1_var = np.zeros(n_rows, dtype=np.float64)

    p_sorted = np.sort(prob_mat, axis=1)[:, ::-1]
    cumsum = np.cumsum(p_sorted, axis=1)
    conformal_threshold = 0.9
    conformal_size = np.empty(n_rows, dtype=np.float64)
    for i in range(n_rows):
        hits = np.where(cumsum[i] >= conformal_threshold)[0]
        conformal_size[i] = float(hits[0] + 1) if len(hits) else float(n_classes)

    return {
        "max_prob": max_prob,
        "margin": margin,
        "entropy": entropy,
        "top1_prob_variance_across_models": top1_var,
        "conformal_set_size_90": conformal_size,
    }


def load_cohort_knn_reference(data_path=None, fs_method="eta2"):
    """Training-cohort reference for KNN rejection features on new samples."""
    base_path = Path(__file__).resolve().parent.parent
    if data_path is None:
        data_path = base_path / "data"
    X_all, y_all, study_all = train_test.load_data(str(data_path))
    valid_mask = train_test.filtered_cohort_mask(y_all, study_all)
    X_cohort = X_all[valid_mask]
    y_cohort = y_all[valid_mask]
    study_cohort = study_all[valid_mask]
    y_encoded, _ = train_test.encode_labels(y_cohort)
    pipe = train_test.build_knn_rejection_pipe(fs_method=fs_method)
    return {
        "X_cohort": X_cohort,
        "y_cohort": y_encoded,
        "study_cohort": study_cohort,
        "pipe": pipe,
        "fs_method": fs_method,
    }


def compute_knn_rejection_features(X_new, cohort_knn_reference):
    """KNN distance summaries for elastic-net rejectors."""
    knn_feats = train_test.compute_knn_features_full_reference(
        cohort_knn_reference["X_cohort"],
        cohort_knn_reference["y_cohort"],
        cohort_knn_reference["study_cohort"],
        X_new,
        cohort_knn_reference["pipe"],
        fs_method=cohort_knn_reference["fs_method"],
    )
    return {
        col: np.asarray(knn_feats[col], dtype=np.float64)
        for col in ELASTICNET_KNN_COLUMNS
    }


def apply_enet_single_head_confidence(
    global_pred_df, global_prob_df, base_prob_matrices, accept_params, knn_feature_map,
    accept_scales=None,
):
    """Elastic-net single-head: P(accept_combined | full rejection feature set)."""
    feature_map = build_rejection_feature_map(
        global_pred_df, global_prob_df, base_prob_matrices
    )
    feature_map.update(knn_feature_map)
    p_accept = _score_logistic_head(feature_map, accept_params, feature_scales=accept_scales)
    global_pred_df = global_pred_df.copy()
    global_pred_df["prediction_prob_calibrated"] = p_accept
    return global_pred_df


def apply_enet_two_head_confidence(
    global_pred_df,
    global_prob_df,
    base_prob_matrices,
    correctness_params,
    ood_head_params,
    knn_feature_map,
    combine="product",
    correctness_scales=None,
    ood_scales=None,
):
    """Elastic-net two-head on the full rejection feature set (product or min combine)."""
    feature_map = build_rejection_feature_map(
        global_pred_df, global_prob_df, base_prob_matrices
    )
    feature_map.update(knn_feature_map)
    p_correct = _score_logistic_head(
        feature_map, correctness_params, feature_scales=correctness_scales
    )
    p_id = _score_logistic_head(
        feature_map, ood_head_params, feature_scales=ood_scales
    )
    if combine == "product":
        p_combined = p_correct * p_id
    elif combine == "min":
        p_combined = np.minimum(p_correct, p_id)
    else:
        raise ValueError(f"Unsupported two-head combine for elastic-net: {combine}")
    global_pred_df = global_pred_df.copy()
    global_pred_df["prediction_prob_calibrated"] = p_combined
    global_pred_df["prediction_prob_correct_head"] = p_correct
    global_pred_df["prediction_prob_id_head"] = p_id
    return global_pred_df


def apply_enet_two_head_product_confidence(
    global_pred_df,
    global_prob_df,
    base_prob_matrices,
    correctness_params,
    ood_head_params,
    knn_feature_map,
    correctness_scales=None,
    ood_scales=None,
):
    """Elastic-net two-head product on the full rejection feature set."""
    return apply_enet_two_head_confidence(
        global_pred_df,
        global_prob_df,
        base_prob_matrices,
        correctness_params,
        ood_head_params,
        knn_feature_map,
        combine="product",
        correctness_scales=correctness_scales,
        ood_scales=ood_scales,
    )


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
    cutoffs,
    rejector_params,
    rejector_key,
    id_head_params=None,
    merge_classes=False,
    merge_method="sum",
    cohort_knn_reference=None,
    rejector_scales=None,
    id_head_scales=None,
):
    """Run SVM deployment predictions with one rejector recipe."""
    print(f"\n{'='*60}")
    print(f"Running SVM predictions ({'MERGED' if merge_classes else 'UNMERGED'} classes)")
    print(f"Rejector: {rejector_key}")
    print(f"{'='*60}")

    if "SVM" not in models:
        raise ValueError("SVM model is required for deployment.")

    predictions = {}
    prob_matrices = {}
    predictions[DEPLOYMENT_PREDICTOR], prob_matrices[DEPLOYMENT_PREDICTOR] = predict_ovr_models(
        X, models, "SVM", sample_names
    )

    if merge_classes:
        prob_matrices[DEPLOYMENT_PREDICTOR] = merge_probability_classes(
            prob_matrices[DEPLOYMENT_PREDICTOR].copy(), merge_method=merge_method
        )

    if not rejector_params:
        raise ValueError("Rejector parameters are required.")

    deploy_pred = predictions[DEPLOYMENT_PREDICTOR]
    deploy_prob = prob_matrices[DEPLOYMENT_PREDICTOR]
    svm_only_probs = {DEPLOYMENT_PREDICTOR: deploy_prob}

    if is_maxprob_single_rejector(rejector_key):
        deploy_pred = apply_maxprob_single_head_confidence(deploy_pred, rejector_params)
    elif rejector_key == "svm_two_head_min":
        if not id_head_params:
            raise ValueError("ID-head parameters are required for svm_two_head_min.")
        deploy_pred = apply_maxprob_two_head_min_confidence(
            deploy_pred, rejector_params, id_head_params
        )
    elif is_ridge_rejector_key(rejector_key) and not is_two_head_rejector_key(rejector_key):
        knn_map = {}
        if rejector_needs_knn10(rejector_key):
            if cohort_knn_reference is None:
                raise ValueError(f"cohort_knn_reference is required for {rejector_key}.")
            knn_map = compute_knn_rejection_features(X, cohort_knn_reference)
        deploy_pred = apply_enet_single_head_confidence(
            deploy_pred, deploy_prob, svm_only_probs, rejector_params, knn_map,
            accept_scales=rejector_scales,
        )
    elif is_ridge_rejector_key(rejector_key) and rejector_key.endswith("_two_head_min"):
        if not id_head_params:
            raise ValueError(f"ID-head parameters are required for {rejector_key}.")
        knn_map = {}
        if rejector_needs_knn10(rejector_key):
            if cohort_knn_reference is None:
                raise ValueError(f"cohort_knn_reference is required for {rejector_key}.")
            knn_map = compute_knn_rejection_features(X, cohort_knn_reference)
        deploy_pred = apply_enet_two_head_confidence(
            deploy_pred, deploy_prob, svm_only_probs, rejector_params, id_head_params,
            knn_map, combine="min",
            correctness_scales=rejector_scales, ood_scales=id_head_scales,
        )
    else:
        raise ValueError(f"Unsupported rejector_key in run_predictions: {rejector_key}")

    predictions[DEPLOYMENT_PREDICTOR] = deploy_pred
    predictions = apply_cutoffs(predictions, cutoffs)

    return (
        {DEPLOYMENT_PREDICTOR: predictions[DEPLOYMENT_PREDICTOR]},
        {DEPLOYMENT_PREDICTOR: deploy_prob},
    )


def run_predictions_for_label_set(
    X,
    sample_names,
    models,
    label_set_key,
    label_merge_suffix,
    cutoffs_root,
    rejector_keys,
    merge_classes,
    max_accepted_risk_pct,
    output_dir,
    input_filename,
    cohort_knn_reference=None,
    cutoff_sources=None,
    threshold_methods=None,
    merge_method="sum",
):
    """Run one label set (merged/unmerged) for each rejector, cutoff source, and threshold method."""
    if cutoff_sources is None:
        cutoff_sources = [CUTOFF_SOURCE_SELECTION]
    if threshold_methods is None:
        threshold_methods = [THRESHOLD_METHOD_JACKKNIFE]

    for cutoff_source in cutoff_sources:
        print(f"\n=== Cutoff source: {cutoff_source} ===")
        source_tag = prediction_output_tag(cutoff_source)
        for threshold_method in threshold_methods:
            threshold_tag = prediction_threshold_tag(threshold_method)
            print(f"\n--- Threshold method: {threshold_method} ---")
            for rejector_key in rejector_keys:
                print(f"\n--- Rejector pipeline: {rejector_key}{source_tag}{threshold_tag} ---")
                params_path = resolve_final_glm_params_path(
                    cutoffs_root,
                    label_set_key,
                    rejector_key=rejector_key,
                    cutoff_source=cutoff_source,
                )
                print(f"Loading rejector parameters from: {params_path}")
                rejector_params, id_head_params, rejector_scales, id_head_scales = load_final_rejector_params(
                    params_path, rejector_key=rejector_key
                )
                if is_ridge_rejector_key(rejector_key) and rejector_scales is None:
                    raise ValueError(
                        f"Ridge params at {params_path} lack mean_x/sd_x columns. "
                        "Re-run calibration rejector export."
                    )
                if (
                    is_ridge_rejector_key(rejector_key)
                    and is_two_head_rejector_key(rejector_key)
                    and id_head_scales is None
                ):
                    raise ValueError(
                        f"Ridge ID-head params at {params_path} lack mean_x/sd_x columns."
                    )

                cutoffs = load_final_deployment_cutoffs(
                    cutoffs_root,
                    label_set_key,
                    rejector_key=rejector_key,
                    max_accepted_risk_pct=max_accepted_risk_pct,
                    cutoff_source=cutoff_source,
                    threshold_method=threshold_method,
                )

                knn_ref = cohort_knn_reference if rejector_needs_knn10(rejector_key) else None
                predictions, prob_matrices = run_predictions(
                    X,
                    sample_names,
                    models,
                    cutoffs,
                    rejector_params,
                    rejector_key,
                    id_head_params,
                    merge_classes=merge_classes,
                    merge_method=merge_method,
                    cohort_knn_reference=knn_ref,
                    rejector_scales=rejector_scales,
                    id_head_scales=id_head_scales,
                )

                risk_tag = prediction_risk_tag(max_accepted_risk_pct)
                save_predictions(
                    predictions,
                    prob_matrices,
                    output_dir,
                    input_filename,
                    merge_suffix=f"{label_merge_suffix}_{rejector_key}{source_tag}{threshold_tag}{risk_tag}",
                )


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
        help="Maximum accepted error rate (percentage) on accepted predictions. "
             "If unset, uses deploy_cutoffs_{suffix}.csv (default 5%% target). "
             "If set, selects cutoff from deploy_risk_coverage_curve_{suffix}.csv."
    )
    parser.add_argument(
        "--cutoff_source",
        default="selection_loso",
        choices=["selection_loso", "deploy_loso", "both"],
        help="Which calibration track to use for rejector cutoffs/coefs. "
             "'both' runs selection-loso and deploy-loso (Option B) for side-by-side comparison.",
    )
    parser.add_argument(
        "--rejector_mode",
        default="all",
        choices=[
            "svm_single_head",
            "svm_ridge_in_model",
            "all",
        ],
        help="SVM rejector recipe(s): single-head max-prob or single-head ridge "
             "(in-model confidence features). all = both recipes.",
    )
    parser.add_argument(
        "--threshold_method",
        default="jackknife_adjusted",
        choices=["jackknife_adjusted", "pooled_oof", "ucb_95", "both"],
        help="Cutoff derivation on pooled OOF scores. "
             "'jackknife_adjusted' (default) applies jackknife gap correction; "
             "'pooled_oof' uses raw pooled-OOF threshold; "
             "'ucb_95' uses one-sided 95% Wilson upper bound control; "
             "'both' writes separate prediction files for each.",
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
    rejector_keys = resolve_rejector_modes(args.rejector_mode)
    cutoff_sources = resolve_cutoff_sources(args.cutoff_source)
    threshold_methods = resolve_threshold_methods(args.threshold_method)
    print(f"Deployment classifier: {DEPLOYMENT_PREDICTOR}")
    print(f"Rejector key(s): {', '.join(rejector_keys)}")
    print(f"Cutoff source(s): {', '.join(cutoff_sources)}")
    print(f"Threshold method(s): {', '.join(threshold_methods)}")
    if args.max_accepted_risk_pct is not None:
        print(f"Maximum accepted risk (on accepted predictions): {args.max_accepted_risk_pct:.2f}%")
    else:
        print("Using default deployment cutoffs (5% target risk)")

    needs_knn = any(rejector_needs_knn10(k) for k in rejector_keys)
    cohort_knn_reference = load_cohort_knn_reference() if needs_knn else None
    if needs_knn:
        print("Loaded training-cohort KNN reference for elastic-net rejectors.")
    
    # Determine which versions to run
    run_merged = not args.unmerged_only
    run_unmerged = not args.merged_only

    if args.merged_only and args.unmerged_only:
        print("ERROR: Cannot specify both --merged_only and --unmerged_only")
        return

    X, sample_names = load_new_samples(args.input_file)
    models = load_models_and_metadata(args.models_dir, pipelines_dir)

    if run_unmerged:
        print("\n" + "="*60)
        print("UNMERGED VERSION")
        print("="*60)
        run_predictions_for_label_set(
            X,
            sample_names,
            models,
            label_set_key="unmerged_maxprob",
            label_merge_suffix="_unmerged",
            cutoffs_root=args.cutoffs_file,
            rejector_keys=rejector_keys,
            merge_classes=False,
            max_accepted_risk_pct=args.max_accepted_risk_pct,
            output_dir=output_dir,
            input_filename=input_filename,
            cohort_knn_reference=cohort_knn_reference,
            cutoff_sources=cutoff_sources,
            threshold_methods=threshold_methods,
        )

    if run_merged:
        print("\n" + "="*60)
        print("MERGED VERSION")
        print("="*60)
        run_predictions_for_label_set(
            X,
            sample_names,
            models,
            label_set_key="merged_summed",
            label_merge_suffix="_merged_summed",
            cutoffs_root=args.cutoffs_file,
            rejector_keys=rejector_keys,
            merge_classes=True,
            max_accepted_risk_pct=args.max_accepted_risk_pct,
            output_dir=output_dir,
            input_filename=input_filename,
            cohort_knn_reference=cohort_knn_reference,
            cutoff_sources=cutoff_sources,
            threshold_methods=threshold_methods,
        )

    if run_merged:
        print("\n" + "="*60)
        print("MERGED (MAX-PROB) VERSION")
        print("="*60)
        run_predictions_for_label_set(
            X,
            sample_names,
            models,
            label_set_key="merged_maxprob",
            label_merge_suffix="_merged_maxprob",
            cutoffs_root=args.cutoffs_file,
            rejector_keys=rejector_keys,
            merge_classes=True,
            max_accepted_risk_pct=args.max_accepted_risk_pct,
            output_dir=output_dir,
            input_filename=input_filename,
            cohort_knn_reference=cohort_knn_reference,
            cutoff_sources=cutoff_sources,
            threshold_methods=threshold_methods,
            merge_method="max",
        )

    print("\n" + "="*60)
    print("Prediction pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
