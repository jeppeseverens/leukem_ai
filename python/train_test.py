import pandas as pd
import numpy as np
import os
import json
import gc
import hashlib
import pickle
import re

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors

from joblib import Parallel, delayed
import itertools
import ast
from collections import Counter

# Project-wide seed for CV shuffling (StratifiedKFold, left-out assignment).
CV_RANDOM_STATE = 1

# Default KNN settings for rejection feature generation.
KNN_REJECTION_K_VALUES = (10, 20)
KNN_REJECTION_FEATURE_N_GENES = 500
KNN_DISTANCE_COLUMNS = (
    "knn10_mean_d", "knn10_min_d", "knn10_q90_d",
    "knn20_mean_d", "knn20_min_d", "knn20_q90_d",
)


def _stable_hash_array(arr):
    """Build a stable hash for numpy-compatible content."""
    arr_np = np.asarray(arr)
    return hashlib.sha256(arr_np.tobytes()).hexdigest()


def _get_reject_cache_dir():
    """Directory for fold-level reject feature/preprocess caches."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(project_root, "data", "out", "reject_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _build_preprocess_cache_key(split_tag, n_genes, train_indices, test_indices, y_train, study_train, fs_method):
    payload = {
        "split_tag": str(split_tag),
        "n_genes": int(n_genes),
        "train_idx_hash": _stable_hash_array(train_indices),
        "test_idx_hash": _stable_hash_array(test_indices),
        "y_train_hash": _stable_hash_array(y_train),
        "study_train_hash": _stable_hash_array(study_train),
        "fs_method": str(fs_method),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_preprocess_cache(cache_dir, cache_key):
    cache_path = os.path.join(cache_dir, f"preprocess_{cache_key}.pkl")
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _save_preprocess_cache(cache_dir, cache_key, payload):
    cache_path = os.path.join(cache_dir, f"preprocess_{cache_key}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)


def _compute_knn_distance_features(X_train_ref, X_eval_ref, k_values=KNN_REJECTION_K_VALUES):
    """
    Compute KNN distance summaries for each k in k_values.
    Returns dict of feature_name -> np.ndarray.
    """
    out = {}
    if X_train_ref is None or X_eval_ref is None:
        return out

    X_train_ref = np.asarray(X_train_ref, dtype=np.float32)
    X_eval_ref = np.asarray(X_eval_ref, dtype=np.float32)
    if X_train_ref.size == 0 or X_eval_ref.size == 0:
        return out

    for k in k_values:
        k_eff = int(min(max(1, k), X_train_ref.shape[0]))
        nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
        nn.fit(X_train_ref)
        distances, _ = nn.kneighbors(X_eval_ref, return_distance=True)
        out[f"knn{k}_mean_d"] = distances.mean(axis=1).astype(np.float32)
        out[f"knn{k}_min_d"] = distances.min(axis=1).astype(np.float32)
        out[f"knn{k}_q90_d"] = np.quantile(distances, 0.9, axis=1).astype(np.float32)
    return out


def _load_or_compute_knn_features(cache_dir, cache_key, X_train_ref, X_eval_ref):
    cache_path = os.path.join(cache_dir, f"knn_{cache_key}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    features = _compute_knn_distance_features(X_train_ref, X_eval_ref)
    with open(cache_path, "wb") as f:
        pickle.dump(features, f)
    return features


def _parse_index_vector(value):
    """
    Parse index vectors stored as JSON/Python-like strings in CSV cells.
    Supports forms like:
      "[1, 2, 3]"
      "[1 2 3]"
      "1,2,3"
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=np.int64)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.int64)
    txt = str(value).strip()
    if txt == "":
        return np.array([], dtype=np.int64)

    # Try literal parsing first for strict JSON/Python list style.
    try:
        parsed = ast.literal_eval(txt)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return np.asarray(parsed, dtype=np.int64)
    except Exception:
        pass

    # Fallback regex parser for space/comma-delimited numeric vectors.
    nums = re.findall(r"-?\d+", txt)
    if not nums:
        return np.array([], dtype=np.int64)
    return np.asarray([int(n) for n in nums], dtype=np.int64)


def backfill_knn_columns_in_outer_results(
    outer_results_df,
    X,
    y,
    study_labels,
    pipe,
    fold_type="CV",
    fs_method="eta2",
    knn_n_genes=KNN_REJECTION_FEATURE_N_GENES,
    cache_dir=None,
    strict=True,
):
    """
    Compute KNN distance feature vectors for existing outer-CV results and append
    them as columns, without rerunning model training/prediction.
    """
    if outer_results_df is None or len(outer_results_df) == 0:
        return outer_results_df

    result = outer_results_df.copy()
    fold_type_l = str(fold_type).lower()
    local_cache_dir = cache_dir or _get_reject_cache_dir()

    split_features = {}
    if fold_type_l == "cv":
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_RANDOM_STATE)
        combined = build_hybrid_stratify_labels(y, study_labels, 5)
        split_iter = list(outer_cv.split(X, combined))
        for outer_fold, (train_idx, test_idx) in enumerate(split_iter):
            proc, _, _ = pre_process_data(
                [int(knn_n_genes)],
                X,
                y,
                train_idx,
                test_idx,
                study_labels,
                pipe,
                cache_dir=local_cache_dir,
                split_tag=f"outer_cv_{outer_fold}",
                fs_method=fs_method,
            )
            X_train_knn, X_test_knn = proc[int(knn_n_genes)]
            knn_key_payload = json.dumps(
                {
                    "mode": "cv_backfill",
                    "outer_fold": int(outer_fold),
                    "fs_method": str(fs_method),
                    "n_genes": int(knn_n_genes),
                    "train_hash": _stable_hash_array(train_idx),
                    "test_hash": _stable_hash_array(test_idx),
                    "k_values": list(KNN_REJECTION_K_VALUES),
                },
                sort_keys=True,
            ).encode("utf-8")
            knn_key = hashlib.sha256(knn_key_payload).hexdigest()
            split_features[str(outer_fold)] = {
                "sample_indices": np.asarray(test_idx, dtype=np.int64),
                "features": _load_or_compute_knn_features(local_cache_dir, knn_key, X_train_knn, X_test_knn),
            }
    elif fold_type_l == "loso":
        for test_study_name in np.unique(study_labels):
            test_mask = study_labels == test_study_name
            train_mask = ~test_mask
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            study_labels_train = study_labels[train_mask]
            test_idx = np.where(test_mask)[0]

            proc = pre_process_data_loso(
                [int(knn_n_genes)],
                X_train,
                X_test,
                study_labels_train,
                y_train,
                pipe,
                cache_dir=local_cache_dir,
                split_tag=f"outer_loso_{test_study_name}",
                fs_method=fs_method,
            )
            X_train_knn, X_test_knn = proc[int(knn_n_genes)]
            knn_key_payload = json.dumps(
                {
                    "mode": "loso_backfill",
                    "outer_fold": str(test_study_name),
                    "fs_method": str(fs_method),
                    "n_genes": int(knn_n_genes),
                    "train_hash": _stable_hash_array(np.where(train_mask)[0]),
                    "test_hash": _stable_hash_array(test_idx),
                    "k_values": list(KNN_REJECTION_K_VALUES),
                },
                sort_keys=True,
            ).encode("utf-8")
            knn_key = hashlib.sha256(knn_key_payload).hexdigest()
            split_features[str(test_study_name)] = {
                "sample_indices": np.asarray(test_idx, dtype=np.int64),
                "features": _load_or_compute_knn_features(local_cache_dir, knn_key, X_train_knn, X_test_knn),
            }
    else:
        raise ValueError(f"Unsupported fold_type '{fold_type}'. Use 'CV' or 'loso'.")

    for col in KNN_DISTANCE_COLUMNS:
        if col not in result.columns:
            result[col] = None

    for row_idx in result.index:
        fold_key = str(result.at[row_idx, "outer_fold"])
        if fold_key not in split_features:
            if strict:
                raise ValueError(f"Missing fold '{fold_key}' in computed KNN feature bundles.")
            continue
        fold_bundle = split_features[fold_key]
        row_indices = _parse_index_vector(result.at[row_idx, "sample_indices"])
        if len(row_indices) == 0:
            if strict:
                raise ValueError(f"Empty sample_indices vector at row {row_idx}.")
            continue
        fold_indices = fold_bundle["sample_indices"]
        # Align fold features to the exact row sample order.
        index_to_pos = {int(v): i for i, v in enumerate(fold_indices.tolist())}
        pos = [index_to_pos.get(int(v), None) for v in row_indices.tolist()]
        if any(p is None for p in pos):
            if strict:
                missing = [int(v) for v, p in zip(row_indices.tolist(), pos) if p is None]
                raise ValueError(
                    f"Row {row_idx} has sample_indices not found in fold '{fold_key}': {missing[:10]}"
                )
            continue
        pos = np.asarray(pos, dtype=np.int64)
        feats = fold_bundle["features"]
        for col in KNN_DISTANCE_COLUMNS:
            vec = feats.get(col)
            if vec is None:
                if strict:
                    raise ValueError(f"Missing KNN feature '{col}' for fold '{fold_key}'.")
                continue
            result.at[row_idx, col] = json.dumps(np.asarray(vec, dtype=np.float32)[pos].tolist())

    if strict:
        for col in KNN_DISTANCE_COLUMNS:
            missing_mask = result[col].isna() | (result[col].astype(str).str.len() == 0)
            if missing_mask.any():
                n_missing = int(missing_mask.sum())
                raise ValueError(f"KNN backfill incomplete: '{col}' missing on {n_missing} rows.")

    return result


def backfill_knn_columns_in_outer_leftout_results(
    leftout_results_df,
    X,
    y,
    study_labels,
    X_leftout,
    y_leftout,
    study_leftout,
    leftout_global_idx,
    pipe,
    fold_type="CV",
    fs_method="eta2",
    knn_n_genes=KNN_REJECTION_FEATURE_N_GENES,
    cache_dir=None,
    strict=True,
):
    """
    Compute KNN distance feature vectors for existing outer-CV left-out result rows
    and append them as columns, without rerunning model training/prediction.
    """
    if leftout_results_df is None or len(leftout_results_df) == 0:
        return leftout_results_df

    result = leftout_results_df.copy()
    fold_type_l = str(fold_type).lower()
    local_cache_dir = cache_dir or _get_reject_cache_dir()

    split_features = {}
    if fold_type_l == "cv":
        leftout_fold_assignments = assign_leftout_to_cv_folds(
            y_leftout, study_leftout, n_folds=5, random_state=CV_RANDOM_STATE
        )
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_RANDOM_STATE)
        combined = build_hybrid_stratify_labels(y, study_labels, 5)
        split_iter = list(outer_cv.split(X, combined))
        for outer_fold, (train_idx, test_idx) in enumerate(split_iter):
            fold_mask = leftout_fold_assignments == outer_fold
            if not np.any(fold_mask):
                continue
            proc, _, _, fitted = pre_process_data(
                [int(knn_n_genes)],
                X,
                y,
                train_idx,
                test_idx,
                study_labels,
                pipe,
                return_pipelines=True,
                cache_dir=local_cache_dir,
                split_tag=f"outer_cv_{outer_fold}",
                fs_method=fs_method,
            )
            X_train_knn = proc[int(knn_n_genes)][0]
            X_leftout_fold_proc = fitted[int(knn_n_genes)].transform(X_leftout[fold_mask]).astype(np.float32)
            leftout_idx_fold = np.asarray(leftout_global_idx[fold_mask], dtype=np.int64)
            knn_key_payload = json.dumps(
                {
                    "mode": "cv_backfill_leftout",
                    "outer_fold": int(outer_fold),
                    "fs_method": str(fs_method),
                    "n_genes": int(knn_n_genes),
                    "train_hash": _stable_hash_array(train_idx),
                    "leftout_hash": _stable_hash_array(leftout_idx_fold),
                    "k_values": list(KNN_REJECTION_K_VALUES),
                },
                sort_keys=True,
            ).encode("utf-8")
            knn_key = hashlib.sha256(knn_key_payload).hexdigest()
            split_features[str(outer_fold)] = {
                "sample_indices": leftout_idx_fold,
                "features": _load_or_compute_knn_features(local_cache_dir, knn_key, X_train_knn, X_leftout_fold_proc),
            }
    elif fold_type_l == "loso":
        for test_study_name in np.unique(study_labels):
            leftout_study_mask = study_leftout == test_study_name
            if not np.any(leftout_study_mask):
                continue
            test_mask = study_labels == test_study_name
            train_mask = ~test_mask
            X_train = X[train_mask]
            y_train = y[train_mask]
            study_labels_train = study_labels[train_mask]
            proc, fitted = pre_process_data_loso(
                [int(knn_n_genes)],
                X_train,
                X[test_mask],
                study_labels_train,
                y_train,
                pipe,
                return_pipelines=True,
                cache_dir=local_cache_dir,
                split_tag=f"outer_loso_{test_study_name}",
                fs_method=fs_method,
            )
            X_train_knn = proc[int(knn_n_genes)][0]
            X_leftout_fold_proc = fitted[int(knn_n_genes)].transform(X_leftout[leftout_study_mask]).astype(np.float32)
            leftout_idx_fold = np.asarray(leftout_global_idx[leftout_study_mask], dtype=np.int64)
            knn_key_payload = json.dumps(
                {
                    "mode": "loso_backfill_leftout",
                    "outer_fold": str(test_study_name),
                    "fs_method": str(fs_method),
                    "n_genes": int(knn_n_genes),
                    "train_hash": _stable_hash_array(np.where(train_mask)[0]),
                    "leftout_hash": _stable_hash_array(leftout_idx_fold),
                    "k_values": list(KNN_REJECTION_K_VALUES),
                },
                sort_keys=True,
            ).encode("utf-8")
            knn_key = hashlib.sha256(knn_key_payload).hexdigest()
            split_features[str(test_study_name)] = {
                "sample_indices": leftout_idx_fold,
                "features": _load_or_compute_knn_features(local_cache_dir, knn_key, X_train_knn, X_leftout_fold_proc),
            }
    else:
        raise ValueError(f"Unsupported fold_type '{fold_type}'. Use 'CV' or 'loso'.")

    for col in KNN_DISTANCE_COLUMNS:
        if col not in result.columns:
            result[col] = None

    for row_idx in result.index:
        fold_key = str(result.at[row_idx, "outer_fold"])
        if fold_key not in split_features:
            if strict:
                raise ValueError(f"Missing leftout fold '{fold_key}' in computed KNN feature bundles.")
            continue
        fold_bundle = split_features[fold_key]
        row_indices = _parse_index_vector(result.at[row_idx, "sample_indices"])
        if len(row_indices) == 0:
            if strict:
                raise ValueError(f"Empty sample_indices vector at leftout row {row_idx}.")
            continue
        fold_indices = fold_bundle["sample_indices"]
        index_to_pos = {int(v): i for i, v in enumerate(fold_indices.tolist())}
        pos = [index_to_pos.get(int(v), None) for v in row_indices.tolist()]
        if any(p is None for p in pos):
            if strict:
                missing = [int(v) for v, p in zip(row_indices.tolist(), pos) if p is None]
                raise ValueError(
                    f"Leftout row {row_idx} has sample_indices not found in fold '{fold_key}': {missing[:10]}"
                )
            continue
        pos = np.asarray(pos, dtype=np.int64)
        feats = fold_bundle["features"]
        for col in KNN_DISTANCE_COLUMNS:
            vec = feats.get(col)
            if vec is None:
                if strict:
                    raise ValueError(f"Missing KNN feature '{col}' for fold '{fold_key}'.")
                continue
            result.at[row_idx, col] = json.dumps(np.asarray(vec, dtype=np.float32)[pos].tolist())

    # Strict post-check: all rows must have all KNN columns filled.
    if strict:
        for col in KNN_DISTANCE_COLUMNS:
            missing_mask = result[col].isna() | (result[col].astype(str).str.len() == 0)
            if missing_mask.any():
                n_missing = int(missing_mask.sum())
                raise ValueError(f"Leftout KNN backfill incomplete: '{col}' missing on {n_missing} rows.")

    return result

###################################################################################
# Helper functions                                                                #
###################################################################################


def load_data(directory):
    """
    Loads data from CSV files in a given directory,
    returning NumPy arrays.

    This function searches the provided directory for CSV files starting with specific
    file types (e.g. 'meta', 'counts', and 'RGAs'). The data is then loaded
    using pandas and converted to NumPy arrays.

    Parameters
    ----------
    directory : str
        Path to the directory containing the CSV files.

    Returns
    -------
    studies : np.ndarray
        NumPy array loaded from the 'Studies' column of the newest 'meta' file.
    X : np.ndarray
        NumPy array of gene count data loaded from the newest 'GDC_counts' file.
        Rows correspond to samples/observations, columns to genes/features.
    y : np.ndarray
        NumPy array of the target variable ('ICC_Subtype') loaded from the newest 'RGAs' file.
    """
    # List all files in the directory.
    files = os.listdir(directory)

    # Filter for CSV files first to avoid errors with non-CSV files
    csv_files = [f for f in files if f.lower().endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory}")

    # Select the file that starts with the required prefixes 
    meta_file = next(f for f in csv_files if f.startswith("meta_"))
    counts_file = next(f for f in csv_files if f.startswith("counts_"))
    rgas_file = next(f for f in csv_files if f.startswith("rgas_"))

    # Construct full paths
    meta_path = os.path.join(directory, meta_file)
    counts_path = os.path.join(directory, counts_file)
    rgas_path = os.path.join(directory, rgas_file)

    # Load CSV data into pandas DataFrames/Series.
    X_df = pd.read_csv(counts_path)
    # X_df = pd.read_csv(counts_path, index_col=0, engine='c')

    X_df = X_df.set_index(X_df.columns[0])

    X_df.index.name = None
    X_df.columns.name = None

    studies_series = pd.read_csv(meta_path)["Studies"]
    y_series = pd.read_csv(rgas_path, index_col=0)["ICC_Subtype"]
    
    print("\n")
    print(f"  studies_series: {len(studies_series)}")
    print(f"  X_df: {X_df.shape}")
    print(f"  y_series: {len(y_series)}")
    # --- Convert to NumPy arrays ---
    # .values returns the underlying numpy array representation
    studies = studies_series.values
    X = X_df.transpose().values.astype(np.float32)
    y = y_series.values

    print(f"  Studies: {len(studies)}")
    print(f"  X shape: {X.shape}")
    print(f"  y: {len(y)}")

    # Check if the number of samples aligns after loading
    if not (len(studies) == X.shape[0] == len(y)):
        raise ValueError("Loaded data dimensions do not align.")

    return X, y, studies


def filter_data(X, y, study_labels, min_n=20):
    """
    Removes samples based on class counts and selected studies.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target labels.
        study_labels (numpy.ndarray): Study labels.

    Returns:
        tuple: Filtered X, y, and study_labels.
    """
    X = np.array(X, dtype=np.float32)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[class_counts >= min_n]

    valid_classes = [c for c in valid_classes if c != "AML NOS" and c != "Missing data" and c != "Multi"]

    valid_indices_classes = np.isin(y, valid_classes)

    selected_studies = [
        "TCGA-LAML",
        "LEUCEGENE",
        "BEATAML1.0-COHORT",
        "AAML0531",
        "AAML1031",
        "AAML03P1",
        "100LUMC",
    ]

    valid_indices_studies = np.isin(study_labels, selected_studies)

    # Combine the indices to keep samples that satisfy both conditions
    valid_indices = valid_indices_classes & valid_indices_studies

    filtered_X = X[valid_indices]
    filtered_y = y[valid_indices]
    filtered_study_labels = study_labels[valid_indices]
    
    print("\n")
    print(f"  Studies: {len(filtered_study_labels)}")
    print(f"  X shape: {filtered_X.shape}")
    print(f"  y: {len(filtered_y)}")

    return filtered_X, filtered_y, filtered_study_labels


def build_hybrid_stratify_labels(y, study_labels, n_splits):
    """
    Build labels for StratifiedKFold on the given slice of data.

    Uses joint (subtype, study) when that cell has >= n_splits samples here;
    otherwise subtype-only so sparse (class, study) pairs do not form tiny
    strata. Matches main CV and left-out fold assignment logic.
    """
    y = np.asarray(y)
    study_labels = np.asarray(study_labels)
    if len(study_labels) != len(y):
        raise ValueError("y and study_labels must have the same length.")
    pairs = list(zip(y, study_labels))
    counts = Counter(pairs)
    return [
        str(a) + " " + str(b) if counts[(a, b)] >= n_splits else str(a)
        for a, b in pairs
    ]


def get_leftout_samples(X, y, study_labels, min_n=20):
    """
    Returns samples excluded from training by filter_data(), except Multi
    and Missing data, within the selected studies.

    These represent real subtypes the model may encounter at deployment but
    was not trained on (AML NOS, rare subtypes with n < min_n).
    """
    X = np.array(X, dtype=np.float32)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[class_counts >= min_n]
    valid_classes = [c for c in valid_classes if c != "AML NOS" and c != "Missing data" and c != "Multi"]

    selected_studies = [
        "TCGA-LAML",
        "LEUCEGENE",
        "BEATAML1.0-COHORT",
        "AAML0531",
        "AAML1031",
        "AAML03P1",
        "100LUMC",
    ]

    in_selected_studies = np.isin(study_labels, selected_studies)
    is_valid_class = np.isin(y, valid_classes)
    is_excluded_from_leftout = np.isin(y, ["Multi", "Missing data"])

    leftout_mask = in_selected_studies & ~is_valid_class & ~is_excluded_from_leftout

    leftout_X = X[leftout_mask]
    leftout_y = y[leftout_mask]
    leftout_study = study_labels[leftout_mask]
    leftout_global_idx = np.where(leftout_mask)[0]

    print(f"\n  Left-out samples: {len(leftout_y)}")
    if len(leftout_y) > 0:
        unique_leftout, counts = np.unique(leftout_y, return_counts=True)
        for cls, cnt in zip(unique_leftout, counts):
            print(f"    {cls}: {cnt}")
    print(f"  Left-out X shape: {leftout_X.shape}")

    return leftout_X, leftout_y, leftout_study, leftout_global_idx


def assign_leftout_to_loso_folds(study_leftout):
    """Assign left-out samples to LOSO folds by cohort (study name)."""
    study_leftout = np.asarray(study_leftout, dtype=object)
    if len(study_leftout) == 0:
        return np.array([], dtype=object)
    return study_leftout.copy()


def export_leftout_fold_assignment_csv(leftout_global_idx, leftout_fold_assignments, output_path):
    """
    Write per-fold left-out sample indices for final calibration augmentation.
    Schema matches outer leftout CSVs read by R/load_leftout_fold_assignment().
    outer_fold may be integer (CV) or study name string (LOSO).
    """
    leftout_global_idx = np.asarray(leftout_global_idx, dtype=np.int64)
    leftout_fold_assignments = np.asarray(leftout_fold_assignments, dtype=object)
    if len(leftout_global_idx) != len(leftout_fold_assignments):
        raise ValueError("leftout_global_idx and leftout_fold_assignments length mismatch.")
    if len(leftout_global_idx) == 0:
        raise ValueError("No left-out samples to export fold assignment for.")

    rows = []
    for fold in sorted(np.unique(leftout_fold_assignments).tolist(), key=str):
        mask = leftout_fold_assignments == fold
        idx = leftout_global_idx[mask].tolist()
        if not idx:
            continue
        outer_fold = int(fold) if str(fold).isdigit() else str(fold)
        rows.append({
            "outer_fold": outer_fold,
            "sample_indices": json.dumps(idx),
        })
    if not rows:
        raise ValueError("Left-out fold assignment produced no fold rows.")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved left-out fold assignment: {output_path}")


def assign_leftout_to_cv_folds(y_leftout, study_leftout, n_folds=5, random_state=None):
    """
    Assign left-out samples to CV folds with stratification and deterministic fallback.

    Same rule as main CV: hybrid strata (joint subtype+study when count >= n_folds
    in the left-out set, else subtype-only), then subtype-only, study-only, then
    balanced random modulo.

    random_state defaults to CV_RANDOM_STATE when None.
    """
    y_leftout = np.asarray(y_leftout)
    study_leftout = np.asarray(study_leftout)
    n_leftout = len(y_leftout)
    if n_leftout != len(study_leftout):
        raise ValueError("y_leftout and study_leftout must have the same length.")
    if n_leftout == 0:
        return np.array([], dtype=int)

    if random_state is None:
        random_state = CV_RANDOM_STATE
    rng = np.random.RandomState(random_state)

    # StratifiedKFold requires every stratum to have at least n_folds samples.
    def can_stratify(strata):
        _, counts = np.unique(strata, return_counts=True)
        return np.all(counts >= n_folds)

    def try_stratified_assignment(strata, label_name):
        if not can_stratify(strata):
            return None
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        assignments = np.zeros(n_leftout, dtype=int)
        dummy_X = np.zeros((n_leftout, 1), dtype=np.int8)
        for fold_idx, (_, test_idx) in enumerate(splitter.split(dummy_X, strata)):
            assignments[test_idx] = fold_idx
        print(f"Left-out fold assignment stratified by {label_name}.")
        return assignments

    hybrid = np.array(
        build_hybrid_stratify_labels(y_leftout, study_leftout, n_folds), dtype=object
    )
    for strata, label_name in (
        (hybrid, "hybrid (joint if n>=k else subtype)"),
        (y_leftout, "subtype"),
        (study_leftout, "study"),
    ):
        assignments = try_stratified_assignment(strata, label_name)
        if assignments is not None:
            return assignments

    # Final fallback keeps folds balanced and reproducible when strata are too sparse.
    assignments = np.zeros(n_leftout, dtype=int)
    shuffled_order = rng.permutation(n_leftout)
    for rank, idx in enumerate(shuffled_order):
        assignments[idx] = rank % n_folds
    print("Left-out fold assignment used balanced random fallback (no feasible stratification).")
    return assignments


def extract_n_genes_list(best_params):
    """Extract sorted unique n_genes values from a best_params DataFrame."""
    n_genes_list = []
    for params in best_params["params"]:
        try:
            parsed = ast.literal_eval(params) if isinstance(params, str) else params
            n_genes_list.append(parsed["n_genes"])
        except (ValueError, SyntaxError, KeyError) as e:
            print(f"Error parsing params: {params}, Error: {e}")
            continue
    if not n_genes_list:
        raise ValueError("No valid n_genes values found in best_params")
    return sorted(set(n_genes_list))


def encode_labels(y):
    """Encodes string labels to integers and returns the mapping."""
    unique_labels = np.unique(y)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_y = np.array([label_to_int[label] for label in y])
    return int_y, label_to_int


def restore_labels(df, label_mapping):
    df = df.dropna()

    int_to_label = {v: k for k, v in label_mapping.items()}
    if "class" in df.columns:
        # OvR case
        df["class_label"] = df["class"].map(int_to_label)
        return df

    elif "class_0" in df.columns and "class_1" in df.columns:
        # OvO case
        df["class_0_label"] = df["class_0"].map(int_to_label)
        df["class_1_label"] = df["class_1"].map(int_to_label)
        return df

    else:
        return df


# Costum metric


def conditional_f1(y_true, preds):
    """
    Calculates the F1-score, treating 1 as positive unless y_true contains only 0s,
    in which case it treats 0 as positive.
    """
    unique_y_true = np.unique(y_true)

    if len(unique_y_true) == 1 and unique_y_true[0] == 0:
        # Only 0s in y_true, treat 0 as positive
        return f1_score(y_true, preds, average="binary", pos_label=0)
    else:
        # Treat 1 as positive
        return f1_score(y_true, preds, average="binary", pos_label=1)


###################################################################################
# Main function to evaluate one set of hyperparameters for inner cross validation #
###################################################################################

def _knn_features_to_json(knn_features, feature_name):
    """Serialize one per-sample KNN vector for CSV output."""
    if not knn_features:
        return None
    vals = knn_features.get(feature_name)
    if vals is None:
        return None
    return json.dumps(np.asarray(vals, dtype=np.float32).tolist())


def _append_knn_columns(result_dict, knn_features):
    """Attach KNN reject-feature columns when vectors are available."""
    if not knn_features:
        return result_dict
    for col in KNN_DISTANCE_COLUMNS:
        result_dict[col] = _knn_features_to_json(knn_features, col)
    return result_dict


def compute_knn_features_full_reference(
    X_train,
    y_train,
    study_labels,
    X_eval,
    pipe,
    fs_method="eta2",
    knn_n_genes=KNN_REJECTION_FEATURE_N_GENES,
    cache_dir=None,
):
    """
    KNN distance summaries with the full training set as the reference space.
    Used for final-deployment left-out scoring (model fit on all included data).
    """
    local_cache_dir = cache_dir or _get_reject_cache_dir()
    pipe_clone = clone(pipe)
    X_train_proc = pipe_clone.fit_transform(
        X_train,
        y_train,
        feature_selection__study_per_patient=study_labels,
        feature_selection__n_genes=int(knn_n_genes),
    ).astype(np.float32)
    X_eval_proc = pipe_clone.transform(X_eval).astype(np.float32)
    knn_key_payload = json.dumps(
        {
            "mode": "final_full_reference",
            "fs_method": str(fs_method),
            "n_genes": int(knn_n_genes),
            "train_hash": _stable_hash_array(np.arange(X_train.shape[0])),
            "eval_hash": _stable_hash_array(np.arange(X_eval.shape[0])),
            "k_values": list(KNN_REJECTION_K_VALUES),
        },
        sort_keys=True,
    ).encode("utf-8")
    knn_key = hashlib.sha256(knn_key_payload).hexdigest()
    return _load_or_compute_knn_features(
        local_cache_dir, knn_key, X_train_proc, X_eval_proc
    )


def backfill_knn_columns_in_final_leftout_results(
    leftout_results_df,
    X,
    y,
    study_labels,
    X_leftout,
    leftout_global_idx,
    pipe,
    fs_method="eta2",
    knn_n_genes=KNN_REJECTION_FEATURE_N_GENES,
    cache_dir=None,
    strict=True,
):
    """
    Backfill KNN columns for final full-data left-out CSV rows (outer_fold = -1).
    Reference space is all included training samples, matching deployment inference.
    """
    if leftout_results_df is None or len(leftout_results_df) == 0:
        return leftout_results_df

    result = leftout_results_df.copy()
    features = compute_knn_features_full_reference(
        X,
        y,
        study_labels,
        X_leftout,
        pipe,
        fs_method=fs_method,
        knn_n_genes=knn_n_genes,
        cache_dir=cache_dir,
    )
    bundle = {
        "sample_indices": np.asarray(leftout_global_idx, dtype=np.int64),
        "features": features,
    }

    for col in KNN_DISTANCE_COLUMNS:
        if col not in result.columns:
            result[col] = None

    for row_idx in result.index:
        row_indices = _parse_index_vector(result.at[row_idx, "sample_indices"])
        if len(row_indices) == 0:
            if strict:
                raise ValueError(f"Empty sample_indices vector at leftout row {row_idx}.")
            continue
        fold_indices = bundle["sample_indices"]
        index_to_pos = {int(v): i for i, v in enumerate(fold_indices.tolist())}
        pos = [index_to_pos.get(int(v), None) for v in row_indices.tolist()]
        if any(p is None for p in pos):
            if strict:
                missing = [int(v) for v, p in zip(row_indices.tolist(), pos) if p is None]
                raise ValueError(
                    f"Leftout row {row_idx} has sample_indices not found in KNN bundle: {missing[:10]}"
                )
            continue
        pos = np.asarray(pos, dtype=np.int64)
        feats = bundle["features"]
        for col in KNN_DISTANCE_COLUMNS:
            vec = feats.get(col)
            if vec is None:
                if strict:
                    raise ValueError(f"Missing KNN feature '{col}' for final leftout backfill.")
                continue
            result.at[row_idx, col] = json.dumps(
                np.asarray(vec, dtype=np.float32)[pos].tolist()
            )

    if strict:
        for col in KNN_DISTANCE_COLUMNS:
            missing_mask = result[col].isna() | (result[col].astype(str).str.len() == 0)
            if missing_mask.any():
                n_missing = int(missing_mask.sum())
                raise ValueError(
                    f"Final leftout KNN backfill incomplete: '{col}' missing on {n_missing} rows."
                )

    return result


def evaluate_inner_fold(
    outer_fold,
    inner_fold,
    processed_X,
    y_train_inner,
    y_val_inner,
    original_val_inner_idx,
    model,
    params,
    multi_type="standard",
    model_type="any",
    knn_features=None,
):

    def standard_eval():
        # Create and fit label encoder ONLY on training data
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train_inner)
        
        # Transform training labels
        y_train_encoded = label_encoder.transform(y_train_inner)

        # Check for unseen classes to determine validation data for NN training
        unseen_classes = set(y_val_inner) - set(label_encoder.classes_)
        
        # Always fit model (with appropriate validation data for NN)
        if model_type == "NN":
            if unseen_classes:
                print(f"Warning: Validation data contains unseen classes: {unseen_classes}")
                # Use filtered validation data for NN training monitoring
                val_mask = np.isin(y_val_inner, list(label_encoder.classes_))
                X_val_for_training = X_val_inner[val_mask]
                y_val_for_training = label_encoder.transform(y_val_inner[val_mask])
                
                if len(X_val_for_training) > 0:
                    clf.fit(X_train_inner, y_train_encoded, validation_data=(X_val_for_training, y_val_for_training))
                else:
                    print("No validation samples with known classes. Training NN without validation monitoring.")
                    clf.fit(X_train_inner, y_train_encoded)
            else:
                # All validation classes are known
                y_val_encoded = label_encoder.transform(y_val_inner)
                clf.fit(X_train_inner, y_train_encoded, validation_data=(X_val_inner, y_val_encoded))
        else:
            clf.fit(X_train_inner, y_train_encoded)
        
        # Make predictions on ALL validation samples
        preds_prob = clf.predict_proba(X_val_inner)
        preds_encoded = np.argmax(preds_prob, axis=1)
        
        # Convert predictions back to original labels
        preds = label_encoder.inverse_transform(preds_encoded)
        
        # Prepare data for metric calculation (only samples with known classes)
        if unseen_classes:
            # Create mask for samples with known classes (for metric calculation)
            mask = np.isin(y_val_inner, list(label_encoder.classes_))
            y_val_inner_for_metrics = y_val_inner[mask]
            preds_for_metrics = preds[mask]
            
            if len(y_val_inner_for_metrics) == 0:
                print("All validation samples have unseen classes. Skipping metric calculation.")
                return None
        else:
            # All classes are known, use all samples for metrics
            y_val_inner_for_metrics = y_val_inner
            preds_for_metrics = preds
                
        if model_type == "NN":
            history = clf.model.history.history
            best_epoch = np.argmin(history['val_loss']) + 1  # Add 1 to match epoch count
            params["best_epoch"] = best_epoch
            
        preds_prob = preds_prob.flatten()
        preds_prob = np.round(preds_prob, 4)
        preds_prob = preds_prob.tolist()
        classes = label_encoder.classes_
        return _append_knn_columns({
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "classes": classes,
            "params": params,
            # Metrics calculated only on samples with known classes
            "accuracy": accuracy_score(y_val_inner_for_metrics, preds_for_metrics),
            "f1_macro": f1_score(y_val_inner_for_metrics, preds_for_metrics, average="macro"),
            "mcc": matthews_corrcoef(y_val_inner_for_metrics, preds_for_metrics),
            "kappa": cohen_kappa_score(y_val_inner_for_metrics, preds_for_metrics),
            # But return predictions and true labels for ALL validation samples
            "y_val": y_val_inner,  # All validation labels (including unseen classes)
            "preds": preds,        # All predictions 
            "preds_prob": json.dumps(preds_prob),  # All prediction probabilities
            "sample_indices": original_val_inner_idx  # All sample indices
        }, knn_features)

    def ovr_eval():
        results = []
        
        classes_train = np.unique(y_train_inner)
        
        for cl in classes_train:
            # Check if class exists in both sets using masks
            train_mask = (y_train_inner == cl)
            val_mask = (y_val_inner == cl)
            
            if not np.any(train_mask) or not np.any(val_mask):
                print(f"Skipping class {cl} for OvR - not present in train or validation")
                continue

            y_train_bin = [1 if yy == cl else 0 for yy in y_train_inner]
            y_val_bin = [1 if yy == cl else 0 for yy in y_val_inner]

            y_train_bin = np.array(y_train_bin, dtype=np.int32)
            y_val_bin = np.array(y_val_bin, dtype=np.int32)
            
            clf.fit(X_train_inner, y_train_bin)
            preds_prob = clf.predict_proba(X_val_inner)
            preds_prob = preds_prob[:, 1]
            preds = (preds_prob >= 0.5).astype(int)

            preds_prob = np.round(preds_prob, 4)
            preds_prob = preds_prob.tolist()
            
            results.append(
                _append_knn_columns({
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "class": cl,
                    "params": params,
                    "accuracy": accuracy_score(y_val_bin, preds),
                    "f1_binary": conditional_f1(y_val_bin, preds),
                    "mcc": matthews_corrcoef(y_val_bin, preds),
                    "kappa": cohen_kappa_score(y_val_bin, preds),
                    "y_val": y_val_bin,
                    "preds": preds,
                    "preds_prob": json.dumps(preds_prob),
                    "sample_indices": original_val_inner_idx
                }, knn_features)
            )
        return results

    def ovo_eval():
        results = []
        
        classes = np.unique(y_train_inner)
        for i, j in itertools.combinations(classes, 2):

            # Create masks for both classes
            train_mask = np.isin(y_train_inner, [i, j])
            val_mask = np.isin(y_val_inner, [i, j])
            
            # Skip if no samples for either class in train or val
            if not np.any(train_mask) or not np.any(val_mask):
                print(f"Skipping classes {i} and {j} - no samples in train or val")
                continue

            # Get filtered data
            X_train_ij = X_train_inner[train_mask]
            y_train_ij = y_train_inner[train_mask]
            y_val_ij = y_val_inner[val_mask]
            
            # Convert to binary (j=1, i=0)
            y_train_bin = (y_train_ij == j).astype(np.int32)
            y_val_bin = (y_val_ij == j).astype(np.int32)
            
            # Skip if only one class present in either set
            if len(np.unique(y_train_bin)) != 2 or len(np.unique(y_val_bin)) != 2:
                print(f"Skipping classes {i} and {j} - only one class present")
                continue

            clf.fit(X_train_ij, y_train_bin)
            preds_prob = clf.predict_proba(X_val_inner)
            pos_class_index = list(clf.classes_).index(1)
            preds_prob = preds_prob[:, pos_class_index]
            preds = (preds_prob >= 0.5).astype(int)

            preds_prob = np.round(preds_prob, 4)
            preds_prob = preds_prob.tolist()
            
            results.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "class_0": i,
                    "class_1": j,
                    "params": params,
                    "accuracy": 0,
                    "f1_binary": 0,
                    "mcc": 0,
                    "kappa": 0,
                    "y_val": y_val_inner,
                    "preds": preds,
                    "preds_prob": json.dumps(preds_prob),
                    "sample_indices": original_val_inner_idx
                }
            )
        return results

    # Dispatch table for clean logic
    eval_dispatch = {
        "standard": standard_eval, 
        "OvR": ovr_eval, 
        "OvO": ovo_eval
        }

    if multi_type not in eval_dispatch:
        raise ValueError(f"Unsupported evaluation type: {multi_type}")

    # Select preprocessed data
    n_genes = params.pop("n_genes")
    X_train_inner, X_val_inner = processed_X[n_genes]

    # Set classifier
    clf = model(**params)
    params["n_genes"] = n_genes

    del model
    gc.collect()
    
    output = eval_dispatch[multi_type]()
    
    return output

def evaluate_outer_fold(
    outer_fold,
    processed_X,
    y_train,
    y_test,
    test_idx,
    model,
    best_params_fold,
    multi_type="standard",
    model_type="any",
    knn_features=None,
):
    def _knn_json(feature_name):
        if not knn_features:
            return None
        vals = knn_features.get(feature_name)
        if vals is None:
            return None
        return json.dumps(np.asarray(vals, dtype=np.float32).tolist())

    def standard_eval():
        # Select preprocessed data
        params = best_params_fold.iloc[0]["params"]
        params = ast.literal_eval(params)
        n_genes = params.pop("n_genes")
        X_train, X_test = processed_X[n_genes]

        # Create label encoder to map labels to consecutive integers
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)  # Fit only on training data
        
        # Fit and transform training labels
        y_train_encoded = label_encoder.transform(y_train)

        # Set classifier
        clf = model(**params)
        params["n_genes"] = n_genes

        if model_type == "NN":
            best_epoch = params.get("best_epoch", None)
            if best_epoch is None:
                raise ValueError("Best Epochs for NN is not available.")
            clf.fit(X_train, y_train_encoded, epochs = best_epoch)
        else:
            clf.fit(X_train, y_train_encoded)

        # Make predictions on ALL test samples
        preds_prob = clf.predict_proba(X_test)
        preds_encoded = np.argmax(preds_prob, axis=1)
        
        # Remap predictions back to original labels
        preds = label_encoder.inverse_transform(preds_encoded)
        classes = label_encoder.classes_
            
        preds_prob = preds_prob.flatten()
        preds_prob = np.round(preds_prob, 4)
        preds_prob = preds_prob.tolist()

        return {
            "outer_fold": outer_fold,
            "classes": classes,
            "params": params,
            # Metrics calculated only on samples with known classes
            "accuracy": 0,
            "f1_macro": 0,
            "mcc": 0,
            "kappa": 0,
            # But return predictions and true labels for ALL test samples
            "y_val": y_test,  # All test labels (including unseen classes)
            "preds": preds,   # All predictions
            "preds_prob": json.dumps(preds_prob),  # All prediction probabilities
            "sample_indices": test_idx,  # All sample indices
            "knn10_mean_d": _knn_json("knn10_mean_d"),
            "knn10_min_d": _knn_json("knn10_min_d"),
            "knn10_q90_d": _knn_json("knn10_q90_d"),
            "knn20_mean_d": _knn_json("knn20_mean_d"),
            "knn20_min_d": _knn_json("knn20_min_d"),
            "knn20_q90_d": _knn_json("knn20_q90_d"),
        }

    def ovr_eval():
        results = []
        
        classes = np.unique(y_train)
        for cl in classes:
            # Check if class exists in both sets using masks
            train_mask = (y_train == cl)
            val_mask = (y_test == cl)
            
            if not np.any(train_mask) or not np.any(val_mask):
                print(f"Skipping class {cl} for OvR - not present in train or validation")
                continue
            
            # Check if best parameters exist for this class
            class_params = best_params_fold[best_params_fold["class"] == cl]
            if len(class_params) == 0:
                print(f"Skipping class {cl} for OvR - no best parameters found")
                continue
            
            # Select preprocessed data
            params = class_params.iloc[0]["params"]
            params = ast.literal_eval(params)
            n_genes = params.pop("n_genes")

            # Set classifier
            clf = model(**params)
            params["n_genes"] = n_genes

            # Get correct data
            X_train, X_test = processed_X[n_genes]

            y_train_bin = [1 if yy == cl else 0 for yy in y_train]
            y_test_bin = [1 if yy == cl else 0 for yy in y_test]

            y_train_bin = np.array(y_train_bin, dtype=np.int32)
            y_test_bin = np.array(y_test_bin, dtype=np.int32)

            clf.fit(X_train, y_train_bin)
            preds_prob = clf.predict_proba(X_test)

            pos_class_index = list(clf.classes_).index(1)
            preds_prob = preds_prob[:, pos_class_index]
            preds = (preds_prob >= 0.5).astype(int)
            
            preds_prob = np.round(preds_prob, 4)
            preds_prob = preds_prob.tolist()
            
            results.append(
                {
                    "outer_fold": outer_fold,
                    "class": cl,
                    "params": params,
                    "accuracy": 0,
                    "f1_binary": 0,
                    "mcc": 0,
                    "kappa": 0,
                    "y_val": y_test_bin,
                    "preds": preds,
                    "preds_prob": json.dumps(preds_prob),
                    "sample_indices": test_idx,
                    "knn10_mean_d": _knn_json("knn10_mean_d"),
                    "knn10_min_d": _knn_json("knn10_min_d"),
                    "knn10_q90_d": _knn_json("knn10_q90_d"),
                    "knn20_mean_d": _knn_json("knn20_mean_d"),
                    "knn20_min_d": _knn_json("knn20_min_d"),
                    "knn20_q90_d": _knn_json("knn20_q90_d"),
                }
            )
        return results

    def ovo_eval():
        results = []
        
        classes = np.unique(y_train)
        for i, j in itertools.combinations(classes, 2):

            # Check if best parameters exist for this class pair
            class_pair_params = best_params_fold[
                (best_params_fold["class_0"] == i) & (best_params_fold["class_1"] == j)
            ]
            if len(class_pair_params) == 0:
                print(f"Skipping classes {i} and {j} - no best parameters found")
                continue

            # Select preprocessed data
            params = class_pair_params.iloc[0]["params"]
            params = ast.literal_eval(params)
            n_genes = params.pop("n_genes")

            # Set classifier
            clf = model(**params)
            params["n_genes"] = n_genes

            # Get correct data
            X_train, X_test = processed_X[n_genes]

            # Create masks for both classes
            train_mask = np.isin(y_train, [i, j])
            test_mask = np.isin(y_test, [i, j])
            
            # Skip if no samples for either class in train or test
            if not np.any(train_mask) or not np.any(test_mask):
                print(f"Skipping classes {i} and {j} - no samples in train or test")
                continue

            # Get filtered data
            X_train_ij = X_train[train_mask]
            y_train_ij = y_train[train_mask]
            y_test_ij = y_test[test_mask]
            
            # Convert to binary (j=1, i=0)
            y_train_ij = (y_train_ij == j).astype(np.int32)
            y_test_ij = (y_test_ij == j).astype(np.int32)
            
            # Skip if only one class present in either set
            if len(np.unique(y_train_ij)) != 2 or len(np.unique(y_test_ij)) != 2:
                print(f"Skipping classes {i} and {j} - only one class present")
                continue

            clf.fit(X_train_ij, y_train_ij)
            preds_prob = clf.predict_proba(X_test)
            pos_class_index = list(clf.classes_).index(1)
            preds_prob = preds_prob[:, pos_class_index]
            preds = (preds_prob >= 0.5).astype(int)

            preds_prob = np.round(preds_prob, 4)
            preds_prob = preds_prob.tolist()
            
            
            results.append(
                {
                    "outer_fold": outer_fold,
                    "class_0": i,
                    "class_1": j,
                    "params": params,
                    "accuracy": 0,
                    "f1_binary": 0,
                    "mcc": 0,
                    "kappa": 0,
                    "y_val": y_test,
                    "preds": preds,
                    "preds_prob": json.dumps(preds_prob),
                    "sample_indices": test_idx,
                    "knn10_mean_d": _knn_json("knn10_mean_d"),
                    "knn10_min_d": _knn_json("knn10_min_d"),
                    "knn10_q90_d": _knn_json("knn10_q90_d"),
                    "knn20_mean_d": _knn_json("knn20_mean_d"),
                    "knn20_min_d": _knn_json("knn20_min_d"),
                    "knn20_q90_d": _knn_json("knn20_q90_d"),
                }
            )
        return results

    # Dispatch table for clean logic
    eval_dispatch = {
        "standard": standard_eval, 
        "OvR": ovr_eval, 
        "OvO": ovo_eval
        }

    if multi_type not in eval_dispatch:
        raise ValueError(f"Unsupported evaluation type: {multi_type}")

    return eval_dispatch[multi_type]()

###################################################################################
# Main functions for standard inner cross validation                              #
###################################################################################


def pre_process_data(
    n_genes_list,
    X_train_outer,
    y_train_outer,
    train_idx,
    val_idx,
    study_labels_outer,
    pipe,
    return_pipelines=False,
    cache_dir=None,
    split_tag=None,
    fs_method="unknown",
):
    """
    Preprocesses training and validation sets for different n_genes.
    Fits the pipeline ONLY on the training set.

    When return_pipelines is True, also returns the fitted pipeline objects
    so they can be reused to transform additional data (e.g. left-out samples).
    """
    X_train = X_train_outer[train_idx]
    X_val = X_train_outer[val_idx]

    study_labels_train = study_labels_outer[train_idx]

    y_train = y_train_outer[train_idx]
    y_val = y_train_outer[val_idx]

    y_train = np.array(y_train, dtype=np.int32)
    y_val = np.array(y_val, dtype=np.int32)

    processed_X = {}
    fitted_pipelines = {}
    local_cache_dir = cache_dir or _get_reject_cache_dir()
    for n_genes_i in n_genes_list:
        cache_key = _build_preprocess_cache_key(
            split_tag=split_tag or "cv",
            n_genes=n_genes_i,
            train_indices=train_idx,
            test_indices=val_idx,
            y_train=y_train,
            study_train=study_labels_train,
            fs_method=fs_method,
        )
        cached = _load_preprocess_cache(local_cache_dir, cache_key)
        if cached is not None:
            X_train_proc = cached["X_train_proc"]
            X_val_proc = cached["X_eval_proc"]
            if return_pipelines and "pipeline" in cached:
                fitted_pipelines[n_genes_i] = cached["pipeline"]
            processed_X[n_genes_i] = [X_train_proc, X_val_proc]
            continue

        pipe_clone = clone(pipe)
        # Pass y_train so FeatureSelectionEta (eta2) receives subtype labels; ignored by MAD selector.
        X_train_proc = pipe_clone.fit_transform(
            X_train,
            y_train,
            feature_selection__study_per_patient=study_labels_train,
            feature_selection__n_genes=n_genes_i,
        ).astype(np.float32)
        X_val_proc = pipe_clone.transform(X_val).astype(np.float32)

        payload = {
            "X_train_proc": X_train_proc,
            "X_eval_proc": X_val_proc,
        }
        if return_pipelines:
            payload["pipeline"] = pipe_clone
            fitted_pipelines[n_genes_i] = pipe_clone
        _save_preprocess_cache(local_cache_dir, cache_key, payload)
        processed_X[n_genes_i] = [X_train_proc, X_val_proc]

    if return_pipelines:
        return processed_X, y_train, y_val, fitted_pipelines
    return processed_X, y_train, y_val


def run_inner_cv(
    X,
    y,
    study_labels,
    model,
    param_grid,
    n_jobs,
    pipe,
    multi_type="standard",
    k_out=5,
    k_in=5,
    model_type = "any"
):
    # Define cv folds
    outer_cv = StratifiedKFold(
        n_splits=k_out, shuffle=True, random_state=CV_RANDOM_STATE
    )
    inner_cv = StratifiedKFold(
        n_splits=k_in, shuffle=True, random_state=CV_RANDOM_STATE
    )

    param_combos = param_grid
    n_genes_list = sorted({params["n_genes"] for params in param_combos})
    # Empty list to append results to
    all_results = []

    combined = build_hybrid_stratify_labels(y, study_labels, k_out)

    # Make outer fold splits
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, combined)):
        print("outer_fold")
        print(outer_fold)
        X_train_outer = X[train_idx]
        y_train_outer = y[train_idx]
        study_labels_outer = study_labels[train_idx]

        combined_outer = build_hybrid_stratify_labels(
            y_train_outer, study_labels_outer, k_in
        )

        # Make inner fold splits
        for inner_fold, (train_inner_idx, val_inner_idx) in enumerate(
            inner_cv.split(X_train_outer, combined_outer)
        ):
            print("inner_fold")
            print(inner_fold)

            # Once per inner fold, data is preprocessed
            processed_X, y_train_inner, y_val_inner = pre_process_data(
                n_genes_list,
                X_train_outer,
                y_train_outer,
                train_inner_idx,
                val_inner_idx,
                study_labels_outer,
                pipe,
            )

            inner_tasks = []

            # Get the original indices of validation samples in the full dataset
            original_val_inner_idx = train_idx[val_inner_idx]  # train_idx contains indices of outer training set, val_inner_idx contains indices within that set
            
            # Then, for every hyperparameter combo performance is evaluated
            for params in param_combos:
                inner_tasks.append(
                    delayed(evaluate_inner_fold)(
                        outer_fold,
                        inner_fold,
                        processed_X,
                        y_train_inner,
                        y_val_inner,
                        original_val_inner_idx,
                        model,
                        params,
                        multi_type=multi_type,  # standard, OvR, OvO
                        model_type = model_type
                    )
                )

            # Run inner CV tasks in parallel
            inner_results = Parallel(n_jobs=n_jobs, verbose=1)(inner_tasks)

            # Filter out None results (from folds with unseen classes)
            valid_results = [res for res in inner_results if res is not None]
            
            if valid_results:
                if isinstance(valid_results[0], dict):
                    # Flat list of dictionaries
                    all_results.extend(valid_results)
                elif isinstance(valid_results[0], list):
                    # List of lists of dictionaries
                    for res in valid_results:
                        all_results.extend(res)
                else:
                    raise ValueError("Unexpected structure in inner_results")
            else:
                print(f"Warning: No valid results for outer fold {outer_fold}, inner fold {inner_fold}")

    # Convert to DataFrame
    df_parallel_results = pd.DataFrame(all_results)
    return df_parallel_results

def run_inner_cv_single_param(
    X,
    y,
    study_labels,
    model,
    single_param,
    pipe,
    multi_type="standard",
    k_out=5,
    k_in=5,
    model_type="any"
):
    """
    Modified version of run_inner_cv that processes only one hyperparameter combination.
    Used for SLURM array jobs where each job handles one parameter set.
    """
    # Define cv folds
    outer_cv = StratifiedKFold(
        n_splits=k_out, shuffle=True, random_state=CV_RANDOM_STATE
    )
    inner_cv = StratifiedKFold(
        n_splits=k_in, shuffle=True, random_state=CV_RANDOM_STATE
    )

    # Single parameter instead of list
    n_genes_list = [single_param["n_genes"]]
    all_results = []

    combined = build_hybrid_stratify_labels(y, study_labels, k_out)

    # Make outer fold splits
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, combined)):
        print(f"outer_fold {outer_fold}")
        X_train_outer = X[train_idx]
        y_train_outer = y[train_idx]
        study_labels_outer = study_labels[train_idx]

        combined_outer = build_hybrid_stratify_labels(
            y_train_outer, study_labels_outer, k_in
        )

        # Make inner fold splits
        for inner_fold, (train_inner_idx, val_inner_idx) in enumerate(
            inner_cv.split(X_train_outer, combined_outer)
        ):
            print(f"inner_fold {inner_fold}")

            # Once per inner fold, data is preprocessed
            processed_X, y_train_inner, y_val_inner = pre_process_data(
                n_genes_list,
                X_train_outer,
                y_train_outer,
                train_inner_idx,
                val_inner_idx,
                study_labels_outer,
                pipe,
            )

            # Get the original indices of validation samples in the full dataset
            original_val_inner_idx = train_idx[val_inner_idx]
            
            # Process single hyperparameter combination (no parallel processing needed)
            result = evaluate_inner_fold(
                outer_fold,
                inner_fold,
                processed_X,
                y_train_inner,
                y_val_inner,
                original_val_inner_idx,
                model,
                single_param.copy(),  # Make copy to avoid modifying original
                multi_type=multi_type,
                model_type=model_type
            )

            # Handle result
            if result is not None:
                if isinstance(result, dict):
                    all_results.append(result)
                elif isinstance(result, list):
                    all_results.extend(result)
            else:
                print(f"Warning: No valid results for outer fold {outer_fold}, inner fold {inner_fold}")

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    return df_results


def run_train_test_single_param(
    X,
    y,
    study_labels,
    model,
    single_param,
    pipe,
    multi_type="standard",
    k_out=5,
    model_type="any",
    fs_method="unknown",
    cache_dir=None,
    knn_n_genes=KNN_REJECTION_FEATURE_N_GENES,
):
    """
    Function for final hyperparameter selection using simple train/test splits.
    Similar to run_inner_cv_single_param but without inner cross-validation.
    Used to select the best hyperparameters after inner CV evaluation.
    """
    # Define cv folds for train/test splits
    outer_cv = StratifiedKFold(
        n_splits=k_out, shuffle=True, random_state=CV_RANDOM_STATE
    )

    # Single parameter instead of list; always include KNN reference space.
    n_genes_list = sorted(list({single_param["n_genes"], int(knn_n_genes)}))
    all_results = []

    combined = build_hybrid_stratify_labels(y, study_labels, k_out)

    # Make train/test splits (no inner CV)
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, combined)):
        print(f"fold {fold}")
        
        # Once per fold, data is preprocessed
        processed_X, y_train, y_test = pre_process_data(
            n_genes_list,
            X,
            y,
            train_idx,
            test_idx,
            study_labels,
            pipe,
            cache_dir=cache_dir,
            split_tag=f"final_selection_{fold}",
            fs_method=fs_method,
        )

        knn_features = None
        if int(knn_n_genes) in processed_X:
            X_train_knn, X_test_knn = processed_X[int(knn_n_genes)]
            knn_cache_dir = cache_dir or _get_reject_cache_dir()
            knn_key_payload = json.dumps(
                {
                    "mode": "final_selection_cv",
                    "outer_fold": int(fold),
                    "fs_method": str(fs_method),
                    "n_genes": int(knn_n_genes),
                    "train_hash": _stable_hash_array(train_idx),
                    "test_hash": _stable_hash_array(test_idx),
                    "k_values": list(KNN_REJECTION_K_VALUES),
                },
                sort_keys=True,
            ).encode("utf-8")
            knn_key = hashlib.sha256(knn_key_payload).hexdigest()
            knn_features = _load_or_compute_knn_features(
                knn_cache_dir,
                knn_key,
                X_train_knn,
                X_test_knn,
            )
        
        # Process single hyperparameter combination
        result = evaluate_inner_fold(
            fold,  # Use fold as outer_fold
            0,     # No inner fold, use 0
            processed_X,
            y_train,
            y_test,
            test_idx,  # Original test indices
            model,
            single_param.copy(),  # Make copy to avoid modifying original
            multi_type=multi_type,
            model_type=model_type,
            knn_features=knn_features,
        )

        # Handle result
        if result is not None:
            if isinstance(result, dict):
                all_results.append(result)
            elif isinstance(result, list):
                all_results.extend(result)
        else:
            print(f"Warning: No valid results for fold {fold}")

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    return df_results


def run_train_test_loso_single_param(
    X,
    y,
    study_labels,
    model,
    single_param,
    pipe,
    multi_type="standard",
    model_type="any"
):
    """
    Function for final hyperparameter selection using LOSO (Leave One Study Out) splits.
    Similar to run_inner_cv_loso_single_param but without inner cross-validation.
    Used to select the best hyperparameters after inner CV evaluation.
    """
    # Single parameter instead of list
    n_genes_list = [single_param["n_genes"]]
    all_results = []
    
    studies_as_folds = np.unique(study_labels)
    
    for test_study_name in studies_as_folds:
        print(f"--- Fold: Holding out Study '{test_study_name}' for Testing ---")

        # Create masks for train/test split
        test_mask = study_labels == test_study_name
        train_mask = ~test_mask

        # Split data
        X_train = X[train_mask]
        y_train = y[train_mask]
        study_labels_train = study_labels[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        # Get test indices
        test_idx = np.where(test_mask)[0]

        # Pre-process Data for this fold
        processed_X = pre_process_data_loso(
            n_genes_list,
            X_train,
            X_test,
            study_labels_train,
            y_train,
            pipe
        )

        # Process single hyperparameter combination
        result = evaluate_inner_fold(
            test_study_name,  # Use study name as fold identifier
            0,               # No inner fold, use 0
            processed_X,
            y_train,
            y_test,
            test_idx,
            model,
            single_param.copy(),  # Make copy to avoid modifying original
            multi_type=multi_type,
            model_type=model_type
        )

        # Handle result
        if result is not None:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, dict):
                all_results.append(result)
        else:
            print(f"  No valid results for fold '{test_study_name}'")

        print(f"  Finished evaluation for fold '{test_study_name}'.")

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    return df_results


def run_outer_cv(
    X,
    y,
    study_labels,
    model,
    pipe,
    best_params,
    multi_type="standard",
    model_type = "any",
    fs_method="unknown",
    cache_dir=None,
    knn_n_genes=KNN_REJECTION_FEATURE_N_GENES,
):

    
    # Import best parameters if a string path is provided
    if isinstance(best_params, str):
        best_params = pd.read_csv(best_params)
    
    # Extract n_genes from params column
    n_genes_list = []
    for params in best_params['params']:
        try:
            if isinstance(params, dict):
                n_genes_list.append(params['n_genes'])
            else:
                # Assume it's a string representation of a dictionary
                parsed_params = ast.literal_eval(params)
                n_genes_list.append(parsed_params['n_genes'])
        except (ValueError, SyntaxError, KeyError) as e:
            print(f"Error parsing params: {params}")
            print(f"Error details: {e}")
            continue
    
    # Ensure we have valid n_genes values
    if not n_genes_list:
        raise ValueError("No valid n_genes values found in best_params")
    
    # Remove duplicates and sort. Ensure KNN reference space is available.
    n_genes_list = sorted(list(set(n_genes_list + [int(knn_n_genes)])))
    # Keep outer CV fold assignments reproducible and aligned with inner-CV setup.
    outer_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=CV_RANDOM_STATE
    )

    # Empty list to append results to
    all_results = []

    combined = build_hybrid_stratify_labels(y, study_labels, 5)

    # Make outer fold splits
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, combined)):
        print("outer_fold")
        print(outer_fold)

        # Once per outer fold, data is preprocessed
        processed_X, y_train, y_test = pre_process_data(
            n_genes_list,
            X,
            y,
            train_idx,
            test_idx,
            study_labels,
            pipe,
            cache_dir=cache_dir,
            split_tag=f"outer_cv_{outer_fold}",
            fs_method=fs_method,
        )

        knn_features = None
        if int(knn_n_genes) in processed_X:
            X_train_knn, X_test_knn = processed_X[int(knn_n_genes)]
            knn_cache_dir = cache_dir or _get_reject_cache_dir()
            knn_key_payload = json.dumps(
                {
                    "mode": "cv",
                    "outer_fold": int(outer_fold),
                    "fs_method": str(fs_method),
                    "n_genes": int(knn_n_genes),
                    "train_hash": _stable_hash_array(train_idx),
                    "test_hash": _stable_hash_array(test_idx),
                    "k_values": list(KNN_REJECTION_K_VALUES),
                },
                sort_keys=True,
            ).encode("utf-8")
            knn_key = hashlib.sha256(knn_key_payload).hexdigest()
            knn_features = _load_or_compute_knn_features(
                knn_cache_dir,
                knn_key,
                X_train_knn,
                X_test_knn,
            )
        
        # Filter best_params to get only rows for current outer fold
        best_params_fold = best_params[best_params['outer_fold'] == outer_fold]
        
        outer_results = evaluate_outer_fold(
                    outer_fold,
                    processed_X,
                    y_train,
                    y_test,
                    test_idx,
                    model,
                    best_params_fold,
                    multi_type=multi_type,  # standard, OvR, OvO
                    model_type = model_type,
                    knn_features=knn_features,
                
            )
        # Flatten inner_results list if needed and append to all_results
        if outer_results is None:
            print(f"Warning: No valid results for outer fold {outer_fold}")
            continue
        elif isinstance(outer_results, dict):
            # Single dictionary result
            all_results.append(outer_results)
        elif isinstance(outer_results, list):
            # List of dictionaries
            all_results.extend(outer_results)
        else:
            raise ValueError("Unexpected structure in outer_results")

    # Convert to DataFrame
    df_parallel_results = pd.DataFrame(all_results)
    return df_parallel_results

###################################################################################
# Main function for leave one study out (loso) inner cross validation             #
###################################################################################


def pre_process_data_loso(
    n_genes_list,
    X_train,
    X_test,
    study_labels_inner,  # Labels corresponding to X_train_inner
    y_train_inner,       # Subtype labels for X_train (required for FeatureSelectionEta)
    pipe,
    return_pipelines=False,
    cache_dir=None,
    split_tag=None,
    fs_method="unknown",
):
    """
    Preprocesses training and test/validation sets for different n_genes.
    Fits the pipeline ONLY on the training set.

    When return_pipelines is True, also returns the fitted pipeline objects
    so they can be reused to transform additional data (e.g. left-out samples).
    """
    processed_X = {}
    fitted_pipelines = {}
    local_cache_dir = cache_dir or _get_reject_cache_dir()
    for n_genes_i in n_genes_list:
        cache_key = _build_preprocess_cache_key(
            split_tag=split_tag or "loso",
            n_genes=n_genes_i,
            train_indices=np.arange(X_train.shape[0]),
            test_indices=np.arange(X_test.shape[0]),
            y_train=y_train_inner,
            study_train=study_labels_inner,
            fs_method=fs_method,
        )
        cached = _load_preprocess_cache(local_cache_dir, cache_key)
        if cached is not None:
            X_train_proc = cached["X_train_proc"]
            X_test_proc = cached["X_eval_proc"]
            if return_pipelines and "pipeline" in cached:
                fitted_pipelines[n_genes_i] = cached["pipeline"]
            processed_X[n_genes_i] = [X_train_proc, X_test_proc]
            continue

        # Clone the pipeline for this specific n_genes setting.
        pipe_inner = clone(pipe)
        # Pass y_train_inner so FeatureSelectionEta (eta2) receives subtype labels.
        X_train_proc = pipe_inner.fit_transform(
            X_train,
            y_train_inner,
            feature_selection__study_per_patient=study_labels_inner,
            feature_selection__n_genes=n_genes_i,
        ).astype(np.float32)
        # Transform validation/test data using the fitted pipeline.
        X_test_proc = pipe_inner.transform(X_test).astype(np.float32)

        payload = {
            "X_train_proc": X_train_proc,
            "X_eval_proc": X_test_proc,
        }
        if return_pipelines:
            payload["pipeline"] = pipe_inner
            fitted_pipelines[n_genes_i] = pipe_inner
        _save_preprocess_cache(local_cache_dir, cache_key, payload)
        processed_X[n_genes_i] = [X_train_proc, X_test_proc]

    if return_pipelines:
        return processed_X, fitted_pipelines
    return processed_X


def run_inner_cv_loso(
    X,
    y,
    study_labels,
    model,
    param_grid,
    n_jobs,
    pipe,
    multi_type= "standard",
    model_type = "any"
):
    
    param_combos = param_grid

    all_results = []
    n_genes_list = sorted({params["n_genes"] for params in param_combos})
    studies_as_folds = np.unique(study_labels)
    for test_study_name in studies_as_folds:
        print(
            f"\n--- Outer Loop: Holding out Study '{test_study_name}' for Testing ---"
        )

        # Create masks for outer split
        test_mask = study_labels == test_study_name
        train_mask = ~test_mask

        # Outer training set (N-1 studies)
        X_train_outer = X[train_mask]
        y_train_outer = y[train_mask]
        study_labels_outer = study_labels[train_mask]  # Labels for outer training set

        # Get the unique studies present in the outer training set
        train_studies = np.unique(study_labels_outer)
        print(f"Outer training set contains studies: {train_studies.tolist()}")

        # Inner Loop: Iterate through each study in the outer training set to be used as VALIDATION set
        for validation_study_name in train_studies:
            print(f"  Inner Loop: Validating on Study '{validation_study_name}'")
            # Create masks for inner split (relative to outer training data)
            val_inner_mask = study_labels_outer == validation_study_name
            train_inner_mask = ~val_inner_mask

            # Inner training set (N-2 studies)
            X_train_inner = X_train_outer[train_inner_mask]
            y_train_inner = y_train_outer[train_inner_mask]
            study_labels_inner = study_labels_outer[
                train_inner_mask
            ]  # Labels for inner training

            # Inner validation set (1 study)
            X_val_inner = X_train_outer[val_inner_mask]
            y_val_inner = y_train_outer[val_inner_mask]

            # Get the original indices of validation samples in the full dataset
            train_indices = np.where(train_mask)[0]  # Indices of outer training set in full dataset
            original_val_inner_idx = train_indices[val_inner_mask]  # Indices of validation samples in full dataset
            
            # --- Pre-process Data ONCE for this inner fold ---
            # This computes processed versions for all n_genes values
            processed_X_inner = (
                pre_process_data_loso(
                    n_genes_list,
                    X_train_inner,
                    X_val_inner,
                    study_labels_inner,  # Pass inner training labels for pipeline fitting
                    y_train_inner,
                    pipe
                )
            )

            tasks = []

            # --- Create tasks for hyperparameter evaluation for THIS inner fold ---
            for params in param_combos:
                # Append a delayed evaluation task for each hyperparameter combination
                tasks.append(
                    delayed(evaluate_inner_fold)(
                        test_study_name,  # Identifier for the outer fold (held-out test study)
                        validation_study_name,  # Identifier for the inner fold (validation study)
                        processed_X_inner,  # Pre-calculated processed data for all n_genes
                        y_train_inner,  # Inner training labels
                        y_val_inner,  # Inner validation labels,
                        original_val_inner_idx, # Inner indices based on whole data
                        model,  # Classifier class
                        params,  # Current hyperparameter combination
                        multi_type=multi_type,  # Choose evaluation type: "standard", "OvR", "OvO"
                        model_type = model_type
                    )
                )
            # --- Execute tasks for the current inner fold in parallel ---
            if tasks:
                inner_results_list = Parallel(n_jobs=n_jobs, verbose=1)(tasks)

                # Filter out None results (from folds with unseen classes)
                valid_results = [res for res in inner_results_list if res is not None]

                # Flatten the results if needed (depends on eval_type)
                for res_item in valid_results:
                    if isinstance(res_item, list):  # OvR or OvO might return lists
                        all_results.extend(res_item)
                    elif isinstance(res_item, dict):  # Standard eval returns dict
                        all_results.append(res_item)
                    else:
                        print(
                            f"Warning: Unexpected result type encountered: {type(res_item)}"
                        )

            else:
                print(
                    f"  No evaluation tasks generated for outer fold '{test_study_name}'."
                )

            # --- End Hyperparameter Loop ---
        print(f"  Finished evaluations for outer fold '{test_study_name}'.")
        # --- End Inner Loop ---

    df_parallel_results_study_as_fold = pd.DataFrame(all_results)
    return df_parallel_results_study_as_fold

def run_inner_cv_loso_single_param(
    X,
    y,
    study_labels,
    model,
    single_param,
    pipe,
    multi_type="standard",
    model_type="any"
):
    """
    Modified version of run_inner_cv_loso that processes only one hyperparameter combination.
    Used for SLURM array jobs where each job handles one parameter set.
    """
    # Single parameter instead of list
    n_genes_list = [single_param["n_genes"]]
    all_results = []
    
    studies_as_folds = np.unique(study_labels)
    
    for test_study_name in studies_as_folds:
        print(f"\n--- Outer Loop: Holding out Study '{test_study_name}' for Testing ---")

        # Create masks for outer split
        test_mask = study_labels == test_study_name
        train_mask = ~test_mask

        # Outer training set (N-1 studies)
        X_train_outer = X[train_mask]
        y_train_outer = y[train_mask]
        study_labels_outer = study_labels[train_mask]

        # Get the unique studies present in the outer training set
        train_studies = np.unique(study_labels_outer)
        print(f"Outer training set contains studies: {train_studies.tolist()}")

        # Inner Loop: Iterate through each study in the outer training set to be used as VALIDATION set
        for validation_study_name in train_studies:
            print(f"  Inner Loop: Validating on Study '{validation_study_name}'")
            
            # Create masks for inner split (relative to outer training data)
            val_inner_mask = study_labels_outer == validation_study_name
            train_inner_mask = ~val_inner_mask

            # Inner training set (N-2 studies)
            X_train_inner = X_train_outer[train_inner_mask]
            y_train_inner = y_train_outer[train_inner_mask]
            study_labels_inner = study_labels_outer[train_inner_mask]

            # Inner validation set (1 study)
            X_val_inner = X_train_outer[val_inner_mask]
            y_val_inner = y_train_outer[val_inner_mask]

            # Get the original indices of validation samples in the full dataset
            train_indices = np.where(train_mask)[0]
            original_val_inner_idx = train_indices[val_inner_mask]
            
            # Pre-process Data ONCE for this inner fold
            processed_X_inner = pre_process_data_loso(
                n_genes_list,
                X_train_inner,
                X_val_inner,
                study_labels_inner,
                y_train_inner,
                pipe
            )

            # Process single hyperparameter combination (no parallel processing needed)
            result = evaluate_inner_fold(
                test_study_name,  # Identifier for the outer fold (held-out test study)
                validation_study_name,  # Identifier for the inner fold (validation study)
                processed_X_inner,  # Pre-calculated processed data
                y_train_inner,  # Inner training labels
                y_val_inner,  # Inner validation labels
                original_val_inner_idx,  # Inner indices based on whole data
                model,  # Classifier class
                single_param.copy(),  # Current hyperparameter combination (copy to avoid modification)
                multi_type=multi_type,
                model_type=model_type
            )

            # Handle result
            if result is not None:
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, dict):
                    all_results.append(result)
            else:
                print(f"  No valid results for outer fold '{test_study_name}', inner fold '{validation_study_name}'")

        print(f"  Finished evaluations for outer fold '{test_study_name}'.")

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    return df_results

def run_outer_cv_loso(
    X,
    y,
    study_labels,
    model,
    pipe,
    best_params,
    multi_type="standard",
    model_type = "any",
    fs_method="unknown",
    cache_dir=None,
    knn_n_genes=KNN_REJECTION_FEATURE_N_GENES,
):
    # Import ast module for parsing string representations
    import ast
    
    # Import best parameters if a string path is provided
    if isinstance(best_params, str):
        best_params = pd.read_csv(best_params)
    
    # Extract n_genes from params column
    n_genes_list = []
    for params in best_params['params']:
        try:
            if isinstance(params, dict):
                n_genes_list.append(params['n_genes'])
            else:
                # Assume it's a string representation of a dictionary
                parsed_params = ast.literal_eval(params)
                n_genes_list.append(parsed_params['n_genes'])
        except (ValueError, SyntaxError, KeyError) as e:
            print(f"Error parsing params: {params}")
            print(f"Error details: {e}")
            continue

    # Ensure we have valid n_genes values
    if not n_genes_list:
        raise ValueError("No valid n_genes values found in best_params")
    
    # Remove duplicates and sort. Ensure KNN reference space is available.
    n_genes_list = sorted(list(set(n_genes_list + [int(knn_n_genes)])))

    # Empty list to append results to
    all_results = []
    studies_as_folds = np.unique(study_labels)
    # Iterate through each study as the test fold
    for test_study_name in studies_as_folds:
        print(f"\n--- Outer Loop: Holding out Study '{test_study_name}' for Testing ---")

        # Create masks for outer split
        test_mask = study_labels == test_study_name
        train_mask = ~test_mask

        # Split data into outer training and test sets
        X_train = X[train_mask]
        y_train = y[train_mask]
        study_labels_train = study_labels[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        test_idx = np.where(test_mask)[0]

        # Once per outer fold, data is preprocessed
        processed_X = pre_process_data_loso(
            n_genes_list,
            X_train,
            X_test,
            study_labels_train,
            y_train,
            pipe,
            cache_dir=cache_dir,
            split_tag=f"outer_loso_{test_study_name}",
            fs_method=fs_method,
        )

        knn_features = None
        if int(knn_n_genes) in processed_X:
            X_train_knn, X_test_knn = processed_X[int(knn_n_genes)]
            knn_cache_dir = cache_dir or _get_reject_cache_dir()
            knn_key_payload = json.dumps(
                {
                    "mode": "loso",
                    "outer_fold": str(test_study_name),
                    "fs_method": str(fs_method),
                    "n_genes": int(knn_n_genes),
                    "train_hash": _stable_hash_array(np.where(train_mask)[0]),
                    "test_hash": _stable_hash_array(test_idx),
                    "k_values": list(KNN_REJECTION_K_VALUES),
                },
                sort_keys=True,
            ).encode("utf-8")
            knn_key = hashlib.sha256(knn_key_payload).hexdigest()
            knn_features = _load_or_compute_knn_features(
                knn_cache_dir,
                knn_key,
                X_train_knn,
                X_test_knn,
            )
        
        # Filter best_params to get all rows for current outer fold (test study)
        best_params_fold = best_params[best_params['outer_fold'] == test_study_name]
        
        outer_results = evaluate_outer_fold(
            test_study_name,  # Use study name as fold identifier
            processed_X,
            y_train,
            y_test,
            test_idx,
            model,
            best_params_fold,
            multi_type=multi_type,  # standard, OvR, OvO
            model_type = model_type,
            knn_features=knn_features,
        )

        # Flatten results list if needed and append to all_results
        if outer_results is None:
            print(f"Warning: No valid results for outer fold {test_study_name}")
            continue
        elif isinstance(outer_results, dict):
            # Single dictionary result
            all_results.append(outer_results)
        elif isinstance(outer_results, list):
            # List of dictionaries
            all_results.extend(outer_results)
        else:
            raise ValueError("Unexpected structure in outer_results")

    # Convert to DataFrame
    df_parallel_results = pd.DataFrame(all_results)
    return df_parallel_results

###################################################################################
# Left-out sample prediction functions for outer CV                               #
###################################################################################


def predict_leftout_for_fold(
    outer_fold,
    fitted_pipelines,
    processed_X,
    y_train,
    X_leftout_raw,
    y_leftout,
    leftout_indices,
    model,
    best_params_fold,
    multi_type="standard",
    model_type="any"
):
    """
    Predict on left-out samples for a single outer fold.

    Re-fits the model on the already-preprocessed training data (fast),
    then transforms and predicts on left-out samples using the fitted
    pipeline from that fold.
    """

    def _standard_leftout():
        params = best_params_fold.iloc[0]["params"]
        params = ast.literal_eval(params) if isinstance(params, str) else params
        n_genes = params.pop("n_genes")

        X_train = processed_X[n_genes][0]
        X_leftout_proc = fitted_pipelines[n_genes].transform(
            X_leftout_raw
        ).astype(np.float32)

        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)
        y_train_encoded = label_encoder.transform(y_train)

        clf = model(**params)
        params["n_genes"] = n_genes

        if model_type == "NN":
            best_epoch = params.get("best_epoch", None)
            if best_epoch is None:
                raise ValueError("Best Epochs for NN is not available.")
            clf.fit(X_train, y_train_encoded, epochs=best_epoch)
        else:
            clf.fit(X_train, y_train_encoded)

        preds_prob = clf.predict_proba(X_leftout_proc)
        preds_encoded = np.argmax(preds_prob, axis=1)
        preds = label_encoder.inverse_transform(preds_encoded)
        classes = label_encoder.classes_

        preds_prob = preds_prob.flatten()
        preds_prob = np.round(preds_prob, 4)
        preds_prob = preds_prob.tolist()

        return {
            "outer_fold": outer_fold,
            "classes": classes,
            "params": params,
            "accuracy": 0,
            "f1_macro": 0,
            "mcc": 0,
            "kappa": 0,
            "y_val": y_leftout,
            "preds": preds,
            "preds_prob": json.dumps(preds_prob),
            "sample_indices": leftout_indices,
        }

    def _ovr_leftout():
        results = []
        classes = np.unique(y_train)

        for cl in classes:
            class_params = best_params_fold[best_params_fold["class"] == cl]
            if len(class_params) == 0:
                print(f"Skipping class {cl} for OvR leftout - no best parameters")
                continue

            params = class_params.iloc[0]["params"]
            params = ast.literal_eval(params) if isinstance(params, str) else params
            n_genes = params.pop("n_genes")

            X_train = processed_X[n_genes][0]
            X_leftout_proc = fitted_pipelines[n_genes].transform(
                X_leftout_raw
            ).astype(np.float32)

            clf = model(**params)
            params["n_genes"] = n_genes

            y_train_bin = np.array(
                [1 if yy == cl else 0 for yy in y_train], dtype=np.int32
            )
            # Left-out true class is never a training class, so always 0
            y_leftout_bin = np.zeros(len(y_leftout), dtype=np.int32)

            clf.fit(X_train, y_train_bin)
            preds_prob = clf.predict_proba(X_leftout_proc)
            pos_class_index = list(clf.classes_).index(1)
            preds_prob = preds_prob[:, pos_class_index]
            preds = (preds_prob >= 0.5).astype(int)

            preds_prob = np.round(preds_prob, 4)
            preds_prob = preds_prob.tolist()

            results.append({
                "outer_fold": outer_fold,
                "class": cl,
                "params": params,
                "accuracy": 0,
                "f1_binary": 0,
                "mcc": 0,
                "kappa": 0,
                "y_val": y_leftout_bin,
                "preds": preds,
                "preds_prob": json.dumps(preds_prob),
                "sample_indices": leftout_indices,
            })
        return results

    dispatch = {"standard": _standard_leftout, "OvR": _ovr_leftout}
    if multi_type not in dispatch:
        raise ValueError(f"Unsupported multi_type for leftout: {multi_type}")

    return dispatch[multi_type]()


def run_outer_cv_leftout(
    X, y, study_labels,
    X_leftout, y_leftout, leftout_global_idx, leftout_fold_assignments,
    model, pipe, best_params,
    multi_type="standard", model_type="any"
):
    """
    Predict on left-out samples for each outer CV fold.

    Mirrors run_outer_cv() fold structure (same StratifiedKFold seed) so
    each fold's model is identical. For each fold, transforms and predicts
    on the left-out samples assigned to that fold.
    """
    if isinstance(best_params, str):
        best_params = pd.read_csv(best_params)

    n_genes_list = extract_n_genes_list(best_params)
    # Mirror run_outer_cv() fold assignments exactly for left-out inference.
    outer_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=CV_RANDOM_STATE
    )
    combined = build_hybrid_stratify_labels(y, study_labels, 5)

    all_leftout_results = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, combined)):
        print(f"Leftout predictions for outer fold {outer_fold}")

        fold_mask = leftout_fold_assignments == outer_fold
        if not np.any(fold_mask):
            print(f"  No left-out samples assigned to fold {outer_fold}, skipping")
            continue

        # Preprocess training data and get fitted pipelines
        processed_X, y_train, y_test, fitted_pipelines = pre_process_data(
            n_genes_list, X, y, train_idx, test_idx, study_labels, pipe,
            return_pipelines=True,
        )

        X_leftout_fold = X_leftout[fold_mask]
        y_leftout_fold = y_leftout[fold_mask]
        leftout_idx_fold = leftout_global_idx[fold_mask]

        best_params_fold = best_params[best_params["outer_fold"] == outer_fold]

        print(f"  Predicting on {len(y_leftout_fold)} left-out samples")
        leftout_results = predict_leftout_for_fold(
            outer_fold, fitted_pipelines, processed_X, y_train,
            X_leftout_fold, y_leftout_fold, leftout_idx_fold,
            model, best_params_fold,
            multi_type=multi_type, model_type=model_type,
        )

        if isinstance(leftout_results, dict):
            all_leftout_results.append(leftout_results)
        elif isinstance(leftout_results, list):
            all_leftout_results.extend(leftout_results)

    return pd.DataFrame(all_leftout_results)


def run_outer_cv_loso_leftout(
    X, y, study_labels,
    X_leftout, y_leftout, study_leftout, leftout_global_idx,
    model, pipe, best_params,
    multi_type="standard", model_type="any"
):
    """
    Predict on left-out samples for each LOSO outer fold.

    Left-out samples are assigned to the fold of their study. Mirrors
    run_outer_cv_loso() fold structure so each fold's model is identical.
    """
    if isinstance(best_params, str):
        best_params = pd.read_csv(best_params)

    n_genes_list = extract_n_genes_list(best_params)
    studies_as_folds = np.unique(study_labels)

    all_leftout_results = []

    for test_study_name in studies_as_folds:
        print(f"Leftout predictions for LOSO fold '{test_study_name}'")

        # Left-out samples belonging to this study
        leftout_study_mask = study_leftout == test_study_name
        if not np.any(leftout_study_mask):
            print(f"  No left-out samples in study '{test_study_name}', skipping")
            continue

        # Same split as normal LOSO
        test_mask = study_labels == test_study_name
        train_mask = ~test_mask

        X_train = X[train_mask]
        y_train = y[train_mask]
        study_labels_train = study_labels[train_mask]

        # Preprocess and get fitted pipelines
        processed_X, fitted_pipelines = pre_process_data_loso(
            n_genes_list, X_train, X[test_mask],
            study_labels_train, y_train, pipe,
            return_pipelines=True,
        )

        X_leftout_fold = X_leftout[leftout_study_mask]
        y_leftout_fold = y_leftout[leftout_study_mask]
        leftout_idx_fold = leftout_global_idx[leftout_study_mask]

        best_params_fold = best_params[best_params["outer_fold"] == test_study_name]

        print(f"  Predicting on {len(y_leftout_fold)} left-out samples")
        leftout_results = predict_leftout_for_fold(
            test_study_name, fitted_pipelines, processed_X, y_train,
            X_leftout_fold, y_leftout_fold, leftout_idx_fold,
            model, best_params_fold,
            multi_type=multi_type, model_type=model_type,
        )

        if isinstance(leftout_results, dict):
            all_leftout_results.append(leftout_results)
        elif isinstance(leftout_results, list):
            all_leftout_results.extend(leftout_results)

    return pd.DataFrame(all_leftout_results)


# Sentinel outer_fold for full-data final models (not a CV fold index).
FINAL_LEFTOUT_OUTER_FOLD = -1


def predict_leftout_final(
    trained_models,
    X_leftout,
    y_leftout,
    leftout_global_idx,
    multi_type="standard",
    knn_features=None,
):
    """
    Score left-out samples with final trained models (fit on all included data).

    Output shape matches predict_leftout_for_fold / outer CV left-out CSVs;
    outer_fold is FINAL_LEFTOUT_OUTER_FOLD (-1) to distinguish from per-fold CV.
    """
    if len(X_leftout) == 0:
        return pd.DataFrame()

    y_leftout = np.asarray(y_leftout)
    leftout_global_idx = np.asarray(leftout_global_idx)
    fold_tag = FINAL_LEFTOUT_OUTER_FOLD

    if multi_type == "standard":
        if not trained_models:
            raise ValueError("No trained models for left-out prediction.")
        model_info, clf = trained_models[0]
        pipe = model_info["preprocessing_pipeline"]
        label_encoder = model_info["label_encoder"]
        n_genes = model_info["n_genes"]
        out_params = dict(model_info["params"])
        out_params["n_genes"] = n_genes

        X_leftout_proc = pipe.transform(X_leftout).astype(np.float32)
        preds_prob = clf.predict_proba(X_leftout_proc)
        preds_encoded = np.argmax(preds_prob, axis=1)
        preds = label_encoder.inverse_transform(preds_encoded)
        classes = label_encoder.classes_
        preds_prob_flat = np.round(preds_prob.flatten(), 4).tolist()

        return pd.DataFrame(
            [
                _append_knn_columns({
                    "outer_fold": fold_tag,
                    "classes": list(classes),
                    "params": out_params,
                    "accuracy": 0,
                    "f1_macro": 0,
                    "mcc": 0,
                    "kappa": 0,
                    "y_val": y_leftout,
                    "preds": preds,
                    "preds_prob": json.dumps(preds_prob_flat),
                    "sample_indices": leftout_global_idx,
                }, knn_features)
            ]
        )

    if multi_type == "OvR":
        rows = []
        for model_info, clf in trained_models:
            class_val = model_info["class"]
            n_genes = model_info["n_genes"]
            out_params = dict(model_info["params"])
            out_params["n_genes"] = n_genes
            pipe = model_info["preprocessing_pipeline"]

            X_leftout_proc = pipe.transform(X_leftout).astype(np.float32)
            y_leftout_bin = np.zeros(len(y_leftout), dtype=np.int32)

            preds_prob = clf.predict_proba(X_leftout_proc)
            pos_class_index = list(clf.classes_).index(1)
            preds_prob_pos = preds_prob[:, pos_class_index]
            preds = (preds_prob_pos >= 0.5).astype(int)
            preds_prob_list = np.round(preds_prob_pos, 4).tolist()

            rows.append(
                _append_knn_columns({
                    "outer_fold": fold_tag,
                    "class": class_val,
                    "params": out_params,
                    "accuracy": 0,
                    "f1_binary": 0,
                    "mcc": 0,
                    "kappa": 0,
                    "y_val": y_leftout_bin,
                    "preds": preds,
                    "preds_prob": json.dumps(preds_prob_list),
                    "sample_indices": leftout_global_idx,
                }, knn_features)
            )
        return pd.DataFrame(rows)

    raise ValueError(
        f"Left-out prediction for final train not implemented for multi_type={multi_type}"
    )


###################################################################################
# Pipeline management functions                                                   #
###################################################################################

def get_or_create_pipeline(X, y, study_labels, pipe, n_genes, pipelines_dir):
    """
    Get an existing fitted pipeline or create and save a new one.
    Pipeline caching is based only on n_genes since preprocessing is identical
    across different models, fold types, and multiclass strategies.
    
    Args:
        X: Feature matrix
        study_labels: Study labels for each sample
        pipe: Base pipeline to clone and fit
        n_genes: Number of genes for feature selection
        pipelines_dir: Directory to save/load pipelines
        
    Returns:
        Fitted pipeline and processed data
    """
    import os
    import joblib
    from pathlib import Path
    
    # Create pipelines directory if it doesn't exist
    os.makedirs(pipelines_dir, exist_ok=True)
    
    # Create pipeline filename based only on n_genes
    pipeline_filename = f"pipeline_ngenes_{n_genes}.pkl"
    pipeline_path = os.path.join(pipelines_dir, pipeline_filename)
    
    # Check if pipeline already exists
    if os.path.exists(pipeline_path):
        print(f"  Loading existing pipeline for n_genes={n_genes}: {pipeline_path}")
        fitted_pipeline = joblib.load(pipeline_path)
        X_processed = fitted_pipeline.transform(X).astype(np.float32)
    else:
        print(f"  Creating new pipeline for n_genes={n_genes}: {pipeline_path}")
        # Clone and fit new pipeline
        fitted_pipeline = clone(pipe)
        X_processed = fitted_pipeline.fit_transform(
            X,
            y,
            feature_selection__study_per_patient=study_labels,
            feature_selection__n_genes=n_genes,
        ).astype(np.float32)
        
        # Save the fitted pipeline
        joblib.dump(fitted_pipeline, pipeline_path)
        print(f"  Saved pipeline: {pipeline_path}")
    
    return fitted_pipeline, X_processed

def get_pipeline_cache_dir():
    """
    Get the directory for caching fitted pipelines.
    Pipelines are cached by n_genes only, making them reusable across
    different models, fold types, and multiclass strategies.
    
    Returns:
        Path to pipeline cache directory (data/out/final_models/pipelines)
    """
    from pathlib import Path
    # Get the project root directory (parent of python directory)
    project_root = Path(__file__).resolve().parent.parent
    return os.path.join(project_root, "data", "out", "final_models", "pipelines")

###################################################################################
# Final model training functions                                                  #
###################################################################################

def train_final_model_standard(X, y, study_labels, model, pipe, best_params, model_type="any", pipelines_dir=None):
    """
    Train a final model using standard multiclass classification.
    
    Args:
        X: Feature matrix
        y: Target labels
        study_labels: Study labels for each sample
        model: Model class to use
        pipe: Preprocessing pipeline
        best_params: Best parameters from train/test analysis
        model_type: Type of model being trained
        pipelines_dir: Directory for caching fitted pipelines (optional)
        
    Returns:
        List of tuples (model_info, trained_model)
    """
    print("Training final standard multiclass model...")
    
    # Final deployment is strict: standard multiclass must have exactly one param row.
    if len(best_params) != 1:
        raise ValueError(
            "Expected exactly 1 best-parameter row for standard final training, "
            f"got {len(best_params)}."
        )
    
    params = best_params.iloc[0]["params"]
    params = ast.literal_eval(params) if isinstance(params, str) else params
    
    # Extract n_genes and prepare data
    n_genes = params.pop("n_genes")
    
    # Use cached pipeline if available
    if pipelines_dir is not None:
        print(f"  Using pipeline cache directory: {pipelines_dir}")
        pipe_clone, X_processed = get_or_create_pipeline(
            X, y, study_labels, pipe, n_genes, pipelines_dir
        )
    else:
        # Fallback to original method if no cache directory provided
        print(f"  Fitting new pipeline for n_genes={n_genes}")
        pipe_clone = clone(pipe)
        X_processed = pipe_clone.fit_transform(
            X,
            y,
            feature_selection__study_per_patient=study_labels,
            feature_selection__n_genes=n_genes,
        ).astype(np.float32)
    
    # Create label encoder to map labels to consecutive integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Create and train the model
    clf = model(**params)
    
    if model_type == "NN":
        # Train on full data using best_epoch from inner CV (no validation split; final model uses all data)
        best_epoch = params.get("best_epoch", None)
        if best_epoch is None:
            print("Warning: No best_epoch found in parameters. Training with default epochs.")
            clf.fit(X_processed, y_encoded)
        else:
            clf.fit(X_processed, y_encoded, epochs=best_epoch)
    else:
        clf.fit(X_processed, y_encoded)
    
    # Store model info
    model_info = {
        "params": params,
        "n_genes": n_genes,
        "classes": label_encoder.classes_.tolist(),
        "preprocessing_pipeline": pipe_clone,
        "label_encoder": label_encoder
    }
    
    return [(model_info, clf)]

def train_final_model_ovr(X, y, study_labels, model, pipe, best_params, model_type="any", pipelines_dir=None):
    """
    Train final models using One-vs-Rest classification.
    
    Args:
        X: Feature matrix
        y: Target labels  
        study_labels: Study labels for each sample
        model: Model class to use
        pipe: Preprocessing pipeline
        best_params: Best parameters from train/test analysis
        model_type: Type of model being trained
        pipelines_dir: Directory for caching fitted pipelines (optional)
        
    Returns:
        List of tuples (model_info, trained_model)
    """
    print("Training final One-vs-Rest models...")
    
    trained_models = []
    classes = np.unique(y)
    
    # Get unique n_genes values to prepare pipelines
    n_genes_values = set()
    for class_val in classes:
        class_params = best_params[best_params["class"] == class_val]
        if len(class_params) > 0:
            params = class_params.iloc[0]["params"]
            params = ast.literal_eval(params) if isinstance(params, str) else params
            n_genes_values.add(params["n_genes"])
    
    # Prepare pipelines for all unique n_genes values
    pipelines_cache = {}
    processed_data_cache = {}
    
    if pipelines_dir is not None:
        print(f"  Using pipeline cache directory: {pipelines_dir}")
        for n_genes in n_genes_values:
            fitted_pipeline, X_processed = get_or_create_pipeline(
                X, y, study_labels, pipe, n_genes, pipelines_dir
            )
            pipelines_cache[n_genes] = fitted_pipeline
            processed_data_cache[n_genes] = X_processed
    
    for class_val in classes:
        print(f"  Training model for class {class_val}...")
        
        # Find best parameters for this class
        class_params = best_params[best_params["class"] == class_val]
        if len(class_params) == 0:
            raise ValueError(
                f"Missing best parameters for OvR class '{class_val}'. "
                "Final deployment requires one trained model per class."
            )
        
        params = class_params.iloc[0]["params"]
        params = ast.literal_eval(params) if isinstance(params, str) else params
        
        # Extract n_genes and prepare data
        n_genes = params.pop("n_genes")
        
        # Use cached pipeline and processed data if available
        if pipelines_dir is not None and n_genes in processed_data_cache:
            print(f"    Using cached pipeline for n_genes={n_genes}")
            fitted_pipeline = pipelines_cache[n_genes]
            X_processed = processed_data_cache[n_genes]
        else:
            # Fallback to original method if no cache directory provided
            print(f"    Fitting new pipeline for n_genes={n_genes}")
            fitted_pipeline = clone(pipe)
            X_processed = fitted_pipeline.fit_transform(
                X,
                feature_selection__study_per_patient=study_labels,
                feature_selection__n_genes=n_genes,
            ).astype(np.float32)
        
        # Create binary labels (1 for target class, 0 for others)
        y_binary = (y == class_val).astype(np.int32)
        
        # Create and train the model
        clf = model(**params)
        clf.fit(X_processed, y_binary)
        
        # Store model info
        model_info = {
            "class": class_val,
            "params": params,
            "n_genes": n_genes,
            "preprocessing_pipeline": fitted_pipeline
        }
        
        trained_models.append((model_info, clf))
    
    return trained_models

def train_final_models(X, y, study_labels, model, pipe, best_params, multi_type="standard", model_type="any", pipelines_dir=None):
    """
    Train final models based on the multiclass strategy.
    
    Args:
        X: Feature matrix
        y: Target labels
        study_labels: Study labels for each sample
        model: Model class to use
        pipe: Preprocessing pipeline
        best_params: Best parameters from train/test analysis
        multi_type: Multiclass strategy ("standard", "OvR", "OvO")
        model_type: Type of model being trained
        pipelines_dir: Directory for caching fitted pipelines (optional)
        
    Returns:
        List of tuples (model_info, trained_model)
    """
    # Dispatch table for training functions
    if multi_type == "standard":
        return train_final_model_standard(X, y, study_labels, model, pipe, best_params, model_type, pipelines_dir)
    elif multi_type == "OvR":
        return train_final_model_ovr(X, y, study_labels, model, pipe, best_params, model_type, pipelines_dir)
    else:
        raise ValueError(f"Unsupported multiclass strategy: {multi_type}")

# old
"""
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'kappa': make_scorer(cohen_kappa_score),
    'mcc': make_scorer(matthews_corrcoef)
}

def run_inner_cv_scikeras(X, y, study_per_patient, pipeline, param_grid,k = 2, n_jobs = 1, inner_state = 1):
    # ---------------------------------------------------------------------------
    # SET UP CROSS-VALIDATION STRATEGIES
    # ---------------------------------------------------------------------------
    
    inner_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=inner_state)
    outer_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=inner_state)

    # Container for inner cross-validation results.
    inner_predictions = {}

    # ---------------------------------------------------------------------------
    # OUTER CROSS-VALIDATION LOOP
    # ---------------------------------------------------------------------------
    for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
        # Split data into outer training and test sets.
        X_train, y_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        
        # Reindex study metadata to match training indices.
        study_train = study_per_patient.reindex(X_train.index)
        pipeline.set_params(feature_selection__study_per_patient=study_train)
        
        # -----------------------------------------------------------------------
        # INNER CROSS-VALIDATION FOR EACH PARAMETER COMBINATION
        # -----------------------------------------------------------------------
        for params in ParameterGrid(param_grid):
            pipeline.set_params(**params)
            
            #predicted_classes = cross_val_predict(
            #    pipeline, X_train, y_train,
            #    cv=inner_cv, method='predict', n_jobs=n_jobs
            #)
            
            cv_results = cross_validate(
                pipeline, 
                X_train, 
                y_train, 
                cv=inner_cv, 
                scoring=scoring, 
                return_estimator=True, 
                n_jobs=n_jobs
            )
            
            epoch_counts = []

            for estimator in cv_results['estimator']:
                n_epochs = len(estimator._final_estimator.history_["val_loss"])
                epoch_counts.append(n_epochs)

            mean_epochs = np.mean(epoch_counts)
            mean_kappa = np.mean(cv_results['test_kappa'])
            mean_mcc = np.mean(cv_results['test_mcc'])
            mean_accuracy = np.mean(cv_results['test_accuracy'])
            
            # Compile the results for this parameter configuration.
            results_dict = {
                'params': params,
                'indices_inner_fold': outer_train_idx,
                'true_class': y_train,
                'mean_epochs': mean_epochs,
                'kappa': mean_kappa,
                'mcc': mean_mcc,
                'accuracy_score': mean_accuracy,
                'estimators': cv_results['estimator']
            }
            
            key = tuple(sorted(params.items()))
            if key not in inner_predictions:
                inner_predictions[key] = []
            inner_predictions[key].append(results_dict)
    return inner_predictions

def run_inner_cv(X, y, study_per_patient, pipeline, param_grid,k = 2, n_jobs = 1, inner_state = 1):
    # ---------------------------------------------------------------------------
    # SET UP CROSS-VALIDATION STRATEGIES
    # ---------------------------------------------------------------------------
    
    inner_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=inner_state)
    outer_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=inner_state)

    # Container for inner cross-validation results.
    inner_predictions = {}

    # ---------------------------------------------------------------------------
    # OUTER CROSS-VALIDATION LOOP
    # ---------------------------------------------------------------------------
    for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
        # Split data into outer training and test sets.
        X_train, y_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        
        # Reindex study metadata to match training indices.
        study_train = study_per_patient.reindex(X_train.index)
        pipeline.set_params(feature_selection__study_per_patient=study_train)
        
        # -----------------------------------------------------------------------
        # INNER CROSS-VALIDATION FOR EACH PARAMETER COMBINATION
        # -----------------------------------------------------------------------
        for params in ParameterGrid(param_grid):
            pipeline.set_params(**params)
            
            predicted_classes = cross_val_predict(
                pipeline, X_train, y_train,
                cv=inner_cv, method='predict', n_jobs=n_jobs
            )
            
            # Compute inner CV probability predictions.
            #inner_proba_preds = cross_val_predict(
            #    pipeline, X_train, y_train,
            #    cv=inner_cv, method='predict_proba', n_jobs=n_jobs
            #)
            # Create a DataFrame for probability predictions with proper class names.
            #inner_preds_df = pd.DataFrame(inner_proba_preds, columns=class_order, index=X_train.index)
            
            # Determine predicted classes by selecting the class with maximum probability.
            #predicted_classes = inner_preds_df.idxmax(axis=1)
            
            
            # Compile the results for this parameter configuration.
            results_dict = {
                'params': params,
                'indices_inner_fold': outer_train_idx,
                #'inner_preds_proba': inner_preds_df,
                'predicted_class': predicted_classes,
                'true_class': y_train,
                'kappa': cohen_kappa_score(y_train, predicted_classes),
                'mcc': matthews_corrcoef(y_train, predicted_classes),
                'accuracy_score': accuracy_score(y_train, predicted_classes)
            }
            
            key = tuple(sorted(params.items()))
            if key not in inner_predictions:
                inner_predictions[key] = []
            inner_predictions[key].append(results_dict)
    return inner_predictions

def cv_to_extracted_dict(inner_predictions):
    fold_data = []
    for result in inner_predictions:
        proba = result['inner_preds_proba'].values  # shape: (n_samples, n_classes)
        class_labels = result['inner_preds_proba'].columns.values
        true_labels = result['true_class'].values
        raw_preds = result['predicted_class'].values
        
        # Precompute maximum probabilities and indices for each sample.
        max_probs = proba.max(axis=1)
        max_indices = proba.argmax(axis=1)
        
        fold_data.append({
            'max_probs': max_probs,
            'max_indices': max_indices,
            'class_labels': class_labels,
            'true_labels': true_labels,
            'raw_preds': raw_preds  
        })
    return fold_data

"""
