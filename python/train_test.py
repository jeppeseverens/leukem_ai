import pandas as pd
import numpy as np
import os
from datetime import datetime

from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone

from joblib import Parallel, delayed
import itertools
import polars as pl

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
    meta_file = next(f for f in csv_files if f.startswith("meta"))
    counts_file = next(f for f in csv_files if f.startswith("counts"))
    rgas_file = next(f for f in csv_files if f.startswith("rgas"))

    # Construct full paths
    meta_path = os.path.join(directory, meta_file)
    counts_path = os.path.join(directory, counts_file)
    rgas_path = os.path.join(directory, rgas_file)

    # Load CSV data into pandas DataFrames/Series.
    X_df = pl.read_csv(counts_path).to_pandas()
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
    X = X_df.transpose().values
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

    valid_classes = [c for c in valid_classes if c != "AML NOS" and c != "Missing data"]

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


def evaluate_inner_fold(
    outer_fold,
    inner_fold,
    processed_X,
    y_train_inner,
    y_val_inner,
    model,
    params,
    multi_type="standard",
    model_type="any"
):

    def standard_eval():
        le = LabelEncoder()
        y_train_inner_enc = le.fit_transform(y_train_inner)
        
        if model_type == "NN":
            y_val_inner_enc = le.transform(y_val_inner)
            clf.fit(X_train_inner, y_train_inner_enc, validation_data =(X_val_inner, y_val_inner_enc))
        else:
            clf.fit(X_train_inner, y_train_inner_enc)
        preds_prob = clf.predict_proba(X_val_inner)
        preds = np.argmax(preds_prob, axis=1)
        preds = le.inverse_transform(preds)
        y_val_inner_enc = le.inverse_transform(y_val_inner)
        
        if model_type == "NN":
            history = clf.model.history.history
            best_epoch = np.argmin(history['val_loss']) + 1  # Add 1 to match epoch count
            params["best_epoch"] = best_epoch
            
        return {
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "params": params,
            "accuracy": accuracy_score(y_val_inner, preds),
            "f1_macro": f1_score(y_val_inner, preds, average="macro"),
            "mcc": matthews_corrcoef(y_val_inner, preds),
            "kappa": cohen_kappa_score(y_val_inner, preds),
            "y_val": y_val_inner_enc,
            "preds": preds,
            "preds_prob": preds_prob
        }

    def ovr_eval():
        results = []
        classes = np.unique(y_train_inner)
        for cl in classes:
            y_train_bin = [1 if yy == cl else 0 for yy in y_train_inner]
            y_val_bin = [1 if yy == cl else 0 for yy in y_val_inner]

            y_train_bin = np.array(y_train_bin, dtype=np.int32)

            if len(np.unique(y_train_bin)) == 1:
                continue

            y_val_bin = np.array(y_val_bin, dtype=np.int32)

            clf.fit(X_train_inner, y_train_bin)
            preds_prob = clf.predict_proba(X_val_inner)
            preds_prob = preds_prob[:, 1]
            preds = (preds_prob >= 0.5).astype(int)

            results.append(
                {
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
                    "preds_prob": preds_prob
                }
            )
        return results

    def ovo_eval():
        results = []
        classes = np.unique(y_train_inner)
        for i, j in itertools.combinations(classes, 2):
            train_mask = [(yy == i or yy == j) for yy in y_train_inner]
            val_mask = [(yy == i or yy == j) for yy in y_val_inner]

            if sum(train_mask) == 0 or sum(val_mask) == 0:
                continue

            X_train_ij = X_train_inner[train_mask]
            y_train_ij = np.array(
                [yy for yy in y_train_inner if yy == i or yy == j], dtype=np.int32
            )
            y_train_ij = (y_train_ij == i).astype(np.int32)

            X_val_ij = X_val_inner[val_mask]
            y_val_ij = np.array(
                [yy for yy in y_val_inner if yy == i or yy == j], dtype=np.int32
            )
            y_val_ij = (y_val_ij == i).astype(np.int32)

            if len(np.unique(y_train_ij)) == 1 or len(np.unique(y_val_ij)) == 1:
                continue

            clf.fit(X_train_ij, y_train_ij)
            preds_prob = clf.predict_proba(X_val_ij)
            preds_prob = preds_prob[:, 1]
            preds = (preds_prob >= 0.5).astype(int)

            results.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "class_0": i,
                    "class_1": j,
                    "params": params,
                    "accuracy": accuracy_score(y_val_ij, preds),
                    "f1_binary": conditional_f1(y_val_ij, preds),
                    "mcc": matthews_corrcoef(y_val_ij, preds),
                    "kappa": cohen_kappa_score(y_val_ij, preds),
                    "y_val": y_val_ij,
                    "preds": preds,
                    "preds_prob": preds_prob
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

    return eval_dispatch[multi_type]()


###################################################################################
# Main functions for standard inner cross validation                              #
###################################################################################


def pre_process_data(
    n_genes_list,
    X_train_outer,
    y_train_outer,
    train_inner_idx,
    val_inner_idx,
    study_labels_outer,
    pipe,
):

    X_train_inner = X_train_outer[train_inner_idx]
    X_val_inner = X_train_outer[val_inner_idx]

    study_labels_inner = study_labels_outer[train_inner_idx]

    y_train_inner = y_train_outer[train_inner_idx]
    y_val_inner = y_train_outer[val_inner_idx]

    y_train_inner = np.array(y_train_inner, dtype=np.int32)
    y_val_inner = np.array(y_val_inner, dtype=np.int32)

    processed_X = {}
    for n_genes_i in n_genes_list:
        pipe_inner = clone(pipe)

        X_train_inner_proc = pipe_inner.fit_transform(
            X_train_inner,
            feature_selection__study_per_patient=study_labels_inner,
            feature_selection__n_genes=n_genes_i,
        )
        X_val_inner_proc = pipe_inner.transform(X_val_inner)

        processed_X[n_genes_i] = [X_train_inner_proc, X_val_inner_proc]
    return processed_X, y_train_inner, y_val_inner


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
    outer_cv = StratifiedKFold(n_splits=k_out, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=k_in, shuffle=True, random_state=42)

    param_combos = param_grid
    n_genes_list = sorted({params["n_genes"] for params in param_combos})
    print(n_genes_list)
    # Empty list to append results to
    all_results = []

    # Combine y and study labels so can be stratified on study and y
    combined = [str(a) + " " + str(b) for a, b in zip(y, study_labels)]

    # Make outer fold splits
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, combined)):
        print("outer_fold")
        print(outer_fold)
        X_train_outer = X[train_idx]
        y_train_outer = y[train_idx]
        study_labels_outer = study_labels[train_idx]

        combined_outer = [
            str(a) + " " + str(b) for a, b in zip(y_train_outer, study_labels_outer)
        ]

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

            # Then, for every hyperparameter combo performance is evaluated
            for params in param_combos:
                inner_tasks.append(
                    delayed(evaluate_inner_fold)(
                        outer_fold,
                        inner_fold,
                        processed_X,
                        y_train_inner,
                        y_val_inner,
                        model,
                        params,
                        multi_type=multi_type,  # standard, OvR, OvO
                        model_type = model_type
                    )
                )

            # Run inner CV tasks in parallel
            inner_results = Parallel(n_jobs=n_jobs, verbose=1)(inner_tasks)

            # Flatten inner_results list if needed and append to all_results
            if isinstance(inner_results[0], dict):
                # Flat list of dictionaries
                all_results.extend(inner_results)
            elif isinstance(inner_results[0], list):
                # List of lists of dictionaries
                for res in inner_results:
                    all_results.extend(res)
            else:
                raise ValueError("Unexpected structure in inner_results")

    # Convert to DataFrame
    df_parallel_results = pd.DataFrame(all_results)
    return df_parallel_results


###################################################################################
# Main function for leave one study out (loso) inner cross validation             #
###################################################################################


def pre_process_data_loso(
    n_genes_list,
    X_train_inner,
    X_val_inner,
    y_train_inner,
    y_val_inner,  # y values aren't strictly needed here but kept for consistency
    study_labels_inner,  # Labels corresponding to X_train_inner
    pipe
):
    """
    Preprocesses inner training and validation sets for different n_genes.
    Fits the pipeline ONLY on the inner training set.
    """
    processed_X = {}
    for n_genes_i in n_genes_list:
        # Clone the pipeline for this specific n_genes setting
        pipe_inner = clone(pipe)

        X_train_inner_proc = pipe_inner.fit_transform(
            X_train_inner,
            y_train_inner,  # Pass y if needed by steps
            feature_selection__study_per_patient=study_labels_inner,
            feature_selection__n_genes=n_genes_i,
        )

        # Transform the inner validation data (1 study) using the fitted pipeline
        X_val_inner_proc = pipe_inner.transform(X_val_inner)

        processed_X[n_genes_i] = [X_train_inner_proc, X_val_inner_proc]

    # Return the dictionary of processed data and the original inner y values
    return processed_X, y_train_inner, y_val_inner


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
    
    # Define the studies to use as folds
    studies_as_folds = [
        "BEATAML1.0-COHORT",
        "AAML0531",
        "AAML1031",
        "AAML03P1",
        "TCGA-LAML",
        "LEUCEGENE",
        "100LUMC",
    ]
    param_combos = param_grid

    all_results = []
    n_genes_list = sorted({params["n_genes"] for params in param_combos})
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

            # --- Pre-process Data ONCE for this inner fold ---
            # This computes processed versions for all n_genes values
            processed_X_inner, y_train_inner_proc, y_val_inner_proc = (
                pre_process_data_loso(
                    n_genes_list,
                    X_train_inner,
                    X_val_inner,
                    y_train_inner,
                    y_val_inner,
                    study_labels_inner,  # Pass inner training labels for pipeline fitting
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
                        y_train_inner_proc,  # Inner training labels
                        y_val_inner_proc,  # Inner validation labels
                        model,  # Classifier class
                        params,  # Current hyperparameter combination
                        multi_type=multi_type,  # Choose evaluation type: "standard", "OvR", "OvO"
                        model_type = model_type
                    )
                )
            # --- Execute tasks for the current inner fold in parallel ---
            if tasks:
                inner_results_list = Parallel(n_jobs=n_jobs, verbose=1)(tasks)

                # Flatten the results if needed (depends on eval_type)
                for res_item in inner_results_list:
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


# old
"""
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'kappa': make_scorer(cohen_kappa_score),
    'mcc': make_scorer(matthews_corrcoef)
}

def run_inner_cv_scikeras(X, y, study_per_patient, pipeline, param_grid,k = 2, n_jobs = 1, inner_state = 42):
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

def run_inner_cv(X, y, study_per_patient, pipeline, param_grid,k = 2, n_jobs = 1, inner_state = 42):
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
