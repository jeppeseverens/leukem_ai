# =============================================================================
# Outer Cross-Validation Analysis for Machine Learning Models
# =============================================================================
# This script analyzes outer cross-validation results for SVM, XGBoost, and
# Neural Network models, generates final prediction probability matrices,
# performs ensemble analysis using optimized weights from inner CV, and
# evaluates final model performance.
# =============================================================================

# =============================================================================
# Configuration and Constants
# =============================================================================

#' Find the most recent CSV file matching a pattern in a directory.
#' Returns NULL if no match is found.
find_latest_csv <- function(directory, pattern) {
  if (!dir.exists(directory)) return(NULL)
  files <- list.files(directory, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(NULL)
  files[order(file.mtime(files), decreasing = TRUE)][1]
}

# Output directories per model
OUTER_CV_DIRS <- list(
  svm     = "../data/out/outer_cv/SVM_n10_fs_eta",
  xgboost = "../data/out/outer_cv/XGBOOST_n10_fs_eta",
  nn      = "../data/out/outer_cv/NN_n10_fs_eta"
)

# Auto-discover the latest outer CV result files (standard + leftout)
OUTER_MODEL_CONFIGS <- list(
  svm = list(
    classification_type = "OvR",
    file_paths = list(
      cv   = find_latest_csv(OUTER_CV_DIRS$svm, "^SVM_outer_cv_CV_OvR_fs_eta_\\d+_\\d+(_with_knn)?\\.csv$"),
      loso = find_latest_csv(OUTER_CV_DIRS$svm, "^SVM_outer_cv_loso_OvR_fs_eta_\\d+_\\d+(_with_knn)?\\.csv$")
    )
  ),
  xgboost = list(
    classification_type = "OvR",
    file_paths = list(
      cv   = find_latest_csv(OUTER_CV_DIRS$xgboost, "^XGBOOST_outer_cv_CV_OvR_fs_eta_\\d+_\\d+(_with_knn)?\\.csv$"),
      loso = find_latest_csv(OUTER_CV_DIRS$xgboost, "^XGBOOST_outer_cv_loso_OvR_fs_eta_\\d+_\\d+(_with_knn)?\\.csv$")
    )
  ),
  neural_net = list(
    classification_type = "standard",
    file_paths = list(
      cv   = find_latest_csv(OUTER_CV_DIRS$nn, "^NN_outer_cv_CV_standard_fs_eta_\\d+_\\d+(_with_knn)?\\.csv$"),
      loso = find_latest_csv(OUTER_CV_DIRS$nn, "^NN_outer_cv_loso_standard_fs_eta_\\d+_\\d+(_with_knn)?\\.csv$")
    )
  )
)

# Auto-discover left-out prediction files (generated with --include_leftout)
LEFTOUT_MODEL_CONFIGS <- list(
  svm = list(
    classification_type = "OvR",
    file_paths = list(
      cv   = find_latest_csv(OUTER_CV_DIRS$svm, "^SVM_outer_cv_CV_OvR_leftout_fs_eta_\\d+_\\d+(_with_knn)?\\.csv$"),
      loso = find_latest_csv(OUTER_CV_DIRS$svm, "^SVM_outer_cv_loso_OvR_leftout_fs_eta_\\d+_\\d+(_with_knn)?\\.csv$")
    )
  ),
  xgboost = list(
    classification_type = "OvR",
    file_paths = list(
      cv   = find_latest_csv(OUTER_CV_DIRS$xgboost, "^XGBOOST_outer_cv_CV_OvR_leftout_fs_eta_\\d+_\\d+(_with_knn)?\\.csv$"),
      loso = find_latest_csv(OUTER_CV_DIRS$xgboost, "^XGBOOST_outer_cv_loso_OvR_leftout_fs_eta_\\d+_\\d+(_with_knn)?\\.csv$")
    )
  ),
  neural_net = list(
    classification_type = "standard",
    file_paths = list(
      cv   = find_latest_csv(OUTER_CV_DIRS$nn, "^NN_outer_cv_CV_standard_leftout_fs_eta_\\d+_\\d+(_with_knn)?\\.csv$"),
      loso = find_latest_csv(OUTER_CV_DIRS$nn, "^NN_outer_cv_loso_standard_leftout_fs_eta_\\d+_\\d+(_with_knn)?\\.csv$")
    )
  )
)

# Data filtering criteria (same as inner CV)
DATA_FILTERS <- list(
  min_samples_per_subtype = 10,
  excluded_subtypes = c("AML NOS", "Missing data", "Multi"),
  selected_studies = c(
    "TCGA-LAML",
    "LEUCEGENE",
    "BEATAML1.0-COHORT",
    "AAML0531",
    "AAML1031",
    "AAML03P1",
    "100LUMC"
  )
)

# Base directory for ensemble weights
WEIGHTS_BASE_DIR_UNMERGED <- "../data/out/inner_cv/ensemble_weights_unmerged_eta2/"
WEIGHTS_BASE_DIR_MERGED <- "../data/out/inner_cv/ensemble_weights_merged_summed_eta2/"

# Nested analysis (R/calibration_reject_models.R) reads outer_cv_results.rds ->
# multivariate_results$with_leftout_ood_aware$Global_Product_Optimized: augmented
# fold matrices with disagreement features (no confidence_multivariate).

# =============================================================================
# Source Utility Functions
# =============================================================================

source("utility_functions.R")

# =============================================================================
# Outer CV Specific Functions
# =============================================================================

#' Map filtered-dataset 0-based indices to full-dataset 0-based indices.
map_filtered_local_to_global_indices_outer <- function(sample_indices_zero_based, filtered_index_map_zero_based) {
  local_one_based <- sample_indices_zero_based + 1L
  if (length(local_one_based) == 0) return(integer(0))
  if (any(local_one_based < 1L | local_one_based > length(filtered_index_map_zero_based))) {
    stop(
      sprintf(
        "Found sample_indices outside filtered index range [0, %d]. Example values: %s",
        length(filtered_index_map_zero_based) - 1L,
        paste(head(sample_indices_zero_based, 15), collapse = ", ")
      )
    )
  }
  as.integer(filtered_index_map_zero_based[local_one_based])
}

KNN_DISTANCE_COLUMNS <- c(
  "knn10_mean_d", "knn10_min_d", "knn10_q90_d",
  "knn20_mean_d", "knn20_min_d", "knn20_q90_d"
)
REJECT_OPTION_EXTRA_FEATURE_COLUMNS <- c(
  "trust_ratio_knn10",
  "conformal_set_size_90"
)

# Not class-probability columns when scanning fold matrices (metadata + every calibration score).
# Keep in sync with augmented-matrix builders / utility_functions merge helpers.
PROB_MATRIX_META_COLUMNS <- c(
  "y", "inner_fold", "outer_fold", "indices", "study", "sample_indices",
  "confidence_multivariate",
  "confidence_id", "confidence_correct", "confidence_two_head",
  "confidence_seen_new_cohort", "confidence_unseen", "confidence_three_head",
  "confidence_two_head_postcal", "confidence_two_head_min_gate", "confidence_two_head_id_veto",
  "is_leftout", "n_models_agree",
  "top1_prob_variance_across_models",
  KNN_DISTANCE_COLUMNS,
  REJECT_OPTION_EXTRA_FEATURE_COLUMNS
)

add_optional_knn_columns <- function(probability_matrix, source_row, num_samples) {
  for (knn_col in KNN_DISTANCE_COLUMNS) {
    if (!knn_col %in% colnames(source_row)) next
    parsed <- parse_numeric_string(source_row[[knn_col]][1])
    if (length(parsed) == num_samples) {
      probability_matrix[[knn_col]] <- parsed
    }
  }
  probability_matrix
}

#' Load outer CV results for a single model
#' @param file_path Path to the CSV file containing outer CV results
#' @param classification_type Classification type: "standard" or "OvR"
#' @return Data frame with outer CV results
load_outer_cv_results <- function(file_path, classification_type) {
  cat(sprintf("Loading outer CV results from: %s\n", file_path))

  if (!file.exists(file_path)) {
    warning(sprintf("File does not exist: %s", file_path))
    return(NULL)
  }

  results <- safe_read_file(file_path, function(f) data.frame(data.table::fread(f, sep = ","), row.names = 1))

  if (is.null(results)) {
    warning(sprintf("Failed to load file: %s", file_path))
    return(NULL)
  }

  # For One-vs-Rest, add class labels if not present
  if (classification_type == "OvR" && !"class_label" %in% colnames(results)) {
    # Load label mapping to add class labels
    label_mapping <- safe_read_file("label_mapping_df_n10.csv", read.csv)
    if (!is.null(label_mapping)) {
      results$class_label <- label_mapping$Label[results$class + 1]
    }
  }

  cat(sprintf("  Loaded %d rows of outer CV results\n", nrow(results)))
  return(results)
}

#' Generate outer CV probability matrices for One-vs-Rest classification
#' @param outer_cv_results Outer CV results data frame
#' @param label_mapping Label mapping data frame
#' @param filter_unseen_classes Whether to filter samples with classes not in training (default: TRUE)
#' @return List of probability matrices organized by outer fold (and filtering statistics if filtered)
generate_outer_ovr_probability_matrices <- function(outer_cv_results, label_mapping, filtered_index_map_zero_based, filter_unseen_classes = TRUE, merge_classes = FALSE) {
  cat("Generating outer One-vs-Rest probability matrices...\n")

  if (filter_unseen_classes) {
    cat("  Filtering samples with classes not present in training set...\n")
  }

  outer_fold_ids <- unique(outer_cv_results$outer_fold)
  probability_matrices <- list()
  filtering_stats <- list()

  for (outer_fold_id in outer_fold_ids) {
    cat(sprintf("  Processing outer fold %s...\n", as.character(outer_fold_id)))

    outer_fold_data <- outer_cv_results[outer_cv_results$outer_fold == outer_fold_id, ]
    # class_labels contains the classes that were in the training set (OvR only creates models for training classes)
    class_labels <- unique(outer_fold_data$class_label)

    # Skip if no data
    if (nrow(outer_fold_data) == 0) {
      next
    }

    # Get the number of samples from the first row
    first_row <- outer_fold_data[1, ]
    num_samples <- length(parse_numeric_string(first_row$preds_prob))

    if (num_samples == 0) {
      warning(sprintf("No valid predictions for outer fold", outer_fold_id))
      next
    }

    # Initialize probability matrix
    probability_matrix <- matrix(NA, nrow = num_samples, ncol = length(class_labels))
    colnames(probability_matrix) <- class_labels
    true_labels_vector <- rep(NA, num_samples)

    # Fill probability matrix for each class
    for (j in seq_along(class_labels)) {
      current_class_label <- class_labels[j]
      class_row <- outer_fold_data[outer_fold_data$class_label == current_class_label, ]

      if (nrow(class_row) == 0) next

      # Extract probabilities for this class
      probs <- parse_numeric_string(class_row$preds_prob)
      if (length(probs) == num_samples) {
        probability_matrix[, j] <- probs
      }

      # Extract true labels (1 = this class, 0 = not this class)
      target_values <- parse_numeric_string(class_row$y_val)
      true_labels_vector[target_values == 1] <- current_class_label
    }

    # Skip if no true labels found
    if (all(is.na(true_labels_vector))) {
      warning(sprintf("No true labels found for outer fold %d", outer_fold_id))
      next
    }

    # Normalize probabilities to sum to 1 for each sample
    probability_matrix <- t(apply(probability_matrix, 1, function(row) {
      if (sum(row, na.rm = TRUE) > 0) {
        row / sum(row, na.rm = TRUE)
      } else {
        row
      }
    }))

    # Convert to data frame and ensure all required columns exist
    probability_matrix <- data.frame(probability_matrix)
    probability_matrix <- ensure_all_class_columns(probability_matrix, label_mapping)

    # Add true labels and fold information
    probability_matrix$y <- make.names(true_labels_vector)
    probability_matrix$outer_fold <- outer_fold_id

    # Store sample indices for reference
    sample_indices_local <- parse_numeric_string(first_row$sample_indices)
    if (length(sample_indices_local) == num_samples) {
      probability_matrix$sample_indices <- map_filtered_local_to_global_indices_outer(
        sample_indices_local, filtered_index_map_zero_based
      )
    }
    probability_matrix <- add_optional_knn_columns(probability_matrix, first_row, num_samples)

    # Apply class merging if requested (before filtering)
    if (merge_classes) {
      probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_prob_method = "sum")
      # Update class_labels after merging for filtering
      class_labels <- colnames(probability_matrix)[!colnames(probability_matrix) %in%
                                                    c("y", "outer_fold", "sample_indices", KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS)]
    }

    # Apply filtering if requested
    if (filter_unseen_classes) {
      filter_result <- filter_samples_by_training_classes(
        probability_matrix,
        class_labels,  # class_labels are the training classes for OvR (may be merged)
        outer_fold_id,
        handle_na_labels = FALSE  # Outer CV doesn't have NA labels
      )
      probability_matrix <- filter_result$filtered_matrix
      if (!is.null(filter_result$stats)) {
        filtering_stats[[as.character(outer_fold_id)]] <- filter_result$stats
      }
    }

    probability_matrices[[as.character(outer_fold_id)]] <- probability_matrix
  }

  # Return both matrices and filtering statistics
  result <- list(matrices = probability_matrices)
  if (filter_unseen_classes && length(filtering_stats) > 0) {
    result$filtering_stats <- do.call(rbind, filtering_stats)
  }

  return(result)
}

#' Generate outer CV probability matrices for standard multiclass classification
#' @param outer_cv_results Outer CV results data frame
#' @param label_mapping Label mapping data frame
#' @param filtered_subtypes Filtered leukemia subtypes
#' @param filter_unseen_classes Whether to filter samples with classes not in training (default: TRUE)
#' @return List of probability matrices organized by outer fold (and filtering statistics if filtered)
generate_outer_standard_probability_matrices <- function(outer_cv_results, label_mapping, filtered_subtypes, filtered_index_map_zero_based, filter_unseen_classes = TRUE, merge_classes = FALSE) {
  cat("Generating outer CV standard probability matrices...\n")

  if (filter_unseen_classes) {
    cat("  Filtering samples with classes not present in training set...\n")
  }

  outer_fold_ids <- unique(outer_cv_results$outer_fold)
  probability_matrices <- list()
  filtering_stats <- list()

  for (outer_fold_id in outer_fold_ids) {
    cat(sprintf("  Processing outer fold %s...\n", as.character(outer_fold_id)))

    fold_data <- outer_cv_results[outer_cv_results$outer_fold == outer_fold_id, ]

    if (nrow(fold_data) == 0) {
      warning(sprintf("No data for outer fold", outer_fold_id))
      next
    }

    # Take the first (and typically only) row for this fold
    fold_row <- fold_data[1, ]

    # Extract class information (these are the training classes)
    class_indices <- parse_numeric_string(fold_row$classes)
    class_labels <- label_mapping$Label[class_indices + 1]

    # Extract sample information
    sample_indices_local <- parse_numeric_string(fold_row$sample_indices)
    num_samples <- length(sample_indices_local)

    if (num_samples == 0) {
      warning(sprintf("No samples for outer fold %d", outer_fold_id))
      next
    }

    # Extract prediction probabilities
    probs <- parse_numeric_string(fold_row$preds_prob)

    if (length(probs) != num_samples * length(class_labels)) {
      warning(sprintf("Probability dimensions don't match for outer fold %d", outer_fold_id))
      next
    }

    # Reshape probabilities into matrix (samples x classes)
    probability_matrix <- t(matrix(probs, ncol = num_samples, nrow = length(class_labels)))
    colnames(probability_matrix) <- make.names(class_labels)

    # Convert to data frame and ensure all required columns exist
    probability_matrix <- data.frame(probability_matrix)
    probability_matrix <- ensure_all_class_columns(probability_matrix, label_mapping)

    # Add true labels using sample indices
    probability_matrix$y <- make.names(filtered_subtypes[sample_indices_local + 1])
    probability_matrix$outer_fold <- outer_fold_id
    probability_matrix$sample_indices <- map_filtered_local_to_global_indices_outer(
      sample_indices_local, filtered_index_map_zero_based
    )
    probability_matrix <- add_optional_knn_columns(probability_matrix, fold_row, num_samples)

    # Apply class merging if requested (before filtering)
    if (merge_classes) {
      probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_prob_method = "sum")
      # Update class_labels after merging for filtering
      class_labels <- colnames(probability_matrix)[!colnames(probability_matrix) %in%
                                                    c("y", "outer_fold", "sample_indices", KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS)]
    }

    # Apply filtering if requested
    if (filter_unseen_classes) {
      filter_result <- filter_samples_by_training_classes(
        probability_matrix,
        class_labels,  # class_labels are the training classes (may be merged)
        outer_fold_id,
        handle_na_labels = FALSE  # Outer CV doesn't have NA labels
      )
      probability_matrix <- filter_result$filtered_matrix
      if (!is.null(filter_result$stats)) {
        filtering_stats[[as.character(outer_fold_id)]] <- filter_result$stats
      }
    }

    probability_matrices[[as.character(outer_fold_id)]] <- probability_matrix
  }

  # Return both matrices and filtering statistics
  result <- list(matrices = probability_matrices)
  if (filter_unseen_classes && length(filtering_stats) > 0) {
    result$filtering_stats <- do.call(rbind, filtering_stats)
  }

  return(result)
}

#' Apply global product-of-experts ensemble weights from inner CV to outer CV probability matrices.
#' Inner CV selects weights with the same PoE rule (inner_cv_analysis.R + evaluate_batch_weights_global).
#' @param outer_prob_matrices Outer CV probability matrices for all models
#' @param ensemble_weights_data Ensemble weights from inner CV analysis
#' @param type Type of analysis ("cv" or "loso")
#' @return List of ensemble probability matrices
apply_ensemble_weights_to_outer_cv <- function(outer_prob_matrices, ensemble_weights_data, type = "cv") {
  cat("Applying global product-of-experts ensemble weights to outer CV results...\n")

  weights_to_use <- ensemble_weights_data$global_weights
  if (is.null(weights_to_use)) {
    warning(sprintf("No global weights available for %s analysis", type))
    return(NULL)
  }

  # Get common folds across all models
  common_folds <- Reduce(intersect, lapply(outer_prob_matrices, function(x) names(x[[type]])))
  ensemble_matrices <- list()

  for (fold_name in common_folds) {
    cat(sprintf("  Processing fold %s...\n", fold_name))

    # Extract probability matrices for this fold
    svm_matrix <- outer_prob_matrices$svm[[type]][[fold_name]]
    xgb_matrix <- outer_prob_matrices$xgboost[[type]][[fold_name]]
    nn_matrix <- outer_prob_matrices$neural_net[[type]][[fold_name]]

    # Check if all matrices exist
    if (is.null(svm_matrix) || is.null(xgb_matrix) || is.null(nn_matrix)) {
      warning(sprintf("Missing probability matrix for fold %s", fold_name))
      next
    }

    # Align samples across all three models (critical after filtering)
    # Find common sample indices to ensure all models have the same samples
    if ("sample_indices" %in% colnames(svm_matrix) &&
        "sample_indices" %in% colnames(xgb_matrix) &&
        "sample_indices" %in% colnames(nn_matrix)) {

      svm_samples <- svm_matrix$sample_indices
      xgb_samples <- xgb_matrix$sample_indices
      nn_samples <- nn_matrix$sample_indices

      # Find common samples
      common_samples <- Reduce(intersect, list(svm_samples, xgb_samples, nn_samples))

      if (length(common_samples) == 0) {
        warning(sprintf("No common samples across models for fold %s, skipping", fold_name))
        next
      }

      # Filter to common samples
      svm_matrix <- svm_matrix[svm_matrix$sample_indices %in% common_samples, ]
      xgb_matrix <- xgb_matrix[xgb_matrix$sample_indices %in% common_samples, ]
      nn_matrix <- nn_matrix[nn_matrix$sample_indices %in% common_samples, ]

      # Sort by sample indices to ensure alignment
      svm_matrix <- svm_matrix[order(svm_matrix$sample_indices), ]
      xgb_matrix <- xgb_matrix[order(xgb_matrix$sample_indices), ]
      nn_matrix <- nn_matrix[order(nn_matrix$sample_indices), ]

      # Log if samples were dropped
      n_dropped <- length(svm_samples) - length(common_samples)
      if (n_dropped > 0) {
        cat(sprintf("    Aligned samples: dropped %d samples to match across models\n", n_dropped))
      }
    } else {
      # If no sample_indices column, check row counts match
      if (nrow(svm_matrix) != nrow(xgb_matrix) || nrow(svm_matrix) != nrow(nn_matrix)) {
        warning(sprintf("Sample counts don't match for fold %s (SVM: %d, XGB: %d, NN: %d), skipping ensemble",
                       fold_name, nrow(svm_matrix), nrow(xgb_matrix), nrow(nn_matrix)))
        next
      }
    }

    # Manual alignment since the function expects a different structure
    truth <- svm_matrix$y

    # Remove non-probability columns
    svm_probs <- svm_matrix[, !colnames(svm_matrix) %in% c("y", "outer_fold", "sample_indices", KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS), drop = FALSE]
    xgb_probs <- xgb_matrix[, !colnames(xgb_matrix) %in% c("y", "outer_fold", "sample_indices", KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS), drop = FALSE]
    nn_probs <- nn_matrix[, !colnames(nn_matrix) %in% c("y", "outer_fold", "sample_indices", KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS), drop = FALSE]

    # Ensure all probability columns are numeric
    svm_probs <- data.frame(lapply(svm_probs, function(x) as.numeric(as.character(x))))
    xgb_probs <- data.frame(lapply(xgb_probs, function(x) as.numeric(as.character(x))))
    nn_probs <- data.frame(lapply(nn_probs, function(x) as.numeric(as.character(x))))

    # Get all class names
    all_classes <- unique(c(colnames(svm_probs), colnames(xgb_probs), colnames(nn_probs)))

    # Ensure all matrices have the same columns
    for (class_name in all_classes) {
      if (!class_name %in% colnames(svm_probs)) svm_probs[[class_name]] <- 0
      if (!class_name %in% colnames(xgb_probs)) xgb_probs[[class_name]] <- 0
      if (!class_name %in% colnames(nn_probs)) nn_probs[[class_name]] <- 0
    }

    # Reorder columns
    svm_probs <- svm_probs[, all_classes, drop = FALSE]
    xgb_probs <- xgb_probs[, all_classes, drop = FALSE]
    nn_probs <- nn_probs[, all_classes, drop = FALSE]

    # Global weights: product-of-experts in probability space, p ∝ Π_m p_m^{w_m}
    fold_weights <- weights_to_use[[fold_name]]
    if (is.null(fold_weights)) {
      warning(sprintf("No global weights for fold %s, using DNN-only fallback", fold_name))
      fold_weights <- list(weights = list(SVM = 0, XGB = 0, NN = 1))
    }

    weights <- fold_weights$weights

    svm_weight <- ifelse(is.null(weights$SVM) || is.na(weights$SVM), 1, as.numeric(weights$SVM))
    xgb_weight <- ifelse(is.null(weights$XGB) || is.na(weights$XGB), 1, as.numeric(weights$XGB))
    nn_weight <- ifelse(is.null(weights$NN) || is.na(weights$NN), 1, as.numeric(weights$NN))

    eps <- 1e-12
    ensemble_matrix <- (pmax(svm_probs, eps) ^ svm_weight) *
      (pmax(xgb_probs, eps) ^ xgb_weight) *
      (pmax(nn_probs, eps) ^ nn_weight)

    # Normalize probabilities
    ensemble_matrix <- t(apply(ensemble_matrix, 1, function(row) {
      # Replace any NA or infinite values with 0
      row[is.na(row) | is.infinite(row)] <- 0

      if (sum(row, na.rm = TRUE) > 0) {
        row / sum(row, na.rm = TRUE)
      } else {
        # If all values are 0, set equal probabilities
        rep(1/length(row), length(row))
      }
    }))

    # Convert to data frame and add metadata
    ensemble_matrix <- data.frame(ensemble_matrix)

    # Ensure all probability columns are numeric and replace any remaining NA values
    for (col in colnames(ensemble_matrix)) {
      if (col != "y" && col != "outer_fold") {
        ensemble_matrix[[col]] <- as.numeric(ensemble_matrix[[col]])
        ensemble_matrix[[col]][is.na(ensemble_matrix[[col]])] <- 0
      }
    }

    ensemble_matrix$y <- truth

    # Handle fold names properly - keep as character for LOSO, convert to numeric for CV when possible
    if (type == "cv" && !is.na(suppressWarnings(as.numeric(fold_name)))) {
      ensemble_matrix$outer_fold <- as.numeric(fold_name)
    } else {
      ensemble_matrix$outer_fold <- fold_name
    }

    # Preserve sample indices if available (important for tracking after filtering)
    if ("sample_indices" %in% colnames(svm_matrix)) {
      ensemble_matrix$sample_indices <- svm_matrix$sample_indices
    }

    # KNN reject features are defined in the same transformed space for every model.
    # After row alignment above, copy from SVM and require XGB/NN to match (no silent NA).
    knn_tol <- 1e-4
    for (kcol in KNN_DISTANCE_COLUMNS) {
      if (!kcol %in% colnames(svm_matrix)) next
      svm_k <- as.numeric(svm_matrix[[kcol]])
      if (!all(is.finite(svm_k))) {
        stop(
          sprintf(
            "Ensemble fold %s (%s): non-finite KNN column '%s' on SVM matrix. Re-run run_outer_cv.py (--include_leftout) for all models.",
            fold_name, type, kcol
          )
        )
      }
      ensemble_matrix[[kcol]] <- svm_k
      if (kcol %in% colnames(xgb_matrix)) {
        xgb_k <- as.numeric(xgb_matrix[[kcol]])
        if (!all(is.finite(xgb_k)) || max(abs(svm_k - xgb_k), na.rm = TRUE) > knn_tol) {
          stop(
            sprintf(
              "Ensemble fold %s (%s): KNN column '%s' mismatch or non-finite on XGB vs SVM.",
              fold_name, type, kcol
            )
          )
        }
      }
      if (kcol %in% colnames(nn_matrix)) {
        nn_k <- as.numeric(nn_matrix[[kcol]])
        if (!all(is.finite(nn_k)) || max(abs(svm_k - nn_k), na.rm = TRUE) > knn_tol) {
          stop(
            sprintf(
              "Ensemble fold %s (%s): KNN column '%s' mismatch or non-finite on NN vs SVM.",
              fold_name, type, kcol
            )
          )
        }
      }
    }

    ensemble_matrices[[fold_name]] <- ensemble_matrix
  }

  return(ensemble_matrices)
}

#' Calculate comprehensive performance metrics for outer CV results
#' @param probability_matrices Probability matrices (individual models and ensembles)
#' @param type Type of analysis ("cv" or "loso")
#' @return List of performance results
calculate_outer_cv_performance <- function(probability_matrices, type = "cv") {
  cat(sprintf("Calculating outer CV performance for %s...\n", toupper(type)))

  performance_results <- list()

  for (model_name in names(probability_matrices)) {
    if (!type %in% names(probability_matrices[[model_name]])) {
      next
    }

    cat(sprintf("  Analyzing %s...\n", toupper(model_name)))
    model_matrices <- probability_matrices[[model_name]][[type]]
    model_performance <- list()

    for (fold_name in names(model_matrices)) {
      prob_matrix <- model_matrices[[fold_name]]

      # Extract true labels and predictions
      truth <- prob_matrix$y
      prob_cols <- prob_matrix[, !colnames(prob_matrix) %in% c("y", "outer_fold", "sample_indices", "study", KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS), drop = FALSE]

      # Get predictions (class with highest probability)
      preds <- colnames(prob_cols)[apply(prob_cols, 1, which.max)]

      # Clean class labels
      truth <- gsub("Class.", "", truth)
      preds <- gsub("Class.", "", preds)

      # Ensure all classes are represented
      all_classes <- unique(c(truth, preds))
      truth <- factor(truth, levels = all_classes)
      preds <- factor(preds, levels = all_classes)

      # Calculate confusion matrix and metrics
      cm <- caret::confusionMatrix(preds, truth)

      # Extract per-class metrics
      per_class_metrics <- list()
      if (!is.null(cm$byClass) && nrow(cm$byClass) > 0) {
        for (class_name in rownames(cm$byClass)) {
          per_class_metrics[[class_name]] <- list(
            Sensitivity = cm$byClass[class_name, "Sensitivity"],
            Specificity = cm$byClass[class_name, "Specificity"],
            Precision = cm$byClass[class_name, "Precision"],
            Recall = cm$byClass[class_name, "Recall"],
            F1 = cm$byClass[class_name, "F1"],
            Balanced_Accuracy = cm$byClass[class_name, "Balanced Accuracy"]
          )
        }
      }

      model_performance[[fold_name]] <- list(
        confusion_matrix = cm,
        predictions = preds,  # Save raw predictions
        truth = truth,        # Save raw truth values
        kappa = as.numeric(cm$overall["Kappa"]),
        accuracy = as.numeric(cm$overall["Accuracy"]),
        balanced_accuracy = mean(cm$byClass[, "Balanced Accuracy"], na.rm = TRUE),
        f1_macro = mean(cm$byClass[, "F1"], na.rm = TRUE),
        sensitivity_macro = mean(cm$byClass[, "Sensitivity"], na.rm = TRUE),
        specificity_macro = mean(cm$byClass[, "Specificity"], na.rm = TRUE),
        per_class_metrics = per_class_metrics
      )
    }

    performance_results[[model_name]] <- model_performance
  }

  return(performance_results)
}

#' Summarize performance across all folds
#' @param performance_results Performance results from calculate_outer_cv_performance
#' @return Data frame with summary statistics
summarize_outer_cv_performance <- function(performance_results) {
  cat("Summarizing outer CV performance...\n")

  summary_data <- data.frame()

  for (model_name in names(performance_results)) {
    model_perf <- performance_results[[model_name]]

    if (length(model_perf) == 0) next

    # Extract metrics across folds
    kappas <- sapply(model_perf, function(x) x$kappa)
    accuracies <- sapply(model_perf, function(x) x$accuracy)
    balanced_accuracies <- sapply(model_perf, function(x) x$balanced_accuracy)
    f1_macros <- sapply(model_perf, function(x) x$f1_macro)

    # Calculate summary statistics
    summary_row <- data.frame(
      Model = model_name,
      N_Folds = length(model_perf),
      Mean_Kappa = mean(kappas, na.rm = TRUE),
      SD_Kappa = sd(kappas, na.rm = TRUE),
      Mean_Accuracy = mean(accuracies, na.rm = TRUE),
      SD_Accuracy = sd(accuracies, na.rm = TRUE),
      Mean_Balanced_Accuracy = mean(balanced_accuracies, na.rm = TRUE),
      SD_Balanced_Accuracy = sd(balanced_accuracies, na.rm = TRUE),
      Mean_F1_Macro = mean(f1_macros, na.rm = TRUE),
      SD_F1_Macro = sd(f1_macros, na.rm = TRUE),
      stringsAsFactors = FALSE
    )

    summary_data <- rbind(summary_data, summary_row)
  }

  # Sort by mean kappa (descending)
  summary_data <- summary_data[order(summary_data$Mean_Kappa, decreasing = TRUE), ]

  return(summary_data)
}

#' Summarize per-class performance metrics across all folds
#' @param performance_results Performance results from calculate_outer_cv_performance
#' @return Data frame with per-class summary statistics
summarize_per_class_performance <- function(performance_results) {
  cat("Summarizing per-class performance metrics...\n")

  per_class_summary <- data.frame()

  for (model_name in names(performance_results)) {
    model_perf <- performance_results[[model_name]]

    if (length(model_perf) == 0) next

    # Get all unique classes across all folds
    all_classes <- unique(unlist(lapply(model_perf, function(x) names(x$per_class_metrics))))

    for (class_name in all_classes) {
      # Extract metrics for this class across all folds
      sensitivities <- numeric(0)
      specificities <- numeric(0)
      precisions <- numeric(0)
      recalls <- numeric(0)
      f1_scores <- numeric(0)
      balanced_accuracies <- numeric(0)

      for (fold_name in names(model_perf)) {
        fold_perf <- model_perf[[fold_name]]
        if (!is.null(fold_perf$per_class_metrics) && class_name %in% names(fold_perf$per_class_metrics)) {
          class_metrics <- fold_perf$per_class_metrics[[class_name]]
          sensitivities <- c(sensitivities, class_metrics$Sensitivity)
          specificities <- c(specificities, class_metrics$Specificity)
          precisions <- c(precisions, class_metrics$Precision)
          recalls <- c(recalls, class_metrics$Recall)
          f1_scores <- c(f1_scores, class_metrics$F1)
          balanced_accuracies <- c(balanced_accuracies, class_metrics$Balanced_Accuracy)
        }
      }

      # Calculate summary statistics for this class
      if (length(sensitivities) > 0) {
        class_summary <- data.frame(
          Model = model_name,
          Class = class_name,
          N_Folds = length(sensitivities),
          Mean_Sensitivity = mean(sensitivities, na.rm = TRUE),
          SD_Sensitivity = sd(sensitivities, na.rm = TRUE),
          Mean_Specificity = mean(specificities, na.rm = TRUE),
          SD_Specificity = sd(specificities, na.rm = TRUE),
          Mean_Precision = mean(precisions, na.rm = TRUE),
          SD_Precision = sd(precisions, na.rm = TRUE),
          Mean_Recall = mean(recalls, na.rm = TRUE),
          SD_Recall = sd(recalls, na.rm = TRUE),
          Mean_F1 = mean(f1_scores, na.rm = TRUE),
          SD_F1 = sd(f1_scores, na.rm = TRUE),
          Mean_Balanced_Accuracy = mean(balanced_accuracies, na.rm = TRUE),
          SD_Balanced_Accuracy = sd(balanced_accuracies, na.rm = TRUE),
          stringsAsFactors = FALSE
        )

        per_class_summary <- rbind(per_class_summary, class_summary)
      }
    }
  }

  # Sort by model and then by mean F1 score (descending)
  per_class_summary <- per_class_summary[order(per_class_summary$Model, -per_class_summary$Mean_F1), ]

  return(per_class_summary)
}

# =============================================================================
# Ensemble-Specific: Model Disagreement Features
# =============================================================================

#' Compute model disagreement features and attach to ensemble probability matrices.
#' For each sample, calculates how many of {SVM, XGB, NN} agree on the top-1 and
#' variance of model top-1 probabilities across models.
#'
#' Class-probability columns are whatever remains after dropping metadata and calibration
#' scores (`confidence_*`) using \code{PROB_MATRIX_META_COLUMNS}; variance uses only
#' SVM/XGB/NN class probabilities.
#' @param ensemble_fold_matrices Named list of fold -> ensemble data.frame
#' @param per_model_matrices Named list: model_name -> list(type -> list(fold -> data.frame))
#' @param type "cv" or "loso"
#' @return Modified ensemble_fold_matrices with n_models_agree and
#'   top1_prob_variance_across_models columns
compute_disagreement_features <- function(ensemble_fold_matrices, per_model_matrices, type) {
  cat("Computing model disagreement features for ensemble...\n")

  add_roi_reject_features <- function(df, alpha = 0.10, eps = 1e-8) {
    if (is.null(df) || nrow(df) == 0) return(df)
    prob_cols <- colnames(df)[!colnames(df) %in% PROB_MATRIX_META_COLUMNS]
    if (length(prob_cols) == 0) {
      df$trust_ratio_knn10 <- NA_real_
      df$conformal_set_size_90 <- NA_real_
      return(df)
    }
    # Force probability block to a strict numeric matrix; mixed schemas can
    # otherwise yield list/object columns that break sort/cumsum.
    prob_df <- df[, prob_cols, drop = FALSE]
    prob_df[] <- lapply(prob_df, function(x) {
      v <- suppressWarnings(as.numeric(x))
      v[!is.finite(v)] <- 0
      v
    })
    prob_mat <- as.matrix(prob_df)
    mode(prob_mat) <- "numeric"
    prob_mat <- pmax(prob_mat, 0)
    rs <- rowSums(prob_mat)
    rs[!is.finite(rs) | rs <= 0] <- 1
    prob_mat <- prob_mat / rs

    # Trust-score style distance-density proxy from KNN summaries.
    if (all(c("knn10_min_d", "knn10_q90_d") %in% colnames(df))) {
      knn_min <- as.numeric(df$knn10_min_d)
      knn_q90 <- as.numeric(df$knn10_q90_d)
      df$trust_ratio_knn10 <- knn_min / pmax(knn_q90, eps)
    } else {
      df$trust_ratio_knn10 <- NA_real_
    }

    # Conformal-style set size: smallest k with cumulative top-k probability >= 1-alpha.
    p_sorted <- t(apply(prob_mat, 1, sort, decreasing = TRUE))
    cum_sorted <- t(apply(p_sorted, 1, cumsum))
    threshold <- 1 - alpha
    n_classes <- ncol(cum_sorted)
    set_size <- apply(cum_sorted, 1, function(cs) {
      idx <- which(cs >= threshold)[1]
      if (is.na(idx)) n_classes else idx
    })
    df$conformal_set_size_90 <- as.numeric(set_size)
    df
  }

  for (fold_name in names(ensemble_fold_matrices)) {
    ens <- ensemble_fold_matrices[[fold_name]]
    if (is.null(ens) || nrow(ens) == 0) next

    # Get aligned per-model matrices for this fold
    svm_mat <- per_model_matrices[["svm"]][[type]][[fold_name]]
    xgb_mat <- per_model_matrices[["xgboost"]][[type]][[fold_name]]
    nn_mat  <- per_model_matrices[["neural_net"]][[type]][[fold_name]]

    if (is.null(svm_mat) || is.null(xgb_mat) || is.null(nn_mat)) {
      ens$n_models_agree <- NA
      ens$top1_prob_variance_across_models <- NA_real_
      ens <- add_roi_reject_features(ens)
      ensemble_fold_matrices[[fold_name]] <- ens
      next
    }

    # Align by the strict intersection of sample_indices across all matrices.
    # De-duplicate each matrix first so alignment is one row per sample.
    if ("sample_indices" %in% colnames(ens) &&
        "sample_indices" %in% colnames(svm_mat) &&
        "sample_indices" %in% colnames(xgb_mat) &&
        "sample_indices" %in% colnames(nn_mat)) {
      dedup_by_sample_idx <- function(df) {
        df <- df[!is.na(df$sample_indices), , drop = FALSE]
        df[!duplicated(df$sample_indices), , drop = FALSE]
      }
      ens <- dedup_by_sample_idx(ens)
      svm_mat <- dedup_by_sample_idx(svm_mat)
      xgb_mat <- dedup_by_sample_idx(xgb_mat)
      nn_mat <- dedup_by_sample_idx(nn_mat)

      common_idx <- Reduce(
        intersect,
        list(ens$sample_indices, svm_mat$sample_indices, xgb_mat$sample_indices, nn_mat$sample_indices)
      )
      if (length(common_idx) == 0) {
        ens$n_models_agree <- NA
        ens$top1_prob_variance_across_models <- NA_real_
        ens <- add_roi_reject_features(ens)
        ensemble_fold_matrices[[fold_name]] <- ens
        next
      }
      # Use ensemble order as canonical ordering; subset each model by exact match.
      ens <- ens[ens$sample_indices %in% common_idx, , drop = FALSE]
      ens <- ens[order(ens$sample_indices), , drop = FALSE]
      key_idx <- ens$sample_indices

      svm_mat <- svm_mat[match(key_idx, svm_mat$sample_indices), , drop = FALSE]
      xgb_mat <- xgb_mat[match(key_idx, xgb_mat$sample_indices), , drop = FALSE]
      nn_mat  <- nn_mat[match(key_idx, nn_mat$sample_indices), , drop = FALSE]

      # Guard against unexpected unmatched rows after match().
      valid_rows <- is.finite(svm_mat$sample_indices) & is.finite(xgb_mat$sample_indices) & is.finite(nn_mat$sample_indices)
      ens <- ens[valid_rows, , drop = FALSE]
      svm_mat <- svm_mat[valid_rows, , drop = FALSE]
      xgb_mat <- xgb_mat[valid_rows, , drop = FALSE]
      nn_mat <- nn_mat[valid_rows, , drop = FALSE]
    } else {
      # Fallback when sample_indices is unavailable on one side: trim to common length.
      n_min <- min(nrow(ens), nrow(svm_mat), nrow(xgb_mat), nrow(nn_mat))
      if (n_min == 0) {
        ens$n_models_agree <- NA
        ens$top1_prob_variance_across_models <- NA_real_
        ens <- add_roi_reject_features(ens)
        ensemble_fold_matrices[[fold_name]] <- ens
        next
      }
      ens <- ens[seq_len(n_min), , drop = FALSE]
      svm_mat <- svm_mat[seq_len(n_min), , drop = FALSE]
      xgb_mat <- xgb_mat[seq_len(n_min), , drop = FALSE]
      nn_mat  <- nn_mat[seq_len(n_min), , drop = FALSE]
    }

    # Extract probability-only columns
    svm_prob_cols <- colnames(svm_mat)[!colnames(svm_mat) %in% PROB_MATRIX_META_COLUMNS]
    xgb_prob_cols <- colnames(xgb_mat)[!colnames(xgb_mat) %in% PROB_MATRIX_META_COLUMNS]
    nn_prob_cols  <- colnames(nn_mat)[!colnames(nn_mat) %in% PROB_MATRIX_META_COLUMNS]
    all_prob_cols <- unique(c(svm_prob_cols, xgb_prob_cols, nn_prob_cols))

    # Build aligned probability matrices (fill 0 for missing columns)
    make_aligned <- function(mat, prob_cols) {
      m <- matrix(0, nrow = nrow(mat), ncol = length(all_prob_cols))
      colnames(m) <- all_prob_cols
      for (col in intersect(prob_cols, all_prob_cols)) {
        vals <- suppressWarnings(as.numeric(mat[[col]]))
        # Mixed merged/unmerged class schemas produce NA in absent-class columns.
        # Treat non-finite class probabilities as 0 before row normalization.
        vals[!is.finite(vals)] <- 0
        m[, col] <- vals
      }
      # Normalise rows
      rs <- rowSums(m, na.rm = TRUE)
      rs[rs == 0] <- 1
      m / rs
    }

    svm_p <- make_aligned(svm_mat, svm_prob_cols)
    xgb_p <- make_aligned(xgb_mat, xgb_prob_cols)
    nn_p  <- make_aligned(nn_mat, nn_prob_cols)

    # Fail fast if any model has rows with no finite class probabilities.
    find_all_nonfinite_rows <- function(m) which(rowSums(is.finite(m)) == 0)
    bad_svm <- find_all_nonfinite_rows(svm_p)
    bad_xgb <- find_all_nonfinite_rows(xgb_p)
    bad_nn <- find_all_nonfinite_rows(nn_p)
    if (length(bad_svm) > 0 || length(bad_xgb) > 0 || length(bad_nn) > 0) {
      idx_vals <- if ("sample_indices" %in% colnames(ens)) ens$sample_indices else seq_len(nrow(ens))
      bad_idx <- unique(c(idx_vals[bad_svm], idx_vals[bad_xgb], idx_vals[bad_nn]))
      stop(sprintf(
        paste0(
          "Disagreement feature computation failed in outer CV: rows with no finite class probabilities ",
          "detected (SVM=%d, XGB=%d, NN=%d). Example sample_indices: %s"
        ),
        length(bad_svm), length(bad_xgb), length(bad_nn),
        paste(head(bad_idx, 15), collapse = ", ")
      ))
    }

    # Final safety guard: enforce identical row counts before row-wise indexing.
    n_eff <- min(nrow(ens), nrow(svm_p), nrow(xgb_p), nrow(nn_p))
    if (n_eff == 0) {
      ens$n_models_agree <- NA
      ens$top1_prob_variance_across_models <- NA_real_
      ens <- add_roi_reject_features(ens)
      ensemble_fold_matrices[[fold_name]] <- ens
      next
    }
    if (nrow(ens) != n_eff || nrow(svm_p) != n_eff || nrow(xgb_p) != n_eff || nrow(nn_p) != n_eff) {
      ens <- ens[seq_len(n_eff), , drop = FALSE]
      svm_p <- svm_p[seq_len(n_eff), , drop = FALSE]
      xgb_p <- xgb_p[seq_len(n_eff), , drop = FALSE]
      nn_p <- nn_p[seq_len(n_eff), , drop = FALSE]
    }

    n <- nrow(ens)
    n_agree <- integer(n)

    svm_pred <- all_prob_cols[max.col(svm_p, ties.method = "first")]
    xgb_pred <- all_prob_cols[max.col(xgb_p, ties.method = "first")]
    nn_pred  <- all_prob_cols[max.col(nn_p, ties.method = "first")]
    top1_prob_mat <- cbind(
      svm_p[cbind(seq_len(n), max.col(svm_p, ties.method = "first"))],
      xgb_p[cbind(seq_len(n), max.col(xgb_p, ties.method = "first"))],
      nn_p[cbind(seq_len(n), max.col(nn_p, ties.method = "first"))]
    )
    top1_prob_var <- apply(top1_prob_mat, 1, var, na.rm = TRUE)

    for (i in seq_len(n)) {
      preds_3 <- c(svm_pred[i], xgb_pred[i], nn_pred[i])
      preds_3 <- preds_3[!is.na(preds_3)]
      if (length(preds_3) == 0) {
        n_agree[i] <- NA_integer_
      } else {
        n_agree[i] <- max(table(preds_3))
      }
    }

    ens$n_models_agree <- n_agree
    if (any(!is.finite(ens$n_models_agree))) {
      bad_rows <- which(!is.finite(ens$n_models_agree))
      idx_vals <- if ("sample_indices" %in% colnames(ens)) ens$sample_indices else seq_len(nrow(ens))
      stop(sprintf(
        "Disagreement feature computation failed in outer CV: n_models_agree not finite for %d rows. Example sample_indices: %s",
        length(bad_rows), paste(head(idx_vals[bad_rows], 15), collapse = ", ")
      ))
    }
    ens$top1_prob_variance_across_models <- pmax(0, as.numeric(top1_prob_var))
    ens <- add_roi_reject_features(ens)
    ensemble_fold_matrices[[fold_name]] <- ens
    cat(sprintf("  Fold %s: %d samples, mean agreement = %.2f, mean top1 variance = %.4f\n",
                fold_name, n, mean(n_agree), mean(ens$top1_prob_variance_across_models, na.rm = TRUE)))
  }

  ensemble_fold_matrices
}


# =============================================================================
# Left-Out Sample Probability Matrix Functions
# =============================================================================

#' Generate probability matrices from left-out OvR prediction CSVs.
#' Left-out sample indices reference the FULL (unfiltered) dataset.
#' No filtering of unseen classes is applied (left-out classes are intentionally unseen).
#' @param leftout_results Data frame loaded from leftout CSV (OvR format)
#' @param label_mapping Label mapping data frame
#' @param all_subtypes Full (unfiltered) vector of ICC_Subtype labels
#' @param merge_classes Whether to merge in-distribution classes (ignored for left-out rows so OOD remains OOD)
#' @return List with matrices element (list of fold -> data.frame)
generate_leftout_ovr_probability_matrices <- function(leftout_results, label_mapping, all_subtypes, merge_classes = FALSE) {
  cat("Generating left-out OvR probability matrices...\n")

  outer_fold_ids <- unique(leftout_results$outer_fold)
  probability_matrices <- list()

  for (outer_fold_id in outer_fold_ids) {
    fold_data <- leftout_results[leftout_results$outer_fold == outer_fold_id, ]
    class_labels <- unique(fold_data$class_label)
    if (nrow(fold_data) == 0 || length(class_labels) == 0) next

    num_samples <- length(parse_numeric_string(fold_data$preds_prob[1]))
    if (num_samples == 0) next

    probability_matrix <- matrix(NA, nrow = num_samples, ncol = length(class_labels))
    colnames(probability_matrix) <- class_labels

    for (j in seq_along(class_labels)) {
      class_row <- fold_data[fold_data$class_label == class_labels[j], ]
      if (nrow(class_row) == 0) next
      probs <- parse_numeric_string(class_row$preds_prob)
      if (length(probs) == num_samples) {
        probability_matrix[, j] <- probs
      }
    }

    probability_matrix <- t(apply(probability_matrix, 1, function(row) {
      s <- sum(row, na.rm = TRUE)
      if (s > 0) row / s else row
    }))

    probability_matrix <- data.frame(probability_matrix)
    probability_matrix <- ensure_all_class_columns(probability_matrix, label_mapping)

    sample_indices <- parse_numeric_string(fold_data$sample_indices[1])
    probability_matrix$y <- make.names(all_subtypes[sample_indices + 1])
    probability_matrix$outer_fold <- outer_fold_id
    probability_matrix$sample_indices <- sample_indices
    # Propagate per-sample KNN vectors from left-out CSV rows.
    probability_matrix <- add_optional_knn_columns(probability_matrix, fold_data[1, , drop = FALSE], num_samples)
    # Propagate per-sample KNN vectors from left-out CSV rows.
    probability_matrix <- add_optional_knn_columns(probability_matrix, fold_data[1, , drop = FALSE], num_samples)

    # Keep left-out rows fully uncollapsed so OOD subtypes are not folded into
    # merged classes such as other.KMT2A.

    probability_matrices[[as.character(outer_fold_id)]] <- probability_matrix
  }

  list(matrices = probability_matrices)
}


#' Generate probability matrices from left-out standard prediction CSVs.
#' Left-out sample indices reference the FULL (unfiltered) dataset.
#' @param leftout_results Data frame loaded from leftout CSV (standard format)
#' @param label_mapping Label mapping data frame
#' @param all_subtypes Full (unfiltered) vector of ICC_Subtype labels
#' @param merge_classes Whether to merge in-distribution classes (ignored for left-out rows so OOD remains OOD)
#' @return List with matrices element (list of fold -> data.frame)
generate_leftout_standard_probability_matrices <- function(leftout_results, label_mapping, all_subtypes, merge_classes = FALSE) {
  cat("Generating left-out standard probability matrices...\n")

  outer_fold_ids <- unique(leftout_results$outer_fold)
  probability_matrices <- list()

  for (outer_fold_id in outer_fold_ids) {
    fold_data <- leftout_results[leftout_results$outer_fold == outer_fold_id, ]
    if (nrow(fold_data) == 0) next

    fold_row <- fold_data[1, ]
    class_indices <- parse_numeric_string(fold_row$classes)
    class_labels <- label_mapping$Label[class_indices + 1]

    sample_indices <- parse_numeric_string(fold_row$sample_indices)
    num_samples <- length(sample_indices)
    if (num_samples == 0) next

    probs <- parse_numeric_string(fold_row$preds_prob)
    if (length(probs) != num_samples * length(class_labels)) next

    probability_matrix <- t(matrix(probs, ncol = num_samples, nrow = length(class_labels)))
    colnames(probability_matrix) <- make.names(class_labels)

    probability_matrix <- data.frame(probability_matrix)
    probability_matrix <- ensure_all_class_columns(probability_matrix, label_mapping)

    probability_matrix$y <- make.names(all_subtypes[sample_indices + 1])
    probability_matrix$outer_fold <- outer_fold_id
    probability_matrix$sample_indices <- sample_indices
    # Propagate per-sample KNN vectors from left-out CSV rows.
    probability_matrix <- add_optional_knn_columns(probability_matrix, fold_row, num_samples)
    # Propagate per-sample KNN vectors from left-out CSV rows.
    probability_matrix <- add_optional_knn_columns(probability_matrix, fold_row, num_samples)

    # Keep left-out rows fully uncollapsed so OOD subtypes are not folded into
    # merged classes such as other.KMT2A.

    probability_matrices[[as.character(outer_fold_id)]] <- probability_matrix
  }

  list(matrices = probability_matrices)
}


#' Append left-out samples to each fold's probability matrix.
#' @param known_fold_matrices List of fold_name -> data.frame (known-class prob matrices)
#' @param leftout_fold_matrices List of fold_name -> data.frame (left-out prob matrices)
#' @return List of fold_name -> augmented data.frame
augment_fold_matrices_with_leftout <- function(known_fold_matrices, leftout_fold_matrices) {
  augmented <- known_fold_matrices
  meta_cols <- c(
    "y", "inner_fold", "outer_fold", "indices", "study", "sample_indices",
    "confidence_multivariate", "confidence_id", "confidence_correct",
    "confidence_two_head", "confidence_seen_new_cohort", "confidence_unseen",
    "confidence_three_head", "confidence_two_head_postcal",
    "confidence_two_head_min_gate", "confidence_two_head_id_veto",
    "is_leftout", "n_models_agree", "top1_prob_variance_across_models",
    KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS
  )

  for (fold_name in names(leftout_fold_matrices)) {
    leftout_df <- leftout_fold_matrices[[fold_name]]
    leftout_df$is_leftout <- TRUE

    if (fold_name %in% names(augmented)) {
      known_df <- augmented[[fold_name]]
      known_df$is_leftout <- FALSE

      # Keep all columns across known + left-out rows. Missing probability
      # columns are true structural absences, so fill with 0 instead of NA to
      # avoid NA propagation in downstream max-probability/prediction steps.
      all_cols <- union(colnames(known_df), colnames(leftout_df))
      prob_cols_union <- setdiff(all_cols, meta_cols)
      for (cn in setdiff(all_cols, colnames(known_df))) {
        known_df[[cn]] <- if (cn %in% prob_cols_union) 0 else NA
      }
      for (cn in setdiff(all_cols, colnames(leftout_df))) {
        leftout_df[[cn]] <- if (cn %in% prob_cols_union) 0 else NA
      }
      known_df <- known_df[, all_cols, drop = FALSE]
      leftout_df <- leftout_df[, all_cols, drop = FALSE]
      combined_df <- rbind(known_df, leftout_df)
      # If KNN columns exist anywhere on this fold, every row must be finite (no silent NA padding).
      for (kcol in KNN_DISTANCE_COLUMNS) {
        if (!kcol %in% colnames(combined_df)) next
        kv <- suppressWarnings(as.numeric(combined_df[[kcol]]))
        if (!all(is.finite(kv))) {
          stop(
            sprintf(
              "Per-model augment fold %s: KNN column '%s' has missing or non-finite values after binding known + leftout. Re-run run_outer_cv.py --include_leftout for this model.",
              fold_name, kcol
            )
          )
        }
      }
      augmented[[fold_name]] <- combined_df
    } else {
      augmented[[fold_name]] <- leftout_df
    }
  }

  augmented
}


#' Build augmented ensemble matrices by combining per-model left-out predictions.
#' Product-of-experts in probability space using the same global inner-CV weights.
#' @param known_ensemble_matrices List of fold -> ensemble prob matrix (known only)
#' @param leftout_per_model List of model -> fold -> leftout prob matrix
#' @param ensemble_weights Weights from inner CV for this analysis type
#' @param type "cv" or "loso"
#' @return List of fold -> augmented ensemble data.frame
build_augmented_ensemble <- function(known_ensemble_matrices, leftout_per_model, ensemble_weights, type) {
  cat("Building augmented ensemble matrices with left-out samples...\n")

  fold_names <- unique(unlist(lapply(leftout_per_model, function(x) names(x))))
  augmented <- known_ensemble_matrices

  for (fold_name in fold_names) {
    # Collect leftout matrices from each model for this fold
    svm_lo <- leftout_per_model[["svm"]][[fold_name]]
    xgb_lo <- leftout_per_model[["xgboost"]][[fold_name]]
    nn_lo  <- leftout_per_model[["neural_net"]][[fold_name]]

    if (is.null(svm_lo) || is.null(xgb_lo) || is.null(nn_lo)) {
      cat(sprintf("  Skipping fold %s - not all models have left-out predictions\n", fold_name))
      next
    }

    meta_cols <- c("y", "outer_fold", "sample_indices", "is_leftout", KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS)

    # Align samples (match by sample_indices)
    common_idx <- Reduce(intersect, list(svm_lo$sample_indices, xgb_lo$sample_indices, nn_lo$sample_indices))
    if (length(common_idx) == 0) next

    svm_lo <- svm_lo[svm_lo$sample_indices %in% common_idx, ]
    xgb_lo <- xgb_lo[xgb_lo$sample_indices %in% common_idx, ]
    nn_lo  <- nn_lo[nn_lo$sample_indices %in% common_idx, ]

    svm_lo <- svm_lo[order(svm_lo$sample_indices), ]
    xgb_lo <- xgb_lo[order(xgb_lo$sample_indices), ]
    nn_lo  <- nn_lo[order(nn_lo$sample_indices), ]

    # Extract probability columns
    prob_cols_svm <- colnames(svm_lo)[!colnames(svm_lo) %in% meta_cols]
    prob_cols_xgb <- colnames(xgb_lo)[!colnames(xgb_lo) %in% meta_cols]
    prob_cols_nn  <- colnames(nn_lo)[!colnames(nn_lo) %in% meta_cols]

    all_prob_cols <- unique(c(prob_cols_svm, prob_cols_xgb, prob_cols_nn))

    # Ensure all matrices have same probability columns (fill 0 for missing)
    for (col in all_prob_cols) {
      if (!col %in% prob_cols_svm) svm_lo[[col]] <- 0
      if (!col %in% prob_cols_xgb) xgb_lo[[col]] <- 0
      if (!col %in% prob_cols_nn)  nn_lo[[col]] <- 0
    }

    # Get global weights for this fold
    fold_weights <- ensemble_weights$global_weights[[fold_name]]
    if (is.null(fold_weights)) {
      cat(sprintf("  No global weights for fold %s, using equal weights\n", fold_name))
      w_svm <- 1/3; w_xgb <- 1/3; w_nn <- 1/3
    } else {
      w <- fold_weights$weights
      w_svm <- as.numeric(w$SVM)
      w_xgb <- as.numeric(w$XGB)
      w_nn  <- as.numeric(w$NN)
    }

    eps <- 1e-12
    ens_probs <- (pmax(as.matrix(svm_lo[, all_prob_cols]), eps) ^ w_svm) *
      (pmax(as.matrix(xgb_lo[, all_prob_cols]), eps) ^ w_xgb) *
      (pmax(as.matrix(nn_lo[, all_prob_cols]), eps) ^ w_nn)

    # Normalise rows
    row_sums <- rowSums(ens_probs, na.rm = TRUE)
    row_sums[row_sums == 0] <- 1
    ens_probs <- ens_probs / row_sums

    ens_df <- data.frame(ens_probs)
    ens_df$y <- svm_lo$y
    ens_df$outer_fold <- svm_lo$outer_fold
    ens_df$sample_indices <- svm_lo$sample_indices
    ens_df$is_leftout <- TRUE

    # Left-out ensemble rows must carry the same KNN geometry as per-model leftout matrices.
    # Previously these columns were never copied from svm_lo/xgb_lo/nn_lo, so union-padding
    # filled them with NA and multivariate+KNN calibration failed on the pooled set.
    knn_tol <- 1e-4
    for (kcol in KNN_DISTANCE_COLUMNS) {
      if (!all(c(kcol %in% colnames(svm_lo), kcol %in% colnames(xgb_lo), kcol %in% colnames(nn_lo)))) {
        stop(
          sprintf(
            "Augmented ensemble fold %s: missing KNN column '%s' on at least one model's leftout matrix. Re-run run_outer_cv.py --include_leftout for SVM, XGBOOST, and NN.",
            fold_name, kcol
          )
        )
      }
      svm_k <- as.numeric(svm_lo[[kcol]])
      xgb_k <- as.numeric(xgb_lo[[kcol]])
      nn_k <- as.numeric(nn_lo[[kcol]])
      if (!all(is.finite(svm_k)) || !all(is.finite(xgb_k)) || !all(is.finite(nn_k))) {
        stop(
          sprintf(
            "Augmented ensemble fold %s: non-finite KNN values in '%s' for leftout rows.",
            fold_name, kcol
          )
        )
      }
      if (max(abs(svm_k - xgb_k), na.rm = TRUE) > knn_tol || max(abs(svm_k - nn_k), na.rm = TRUE) > knn_tol) {
        stop(
          sprintf(
            "Augmented ensemble fold %s: KNN column '%s' disagrees across models on leftout rows.",
            fold_name, kcol
          )
        )
      }
      ens_df[[kcol]] <- svm_k
    }

    # Append to existing ensemble matrix for this fold
    if (fold_name %in% names(augmented) && !is.null(augmented[[fold_name]])) {
      known_ens <- augmented[[fold_name]]
      known_ens$is_leftout <- FALSE
      all_cols <- union(colnames(known_ens), colnames(ens_df))
      prob_cols_union <- setdiff(all_cols, meta_cols)
      for (cn in setdiff(all_cols, colnames(known_ens))) {
        known_ens[[cn]] <- if (cn %in% prob_cols_union) 0 else NA
      }
      for (cn in setdiff(all_cols, colnames(ens_df))) {
        ens_df[[cn]] <- if (cn %in% prob_cols_union) 0 else NA
      }
      known_ens <- known_ens[, all_cols, drop = FALSE]
      ens_df <- ens_df[, all_cols, drop = FALSE]
      combined_ens <- rbind(known_ens, ens_df)
      for (kcol in KNN_DISTANCE_COLUMNS) {
        if (!kcol %in% colnames(combined_ens)) next
        kv <- suppressWarnings(as.numeric(combined_ens[[kcol]]))
        if (!all(is.finite(kv))) {
          stop(
            sprintf(
              "Augmented ensemble fold %s: after binding known + leftout ensemble rows, KNN column '%s' is not finite on all rows. Known ensemble must include KNN (from run_outer_cv.py outputs) before augmentation; leftout ensemble rows must include matching KNN.",
              fold_name, kcol
            )
          )
        }
      }
      augmented[[fold_name]] <- combined_ens
    } else {
      for (kcol in KNN_DISTANCE_COLUMNS) {
        if (!kcol %in% colnames(ens_df)) next
        kv <- suppressWarnings(as.numeric(ens_df[[kcol]]))
        if (!all(is.finite(kv))) {
          stop(
            sprintf(
              "Augmented ensemble fold %s: leftout-only ensemble matrix has non-finite KNN column '%s'.",
              fold_name, kcol
            )
          )
        }
      }
      augmented[[fold_name]] <- ens_df
    }
  }

  augmented
}


# =============================================================================
# Main Outer CV Analysis Function

#' Main function to run outer CV analysis
#' @param merge_classes Whether to merge classes (MDS/TP53 -> MDS.r, other KMT2A -> other.KMT2A, MECOM -> MECOM)
main_outer_cv <- function(merge_classes = FALSE) {
  # Load required libraries
  load_library_quietly("plyr")
  load_library_quietly("dplyr")
  load_library_quietly("stringr")
  load_library_quietly("caret")
  load_library_quietly("data.table")

  cat("=== Starting Outer Cross-Validation Analysis ===\n")

  # Load label mapping and data
  cat("Loading label mapping and data...\n")
  label_mapping <- safe_read_file("../data/label_mapping_all.csv", read.csv)
  if (is.null(label_mapping)) {
    stop("Failed to load label mapping file")
  }

  # Load leukemia subtype data
  leukemia_subtypes <- safe_read_file("../data/rgas_10feb26.csv", function(f) read.csv(f)$ICC_Subtype)
  if (is.null(leukemia_subtypes)) {
    stop("Failed to load leukemia subtype data")
  }

  # Load study metadata
  study_names <- safe_read_file("../data/meta_20aug25.csv", function(f) read.csv(f)$Studies)
  if (is.null(study_names)) {
    stop("Failed to load study metadata")
  }

  # Filter data based on criteria and keep index mapping to full dataset.
  subtypes_with_sufficient_samples <- names(which(table(leukemia_subtypes) >= DATA_FILTERS$min_samples_per_subtype))
  filter <- which(
    leukemia_subtypes %in% subtypes_with_sufficient_samples &
      !leukemia_subtypes %in% DATA_FILTERS$excluded_subtypes &
      study_names %in% DATA_FILTERS$selected_studies
  )
  filtered_index_map_zero_based <- filter - 1L
  filtered_leukemia_subtypes <- leukemia_subtypes[filter]

  # Load outer CV results for all models
  cat("Loading outer CV results...\n")
  outer_cv_data <- list()

  for (model_name in names(OUTER_MODEL_CONFIGS)) {
    config <- OUTER_MODEL_CONFIGS[[model_name]]
    cat(sprintf("Loading %s outer CV data...\n", toupper(model_name)))

    outer_cv_data[[model_name]] <- list()

    for (fold_type in names(config$file_paths)) {
      file_path <- config$file_paths[[fold_type]]
      if (is.null(file_path)) {
        cat(sprintf("  No %s file found for %s\n", fold_type, toupper(model_name)))
        next
      }
      cat(sprintf("  Discovered: %s\n", file_path))
      results <- load_outer_cv_results(file_path, config$classification_type)

      if (!is.null(results)) {
        outer_cv_data[[model_name]][[fold_type]] <- results
      }
    }
  }

  # Generate probability matrices from outer CV results
  cat("Generating outer CV probability matrices...\n")
  outer_probability_matrices <- list()
  filtering_statistics <- list()  # Store filtering stats for reporting

  for (model_name in names(outer_cv_data)) {
    config <- OUTER_MODEL_CONFIGS[[model_name]]
    cat(sprintf("Processing %s probabilities...\n", toupper(model_name)))

    outer_probability_matrices[[model_name]] <- list()

    for (fold_type in names(outer_cv_data[[model_name]])) {
      results <- outer_cv_data[[model_name]][[fold_type]]

      if (!is.null(results)) {
        # Generate probability matrices (with filtering and optional merging)
        if (config$classification_type == "OvR") {
          result <- generate_outer_ovr_probability_matrices(
            results,
            label_mapping,
            filtered_index_map_zero_based,
            filter_unseen_classes = TRUE,
            merge_classes = merge_classes
          )
        } else {
          result <- generate_outer_standard_probability_matrices(
            results,
            label_mapping,
            filtered_leukemia_subtypes,
            filtered_index_map_zero_based,
            filter_unseen_classes = TRUE,
            merge_classes = merge_classes
          )
        }

        # Extract matrices and filtering stats
        probs <- result$matrices
        if (!is.null(result$filtering_stats)) {
          if (!model_name %in% names(filtering_statistics)) {
            filtering_statistics[[model_name]] <- list()
          }
          result$filtering_stats$model <- model_name
          result$filtering_stats$type <- fold_type
          filtering_statistics[[model_name]][[fold_type]] <- result$filtering_stats
        }

        # No class grouping - keep original classes
        # probs remain unchanged
        outer_probability_matrices[[model_name]][[fold_type]] <- probs
      }
    }
  }

  # Load ensemble weights from inner CV analysis (use correct directory based on merge_classes)
  cat("Loading ensemble weights from inner CV analysis...\n")
  if (merge_classes) {
    weights_base_dir <- WEIGHTS_BASE_DIR_MERGED
  } else {
    weights_base_dir <- WEIGHTS_BASE_DIR_UNMERGED
  }
  cat(sprintf("Using weights directory: %s\n", weights_base_dir))

  ensemble_weights <- list()

  for (type in c("cv", "loso")) {
    weights_data <- tryCatch({
      load_ensemble_weights(weights_base_dir, type)
    }, error = function(e) {
      warning(sprintf("Failed to load ensemble weights for %s: %s", type, e$message))
      NULL
    })

    if (!is.null(weights_data)) {
      ensemble_weights[[type]] <- weights_data
    }
  }

  # Apply ensemble weights to generate ensemble predictions
  cat("Generating ensemble predictions...\n")
  ensemble_matrices <- list()

  for (type in c("cv", "loso")) {
    if (!type %in% names(ensemble_weights)) {
      warning(sprintf("No ensemble weights available for %s", type))
      next
    }

    cat(sprintf("Processing %s ensemble (global product-of-experts)...\n", toupper(type)))
    ensemble_matrices[[type]] <- list()

    global_product_ensemble <- apply_ensemble_weights_to_outer_cv(
      outer_probability_matrices, ensemble_weights[[type]], type
    )
    if (!is.null(global_product_ensemble)) {
      ensemble_matrices[[type]][["global_product_ensemble"]] <- global_product_ensemble
    }

  }

  # Combine individual models and ensemble results for performance calculation
  cat("Combining results for performance analysis...\n")
  all_probability_matrices <- outer_probability_matrices

  for (type in c("cv", "loso")) {
    if (type %in% names(ensemble_matrices)) {
      if ("global_product_ensemble" %in% names(ensemble_matrices[[type]])) {
        if (!"Global_Product_Optimized" %in% names(all_probability_matrices)) {
          all_probability_matrices[["Global_Product_Optimized"]] <- list()
        }
        all_probability_matrices[["Global_Product_Optimized"]][[type]] <- ensemble_matrices[[type]][["global_product_ensemble"]]
      }
    }
  }

  # Calculate performance for all models and ensembles
  cat("Calculating performance metrics...\n")
  detailed_performance <- list()
  performance_summaries <- list()
  per_class_summaries <- list()

  for (type in c("cv", "loso")) {

    # Calculate detailed performance for all models and ensembles
    detailed_performance[[type]] <- calculate_outer_cv_performance(all_probability_matrices, type)

    # Generate performance summary
    performance_summaries[[type]] <- summarize_outer_cv_performance(detailed_performance[[type]])

    # Generate per-class performance summary
    per_class_summaries[[type]] <- summarize_per_class_performance(detailed_performance[[type]])
  }

  # Snapshot before augmentation (same rows as all_probability_matrices).
  all_probability_matrices_raw <- all_probability_matrices

  # -------------------------------------------------------------------------
  # Left-out-aware analysis (optional, requires leftout prediction CSVs)
  # -------------------------------------------------------------------------

  # Check if any leftout file paths are configured
  has_leftout_data <- any(sapply(LEFTOUT_MODEL_CONFIGS, function(cfg) {
    any(sapply(cfg$file_paths, function(p) !is.null(p) && file.exists(p)))
  }))

  if (has_leftout_data) {
    cat("\n=== Loading left-out predictions for augmented analysis ===\n")

    leftout_probability_matrices <- list()

    for (model_name in names(LEFTOUT_MODEL_CONFIGS)) {
      lo_config <- LEFTOUT_MODEL_CONFIGS[[model_name]]
      leftout_probability_matrices[[model_name]] <- list()

      for (fold_type in names(lo_config$file_paths)) {
        lo_path <- lo_config$file_paths[[fold_type]]
        if (is.null(lo_path) || !file.exists(lo_path)) next

        cat(sprintf("  Loading %s leftout data (%s): %s\n", toupper(model_name), fold_type, lo_path))
        lo_results <- load_outer_cv_results(lo_path, lo_config$classification_type)
        if (is.null(lo_results)) next

        if (lo_config$classification_type == "OvR") {
          lo_probs <- generate_leftout_ovr_probability_matrices(
            lo_results, label_mapping, leukemia_subtypes, merge_classes = merge_classes
          )
        } else {
          lo_probs <- generate_leftout_standard_probability_matrices(
            lo_results, label_mapping, leukemia_subtypes, merge_classes = merge_classes
          )
        }

        leftout_probability_matrices[[model_name]][[fold_type]] <- lo_probs$matrices
      }
    }

    # Build augmented matrices (raw known-class + leftout) per model and type
    cat("Building augmented probability matrices (known + left-out)...\n")
    all_augmented_matrices <- list()

    for (type in c("cv", "loso")) {
      # Per-model augmentation
      for (model_name in names(all_probability_matrices_raw)) {
        if (is.null(all_augmented_matrices[[model_name]])) {
          all_augmented_matrices[[model_name]] <- list()
        }

        known_folds <- all_probability_matrices_raw[[model_name]][[type]]
        lo_folds <- leftout_probability_matrices[[model_name]][[type]]

        if (!is.null(known_folds) && !is.null(lo_folds)) {
          all_augmented_matrices[[model_name]][[type]] <- augment_fold_matrices_with_leftout(
            known_folds, lo_folds
          )
          n_lo <- sum(sapply(lo_folds, nrow))
          cat(sprintf("  %s (%s): augmented with %d left-out samples\n", model_name, type, n_lo))
        } else {
          all_augmented_matrices[[model_name]][[type]] <- known_folds
        }
      }

      # Augmented product ensemble only
      if (type %in% names(ensemble_weights)) {
        lo_per_model <- list()
        for (mn in c("svm", "xgboost", "neural_net")) {
          lo_per_model[[mn]] <- leftout_probability_matrices[[mn]][[type]]
        }
        has_all_lo <- all(sapply(lo_per_model, function(x) !is.null(x) && length(x) > 0))
        ens_model <- "Global_Product_Optimized"
        known_ens <- all_probability_matrices_raw[[ens_model]][[type]]
        if (!is.null(known_ens)) {
          if (is.null(all_augmented_matrices[[ens_model]])) {
            all_augmented_matrices[[ens_model]] <- list()
          }
          if (has_all_lo) {
            all_augmented_matrices[[ens_model]][[type]] <- build_augmented_ensemble(
              known_ens, lo_per_model, ensemble_weights[[type]], type
            )
          } else {
            all_augmented_matrices[[ens_model]][[type]] <- known_ens
          }
        }
      }
    }

  } else {
    cat("\nNo left-out prediction files configured. Skipping augmented analysis.\n")
    cat("To enable: run run_outer_cv.py --include_leftout, then set paths in LEFTOUT_MODEL_CONFIGS.\n")
  }

  # -------------------------------------------------------------------------
  # Fold bundles for R/calibration_reject_models.R (SCENARIO_KEY with_leftout_ood_aware):
  # augmented Global_Product_Optimized folds + disagreement / ROI-style columns.
  # No confidence_multivariate: nested script uses get_rejection_features_from_matrix() only.
  # -------------------------------------------------------------------------
  CALIBRATION_MV_BASE_MODEL <- "Global_Product_Optimized"
  CALIBRATION_MV_MODEL_LABEL <- "Global_Product_Optimized_augmented_disagreement_folds"

  multivariate_results <- list(with_leftout_ood_aware = list())
  if (!has_leftout_data || !exists("all_augmented_matrices")) {
    cat("\nSkipping with_leftout_ood_aware fold bundles (no left-out matrices).\n")
  } else if (!CALIBRATION_MV_BASE_MODEL %in% names(all_augmented_matrices)) {
    cat("\nSkipping with_leftout_ood_aware fold bundles (", CALIBRATION_MV_BASE_MODEL, " missing).\n", sep = "")
  } else {
    cat(
      "\n=== Augmented ensemble fold bundles (",
      CALIBRATION_MV_BASE_MODEL,
      "; for R/calibration_reject_models.R) ===\n",
      sep = ""
    )
    multivariate_results$with_leftout_ood_aware[[CALIBRATION_MV_BASE_MODEL]] <- list()
    for (type in c("cv", "loso")) {
      if (!type %in% names(all_augmented_matrices[[CALIBRATION_MV_BASE_MODEL]])) next
      ens_folds <- copy_fold_matrix_list(all_augmented_matrices[[CALIBRATION_MV_BASE_MODEL]][[type]])
      if (!is.list(ens_folds) || length(ens_folds) < 2L) {
        multivariate_results$with_leftout_ood_aware[[CALIBRATION_MV_BASE_MODEL]][[type]] <- NULL
        next
      }
      cat(sprintf("  %s (%s): disagreement features on augmented folds...\n", toupper(type), CALIBRATION_MV_BASE_MODEL))
      ens_folds <- compute_disagreement_features(ens_folds, all_augmented_matrices, type)
      multivariate_results$with_leftout_ood_aware[[CALIBRATION_MV_BASE_MODEL]][[type]] <- list(
        fold_matrices = ens_folds,
        model_label = CALIBRATION_MV_MODEL_LABEL
      )
    }
  }

  # Print summary of filtering statistics
  cat("\n=== Sample Filtering Summary ===\n")
  if (length(filtering_statistics) > 0) {
    total_filtered <- 0
    total_samples <- 0
    for (model_name in names(filtering_statistics)) {
      for (fold_type in names(filtering_statistics[[model_name]])) {
        stats <- filtering_statistics[[model_name]][[fold_type]]
        model_total_filtered <- sum(stats$n_filtered)
        model_total_samples <- sum(stats$n_total)
        total_filtered <- total_filtered + model_total_filtered
        total_samples <- total_samples + model_total_samples
        cat(sprintf("%s (%s): Filtered %d/%d samples (%.1f%%)\n",
                    toupper(model_name), toupper(fold_type),
                    model_total_filtered, model_total_samples,
                    100 * model_total_filtered / model_total_samples))
      }
    }
    cat(sprintf("OVERALL: Filtered %d/%d samples (%.1f%%)\n",
                total_filtered, total_samples,
                100 * total_filtered / total_samples))
  } else {
    cat("No samples were filtered (all test samples had classes in training)\n")
  }
  cat("================================\n\n")

  # Save all results
  outer_cv_results <- list(
    outer_cv_data = outer_cv_data,
    outer_probability_matrices = outer_probability_matrices,
    filtering_statistics = filtering_statistics,
    ensemble_matrices = ensemble_matrices,
    detailed_performance = detailed_performance,
    performance_summaries = performance_summaries,
    per_class_summaries = per_class_summaries,
    ensemble_weights_used = ensemble_weights,
    multivariate_results = multivariate_results
  )

  # Determine suffix for file paths (maxprob method - uses max probability instead of summing)
  if (!merge_classes) {
    merge_suffix <- "_unmerged_maxprob"
  } else {
    merge_suffix <- "_merged_summed"
  }
  outer_cv_results$merge_classes <- merge_classes  # Store merge status in results

  # One clear output root for everything produced by this script
  analysis_output_dir <- file.path(
    "../data/out/outer_cv",
    paste0("outer_cv_analysis_outputs", merge_suffix)
  )
  dir.create(analysis_output_dir, recursive = TRUE, showWarnings = FALSE)

  # Manifest only; nested calibration reads outer_cv_results.rds (no per-regime RDS sidecars).
  manifest_rows <- list(
    data.frame(key = "timestamp_utc", value = format(Sys.time(), tz = "UTC", usetz = TRUE), stringsAsFactors = FALSE),
    data.frame(key = "merge_suffix", value = merge_suffix, stringsAsFactors = FALSE),
    data.frame(key = "merge_classes", value = as.character(merge_classes), stringsAsFactors = FALSE),
    data.frame(key = "has_leftout_data", value = as.character(has_leftout_data), stringsAsFactors = FALSE),
    data.frame(key = "global_ensemble_model", value = "Global_Product_Optimized (product-of-experts)", stringsAsFactors = FALSE),
    data.frame(
      key = "multivariate_results",
      value = "with_leftout_ood_aware$Global_Product_Optimized: augmented folds + disagreement (R/calibration_reject_models.R)",
      stringsAsFactors = FALSE
    ),
    data.frame(
      key = "regimes",
      value = "with_leftout_ood_aware",
      stringsAsFactors = FALSE
    )
  )
  write.csv(
    do.call(rbind, manifest_rows),
    file.path(analysis_output_dir, "analysis_manifest.csv"),
    row.names = FALSE
  )

  saveRDS(outer_cv_results, file.path(analysis_output_dir, "outer_cv_results.rds"))

  return(outer_cv_results)
}

outer_cv_results_unmerged <- main_outer_cv(merge_classes = FALSE)
outer_cv_results_merged <- main_outer_cv(merge_classes = TRUE)
