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
      cv   = find_latest_csv(OUTER_CV_DIRS$svm, "^SVM_outer_cv_CV_OvR_fs_eta_\\d+_\\d+\\.csv$"),
      loso = find_latest_csv(OUTER_CV_DIRS$svm, "^SVM_outer_cv_loso_OvR_fs_eta_\\d+_\\d+\\.csv$")
    )
  ),
  xgboost = list(
    classification_type = "OvR",
    file_paths = list(
      cv   = find_latest_csv(OUTER_CV_DIRS$xgboost, "^XGBOOST_outer_cv_CV_OvR_fs_eta_\\d+_\\d+\\.csv$"),
      loso = find_latest_csv(OUTER_CV_DIRS$xgboost, "^XGBOOST_outer_cv_loso_OvR_fs_eta_\\d+_\\d+\\.csv$")
    )
  ),
  neural_net = list(
    classification_type = "standard",
    file_paths = list(
      cv   = find_latest_csv(OUTER_CV_DIRS$nn, "^NN_outer_cv_CV_standard_fs_eta_\\d+_\\d+\\.csv$"),
      loso = find_latest_csv(OUTER_CV_DIRS$nn, "^NN_outer_cv_loso_standard_fs_eta_\\d+_\\d+\\.csv$")
    )
  )
)

# Auto-discover left-out prediction files (generated with --include_leftout)
LEFTOUT_MODEL_CONFIGS <- list(
  svm = list(
    classification_type = "OvR",
    file_paths = list(
      cv   = find_latest_csv(OUTER_CV_DIRS$svm, "^SVM_outer_cv_CV_OvR_leftout_fs_eta_\\d+_\\d+\\.csv$"),
      loso = find_latest_csv(OUTER_CV_DIRS$svm, "^SVM_outer_cv_loso_OvR_leftout_fs_eta_\\d+_\\d+\\.csv$")
    )
  ),
  xgboost = list(
    classification_type = "OvR",
    file_paths = list(
      cv   = find_latest_csv(OUTER_CV_DIRS$xgboost, "^XGBOOST_outer_cv_CV_OvR_leftout_fs_eta_\\d+_\\d+\\.csv$"),
      loso = find_latest_csv(OUTER_CV_DIRS$xgboost, "^XGBOOST_outer_cv_loso_OvR_leftout_fs_eta_\\d+_\\d+\\.csv$")
    )
  ),
  neural_net = list(
    classification_type = "standard",
    file_paths = list(
      cv   = find_latest_csv(OUTER_CV_DIRS$nn, "^NN_outer_cv_CV_standard_leftout_fs_eta_\\d+_\\d+\\.csv$"),
      loso = find_latest_csv(OUTER_CV_DIRS$nn, "^NN_outer_cv_loso_standard_leftout_fs_eta_\\d+_\\d+\\.csv$")
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

# NOTE: Rejection cutoffs are now computed via leave-one-fold-out cross-fitting
# directly on outer CV results, rather than loaded from inner CV.
# Set to FALSE to inspect monotonic risk-coverage curves without cutoff selection.
ENABLE_CUTOFF_SELECTION <- FALSE

# =============================================================================
# Source Utility Functions
# =============================================================================

source("utility_functions.R")

# =============================================================================
# Outer CV Specific Functions
# =============================================================================

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
generate_outer_ovr_probability_matrices <- function(outer_cv_results, label_mapping, filter_unseen_classes = TRUE, merge_classes = FALSE) {
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
    sample_indices <- parse_numeric_string(first_row$sample_indices)
    if (length(sample_indices) == num_samples) {
      probability_matrix$sample_indices <- sample_indices
    }

    # Apply class merging if requested (before filtering)
    if (merge_classes) {
      probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_prob_method = "sum")
      # Update class_labels after merging for filtering
      class_labels <- colnames(probability_matrix)[!colnames(probability_matrix) %in%
                                                    c("y", "outer_fold", "sample_indices")]
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
generate_outer_standard_probability_matrices <- function(outer_cv_results, label_mapping, filtered_subtypes, filter_unseen_classes = TRUE, merge_classes = FALSE) {
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
    sample_indices <- parse_numeric_string(fold_row$sample_indices)
    num_samples <- length(sample_indices)

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
    probability_matrix$y <- make.names(filtered_subtypes[sample_indices + 1])
    probability_matrix$outer_fold <- outer_fold_id
    probability_matrix$sample_indices <- sample_indices

    # Apply class merging if requested (before filtering)
    if (merge_classes) {
      probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_prob_method = "sum")
      # Update class_labels after merging for filtering
      class_labels <- colnames(probability_matrix)[!colnames(probability_matrix) %in%
                                                    c("y", "outer_fold", "sample_indices")]
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

#' Apply ensemble weights from inner CV to outer CV probability matrices
#' @param outer_prob_matrices Outer CV probability matrices for all models
#' @param ensemble_weights_data Ensemble weights from inner CV analysis
#' @param type Type of analysis ("cv" or "loso")
#' @param ensemble_method Method to use ("ovr" or "global")
#' @return List of ensemble probability matrices
apply_ensemble_weights_to_outer_cv <- function(outer_prob_matrices, ensemble_weights_data, type = "cv", ensemble_method = "ovr") {
  cat(sprintf("Applying %s ensemble weights to outer CV results...\n", ensemble_method))

  # Get the appropriate weights
  weights_to_use <- if (ensemble_method == "ovr") {
    ensemble_weights_data$ovr_weights
  } else {
    ensemble_weights_data$global_weights
  }

  if (is.null(weights_to_use)) {
    warning(sprintf("No %s weights available for %s analysis", ensemble_method, type))
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
    svm_probs <- svm_matrix[, !colnames(svm_matrix) %in% c("y", "outer_fold", "sample_indices"), drop = FALSE]
    xgb_probs <- xgb_matrix[, !colnames(xgb_matrix) %in% c("y", "outer_fold", "sample_indices"), drop = FALSE]
    nn_probs <- nn_matrix[, !colnames(nn_matrix) %in% c("y", "outer_fold", "sample_indices"), drop = FALSE]

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

    # Apply ensemble weights
    if (ensemble_method == "ovr") {
      # Use class-specific weights
      fold_weights <- weights_to_use[[fold_name]]
      if (is.null(fold_weights)) {
        warning(sprintf("No OvR weights for fold %s, using DNN-only fallback", fold_name))
        fold_weights <- list()
        for (class_name in all_classes) {
          fold_weights[[gsub("Class.", "", class_name)]] <- list(weights = list(SVM = 0, XGB = 0, NN = 1))
        }
      }

      # Initialize ensemble matrix
      ensemble_matrix <- matrix(0, nrow = nrow(svm_probs), ncol = length(all_classes))
      colnames(ensemble_matrix) <- all_classes

      # Apply class-specific weights
      for (class_name in all_classes) {
        clean_class_name <- gsub("Class.", "", class_name)
        clean_class_name_no_dots <- gsub("\\.", "", clean_class_name)

        # Find weights for this class
        class_weights <- NULL
        if (clean_class_name %in% names(fold_weights)) {
          class_weights <- fold_weights[[clean_class_name]]$weights
        } else if (clean_class_name_no_dots %in% names(fold_weights)) {
          class_weights <- fold_weights[[clean_class_name_no_dots]]$weights
        } else {
          # Use DNN-only as fallback
          class_weights <- list(SVM = 0, XGB = 0, NN = 1)
        }

        # Calculate weighted ensemble for this class
        # Ensure weights are numeric and handle any NA values
        svm_weight <- ifelse(is.null(class_weights$SVM) || is.na(class_weights$SVM), 1, as.numeric(class_weights$SVM))
        xgb_weight <- ifelse(is.null(class_weights$XGB) || is.na(class_weights$XGB), 1, as.numeric(class_weights$XGB))
        nn_weight <- ifelse(is.null(class_weights$NN) || is.na(class_weights$NN), 1, as.numeric(class_weights$NN))

        ensemble_matrix[, class_name] <-
          svm_probs[[class_name]] * svm_weight +
          xgb_probs[[class_name]] * xgb_weight +
          nn_probs[[class_name]] * nn_weight
      }

    } else {
      # Use global weights
      fold_weights <- weights_to_use[[fold_name]]
      if (is.null(fold_weights)) {
        warning(sprintf("No global weights for fold %s, using DNN-only fallback", fold_name))
        fold_weights <- list(weights = list(SVM = 0, XGB = 0, NN = 1))
      }

      weights <- fold_weights$weights

      # Ensure weights are numeric and handle any NA values
      svm_weight <- ifelse(is.null(weights$SVM) || is.na(weights$SVM), 1, as.numeric(weights$SVM))
      xgb_weight <- ifelse(is.null(weights$XGB) || is.na(weights$XGB), 1, as.numeric(weights$XGB))
      nn_weight <- ifelse(is.null(weights$NN) || is.na(weights$NN), 1, as.numeric(weights$NN))

      # Calculate weighted ensemble
      ensemble_matrix <- svm_probs * svm_weight +
                        xgb_probs * xgb_weight +
                        nn_probs * nn_weight
    }

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
      prob_cols <- prob_matrix[, !colnames(prob_matrix) %in% c("y", "outer_fold", "sample_indices", "study"), drop = FALSE]

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
# Rejection Analysis Functions
# =============================================================================

#' Load optimal cutoffs from inner CV analysis
#' @param rejection_base_dir Base directory containing rejection analysis results
#' @param type Type of analysis ("cv" or "loso")
#' @return List containing optimal cutoffs and summary statistics
load_optimal_cutoffs <- function(rejection_base_dir, type = "cv") {
  cat(sprintf("Loading optimal cutoffs for %s analysis...\n", toupper(type)))

  # Load cutoffs from the main directory
  optimal_cutoffs_file <- file.path(rejection_base_dir, "cutoffs.csv")

  if (!file.exists(optimal_cutoffs_file)) {
    warning(sprintf("Cutoffs file not found: %s", optimal_cutoffs_file))
    return(NULL)
  }

  # Load all cutoffs and filter by type
  all_cutoffs <- safe_read_file(optimal_cutoffs_file, read.csv)
  if (is.null(all_cutoffs)) {
    warning("Failed to load cutoffs file")
    return(NULL)
  }

  # Filter cutoffs by type
  optimal_cutoffs <- all_cutoffs[all_cutoffs$source == type, ]

  if (nrow(optimal_cutoffs) == 0) {
    warning(sprintf("No cutoffs found for type: %s", type))
    return(NULL)
  }

  cat(sprintf("  Loaded %d cutoffs for %s analysis from: %s\n", nrow(optimal_cutoffs), toupper(type), optimal_cutoffs_file))

  return(list(
    optimal_cutoffs = optimal_cutoffs
  ))
}

#' Apply rejection analysis to outer CV probability matrices using inner CV cutoffs
#' @param probability_matrices Probability matrices for all models and ensembles
#' @param optimal_cutoffs Optimal cutoffs from inner CV analysis
#' @param type Type of analysis ("cv" or "loso")
#' @return List of rejection analysis results
apply_rejection_analysis_to_outer_cv <- function(probability_matrices, optimal_cutoffs, type = "cv") {
  cat(sprintf("Applying rejection analysis to outer CV results for %s...\n", toupper(type)))

  if (is.null(optimal_cutoffs) || is.null(optimal_cutoffs$optimal_cutoffs)) {
    warning("No optimal cutoffs available for rejection analysis")
    return(NULL)
  }

  rejection_results <- list()

  # Get unique models from optimal cutoffs
  models_with_cutoffs <- unique(optimal_cutoffs$optimal_cutoffs$model)

  for (model_name in models_with_cutoffs) {
    cat(sprintf("  Processing %s...\n", model_name))

    # Use mean cutoff across folds
    # mean_cutoff <- mean(model_cutoffs$prob_cutoff, na.rm = TRUE)
    # cat(sprintf("    Using cutoff %.3f for %s\n", mean_cutoff, model_name))

    # Find corresponding probability matrix
    if (model_name %in% names(probability_matrices) && type %in% names(probability_matrices[[model_name]])) {
      model_matrices <- probability_matrices[[model_name]][[type]]

      for (fold_name in names(model_matrices)) {
        prob_matrix <- model_matrices[[fold_name]]

        # Get cutoff for this model (use mean across inner folds)
        mean_cutoff <- optimal_cutoffs$optimal_cutoffs[
          optimal_cutoffs$optimal_cutoffs$model == model_name &
            optimal_cutoffs$optimal_cutoffs$source == type &
            optimal_cutoffs$optimal_cutoffs$outer_fold == fold_name, "mean_cutoff"
          ]
        if (length(mean_cutoff) == 0) {
          cat(sprintf("    No cutoffs found for %s, skipping\n", model_name))
          next
        }

        if (!is.null(prob_matrix) && nrow(prob_matrix) > 0) {
          # Apply rejection analysis
          rejection_result <- evaluate_single_matrix_with_rejection_and_cutoff(
            prob_matrix, fold_name, model_name, type, mean_cutoff
          )

          if (!is.null(rejection_result)) {
            rejection_results[[paste(model_name, fold_name, sep = "_")]] <- rejection_result
          }
        }
      }
    } else if (model_name == "Global_Optimized") {
      # Ensemble method (Global only; OvR removed)
      ensemble_found <- FALSE
      if ("Global_Optimized" %in% names(probability_matrices) && type %in% names(probability_matrices[["Global_Optimized"]])) {
        ensemble_matrices <- probability_matrices[["Global_Optimized"]][[type]]

        for (fold_name in names(ensemble_matrices)) {
          prob_matrix <- ensemble_matrices[[fold_name]]
          mean_cutoff <- optimal_cutoffs$optimal_cutoffs[
            optimal_cutoffs$optimal_cutoffs$model == "Global_Optimized" &
              optimal_cutoffs$optimal_cutoffs$source == type &
              optimal_cutoffs$optimal_cutoffs$outer_fold == fold_name, "mean_cutoff"
          ]
          if (length(mean_cutoff) > 0 && !is.null(prob_matrix) && nrow(prob_matrix) > 0) {
            rejection_result <- evaluate_single_matrix_with_rejection_and_cutoff(
              prob_matrix, fold_name, "Global_Optimized", type, mean_cutoff[1]
            )
            if (!is.null(rejection_result)) {
              rejection_results[[paste("Global_Optimized", fold_name, sep = "_")]] <- rejection_result
            }
            ensemble_found <- TRUE
          }
        }
      }

      if (!ensemble_found) {
        cat(sprintf("    No probability matrices found for %s, skipping\n", model_name))
      }
    } else {
      cat(sprintf("    Skipping %s (no longer used)\n", model_name))
    }
  }

  return(rejection_results)
}

#' Evaluate single matrix with rejection using specific cutoff
#' @param prob_matrix Probability matrix with class probabilities and true labels
#' @param fold_name Name of the fold being analyzed
#' @param model_name Name of the model being analyzed
#' @param type Type of analysis ("cv" or "loso")
#' @param cutoff Probability cutoff to apply
#' @return List with rejection analysis results and per-class metrics
evaluate_single_matrix_with_rejection_and_cutoff <- function(prob_matrix, fold_name, model_name, type, cutoff) {
  # Exclude meta columns and optional Platt-calibrated confidence from prob columns
  meta_cols <- c(
    "y", "outer_fold", "sample_indices", "confidence_calibrated", "confidence_multivariate",
    "is_leftout", "n_models_agree", "mean_js_convergence", "top1_prob_variance_across_models"
  )
  prob_matrix_clean <- prob_matrix[, !colnames(prob_matrix) %in% meta_cols, drop = FALSE]

  truth <- prob_matrix$y
  truth <- gsub("Class.", "", truth)

  # Predictions from class probabilities
  pred_indices <- apply(prob_matrix_clean, 1, which.max)
  preds <- colnames(prob_matrix_clean)[pred_indices]
  preds <- gsub("Class.", "", preds)

  # Use Platt-calibrated confidence for rejection if present, else max probability
  if ("confidence_calibrated" %in% colnames(prob_matrix)) {
    confidence_vals <- prob_matrix$confidence_calibrated
  } else {
    confidence_vals <- apply(prob_matrix_clean, 1, max)
  }

  all_classes <- unique(c(truth, preds))
  truth <- factor(truth, levels = all_classes)
  preds <- factor(preds, levels = all_classes)

  # Apply rejection using the specific cutoff
  rejected_indices <- which(confidence_vals < cutoff)
  accepted_indices <- which(confidence_vals >= cutoff)

  # Calculate metrics for accepted samples only
  if (length(accepted_indices) == 0) {
    # If all samples are rejected, return NULL
    return(NULL)
  }

  accepted_truth <- truth[accepted_indices]
  accepted_preds <- preds[accepted_indices]

  # Calculate confusion matrix and metrics for accepted samples
  cm <- caret::confusionMatrix(accepted_preds, accepted_truth)
  kappa <- as.numeric(cm$overall["Kappa"])
  accuracy <- as.numeric(cm$overall["Accuracy"])

  # Extract per-class metrics for accepted samples
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

  # Calculate metrics for rejected samples (if any)
  rejected_accuracy <- NA
  if (length(rejected_indices) > 0) {
    rejected_truth <- truth[rejected_indices]
    rejected_preds <- preds[rejected_indices]
    rejected_accuracy <- sum(rejected_truth == rejected_preds) / length(rejected_indices)
  }

  # Return results as a list
  list(
    summary = data.frame(
      model = model_name,
      type = type,
      fold = fold_name,
      prob_cutoff = cutoff,
      kappa = kappa,
      accuracy = accuracy,
      n_accepted = length(accepted_indices),
      n_rejected = length(rejected_indices),
      perc_rejected = length(rejected_indices) / nrow(prob_matrix),
      rejected_accuracy = rejected_accuracy,
      total_samples = nrow(prob_matrix),
      stringsAsFactors = FALSE
    ),
    per_class_metrics = per_class_metrics
  )
}

#' Generate rejection analysis summary for outer CV
#' @param rejection_results Rejection analysis results
#' @param type Type of analysis ("cv" or "loso")
#' @return Data frame with summary statistics
summarize_rejection_analysis <- function(rejection_results, type = "cv") {
  cat(sprintf("Summarizing rejection analysis for %s...\n", toupper(type)))

  if (is.null(rejection_results) || length(rejection_results) == 0) {
    return(NULL)
  }

  # Extract summary results
  all_summaries <- lapply(rejection_results, function(x) x$summary)
  all_results <- do.call(rbind, all_summaries)

  if (nrow(all_results) == 0) {
    return(NULL)
  }

  # Calculate summary statistics across folds for each model
  summary_stats <- all_results %>%
    group_by(model) %>%
    summarise(
      mean_cutoff = mean(prob_cutoff, na.rm = TRUE),
      sd_cutoff = sd(prob_cutoff, na.rm = TRUE),
      mean_kappa = mean(kappa, na.rm = TRUE),
      sd_kappa = sd(kappa, na.rm = TRUE),
      mean_accuracy = mean(accuracy, na.rm = TRUE),
      sd_accuracy = sd(accuracy, na.rm = TRUE),
      mean_perc_rejected = mean(perc_rejected, na.rm = TRUE),
      sd_perc_rejected = sd(perc_rejected, na.rm = TRUE),
      mean_n_accepted = mean(n_accepted, na.rm = TRUE),
      mean_n_rejected = mean(n_rejected, na.rm = TRUE),
      n_folds = n(),
      .groups = "drop"
    )

  # Extract and summarize per-class metrics
  per_class_summary <- data.frame()

  for (model_name in unique(all_results$model)) {
    model_results <- rejection_results[grepl(paste0("^", model_name, "_"), names(rejection_results))]

    # Get all unique classes across all folds for this model
    all_classes <- unique(unlist(lapply(model_results, function(x) names(x$per_class_metrics))))

    for (class_name in all_classes) {
      # Extract metrics for this class across all folds
      sensitivities <- numeric(0)
      specificities <- numeric(0)
      precisions <- numeric(0)
      recalls <- numeric(0)
      f1_scores <- numeric(0)
      balanced_accuracies <- numeric(0)

      for (fold_result in model_results) {
        if (!is.null(fold_result$per_class_metrics) && class_name %in% names(fold_result$per_class_metrics)) {
          class_metrics <- fold_result$per_class_metrics[[class_name]]
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
          Type = type,
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
  if (nrow(per_class_summary) > 0) {
    per_class_summary <- per_class_summary[order(per_class_summary$Model, -per_class_summary$Mean_F1), ]
  }

  return(list(
    detailed_results = all_results,
    summary_stats = summary_stats,
    per_class_summary = per_class_summary
  ))
}


#' Compare performance with and without rejection analysis
#' @param detailed_performance Performance results without rejection
#' @param rejection_summary Rejection analysis summary
#' @param type Type of analysis ("cv" or "loso")
#' @return Data frame with performance comparison
compare_performance_with_rejection <- function(detailed_performance, rejection_summary, type = "cv") {
  cat(sprintf("Comparing performance with and without rejection for %s...\n", toupper(type)))

  if (is.null(rejection_summary) || is.null(detailed_performance)) {
    warning("Missing data for performance comparison")
    return(NULL)
  }

  comparison_results <- data.frame()

  # Get models that have both performance data and rejection analysis
  models_with_rejection <- unique(rejection_summary$detailed_results$model)

  for (model_name in models_with_rejection) {
    # Get rejection performance for this model
    model_rejection <- rejection_summary$detailed_results[rejection_summary$detailed_results$model == model_name, ]

    if (nrow(model_rejection) == 0) next

    # Calculate mean rejection performance across folds
    mean_rejection_kappa <- mean(model_rejection$kappa, na.rm = TRUE)
    mean_rejection_accuracy <- mean(model_rejection$accuracy, na.rm = TRUE)
    mean_perc_rejected <- mean(model_rejection$perc_rejected, na.rm = TRUE)

    # Get original performance for this model (if available)
    original_kappa <- NA
    original_accuracy <- NA

    if (model_name %in% names(detailed_performance)) {
      model_perf <- detailed_performance[[model_name]]
      if (length(model_perf) > 0) {
        # Calculate mean original performance across folds
        kappas <- sapply(model_perf, function(x) x$kappa)
        accuracies <- sapply(model_perf, function(x) x$accuracy)
        original_kappa <- mean(kappas, na.rm = TRUE)
        original_accuracy <- mean(accuracies, na.rm = TRUE)
      }
    }

    # Create comparison row
    comparison_row <- data.frame(
      Model = model_name,
      Type = type,
      Original_Kappa = original_kappa,
      Rejection_Kappa = mean_rejection_kappa,
      Kappa_Improvement = mean_rejection_kappa - original_kappa,
      Original_Accuracy = original_accuracy,
      Rejection_Accuracy = mean_rejection_accuracy,
      Accuracy_Improvement = mean_rejection_accuracy - original_accuracy,
      Mean_Percent_Rejected = mean_perc_rejected * 100,
      stringsAsFactors = FALSE
    )

    comparison_results <- rbind(comparison_results, comparison_row)
  }

  # Sort by kappa improvement (descending)
  if (nrow(comparison_results) > 0) {
    comparison_results <- comparison_results[order(comparison_results$Rejection_Kappa, decreasing = TRUE), ]
  }

  return(comparison_results)
}



# =============================================================================
# Ensemble-Specific: Model Disagreement Features
# =============================================================================

#' Compute model disagreement features and attach to ensemble probability matrices.
#' For each sample, calculates how many of {SVM, XGB, NN} agree on the top-1,
#' mean Jensen-Shannon convergence, and variance of model top-1 probabilities.
#' @param ensemble_fold_matrices Named list of fold -> ensemble data.frame
#' @param per_model_matrices Named list: model_name -> list(type -> list(fold -> data.frame))
#' @param type "cv" or "loso"
#' @return Modified ensemble_fold_matrices with n_models_agree, mean_js_convergence,
#' and top1_prob_variance_across_models columns
compute_disagreement_features <- function(ensemble_fold_matrices, per_model_matrices, type) {
  cat("Computing model disagreement features for ensemble...\n")

  meta_cols <- c("y", "inner_fold", "outer_fold", "indices", "study",
                 "sample_indices", "confidence_calibrated", "confidence_multivariate",
                 "is_leftout", "n_models_agree", "mean_js_convergence",
                 "top1_prob_variance_across_models")

  for (fold_name in names(ensemble_fold_matrices)) {
    ens <- ensemble_fold_matrices[[fold_name]]
    if (is.null(ens) || nrow(ens) == 0) next

    # Get aligned per-model matrices for this fold
    svm_mat <- per_model_matrices[["svm"]][[type]][[fold_name]]
    xgb_mat <- per_model_matrices[["xgboost"]][[type]][[fold_name]]
    nn_mat  <- per_model_matrices[["neural_net"]][[type]][[fold_name]]

    if (is.null(svm_mat) || is.null(xgb_mat) || is.null(nn_mat)) {
      ens$n_models_agree <- NA
      ens$mean_js_convergence <- NA_real_
      ens$top1_prob_variance_across_models <- NA_real_
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
        ens$mean_js_convergence <- NA_real_
        ens$top1_prob_variance_across_models <- NA_real_
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
        ens$mean_js_convergence <- NA_real_
        ens$top1_prob_variance_across_models <- NA_real_
        ensemble_fold_matrices[[fold_name]] <- ens
        next
      }
      ens <- ens[seq_len(n_min), , drop = FALSE]
      svm_mat <- svm_mat[seq_len(n_min), , drop = FALSE]
      xgb_mat <- xgb_mat[seq_len(n_min), , drop = FALSE]
      nn_mat  <- nn_mat[seq_len(n_min), , drop = FALSE]
    }

    # Extract probability-only columns
    svm_prob_cols <- colnames(svm_mat)[!colnames(svm_mat) %in% meta_cols]
    xgb_prob_cols <- colnames(xgb_mat)[!colnames(xgb_mat) %in% meta_cols]
    nn_prob_cols  <- colnames(nn_mat)[!colnames(nn_mat) %in% meta_cols]
    all_prob_cols <- unique(c(svm_prob_cols, xgb_prob_cols, nn_prob_cols))

    # Build aligned probability matrices (fill 0 for missing columns)
    make_aligned <- function(mat, prob_cols) {
      m <- matrix(0, nrow = nrow(mat), ncol = length(all_prob_cols))
      colnames(m) <- all_prob_cols
      for (col in intersect(prob_cols, all_prob_cols)) {
        m[, col] <- as.numeric(mat[[col]])
      }
      # Normalise rows
      rs <- rowSums(m)
      rs[rs == 0] <- 1
      m / rs
    }

    svm_p <- make_aligned(svm_mat, svm_prob_cols)
    xgb_p <- make_aligned(xgb_mat, xgb_prob_cols)
    nn_p  <- make_aligned(nn_mat, nn_prob_cols)

    # Final safety guard: enforce identical row counts before row-wise indexing.
    n_eff <- min(nrow(ens), nrow(svm_p), nrow(xgb_p), nrow(nn_p))
    if (n_eff == 0) {
      ens$n_models_agree <- NA
      ens$mean_js_convergence <- NA_real_
      ens$top1_prob_variance_across_models <- NA_real_
      ensemble_fold_matrices[[fold_name]] <- ens
      next
    }
    if (nrow(ens) != n_eff || nrow(svm_p) != n_eff || nrow(xgb_p) != n_eff || nrow(nn_p) != n_eff) {
      ens <- ens[seq_len(n_eff), , drop = FALSE]
      svm_p <- svm_p[seq_len(n_eff), , drop = FALSE]
      xgb_p <- xgb_p[seq_len(n_eff), , drop = FALSE]
      nn_p <- nn_p[seq_len(n_eff), , drop = FALSE]
    }

    # Final safety guard: enforce identical row counts before row-wise indexing.
    n_eff <- min(nrow(ens), nrow(svm_p), nrow(xgb_p), nrow(nn_p))
    if (n_eff == 0) {
      ens$n_models_agree <- NA
      ens$mean_js_convergence <- NA_real_
      ens$top1_prob_variance_across_models <- NA_real_
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
    mean_js_convergence <- numeric(n)

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
      n_agree[i] <- max(table(preds_3))

      # Mean Jensen-Shannon divergence across the 3 model distributions.
      # Convert to a convergence score in [0, 1] where higher means more agreement.
      p1 <- pmax(as.numeric(svm_p[i, ]), 1e-12)
      p2 <- pmax(as.numeric(xgb_p[i, ]), 1e-12)
      p3 <- pmax(as.numeric(nn_p[i, ]), 1e-12)
      m <- (p1 + p2 + p3) / 3
      kl1 <- sum(p1 * log(p1 / m))
      kl2 <- sum(p2 * log(p2 / m))
      kl3 <- sum(p3 * log(p3 / m))
      js_mean <- (kl1 + kl2 + kl3) / 3
      mean_js_convergence[i] <- 1 - (js_mean / log(3))
    }

    ens$n_models_agree <- n_agree
    ens$mean_js_convergence <- pmin(1, pmax(0, mean_js_convergence))
    ens$top1_prob_variance_across_models <- pmax(0, as.numeric(top1_prob_var))
    ensemble_fold_matrices[[fold_name]] <- ens
    cat(sprintf("  Fold %s: %d samples, mean agreement = %.2f, mean top1 variance = %.4f\n",
                fold_name, n, mean(n_agree), mean(ens$top1_prob_variance_across_models, na.rm = TRUE)))
  }

  ensemble_fold_matrices
}


# =============================================================================
# Ensemble-Specific: Per-Class Rejection Analysis
# =============================================================================

#' Sweep rejection cutoffs per predicted class for a single probability matrix.
#' Same logic as evaluate_single_matrix_with_rejection_vectorized but grouped by
#' predicted class.
#' @param prob_matrix Probability matrix with confidence column
#' @param fold_name Fold identifier
#' @param model_name Model identifier
#' @param type Analysis type
#' @param cutoff_step Step for cutoff sweep
#' @param confidence_col Which confidence column to use
#' @return Data frame with per-class risk-coverage rows
evaluate_per_class_rejection <- function(prob_matrix, fold_name, model_name, type,
                                         cutoff_step = 0.01,
                                         confidence_col = "confidence_multivariate") {
  meta_cols <- c("y", "inner_fold", "outer_fold", "indices", "study",
                 "sample_indices", "confidence_calibrated", "confidence_multivariate",
                 "is_leftout", "n_models_agree", "mean_js_convergence",
                 "top1_prob_variance_across_models")
  prob_cols <- colnames(prob_matrix)[!colnames(prob_matrix) %in% meta_cols]
  prob_mat <- as.matrix(prob_matrix[, prob_cols, drop = FALSE])

  truth <- gsub("Class\\. ", "", prob_matrix$y)
  pred_indices <- max.col(prob_mat, ties.method = "first")
  preds <- gsub("Class\\. ", "", prob_cols[pred_indices])

  if (confidence_col %in% colnames(prob_matrix)) {
    confidence_vals <- prob_matrix[[confidence_col]]
  } else {
    confidence_vals <- prob_mat[cbind(seq_len(nrow(prob_mat)), pred_indices)]
  }

  correct <- as.integer(truth == preds)
  prob_cutoffs <- seq(0.00, 1.00, by = cutoff_step)

  results <- list()
  idx <- 1L

  for (cls in unique(preds)) {
    cls_mask <- preds == cls
    cls_conf <- confidence_vals[cls_mask]
    cls_correct <- correct[cls_mask]
    n_cls <- sum(cls_mask)

    if (n_cls == 0) next

    for (cutoff in prob_cutoffs) {
      accepted <- cls_conf >= cutoff
      n_accepted <- sum(accepted)

      if (n_accepted == 0) next

      results[[idx]] <- data.frame(
        model = model_name,
        type = type,
        fold = fold_name,
        pred_class = cls,
        prob_cutoff = cutoff,
        accuracy = mean(cls_correct[accepted]),
        n_accepted = n_accepted,
        n_total_class = n_cls,
        perc_rejected = 1 - n_accepted / n_cls,
        stringsAsFactors = FALSE
      )
      idx <- idx + 1L
    }
  }

  if (length(results) == 0) return(data.frame())
  do.call(rbind, results)
}


#' Find per-class cutoffs via leave-one-fold-out cross-fitting (ensemble only).
#' Classes with fewer than min_class_n predicted samples in the training folds
#' fall back to the global cutoff.
#' @param per_class_rc Per-class risk-coverage data (from evaluate_per_class_rejection)
#' @param global_cutoffs Global cross-fitted cutoffs (from find_cutoffs_leave_one_fold_out)
#' @param target_risk Maximum acceptable error rate
#' @param min_class_n Minimum predicted samples in training folds for class-specific cutoff
#' @return Data frame of per-fold, per-class cutoffs with cutoff_source column
find_per_class_cutoffs_leave_one_fold_out <- function(per_class_rc, global_cutoffs,
                                                       target_risk = 0.02,
                                                       min_class_n = 20) {
  cat("Finding per-class cutoffs via leave-one-fold-out cross-fitting...\n")

  all_folds <- unique(per_class_rc$fold)
  all_models <- unique(per_class_rc$model)
  all_classes <- unique(per_class_rc$pred_class)
  cutoff_rows <- list()
  idx <- 1L

  for (held_out_fold in all_folds) {
    for (m in all_models) {
      # Get global cutoff for this model/fold as fallback
      global_row <- global_cutoffs$per_fold_cutoffs[
        global_cutoffs$per_fold_cutoffs$model == m &
          global_cutoffs$per_fold_cutoffs$fold == held_out_fold, ,
        drop = FALSE
      ]
      global_co <- if (nrow(global_row) > 0) global_row$prob_cutoff[1] else 0.5

      for (cls in all_classes) {
        # Training data: other folds, this model, this class
        train_data <- per_class_rc[
          per_class_rc$fold != held_out_fold &
            per_class_rc$model == m &
            per_class_rc$pred_class == cls, ,
          drop = FALSE
        ]

        # Count total predicted samples for this class across training folds
        n_train_samples <- sum(train_data$n_total_class[!duplicated(train_data$fold)])

        if (n_train_samples < min_class_n || nrow(train_data) == 0) {
          # Fall back to global cutoff
          cutoff_rows[[idx]] <- data.frame(
            model = m,
            fold = held_out_fold,
            pred_class = cls,
            prob_cutoff = global_co,
            cutoff_source = "global_fallback",
            n_train_samples = n_train_samples,
            stringsAsFactors = FALSE
          )
          idx <- idx + 1L
          next
        }

        # Aggregate across training folds per cutoff
        agg <- train_data %>%
          dplyr::group_by(prob_cutoff) %>%
          dplyr::summarise(
            mean_accuracy = mean(accuracy, na.rm = TRUE),
            mean_perc_rejected = mean(perc_rejected, na.rm = TRUE),
            mean_risk = 1 - mean(accuracy, na.rm = TRUE),
            mean_coverage = 1 - mean(perc_rejected, na.rm = TRUE),
            .groups = "drop"
          )

        # Pick cutoff using same logic as global
        meets <- agg$mean_risk <= target_risk
        if (any(meets)) {
          df_ok <- agg[meets, , drop = FALSE]
          best_cov <- max(df_ok$mean_coverage, na.rm = TRUE)
          df_ok <- df_ok[df_ok$mean_coverage == best_cov, , drop = FALSE]
          chosen <- df_ok[1, , drop = FALSE]
        } else {
          best_risk <- min(agg$mean_risk, na.rm = TRUE)
          df_r <- agg[agg$mean_risk == best_risk, , drop = FALSE]
          best_cov <- max(df_r$mean_coverage, na.rm = TRUE)
          df_r <- df_r[df_r$mean_coverage == best_cov, , drop = FALSE]
          chosen <- df_r[1, , drop = FALSE]
        }

        cutoff_rows[[idx]] <- data.frame(
          model = m,
          fold = held_out_fold,
          pred_class = cls,
          prob_cutoff = chosen$prob_cutoff,
          cutoff_source = "class_specific",
          n_train_samples = n_train_samples,
          stringsAsFactors = FALSE
        )
        idx <- idx + 1L
      }
    }
  }

  if (length(cutoff_rows) == 0) return(NULL)
  result <- do.call(rbind, cutoff_rows)

  n_specific <- sum(result$cutoff_source == "class_specific")
  n_fallback <- sum(result$cutoff_source == "global_fallback")
  cat(sprintf("  %d class-specific cutoffs, %d global fallbacks (min_class_n=%d)\n",
              n_specific, n_fallback, min_class_n))

  result
}

#' Build per-class cutoff table using only global fold cutoffs.
#' This creates a no-class-specific variant aligned to the per-class output shape.
#' @param per_class_rc Per-class risk-coverage data (from evaluate_per_class_rejection)
#' @param global_cutoffs Global cross-fitted cutoffs (from find_cutoffs_leave_one_fold_out)
#' @return Data frame of per-fold, per-class rows with global-only cutoffs
build_global_only_per_class_cutoffs <- function(per_class_rc, global_cutoffs) {
  if (is.null(per_class_rc) || nrow(per_class_rc) == 0 ||
      is.null(global_cutoffs) || is.null(global_cutoffs$per_fold_cutoffs) ||
      nrow(global_cutoffs$per_fold_cutoffs) == 0) {
    return(NULL)
  }

  classes_by_fold_model <- unique(per_class_rc[, c("model", "fold", "pred_class"), drop = FALSE])
  global_pf <- global_cutoffs$per_fold_cutoffs[, c("model", "fold", "prob_cutoff"), drop = FALSE]

  global_only <- merge(classes_by_fold_model, global_pf, by = c("model", "fold"), all.x = TRUE)
  global_only <- global_only[!is.na(global_only$prob_cutoff), , drop = FALSE]
  if (nrow(global_only) == 0) return(NULL)

  # Keep output schema close to per-class cutoffs for easy downstream reuse.
  global_only$cutoff_source <- "global_only"
  global_only$n_train_samples <- NA_integer_
  global_only <- global_only[, c("model", "fold", "pred_class", "prob_cutoff", "cutoff_source", "n_train_samples")]
  rownames(global_only) <- NULL
  global_only
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
#' @param merge_classes Whether to merge classes
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

    if (merge_classes) {
      probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_prob_method = "sum")
    }

    probability_matrices[[as.character(outer_fold_id)]] <- probability_matrix
  }

  list(matrices = probability_matrices)
}


#' Generate probability matrices from left-out standard prediction CSVs.
#' Left-out sample indices reference the FULL (unfiltered) dataset.
#' @param leftout_results Data frame loaded from leftout CSV (standard format)
#' @param label_mapping Label mapping data frame
#' @param all_subtypes Full (unfiltered) vector of ICC_Subtype labels
#' @param merge_classes Whether to merge classes
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

    if (merge_classes) {
      probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_prob_method = "sum")
    }

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

  for (fold_name in names(leftout_fold_matrices)) {
    leftout_df <- leftout_fold_matrices[[fold_name]]
    leftout_df$is_leftout <- TRUE

    if (fold_name %in% names(augmented)) {
      known_df <- augmented[[fold_name]]
      known_df$is_leftout <- FALSE

      # Remove columns present only in one side (e.g. confidence_calibrated)
      shared_cols <- intersect(colnames(known_df), colnames(leftout_df))
      augmented[[fold_name]] <- rbind(known_df[, shared_cols, drop = FALSE],
                                      leftout_df[, shared_cols, drop = FALSE])
    } else {
      augmented[[fold_name]] <- leftout_df
    }
  }

  augmented
}


#' Build augmented ensemble matrices by combining per-model left-out predictions.
#' Applies ensemble weights to left-out probability rows and appends to existing
#' ensemble matrices.
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

    meta_cols <- c("y", "outer_fold", "sample_indices", "is_leftout")

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

    # Weighted ensemble
    ens_probs <- as.matrix(svm_lo[, all_prob_cols]) * w_svm +
                 as.matrix(xgb_lo[, all_prob_cols]) * w_xgb +
                 as.matrix(nn_lo[, all_prob_cols])  * w_nn

    # Normalise rows
    row_sums <- rowSums(ens_probs, na.rm = TRUE)
    row_sums[row_sums == 0] <- 1
    ens_probs <- ens_probs / row_sums

    ens_df <- data.frame(ens_probs)
    ens_df$y <- svm_lo$y
    ens_df$outer_fold <- svm_lo$outer_fold
    ens_df$sample_indices <- svm_lo$sample_indices
    ens_df$is_leftout <- TRUE

    # Append to existing ensemble matrix for this fold
    if (fold_name %in% names(augmented) && !is.null(augmented[[fold_name]])) {
      known_ens <- augmented[[fold_name]]
      known_ens$is_leftout <- FALSE
      shared_cols <- intersect(colnames(known_ens), colnames(ens_df))
      augmented[[fold_name]] <- rbind(known_ens[, shared_cols, drop = FALSE],
                                      ens_df[, shared_cols, drop = FALSE])
    } else {
      augmented[[fold_name]] <- ens_df
    }
  }

  augmented
}


# =============================================================================
# Platt Scaling for Augmented (Known + Left-Out) Matrices
# =============================================================================

#' Apply Platt calibration to augmented fold matrices.
#' Fits the logistic model on known-class rows (is_leftout == FALSE) from OTHER
#' folds only, but applies calibrated confidence to ALL rows (known + leftout)
#' of the target fold.
#' @param fold_matrices Named list of fold_name -> data.frame (augmented probability matrices)
#' @return List of fold matrices with confidence_calibrated column added
apply_platt_to_augmented_fold_matrices <- function(fold_matrices) {
  n_folds <- length(fold_matrices)
  if (n_folds < 2L) return(fold_matrices)
  fold_names <- names(fold_matrices)
  result <- list()

  for (k in seq_along(fold_names)) {
    target <- fold_matrices[[fold_names[k]]]
    others <- fold_matrices[setdiff(fold_names, fold_names[k])]

    # Filter pool to known-class rows only
    others_known <- lapply(others, function(m) {
      if ("is_leftout" %in% colnames(m)) {
        m[!m$is_leftout, , drop = FALSE]
      } else {
        m
      }
    })

    # Fit on known-class pool, apply to full target (known + leftout)
    result[[fold_names[k]]] <- apply_platt_to_target_from_pool(others_known, target)
  }
  result
}


# =============================================================================
# Leave-One-Fold-Out Cutoff Optimization (Cross-Fitting)
# =============================================================================

#' Find optimal cutoff for each fold using leave-one-fold-out cross-fitting.
#' For each target fold, the cutoff is optimized on all OTHER folds' risk-coverage
#' data, then applied to the target fold. This prevents data leakage.
#'
#' @param risk_coverage_raw Raw per-fold risk-coverage data from
#'   evaluate_all_matrices_with_rejection_unified (columns: model, fold,
#'   prob_cutoff, kappa, accuracy, perc_rejected, ...)
#' @param target_risk Maximum acceptable error rate (default 0.02)
#' @return List with per-fold cutoffs and summary statistics
find_cutoffs_leave_one_fold_out <- function(risk_coverage_raw, target_risk = 0.02) {
  cat("Finding cutoffs via leave-one-fold-out cross-fitting...\n")

  all_folds <- unique(risk_coverage_raw$fold)
  all_models <- unique(risk_coverage_raw$model)
  cutoff_rows <- list()
  idx <- 1L

  for (held_out_fold in all_folds) {
    for (m in all_models) {
      # Train on all folds EXCEPT held_out_fold
      train_data <- risk_coverage_raw[
        risk_coverage_raw$fold != held_out_fold & risk_coverage_raw$model == m, ,
        drop = FALSE
      ]
      if (nrow(train_data) == 0) next

      # Aggregate across training folds per cutoff
      agg <- train_data %>%
        dplyr::group_by(prob_cutoff) %>%
        dplyr::summarise(
          mean_kappa    = mean(kappa, na.rm = TRUE),
          mean_accuracy = mean(accuracy, na.rm = TRUE),
          mean_perc_rejected = mean(perc_rejected, na.rm = TRUE),
          mean_risk     = 1 - mean(accuracy, na.rm = TRUE),
          mean_coverage = 1 - mean(perc_rejected, na.rm = TRUE),
          .groups = "drop"
        )

      # Pick cutoff: among those meeting target_risk, highest coverage then kappa
      meets <- agg$mean_risk <= target_risk
      if (any(meets)) {
        df_ok <- agg[meets, , drop = FALSE]
        best_cov <- max(df_ok$mean_coverage, na.rm = TRUE)
        df_ok <- df_ok[df_ok$mean_coverage == best_cov, , drop = FALSE]
        best_kappa <- max(df_ok$mean_kappa, na.rm = TRUE)
        df_ok <- df_ok[df_ok$mean_kappa == best_kappa, , drop = FALSE]
        chosen <- df_ok[1, , drop = FALSE]
      } else {
        best_risk <- min(agg$mean_risk, na.rm = TRUE)
        df_r <- agg[agg$mean_risk == best_risk, , drop = FALSE]
        best_cov <- max(df_r$mean_coverage, na.rm = TRUE)
        df_r <- df_r[df_r$mean_coverage == best_cov, , drop = FALSE]
        best_kappa <- max(df_r$mean_kappa, na.rm = TRUE)
        df_r <- df_r[df_r$mean_kappa == best_kappa, , drop = FALSE]
        chosen <- df_r[1, , drop = FALSE]
      }

      cutoff_rows[[idx]] <- data.frame(
        model = m,
        fold = held_out_fold,
        prob_cutoff = chosen$prob_cutoff,
        train_risk = chosen$mean_risk,
        train_coverage = chosen$mean_coverage,
        train_kappa = chosen$mean_kappa,
        stringsAsFactors = FALSE
      )
      idx <- idx + 1L
    }
  }

  if (length(cutoff_rows) == 0) {
    warning("No cutoffs could be determined via cross-fitting")
    return(NULL)
  }

  cutoffs_df <- do.call(rbind, cutoff_rows)

  # Summary across folds per model
  summary_stats <- cutoffs_df %>%
    dplyr::group_by(model) %>%
    dplyr::summarise(
      mean_cutoff   = mean(prob_cutoff, na.rm = TRUE),
      sd_cutoff     = sd(prob_cutoff, na.rm = TRUE),
      mean_train_risk = mean(train_risk, na.rm = TRUE),
      mean_train_coverage = mean(train_coverage, na.rm = TRUE),
      mean_train_kappa = mean(train_kappa, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::arrange(dplyr::desc(mean_train_kappa))

  cat("  Cross-fitted cutoff summary:\n")
  print(summary_stats)

  return(list(
    per_fold_cutoffs = cutoffs_df,
    summary_stats = summary_stats
  ))
}


#' Evaluate cross-fitted cutoffs on the held-out fold they were trained for.
#' @param risk_coverage_raw Raw per-fold risk-coverage data
#' @param cross_fitted_cutoffs Output of find_cutoffs_leave_one_fold_out()
#' @return Data frame with per-fold evaluation metrics
evaluate_cross_fitted_cutoffs <- function(risk_coverage_raw, cross_fitted_cutoffs) {
  if (is.null(cross_fitted_cutoffs)) return(NULL)

  cutoffs_df <- cross_fitted_cutoffs$per_fold_cutoffs
  eval_rows <- list()
  idx <- 1L

  for (i in seq_len(nrow(cutoffs_df))) {
    row <- cutoffs_df[i, ]
    fold_data <- risk_coverage_raw[
      risk_coverage_raw$fold == row$fold & risk_coverage_raw$model == row$model, ,
      drop = FALSE
    ]
    if (nrow(fold_data) == 0) next

    # Find the closest cutoff in the held-out fold's data
    cutoff_match <- fold_data[which.min(abs(fold_data$prob_cutoff - row$prob_cutoff)), , drop = FALSE]
    if (nrow(cutoff_match) == 0) next

    eval_rows[[idx]] <- data.frame(
      model = row$model,
      fold = row$fold,
      prob_cutoff = row$prob_cutoff,
      eval_kappa = cutoff_match$kappa,
      eval_accuracy = cutoff_match$accuracy,
      eval_perc_rejected = cutoff_match$perc_rejected,
      eval_risk = 1 - cutoff_match$accuracy,
      eval_coverage = 1 - cutoff_match$perc_rejected,
      stringsAsFactors = FALSE
    )
    idx <- idx + 1L
  }

  if (length(eval_rows) == 0) return(NULL)
  do.call(rbind, eval_rows)
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

  # Filter data based on criteria
  subtypes_with_sufficient_samples <- names(which(table(leukemia_subtypes) >= DATA_FILTERS$min_samples_per_subtype))
  filtered_leukemia_subtypes <- leukemia_subtypes[
    leukemia_subtypes %in% subtypes_with_sufficient_samples &
    !leukemia_subtypes %in% DATA_FILTERS$excluded_subtypes &
    study_names %in% DATA_FILTERS$selected_studies
  ]

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
          result <- generate_outer_ovr_probability_matrices(results, label_mapping, filter_unseen_classes = TRUE, merge_classes = merge_classes)
        } else {
          result <- generate_outer_standard_probability_matrices(results, label_mapping, filtered_leukemia_subtypes, filter_unseen_classes = TRUE, merge_classes = merge_classes)
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

    cat(sprintf("Processing %s ensemble...\n", toupper(type)))
    ensemble_matrices[[type]] <- list()

    # Generate global ensemble only (OvR removed)
    global_ensemble <- apply_ensemble_weights_to_outer_cv(
      outer_probability_matrices, ensemble_weights[[type]], type, "global"
    )
    if (!is.null(global_ensemble)) {
      ensemble_matrices[[type]][["global_ensemble"]] <- global_ensemble
    }
  }

  # Combine individual models and ensemble results for performance calculation
  cat("Combining results for performance analysis...\n")
  all_probability_matrices <- outer_probability_matrices

  for (type in c("cv", "loso")) {
    if (type %in% names(ensemble_matrices)) {
      if ("global_ensemble" %in% names(ensemble_matrices[[type]])) {
        if (!"Global_Optimized" %in% names(all_probability_matrices)) {
          all_probability_matrices[["Global_Optimized"]] <- list()
        }
        all_probability_matrices[["Global_Optimized"]][[type]] <- ensemble_matrices[[type]][["global_ensemble"]]
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

  # Save pre-Platt matrices for leftout-aware analysis (uses raw max-prob, not calibrated)
  all_probability_matrices_raw <- all_probability_matrices

  # Apply Platt scaling for rejection confidence (out-of-sample per outer fold)
  cat("Applying Platt scaling to probability matrices for rejection...\n")
  for (model_name in names(all_probability_matrices)) {
    for (type in c("cv", "loso")) {
      if (!type %in% names(all_probability_matrices[[model_name]])) next
      fold_list <- all_probability_matrices[[model_name]][[type]]
      if (!is.list(fold_list) || length(fold_list) < 2L) next
      calibrated_list <- apply_platt_to_inner_fold_matrices(fold_list)
      for (fold_name in names(calibrated_list)) {
        all_probability_matrices[[model_name]][[type]][[fold_name]] <- calibrated_list[[fold_name]]
      }
    }
  }

  # Run full risk–coverage rejection analysis (independent of fixed inner-CV cutoffs).
  # Pass ensemble_matrices as NULL so Global_Optimized is only taken from
  # all_probability_matrices (already includes it), avoiding duplicate rows per fold/cutoff.
  cat("Running full risk–coverage rejection analysis on outer CV results...\n")
  risk_coverage_results <- list()
  for (type in c("cv", "loso")) {
    if (!type %in% names(ensemble_matrices)) next
    cat(sprintf("  Processing %s risk–coverage analysis...\n", toupper(type)))
    full_res <- evaluate_all_matrices_with_rejection_unified(
      all_probability_matrices,
      list(global_optimized_ensemble_matrices = NULL),
      type,
      has_inner_folds = FALSE
    )
    risk_coverage_results[[type]] <- full_res
  }

  # -------------------------------------------------------------------------
  # Optional: leave-one-fold-out cutoff optimization (cross-fitting)
  # -------------------------------------------------------------------------
  cross_fitted_cutoffs <- list()
  cross_fitted_evaluation <- list()
  if (ENABLE_CUTOFF_SELECTION) {
    cat("Running leave-one-fold-out cutoff optimization...\n")
    for (type in c("cv", "loso")) {
      if (!type %in% names(risk_coverage_results)) next
      rc_raw <- risk_coverage_results[[type]]
      if (is.null(rc_raw) || nrow(rc_raw) == 0) next

      cat(sprintf("  Cross-fitting cutoffs for %s...\n", toupper(type)))
      cross_fitted_cutoffs[[type]] <- find_cutoffs_leave_one_fold_out(rc_raw)
      cross_fitted_evaluation[[type]] <- evaluate_cross_fitted_cutoffs(
        rc_raw, cross_fitted_cutoffs[[type]]
      )
    }
  } else {
    cat("Skipping cutoff selection (ENABLE_CUTOFF_SELECTION=FALSE); exporting risk-coverage curves only.\n")
  }

  # -------------------------------------------------------------------------
  # Left-out-aware analysis (optional, requires leftout prediction CSVs)
  # -------------------------------------------------------------------------
  augmented_cross_fitted_cutoffs <- list()
  augmented_cross_fitted_evaluation <- list()
  augmented_risk_coverage_results <- list()

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

      # Ensemble augmentation
      if (type %in% names(ensemble_weights) && "Global_Optimized" %in% names(all_probability_matrices_raw)) {
        known_ens <- all_probability_matrices_raw[["Global_Optimized"]][[type]]
        lo_per_model <- list()
        for (mn in c("svm", "xgboost", "neural_net")) {
          lo_per_model[[mn]] <- leftout_probability_matrices[[mn]][[type]]
        }
        has_all_lo <- all(sapply(lo_per_model, function(x) !is.null(x) && length(x) > 0))

        if (!is.null(known_ens) && has_all_lo) {
          if (is.null(all_augmented_matrices[["Global_Optimized"]])) {
            all_augmented_matrices[["Global_Optimized"]] <- list()
          }
          all_augmented_matrices[["Global_Optimized"]][[type]] <- build_augmented_ensemble(
            known_ens, lo_per_model, ensemble_weights[[type]], type
          )
        } else if (!is.null(known_ens)) {
          if (is.null(all_augmented_matrices[["Global_Optimized"]])) {
            all_augmented_matrices[["Global_Optimized"]] <- list()
          }
          all_augmented_matrices[["Global_Optimized"]][[type]] <- known_ens
        }
      }
    }

    # Apply Platt scaling to augmented matrices (fit on known-class only, apply to all)
    cat("Applying Platt scaling to augmented matrices...\n")
    for (model_name in names(all_augmented_matrices)) {
      for (type in c("cv", "loso")) {
        if (!type %in% names(all_augmented_matrices[[model_name]])) next
        fold_list <- all_augmented_matrices[[model_name]][[type]]
        if (!is.list(fold_list) || length(fold_list) < 2L) next
        calibrated_list <- apply_platt_to_augmented_fold_matrices(fold_list)
        for (fold_name in names(calibrated_list)) {
          all_augmented_matrices[[model_name]][[type]][[fold_name]] <- calibrated_list[[fold_name]]
        }
      }
    }

    # Run augmented risk-coverage analysis
    cat("Running augmented risk-coverage analysis (known + left-out)...\n")
    for (type in c("cv", "loso")) {
      if (!type %in% names(ensemble_matrices)) next

      aug_res <- evaluate_all_matrices_with_rejection_unified(
        all_augmented_matrices,
        list(global_optimized_ensemble_matrices = NULL),
        type,
        has_inner_folds = FALSE
      )
      augmented_risk_coverage_results[[type]] <- aug_res
    }

    # Optional cutoff selection on augmented data
    if (ENABLE_CUTOFF_SELECTION) {
      cat("Running leave-one-fold-out cutoff optimization on augmented data...\n")
      for (type in c("cv", "loso")) {
        if (!type %in% names(augmented_risk_coverage_results)) next
        rc_aug <- augmented_risk_coverage_results[[type]]
        if (is.null(rc_aug) || nrow(rc_aug) == 0) next

        cat(sprintf("  Cross-fitting augmented cutoffs for %s...\n", toupper(type)))
        augmented_cross_fitted_cutoffs[[type]] <- find_cutoffs_leave_one_fold_out(rc_aug)
        augmented_cross_fitted_evaluation[[type]] <- evaluate_cross_fitted_cutoffs(
          rc_aug, augmented_cross_fitted_cutoffs[[type]]
        )
      }
    } else {
      cat("Skipping augmented cutoff selection (ENABLE_CUTOFF_SELECTION=FALSE).\n")
    }
  } else {
    cat("\nNo left-out prediction files configured. Skipping augmented analysis.\n")
    cat("To enable: run run_outer_cv.py --include_leftout, then set paths in LEFTOUT_MODEL_CONFIGS.\n")
  }

  # -------------------------------------------------------------------------
  # Ensemble-specific: multivariate calibration + per-class cutoffs (additive)
  # Runs only on Global_Optimized ensemble, after all existing analyses.
  # Computes both:
  #   - standard (known-only) multivariate analysis
  #   - augmented (known + leftout) multivariate analysis, when leftout is enabled
  # -------------------------------------------------------------------------
  multivariate_results <- list(standard = list(), with_leftout = list())

  run_multivariate_for_ensemble <- function(ens_folds, per_model_matrices, type, model_label) {
    if (!is.list(ens_folds) || length(ens_folds) < 2L) return(NULL)

    cat(sprintf("  Processing %s (%s)...\n", toupper(type), model_label))

    # 1. Disagreement features (ensemble only)
    ens_folds <- compute_disagreement_features(ens_folds, per_model_matrices, type)

    # 2. Multivariate Platt calibration (leave-one-fold-out)
    cat("  Applying multivariate Platt calibration to ensemble...\n")
    fold_names <- names(ens_folds)
    for (k in seq_along(fold_names)) {
      target <- ens_folds[[fold_names[k]]]
      others <- ens_folds[setdiff(fold_names, fold_names[k])]

      # If augmented matrices, fit only on known-class rows
      others_for_fit <- lapply(others, function(m) {
        if ("is_leftout" %in% colnames(m)) m[!m$is_leftout, , drop = FALSE] else m
      })

      ens_folds[[fold_names[k]]] <- apply_multivariate_platt_to_target_from_pool(
        others_for_fit, target
      )
    }

    # 3. Risk-coverage sweep using confidence_multivariate
    cat("  Running multivariate risk-coverage sweep...\n")
    ens_folds_mv <- lapply(ens_folds, function(m) {
      if ("confidence_multivariate" %in% colnames(m)) {
        m$confidence_calibrated <- m$confidence_multivariate
      }
      m
    })

    mv_rc_list <- list()
    for (fold_name in names(ens_folds_mv)) {
      mv_rc_list[[length(mv_rc_list) + 1]] <- evaluate_single_matrix_with_rejection_vectorized(
        ens_folds_mv[[fold_name]], fold_name, model_label, type
      )
    }
    mv_rc_raw <- do.call(rbind, mv_rc_list)

    # 4. Optional cross-fitted global cutoffs
    mv_global_cutoffs <- NULL
    mv_global_evaluation <- NULL
    if (ENABLE_CUTOFF_SELECTION) {
      cat("  Cross-fitting global cutoffs on multivariate confidence...\n")
      mv_global_cutoffs <- find_cutoffs_leave_one_fold_out(mv_rc_raw)
      mv_global_evaluation <- evaluate_cross_fitted_cutoffs(mv_rc_raw, mv_global_cutoffs)
    }

    # 5. Per-class rejection + cutoffs (with fallback)
    cat("  Running per-class rejection analysis...\n")
    per_class_rc_list <- list()
    for (fold_name in names(ens_folds_mv)) {
      per_class_rc_list[[length(per_class_rc_list) + 1]] <- evaluate_per_class_rejection(
        ens_folds_mv[[fold_name]], fold_name, model_label, type,
        confidence_col = "confidence_calibrated"
      )
    }
    per_class_rc <- do.call(rbind, per_class_rc_list)

    mv_per_class_cutoffs <- NULL
    if (ENABLE_CUTOFF_SELECTION && !is.null(mv_global_cutoffs) && !is.null(per_class_rc) && nrow(per_class_rc) > 0) {
      mv_per_class_cutoffs <- find_per_class_cutoffs_leave_one_fold_out(
        per_class_rc, mv_global_cutoffs
      )
    }
    mv_per_class_cutoffs_global_only <- NULL
    if (ENABLE_CUTOFF_SELECTION && !is.null(mv_global_cutoffs)) {
      mv_per_class_cutoffs_global_only <- build_global_only_per_class_cutoffs(
        per_class_rc, mv_global_cutoffs
      )
    }

    list(
      risk_coverage = mv_rc_raw,
      global_cutoffs = mv_global_cutoffs,
      global_evaluation = mv_global_evaluation,
      per_class_risk_coverage = per_class_rc,
      per_class_cutoffs = mv_per_class_cutoffs,
      per_class_cutoffs_global_only = mv_per_class_cutoffs_global_only,
      fold_matrices = ens_folds_mv,
      model_label = model_label
    )
  }

  if ("Global_Optimized" %in% names(all_probability_matrices)) {
    cat("\n=== Ensemble Multivariate Rejection Analysis (standard) ===\n")
    for (type in c("cv", "loso")) {
      if (!type %in% names(all_probability_matrices[["Global_Optimized"]])) next
      ens_folds_std <- all_probability_matrices[["Global_Optimized"]][[type]]
      multivariate_results$standard[[type]] <- run_multivariate_for_ensemble(
        ens_folds_std, all_probability_matrices_raw, type, "Global_Optimized_MV"
      )
    }
  }

  if (has_leftout_data) {
    cat("\n=== Ensemble Multivariate Rejection Analysis (with leftout) ===\n")
    if (exists("all_augmented_matrices") && "Global_Optimized" %in% names(all_augmented_matrices)) {
      for (type in c("cv", "loso")) {
        if (!type %in% names(all_augmented_matrices[["Global_Optimized"]])) next
        ens_folds_aug <- all_augmented_matrices[["Global_Optimized"]][[type]]
        # Use augmented per-model matrices for disagreement calculation (includes leftout rows)
        multivariate_results$with_leftout[[type]] <- run_multivariate_for_ensemble(
          ens_folds_aug, all_augmented_matrices, type, "Global_Optimized_MV_leftout"
        )
      }
    } else {
      cat("  No augmented Global_Optimized matrices available; skipping multivariate-with-leftout.\n")
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
    cross_fitted_cutoffs = cross_fitted_cutoffs,
    cross_fitted_evaluation = cross_fitted_evaluation,
    augmented_cross_fitted_cutoffs = augmented_cross_fitted_cutoffs,
    augmented_cross_fitted_evaluation = augmented_cross_fitted_evaluation,
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

  # Standard outputs (known-only)
  out_standard_dir <- file.path(analysis_output_dir, "standard")
  dir.create(out_standard_dir, recursive = TRUE, showWarnings = FALSE)

  # Leftout-aware outputs (known + leftout)
  out_with_leftout_dir <- file.path(analysis_output_dir, "with_leftout")
  dir.create(out_with_leftout_dir, recursive = TRUE, showWarnings = FALSE)

  # Multivariate ensemble outputs
  out_mv_standard_dir <- file.path(analysis_output_dir, "ensemble_multivariate", "standard")
  dir.create(out_mv_standard_dir, recursive = TRUE, showWarnings = FALSE)
  out_mv_leftout_dir <- file.path(analysis_output_dir, "ensemble_multivariate", "with_leftout")
  dir.create(out_mv_leftout_dir, recursive = TRUE, showWarnings = FALSE)

  # -------------------------------------------------------------------------
  # Risk–coverage outputs (clear naming)
  #
  # We export two related views:
  # 1) heldout_curve_raw: per (model, held_out_fold, prob_cutoff) measured on that fold
  # 2) trainpool_curve: for each held_out_fold, aggregate OTHER folds (cross-fitting pool)
  # -------------------------------------------------------------------------

  build_trainpool_risk_coverage <- function(rc_unique) {
    folds <- unique(rc_unique$fold)
    out <- list()
    idx <- 1L
    for (held in folds) {
      train <- rc_unique[rc_unique$fold != held, , drop = FALSE]
      if (nrow(train) == 0) next
      # Compute pool metrics by aggregating counts across folds (not averaging fold metrics).
      # This matches the intent of a "train pool" curve: what you get on all OTHER folds combined.
      agg <- train %>%
        dplyr::group_by(model, prob_cutoff) %>%
        dplyr::summarise(
          train_total_samples = sum(total_samples, na.rm = TRUE),
          train_n_accepted = sum(n_accepted, na.rm = TRUE),
          train_n_rejected = sum(n_rejected, na.rm = TRUE),
          # Weighted by accepted sample count
          train_accuracy = sum(accuracy * n_accepted, na.rm = TRUE) / pmax(1, train_n_accepted),
          # Kappa can't be pooled exactly without a confusion matrix; provide a weighted summary
          train_kappa_weighted = sum(kappa * n_accepted, na.rm = TRUE) / pmax(1, train_n_accepted),
          train_perc_rejected = train_n_rejected / pmax(1, train_total_samples),
          train_risk = 1 - train_accuracy,
          train_coverage = 1 - train_perc_rejected,
          .groups = "drop"
        )
      agg$held_out_fold <- held
      out[[idx]] <- agg
      idx <- idx + 1L
    }
    if (length(out) == 0) return(data.frame())
    do.call(rbind, out)
  }

  build_monotonic_frontier_foldwise <- function(heldout_rc) {
    if (is.null(heldout_rc) || nrow(heldout_rc) == 0) return(data.frame())
    heldout_rc %>%
      dplyr::mutate(
        risk = 1 - accuracy,
        coverage = 1 - perc_rejected
      ) %>%
      dplyr::arrange(model, fold, risk, dplyr::desc(coverage)) %>%
      dplyr::group_by(model, fold) %>%
      dplyr::mutate(coverage_mono = cummax(coverage)) %>%
      dplyr::ungroup()
  }

  build_pareto_frontier <- function(heldout_rc) {
    if (is.null(heldout_rc) || nrow(heldout_rc) == 0) return(data.frame())
    agg <- heldout_rc %>%
      dplyr::mutate(
        risk = 1 - accuracy,
        coverage = 1 - perc_rejected
      ) %>%
      dplyr::group_by(model, prob_cutoff) %>%
      dplyr::summarise(
        mean_risk = mean(risk, na.rm = TRUE),
        mean_coverage = mean(coverage, na.rm = TRUE),
        .groups = "drop"
      )
    if (nrow(agg) == 0) return(data.frame())

    out <- list()
    idx <- 1L
    for (m in unique(agg$model)) {
      df <- agg[agg$model == m, , drop = FALSE]
      keep <- rep(TRUE, nrow(df))
      for (i in seq_len(nrow(df))) {
        dominated <- any(
          df$mean_risk <= df$mean_risk[i] &
            df$mean_coverage >= df$mean_coverage[i] &
            (df$mean_risk < df$mean_risk[i] | df$mean_coverage > df$mean_coverage[i]),
          na.rm = TRUE
        )
        keep[i] <- !dominated
      }
      out[[idx]] <- df[keep, , drop = FALSE]
      idx <- idx + 1L
    }
    if (length(out) == 0) return(data.frame())
    do.call(rbind, out)
  }

  build_trainpool_heldout_gap <- function(heldout_rc, trainpool_rc) {
    if (is.null(heldout_rc) || nrow(heldout_rc) == 0 ||
        is.null(trainpool_rc) || nrow(trainpool_rc) == 0) return(data.frame())
    held <- heldout_rc %>%
      dplyr::mutate(
        heldout_risk = 1 - accuracy,
        heldout_coverage = 1 - perc_rejected
      ) %>%
      dplyr::group_by(model, prob_cutoff) %>%
      dplyr::summarise(
        heldout_risk = mean(heldout_risk, na.rm = TRUE),
        heldout_coverage = mean(heldout_coverage, na.rm = TRUE),
        .groups = "drop"
      )
    train <- trainpool_rc %>%
      dplyr::group_by(model, prob_cutoff) %>%
      dplyr::summarise(
        trainpool_risk = mean(train_risk, na.rm = TRUE),
        trainpool_coverage = mean(train_coverage, na.rm = TRUE),
        .groups = "drop"
      )
    dplyr::left_join(held, train, by = c("model", "prob_cutoff")) %>%
      dplyr::mutate(
        risk_gap = heldout_risk - trainpool_risk,
        coverage_gap = heldout_coverage - trainpool_coverage
      )
  }

  export_global_rc_diagnostics <- function(heldout_rc, trainpool_rc, output_dir, type, model_labels) {
    if (is.null(heldout_rc) || nrow(heldout_rc) == 0) return(invisible(NULL))
    heldout_model <- heldout_rc[heldout_rc$model %in% model_labels, , drop = FALSE]
    if (nrow(heldout_model) == 0) return(invisible(NULL))
    trainpool_model <- trainpool_rc[trainpool_rc$model %in% model_labels, , drop = FALSE]

    frontier_foldwise <- build_monotonic_frontier_foldwise(heldout_model)
    pareto <- build_pareto_frontier(heldout_model)
    gap_tbl <- build_trainpool_heldout_gap(heldout_model, trainpool_model)

    write.csv(frontier_foldwise,
              file.path(output_dir, paste0("risk_coverage_heldout_frontier_foldwise_", type, ".csv")),
              row.names = FALSE)
    write.csv(pareto,
              file.path(output_dir, paste0("risk_coverage_heldout_pareto_", type, ".csv")),
              row.names = FALSE)
    write.csv(gap_tbl,
              file.path(output_dir, paste0("risk_coverage_gap_trainpool_vs_heldout_", type, ".csv")),
              row.names = FALSE)
    invisible(NULL)
  }

  build_global_prediction_decisions <- function(fold_matrices, type, regime, method, model_label,
                                                confidence_col = "confidence_calibrated") {
    if (is.null(fold_matrices) || !is.list(fold_matrices) || length(fold_matrices) == 0) {
      return(data.frame())
    }
    meta_cols <- c("y", "inner_fold", "outer_fold", "indices", "study", "sample_indices",
                   "confidence_calibrated", "confidence_multivariate", "is_leftout",
                   "n_models_agree", "mean_js_convergence", "top1_prob_variance_across_models")
  }

  build_global_prediction_decisions <- function(fold_matrices, type, regime, method, model_label,
                                                confidence_col = "confidence_calibrated") {
    if (is.null(fold_matrices) || !is.list(fold_matrices) || length(fold_matrices) == 0) {
      return(data.frame())
    }
    meta_cols <- c("y", "inner_fold", "outer_fold", "indices", "study", "sample_indices",
                   "confidence_calibrated", "confidence_multivariate", "is_leftout",
                   "n_models_agree", "mean_js_convergence", "top1_prob_variance_across_models")
    out <- list()
    idx <- 1L
    for (fold_name in names(fold_matrices)) {
      m <- fold_matrices[[fold_name]]
      if (is.null(m) || nrow(m) == 0) next
      prob_cols <- colnames(m)[!colnames(m) %in% meta_cols]
      if (length(prob_cols) == 0) next
      prob_mat <- as.matrix(m[, prob_cols, drop = FALSE])
      pred_idx <- max.col(prob_mat, ties.method = "first")
      pred <- gsub("Class\\. ", "", prob_cols[pred_idx])
      truth <- gsub("Class\\. ", "", m$y)
      confidence <- if (confidence_col %in% colnames(m)) {
        as.numeric(m[[confidence_col]])
      } else {
        prob_mat[cbind(seq_len(nrow(prob_mat)), pred_idx)]
      }
      out[[idx]] <- data.frame(
        method = method,
        model = model_label,
        regime = regime,
        split_type = type,
        fold = as.character(fold_name),
        sample_indices = if ("sample_indices" %in% colnames(m)) m$sample_indices else NA_integer_,
        is_leftout = if ("is_leftout" %in% colnames(m)) as.logical(m$is_leftout) else FALSE,
        truth = truth,
        pred = pred,
        pred_class = pred,
        confidence = confidence,
        stringsAsFactors = FALSE
      )
      idx <- idx + 1L
    }
    if (length(out) == 0) return(data.frame())
    do.call(rbind, out)
  }

  build_accepted_support_by_class <- function(decisions_df, type, regime, method, model_label, cutoff_step = 0.01) {
    if (is.null(decisions_df) || nrow(decisions_df) == 0) return(data.frame())
    cutoffs <- seq(0, 1, by = cutoff_step)
    out <- list()
    idx <- 1L
    for (fold_name in unique(decisions_df$fold)) {
      fold_df <- decisions_df[decisions_df$fold == fold_name, , drop = FALSE]
      if (nrow(fold_df) == 0) next
      classes <- sort(unique(fold_df$truth))
      for (co in cutoffs) {
        accepted <- fold_df[fold_df$confidence >= co, , drop = FALSE]
        acc_counts <- table(factor(accepted$truth, levels = classes))
        total_counts <- table(factor(fold_df$truth, levels = classes))
        out[[idx]] <- data.frame(
          method = method,
          model = model_label,
          regime = regime,
          split_type = type,
          fold = fold_name,
          prob_cutoff = co,
          truth_class = classes,
          n_total_class = as.integer(total_counts),
          n_accepted_class = as.integer(acc_counts),
          accepted_rate_class = as.integer(acc_counts) / pmax(1, as.integer(total_counts)),
          stringsAsFactors = FALSE
        )
        idx <- idx + 1L
      }
    }
    if (length(out) == 0) return(data.frame())
    do.call(rbind, out)
  }

  # Save risk–coverage summaries for outer CV (standard / known-only)
  for (type in c("cv", "loso")) {
    if (!type %in% names(risk_coverage_results)) next
    rc_raw <- risk_coverage_results[[type]]
    if (is.null(rc_raw) || nrow(rc_raw) == 0) next

    # Deduplicate: keep one row per (model, fold, prob_cutoff) to handle any duplicate
    # fold entries (e.g. from nested structure or multiple sources)
    rc_unique <- rc_raw %>%
      dplyr::group_by(model, fold, prob_cutoff) %>%
      dplyr::slice(1L) %>%
      dplyr::ungroup()
    n_dup <- nrow(rc_raw) - nrow(rc_unique)
    if (n_dup > 0) {
      warning(sprintf(
        "Risk-coverage %s: removed %d duplicate (model, fold, prob_cutoff) rows. Check probability matrix structure.",
        toupper(type), n_dup
      ))
    }

    # (1) Held-out fold curves (raw, per fold)
    heldout_curve_raw_path <- file.path(
      out_standard_dir,
      paste0("risk_coverage_heldout_curve_raw_", type, ".csv")
    )
    write.csv(rc_unique, heldout_curve_raw_path, row.names = FALSE)

    # (2) Training-pool curves: for each held-out fold, aggregate all other folds
    trainpool_curve <- build_trainpool_risk_coverage(rc_unique)
    trainpool_curve_path <- file.path(
      out_standard_dir,
      paste0("risk_coverage_trainpool_curve_", type, ".csv")
    )
    if (nrow(trainpool_curve) > 0) {
      write.csv(trainpool_curve, trainpool_curve_path, row.names = FALSE)
    }

    cat(sprintf("Saved risk–coverage exports for %s:\\n", toupper(type)))
    cat(sprintf("  heldout_curve_raw:  %s\\n", heldout_curve_raw_path))
    cat(sprintf("  trainpool_curve:    %s\\n", trainpool_curve_path))

    # Global Ensemble diagnostics for monotonic-cutoff analysis.
    export_global_rc_diagnostics(
      rc_unique, trainpool_curve, out_standard_dir, type,
      model_labels = c("Global_Optimized")
    )

  }

  # Save cross-fitted cutoffs (standard / known-only) when enabled
  if (ENABLE_CUTOFF_SELECTION) {
    cutoff_dir <- file.path(out_standard_dir, "cutoffs")
    dir.create(cutoff_dir, recursive = TRUE, showWarnings = FALSE)

    for (type in c("cv", "loso")) {
      if (!type %in% names(cross_fitted_cutoffs) || is.null(cross_fitted_cutoffs[[type]])) next

      # Per-fold cutoffs
      cutoff_path <- file.path(cutoff_dir, paste0("cutoffs_", type, ".csv"))
      write.csv(cross_fitted_cutoffs[[type]]$per_fold_cutoffs, cutoff_path, row.names = FALSE)
      cat(sprintf("Saved cross-fitted cutoffs for %s to: %s\n", toupper(type), cutoff_path))

      # Summary
      summary_path <- file.path(cutoff_dir, paste0("cutoffs_summary_", type, ".csv"))
      write.csv(cross_fitted_cutoffs[[type]]$summary_stats, summary_path, row.names = FALSE)

      # Evaluation on held-out folds
      if (!is.null(cross_fitted_evaluation[[type]])) {
        eval_path <- file.path(cutoff_dir, paste0("cutoffs_evaluation_", type, ".csv"))
        write.csv(cross_fitted_evaluation[[type]], eval_path, row.names = FALSE)
        cat(sprintf("Saved cross-fitted evaluation for %s to: %s\n", toupper(type), eval_path))
      }
    }
  }

  # Save left-out-aware cutoffs and risk-coverage if available
  if (has_leftout_data) {
    if (ENABLE_CUTOFF_SELECTION) {
      aug_cutoff_dir <- file.path(out_with_leftout_dir, "cutoffs")
      dir.create(aug_cutoff_dir, recursive = TRUE, showWarnings = FALSE)
    }

    for (type in c("cv", "loso")) {
      # Augmented risk-coverage (deduplicate like the standard path)
      if (type %in% names(augmented_risk_coverage_results)) {
        rc_aug <- augmented_risk_coverage_results[[type]]
        if (!is.null(rc_aug) && nrow(rc_aug) > 0) {
          rc_aug_unique <- rc_aug %>%
            dplyr::group_by(model, fold, prob_cutoff) %>%
            dplyr::slice(1L) %>%
            dplyr::ungroup()

          # Also export heldout/trainpool curves with clear naming
          heldout_curve_raw_path <- file.path(
            out_with_leftout_dir,
            paste0("risk_coverage_heldout_curve_raw_", type, ".csv")
          )
          write.csv(rc_aug_unique, heldout_curve_raw_path, row.names = FALSE)

          trainpool_curve <- build_trainpool_risk_coverage(rc_aug_unique)
          trainpool_curve_path <- file.path(
            out_with_leftout_dir,
            paste0("risk_coverage_trainpool_curve_", type, ".csv")
          )
          if (nrow(trainpool_curve) > 0) {
            write.csv(trainpool_curve, trainpool_curve_path, row.names = FALSE)
          }

          # Keep a fully deduplicated raw table for debugging inside the same folder
          rc_aug_dedup_path <- file.path(
            out_with_leftout_dir,
            paste0("risk_coverage_heldout_curve_raw_dedup_", type, ".csv")
          )
          write.csv(rc_aug_unique, rc_aug_dedup_path, row.names = FALSE)

          cat(sprintf("Saved leftout-aware risk-coverage exports for %s to: %s\n", toupper(type), out_with_leftout_dir))

          # Global Ensemble diagnostics for monotonic-cutoff analysis (with leftout).
          export_global_rc_diagnostics(
            rc_aug_unique, trainpool_curve, out_with_leftout_dir, type,
            model_labels = c("Global_Optimized")
          )
        }
      }

      # Augmented cutoffs
      if (ENABLE_CUTOFF_SELECTION &&
          type %in% names(augmented_cross_fitted_cutoffs) &&
          !is.null(augmented_cross_fitted_cutoffs[[type]])) {
        cutoff_path <- file.path(aug_cutoff_dir, paste0("cutoffs_", type, ".csv"))
        write.csv(augmented_cross_fitted_cutoffs[[type]]$per_fold_cutoffs, cutoff_path, row.names = FALSE)
        cat(sprintf("Saved augmented cross-fitted cutoffs for %s to: %s\n", toupper(type), cutoff_path))

        summary_path <- file.path(aug_cutoff_dir, paste0("cutoffs_summary_", type, ".csv"))
        write.csv(augmented_cross_fitted_cutoffs[[type]]$summary_stats, summary_path, row.names = FALSE)

        if (!is.null(augmented_cross_fitted_evaluation[[type]])) {
          eval_path <- file.path(aug_cutoff_dir, paste0("cutoffs_evaluation_", type, ".csv"))
          write.csv(augmented_cross_fitted_evaluation[[type]], eval_path, row.names = FALSE)
          cat(sprintf("Saved augmented cutoff evaluation for %s to: %s\n", toupper(type), eval_path))
        }
      }
    }
  }

  # Save multivariate ensemble results (standard + with_leftout)
  save_mv_bundle <- function(bundle, dir_name) {
    if (is.null(bundle) || length(bundle) == 0) return(invisible(NULL))
    mv_dir <- file.path(analysis_output_dir, "ensemble_multivariate", dir_name)
    dir.create(mv_dir, recursive = TRUE, showWarnings = FALSE)

    for (type in c("cv", "loso")) {
      if (!type %in% names(bundle) || is.null(bundle[[type]])) next
      mv <- bundle[[type]]

      if (!is.null(mv$risk_coverage) && nrow(mv$risk_coverage) > 0) {
        # Match the standard naming convention: heldout raw/mean + trainpool curves
        rc_raw <- mv$risk_coverage
        rc_unique <- rc_raw %>%
          dplyr::group_by(model, fold, prob_cutoff) %>%
          dplyr::slice(1L) %>%
          dplyr::ungroup()

        heldout_curve_raw_path <- file.path(mv_dir, paste0("risk_coverage_heldout_curve_raw_", type, ".csv"))
        write.csv(rc_unique, heldout_curve_raw_path, row.names = FALSE)

        trainpool_curve <- build_trainpool_risk_coverage(rc_unique)
        trainpool_curve_path <- file.path(mv_dir, paste0("risk_coverage_trainpool_curve_", type, ".csv"))
        if (nrow(trainpool_curve) > 0) {
          write.csv(trainpool_curve, trainpool_curve_path, row.names = FALSE)
        }

        export_global_rc_diagnostics(
          rc_unique, trainpool_curve, mv_dir, type,
          model_labels = unique(rc_unique$model)
        )
      }
      if (!is.null(mv$global_cutoffs)) {
        write.csv(mv$global_cutoffs$per_fold_cutoffs,
                  file.path(mv_dir, paste0("cutoffs_", type, ".csv")), row.names = FALSE)
        write.csv(mv$global_cutoffs$summary_stats,
                  file.path(mv_dir, paste0("cutoffs_summary_", type, ".csv")), row.names = FALSE)
      }
      if (!is.null(mv$global_evaluation)) {
        write.csv(mv$global_evaluation,
                  file.path(mv_dir, paste0("cutoffs_evaluation_", type, ".csv")), row.names = FALSE)
      }
      if (!is.null(mv$per_class_risk_coverage) && nrow(mv$per_class_risk_coverage) > 0) {
        write.csv(mv$per_class_risk_coverage,
                  file.path(mv_dir, paste0("per_class_risk_coverage_", type, ".csv")), row.names = FALSE)
      }
      if (!is.null(mv$per_class_cutoffs)) {
        write.csv(mv$per_class_cutoffs,
                  file.path(mv_dir, paste0("per_class_cutoffs_", type, ".csv")), row.names = FALSE)
      }
      if (!is.null(mv$per_class_cutoffs_global_only)) {
        write.csv(
          mv$per_class_cutoffs_global_only,
          file.path(mv_dir, paste0("per_class_cutoffs_global_only_", type, ".csv")),
          row.names = FALSE
        )
      }

      # Prediction-level decisions and accepted-class support for this MV bundle.
      if (!is.null(mv$fold_matrices)) {
        mv_decisions <- build_global_prediction_decisions(
          mv$fold_matrices,
          type = type,
          regime = dir_name,
          method = "multivariate",
          model_label = if (!is.null(mv$model_label)) mv$model_label else "Global_Optimized_MV",
          confidence_col = "confidence_multivariate"
        )
        if (nrow(mv_decisions) > 0) {
          saveRDS(
            mv_decisions,
            file.path(mv_dir, paste0("global_ensemble_prediction_decisions_", dir_name, "_", type, ".rds"))
          )
          mv_support <- build_accepted_support_by_class(
            mv_decisions, type, dir_name, "multivariate", unique(mv_decisions$model)[1]
          )
          write.csv(
            mv_support,
            file.path(mv_dir, paste0("accepted_support_by_class_", type, ".csv")),
            row.names = FALSE
          )
        }
      }
    }

    cat(sprintf("Saved multivariate ensemble outputs to: %s\n", mv_dir))
    invisible(NULL)
  }

  if (!is.null(multivariate_results$standard) && length(multivariate_results$standard) > 0) {
    save_mv_bundle(multivariate_results$standard, "standard")
  }
  if (!is.null(multivariate_results$with_leftout) && length(multivariate_results$with_leftout) > 0) {
    save_mv_bundle(multivariate_results$with_leftout, "with_leftout")
  }

  # Prediction-level decisions and accepted support (simple Platt Global Ensemble).
  for (type in c("cv", "loso")) {
    if ("Global_Optimized" %in% names(all_probability_matrices) &&
        type %in% names(all_probability_matrices[["Global_Optimized"]])) {
      std_decisions <- build_global_prediction_decisions(
        all_probability_matrices[["Global_Optimized"]][[type]],
        type = type,
        regime = "standard",
        method = "platt_simple",
        model_label = "Global_Optimized",
        confidence_col = "confidence_calibrated"
      )
      if (nrow(std_decisions) > 0) {
        saveRDS(
          std_decisions,
          file.path(out_standard_dir, paste0("global_ensemble_prediction_decisions_standard_", type, ".rds"))
        )
        std_support <- build_accepted_support_by_class(
          std_decisions, type, "standard", "platt_simple", "Global_Optimized"
        )
        write.csv(
          std_support,
          file.path(out_standard_dir, paste0("accepted_support_by_class_", type, ".csv")),
          row.names = FALSE
        )
      }
    }

    if (has_leftout_data &&
        exists("all_augmented_matrices") &&
        "Global_Optimized" %in% names(all_augmented_matrices) &&
        type %in% names(all_augmented_matrices[["Global_Optimized"]])) {
      aug_decisions <- build_global_prediction_decisions(
        all_augmented_matrices[["Global_Optimized"]][[type]],
        type = type,
        regime = "with_leftout",
        method = "platt_simple",
        model_label = "Global_Optimized",
        confidence_col = "confidence_calibrated"
      )
      if (nrow(aug_decisions) > 0) {
        saveRDS(
          aug_decisions,
          file.path(out_with_leftout_dir, paste0("global_ensemble_prediction_decisions_with_leftout_", type, ".rds"))
        )
        aug_support <- build_accepted_support_by_class(
          aug_decisions, type, "with_leftout", "platt_simple", "Global_Optimized"
        )
        write.csv(
          aug_support,
          file.path(out_with_leftout_dir, paste0("accepted_support_by_class_", type, ".csv")),
          row.names = FALSE
        )
      }
    }
  }

  # Manifest to make downstream analysis reproducible.
  manifest_rows <- list(
    data.frame(key = "timestamp_utc", value = format(Sys.time(), tz = "UTC", usetz = TRUE), stringsAsFactors = FALSE),
    data.frame(key = "merge_suffix", value = merge_suffix, stringsAsFactors = FALSE),
    data.frame(key = "merge_classes", value = as.character(merge_classes), stringsAsFactors = FALSE),
    data.frame(key = "enable_cutoff_selection", value = as.character(ENABLE_CUTOFF_SELECTION), stringsAsFactors = FALSE),
    data.frame(key = "has_leftout_data", value = as.character(has_leftout_data), stringsAsFactors = FALSE),
    data.frame(key = "global_model_simple", value = "Global_Optimized", stringsAsFactors = FALSE),
    data.frame(key = "global_model_multivariate", value = "Global_Optimized_MV / Global_Optimized_MV_leftout", stringsAsFactors = FALSE)
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
