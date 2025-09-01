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

# Model types and their configurations for outer CV
OUTER_MODEL_CONFIGS <- list(
  svm = list(
    classification_type = "OvR",
    file_paths = list(
      cv = "../data/out/outer_cv/SVM_n10/SVM_outer_cv_CV_OvR_20250821_0926.csv",
      loso = "../data/out/outer_cv/SVM_n10/SVM_outer_cv_loso_OvR_20250821_0929.csv"
    )
  ),
  xgboost = list(
    classification_type = "OvR",
    file_paths = list(
      cv = "../data/out/outer_cv/XGBOOST_n10/XGBOOST_outer_cv_CV_OvR_20250821_0932.csv",
      loso = "../data/out/outer_cv/XGBOOST_n10/XGBOOST_outer_cv_loso_OvR_20250821_0936.csv"
    )
  ),
  neural_net = list(
    classification_type = "standard",
    file_paths = list(
      cv = "../data/out/outer_cv/NN_n10/NN_outer_cv_CV_standard_20250821_0940.csv",
      loso = "../data/out/outer_cv/NN_n10/NN_outer_cv_loso_standard_20250821_0951.csv"
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
WEIGHTS_BASE_DIR <- "inner_cv_best_params_n10/ensemble_weights_20aug/ensemble_weights"

# Base directory for rejection cut offs
REJECTION_BASE_DIR <- "inner_cv_best_params_n10/cutoffs_20aug"

# =============================================================================
# Source Utility Functions
# =============================================================================

source("R/utility_functions.R")

# =============================================================================
# Class Modification Functions
# =============================================================================

#' Modify class labels to group related subtypes
#' @param vector Vector of class labels
#' @return Modified vector with grouped classes
modify_classes <- function(vector) {
  vector[grepl("MDS|TP53|MECOM", vector)] <- "MDS.r.and.MECOM"
  vector[!grepl("MLLT3", vector) & grepl("KMT2A", vector)] <- "other.KMT2A"
  vector
}

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
#' @return List of probability matrices organized by outer fold
generate_outer_ovr_probability_matrices <- function(outer_cv_results, label_mapping) {
  cat("Generating outer One-vs-Rest probability matrices...\n")
  
  outer_fold_ids <- unique(outer_cv_results$outer_fold)
  probability_matrices <- list()
  
  for (outer_fold_id in outer_fold_ids) {
    cat(sprintf("  Processing outer fold %s...\n", as.character(outer_fold_id)))
    
    outer_fold_data <- outer_cv_results[outer_cv_results$outer_fold == outer_fold_id, ]
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
    
    probability_matrices[[as.character(outer_fold_id)]] <- probability_matrix
  }
  
  return(probability_matrices)
}

#' Generate outer CV probability matrices for standard multiclass classification
#' @param outer_cv_results Outer CV results data frame
#' @param label_mapping Label mapping data frame
#' @param filtered_subtypes Filtered leukemia subtypes
#' @return List of probability matrices organized by outer fold
generate_outer_standard_probability_matrices <- function(outer_cv_results, label_mapping, filtered_subtypes) {
  cat("Generating outer CV standard probability matrices...\n")
  
  outer_fold_ids <- unique(outer_cv_results$outer_fold)
  probability_matrices <- list()
  
  for (outer_fold_id in outer_fold_ids) {
    cat(sprintf("  Processing outer fold %s...\n", as.character(outer_fold_id)))
    
    fold_data <- outer_cv_results[outer_cv_results$outer_fold == outer_fold_id, ]
    
    if (nrow(fold_data) == 0) {
      warning(sprintf("No data for outer fold", outer_fold_id))
      next
    }
    
    # Take the first (and typically only) row for this fold
    fold_row <- fold_data[1, ]
    
    # Extract class information
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
    
    probability_matrices[[as.character(outer_fold_id)]] <- probability_matrix
  }
  
  return(probability_matrices)
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
        warning(sprintf("No OvR weights for fold %s, using equal weights", fold_name))
        fold_weights <- list()
        for (class_name in all_classes) {
          fold_weights[[gsub("Class.", "", class_name)]] <- list(weights = list(SVM = 1, XGB = 0, NN = 0))
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
          # Use equal weights as fallback
          class_weights <- list(SVM = 1, XGB = 0, NN = 0)
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
        warning(sprintf("No global weights for fold %s, using equal weights", fold_name))
        fold_weights <- list(weights = list(SVM = 1, XGB = 0, NN = 0))
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
      truth <- modify_classes(truth)
      preds <- modify_classes(preds)
      
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
  optimal_cutoffs <- all_cutoffs[all_cutoffs$type == type, ]
  
  if (nrow(optimal_cutoffs) == 0) {
    warning(sprintf("No cutoffs found for type: %s", type))
    return(NULL)
  }
  
  cat(sprintf("  Loaded %d cutoffs for %s analysis from: %s\n", nrow(optimal_cutoffs), toupper(type), optimal_cutoffs_file))
  
  # Calculate summary statistics
  summary_stats <- optimal_cutoffs %>%
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
      n_folds = n(),
      .groups = "drop"
    )
  
  return(list(
    optimal_cutoffs = optimal_cutoffs,
    summary_stats = summary_stats
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
    
    # Get cutoff for this model (use mean across folds)
    model_cutoffs <- optimal_cutoffs$optimal_cutoffs[optimal_cutoffs$optimal_cutoffs$model == model_name, ]
    if (nrow(model_cutoffs) == 0) {
      cat(sprintf("    No cutoffs found for %s, skipping\n", model_name))
      next
    }
    
    # Use mean cutoff across folds
    mean_cutoff <- mean(model_cutoffs$prob_cutoff, na.rm = TRUE)
    cat(sprintf("    Using cutoff %.3f for %s\n", mean_cutoff, model_name))
    
    # Find corresponding probability matrix
    if (model_name %in% names(probability_matrices) && type %in% names(probability_matrices[[model_name]])) {
      model_matrices <- probability_matrices[[model_name]][[type]]
      
      for (fold_name in names(model_matrices)) {
        prob_matrix <- model_matrices[[fold_name]]
        
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
    } else {
      # Check if it's an ensemble method
      ensemble_found <- FALSE
      for (ensemble_type in c("OvR_Ensemble", "Global_Optimized")) {
        if (ensemble_type %in% names(probability_matrices) && type %in% names(probability_matrices[[ensemble_type]])) {
          ensemble_matrices <- probability_matrices[[ensemble_type]][[type]]
          
          for (fold_name in names(ensemble_matrices)) {
            prob_matrix <- ensemble_matrices[[fold_name]]
            
            if (!is.null(prob_matrix) && nrow(prob_matrix) > 0) {
              # Apply rejection analysis
              rejection_result <- evaluate_single_matrix_with_rejection_and_cutoff(
                prob_matrix, fold_name, model_name, type, mean_cutoff
              )
              
              if (!is.null(rejection_result)) {
                rejection_results[[paste(model_name, fold_name, sep = "_")]] <- rejection_result
              }
              ensemble_found <- TRUE
            }
          }
        }
      }
      
      if (!ensemble_found) {
        cat(sprintf("    No probability matrices found for %s, skipping\n", model_name))
      }
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
  # Extract true labels and remove from probability matrix
  truth <- prob_matrix$y
  prob_matrix_clean <- prob_matrix[, !colnames(prob_matrix) %in% c("y", "outer_fold", "sample_indices"), drop = FALSE]
  
  # Clean class labels
  truth <- gsub("Class.", "", truth)
  truth <- modify_classes(truth)
  
  # Get predictions (class with highest probability)
  pred_indices <- apply(prob_matrix_clean, 1, which.max)
  preds <- colnames(prob_matrix_clean)[pred_indices]
  preds <- gsub("Class.", "", preds)
  preds <- modify_classes(preds)
  
  # Get max probabilities for each sample
  max_probs <- apply(prob_matrix_clean, 1, max)
  
  # Ensure all classes are represented
  all_classes <- unique(c(truth, preds))
  truth <- factor(truth, levels = all_classes)
  preds <- factor(preds, levels = all_classes)
  
  # Apply rejection using the specific cutoff
  rejected_indices <- which(max_probs < cutoff)
  accepted_indices <- which(max_probs >= cutoff)
  
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
# Main Outer CV Analysis Function
# =============================================================================

#' Main function to run outer CV analysis
main_outer_cv <- function() {
  # Load required libraries
  load_library_quietly("plyr")
  load_library_quietly("dplyr")
  load_library_quietly("stringr")
  load_library_quietly("caret")
  load_library_quietly("data.table")
  
  cat("=== Starting Outer Cross-Validation Analysis ===\n")
  
  # Load label mapping and data
  cat("Loading label mapping and data...\n")
  label_mapping <- safe_read_file("label_mapping_all.csv", read.csv)
  if (is.null(label_mapping)) {
    stop("Failed to load label mapping file")
  }
  
  # Load leukemia subtype data
  leukemia_subtypes <- safe_read_file("data/rgas_20aug25.csv", function(f) read.csv(f)$ICC_Subtype)
  if (is.null(leukemia_subtypes)) {
    stop("Failed to load leukemia subtype data")
  }
  
  # Load study metadata
  study_names <- safe_read_file("data/meta_20aug25.csv", function(f) read.csv(f)$Studies)
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
      results <- load_outer_cv_results(file_path, config$classification_type)
      
      if (!is.null(results)) {
        outer_cv_data[[model_name]][[fold_type]] <- results
      }
    }
  }
  
  # Generate probability matrices from outer CV results
  cat("Generating outer CV probability matrices...\n")
  outer_probability_matrices <- list()
  
  for (model_name in names(outer_cv_data)) {
    config <- OUTER_MODEL_CONFIGS[[model_name]]
    cat(sprintf("Processing %s probabilities...\n", toupper(model_name)))
    
    outer_probability_matrices[[model_name]] <- list()
    
    for (fold_type in names(outer_cv_data[[model_name]])) {
      results <- outer_cv_data[[model_name]][[fold_type]]
      
      if (!is.null(results)) {
        if (config$classification_type == "OvR") {
          probs <- generate_outer_ovr_probability_matrices(results, label_mapping)
        } else {
          probs <- generate_outer_standard_probability_matrices(results, label_mapping, filtered_leukemia_subtypes)
        }
        probs <- lapply(probs, function(prob) {
          MDS_cols <- grepl("MDS|TP53|MECOM", colnames(prob))
          MDS <- rowSums(prob[,MDS_cols])
          
          other_KMT2A_cols <- grepl("KMT2A", colnames(prob)) & !grepl("MLLT3", colnames(prob))
          other_KMT2A <- rowSums(prob[,other_KMT2A_cols])
          
          prob <- prob[, !(MDS_cols | other_KMT2A_cols)]
          
          prob$MDS.r <- MDS
          prob$other.KMT2A <- other_KMT2A
          prob$y <- modify_classes(prob$y)
          colnames(prob) <- modify_classes(colnames(prob))
          return(prob)
        })
        outer_probability_matrices[[model_name]][[fold_type]] <- probs
      }
    }
  }
  
  # Load ensemble weights from inner CV analysis
  cat("Loading ensemble weights from inner CV analysis...\n")
  ensemble_weights <- list()
  
  for (type in c("cv", "loso")) {
    weights_data <- tryCatch({
      load_ensemble_weights(WEIGHTS_BASE_DIR, type)
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
    
    # Generate OvR ensemble
    ovr_ensemble <- apply_ensemble_weights_to_outer_cv(
      outer_probability_matrices, ensemble_weights[[type]], type, "ovr"
    )
    if (!is.null(ovr_ensemble)) {
      ensemble_matrices[[type]][["ovr_ensemble"]] <- ovr_ensemble
    }
    
    # Generate global ensemble
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
      for (ensemble_name in names(ensemble_matrices[[type]])) {
        # Map ensemble names to match rejection analysis expectations
        mapped_name <- if (ensemble_name == "ovr_ensemble") "OvR_Ensemble" else "Global_Optimized"
        
        if (!mapped_name %in% names(all_probability_matrices)) {
          all_probability_matrices[[mapped_name]] <- list()
        }
        all_probability_matrices[[mapped_name]][[type]] <- ensemble_matrices[[type]][[ensemble_name]]
      }
    }
  }
  
  # Calculate performance for all models and ensembles
  cat("Calculating performance metrics...\n")
  detailed_performance <- list()
  performance_summaries <- list()
  per_class_summaries <- list()
  
  for (type in c("cv", "loso")) {
    cat(sprintf("Analyzing %s performance...\n", toupper(type)))
    
    # Calculate detailed performance for all models and ensembles
    detailed_performance[[type]] <- calculate_outer_cv_performance(all_probability_matrices, type)
    
    # Generate performance summary
    performance_summaries[[type]] <- summarize_outer_cv_performance(detailed_performance[[type]])
    
    # Generate per-class performance summary
    per_class_summaries[[type]] <- summarize_per_class_performance(detailed_performance[[type]])
    
    cat(sprintf("\n=== %s Performance Summary ===\n", toupper(type)))
    print(performance_summaries[[type]])
    
    cat(sprintf("\n=== %s Per-Class Performance Summary ===\n", toupper(type)))
    print(per_class_summaries[[type]])
  }
  
  # Load optimal cutoffs for rejection analysis
  cat("Loading optimal cutoffs for rejection analysis...\n")
  optimal_cutoffs_data <- list()
  
  for (type in c("cv", "loso")) {
    optimal_cutoffs_data[[type]] <- load_optimal_cutoffs(REJECTION_BASE_DIR, type)
  }
  
  # Apply rejection analysis to outer CV results
  cat("Applying rejection analysis to outer CV results...\n")
  rejection_results <- list()
  
  for (type in c("cv", "loso")) {
    if (!type %in% names(optimal_cutoffs_data)) {
      warning(sprintf("No optimal cutoffs available for %s rejection analysis, skipping", type))
      next
    }
    
    cat(sprintf("Processing %s rejection analysis...\n", toupper(type)))
    rejection_results[[type]] <- apply_rejection_analysis_to_outer_cv(
      all_probability_matrices, optimal_cutoffs_data[[type]], type
    )
  }
  
  # Summarize rejection analysis
  cat("Summarizing rejection analysis...\n")
  rejection_summary <- list()
  
  for (type in c("cv", "loso")) {
    if (!type %in% names(rejection_results)) {
      warning(sprintf("No rejection analysis results for %s, skipping", type))
      next
    }
    
    rejection_summary[[type]] <- summarize_rejection_analysis(rejection_results[[type]], type)
    
    # Display per-class rejection analysis summary
    if (!is.null(rejection_summary[[type]]$per_class_summary)) {
      cat(sprintf("\n=== %s Rejection Analysis Per-Class Performance Summary ===\n", toupper(type)))
      print(rejection_summary[[type]]$per_class_summary)
    }
  }
  
  # Compare performance with and without rejection
  cat("Comparing performance with and without rejection...\n")
  performance_comparison <- list()
  
  for (type in c("cv", "loso")) {
    if (!type %in% names(rejection_summary)) {
      warning(sprintf("No rejection analysis results for %s, skipping performance comparison", type))
      next
    }
    
    performance_comparison[[type]] <- compare_performance_with_rejection(
      detailed_performance[[type]], rejection_summary[[type]], type
    )
  }
  
  # Save all results
  outer_cv_results <- list(
    outer_cv_data = outer_cv_data,
    outer_probability_matrices = outer_probability_matrices,
    ensemble_matrices = ensemble_matrices,
    detailed_performance = detailed_performance,
    performance_summaries = performance_summaries,
    per_class_summaries = per_class_summaries,
    ensemble_weights_used = ensemble_weights,
    rejection_analysis_results = rejection_results,
    rejection_summary = rejection_summary,
    rejection_per_class_summaries = lapply(rejection_summary, function(x) x$per_class_summary),
    performance_comparison = performance_comparison
  )
  saveRDS(inner_cv_results, "../data/out/inner_cv/inner_cv_results.rds")
  return(outer_cv_results)
}

# Run the analysis if this script is executed directly
if (!exists("SKIP_OUTER_CV_EXECUTION")) {
  outer_cv_results <- main_outer_cv()
} 
