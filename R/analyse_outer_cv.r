# =============================================================================
# Outer Cross-Validation Analysis for Machine Learning Models
# =============================================================================
# This script analyzes outer cross-validation results for SVM, XGBoost, and 
# Neural Network models, generates final prediction probability matrices,
# performs ensemble analysis using optimized weights from inner CV, and 
# evaluates final model performance.
# =============================================================================

# Set working directory
setwd("~/Documents/AML_PhD/leukem_ai")

# =============================================================================
# Configuration and Constants
# =============================================================================

# Model types and their configurations for outer CV
OUTER_MODEL_CONFIGS <- list(
  svm = list(
    classification_type = "OvR",
    file_paths = list(
      cv = "out/outer_cv/SVM_n10/SVM_outer_cv_CV_OvR_20250703_1254.csv",
      loso = "out/outer_cv/SVM_n10/SVM_outer_cv_loso_OvR_20250703_1309.csv"
    ),
    output_dir = "inner_cv_best_params_n10/SVM"
  ),
  xgboost = list(
    classification_type = "OvR",
    file_paths = list(
      cv = "out/outer_cv/XGBOOST_n10/XGBOOST_outer_cv_CV_OvR_20250703_1259.csv",
      loso = "out/outer_cv/XGBOOST_n10/XGBOOST_outer_cv_loso_OvR_20250703_1312.csv"
    ),
    output_dir = "inner_cv_best_params_n10/XGBOOST"
  ),
  neural_net = list(
    classification_type = "standard",
    file_paths = list(
      cv = "out/outer_cv/NN_n10/NN_outer_cv_CV_standard_20250731_1756.csv",
      loso = "out/outer_cv/NN_n10/NN_outer_cv_loso_standard_20250731_1807.csv"
    ),
    output_dir = "inner_cv_best_params_n10/NN"
  )
)

# Data filtering criteria (same as inner CV)
DATA_FILTERS <- list(
  min_samples_per_subtype = 10,
  excluded_subtypes = c("AML NOS", "Missing data"),
  selected_studies = c(
    "TCGA-LAML",
    "LEUCEGENE", 
    "BEATAML1.0-COHORT",
    "AAML0531",
    "AAML1031"
  )
)

# Base directory for ensemble weights
WEIGHTS_BASE_DIR <- "inner_cv_best_params_n10/ensemble_weights"

# Base directory for rejection cut offs
REJECTION_BASE_DIR <- "inner_cv_best_params_n10/rejection_analysis"

# =============================================================================
# Source Utility Functions
# =============================================================================

source("R/utility_functions.R")

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
          fold_weights[[gsub("Class.", "", class_name)]] <- list(weights = list(SVM = 1, XGB = 1, NN = 1))
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
          class_weights <- list(SVM = 1, XGB = 1, NN = 1)
        }
        
        # Calculate weighted ensemble for this class
        ensemble_matrix[, class_name] <- 
          svm_probs[[class_name]] * class_weights$SVM +
          xgb_probs[[class_name]] * class_weights$XGB +
          nn_probs[[class_name]] * class_weights$NN
      }
      
    } else {
      # Use global weights
      fold_weights <- weights_to_use[[fold_name]]
      if (is.null(fold_weights)) {
        warning(sprintf("No global weights for fold %s, using equal weights", fold_name))
        fold_weights <- list(weights = list(SVM = 1, XGB = 1, NN = 1))
      }
      
      weights <- fold_weights$weights
      
      # Calculate weighted ensemble
      ensemble_matrix <- svm_probs * weights$SVM +
                        xgb_probs * weights$XGB +
                        nn_probs * weights$NN
    }
    
    # Normalize probabilities
    ensemble_matrix <- t(apply(ensemble_matrix, 1, function(row) {
      if (sum(row, na.rm = TRUE) > 0) {
        row / sum(row, na.rm = TRUE)
      } else {
        row
      }
    }))
    
    # Convert to data frame and add metadata
    ensemble_matrix <- data.frame(ensemble_matrix)
    ensemble_matrix$y <- truth
    ensemble_matrix$outer_fold <- as.numeric(fold_name)
    
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
      prob_cols <- prob_matrix[, !colnames(prob_matrix) %in% c("y", "outer_fold", "sample_indices"), drop = FALSE]
      
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
      
      model_performance[[fold_name]] <- list(
        confusion_matrix = cm,
        kappa = as.numeric(cm$overall["Kappa"]),
        accuracy = as.numeric(cm$overall["Accuracy"]),
        balanced_accuracy = mean(cm$byClass[, "Balanced Accuracy"], na.rm = TRUE),
        f1_macro = mean(cm$byClass[, "F1"], na.rm = TRUE),
        sensitivity_macro = mean(cm$byClass[, "Sensitivity"], na.rm = TRUE),
        specificity_macro = mean(cm$byClass[, "Specificity"], na.rm = TRUE)
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

#' Generate comprehensive performance comparison plots
#' @param performance_summary Summary data frame from summarize_outer_cv_performance
#' @param output_dir Directory to save plots
#' @param type Type of analysis ("cv" or "loso")
generate_outer_cv_performance_plots <- function(performance_summary, output_dir, type = "cv") {
  cat("Generating outer CV performance plots...\n")
  
  # Load plotting libraries
  load_library_quietly("ggplot2")
  load_library_quietly("reshape2")
  
  # Create output directory
  create_directory_safely(output_dir)
  
  # Prepare data for plotting
  plot_data <- performance_summary[, c("Model", "Mean_Kappa", "Mean_Accuracy", "Mean_Balanced_Accuracy", "Mean_F1_Macro")]
  plot_data_long <- reshape2::melt(plot_data, id.vars = "Model", variable.name = "Metric", value.name = "Value")
  plot_data_long$Metric <- gsub("Mean_", "", plot_data_long$Metric)
  
  # Plot 1: Performance comparison across metrics
  p1 <- ggplot(plot_data_long, aes(x = Model, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = sprintf("Outer CV Performance Comparison (%s)", toupper(type)),
         x = "Model",
         y = "Performance Value") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    facet_wrap(~Metric, scales = "free_y")
  
  # Plot 2: Kappa comparison with error bars
  error_data <- performance_summary[, c("Model", "Mean_Kappa", "SD_Kappa")]
  p2 <- ggplot(error_data, aes(x = reorder(Model, Mean_Kappa), y = Mean_Kappa)) +
    geom_col(fill = "steelblue", alpha = 0.7) +
    geom_errorbar(aes(ymin = Mean_Kappa - SD_Kappa, ymax = Mean_Kappa + SD_Kappa), 
                  width = 0.2, color = "black") +
    labs(title = sprintf("Kappa Performance with Standard Deviation (%s)", toupper(type)),
         x = "Model",
         y = "Kappa") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    coord_flip()
  
  # Save plots
  ggsave(file.path(output_dir, sprintf("outer_cv_performance_comparison_%s.png", type)), p1, width = 12, height = 8)
  ggsave(file.path(output_dir, sprintf("outer_cv_kappa_comparison_%s.png", type)), p2, width = 10, height = 6)
  
  cat(sprintf("  Plots saved to: %s\n", output_dir))
}

#' Save outer CV results and performance metrics
#' @param outer_cv_results All outer CV results and performance data
#' @param output_base_dir Base directory for saving results
save_outer_cv_results <- function(outer_cv_results, output_base_dir) {
  cat("Saving outer CV results...\n")
  
  for (type in c("cv", "loso")) {
    if (!type %in% names(outer_cv_results$performance_summaries)) next
    
    type_output_dir <- file.path(output_base_dir, "outer_cv_analysis", type)
    create_directory_safely(type_output_dir)
    
    # Save performance summary
    performance_summary <- outer_cv_results$performance_summaries[[type]]
    write.csv(performance_summary, 
              file.path(type_output_dir, "performance_summary.csv"), 
              row.names = FALSE)
    
    # Save detailed performance results
    detailed_performance <- outer_cv_results$detailed_performance[[type]]
    for (model_name in names(detailed_performance)) {
      model_dir <- file.path(type_output_dir, model_name)
      create_directory_safely(model_dir)
      
      for (fold_name in names(detailed_performance[[model_name]])) {
        fold_result <- detailed_performance[[model_name]][[fold_name]]
        
        # Save confusion matrix
        cm_file <- file.path(model_dir, sprintf("confusion_matrix_fold_%s.txt", fold_name))
        capture.output(print(fold_result$confusion_matrix), file = cm_file)
        
        # Save metrics
        metrics <- data.frame(
          Metric = c("Kappa", "Accuracy", "Balanced_Accuracy", "F1_Macro", "Sensitivity_Macro", "Specificity_Macro"),
          Value = c(fold_result$kappa, fold_result$accuracy, fold_result$balanced_accuracy, 
                   fold_result$f1_macro, fold_result$sensitivity_macro, fold_result$specificity_macro),
          stringsAsFactors = FALSE
        )
        write.csv(metrics, 
                  file.path(model_dir, sprintf("metrics_fold_%s.csv", fold_name)), 
                  row.names = FALSE)
      }
    }
    
    cat(sprintf("  Saved %s results to: %s\n", toupper(type), type_output_dir))
  }
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
  label_mapping <- safe_read_file("label_mapping_df_n10.csv", read.csv)
  if (is.null(label_mapping)) {
    stop("Failed to load label mapping file")
  }
  
  # Load leukemia subtype data
  leukemia_subtypes <- safe_read_file("data/rgas.csv", function(f) read.csv(f)$ICC_Subtype)
  if (is.null(leukemia_subtypes)) {
    stop("Failed to load leukemia subtype data")
  }
  
  # Load study metadata
  study_names <- safe_read_file("data/meta.csv", function(f) read.csv(f)$Studies)
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
        if (!ensemble_name %in% names(all_probability_matrices)) {
          all_probability_matrices[[ensemble_name]] <- list()
        }
        all_probability_matrices[[ensemble_name]][[type]] <- ensemble_matrices[[type]][[ensemble_name]]
      }
    }
  }
  
  # Calculate performance for all models and ensembles
  cat("Calculating performance metrics...\n")
  detailed_performance <- list()
  performance_summaries <- list()
  
  for (type in c("cv", "loso")) {
    cat(sprintf("Analyzing %s performance...\n", toupper(type)))
    
    # Calculate detailed performance
    detailed_performance[[type]] <- calculate_outer_cv_performance(all_probability_matrices, type)
    
    # Generate performance summary
    performance_summaries[[type]] <- summarize_outer_cv_performance(detailed_performance[[type]])
    
    cat(sprintf("\n=== %s Performance Summary ===\n", toupper(type)))
    print(performance_summaries[[type]])
  }
  
  # Generate plots and save results
  output_base_dir <- "inner_cv_best_params_n10"
  
  for (type in c("cv", "loso")) {
    if (type %in% names(performance_summaries)) {
      plot_output_dir <- file.path(output_base_dir, "outer_cv_analysis", type, "plots")
      generate_outer_cv_performance_plots(performance_summaries[[type]], plot_output_dir, type)
    }
  }
  
  # Save all results
  outer_cv_results <- list(
    outer_cv_data = outer_cv_data,
    outer_probability_matrices = outer_probability_matrices,
    ensemble_matrices = ensemble_matrices,
    detailed_performance = detailed_performance,
    performance_summaries = performance_summaries,
    ensemble_weights_used = ensemble_weights
  )
  
  save_outer_cv_results(outer_cv_results, output_base_dir)
  
  cat("=== Outer Cross-Validation Analysis Complete! ===\n")
  
  return(outer_cv_results)
}

# Run the analysis if this script is executed directly
if (!exists("SKIP_OUTER_CV_EXECUTION")) {
  outer_cv_results <- main_outer_cv()
} 
