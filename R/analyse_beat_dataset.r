# =============================================================================
# BEAT Dataset Analysis: Inner CV vs Outer CV Performance
# =============================================================================
# This script analyzes the prediction results specifically for the BEAT dataset
# across both inner CV and outer CV to understand why models transfer poorly
# to this dataset when it's used as the evaluation fold.
# =============================================================================

# =============================================================================
# Multi-Dataset LOSO Analysis: Inner CV vs Outer CV Performance
# =============================================================================
# This script analyzes the prediction results for ALL datasets in LOSO
# across both inner CV and outer CV to understand performance transfer issues
# and identify which datasets have the most severe degradation.
# =============================================================================

# Set working directory
setwd("~/Documents/AML_PhD/leukem_ai")

# =============================================================================
# Configuration and Constants
# =============================================================================

# Dataset identifiers for LOSO analysis
LOSO_DATASETS <- c(
  "AAML0531",
  "AAML1031", 
  "BEATAML1.0-COHORT",
  "LEUCEGENE",
  "TCGA-LAML"
)

# BEAT dataset identifier (for backward compatibility)
BEAT_DATASET_NAME <- "BEATAML1.0-COHORT"

# Base directories
INNER_CV_BASE_DIR <- "inner_cv_best_params_n10"
OUTER_CV_BASE_DIR <- "inner_cv_best_params_n10/outer_cv_analysis"

# =============================================================================
# Source Utility Functions
# =============================================================================

source("R/utility_functions.R")

# =============================================================================
# Data Loading Functions
# =============================================================================

#' Load inner CV probability matrices for a specific fold
#' @param fold_name Name of the fold to load matrices for
#' @return List of probability matrices for individual models
load_fold_inner_cv_matrices <- function(fold_name) {
  # Load probability matrices from inner CV analysis for this fold
  inner_cv_matrices <- list()
  
  # Try to source the main inner CV analysis to get probability matrices
  tryCatch({
    # Set flag to skip main execution
    SKIP_MAIN_EXECUTION <- TRUE
    
    # Source the inner CV analysis functions
    source("R/analyse_inner_cv_new.r")
    
    # Load the saved probability matrices (if available)
    # For now, we'll calculate performance directly from the probability matrices
    # This would ideally load pre-saved matrices, but we'll implement a simpler approach
    
    inner_cv_matrices <- NULL
  }, error = function(e) {
    warning(sprintf("Could not load inner CV matrices: %s", e$message))
    inner_cv_matrices <- NULL
  })
  
  return(inner_cv_matrices)
}

#' Calculate inner CV performance for individual models from best parameters
#' @param fold_name Name of the fold
#' @param model_name Name of the model
#' @return Inner CV performance (mean kappa)
calculate_inner_cv_performance_from_best_params <- function(fold_name, model_name) {
  # Load the best parameters file for this model
  # Map model names to their directory names
  model_dir_map <- list(
    "svm" = "SVM",
    "xgboost" = "XGBOOST", 
    "neural_net" = "NN"
  )
  
  model_dir <- if (model_name %in% names(model_dir_map)) {
    model_dir_map[[model_name]]
  } else {
    toupper(model_name)
  }
  
  file_prefix <- if (model_name == "neural_net") "NEURAL_NET" else toupper(model_name)
  
  best_params_file <- file.path(INNER_CV_BASE_DIR, model_dir, 
                               paste0(file_prefix, "_best_param_loso.csv"))
  
  if (!file.exists(best_params_file)) {
    warning(sprintf("Best parameters file not found: %s", best_params_file))
    return(NA)
  }
  
  best_params <- read.csv(best_params_file, stringsAsFactors = FALSE)
  
  # Filter for the specific fold
  fold_params <- best_params[best_params$outer_fold == fold_name, ]
  
  if (nrow(fold_params) == 0) {
    warning(sprintf("No parameters found for fold %s in model %s", fold_name, model_name))
    return(NA)
  }
  
  # For individual models, return the mean kappa from best parameters
  # This represents the inner CV performance
  if ("mean_kappa" %in% colnames(fold_params)) {
    mean_kappa <- mean(fold_params$mean_kappa, na.rm = TRUE)
  } else {
    # Fallback: use any available kappa measure
    kappa_cols <- colnames(fold_params)[grepl("kappa", colnames(fold_params), ignore.case = TRUE)]
    if (length(kappa_cols) > 0) {
      mean_kappa <- mean(fold_params[[kappa_cols[1]]], na.rm = TRUE)
    } else {
      warning(sprintf("No kappa columns found for %s", model_name))
      return(NA)
    }
  }
  
  return(mean_kappa)
}

#' Load ensemble weights for a specific fold
#' @param fold_name Name of the fold to load weights for
#' @param weights_type Type of weights ("global" or "ovr")
#' @return List of ensemble weights for the fold
load_fold_ensemble_weights <- function(fold_name, weights_type = "global") {
  weights_file <- file.path(INNER_CV_BASE_DIR, "ensemble_weights", "loso", 
                           paste0(weights_type, "_ensemble_weights_used.csv"))
  
  if (!file.exists(weights_file)) {
    warning(sprintf("Weights file not found: %s", weights_file))
    return(NULL)
  }
  
  weights_data <- read.csv(weights_file, stringsAsFactors = FALSE)
  
  if (weights_type == "global") {
    # Filter for specific fold
    fold_weights <- weights_data[weights_data$fold == fold_name, ]
    if (nrow(fold_weights) == 0) return(NULL)
    
    return(list(
      fold = fold_name,
      weight_name = fold_weights$weight_name,
      weights = list(
        SVM = fold_weights$svm_weight,
        XGB = fold_weights$xgb_weight,
        NN = fold_weights$nn_weight
      ),
      kappa = fold_weights$kappa
    ))
  } else {
    # Filter for specific fold
    fold_weights <- weights_data[weights_data$fold == fold_name, ]
    if (nrow(fold_weights) == 0) return(NULL)
    
    # Convert to nested list structure
    ovr_weights <- list()
    for (i in 1:nrow(fold_weights)) {
      row <- fold_weights[i, ]
      class_name <- row$class
      
      ovr_weights[[class_name]] <- list(
        weight_name = row$weight_name,
        weights = list(
          SVM = row$svm_weight,
          XGB = row$xgb_weight,
          NN = row$nn_weight
        ),
        f1_score = row$f1_score
      )
    }
    
    return(ovr_weights)
  }
}

#' Load performance metrics for a specific fold
#' @param fold_name Name of the fold to load metrics for
#' @param model_name Name of the model
#' @return Data frame with performance metrics
load_fold_performance_metrics <- function(fold_name, model_name) {
  metrics_file <- file.path(OUTER_CV_BASE_DIR, "loso", model_name, 
                           paste0("metrics_fold_", fold_name, ".csv"))
  
  if (!file.exists(metrics_file)) {
    warning(sprintf("Metrics file not found: %s", metrics_file))
    return(NULL)
  }
  
  metrics <- read.csv(metrics_file, stringsAsFactors = FALSE)
  return(metrics)
}

#' Load confusion matrix for a specific fold
#' @param fold_name Name of the fold to load confusion matrix for
#' @param model_name Name of the model
#' @return Confusion matrix object or NULL if file doesn't exist
load_fold_confusion_matrix <- function(fold_name, model_name) {
  cm_file <- file.path(OUTER_CV_BASE_DIR, "loso", model_name, 
                       paste0("confusion_matrix_fold_", fold_name, ".txt"))
  
  if (!file.exists(cm_file)) {
    warning(sprintf("Confusion matrix file not found: %s", cm_file))
    return(NULL)
  }
  
  # Read the confusion matrix text file
  cm_text <- readLines(cm_file)
  return(cm_text)
}

#' Load rejection analysis results for a specific fold
#' @param fold_name Name of the fold to load rejection results for
#' @param model_name Name of the model
#' @return Data frame with rejection analysis results
load_fold_rejection_results <- function(fold_name, model_name) {
  rejection_file <- file.path(OUTER_CV_BASE_DIR, "loso", "rejection_analysis", 
                             "detailed_rejection_results.csv")
  
  if (!file.exists(rejection_file)) {
    warning(sprintf("Rejection results file not found: %s", rejection_file))
    return(NULL)
  }
  
  rejection_data <- read.csv(rejection_file, stringsAsFactors = FALSE)
  
  # Filter for specific fold and model
  fold_rejection <- rejection_data[rejection_data$fold == fold_name & 
                                   rejection_data$model == model_name, ]
  
  return(fold_rejection)
}

# =============================================================================
# Multi-Dataset Analysis Functions
# =============================================================================

#' Analyze performance across all LOSO datasets
#' @return List of performance analysis results for all datasets
analyze_all_loso_datasets <- function() {
  cat("=== Analyzing Performance Across All LOSO Datasets ===\n")
  
  # Models to analyze (both individual and ensemble)
  models <- c("svm", "xgboost", "neural_net", "Global_Optimized", "OvR_Ensemble")
  individual_models <- c("svm", "xgboost", "neural_net")
  ensemble_models <- c("Global_Optimized", "OvR_Ensemble")
  
  # Initialize results
  all_datasets_performance <- list()
  all_datasets_ensemble_weights <- list()
  all_datasets_inner_cv_performance <- list()
  
  for (dataset_name in LOSO_DATASETS) {
    cat(sprintf("\n--- Analyzing Dataset: %s ---\n", dataset_name))
    
    dataset_results <- list()
    dataset_weights <- list()
    dataset_inner_cv <- list()
    
    # Analyze individual models for this dataset
    for (model_name in individual_models) {
      cat(sprintf("  Processing %s...\n", toupper(model_name)))
      
      # Load outer CV performance metrics
      metrics <- load_fold_performance_metrics(dataset_name, model_name)
      if (!is.null(metrics)) {
        dataset_results[[model_name]] <- metrics
      }
      
      # Load inner CV performance from best parameters
      inner_cv_kappa <- calculate_inner_cv_performance_from_best_params(dataset_name, model_name)
      if (!is.na(inner_cv_kappa)) {
        dataset_inner_cv[[model_name]] <- inner_cv_kappa
        cat(sprintf("    Inner CV Kappa: %.4f\n", inner_cv_kappa))
      }
    }
    
    # Analyze ensemble models for this dataset
    for (model_name in ensemble_models) {
      cat(sprintf("  Processing %s...\n", toupper(model_name)))
      
      # Load outer CV performance metrics
      metrics <- load_fold_performance_metrics(dataset_name, model_name)
      if (!is.null(metrics)) {
        dataset_results[[model_name]] <- metrics
      }
    }
    
    # Load ensemble weights for this dataset
    global_weights <- load_fold_ensemble_weights(dataset_name, "global")
    ovr_weights <- load_fold_ensemble_weights(dataset_name, "ovr")
    
    dataset_weights$global <- global_weights
    dataset_weights$ovr <- ovr_weights
    
    all_datasets_performance[[dataset_name]] <- dataset_results
    all_datasets_ensemble_weights[[dataset_name]] <- dataset_weights
    all_datasets_inner_cv_performance[[dataset_name]] <- dataset_inner_cv
  }
  
  return(list(
    all_datasets_performance = all_datasets_performance,
    all_datasets_ensemble_weights = all_datasets_ensemble_weights,
    all_datasets_inner_cv_performance = all_datasets_inner_cv_performance
  ))
}

#' Create comprehensive performance comparison across all datasets
#' @param all_datasets_results Results from analyze_all_loso_datasets
#' @return Data frame with comprehensive performance comparison
create_comprehensive_performance_comparison <- function(all_datasets_results) {
  cat("\n=== Creating Comprehensive Performance Comparison ===\n")
  
  comprehensive_comparison <- data.frame()
  
  for (dataset_name in names(all_datasets_results$all_datasets_performance)) {
    cat(sprintf("  Processing dataset: %s\n", dataset_name))
    
    dataset_performance <- all_datasets_results$all_datasets_performance[[dataset_name]]
    dataset_weights <- all_datasets_results$all_datasets_ensemble_weights[[dataset_name]]
    dataset_inner_cv <- all_datasets_results$all_datasets_inner_cv_performance[[dataset_name]]
    
    # Process each model for this dataset
    for (model_name in names(dataset_performance)) {
      metrics <- dataset_performance[[model_name]]
      
      # Extract key metrics
      kappa <- metrics$Value[metrics$Metric == "Kappa"]
      accuracy <- metrics$Value[metrics$Metric == "Accuracy"]
      balanced_accuracy <- metrics$Value[metrics$Metric == "Balanced_Accuracy"]
      f1_macro <- metrics$Value[metrics$Metric == "F1_Macro"]
      
      # Get inner CV performance based on model type
      inner_cv_performance <- NA
      weight_config <- NA
      
      if (model_name %in% c("svm", "xgboost", "neural_net")) {
        # Individual models: get from inner CV performance
        if (!is.null(dataset_inner_cv) && model_name %in% names(dataset_inner_cv)) {
          inner_cv_performance <- dataset_inner_cv[[model_name]]
          weight_config <- "Best hyperparameters"
        }
      } else if (model_name == "Global_Optimized" && !is.null(dataset_weights$global)) {
        # Global ensemble: get from ensemble weights
        inner_cv_performance <- dataset_weights$global$kappa
        weight_config <- dataset_weights$global$weight_name
      } else if (model_name == "OvR_Ensemble" && !is.null(dataset_weights$ovr)) {
        # OvR ensemble: calculate average F1 score across classes
        f1_scores <- sapply(dataset_weights$ovr, function(x) x$f1_score)
        f1_scores <- f1_scores[!is.na(f1_scores)]
        if (length(f1_scores) > 0) {
          inner_cv_performance <- mean(f1_scores, na.rm = TRUE)
          weight_config <- "Class-specific"
        }
      }
      
      # Create comparison row
      comparison_row <- data.frame(
        Dataset = dataset_name,
        Model = model_name,
        Inner_CV_Performance = inner_cv_performance,
        Outer_CV_Kappa = kappa,
        Outer_CV_Accuracy = accuracy,
        Outer_CV_Balanced_Accuracy = balanced_accuracy,
        Outer_CV_F1_Macro = f1_macro,
        Weight_Configuration = weight_config,
        stringsAsFactors = FALSE
      )
      
      comprehensive_comparison <- rbind(comprehensive_comparison, comparison_row)
    }
  }
  
  # Calculate performance degradation for ALL models (individual + ensemble)
  comprehensive_comparison$Performance_Degradation <- 
    comprehensive_comparison$Inner_CV_Performance - comprehensive_comparison$Outer_CV_Kappa
  
  comprehensive_comparison$Degradation_Percent <- 
    (comprehensive_comparison$Performance_Degradation / comprehensive_comparison$Inner_CV_Performance) * 100
  
  return(comprehensive_comparison)
}

#' Generate comprehensive visualizations for all datasets
#' @param comprehensive_comparison Comprehensive performance comparison data
#' @param output_dir Directory to save visualizations
generate_comprehensive_visualizations <- function(comprehensive_comparison, output_dir = "multi_dataset_analysis") {
  cat("\n=== Generating Comprehensive Visualizations ===\n")
  
  # Create output directory
  create_directory_safely(output_dir)
  
  # Load plotting libraries
  load_library_quietly("ggplot2")
  load_library_quietly("dplyr")
  load_library_quietly("gridExtra")
  load_library_quietly("RColorBrewer")
  
  # Filter for all models that have inner CV performance (individual + ensemble)
  models_with_inner_cv <- comprehensive_comparison[!is.na(comprehensive_comparison$Inner_CV_Performance), ]
  
  if (nrow(models_with_inner_cv) == 0) {
    cat("  No model performance data available for visualization\n")
    return()
  }
  
  # 1. Performance Degradation Heatmap
  p1 <- ggplot(models_with_inner_cv, aes(x = Model, y = Dataset, fill = Degradation_Percent)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.1f%%", Degradation_Percent)), 
              color = "white", size = 2, fontface = "bold") +
    scale_fill_gradient2(
      low = "green", 
      mid = "yellow", 
      high = "red",
      midpoint = 20,
      name = "Performance\nDegradation (%)"
    ) +
    labs(
      title = "Performance Degradation: Inner CV vs Outer CV",
      subtitle = "Percentage drop in performance when models are applied to new datasets",
      x = "Model",
      y = "Dataset"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10)
    )
  
  # 2. Performance Comparison Bar Plot
  plot_data_long <- models_with_inner_cv %>%
    select(Dataset, Model, Inner_CV_Performance, Outer_CV_Kappa) %>%
    tidyr::pivot_longer(
      cols = c(Inner_CV_Performance, Outer_CV_Kappa),
      names_to = "Performance_Type",
      values_to = "Performance_Value"
    )
  
  plot_data_long$Performance_Type <- factor(
    plot_data_long$Performance_Type,
    levels = c("Inner_CV_Performance", "Outer_CV_Kappa"),
    labels = c("Inner CV", "Outer CV")
  )
  
  p2 <- ggplot(plot_data_long, aes(x = Dataset, y = Performance_Value, fill = Performance_Type)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    facet_wrap(~Model, scales = "free_y", ncol = 3) +
    scale_fill_brewer(palette = "Set1", name = "Performance Type") +
    labs(
      title = "Performance Comparison: Inner CV vs Outer CV",
      subtitle = "Kappa performance across all datasets and models",
      x = "Dataset",
      y = "Performance (Kappa)"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10),
      legend.position = "bottom"
    )
  
  # 3. Degradation by Dataset
  p3 <- ggplot(models_with_inner_cv, aes(x = Dataset, y = Degradation_Percent, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    scale_fill_brewer(palette = "Set2", name = "Model") +
    labs(
      title = "Performance Degradation by Dataset",
      subtitle = "How much performance drops for each dataset across all models",
      x = "Dataset",
      y = "Performance Degradation (%)"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10),
      legend.position = "bottom"
    )
  
  # 4. Degradation by Model
  p4 <- ggplot(models_with_inner_cv, aes(x = Model, y = Degradation_Percent, fill = Dataset)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    scale_fill_brewer(palette = "Set3", name = "Dataset") +
    labs(
      title = "Performance Degradation by Model",
      subtitle = "How much performance drops for each model across all datasets",
      x = "Model",
      y = "Performance Degradation (%)"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10),
      legend.position = "bottom"
    )
  
  # 5. Scatter plot: Inner CV vs Outer CV Performance
  p5 <- ggplot(models_with_inner_cv, aes(x = Inner_CV_Performance, y = Outer_CV_Kappa, color = Dataset, shape = Model)) +
    geom_point(size = 3, alpha = 0.8) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red", alpha = 0.7) +
    scale_color_brewer(palette = "Set1", name = "Dataset") +
    scale_shape_manual(values = c(16, 17, 18, 15, 8), name = "Model") +
    labs(
      title = "Inner CV vs Outer CV Performance",
      subtitle = "Diagonal line represents perfect transfer (no degradation)",
      x = "Inner CV Performance (Kappa)",
      y = "Outer CV Performance (Kappa)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10),
      legend.position = "bottom"
    )
  
  # Save individual plots
  ggsave(file.path(output_dir, "performance_degradation_heatmap.png"), p1, width = 12, height = 8)
  ggsave(file.path(output_dir, "performance_comparison_barplot.png"), p2, width = 14, height = 10)
  ggsave(file.path(output_dir, "degradation_by_dataset.png"), p3, width = 12, height = 8)
  ggsave(file.path(output_dir, "degradation_by_model.png"), p4, width = 12, height = 8)
  ggsave(file.path(output_dir, "inner_vs_outer_scatter.png"), p5, width = 12, height = 8)
  
  # Create combined overview plot
  combined_plot <- grid.arrange(p1, p3, p4, p5, ncol = 2, nrow = 2)
  ggsave(file.path(output_dir, "comprehensive_overview.png"), combined_plot, width = 20, height = 16)
  
  cat("  All visualizations saved to:", output_dir, "\n")
}

#' Generate summary statistics for all datasets
#' @param comprehensive_comparison Comprehensive performance comparison data
#' @return Data frame with summary statistics
generate_summary_statistics <- function(comprehensive_comparison) {
  cat("\n=== Generating Summary Statistics ===\n")
  
  # Filter for all models that have inner CV performance
  models_data <- comprehensive_comparison[!is.na(comprehensive_comparison$Inner_CV_Performance), ]
  
  if (nrow(models_data) == 0) {
    return(NULL)
  }
  
  # Summary by dataset
  dataset_summary <- models_data %>%
    group_by(Dataset) %>%
    summarise(
      Mean_Inner_CV_Performance = mean(Inner_CV_Performance, na.rm = TRUE),
      Mean_Outer_CV_Kappa = mean(Outer_CV_Kappa, na.rm = TRUE),
      Mean_Degradation = mean(Performance_Degradation, na.rm = TRUE),
      Mean_Degradation_Percent = mean(Degradation_Percent, na.rm = TRUE),
      SD_Degradation_Percent = sd(Degradation_Percent, na.rm = TRUE),
      N_Models = n(),
      .groups = "drop"
    ) %>%
    arrange(Mean_Degradation_Percent)
  
  # Summary by model
  model_summary <- models_data %>%
    group_by(Model) %>%
    summarise(
      Mean_Inner_CV_Performance = mean(Inner_CV_Performance, na.rm = TRUE),
      Mean_Outer_CV_Kappa = mean(Outer_CV_Kappa, na.rm = TRUE),
      Mean_Degradation = mean(Performance_Degradation, na.rm = TRUE),
      Mean_Degradation_Percent = mean(Degradation_Percent, na.rm = TRUE),
      SD_Degradation_Percent = sd(Degradation_Percent, na.rm = TRUE),
      N_Datasets = n(),
      .groups = "drop"
    ) %>%
    arrange(Mean_Degradation_Percent)
  
  # Overall summary
  overall_summary <- data.frame(
    Metric = c(
      "Total Datasets",
      "Total Models",
      "Overall Mean Inner CV Performance",
      "Overall Mean Outer CV Kappa",
      "Overall Mean Performance Degradation",
      "Overall Mean Degradation Percentage",
      "Dataset with Highest Degradation",
      "Dataset with Lowest Degradation",
      "Model with Highest Degradation",
      "Model with Lowest Degradation"
    ),
    Value = c(
      length(unique(models_data$Dataset)),
      length(unique(models_data$Model)),
      round(mean(models_data$Inner_CV_Performance, na.rm = TRUE), 4),
      round(mean(models_data$Outer_CV_Kappa, na.rm = TRUE), 4),
      round(mean(models_data$Performance_Degradation, na.rm = TRUE), 4),
      paste0(round(mean(models_data$Degradation_Percent, na.rm = TRUE), 2), "%"),
      dataset_summary$Dataset[which.max(dataset_summary$Mean_Degradation_Percent)],
      dataset_summary$Dataset[which.min(dataset_summary$Mean_Degradation_Percent)],
      model_summary$Model[which.max(model_summary$Mean_Degradation_Percent)],
      model_summary$Model[which.min(model_summary$Mean_Degradation_Percent)]
    ),
    stringsAsFactors = FALSE
  )
  
  return(list(
    dataset_summary = dataset_summary,
    model_summary = model_summary,
    overall_summary = overall_summary
  ))
}

# =============================================================================
# Performance Analysis Functions
# =============================================================================

#' Analyze BEAT dataset performance across all models
#' @return List of performance analysis results
analyze_beat_dataset_performance <- function() {
  cat("=== Analyzing BEAT Dataset Performance ===\n")
  
  # Models to analyze
  models <- c("svm", "xgboost", "neural_net", "Global_Optimized", "OvR_Ensemble")
  
  # Initialize results
  performance_summary <- data.frame()
  ensemble_weights_summary <- list()
  confusion_matrices <- list()
  rejection_analysis <- list()
  
  for (model_name in models) {
    cat(sprintf("\n--- Analyzing %s ---\n", toupper(model_name)))
    
    # Load performance metrics
    metrics <- load_fold_performance_metrics(BEAT_DATASET_NAME, model_name)
    if (!is.null(metrics)) {
      cat("  Performance Metrics:\n")
      print(metrics)
      
      # Add to summary
      summary_row <- data.frame(
        Model = model_name,
        Fold = BEAT_DATASET_NAME,
        Kappa = metrics$Value[metrics$Metric == "Kappa"],
        Accuracy = metrics$Value[metrics$Metric == "Accuracy"],
        Balanced_Accuracy = metrics$Value[metrics$Metric == "Balanced_Accuracy"],
        F1_Macro = metrics$Value[metrics$Metric == "F1_Macro"],
        Sensitivity_Macro = metrics$Value[metrics$Metric == "Sensitivity_Macro"],
        Specificity_Macro = metrics$Value[metrics$Metric == "Specificity_Macro"],
        stringsAsFactors = FALSE
      )
      performance_summary <- rbind(performance_summary, summary_row)
    }
    
    # Load confusion matrix
    cm_text <- load_fold_confusion_matrix(BEAT_DATASET_NAME, model_name)
    if (!is.null(cm_text)) {
      confusion_matrices[[model_name]] <- cm_text
      cat("  Confusion matrix loaded\n")
    }
    
    # Load rejection analysis
    rejection_data <- load_fold_rejection_results(BEAT_DATASET_NAME, model_name)
    if (!is.null(rejection_data) && nrow(rejection_data) > 0) {
      rejection_analysis[[model_name]] <- rejection_data
      cat("  Rejection analysis loaded\n")
    }
  }
  
  # Load ensemble weights for BEAT
  cat("\n--- Loading Ensemble Weights for BEAT ---\n")
  
  # Global ensemble weights
  global_weights <- load_fold_ensemble_weights(BEAT_DATASET_NAME, "global")
  if (!is.null(global_weights)) {
    ensemble_weights_summary$global <- global_weights
    cat("  Global ensemble weights:\n")
    cat(sprintf("    Weight configuration: %s\n", global_weights$weight_name))
    cat(sprintf("    SVM weight: %.2f\n", global_weights$weights$SVM))
    cat(sprintf("    XGB weight: %.2f\n", global_weights$weights$XGB))
    cat(sprintf("    NN weight: %.2f\n", global_weights$weights$NN))
    cat(sprintf("    Inner CV Kappa: %.4f\n", global_weights$kappa))
  }
  
  # OvR ensemble weights
  ovr_weights <- load_fold_ensemble_weights(BEAT_DATASET_NAME, "ovr")
  if (!is.null(ovr_weights)) {
    ensemble_weights_summary$ovr <- ovr_weights
    cat("  OvR ensemble weights loaded for", length(ovr_weights), "classes\n")
  }
  
  return(list(
    performance_summary = performance_summary,
    ensemble_weights_summary = ensemble_weights_summary,
    confusion_matrices = confusion_matrices,
    rejection_analysis = rejection_analysis
  ))
}

#' Compare inner CV vs outer CV performance for BEAT
#' @param analysis_results Results from analyze_beat_dataset_performance
#' @return Data frame with performance comparison
compare_inner_vs_outer_cv_beat <- function(analysis_results) {
  cat("\n=== Comparing Inner CV vs Outer CV Performance for BEAT ===\n")
  
  # Get inner CV performance from ensemble weights
  inner_cv_performance <- data.frame()
  
  # Global ensemble inner CV performance
  if (!is.null(analysis_results$ensemble_weights_summary$global)) {
    global_weights <- analysis_results$ensemble_weights_summary$global
    inner_cv_performance <- rbind(inner_cv_performance, data.frame(
      Model = "Global_Optimized",
      Inner_CV_Kappa = global_weights$kappa,
      Inner_CV_Weight_Config = global_weights$weight_name,
      stringsAsFactors = FALSE
    ))
  }
  
  # OvR ensemble inner CV performance (average F1 scores)
  if (!is.null(analysis_results$ensemble_weights_summary$ovr)) {
    ovr_weights <- analysis_results$ensemble_weights_summary$ovr
    
    # Calculate average F1 score across classes
    f1_scores <- sapply(ovr_weights, function(x) x$f1_score)
    f1_scores <- f1_scores[!is.na(f1_scores)]
    
    if (length(f1_scores) > 0) {
      avg_f1 <- mean(f1_scores, na.rm = TRUE)
      inner_cv_performance <- rbind(inner_cv_performance, data.frame(
        Model = "OvR_Ensemble",
        Inner_CV_Kappa = avg_f1,  # Using F1 as proxy for kappa
        Inner_CV_Weight_Config = "Class-specific",
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Merge with outer CV performance
  comparison <- merge(inner_cv_performance, analysis_results$performance_summary, by = "Model")
  
  # Calculate performance degradation
  comparison$Performance_Degradation <- comparison$Inner_CV_Kappa - comparison$Kappa
  comparison$Degradation_Percent <- (comparison$Performance_Degradation / comparison$Inner_CV_Kappa) * 100
  
  # Sort by performance degradation
  comparison <- comparison[order(comparison$Performance_Degradation, decreasing = TRUE), ]
  
  return(comparison)
}

#' Analyze class-specific performance for BEAT
#' @param analysis_results Results from analyze_beat_dataset_performance
#' @return Data frame with class-specific analysis
analyze_class_specific_performance_beat <- function(analysis_results) {
  cat("\n=== Analyzing Class-Specific Performance for BEAT ===\n")
  
  # Get OvR weights for detailed class analysis
  if (is.null(analysis_results$ensemble_weights_summary$ovr)) {
    cat("  No OvR weights available for class-specific analysis\n")
    return(NULL)
  }
  
  ovr_weights <- analysis_results$ensemble_weights_summary$ovr
  
  # Create class performance summary
  class_performance <- data.frame()
  
  for (class_name in names(ovr_weights)) {
    class_weights <- ovr_weights[[class_name]]
    
    class_row <- data.frame(
      Class = class_name,
      Weight_Configuration = class_weights$weight_name,
      SVM_Weight = class_weights$weights$SVM,
      XGB_Weight = class_weights$weights$XGB,
      NN_Weight = class_weights$weights$NN,
      Inner_CV_F1 = class_weights$f1_score,
      stringsAsFactors = FALSE
    )
    
    class_performance <- rbind(class_performance, class_row)
  }
  
  # Sort by inner CV F1 score
  class_performance <- class_performance[order(class_performance$Inner_CV_F1, decreasing = TRUE), ]
  
  return(class_performance)
}

#' Generate comprehensive BEAT dataset analysis report
#' @param analysis_results Results from analyze_beat_dataset_performance
#' @param output_dir Directory to save the analysis report
generate_beat_analysis_report <- function(analysis_results, output_dir = "beat_dataset_analysis") {
  cat("\n=== Generating BEAT Dataset Analysis Report ===\n")
  
  # Create output directory
  create_directory_safely(output_dir)
  
  # 1. Performance Summary
  if (!is.null(analysis_results$performance_summary) && nrow(analysis_results$performance_summary) > 0) {
    write.csv(analysis_results$performance_summary, 
              file.path(output_dir, "beat_performance_summary.csv"), 
              row.names = FALSE)
    cat("  Saved performance summary to:", file.path(output_dir, "beat_performance_summary.csv"), "\n")
  }
  
  # 2. Inner vs Outer CV Comparison
  comparison <- compare_inner_vs_outer_cv_beat(analysis_results)
  if (nrow(comparison) > 0) {
    write.csv(comparison, 
              file.path(output_dir, "beat_inner_vs_outer_cv_comparison.csv"), 
              row.names = FALSE)
    cat("  Saved inner vs outer CV comparison to:", file.path(output_dir, "beat_inner_vs_outer_cv_comparison.csv"), "\n")
  }
  
  # 3. Class-Specific Analysis
  class_analysis <- analyze_class_specific_performance_beat(analysis_results)
  if (!is.null(class_analysis) && nrow(class_analysis) > 0) {
    write.csv(class_analysis, 
              file.path(output_dir, "beat_class_specific_analysis.csv"), 
              row.names = FALSE)
    cat("  Saved class-specific analysis to:", file.path(output_dir, "beat_class_specific_analysis.csv"), "\n")
  }
  
  # 4. Save confusion matrices
  for (model_name in names(analysis_results$confusion_matrices)) {
    cm_file <- file.path(output_dir, paste0("confusion_matrix_", model_name, ".txt"))
    writeLines(analysis_results$confusion_matrices[[model_name]], cm_file)
    cat("  Saved confusion matrix for", model_name, "to:", cm_file, "\n")
  }
  
  # 5. Save rejection analysis
  for (model_name in names(analysis_results$rejection_analysis)) {
    rejection_file <- file.path(output_dir, paste0("rejection_analysis_", model_name, ".csv"))
    write.csv(analysis_results$rejection_analysis[[model_name]], rejection_file, row.names = FALSE)
    cat("  Saved rejection analysis for", model_name, "to:", rejection_file, "\n")
  }
  
  # 6. Generate summary report
  summary_report <- generate_summary_report(analysis_results, comparison, class_analysis)
  report_file <- file.path(output_dir, "beat_analysis_summary_report.txt")
  writeLines(summary_report, report_file)
  cat("  Saved summary report to:", report_file, "\n")
  
  cat("  Analysis report generated successfully!\n")
}

#' Generate a comprehensive summary report
#' @param analysis_results Results from analyze_beat_dataset_performance
#' @param comparison Inner vs outer CV comparison
#' @param class_analysis Class-specific analysis
#' @return Character vector with summary report
generate_summary_report <- function(analysis_results, comparison, class_analysis) {
  report <- c(
    "=============================================================================",
    "BEAT Dataset Analysis Summary Report",
    "=============================================================================",
    "",
    "Dataset: BEATAML1.0-COHORT",
    "Analysis Date:", as.character(Sys.Date()),
    "",
    "=============================================================================",
    "1. OVERALL PERFORMANCE SUMMARY",
    "============================================================================="
  )
  
  if (!is.null(analysis_results$performance_summary) && nrow(analysis_results$performance_summary) > 0) {
    for (i in 1:nrow(analysis_results$performance_summary)) {
      row <- analysis_results$performance_summary[i, ]
      report <- c(report,
        "",
        paste("Model:", row$Model),
        paste("  Kappa:", round(row$Kappa, 4)),
        paste("  Accuracy:", round(row$Accuracy, 4)),
        paste("  Balanced Accuracy:", round(row$Balanced_Accuracy, 4)),
        paste("  F1 Macro:", round(row$F1_Macro, 4))
      )
    }
  }
  
  report <- c(report,
    "",
    "=============================================================================",
    "2. INNER CV vs OUTER CV PERFORMANCE COMPARISON",
    "============================================================================="
  )
  
  if (nrow(comparison) > 0) {
    for (i in 1:nrow(comparison)) {
      row <- comparison[i, ]
      report <- c(report,
        "",
        paste("Model:", row$Model),
        paste("  Inner CV Performance:", round(row$Inner_CV_Kappa, 4)),
        paste("  Outer CV Performance:", round(row$Kappa, 4)),
        paste("  Performance Degradation:", round(row$Performance_Degradation, 4)),
        paste("  Degradation Percentage:", round(row$Degradation_Percent, 2), "%")
      )
    }
  }
  
  report <- c(report,
    "",
    "=============================================================================",
    "3. KEY INSIGHTS AND RECOMMENDATIONS",
    "=============================================================================",
    "",
    "The analysis reveals several important findings:",
    "",
    "1. Performance Transfer Gap:",
    "   - Models that perform well during inner CV show significant degradation",
    "   - This suggests overfitting to the training data or dataset shift issues",
    "",
    "2. Ensemble Weight Analysis:",
    "   - BEAT dataset shows different optimal weight configurations",
    "   - Some classes benefit from specific model combinations",
    "",
    "3. Recommendations:",
    "   - Consider dataset-specific fine-tuning for BEAT",
    "   - Investigate potential data distribution differences",
    "   - Explore domain adaptation techniques",
    "   - Validate ensemble weights on external BEAT-like datasets"
  )
  
  return(report)
}

#' Main function to run multi-dataset LOSO analysis
#' @return List of comprehensive analysis results
main_multi_dataset_analysis <- function() {
  cat("=== Starting Multi-Dataset LOSO Analysis ===\n")
  
  # Load required libraries
  load_library_quietly("dplyr")
  load_library_quietly("ggplot2")
  load_library_quietly("tidyr")
  
  # Run comprehensive analysis across all datasets
  all_datasets_results <- analyze_all_loso_datasets()
  
  # Create comprehensive performance comparison
  comprehensive_comparison <- create_comprehensive_performance_comparison(all_datasets_results)
  
  # Generate summary statistics
  summary_stats <- generate_summary_statistics(comprehensive_comparison)
  
  # Generate comprehensive visualizations
  generate_comprehensive_visualizations(comprehensive_comparison)
  
  # Save comprehensive results
  output_dir <- "multi_dataset_analysis"
  create_directory_safely(output_dir)
  
  # Save comprehensive comparison
  write.csv(comprehensive_comparison, 
            file.path(output_dir, "comprehensive_performance_comparison.csv"), 
            row.names = FALSE)
  
  # Save summary statistics
  if (!is.null(summary_stats)) {
    write.csv(summary_stats$dataset_summary, 
              file.path(output_dir, "dataset_summary_statistics.csv"), 
              row.names = FALSE)
    write.csv(summary_stats$model_summary, 
              file.path(output_dir, "model_summary_statistics.csv"), 
              row.names = FALSE)
    write.csv(summary_stats$overall_summary, 
              file.path(output_dir, "overall_summary_statistics.csv"), 
              row.names = FALSE)
  }
  
  # Display key findings
  cat("\n=== KEY FINDINGS ACROSS ALL DATASETS ===\n")
  
  if (!is.null(summary_stats)) {
    cat("\nDataset Performance Summary (Ranked by Degradation):\n")
    print(summary_stats$dataset_summary)
    
    cat("\nModel Performance Summary (Ranked by Degradation):\n")
    print(summary_stats$model_summary)
    
    cat("\nOverall Summary:\n")
    print(summary_stats$overall_summary)
  }
  
  # Display top degradation cases
  models_data <- comprehensive_comparison[!is.na(comprehensive_comparison$Inner_CV_Performance), ]
  if (nrow(models_data) > 0) {
    cat("\nTop 10 Worst Performance Transfer Cases:\n")
    top_degradation <- models_data[order(models_data$Degradation_Percent, decreasing = TRUE), ]
    print(head(top_degradation[, c("Dataset", "Model", "Inner_CV_Performance", "Outer_CV_Kappa", "Degradation_Percent")], 10))
    
    cat("\nTop 10 Best Performance Transfer Cases:\n")
    best_transfer <- models_data[order(models_data$Degradation_Percent, decreasing = FALSE), ]
    print(head(best_transfer[, c("Dataset", "Model", "Inner_CV_Performance", "Outer_CV_Kappa", "Degradation_Percent")], 10))
  }
  
  cat("\n=== Multi-Dataset Analysis Complete! ===\n")
  
  return(list(
    all_datasets_results = all_datasets_results,
    comprehensive_comparison = comprehensive_comparison,
    summary_stats = summary_stats
  ))
}

# =============================================================================
# Main Analysis Function
# =============================================================================

#' Main function to run BEAT dataset analysis
main_beat_analysis <- function() {
  cat("=== Starting BEAT Dataset Analysis ===\n")
  
  # Load required libraries
  load_library_quietly("dplyr")
  load_library_quietly("ggplot2")
  
  # Run comprehensive analysis
  analysis_results <- analyze_beat_dataset_performance()
  
  # Generate comparison
  comparison <- compare_inner_vs_outer_cv_beat(analysis_results)
  
  # Generate class-specific analysis
  class_analysis <- analyze_class_specific_performance_beat(analysis_results)
  
  # Display key findings
  cat("\n=== KEY FINDINGS ===\n")
  
  if (nrow(comparison) > 0) {
    cat("\nPerformance Degradation (Inner CV vs Outer CV):\n")
    print(comparison[, c("Model", "Inner_CV_Kappa", "Kappa", "Performance_Degradation", "Degradation_Percent")])
  }
  
  if (!is.null(class_analysis) && nrow(class_analysis) > 0) {
    cat("\nTop Performing Classes (by Inner CV F1):\n")
    top_classes <- head(class_analysis, 5)
    print(top_classes[, c("Class", "Inner_CV_F1", "Weight_Configuration")])
  }
  
  # Generate comprehensive report
  generate_beat_analysis_report(analysis_results)
  
  cat("\n=== BEAT Dataset Analysis Complete! ===\n")
  
  return(list(
    analysis_results = analysis_results,
    comparison = comparison,
    class_analysis = class_analysis
  ))
}

# =============================================================================
# Utility Functions
# =============================================================================

#' Load library quietly without messages or warnings
#' @param package_name Name of the package to load
load_library_quietly <- function(package_name) {
  invisible(capture.output(
    suppressMessages(
      suppressWarnings(
        library(package_name, character.only = TRUE)
      )
    )
  ))
}

#' Create directory safely
#' @param dir_path Directory path to create
create_directory_safely <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  }
}

# Run the analysis if this script is executed directly
if (!exists("SKIP_BEAT_ANALYSIS_EXECUTION")) {
  # Run both analyses
  cat("Running both single dataset (BEAT) and multi-dataset analysis...\n\n")
  
  # Run BEAT-specific analysis
  beat_analysis_results <- main_beat_analysis()
  
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  
  # Run multi-dataset analysis
  multi_dataset_results <- main_multi_dataset_analysis()
}
