# =============================================================================
# Enhanced Performance Plot with Corrected Error Bars
# =============================================================================
# This script generates performance comparison plots with correct error bars:
# - Inner CV: Standard deviation across inner folds for best parameters
# - Outer CV: No error bars (single evaluation per dataset)
# - Ensemble methods: Standard deviation calculated from fold-specific metrics
# =============================================================================

# Set working directory
setwd("~/Documents/AML_PhD/leukem_ai")

# =============================================================================
# Source Utility Functions
# =============================================================================

source("R/utility_functions.R")

# =============================================================================
# Data Loading Functions
# =============================================================================

#' Load inner CV variation data for error bars
#' @return List of inner CV variation data by model
load_inner_cv_variation_data <- function() {
  cat("Loading inner CV variation data for error bars...\n")
  
  variation_data <- list()
  models <- c("svm", "xgboost", "neural_net")
  
  for (model in models) {
    variation_file <- file.path("inner_cv_variation_analysis", 
                               paste0(model, "_inner_cv_variation_summary.csv"))
    
    if (file.exists(variation_file)) {
      variation_data[[model]] <- read.csv(variation_file, stringsAsFactors = FALSE)
      cat(sprintf("  Loaded %s variation data\n", model))
    } else {
      warning(sprintf("Variation file not found: %s", variation_file))
    }
  }
  
  return(variation_data)
}

#' Load ensemble weights for error propagation
#' @return List of ensemble weights by fold and model
load_ensemble_weights_for_sd <- function() {
  cat("Loading ensemble weights for error propagation...\n")
  
  ensemble_weights <- list()
  cv_types <- c("cv", "loso")
  
  for (cv_type in cv_types) {
    weights_dir <- file.path("inner_cv_best_params_n10", "ensemble_weights", cv_type)
    
    if (dir.exists(weights_dir)) {
      # Load global ensemble weights
      global_weights_file <- file.path(weights_dir, "global_ensemble_weights_used.csv")
      if (file.exists(global_weights_file)) {
        global_weights <- read.csv(global_weights_file, stringsAsFactors = FALSE)
        ensemble_weights[[cv_type]][["global"]] <- global_weights
        cat(sprintf("  Loaded global ensemble weights for %s: %d folds\n", cv_type, nrow(global_weights)))
      }
      
      # Load OvR ensemble weights
      ovr_weights_file <- file.path(weights_dir, "ovr_ensemble_weights_used.csv")
      if (file.exists(ovr_weights_file)) {
        ovr_weights <- read.csv(ovr_weights_file, stringsAsFactors = FALSE)
        ensemble_weights[[cv_type]][["ovr"]] <- ovr_weights
        cat(sprintf("  Loaded OvR ensemble weights for %s: %d folds\n", cv_type, nrow(ovr_weights)))
      }
    }
  }
  
  return(ensemble_weights)
}

#' Load ensemble fold-specific metrics for error bars
#' @return List of ensemble fold-specific metrics by model
load_ensemble_fold_metrics <- function() {
  cat("Loading ensemble fold-specific metrics for error bars...\n")
  
  ensemble_data <- list()
  ensemble_models <- c("Global_Optimized", "OvR_Ensemble")
  cv_types <- c("cv", "loso")
  
  for (model in ensemble_models) {
    ensemble_data[[model]] <- list()
    
    for (cv_type in cv_types) {
      # Map model names to directory names
      dir_name <- if (model == "Global_Optimized") "global_ensemble" else "ovr_ensemble"
      
      metrics_dir <- file.path("inner_cv_best_params_n10", "outer_cv_analysis", cv_type, dir_name)
      
      if (dir.exists(metrics_dir)) {
        # Get all metrics files
        metrics_files <- list.files(metrics_dir, pattern = "metrics_fold_.*\\.csv", full.names = TRUE)
        
        if (length(metrics_files) > 0) {
          fold_metrics <- list()
          
          for (file in metrics_files) {
            # Extract fold name from filename
            fold_name <- gsub(".*metrics_fold_(.*)\\.csv", "\\1", basename(file))
            
            if (file.exists(file)) {
              metrics <- read.csv(file, stringsAsFactors = FALSE)
              fold_metrics[[fold_name]] <- metrics
            }
          }
          
          ensemble_data[[model]][[cv_type]] <- fold_metrics
          cat(sprintf("  Loaded %s %s metrics: %d folds\n", model, cv_type, length(fold_metrics)))
        }
      }
    }
  }
  
  return(ensemble_data)
}

#' Load comprehensive performance comparison data
#' @return Data frame with performance data
load_performance_comparison_data <- function() {
  cat("Loading performance comparison data...\n")
  
  perf_file <- "multi_dataset_analysis/comprehensive_performance_comparison.csv"
  
  if (file.exists(perf_file)) {
    perf_data <- read.csv(perf_file, stringsAsFactors = FALSE)
    cat(sprintf("  Loaded performance data: %d rows\n", nrow(perf_data)))
    return(perf_data)
  } else {
    stop(sprintf("Performance comparison file not found: %s", perf_file))
  }
}

#' Calculate ensemble standard deviation using error propagation
#' @param ensemble_weights Ensemble weights data
#' @param individual_model_sds Individual model standard deviations
#' @param model Model name (Global_Optimized or OvR_Ensemble)
#' @param dataset Dataset name
#' @return Standard deviation value
calculate_ensemble_sd_propagated <- function(ensemble_weights, individual_model_sds, model, dataset) {
  # Map model names to ensemble type
  ensemble_type <- if (model == "Global_Optimized") "global" else "ovr"
  
  # Use CV type data for now (could be enhanced to use LOSO for specific datasets)
  cv_weights <- ensemble_weights[["cv"]][[ensemble_type]]
  
  if (!is.null(cv_weights) && nrow(cv_weights) > 0) {
    # Get individual model SDs for this dataset
    svm_sd <- individual_model_sds$svm[individual_model_sds$dataset == dataset]
    xgb_sd <- individual_model_sds$xgb[individual_model_sds$dataset == dataset]
    nn_sd <- individual_model_sds$nn[individual_model_sds$dataset == dataset]
    
    # If we have SDs for all models, calculate ensemble SD using error propagation
    if (length(svm_sd) > 0 && length(xgb_sd) > 0 && length(nn_sd) > 0) {
      # Calculate weighted ensemble SD using error propagation formula:
      # SD_ensemble = sqrt(sum((weight_i * SD_i)^2))
      ensemble_sds <- numeric(0)
      
      for (i in 1:nrow(cv_weights)) {
        svm_weight <- cv_weights$svm_weight[i]
        xgb_weight <- cv_weights$xgb_weight[i]
        nn_weight <- cv_weights$nn_weight[i]
        
        # Error propagation: SD_ensemble = sqrt(sum((weight_i * SD_i)^2))
        ensemble_sd <- sqrt((svm_weight * svm_sd)^2 + (xgb_weight * xgb_sd)^2 + (nn_weight * nn_sd)^2)
        ensemble_sds <- c(ensemble_sds, ensemble_sd)
      }
      
      # Return mean ensemble SD across folds
      if (length(ensemble_sds) > 0) {
        return(mean(ensemble_sds))
      }
    }
  }
  
  # Fallback: return average of individual model SDs for this dataset
  if (length(svm_sd) > 0 && length(xgb_sd) > 0 && length(nn_sd) > 0) {
    return(mean(c(svm_sd, xgb_sd, nn_sd)))
  }
  
  return(0)  # Return 0 if no data available
}

#' Calculate ensemble standard deviation from fold metrics
#' @param ensemble_data Ensemble fold-specific metrics
#' @param model Model name
#' @param dataset Dataset name
#' @return Standard deviation value
calculate_ensemble_sd <- function(ensemble_data, model, dataset) {
  # For now, we'll use CV type data as a proxy for dataset-specific variation
  # In a more sophisticated approach, we could map datasets to specific CV folds
  
  if (model %in% names(ensemble_data)) {
    # Use CV type data (5-fold CV) for standard deviation calculation
    cv_metrics <- ensemble_data[[model]][["cv"]]
    
    if (!is.null(cv_metrics) && length(cv_metrics) > 0) {
      # Extract kappa values from all folds
      kappa_values <- numeric(0)
      
      for (fold_name in names(cv_metrics)) {
        fold_data <- cv_metrics[[fold_name]]
        if ("Kappa" %in% fold_data$Metric) {
          kappa_values <- c(kappa_values, fold_data$Value[fold_data$Metric == "Kappa"])
        }
      }
      
      if (length(kappa_values) > 1) {
        return(sd(kappa_values))
      }
    }
  }
  
  return(0)  # Return 0 if no variation data available
}

#' Merge performance data with variation data for error bars
#' @param perf_data Performance comparison data
#' @param variation_data Inner CV variation data
#' @param ensemble_data Ensemble fold-specific metrics
#' @param ensemble_weights Ensemble weights for error propagation
#' @return Enhanced data frame with error bar information
merge_performance_with_variation <- function(perf_data, variation_data, ensemble_data, ensemble_weights) {
  cat("Merging performance data with variation data...\n")
  
  # Initialize Inner_CV_SD column
  perf_data$Inner_CV_SD <- 0
  
  # Add inner CV standard deviation for individual models
  individual_models <- perf_data[perf_data$Model %in% c("svm", "xgboost", "neural_net"), ]
  
  for (i in 1:nrow(individual_models)) {
    dataset <- individual_models$Dataset[i]
    model <- individual_models$Model[i]
    
    if (model %in% names(variation_data)) {
      model_variation <- variation_data[[model]]
      dataset_row <- model_variation[model_variation$outer_fold == dataset, ]
      
      if (nrow(dataset_row) > 0) {
        perf_data$Inner_CV_SD[perf_data$Dataset == dataset & perf_data$Model == model] <- 
          dataset_row$sd_kappa[1]
      }
    }
  }
  
  # Create a lookup table for individual model SDs by dataset
  individual_model_sds <- data.frame(
    dataset = character(),
    svm = numeric(),
    xgb = numeric(),
    nn = numeric(),
    stringsAsFactors = FALSE
  )
  
  datasets <- unique(perf_data$Dataset)
  for (dataset in datasets) {
    svm_sd <- perf_data$Inner_CV_SD[perf_data$Dataset == dataset & perf_data$Model == "svm"]
    xgb_sd <- perf_data$Inner_CV_SD[perf_data$Dataset == dataset & perf_data$Model == "xgboost"]
    nn_sd <- perf_data$Inner_CV_SD[perf_data$Dataset == dataset & perf_data$Model == "neural_net"]
    
    if (length(svm_sd) > 0 && length(xgb_sd) > 0 && length(nn_sd) > 0) {
      individual_model_sds <- rbind(individual_model_sds, data.frame(
        dataset = dataset,
        svm = svm_sd,
        xgb = xgb_sd,
        nn = nn_sd,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Add standard deviation for ensemble models using error propagation
  ensemble_models <- perf_data[perf_data$Model %in% c("Global_Optimized", "OvR_Ensemble"), ]
  
  for (i in 1:nrow(ensemble_models)) {
    dataset <- ensemble_models$Dataset[i]
    model <- ensemble_models$Model[i]
    
    # Use error propagation instead of outer CV fold variation
    sd_value <- calculate_ensemble_sd_propagated(ensemble_weights, individual_model_sds, model, dataset)
    perf_data$Inner_CV_SD[perf_data$Dataset == dataset & perf_data$Model == model] <- sd_value
  }
  
  cat(sprintf("  Enhanced data created: %d rows\n", nrow(perf_data)))
  return(perf_data)
}

#' Generate enhanced performance plot with correct error bars
#' @param enhanced_data Enhanced performance data with variation information
#' @param output_dir Output directory
generate_corrected_performance_plot <- function(enhanced_data, output_dir = "multi_dataset_analysis") {
  cat("Generating corrected performance comparison plot...\n")
  
  # Load required libraries
  load_library_quietly("ggplot2")
  load_library_quietly("dplyr")
  load_library_quietly("tidyr")
  
  # Filter for models with inner CV performance
  models_with_inner_cv <- enhanced_data[!is.na(enhanced_data$Inner_CV_Performance), ]
  
  if (nrow(models_with_inner_cv) == 0) {
    cat("  No model performance data available for visualization\n")
    return()
  }
  
  # Reshape data for plotting
  plot_data_long <- models_with_inner_cv %>%
    select(Dataset, Model, Inner_CV_Performance, Inner_CV_SD, Outer_CV_Kappa) %>%
    tidyr::pivot_longer(
      cols = c(Inner_CV_Performance, Outer_CV_Kappa),
      names_to = "Performance_Type",
      values_to = "Performance_Value"
    )
  
  # Add corresponding SD values (only for Inner CV)
  plot_data_long$SD_Value <- ifelse(
    plot_data_long$Performance_Type == "Inner_CV_Performance",
    models_with_inner_cv$Inner_CV_SD[match(
      paste(plot_data_long$Dataset, plot_data_long$Model), 
      paste(models_with_inner_cv$Dataset, models_with_inner_cv$Model)
    )],
    0  # No error bars for Outer CV
  )
  
  # Set SD to 0 for models without variation data
  plot_data_long$SD_Value[is.na(plot_data_long$SD_Value)] <- 0
  
  plot_data_long$Performance_Type <- factor(
    plot_data_long$Performance_Type,
    levels = c("Inner_CV_Performance", "Outer_CV_Kappa"),
    labels = c("Inner CV", "Outer CV")
  )
  
  # Create the plot with corrected error bar positioning
  p_corrected <- ggplot(plot_data_long, aes(x = Dataset, y = Performance_Value, fill = Performance_Type)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8, width = 0.7) +
    # Add error bars only where SD > 0 (Inner CV for all models)
    # Position error bars to align with Inner CV bars specifically
    geom_errorbar(
      data = subset(plot_data_long, SD_Value > 0 & Performance_Type == "Inner CV"),
      aes(ymin = Performance_Value - SD_Value, ymax = Performance_Value + SD_Value),
      position = position_nudge(x = -0.175),  # Nudge left to align with Inner CV bars
      width = 0.2,
      color = "black",
      alpha = 0.8
    ) +
    facet_wrap(~Model, scales = "free_y", ncol = 3) +
    scale_fill_brewer(palette = "Set1", name = "Performance Type") +
    labs(
      title = "Performance Comparison: Inner CV vs Outer CV (Corrected Error Bars)",
      subtitle = "Error bars show SD across inner folds for best hyperparameters (Inner CV only)",
      x = "Dataset",
      y = "Performance (Kappa)"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10),
      legend.position = "bottom",
      strip.text = element_text(size = 10, face = "bold")
    )
  
  # Save corrected plot
  create_directory_safely(output_dir)
  ggsave(file.path(output_dir, "performance_comparison_corrected_error_bars.png"), 
         p_corrected, width = 14, height = 10)
  
  cat(sprintf("  Corrected plot saved to: %s\n", 
             file.path(output_dir, "performance_comparison_corrected_error_bars.png")))
  
  # Also create a detailed summary table
  summary_table <- models_with_inner_cv %>%
    select(Dataset, Model, Inner_CV_Performance, Inner_CV_SD, Outer_CV_Kappa, 
           Performance_Degradation, Degradation_Percent) %>%
    arrange(Dataset, Model)
  
  write.csv(summary_table, 
            file.path(output_dir, "performance_summary_with_error_bars.csv"), 
            row.names = FALSE)
  
  cat(sprintf("  Summary table saved to: %s\n", 
             file.path(output_dir, "performance_summary_with_error_bars.csv")))
  
  return(summary_table)
}

#' Generate comparison of error bar magnitudes
#' @param enhanced_data Enhanced performance data
#' @param output_dir Output directory
generate_error_bar_analysis <- function(enhanced_data, output_dir = "multi_dataset_analysis") {
  cat("Generating error bar magnitude analysis...\n")
  
  # Load required libraries
  load_library_quietly("ggplot2")
  load_library_quietly("dplyr")
  
  # Filter for models with error bars (all models now)
  models_with_errors <- enhanced_data[
    !is.na(enhanced_data$Inner_CV_SD) & 
    enhanced_data$Inner_CV_SD > 0, 
  ]
  
  if (nrow(models_with_errors) == 0) {
    cat("  No models with error bar data available\n")
    return()
  }
  
  # Plot 1: Error bar magnitude by dataset and model
  p1 <- ggplot(models_with_errors, aes(x = Dataset, y = Inner_CV_SD, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    scale_fill_brewer(palette = "Set2", name = "Model") +
    labs(
      title = "Inner CV Performance Variability by Dataset and Model",
      subtitle = "Standard deviation of kappa across inner folds (for best hyperparameters)",
      x = "Dataset",
      y = "Standard Deviation (Kappa)"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(size = 14, face = "bold"),
      legend.position = "bottom"
    )
  
  # Plot 2: Coefficient of variation (relative variability)
  models_with_errors$CV <- models_with_errors$Inner_CV_SD / models_with_errors$Inner_CV_Performance
  
  p2 <- ggplot(models_with_errors, aes(x = Dataset, y = CV, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    scale_fill_brewer(palette = "Set2", name = "Model") +
    labs(
      title = "Inner CV Performance Relative Variability",
      subtitle = "Coefficient of variation (SD/Mean) across inner folds",
      x = "Dataset",
      y = "Coefficient of Variation"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(size = 14, face = "bold"),
      legend.position = "bottom"
    )
  
  # Save plots
  ggsave(file.path(output_dir, "inner_cv_variability_by_dataset.png"), p1, width = 12, height = 8)
  ggsave(file.path(output_dir, "inner_cv_relative_variability.png"), p2, width = 12, height = 8)
  
  cat(sprintf("  Variability analysis plots saved to: %s\n", output_dir))
  
  # Summary statistics
  variability_summary <- models_with_errors %>%
    group_by(Model) %>%
    summarise(
      Mean_SD = mean(Inner_CV_SD, na.rm = TRUE),
      Mean_CV = mean(CV, na.rm = TRUE),
      Max_SD = max(Inner_CV_SD, na.rm = TRUE),
      Min_SD = min(Inner_CV_SD, na.rm = TRUE),
      Most_Variable_Dataset = Dataset[which.max(Inner_CV_SD)],
      Least_Variable_Dataset = Dataset[which.min(Inner_CV_SD)],
      .groups = "drop"
    )
  
  cat("\nInner CV Variability Summary by Model:\n")
  print(variability_summary)
  
  write.csv(variability_summary, 
            file.path(output_dir, "inner_cv_variability_summary.csv"), 
            row.names = FALSE)
  
  return(variability_summary)
}

# =============================================================================
# Main Analysis Function
# =============================================================================

#' Main function to generate corrected performance plots
main_corrected_performance_analysis <- function() {
  cat("=== Starting Corrected Performance Plot Analysis ===\n")
  
  # Load required libraries
  load_library_quietly("ggplot2")
  load_library_quietly("dplyr")
  load_library_quietly("tidyr")
  
  # Load data
  variation_data <- load_inner_cv_variation_data()
  ensemble_data <- load_ensemble_fold_metrics()
  perf_data <- load_performance_comparison_data()
  ensemble_weights <- load_ensemble_weights_for_sd()
  
  # Merge data
  enhanced_data <- merge_performance_with_variation(perf_data, variation_data, ensemble_data, ensemble_weights)
  
  # Generate corrected performance plot
  summary_table <- generate_corrected_performance_plot(enhanced_data)
  
  # Generate error bar analysis
  variability_summary <- generate_error_bar_analysis(enhanced_data)
  
  cat("\n=== Key Findings ===\n")
  cat("Corrected error bar logic:\n")
  cat("- Inner CV: Error bars show SD across inner folds for best hyperparameters\n")
  cat("- Outer CV: No error bars (single evaluation per dataset)\n")
  cat("- Individual models (SVM, XGBoost, Neural Net): SD from inner CV variation analysis\n")
  cat("- Ensemble models (Global_Optimized, OvR_Ensemble): SD calculated from fold-specific metrics\n")
  cat("- Error bars are now properly centered above Inner CV bars\n")
  
  cat("\n=== Analysis Complete! ===\n")
  
  return(list(
    enhanced_data = enhanced_data,
    summary_table = summary_table,
    variability_summary = variability_summary
  ))
}

# Run the analysis if this script is executed directly
if (!exists("SKIP_CORRECTED_PERFORMANCE_EXECUTION")) {
  corrected_performance_results <- main_corrected_performance_analysis()
}
