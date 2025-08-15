# =============================================================================
# Inner CV Performance Variation Analysis
# =============================================================================
# This script analyzes the variation in inner CV performance across different
# 3-dataset combinations for each outer fold to understand if poor outer CV
# performance is due to inconsistent inner CV optimization.
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

# Model configurations (from inner CV analysis)
MODEL_CONFIGS <- list(
  svm = list(
    classification_type = "OvR",
    file_paths = list(
      loso = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250611_1927/SVM_inner_cv_loso_OvR_20250611_1927.csv"
    )
  ),
  xgboost = list(
    classification_type = "OvR",
    file_paths = list(
      loso = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250612_0516/XGBOOST_inner_cv_loso_OvR_20250612_0516.csv"
    )
  ),
  neural_net = list(
    classification_type = "standard",
    file_paths = list(
      loso = "/Users/jsevere2/Documents/AML_PhD/predictor_out/NN/loso_7aug25/"
    )
  )
)

# =============================================================================
# Source Utility Functions
# =============================================================================

source("R/utility_functions.R")

# =============================================================================
# Data Loading Functions
# =============================================================================

#' Load and combine neural network CSV files
#' @param directory_path Path to directory containing CSV files
#' @return Combined data frame
combine_csv_files <- function(directory_path) {
  if (!dir.exists(directory_path)) {
    stop(sprintf("Directory does not exist: %s", directory_path))
  }
  
  csv_files <- list.files(directory_path, recursive = TRUE, full.names = TRUE, pattern = "\\.csv$")
  
  if (length(csv_files) == 0) {
    stop(sprintf("No CSV files found in directory: %s", directory_path))
  }
  
  combined_results <- lapply(csv_files, function(file) {
    safe_read_file(file, function(f) data.frame(data.table::fread(f, sep = ",", drop = 1)))
  })
  
  # Remove NULL results from failed reads
  combined_results <- combined_results[!sapply(combined_results, is.null)]
  
  if (length(combined_results) == 0) {
    stop("No files could be read successfully")
  }
  
  do.call(rbind, combined_results)
}

#' Process neural network results to clean epoch information
#' @param nn_results Neural network results data frame
#' @return Processed data frame
process_neural_net_results <- function(nn_results) {
  load_library_quietly("dplyr")
  load_library_quietly("stringr")
  
  # Extract best_epoch as numeric
  nn_results$epochs <- str_match(nn_results$params, "best_epoch': np\\.int64\\((\\d+)\\)")[,2] |> as.integer()
  
  # Remove best_epoch from param string
  nn_results$params <- gsub(", 'best_epoch'.+", "", nn_results$params)
  
  # Add mean best_epoch per group back into param string
  nn_results %>%
    group_by(outer_fold, params) %>%
    mutate(params = paste0(params, ", 'best_epoch': ", round(mean(epochs)), "}")) %>%
    ungroup()
}

#' Load raw inner CV results for a specific model
#' @param model_name Name of the model ("svm", "xgboost", "neural_net")
#' @return Data frame with raw inner CV results
load_raw_inner_cv_results <- function(model_name) {
  cat(sprintf("Loading raw inner CV results for %s...\n", toupper(model_name)))
  
  config <- MODEL_CONFIGS[[model_name]]
  file_path <- config$file_paths$loso
  
  if (model_name == "neural_net") {
    # Neural networks use directory of CSV files
    results <- safe_read_file(file_path, combine_csv_files)
    if (!is.null(results)) {
      results <- process_neural_net_results(results)
    }
  } else {
    # SVM and XGBoost use single CSV files
    results <- safe_read_file(file_path, function(f) {
      data.frame(data.table::fread(f, sep = ","), row.names = 1)
    })
  }
  
  if (is.null(results)) {
    warning(sprintf("Failed to load %s data", model_name))
    return(NULL)
  }
  
  cat(sprintf("  Loaded %d rows of inner CV results\n", nrow(results)))
  return(results)
}

# =============================================================================
# Inner CV Variation Analysis Functions
# =============================================================================

#' Analyze inner CV performance variation for each outer fold
#' @param raw_results Raw inner CV results
#' @param model_name Name of the model
#' @return Data frame with performance variation analysis
analyze_inner_cv_variation <- function(raw_results, model_name) {
  cat(sprintf("Analyzing inner CV variation for %s...\n", toupper(model_name)))
  
  load_library_quietly("dplyr")
  
  variation_results <- data.frame()
  
  for (outer_fold in LOSO_DATASETS) {
    cat(sprintf("  Processing outer fold: %s\n", outer_fold))
    
    # Filter data for this outer fold
    fold_data <- raw_results[raw_results$outer_fold == outer_fold, ]
    
    if (nrow(fold_data) == 0) {
      cat(sprintf("    No data found for %s\n", outer_fold))
      next
    }
    
    # Get unique inner folds for this outer fold
    inner_folds <- unique(fold_data$inner_fold)
    cat(sprintf("    Found %d inner folds\n", length(inner_folds)))
    
    # Get unique parameter combinations
    param_combinations <- unique(fold_data$params)
    cat(sprintf("    Found %d parameter combinations\n", length(param_combinations)))
    
    # For each parameter combination, calculate performance across inner folds
    for (param_combo in param_combinations) {
      param_data <- fold_data[fold_data$params == param_combo, ]
      
      if (nrow(param_data) == 0) next
      
      # Calculate statistics for this parameter combination
      kappas <- param_data$kappa
      kappas <- kappas[!is.na(kappas)]
      
      if (length(kappas) == 0) next
      
      # Create summary for this parameter combination
      param_summary <- data.frame(
        outer_fold = outer_fold,
        model = model_name,
        params = param_combo,
        n_inner_folds = length(kappas),
        mean_kappa = mean(kappas),
        sd_kappa = sd(kappas),
        min_kappa = min(kappas),
        max_kappa = max(kappas),
        range_kappa = max(kappas) - min(kappas),
        cv_kappa = ifelse(mean(kappas) != 0, sd(kappas) / mean(kappas), NA),
        stringsAsFactors = FALSE
      )
      
      variation_results <- rbind(variation_results, param_summary)
    }
    
    # Find best parameter combination for this outer fold
    if (nrow(variation_results) > 0) {
      fold_results <- variation_results[variation_results$outer_fold == outer_fold, ]
      best_idx <- which.max(fold_results$mean_kappa)
      
      if (length(best_idx) > 0) {
        best_params <- fold_results[best_idx, ]
        cat(sprintf("    Best params: Mean Kappa = %.4f ± %.4f (Range: %.4f)\n", 
                   best_params$mean_kappa, best_params$sd_kappa, best_params$range_kappa))
      }
    }
  }
  
  return(variation_results)
}

#' Summarize inner CV variation across all outer folds
#' @param variation_results Results from analyze_inner_cv_variation
#' @return Summary data frame
summarize_inner_cv_variation <- function(variation_results) {
  cat("Summarizing inner CV variation across outer folds...\n")
  
  load_library_quietly("dplyr")
  
  # Get best parameter combination for each outer fold
  best_params_per_fold <- variation_results %>%
    group_by(outer_fold, model) %>%
    filter(mean_kappa == max(mean_kappa, na.rm = TRUE)) %>%
    slice(1) %>%
    ungroup()
  
  # Summary by outer fold
  fold_summary <- best_params_per_fold %>%
    select(outer_fold, model, mean_kappa, sd_kappa, range_kappa, cv_kappa, n_inner_folds) %>%
    arrange(desc(mean_kappa))
  
  # Overall summary statistics
  overall_summary <- best_params_per_fold %>%
    summarise(
      model = first(model),
      n_outer_folds = n(),
      overall_mean_kappa = mean(mean_kappa, na.rm = TRUE),
      overall_sd_kappa = sd(mean_kappa, na.rm = TRUE),
      mean_variation_sd = mean(sd_kappa, na.rm = TRUE),
      mean_variation_range = mean(range_kappa, na.rm = TRUE),
      mean_cv = mean(cv_kappa, na.rm = TRUE),
      worst_fold = outer_fold[which.min(mean_kappa)],
      best_fold = outer_fold[which.max(mean_kappa)],
      .groups = "drop"
    )
  
  return(list(
    fold_summary = fold_summary,
    overall_summary = overall_summary,
    best_params_per_fold = best_params_per_fold
  ))
}

#' Generate inner CV variation plots
#' @param variation_results Results from analyze_inner_cv_variation
#' @param model_name Name of the model
#' @param output_dir Directory to save plots
generate_inner_cv_variation_plots <- function(variation_results, model_name, output_dir = "inner_cv_variation_analysis") {
  cat(sprintf("Generating inner CV variation plots for %s...\n", toupper(model_name)))
  
  # Load plotting libraries
  load_library_quietly("ggplot2")
  load_library_quietly("dplyr")
  
  # Create output directory
  create_directory_safely(output_dir)
  
  # Get best parameter combination for each outer fold
  best_params_per_fold <- variation_results %>%
    group_by(outer_fold) %>%
    filter(mean_kappa == max(mean_kappa, na.rm = TRUE)) %>%
    slice(1) %>%
    ungroup()
  
  # Plot 1: Mean kappa by outer fold with error bars
  p1 <- ggplot(best_params_per_fold, aes(x = reorder(outer_fold, mean_kappa), y = mean_kappa)) +
    geom_col(fill = "steelblue", alpha = 0.7) +
    geom_errorbar(aes(ymin = mean_kappa - sd_kappa, ymax = mean_kappa + sd_kappa), 
                  width = 0.2, color = "darkred") +
    labs(
      title = sprintf("Inner CV Performance by Outer Fold (%s)", toupper(model_name)),
      subtitle = "Error bars show standard deviation across inner folds",
      x = "Outer Fold (Test Dataset)",
      y = "Mean Kappa (Inner CV)"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Plot 2: Kappa range (variability) by outer fold
  p2 <- ggplot(best_params_per_fold, aes(x = reorder(outer_fold, range_kappa), y = range_kappa)) +
    geom_col(fill = "orange", alpha = 0.7) +
    labs(
      title = sprintf("Inner CV Performance Variability (%s)", toupper(model_name)),
      subtitle = "Range of kappa values across inner folds",
      x = "Outer Fold (Test Dataset)",
      y = "Kappa Range (Max - Min)"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Plot 3: Coefficient of variation
  p3 <- ggplot(best_params_per_fold, aes(x = reorder(outer_fold, cv_kappa), y = cv_kappa)) +
    geom_col(fill = "green", alpha = 0.7) +
    labs(
      title = sprintf("Inner CV Performance Stability (%s)", toupper(model_name)),
      subtitle = "Coefficient of variation (lower = more stable)",
      x = "Outer Fold (Test Dataset)",
      y = "Coefficient of Variation"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Plot 4: Mean vs variability scatter
  p4 <- ggplot(best_params_per_fold, aes(x = mean_kappa, y = sd_kappa, color = outer_fold)) +
    geom_point(size = 3, alpha = 0.8) +
    geom_text(aes(label = outer_fold), vjust = -0.5, size = 3) +
    labs(
      title = sprintf("Inner CV Performance vs Variability (%s)", toupper(model_name)),
      subtitle = "Higher and to the right = better and more stable",
      x = "Mean Kappa (Inner CV)",
      y = "Standard Deviation (Inner CV)"
    ) +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Save plots
  ggsave(file.path(output_dir, sprintf("%s_inner_cv_performance_by_fold.png", model_name)), 
         p1, width = 10, height = 6)
  ggsave(file.path(output_dir, sprintf("%s_inner_cv_variability_by_fold.png", model_name)), 
         p2, width = 10, height = 6)
  ggsave(file.path(output_dir, sprintf("%s_inner_cv_stability_by_fold.png", model_name)), 
         p3, width = 10, height = 6)
  ggsave(file.path(output_dir, sprintf("%s_inner_cv_performance_vs_variability.png", model_name)), 
         p4, width = 10, height = 6)
  
  cat(sprintf("  Plots saved to: %s\n", output_dir))
}

# =============================================================================
# Main Analysis Function
# =============================================================================

#' Main function to run inner CV variation analysis for all models
main_inner_cv_variation_analysis <- function() {
  cat("=== Starting Inner CV Variation Analysis ===\n")
  
  # Load required libraries
  load_library_quietly("dplyr")
  load_library_quietly("data.table")
  load_library_quietly("stringr")
  
  all_models_variation <- list()
  all_models_summary <- list()
  
  for (model_name in names(MODEL_CONFIGS)) {
    cat(sprintf("\n--- Analyzing %s ---\n", toupper(model_name)))
    
    # Load raw inner CV results
    raw_results <- load_raw_inner_cv_results(model_name)
    
    if (is.null(raw_results)) {
      cat(sprintf("Skipping %s due to data loading issues\n", model_name))
      next
    }
    
    # Analyze variation
    variation_results <- analyze_inner_cv_variation(raw_results, model_name)
    
    if (nrow(variation_results) == 0) {
      cat(sprintf("No variation results for %s\n", model_name))
      next
    }
    
    # Summarize results
    summary_results <- summarize_inner_cv_variation(variation_results)
    
    # Generate plots
    generate_inner_cv_variation_plots(variation_results, model_name)
    
    # Store results
    all_models_variation[[model_name]] <- variation_results
    all_models_summary[[model_name]] <- summary_results
    
    # Display summary
    cat(sprintf("\n=== %s Summary ===\n", toupper(model_name)))
    cat("Fold-by-fold performance:\n")
    print(summary_results$fold_summary)
    cat("\nOverall summary:\n")
    print(summary_results$overall_summary)
  }
  
  # Save results
  output_dir <- "inner_cv_variation_analysis"
  create_directory_safely(output_dir)
  
  for (model_name in names(all_models_variation)) {
    # Save detailed variation results
    write.csv(all_models_variation[[model_name]], 
              file.path(output_dir, sprintf("%s_inner_cv_variation_detailed.csv", model_name)), 
              row.names = FALSE)
    
    # Save summary results
    write.csv(all_models_summary[[model_name]]$fold_summary, 
              file.path(output_dir, sprintf("%s_inner_cv_variation_summary.csv", model_name)), 
              row.names = FALSE)
    
    write.csv(all_models_summary[[model_name]]$overall_summary, 
              file.path(output_dir, sprintf("%s_inner_cv_variation_overall.csv", model_name)), 
              row.names = FALSE)
  }
  
  cat("\n=== Inner CV Variation Analysis Complete! ===\n")
  cat(sprintf("Results saved to: %s\n", output_dir))
  
  return(list(
    variation_results = all_models_variation,
    summary_results = all_models_summary
  ))
}

# Run the analysis if this script is executed directly
if (!exists("SKIP_INNER_CV_VARIATION_EXECUTION")) {
  inner_cv_variation_results <- main_inner_cv_variation_analysis()
}
