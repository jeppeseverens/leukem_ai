# =============================================================================
# Inner CV Variability Analysis
# =============================================================================
# This script creates comprehensive visualizations and tables showing inner CV
# standard deviations across models and evaluation datasets to understand
# performance variability patterns.
# =============================================================================

# Set working directory
setwd("~/Documents/AML_PhD/leukem_ai")

# =============================================================================
# Source Utility Functions
# =============================================================================

source("R/utility_functions.R")

# =============================================================================
# Data Loading and Processing Functions
# =============================================================================

#' Load and combine all inner CV variability data
#' @return Data frame with all inner CV variability information
load_all_inner_cv_variability <- function() {
  cat("Loading all inner CV variability data...\n")
  
  # Load individual model variation data
  models <- c("svm", "xgboost", "neural_net")
  all_variability <- data.frame()
  
  for (model in models) {
    summary_file <- file.path("inner_cv_variation_analysis", 
                             paste0(model, "_inner_cv_variation_summary.csv"))
    
    if (file.exists(summary_file)) {
      model_data <- read.csv(summary_file, stringsAsFactors = FALSE)
      model_data$model <- model  # Ensure consistent column naming
      all_variability <- rbind(all_variability, model_data)
      cat(sprintf("  Loaded %s: %d datasets\n", model, nrow(model_data)))
    } else {
      warning(sprintf("Summary file not found: %s", summary_file))
    }
  }
  
  # Load performance comparison data for reference
  perf_file <- "multi_dataset_analysis/comprehensive_performance_comparison.csv"
  if (file.exists(perf_file)) {
    perf_data <- read.csv(perf_file, stringsAsFactors = FALSE)
    
    # Filter for individual models and add outer CV performance
    individual_perf <- perf_data[perf_data$Model %in% models, 
                                c("Dataset", "Model", "Outer_CV_Kappa", "Performance_Degradation")]
    
    # Merge with variability data
    all_variability <- merge(all_variability, individual_perf, 
                            by.x = c("outer_fold", "model"), 
                            by.y = c("Dataset", "Model"),
                            all.x = TRUE)
  }
  
  cat(sprintf("Combined variability data: %d rows\n", nrow(all_variability)))
  return(all_variability)
}

#' Create comprehensive inner CV variability table
#' @param variability_data Combined variability data
#' @return Formatted table for display
create_variability_summary_table <- function(variability_data) {
  cat("Creating comprehensive variability summary table...\n")
  
  load_library_quietly("dplyr")
  
  # Create a comprehensive summary table
  summary_table <- variability_data %>%
    select(outer_fold, model, mean_kappa, sd_kappa, cv_kappa, 
           Outer_CV_Kappa, Performance_Degradation) %>%
    arrange(model, outer_fold) %>%
    mutate(
      Dataset = outer_fold,
      Model = toupper(model),
      Inner_CV_Mean = round(mean_kappa, 4),
      Inner_CV_SD = round(sd_kappa, 4),
      Inner_CV_CV = round(cv_kappa, 4),
      Outer_CV_Kappa = round(Outer_CV_Kappa, 4),
      Performance_Drop = round(Performance_Degradation, 4),
      Stability_Rating = case_when(
        cv_kappa < 0.2 ~ "Very Stable",
        cv_kappa < 0.4 ~ "Stable", 
        cv_kappa < 0.6 ~ "Moderate",
        cv_kappa < 0.8 ~ "Unstable",
        TRUE ~ "Very Unstable"
      )
    ) %>%
    select(Dataset, Model, Inner_CV_Mean, Inner_CV_SD, Inner_CV_CV, 
           Outer_CV_Kappa, Performance_Drop, Stability_Rating)
  
  return(summary_table)
}

#' Generate comprehensive variability visualizations
#' @param variability_data Combined variability data
#' @param summary_table Formatted summary table
#' @param output_dir Output directory
generate_comprehensive_variability_plots <- function(variability_data, summary_table, output_dir = "inner_cv_variability_analysis") {
  cat("Generating comprehensive variability visualizations...\n")
  
  # Load required libraries
  load_library_quietly("ggplot2")
  load_library_quietly("dplyr")
  load_library_quietly("gridExtra")
  load_library_quietly("RColorBrewer")
  load_library_quietly("reshape2")
  
  # Create output directory
  create_directory_safely(output_dir)
  
  # Prepare data for plotting
  plot_data <- variability_data %>%
    mutate(
      Dataset = outer_fold,
      Model = factor(toupper(model), levels = c("SVM", "XGBOOST", "NEURAL_NET"))
    )
  
  # Plot 1: Inner CV Standard Deviation Heatmap
  p1 <- ggplot(plot_data, aes(x = Model, y = Dataset, fill = sd_kappa)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = sprintf("%.3f", sd_kappa)), 
              color = "white", size = 4, fontface = "bold") +
    scale_fill_gradient2(
      low = "green", 
      mid = "yellow", 
      high = "red",
      midpoint = 0.25,
      name = "Inner CV\nStandard\nDeviation",
      limits = c(0, max(plot_data$sd_kappa, na.rm = TRUE))
    ) +
    labs(
      title = "Inner CV Performance Variability (Standard Deviation)",
      subtitle = "Lower values indicate more stable hyperparameter optimization",
      x = "Model",
      y = "Evaluation Dataset"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )
  
  # Plot 2: Coefficient of Variation Heatmap
  p2 <- ggplot(plot_data, aes(x = Model, y = Dataset, fill = cv_kappa)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = sprintf("%.3f", cv_kappa)), 
              color = "white", size = 4, fontface = "bold") +
    scale_fill_gradient2(
      low = "green", 
      mid = "yellow", 
      high = "red",
      midpoint = 0.4,
      name = "Coefficient\nof Variation",
      limits = c(0, max(plot_data$cv_kappa, na.rm = TRUE))
    ) +
    labs(
      title = "Inner CV Performance Relative Variability (CV)",
      subtitle = "Coefficient of Variation = SD/Mean (lower = more stable)",
      x = "Model",
      y = "Evaluation Dataset"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )
  
  # Plot 3: Inner CV Mean vs SD Scatter
  p3 <- ggplot(plot_data, aes(x = mean_kappa, y = sd_kappa, color = Model)) +
    geom_point(size = 4, alpha = 0.8) +
    geom_text(aes(label = Dataset), vjust = -0.5, size = 3) +
    scale_color_brewer(palette = "Set1", name = "Model") +
    labs(
      title = "Inner CV Performance vs Variability",
      subtitle = "Top-right = good performance but unstable, Bottom-right = good and stable",
      x = "Inner CV Mean Performance (Kappa)",
      y = "Inner CV Standard Deviation"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10),
      legend.position = "bottom"
    )
  
  # Plot 4: Variability vs Performance Degradation
  if ("Performance_Degradation" %in% colnames(plot_data)) {
    p4 <- ggplot(plot_data, aes(x = sd_kappa, y = Performance_Degradation, color = Model)) +
      geom_point(size = 4, alpha = 0.8) +
      geom_text(aes(label = Dataset), vjust = -0.5, size = 3) +
      geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
      scale_color_brewer(palette = "Set1", name = "Model") +
      labs(
        title = "Inner CV Variability vs Performance Degradation",
        subtitle = "Does higher inner CV variability predict worse outer CV performance?",
        x = "Inner CV Standard Deviation",
        y = "Performance Degradation (Inner - Outer)"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 10),
        legend.position = "bottom"
      )
  }
  
  # Plot 5: Side-by-side comparison for BEAT
  beat_data <- plot_data[plot_data$Dataset == "BEATAML1.0-COHORT", ]
  if (nrow(beat_data) > 0) {
    beat_comparison <- data.frame(
      Model = beat_data$Model,
      Inner_CV_Mean = beat_data$mean_kappa,
      Inner_CV_SD = beat_data$sd_kappa,
      Outer_CV = beat_data$Outer_CV_Kappa
    )
    
    beat_long <- reshape2::melt(beat_comparison, 
                               id.vars = c("Model", "Inner_CV_SD"),
                               measure.vars = c("Inner_CV_Mean", "Outer_CV"),
                               variable.name = "CV_Type",
                               value.name = "Performance")
    
    p5 <- ggplot(beat_long, aes(x = Model, y = Performance, fill = CV_Type)) +
      geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
      geom_errorbar(
        data = subset(beat_long, CV_Type == "Inner_CV_Mean"),
        aes(ymin = Performance - Inner_CV_SD, ymax = Performance + Inner_CV_SD),
        position = position_dodge(width = 0.9),
        width = 0.2,
        color = "black"
      ) +
      scale_fill_brewer(palette = "Set2", name = "CV Type",
                       labels = c("Inner CV (Mean ± SD)", "Outer CV")) +
      labs(
        title = "BEAT Dataset: Inner CV vs Outer CV Performance",
        subtitle = "Error bars show inner CV standard deviation across folds",
        x = "Model",
        y = "Performance (Kappa)"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 10),
        legend.position = "bottom"
      )
  }
  
  # Save individual plots
  ggsave(file.path(output_dir, "inner_cv_sd_heatmap.png"), p1, width = 10, height = 8)
  ggsave(file.path(output_dir, "inner_cv_cv_heatmap.png"), p2, width = 10, height = 8)
  ggsave(file.path(output_dir, "performance_vs_variability_scatter.png"), p3, width = 10, height = 8)
  
  if (exists("p4")) {
    ggsave(file.path(output_dir, "variability_vs_degradation.png"), p4, width = 10, height = 8)
  }
  
  if (exists("p5")) {
    ggsave(file.path(output_dir, "beat_detailed_comparison.png"), p5, width = 10, height = 8)
  }
  
  # Create combined overview plot
  if (exists("p4") && exists("p5")) {
    combined_plot <- grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)
    ggsave(file.path(output_dir, "comprehensive_variability_overview.png"), 
           combined_plot, width = 16, height = 12)
  }
  
  cat(sprintf("  All plots saved to: %s\n", output_dir))
}

#' Create detailed variability statistics
#' @param variability_data Combined variability data
#' @return List of detailed statistics
generate_detailed_variability_stats <- function(variability_data) {
  cat("Generating detailed variability statistics...\n")
  
  load_library_quietly("dplyr")
  
  # Overall model statistics
  model_stats <- variability_data %>%
    group_by(model) %>%
    summarise(
      n_datasets = n(),
      mean_inner_cv_performance = mean(mean_kappa, na.rm = TRUE),
      sd_inner_cv_performance = sd(mean_kappa, na.rm = TRUE),
      mean_variability = mean(sd_kappa, na.rm = TRUE),
      min_variability = min(sd_kappa, na.rm = TRUE),
      max_variability = max(sd_kappa, na.rm = TRUE),
      mean_cv = mean(cv_kappa, na.rm = TRUE),
      most_stable_dataset = outer_fold[which.min(sd_kappa)],
      least_stable_dataset = outer_fold[which.max(sd_kappa)],
      .groups = "drop"
    ) %>%
    arrange(mean_variability)
  
  # Dataset-specific statistics
  dataset_stats <- variability_data %>%
    group_by(outer_fold) %>%
    summarise(
      n_models = n(),
      mean_inner_cv_performance = mean(mean_kappa, na.rm = TRUE),
      mean_variability = mean(sd_kappa, na.rm = TRUE),
      most_stable_model = model[which.min(sd_kappa)],
      least_stable_model = model[which.max(sd_kappa)],
      stability_range = max(sd_kappa, na.rm = TRUE) - min(sd_kappa, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(mean_variability)
  
  # Stability rankings
  stability_ranking <- variability_data %>%
    arrange(sd_kappa) %>%
    mutate(
      rank = row_number(),
      combination = paste(outer_fold, model, sep = " - ")
    ) %>%
    select(rank, combination, mean_kappa, sd_kappa, cv_kappa)
  
  # Correlations (if performance degradation data is available)
  correlations <- NULL
  if ("Performance_Degradation" %in% colnames(variability_data)) {
    correlations <- data.frame(
      Metric_Pair = c(
        "Inner_CV_SD vs Performance_Degradation",
        "Inner_CV_CV vs Performance_Degradation", 
        "Inner_CV_Mean vs Performance_Degradation"
      ),
      Correlation = c(
        cor(variability_data$sd_kappa, variability_data$Performance_Degradation, use = "complete.obs"),
        cor(variability_data$cv_kappa, variability_data$Performance_Degradation, use = "complete.obs"),
        cor(variability_data$mean_kappa, variability_data$Performance_Degradation, use = "complete.obs")
      )
    )
  }
  
  return(list(
    model_stats = model_stats,
    dataset_stats = dataset_stats,
    stability_ranking = stability_ranking,
    correlations = correlations
  ))
}

# =============================================================================
# Main Analysis Function
# =============================================================================

#' Main function for comprehensive inner CV variability analysis
main_inner_cv_variability_analysis <- function() {
  cat("=== Starting Comprehensive Inner CV Variability Analysis ===\n")
  
  # Load required libraries
  load_library_quietly("ggplot2")
  load_library_quietly("dplyr")
  load_library_quietly("gridExtra")
  load_library_quietly("reshape2")
  
  # Load and process data
  variability_data <- load_all_inner_cv_variability()
  summary_table <- create_variability_summary_table(variability_data)
  
  # Generate visualizations
  generate_comprehensive_variability_plots(variability_data, summary_table)
  
  # Generate detailed statistics
  detailed_stats <- generate_detailed_variability_stats(variability_data)
  
  # Save results
  output_dir <- "inner_cv_variability_analysis"
  create_directory_safely(output_dir)
  
  # Save comprehensive summary table
  write.csv(summary_table, 
            file.path(output_dir, "comprehensive_variability_summary.csv"), 
            row.names = FALSE)
  
  # Save detailed statistics
  write.csv(detailed_stats$model_stats, 
            file.path(output_dir, "model_variability_statistics.csv"), 
            row.names = FALSE)
  
  write.csv(detailed_stats$dataset_stats, 
            file.path(output_dir, "dataset_variability_statistics.csv"), 
            row.names = FALSE)
  
  write.csv(detailed_stats$stability_ranking, 
            file.path(output_dir, "stability_ranking.csv"), 
            row.names = FALSE)
  
  if (!is.null(detailed_stats$correlations)) {
    write.csv(detailed_stats$correlations, 
              file.path(output_dir, "variability_correlations.csv"), 
              row.names = FALSE)
  }
  
  # Display key findings
  cat("\n=== KEY FINDINGS ===\n")
  
  cat("\nModel Variability Rankings (most to least stable):\n")
  print(detailed_stats$model_stats[, c("model", "mean_variability", "mean_cv")])
  
  cat("\nDataset Variability Rankings (most to least stable):\n") 
  print(detailed_stats$dataset_stats[, c("outer_fold", "mean_variability", "most_stable_model")])
  
  cat("\nTop 5 Most Stable Combinations:\n")
  print(head(detailed_stats$stability_ranking, 5))
  
  cat("\nTop 5 Least Stable Combinations:\n")
  print(tail(detailed_stats$stability_ranking, 5))
  
  if (!is.null(detailed_stats$correlations)) {
    cat("\nCorrelations with Performance Degradation:\n")
    print(detailed_stats$correlations)
  }
  
  cat("\n=== BEAT Dataset Specific Findings ===\n")
  beat_data <- summary_table[summary_table$Dataset == "BEATAML1.0-COHORT", ]
  if (nrow(beat_data) > 0) {
    print(beat_data)
    
    cat("\nBEAT Stability Analysis:\n")
    cat(sprintf("- Neural Net: %.4f ± %.4f (CV = %.3f) - %s\n",
               beat_data$Inner_CV_Mean[beat_data$Model == "NEURAL_NET"],
               beat_data$Inner_CV_SD[beat_data$Model == "NEURAL_NET"],
               beat_data$Inner_CV_CV[beat_data$Model == "NEURAL_NET"],
               beat_data$Stability_Rating[beat_data$Model == "NEURAL_NET"]))
  }
  
  cat("\n=== Analysis Complete! ===\n")
  cat(sprintf("Results saved to: %s\n", output_dir))
  
  return(list(
    variability_data = variability_data,
    summary_table = summary_table,
    detailed_stats = detailed_stats
  ))
}

# Run the analysis if this script is executed directly
if (!exists("SKIP_INNER_CV_VARIABILITY_EXECUTION")) {
  inner_cv_variability_results <- main_inner_cv_variability_analysis()
}
