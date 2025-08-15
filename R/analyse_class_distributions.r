# =============================================================================
# Class Distribution Analysis for Dataset Combinations
# =============================================================================
# This script analyzes class distributions across individual datasets and
# when datasets are combined for inner CV (3 datasets) vs outer CV (4 datasets)
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

# =============================================================================
# Source Utility Functions
# =============================================================================

source("R/utility_functions.R")

# =============================================================================
# Data Loading Functions
# =============================================================================

#' Load and analyze class distributions
#' @return List containing class distribution analysis
analyze_class_distributions <- function() {
  cat("=== Analyzing Class Distributions Across Datasets ===\n")
  
  # Load required libraries
  load_library_quietly("dplyr")
  load_library_quietly("ggplot2")
  load_library_quietly("RColorBrewer")
  load_library_quietly("gridExtra")
  load_library_quietly("reshape2")
  load_library_quietly("tidyr")
  
  # Load data
  cat("Loading data files...\n")
  
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
  
  # Create data frame
  data_df <- data.frame(
    subtype = leukemia_subtypes,
    study = study_names,
    stringsAsFactors = FALSE
  )
  
  # Apply same filters as in the analysis
  DATA_FILTERS <- list(
    min_samples_per_subtype = 10,
    excluded_subtypes = c("AML NOS", "Missing data"),
    selected_studies = LOSO_DATASETS
  )
  
  # Filter data
  subtypes_with_sufficient_samples <- names(which(table(leukemia_subtypes) >= DATA_FILTERS$min_samples_per_subtype))
  
  filtered_data <- data_df[
    data_df$subtype %in% subtypes_with_sufficient_samples & 
    !data_df$subtype %in% DATA_FILTERS$excluded_subtypes & 
    data_df$study %in% DATA_FILTERS$selected_studies,
  ]
  
  # Apply class modifications (same as in analysis)
  filtered_data$subtype_modified <- modify_classes(filtered_data$subtype)
  
  cat(sprintf("Total samples after filtering: %d\n", nrow(filtered_data)))
  cat(sprintf("Number of classes: %d\n", length(unique(filtered_data$subtype_modified))))
  cat(sprintf("Number of studies: %d\n", length(unique(filtered_data$study))))
  
  return(filtered_data)
}

#' Analyze individual dataset distributions
#' @param filtered_data Filtered dataset
#' @return Data frame with individual dataset class distributions
analyze_individual_dataset_distributions <- function(filtered_data) {
  cat("\nAnalyzing individual dataset class distributions...\n")
  
  individual_distributions <- data.frame()
  
  for (dataset in LOSO_DATASETS) {
    dataset_data <- filtered_data[filtered_data$study == dataset, ]
    
    if (nrow(dataset_data) == 0) {
      cat(sprintf("  No data for %s\n", dataset))
      next
    }
    
    class_counts <- table(dataset_data$subtype_modified)
    class_props <- prop.table(class_counts)
    
    dataset_summary <- data.frame(
      Dataset = dataset,
      Class = names(class_counts),
      Count = as.numeric(class_counts),
      Proportion = as.numeric(class_props),
      Total_Samples = nrow(dataset_data),
      stringsAsFactors = FALSE
    )
    
    individual_distributions <- rbind(individual_distributions, dataset_summary)
    
    cat(sprintf("  %s: %d samples, %d classes\n", 
               dataset, nrow(dataset_data), length(unique(dataset_data$subtype_modified))))
  }
  
  return(individual_distributions)
}

#' Analyze 3-dataset combinations (inner CV scenarios)
#' @param filtered_data Filtered dataset
#' @return Data frame with 3-dataset combination distributions
analyze_three_dataset_combinations <- function(filtered_data) {
  cat("\nAnalyzing 3-dataset combinations (inner CV scenarios)...\n")
  
  three_dataset_distributions <- data.frame()
  
  for (test_dataset in LOSO_DATASETS) {
    # For inner CV: test_dataset is excluded, and we need 3-dataset combinations
    # from the remaining 4 datasets
    other_datasets <- LOSO_DATASETS[LOSO_DATASETS != test_dataset]
    
    cat(sprintf("  Test dataset: %s\n", test_dataset))
    
    # Generate all 3-dataset combinations from the 4 remaining datasets
    for (i in 1:length(other_datasets)) {
      excluded_from_training <- other_datasets[i]
      training_datasets <- other_datasets[other_datasets != excluded_from_training]
      
      cat(sprintf("    Inner CV fold %d: Train on %s, Validate on %s\n", 
                 i, paste(training_datasets, collapse = "+"), excluded_from_training))
      
      # Get data for the 3 training datasets
      combo_data <- filtered_data[filtered_data$study %in% training_datasets, ]
      
      if (nrow(combo_data) == 0) {
        cat(sprintf("      No data for combination: %s\n", paste(training_datasets, collapse = "+")))
        next
      }
      
      class_counts <- table(combo_data$subtype_modified)
      class_props <- prop.table(class_counts)
      
      combo_summary <- data.frame(
        Test_Dataset = test_dataset,
        Inner_Fold = i,
        Validation_Dataset = excluded_from_training,
        Combination_Type = "3_datasets_inner_cv",
        Class = names(class_counts),
        Count = as.numeric(class_counts),
        Proportion = as.numeric(class_props),
        Total_Samples = nrow(combo_data),
        Training_Datasets = paste(training_datasets, collapse = "+"),
        stringsAsFactors = FALSE
      )
      
      three_dataset_distributions <- rbind(three_dataset_distributions, combo_summary)
      
      cat(sprintf("      Total samples: %d, Classes: %d\n", 
                 nrow(combo_data), length(unique(combo_data$subtype_modified))))
    }
  }
  
  return(three_dataset_distributions)
}

#' Analyze 4-dataset combinations (outer CV scenarios)
#' @param filtered_data Filtered dataset
#' @return Data frame with 4-dataset combination distributions
analyze_four_dataset_combinations <- function(filtered_data) {
  cat("\nAnalyzing 4-dataset combinations (outer CV scenarios)...\n")
  
  four_dataset_distributions <- data.frame()
  
  for (excluded_dataset in LOSO_DATASETS) {
    included_datasets <- LOSO_DATASETS[LOSO_DATASETS != excluded_dataset]
    
    cat(sprintf("  Outer test: %s, Training: %s\n", 
               excluded_dataset, paste(included_datasets, collapse = ", ")))
    
    # Get data for the 4 included datasets (all except the test dataset)
    combo_data <- filtered_data[filtered_data$study %in% included_datasets, ]
    
    if (nrow(combo_data) == 0) {
      cat(sprintf("    No data for combination excluding %s\n", excluded_dataset))
      next
    }
    
    class_counts <- table(combo_data$subtype_modified)
    class_props <- prop.table(class_counts)
    
    combo_summary <- data.frame(
      Test_Dataset = excluded_dataset,
      Combination_Type = "4_datasets",
      Class = names(class_counts),
      Count = as.numeric(class_counts),
      Proportion = as.numeric(class_props),
      Total_Samples = nrow(combo_data),
      Training_Datasets = paste(included_datasets, collapse = "+"),
      stringsAsFactors = FALSE
    )
    
    four_dataset_distributions <- rbind(four_dataset_distributions, combo_summary)
    
    cat(sprintf("    Total samples: %d, Classes: %d\n", 
               nrow(combo_data), length(unique(combo_data$subtype_modified))))
  }
  
  return(four_dataset_distributions)
}

#' Compare class distribution changes for BEAT
#' @param three_dataset_dist 3-dataset distributions
#' @param four_dataset_dist 4-dataset distributions
#' @return Comparison analysis for BEAT
analyze_dataset_distribution_changes <- function(three_dataset_dist, four_dataset_dist, dataset_name) {
  cat(paste0("  Analyzing class distribution changes for ", dataset_name, "...\n"))
  
  # Get dataset 3-dataset scenarios (inner CV training combinations)
  dataset_3dataset <- three_dataset_dist[three_dataset_dist$Test_Dataset == dataset_name, ]
  
  # Get dataset 4-dataset scenario (outer CV training)
  dataset_4dataset <- four_dataset_dist[four_dataset_dist$Test_Dataset == dataset_name, ]
  
  if (nrow(dataset_3dataset) == 0 || nrow(dataset_4dataset) == 0) {
    cat(paste0("    Unable to compare ", dataset_name, " scenarios\n"))
    return(NULL)
  }
  
  # Calculate average proportions and standard deviations across all 3-dataset inner CV combinations
  dataset_3dataset_avg <- dataset_3dataset %>%
    group_by(Class) %>%
    summarise(
      Count_3dataset = mean(Count, na.rm = TRUE),
      Proportion_3dataset = mean(Proportion, na.rm = TRUE),
      Proportion_3dataset_SD = sd(Proportion, na.rm = TRUE),
      Total_Samples_3dataset = mean(Total_Samples, na.rm = TRUE),
      N_Combinations = n(),
      .groups = "drop"
    )
  
  # Get 4-dataset scenario data
  dataset_4dataset_summary <- dataset_4dataset %>%
    select(Class, Count, Proportion, Total_Samples) %>%
    rename(
      Count_4dataset = Count,
      Proportion_4dataset = Proportion,
      Total_Samples_4dataset = Total_Samples
    )
  
  # Merge data for comparison
  comparison <- merge(
    dataset_3dataset_avg,
    dataset_4dataset_summary,
    by = "Class",
    all = TRUE
  )
  
  # Handle missing values
  comparison[is.na(comparison)] <- 0
  
  # Calculate changes
  comparison$Count_Change <- comparison$Count_4dataset - comparison$Count_3dataset
  comparison$Proportion_Change <- comparison$Proportion_4dataset - comparison$Proportion_3dataset
  comparison$Relative_Change <- ifelse(comparison$Proportion_3dataset > 0, 
                                      (comparison$Proportion_Change / comparison$Proportion_3dataset) * 100, 
                                      0)
  
  # Add scenario details
  comparison$Total_Samples_3dataset_avg <- comparison$Total_Samples_3dataset[1]
  comparison$Total_Samples_4dataset <- comparison$Total_Samples_4dataset[1]
  
  cat(sprintf("  3-dataset scenarios (average): %.0f total samples\n", comparison$Total_Samples_3dataset_avg[1]))
  cat(sprintf("  4-dataset scenario: %d total samples\n", comparison$Total_Samples_4dataset[1]))
  cat(sprintf("  Sample increase: %.0f (%.1f%%)\n", 
             comparison$Total_Samples_4dataset[1] - comparison$Total_Samples_3dataset_avg[1],
             ((comparison$Total_Samples_4dataset[1] / comparison$Total_Samples_3dataset_avg[1]) - 1) * 100))
  
  return(comparison)
}

# Backward compatibility wrapper for BEAT
analyze_beat_distribution_changes <- function(three_dataset_dist, four_dataset_dist) {
  cat("\nAnalyzing class distribution changes for BEAT...\n")
  return(analyze_dataset_distribution_changes(three_dataset_dist, four_dataset_dist, "BEATAML1.0-COHORT"))
}

#' Generate class distribution visualizations
#' @param individual_dist Individual dataset distributions
#' @param three_dataset_dist 3-dataset distributions
#' @param four_dataset_dist 4-dataset distributions
#' @param beat_comparison BEAT comparison analysis
#' @param output_dir Output directory
generate_class_distribution_plots <- function(individual_dist, three_dataset_dist, four_dataset_dist, all_dataset_comparisons, output_dir = "class_distribution_analysis") {
  cat("\nGenerating class distribution visualizations...\n")
  
  # Create output directory
  create_directory_safely(output_dir)
  
  # Plot 1: Individual dataset class distributions
  p1 <- ggplot(individual_dist, aes(x = Class, y = Count, fill = Dataset)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    facet_wrap(~Dataset, scales = "free_y", ncol = 2) +
    labs(
      title = "Class Distributions Across Individual Datasets",
      subtitle = "Count of samples per class in each study",
      x = "Class",
      y = "Count"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      plot.title = element_text(size = 14, face = "bold"),
      legend.position = "none"
    )
  
  # Plot 2: Individual dataset proportions
  p2 <- ggplot(individual_dist, aes(x = Class, y = Proportion, fill = Dataset)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    facet_wrap(~Dataset, scales = "free_y", ncol = 2) +
    labs(
      title = "Class Proportions Across Individual Datasets",
      subtitle = "Proportion of samples per class in each study",
      x = "Class",
      y = "Proportion"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
      plot.title = element_text(size = 14, face = "bold"),
      legend.position = "none"
    )
  
  # Plot 3+: Dataset comparisons (if available)
  dataset_plots <- list()
  relative_change_plots <- list()
  
  if (!is.null(all_dataset_comparisons) && length(all_dataset_comparisons) > 0) {
    for (dataset_name in names(all_dataset_comparisons)) {
      comparison <- all_dataset_comparisons[[dataset_name]]
      
      if (!is.null(comparison)) {
        # Reshape data for plotting with error bars
        plot_data <- comparison %>%
          select(Class, Proportion_3dataset, Proportion_3dataset_SD, Proportion_4dataset) %>%
          tidyr::pivot_longer(
            cols = c(Proportion_3dataset, Proportion_4dataset),
            names_to = "Scenario",
            values_to = "Proportion"
          )
        
        # Add corresponding SD values (only for 3-dataset scenario)
        plot_data$SD_Value <- ifelse(
          plot_data$Scenario == "Proportion_3dataset",
          comparison$Proportion_3dataset_SD[match(plot_data$Class, comparison$Class)],
          0  # No error bars for 4-dataset (single evaluation)
        )
        
        plot_data$Scenario <- factor(
          plot_data$Scenario,
          levels = c("Proportion_3dataset", "Proportion_4dataset"),
          labels = c("3 Datasets (Inner CV)", "4 Datasets (Outer CV)")
        )
        
        # Create distribution comparison plot
        p_dist <- ggplot(plot_data, aes(x = Class, y = Proportion, fill = Scenario)) +
          geom_bar(stat = "identity", position = position_dodge(width = 0.9), alpha = 0.8) +
          # Add error bars only where SD > 0 (Inner CV for 3-dataset combinations)
          # Position them manually to align with the 3-dataset bars (left side)
          geom_errorbar(
            data = subset(plot_data, SD_Value > 0),
            aes(ymin = Proportion - SD_Value, ymax = Proportion + SD_Value,
                x = as.numeric(as.factor(Class)) - 0.225),  # Shift left to align with 3-dataset bars
            width = 0.2,
            color = "black",
            alpha = 0.8,
            show.legend = FALSE,
            inherit.aes = FALSE
          ) +
          scale_fill_brewer(palette = "Set1", name = "Training Scenario") +
          labs(
            title = paste0(dataset_name, ": Class Distribution Changes (with Variability)"),
            subtitle = "Error bars show SD across 3-dataset combinations (Inner CV only)",
            x = "Class",
            y = "Proportion"
          ) +
          theme_minimal() +
          theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            plot.title = element_text(size = 12, face = "bold"),
            legend.position = "bottom"
          )
        
        # Create relative changes plot
        p_rel <- ggplot(comparison, aes(x = reorder(Class, Relative_Change), y = Relative_Change)) +
          geom_col(aes(fill = ifelse(Relative_Change > 0, "Increase", "Decrease")), alpha = 0.8) +
          geom_hline(yintercept = 0, linetype = "dashed") +
          scale_fill_manual(values = c("Increase" = "red", "Decrease" = "blue"), name = "Change Direction") +
          labs(
            title = paste0(dataset_name, ": Relative Class Proportion Changes"),
            subtitle = "% change from 3-dataset to 4-dataset training scenarios",
            x = "Class",
            y = "Relative Change (%)"
          ) +
          theme_minimal() +
          theme(
            axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
            plot.title = element_text(size = 12, face = "bold"),
            legend.position = "bottom"
          ) +
          coord_flip()
        
        dataset_plots[[dataset_name]] <- p_dist
        relative_change_plots[[dataset_name]] <- p_rel
      }
    }
  }
  
  # Save plots
  ggsave(file.path(output_dir, "individual_dataset_counts.png"), p1, width = 14, height = 10)
  ggsave(file.path(output_dir, "individual_dataset_proportions.png"), p2, width = 14, height = 10)
  
  # Save dataset-specific plots
  if (length(dataset_plots) > 0) {
    for (dataset_name in names(dataset_plots)) {
      # Clean dataset name for filename
      clean_name <- gsub("[^A-Za-z0-9]", "_", dataset_name)
      
      ggsave(
        file.path(output_dir, paste0(clean_name, "_distribution_comparison.png")),
        dataset_plots[[dataset_name]], 
        width = 12, height = 8
      )
      
      ggsave(
        file.path(output_dir, paste0(clean_name, "_relative_changes.png")),
        relative_change_plots[[dataset_name]], 
        width = 10, height = 8
      )
    }
  }
  
  cat(sprintf("  Plots saved to: %s\n", output_dir))
}

#' Enhanced performance comparison plot with error bars
#' @param comprehensive_comparison Performance comparison data
#' @param output_dir Output directory
generate_enhanced_performance_plot_with_error_bars <- function(comprehensive_comparison, output_dir = "multi_dataset_analysis") {
  cat("\nGenerating enhanced performance comparison plot with error bars...\n")
  
  # Load required libraries
  load_library_quietly("ggplot2")
  load_library_quietly("dplyr")
  load_library_quietly("tidyr")
  
  # Filter for models with inner CV performance
  models_with_inner_cv <- comprehensive_comparison[!is.na(comprehensive_comparison$Inner_CV_Performance), ]
  
  if (nrow(models_with_inner_cv) == 0) {
    cat("  No model performance data available for visualization\n")
    return()
  }
  
  # Calculate summary statistics for error bars
  summary_stats <- models_with_inner_cv %>%
    group_by(Dataset, Model) %>%
    summarise(
      Inner_CV_Mean = mean(Inner_CV_Performance, na.rm = TRUE),
      Inner_CV_SD = sd(Inner_CV_Performance, na.rm = TRUE),
      Outer_CV_Mean = mean(Outer_CV_Kappa, na.rm = TRUE),
      Outer_CV_SD = sd(Outer_CV_Kappa, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Handle cases where SD is NA (single observation)
  summary_stats$Inner_CV_SD[is.na(summary_stats$Inner_CV_SD)] <- 0
  summary_stats$Outer_CV_SD[is.na(summary_stats$Outer_CV_SD)] <- 0
  
  # Reshape data for plotting
  plot_data_long <- summary_stats %>%
    select(Dataset, Model, Inner_CV_Mean, Inner_CV_SD, Outer_CV_Mean, Outer_CV_SD) %>%
    tidyr::pivot_longer(
      cols = c(Inner_CV_Mean, Outer_CV_Mean),
      names_to = "Performance_Type",
      values_to = "Performance_Value"
    )
  
  # Add corresponding SD values
  plot_data_long$SD_Value <- ifelse(
    plot_data_long$Performance_Type == "Inner_CV_Mean",
    summary_stats$Inner_CV_SD[match(paste(plot_data_long$Dataset, plot_data_long$Model), 
                                   paste(summary_stats$Dataset, summary_stats$Model))],
    summary_stats$Outer_CV_SD[match(paste(plot_data_long$Dataset, plot_data_long$Model), 
                                   paste(summary_stats$Dataset, summary_stats$Model))]
  )
  
  plot_data_long$Performance_Type <- factor(
    plot_data_long$Performance_Type,
    levels = c("Inner_CV_Mean", "Outer_CV_Mean"),
    labels = c("Inner CV", "Outer CV")
  )
  
  # Enhanced performance comparison plot with error bars
  p_enhanced <- ggplot(plot_data_long, aes(x = Dataset, y = Performance_Value, fill = Performance_Type)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8, width = 0.7) +
    geom_errorbar(
      aes(ymin = Performance_Value - SD_Value, ymax = Performance_Value + SD_Value),
      position = position_dodge(width = 0.7),
      width = 0.2,
      color = "black",
      alpha = 0.8
    ) +
    facet_wrap(~Model, scales = "free_y", ncol = 3) +
    scale_fill_brewer(palette = "Set1", name = "Performance Type") +
    labs(
      title = "Performance Comparison: Inner CV vs Outer CV (with Error Bars)",
      subtitle = "Kappa performance across all datasets and models (error bars show standard deviation)",
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
  
  # Save enhanced plot
  create_directory_safely(output_dir)
  ggsave(file.path(output_dir, "performance_comparison_barplot_with_error_bars.png"), 
         p_enhanced, width = 14, height = 10)
  
  cat(sprintf("  Enhanced plot saved to: %s\n", 
             file.path(output_dir, "performance_comparison_barplot_with_error_bars.png")))
}

# =============================================================================
# Main Analysis Function
# =============================================================================

#' Main function to run class distribution analysis
main_class_distribution_analysis <- function() {
  cat("=== Starting Class Distribution Analysis ===\n")
  
  # Load required libraries
  load_library_quietly("dplyr")
  load_library_quietly("ggplot2")
  load_library_quietly("tidyr")
  
  # Load and filter data
  filtered_data <- analyze_class_distributions()
  
  # Analyze individual dataset distributions
  individual_dist <- analyze_individual_dataset_distributions(filtered_data)
  
  # Analyze 3-dataset combinations (inner CV scenarios)
  three_dataset_dist <- analyze_three_dataset_combinations(filtered_data)
  
  # Analyze 4-dataset combinations (outer CV scenarios)
  four_dataset_dist <- analyze_four_dataset_combinations(filtered_data)
  
  # Generate distribution comparison for all LOSO datasets
  all_dataset_comparisons <- list()
  datasets <- c("AAML0531", "AAML1031", "BEATAML1.0-COHORT", "LEUCEGENE", "TCGA-LAML")
  
  for (dataset in datasets) {
    cat(paste0("Analyzing distribution changes for ", dataset, "...\n"))
    comparison <- analyze_dataset_distribution_changes(three_dataset_dist, four_dataset_dist, dataset)
    all_dataset_comparisons[[dataset]] <- comparison
  }
  
  # Generate visualizations
  generate_class_distribution_plots(individual_dist, three_dataset_dist, four_dataset_dist, all_dataset_comparisons)
  
  # Load performance comparison data and generate enhanced plot
  perf_comparison_file <- "multi_dataset_analysis/comprehensive_performance_comparison.csv"
  if (file.exists(perf_comparison_file)) {
    comprehensive_comparison <- read.csv(perf_comparison_file)
    generate_enhanced_performance_plot_with_error_bars(comprehensive_comparison)
  }
  
  # Save results
  output_dir <- "class_distribution_analysis"
  create_directory_safely(output_dir)
  
  write.csv(individual_dist, 
            file.path(output_dir, "individual_dataset_distributions.csv"), 
            row.names = FALSE)
  
  write.csv(three_dataset_dist, 
            file.path(output_dir, "three_dataset_combinations.csv"), 
            row.names = FALSE)
  
  write.csv(four_dataset_dist, 
            file.path(output_dir, "four_dataset_combinations.csv"), 
            row.names = FALSE)
  
  # Save dataset-specific comparison data
  for (dataset_name in names(all_dataset_comparisons)) {
    if (!is.null(all_dataset_comparisons[[dataset_name]])) {
      clean_name <- gsub("[^A-Za-z0-9]", "_", dataset_name)
      write.csv(all_dataset_comparisons[[dataset_name]], 
                file.path(output_dir, paste0(clean_name, "_distribution_changes.csv")), 
                row.names = FALSE)
    }
  }
  
  cat("\n=== Class Distribution Analysis Complete! ===\n")
  cat(sprintf("Results saved to: %s\n", output_dir))
  
  # Display key findings for BEAT (if available)
  beat_comparison <- all_dataset_comparisons[["BEATAML1.0-COHORT"]]
  if (!is.null(beat_comparison)) {
    cat("\n=== KEY FINDINGS FOR BEAT ===\n")
    cat(sprintf("Sample size change: %.0f → %d (+%.1f%%)\n",
               beat_comparison$Total_Samples_3dataset_avg[1],
               beat_comparison$Total_Samples_4dataset[1],
               ((beat_comparison$Total_Samples_4dataset[1] / beat_comparison$Total_Samples_3dataset_avg[1]) - 1) * 100))
    
    cat("\nTop 5 classes with largest relative proportion changes:\n")
    top_changes <- beat_comparison[order(abs(beat_comparison$Relative_Change), decreasing = TRUE)[1:5], ]
    print(top_changes[, c("Class", "Proportion_3dataset", "Proportion_4dataset", "Relative_Change")])
  }
  
  return(list(
    individual_distributions = individual_dist,
    three_dataset_distributions = three_dataset_dist,
    four_dataset_distributions = four_dataset_dist,
    all_dataset_comparisons = all_dataset_comparisons,
    filtered_data = filtered_data
  ))
}

# Run the analysis if this script is executed directly
if (!exists("SKIP_CLASS_DISTRIBUTION_EXECUTION")) {
  class_distribution_results <- main_class_distribution_analysis()
}
