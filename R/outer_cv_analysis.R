# Outer CV Analysis Script
# Refactored to work with both XGBOOST and SVM data

library(dplyr)
library(stringr)
library(caret)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

#' Clean probability strings by removing brackets and converting to numeric
#' @param probs_str Character string containing probabilities
#' @return Numeric vector of probabilities
clean_probs_str <- function(probs_str) {
  probs_str %>%
    str_replace_all("\\[|\\]|\\{|\\}|\\\n", "") %>%  # remove brackets and line breaks
    str_squish() %>%                                 # collapse multiple spaces
    str_split(" ") %>%                               # split by space
    unlist() %>%
    as.numeric()
}

#' Convert text to matrix format
#' @param raw_text Raw text containing matrix data
#' @param ncol Number of columns in the matrix
#' @return Matrix object
text_to_matrix <- function(raw_text, ncol) {
  cleaned_text <- raw_text |>
    gsub("\\[|\\]", "", x = _)
  
  num_vec <- as.numeric(unlist(strsplit(cleaned_text, ", ")))
  mat <- matrix(num_vec, ncol = ncol, byrow = TRUE)
  mat
}

#' Clean and standardize label names
#' @param labels Vector of label names
#' @return Cleaned label names
clean_labels <- function(labels, OvO = FALSE) {
  
  if(OvO){
    labels <- make.names(labels)
  }
  labels <- gsub("^AML.with.", "", labels)
  labels <- gsub(".*?/", "", labels)
  labels <- gsub("mutated.", "", labels)
  labels <- gsub("cytogenetic.abnormalities", "CA", labels)
  labels
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

#' Load label mapping and sample indices
#' @param base_path Base path to the data directory
#' @return List containing label_mapping and sample_indices
load_metadata <- function(base_path = "..") {
  label_mapping <- read.csv(file.path(base_path, "label_mapping_df.csv"))
  sample_indices <- read.csv(file.path(base_path, "sample_indices.csv"))
  
  list(
    label_mapping = label_mapping,
    sample_indices = sample_indices
  )
}

#' Load outer CV results for a specific model and method
#' @param model_type Either "XGBOOST" or "SVM"
#' @param method Either "standard", "OvR", or "OvO"
#' @param cv_type Either "CV" or "loso"
#' @param base_path Base path to the outer_cv directory
#' @return Data frame with outer CV results
load_outer_cv_data <- function(model_type, method, cv_type = "CV", base_path = "../out/outer_cv") {
  # Construct filename pattern
  filename_pattern <- paste0(
    model_type, "_outer_cv_", cv_type, "_", method, "_.+.csv"
  )
  
  # Find matching file
  model_dir <- file.path(base_path, model_type)
  files <- list.files(model_dir, pattern = filename_pattern, full.names = TRUE, recursive = TRUE)
  
  if (length(files) == 0) {
    stop(paste("No files found matching pattern:", filename_pattern))
  }
  
  # Load the most recent file (assuming timestamp in filename)
  data <- read.csv(files[1])
  
  # For OvO method, create class column if it doesn't exist
  if (method == "OvO" && !"class" %in% colnames(data)) {
    data$class <- paste(data$class_0, data$class_1, sep = "_")
  }
  
  data
}

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

#' Analyze standard classification results
#' @param data Outer CV results data frame
#' @param label_mapping Label mapping data frame
#' @return List with predictions, true values, and confusion matrix
analyze_standard <- function(data, label_mapping) {
  # Extract predictions and true values
  y_val <- clean_probs_str(unlist(data$y_val))
  preds <- clean_probs_str(unlist(data$preds))
  
  # Ensure consistent factor levels
  levels_un <- unique(y_val)
  preds <- factor(preds, levels = levels_un)
  y_val <- factor(y_val, levels = levels_un)
  
  # Map to label names
  levels_labels <- label_mapping$Label[levels_un + 1]
  levels_labels <- clean_labels(levels_labels)
  
  levels(preds) <- levels_labels
  levels(y_val) <- levels_labels
  
  preds <- factor(preds, (levels_labels))
  y_val <- factor(y_val, (levels_labels))
  
  # Calculate confusion matrix
  res <- caret::confusionMatrix(preds, y_val)
  
  list(
    predictions = preds,
    true_values = y_val,
    confusion_matrix = res,
    kappa = res$overall["Kappa"]
  )
}

#' Analyze One-vs-Rest (OvR) classification results
#' @param data_standard Standard CV results (for y_val reference)
#' @param data_ovr OvR CV results
#' @param label_mapping Label mapping data frame
#' @return List with predictions, true values, and confusion matrix
analyze_ovr <- function(data_standard, data_ovr, label_mapping) {
  # Extract outer fold IDs
  outer_folds <- unique(data_standard$outer_fold)
  
  preds <- list()
  y_val <- list()
  
  # Loop over outer folds
  for (outer_fold_i in outer_folds) {
    # Get y_val for current fold
    probs_str <- data_standard %>%
      filter(outer_fold == outer_fold_i) %>%
      pull(y_val) %>%
      .[1]
    probs_vec <- clean_probs_str(probs_str)
    
    # Initialize probability dataframe
    n_samples <- length(probs_vec)
    prob_df <- data.frame(matrix(nrow = n_samples, ncol = 0))
    
    # Process each target class
    for (target in sort(unique(data_ovr$class))) {
      target_probs <- data_ovr %>%
        filter(outer_fold == outer_fold_i, class == target)
      
      probs_str <- target_probs$preds_prob
      probs_vec <- gsub("\\[|\\]", "", probs_str)
      
      # Add class probabilities
      prob_df[[as.character(target)]] <- as.numeric(unlist(strsplit(probs_vec, ", ")))
    }
    
    # Get predictions and true values for this fold
    preds[[as.character(outer_fold_i)]] <- (apply(prob_df, 1, which.max) - 1)
    y_val[[as.character(outer_fold_i)]] <- clean_probs_str(unlist(data_standard$y_val[data_standard$outer_fold == outer_fold_i]))
  }
  
  # Combine all folds
  y_val <- unlist(y_val)
  preds <- unlist(preds)
  
  # Ensure consistent factor levels
  levels_un <- unique(y_val)
  preds <- factor(preds, levels = levels_un)
  y_val <- factor(y_val, levels = levels_un)
  
  # Map to label names
  levels_labels <- label_mapping$Label[levels_un + 1]
  levels_labels <- clean_labels(levels_labels)
  
  levels(preds) <- levels_labels
  levels(y_val) <- levels_labels
  
  preds <- factor(preds, (levels_labels))
  y_val <- factor(y_val, (levels_labels))
  
  # Calculate confusion matrix
  res <- caret::confusionMatrix(preds, y_val)
  
  list(
    predictions = preds,
    true_values = y_val,
    confusion_matrix = res,
    kappa = res$overall["Kappa"]
  )
}

#' Analyze One-vs-One (OvO) classification results
#' @param data_standard Standard CV results (for y_val reference)
#' @param data_ovo OvO CV results
#' @param label_mapping Label mapping data frame
#' @return List with predictions, true values, and confusion matrix
analyze_ovo <- function(data_standard, data_ovo, label_mapping) {
  # Extract outer fold IDs
  outer_folds <- unique(data_standard$outer_fold)
  
  preds <- list()
  y_val <- list()
  
  # Loop over outer folds
  for (outer_fold_i in outer_folds) {
    # Get y_val for current fold
    probs_str <- data_standard %>%
      filter(outer_fold == outer_fold_i) %>%
      pull(y_val) %>%
      .[1]
    probs_vec <- clean_probs_str(probs_str)
    
    # Initialize voting dataframe
    n_samples <- length(probs_vec)
    target_duos <- sort(unique(data_ovo$class))
    all_classes <- unique(c(data_ovo$class_0_label, data_ovo$class_1_label))
    votes_df <- data.frame(matrix(nrow = n_samples, ncol = length(all_classes), data = 0))
    colnames(votes_df) <- make.names(all_classes)
    
    # Process each class pair
    for (target in target_duos) {
      target_probs <- data_ovo %>%
        filter(outer_fold == outer_fold_i, class == target)
      
      neg <- make.names(target_probs$class_1_label)
      pos <- make.names(target_probs$class_0_label)
      
      probs_str <- target_probs$preds_prob
      probs_vec <- gsub("\\[|\\]", "", probs_str)
      probs <- as.numeric(unlist(strsplit(probs_vec, ", ")))
      
      # Vote based on probability threshold
      votes_df[probs > 0.5, pos] <- votes_df[probs > 0.5, pos] + 1
      votes_df[probs <= 0.5, neg] <- votes_df[probs <= 0.5, neg] + 1
    }
    
    # Get predictions for this fold
    preds_fold <- colnames(votes_df)[apply(votes_df, 1, which.max)]
    preds[[as.character(outer_fold_i)]] <- preds_fold
    y_val[[as.character(outer_fold_i)]] <- clean_probs_str(unlist(data_standard$y_val[data_standard$outer_fold == outer_fold_i]))
  }
  
  # Combine all folds
  y_val <- unlist(y_val)
  preds <- unlist(preds)
  
  # Ensure consistent factor levels
  levels_un <- unique(y_val)
  y_val <- factor(y_val, levels = levels_un)
  
  # Map to label names
  levels_labels <- make.names(label_mapping$Label[levels_un + 1])
  levels_labels <- clean_labels(levels_labels, OvO = TRUE)
  
  levels(y_val) <- levels_labels
  preds <- factor(preds, levels = levels_labels)
  
  preds <- factor(preds, (levels_labels))
  y_val <- factor(y_val, (levels_labels))
  # Calculate confusion matrix
  res <- caret::confusionMatrix(preds, y_val)
  
  list(
    predictions = preds,
    true_values = y_val,
    confusion_matrix = res,
    kappa = res$overall["Kappa"]
  )
}

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

#' Run complete outer CV analysis for a model
#' @param model_type Either "XGBOOST" or "SVM"
#' @param cv_type Either "CV" or "loso"
#' @param base_path Base path to the outer_cv directory
#' @return List with results for all methods
run_outer_cv_analysis <- function(model_type, cv_type = "CV", base_path = "../out/outer_cv") {
  # Load metadata
  metadata <- load_metadata()
  
  # Load data for all methods
  data_standard <- load_outer_cv_data(model_type, "standard", cv_type, base_path)
  data_ovr <- load_outer_cv_data(model_type, "OvR", cv_type, base_path)
  data_ovo <- load_outer_cv_data(model_type, "OvO", cv_type, base_path)
  
  # Run analyses
  results <- list()
  
  cat("Analyzing", model_type, "Standard method...\n")
  results$standard <- analyze_standard(data_standard, metadata$label_mapping)
  
  cat("Analyzing", model_type, "OvR method...\n")
  results$ovr <- analyze_ovr(data_standard, data_ovr, metadata$label_mapping)
  
  cat("Analyzing", model_type, "OvO method...\n")
  results$ovo <- analyze_ovo(data_standard, data_ovo, metadata$label_mapping)
  
  # Print summary
  cat("\n=== RESULTS SUMMARY ===\n")
  cat("Model:", model_type, "\n")
  cat("CV Type:", cv_type, "\n")
  cat("Standard Kappa:", round(results$standard$kappa, 4), "\n")
  cat("OvR Kappa:", round(results$ovr$kappa, 4), "\n")
  cat("OvO Kappa:", round(results$ovo$kappa, 4), "\n")
  
  results
}

