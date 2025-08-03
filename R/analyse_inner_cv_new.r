# =============================================================================
# Cross-Validation Analysis for Machine Learning Models
# =============================================================================
# This script analyzes inner cross-validation results for SVM, XGBoost, and 
# Neural Network models, extracts best hyperparameters, and generates prediction
# probability matrices for further analysis.
# =============================================================================

# =============================================================================
# Configuration and Constants
# =============================================================================

# Model types and their configurations
MODEL_CONFIGS <- list(
  svm = list(
    classification_type = "OvR",
    file_paths = list(
      cv = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250611_1318/SVM_inner_cv_OvR_20250611_1318.csv",
      loso = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250611_1927/SVM_inner_cv_loso_OvR_20250611_1927.csv"
    ),
    output_dir = "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/SVM"
  ),
  xgboost = list(
    classification_type = "OvR",
    file_paths = list(
      cv = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250612_0024/XGBOOST_inner_cv_OvR_20250612_0024.csv",
      loso = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250612_0516/XGBOOST_inner_cv_loso_OvR_20250612_0516.csv"
    ),
    output_dir = "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/XGBOOST"
  ),
  neural_net = list(
    classification_type = "standard",
    file_paths = list(
      cv = "/Users/jsevere2/Documents/AML_PhD/predictor_out/NN/cv_27jul25",
      loso = "/Users/jsevere2/Documents/AML_PhD/predictor_out/NN/loso_27jul25/"
    ),
    output_dir = "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/NN"
  )
)

# Data filtering criteria
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

# Ensemble weight configurations
ENSEMBLE_WEIGHTS <- list(
  # Individual models
  SVM = list(SVM = 1, XGB = 0, NN = 0),
  XGB = list(SVM = 0, XGB = 1, NN = 0),
  NN = list(SVM = 0, XGB = 0, NN = 1),

  # Equal weight combinations
  SVM_XGB = list(SVM = 1, XGB = 1, NN = 0),
  SVM_NN = list(SVM = 1, XGB = 0, NN = 1),
  XGB_NN = list(SVM = 0, XGB = 1, NN = 1),
  ALL = list(SVM = 1, XGB = 1, NN = 1)# ,

  # Heavy weight combinations (2:1:1 ratios)
  # SVM_heavy = list(SVM = 2, XGB = 1, NN = 1),
  # XGB_heavy = list(SVM = 1, XGB = 2, NN = 1),
  # NN_heavy = list(SVM = 1, XGB = 1, NN = 2),
  # 
  # # Very heavy weight combinations (3:1:1 ratios)
  # SVM_very_heavy = list(SVM = 3, XGB = 1, NN = 1),
  # XGB_very_heavy = list(SVM = 1, XGB = 3, NN = 1),
  # NN_very_heavy = list(SVM = 1, XGB = 1, NN = 3),
  # 
  # Dominant weight combinations (4:1:1 ratios)
  # SVM_dominant = list(SVM = 4, XGB = 1, NN = 1),
  # XGB_dominant = list(SVM = 1, XGB = 4, NN = 1),
  # NN_dominant = list(SVM = 1, XGB = 1, NN = 4),
  # 
  # # Balanced pairs with light third
  # SVM_XGB_light_NN = list(SVM = 2, XGB = 2, NN = 1),
  # SVM_NN_light_XGB = list(SVM = 2, XGB = 1, NN = 2),
  # XGB_NN_light_SVM = list(SVM = 1, XGB = 2, NN = 2)
)

# Generate all combinations of weights from 0 to 1 in 0.1 steps
steps <- seq(0, 1, by = 0.1)
grid <- expand.grid(SVM = steps, XGB = steps, NN = steps)

# Filter combinations that sum to 1
valid_combinations <- subset(grid, abs(SVM + XGB + NN - 1) < 0.1)

# Convert to a named list
ENSEMBLE_WEIGHTS <- apply(valid_combinations, 1, function(row) {
  list(SVM = row["SVM"], XGB = row["XGB"], NN = row["NN"])
})

# Name the list elements for clarity (optional)
names(ENSEMBLE_WEIGHTS) <- paste0("W", seq_along(ENSEMBLE_WEIGHTS))
ENSEMBLE_WEIGHTS[["ALL"]] <- list(SVM = 1, XGB = 1, NN = 1)
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

#' Safe file reading with error handling
#' @param file_path Path to the file
#' @param reader_function Function to use for reading
#' @return Data frame or NULL if error
safe_read_file <- function(file_path, reader_function) {
  tryCatch({
    reader_function(file_path)
  }, error = function(e) {
    warning(sprintf("Failed to read %s: %s", file_path, e$message))
    NULL
  })
}

#' Create directory safely
#' @param dir_path Directory path to create
create_directory_safely <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  }
}

# =============================================================================
# Class Modification Functions
# =============================================================================

#' Modify class labels to group related subtypes
#' @param vector Vector of class labels
#' @return Modified vector with grouped classes
modify_classes <- function(vector) {
  vector[grepl("MDS|TP53", vector)] <- "MDS.r"
  vector[!grepl("MLLT3", vector) & grepl("KMT2A", vector)] <- "other.KMT2A"
  vector
}

# =============================================================================
# Data Processing Functions
# =============================================================================

#' Combine multiple CSV files from a directory into a single data frame
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

#' Clean and parse string data consistently
#' @param input_string String to clean and parse
#' @return Numeric vector
parse_numeric_string <- function(input_string) {
  if (is.null(input_string) || is.na(input_string) || input_string == "") {
    return(numeric(0))
  }
  
  cleaned_string <- input_string %>%
    str_replace_all(",|\\[|\\]|\\{|\\}|\\\n", "") %>%
    str_squish()
  
  if (cleaned_string == "") {
    return(numeric(0))
  }
  
  numeric_values <- as.numeric(unlist(strsplit(cleaned_string, " ")))
  numeric_values[!is.na(numeric_values)]
}

#' Read CSV files and optionally process for One-vs-One classification
#' @param file_path Path to the CSV file
#' @param is_one_vs_one Logical indicating if this is One-vs-One classification
#' @return Data frame with processed data
read_and_process_csv <- function(file_path, is_one_vs_one = FALSE) {
  data_frame <- safe_read_file(file_path, function(f) data.frame(data.table::fread(f)))
  
  if (is.null(data_frame)) {
    stop(sprintf("Failed to read file: %s", file_path))
  }
  
  if (is_one_vs_one) {
    data_frame$class <- paste(data_frame$class_0, data_frame$class_1, sep = "_")
    data_frame$kappa <- abs(data_frame$kappa)
  }
  
  data_frame
}

#' Add descriptive class labels to data frame
#' @param data_frame Input data frame
#' @param label_mapping Label mapping data frame
#' @return Data frame with added class labels
add_class_labels <- function(data_frame, label_mapping) {
  if (!"class" %in% colnames(data_frame)) {
    stop("Data frame must contain 'class' column")
  }
  
  data_frame$class_label <- label_mapping$Label[data_frame$class + 1]
  data_frame
}

#' Process neural network results to clean epoch information
#' @param nn_results Neural network results data frame
#' @return Processed data frame
process_neural_net_results <- function(nn_results) {
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

# =============================================================================
# Hyperparameter Extraction Functions
# =============================================================================

#' Extract the best hyperparameters per outer fold based on mean kappa
#' @param inner_cv_results Data frame with inner cross-validation results
#' @param classification_type Classification type: "standard", "OvR", or "OvO"
#' @return Data frame with best parameters for each outer fold
extract_best_hyperparameters <- function(inner_cv_results, classification_type) {
  # Validate inputs
  required_cols <- c("outer_fold", "params", "kappa", "accuracy")
  missing_cols <- setdiff(required_cols, colnames(inner_cv_results))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
  }
  
  # Choose grouping variables based on classification type
  grouping_vars <- if (classification_type == "standard") {
    c("outer_fold", "params")
  } else {
    c("outer_fold", "class", "params")
  }
  
  # Step 1: Compute mean kappa and accuracy across inner folds for each param set
  best_parameters <- inner_cv_results %>%
    group_by(across(all_of(grouping_vars))) %>%
    summarise(
      mean_kappa = mean(kappa, na.rm = TRUE),
      mean_accuracy = mean(accuracy, na.rm = TRUE),
      across(any_of(c("class_0", "class_1")), first),
      .groups = "drop_last"
    )
  
  # Step 2: For each group, retain the param set with the highest mean_kappa
  best_parameters %>%
    group_by(across(all_of(grouping_vars[-length(grouping_vars)]))) %>%
    filter(mean_kappa == max(mean_kappa, na.rm = TRUE)) %>%
    slice(1) %>%
    ungroup()
}

# =============================================================================
# Probability Matrix Generation Functions
# =============================================================================

#' Ensure all required class columns exist in probability matrix
#' @param prob_matrix Probability matrix
#' @param label_mapping Label mapping data frame
#' @return Matrix with all required columns
ensure_all_class_columns <- function(prob_matrix, label_mapping) {
  required_cols <- make.names(label_mapping$Label)
  missing_cols <- setdiff(required_cols, colnames(prob_matrix))
  
  for (col_name in missing_cols) {
    prob_matrix[[col_name]] <- 0
  }
  
  prob_matrix[, required_cols, drop = FALSE]
}

#' Generate probability data frames for One-vs-Rest classification
#' @param cv_results_df Cross-validation results data frame
#' @param best_params_df Best parameters data frame
#' @param label_mapping Label mapping data frame
#' @return List of probability data frames organized by outer fold
generate_ovr_probability_matrices <- function(cv_results_df, best_params_df, label_mapping) {
  best_params_with_labels <- add_class_labels(best_params_df, label_mapping)
  outer_fold_ids <- unique(cv_results_df$outer_fold)
  
  probability_matrices <- list()
  
  for (outer_fold_id in outer_fold_ids) {
    outer_fold_data <- cv_results_df[cv_results_df$outer_fold == outer_fold_id, ]
    inner_fold_ids <- unique(outer_fold_data$inner_fold)
    
    fold_matrices <- list()
    
    for (inner_fold_id in inner_fold_ids) {
      inner_fold_data <- outer_fold_data[outer_fold_data$inner_fold == inner_fold_id, ]
      class_labels <- unique(inner_fold_data$class_label)
      
      # Skip if no data or invalid predictions
      if (nrow(inner_fold_data) == 0 || 
          is.null(inner_fold_data$preds_prob[1]) || 
          is.na(inner_fold_data$preds_prob[1])) {
        next
      }
      
      num_samples <- length(parse_numeric_string(inner_fold_data$preds_prob[1]))
      if (num_samples == 0) next
      
      probability_matrix <- matrix(NA, nrow = num_samples, ncol = length(class_labels))
      colnames(probability_matrix) <- class_labels
      true_labels_vector <- rep(NA, num_samples)
      
      for (j in seq_along(class_labels)) {
        current_class_label <- class_labels[j]
        best_param_for_class <- best_params_with_labels[
          best_params_with_labels$outer_fold == outer_fold_id & 
          best_params_with_labels$class_label == current_class_label,
        ]$params
        
        if (length(best_param_for_class) == 0) next
        
        best_param_row <- inner_fold_data[
          inner_fold_data$class_label == current_class_label & 
          inner_fold_data$params == best_param_for_class,
        ]
        
        if (nrow(best_param_row) == 0) next
        
        probs <- parse_numeric_string(best_param_row$preds_prob)
        if (length(probs) == num_samples) {
          probability_matrix[, j] <- probs
        }
        
        target_values <- parse_numeric_string(best_param_row$y_val)
        true_labels_vector[target_values == 1] <- current_class_label
      }
      
      if (all(is.na(true_labels_vector))) next
      
      probability_matrix <- t(apply(probability_matrix, 1, function(row) row / sum(row)))
      probability_matrix <- data.frame(probability_matrix)
      probability_matrix <- ensure_all_class_columns(probability_matrix, label_mapping)
      
      # Add true labels
      probability_matrix$y <- make.names(true_labels_vector)
      # Add inner fold name
      probability_matrix$inner_fold <- inner_fold_id
      # Add outer fold name
      probability_matrix$outer_fold <- outer_fold_id
      
      fold_matrices[[as.character(inner_fold_id)]] <- probability_matrix
    }
    
    if (length(fold_matrices) > 0) {
      probability_matrices[[as.character(outer_fold_id)]] <- do.call(rbind, fold_matrices)
      probability_matrices[[as.character(outer_fold_id)]][is.na(probability_matrices[[as.character(outer_fold_id)]])] <- 0
    }
  }
  
  probability_matrices
}

#' Generate probability data frames for standard multiclass classification
#' @param cv_results_df Cross-validation results data frame
#' @param best_params_df Best parameters data frame
#' @param label_mapping Label mapping data frame
#' @param filtered_subtypes Filtered leukemia subtypes
#' @return List of probability data frames organized by outer fold

generate_standard_probability_matrices <- function(cv_results_df, best_params_df, label_mapping, filtered_subtypes) {
  outer_fold_ids <- unique(cv_results_df$outer_fold)
  probability_matrices <- list()
  
  for (outer_fold_id in outer_fold_ids) {
    outer_fold_data <- cv_results_df[cv_results_df$outer_fold == outer_fold_id, ]
    inner_fold_ids <- unique(outer_fold_data$inner_fold)
    
    fold_matrices <- list()
    
    for (inner_fold_id in inner_fold_ids) {
      best_param <- best_params_df[best_params_df$outer_fold == outer_fold_id, ]$params
      
      
      inner_fold_data <- outer_fold_data[
        outer_fold_data$inner_fold == inner_fold_id & 
        outer_fold_data$params == best_param, 
      ]
      
      
      class_indices <- unique(parse_numeric_string(inner_fold_data$classes))
      class_labels <- label_mapping$Label[class_indices + 1]
      
      num_samples <- length(parse_numeric_string(inner_fold_data$sample_indices))
      
      probs <- parse_numeric_string(inner_fold_data$preds_prob)
      
      probability_matrix <- t(matrix(probs, ncol = num_samples, nrow = length(class_labels)))
      colnames(probability_matrix) <- make.names(class_labels)
      
      probability_matrix <- data.frame(probability_matrix)
      probability_matrix <- ensure_all_class_columns(probability_matrix, label_mapping)
      
      probability_matrix <- t(apply(probability_matrix, 1, function(row) row / sum(row)))
      probability_matrix <- data.frame(probability_matrix)
      
      sample_indices <- parse_numeric_string(inner_fold_data$sample_indices)
  
      probability_matrix$y <- make.names(filtered_subtypes[sample_indices + 1])

      probability_matrix <- data.frame(probability_matrix)
      probability_matrix$inner_fold <- inner_fold_id
      probability_matrix$outer_fold <- outer_fold_id
      
      fold_matrices[[as.character(inner_fold_id)]] <- probability_matrix
    }
    
    probability_matrices[[as.character(outer_fold_id)]] <- do.call(rbind, fold_matrices)
  }
  
  probability_matrices
}

# =============================================================================
# Improved Ensemble Analysis Functions
# =============================================================================

#' Perform One-vs-Rest ensemble analysis for each class separately
#' @param results Analysis results containing probability matrices
#' @param weights Weight configurations for ensemble
#' @param type Type of analysis ("cv" or "loso")
#' @return List of performance metrics for each fold, weight configuration, and class
perform_ovr_ensemble_analysis <- function(results, weights, type = "cv") {
  cat("Performing One-vs-Rest ensemble analysis...\n")
  
  folds <- names(results$probability_matrices$svm[[type]])
  df_list <- list()
  
  for (fold in folds) {
    cat(sprintf("  Processing fold %s...\n", fold))
    df_list[[fold]] <- list()
    
    # Align probability matrices for this fold
    aligned_matrices <- align_probability_matrices(results$probability_matrices, fold, type)
    if (is.null(aligned_matrices)) {
      cat(sprintf("    Skipping fold %s - unable to align matrices\n", fold))
      next
    }
    
    for (i in seq_along(weights)) {
      weight_i <- names(weights)[i]
      
      # Extract aligned probability data frames
      prob_df_SVM <- aligned_matrices$svm
      prob_df_XGB <- aligned_matrices$xgboost
      prob_df_NN <- aligned_matrices$neural_net
      truth <- aligned_matrices$truth
      
      # Get all class names
      all_classes <- colnames(prob_df_SVM)
      
      # Analyze each class separately as a binary classification problem
      for (class_name in all_classes) {
        # Calculate weighted ensemble probabilities for this class only
        class_probs <- prob_df_SVM[[class_name]] * weights[[weight_i]]$SVM +
                      prob_df_XGB[[class_name]] * weights[[weight_i]]$XGB +
                      prob_df_NN[[class_name]] * weights[[weight_i]]$NN
        
        # Create binary predictions: class vs not class
        # Threshold at 0.5 for binary classification
        binary_preds <- ifelse(class_probs > 0.5, "Class", "Not_Class")
        
        # Create binary truth: class vs not class
        binary_truth <- ifelse(truth == class_name, "Class", "Not_Class")
        
        # Ensure factor levels
        binary_truth <- factor(binary_truth, levels = c("Not_Class", "Class"))
        binary_preds <- factor(binary_preds, levels = c("Not_Class", "Class"))
        
        # Calculate binary confusion matrix and metrics
        cm <- caret::confusionMatrix(binary_preds, binary_truth, positive = "Class")
        
        # Extract binary performance metrics
        sensitivity <- cm$byClass["Sensitivity"]
        specificity <- cm$byClass["Specificity"]
        balanced_accuracy <- cm$byClass["Balanced Accuracy"]
        f1_score <- cm$byClass["F1"]
        
        # Create results data frame for this class
        df <- data.frame(
          weights = weight_i,
          type = type,
          class = gsub("Class.", "", class_name),
          sensitivity = sensitivity,
          specificity = specificity,
          balanced_accuracy = balanced_accuracy,
          f1_score = f1_score,
          stringsAsFactors = FALSE
        )
        
        df_list[[fold]][[paste(weight_i, class_name, sep = "_")]] <- df
      }
    }
    
    # Combine results for this fold
    df_list[[fold]] <- do.call(rbind, df_list[[fold]])
  }
  
  df_list
}

#' Generate One-vs-Rest optimized ensemble probability matrices
#' @param results Analysis results containing probability matrices
#' @param weights Weight configurations for ensemble
#' @param type Type of analysis ("cv" or "loso")
#' @return List containing optimized probability matrices and weights used for each fold
generate_ovr_optimized_ensemble_matrices <- function(results, weights, type = "cv", ensemble_performance) {
  cat("Generating One-vs-Rest optimized ensemble probability matrices...\n")
  
  folds <- names(results$probability_matrices$svm[[type]])
  optimized_matrices <- list()
  weights_used <- list()  # Store weights used for each fold
  
  for (fold in folds) {
    cat(sprintf("  Creating OvR optimized matrix for fold %s...\n", fold))
    
    # Align probability matrices for this fold
    aligned_matrices <- align_probability_matrices(results$probability_matrices, fold, type)
    if (is.null(aligned_matrices)) {
      cat(sprintf("    Skipping fold %s - unable to align matrices\n", fold))
      next
    }
    
    # Get best weight configuration for each class in this fold
    fold_performance <- ensemble_performance[[fold]]
    if (nrow(fold_performance) == 0) {
      cat(sprintf("    Skipping fold %s - no performance data available\n", fold))
      next
    }
    
    # Extract aligned probability data frames
    prob_df_SVM <- aligned_matrices$svm
    prob_df_XGB <- aligned_matrices$xgboost
    prob_df_NN <- aligned_matrices$neural_net
    truth <- aligned_matrices$truth
    
    # Get all class names
    all_classes <- colnames(prob_df_SVM)
    
    # Get classes that actually have performance data (i.e., were present in this fold)
    available_classes <- unique(fold_performance$class)
    
    # Initialize optimized probability matrix
    optimized_matrix <- matrix(0, nrow = nrow(prob_df_SVM), ncol = length(all_classes))
    colnames(optimized_matrix) <- all_classes
    
    # Store weights used for each class in this fold
    fold_weights_used <- list()
    
    # For each class, use the best ensemble weights based on F1 score
    for (class_name in all_classes) {
      # Clean class name for matching - try multiple variations
      clean_class_name <- gsub("Class.", "", class_name)
      clean_class_name_no_dots <- gsub("\\.", "", clean_class_name)
      
      # Check if this class has performance data (was present in this fold)
      class_has_data <- FALSE
      class_performance <- data.frame()
      
      # Try to find performance data for this class
      if (clean_class_name %in% available_classes) {
        class_performance <- fold_performance[fold_performance$class == clean_class_name, ]
        class_has_data <- TRUE
      } else if (clean_class_name_no_dots %in% available_classes) {
        class_performance <- fold_performance[fold_performance$class == clean_class_name_no_dots, ]
        class_has_data <- TRUE
      } else {
        # Try partial matching
        matching_classes <- available_classes[
          grepl(clean_class_name, available_classes, ignore.case = TRUE) |
          grepl(clean_class_name_no_dots, available_classes, ignore.case = TRUE)
        ]
        if (length(matching_classes) > 0) {
          class_performance <- fold_performance[fold_performance$class == matching_classes[1], ]
          class_has_data <- TRUE
        }
      }
      
      if (class_has_data && nrow(class_performance) > 0) {
        # Use the best weights found for this class
        best_weight_indices <- which.max(class_performance$f1_score)
        best_weight_name <- class_performance$weights[best_weight_indices]
        
        # Ensure we have a single weight name (take first if multiple)
        if (length(best_weight_name) > 1) {
          best_weight_name <- best_weight_name[1]
          cat(sprintf("    Warning: Multiple best weights found for class %s, using first one\n", clean_class_name))
        }
        
        # Debug: Check if best_weight_name is valid
        if (is.null(best_weight_name) || is.na(best_weight_name) || length(best_weight_name) == 0 || best_weight_name == "") {
          cat(sprintf("    Warning: Invalid weight name for class %s, using default weights\n", clean_class_name))
          best_weights <- weights[["ALL"]]
          best_weight_name <- "ALL"
        } else if (!best_weight_name %in% names(weights)) {
          cat(sprintf("    Warning: Weight '%s' not found in weights list for class %s, using default weights\n", best_weight_name, clean_class_name))
          best_weights <- weights[["ALL"]]
          best_weight_name <- "ALL"
        } else {
          best_weights <- weights[[best_weight_name]]
          cat(sprintf("    Using optimized weights (%s) for class %s\n", best_weight_name, clean_class_name))
        }
        
        # Store the weight configuration used for this class
        fold_weights_used[[clean_class_name]] <- list(
          weight_name = best_weight_name,
          weights = best_weights,
          f1_score = max(class_performance$f1_score)
        )
        
        # Calculate weighted ensemble probabilities for this class
        class_probs <- prob_df_SVM[[class_name]] * best_weights$SVM +
                      prob_df_XGB[[class_name]] * best_weights$XGB +
                      prob_df_NN[[class_name]] * best_weights$NN
        
        optimized_matrix[, class_name] <- class_probs
      } else {
        # This class wasn't present in the performance data, so use default weights
        cat(sprintf("    Using default weights for class %s (not present in this fold)\n", clean_class_name))
        best_weights <- weights[["ALL"]]
        best_weight_name <- "ALL"
        
        # Store the default weight configuration used for this class
        fold_weights_used[[clean_class_name]] <- list(
          weight_name = best_weight_name,
          weights = best_weights,
          f1_score = NA
        )
        
        # Calculate weighted ensemble probabilities for this class
        class_probs <- prob_df_SVM[[class_name]] * best_weights$SVM +
                      prob_df_XGB[[class_name]] * best_weights$XGB +
                      prob_df_NN[[class_name]] * best_weights$NN
        
        optimized_matrix[, class_name] <- class_probs
      }
    }
    
    # Convert to data frame and add true labels
    optimized_matrix <- data.frame(optimized_matrix)
    optimized_matrix <- t(apply(optimized_matrix, 1, function(row) row / sum(row)))
    optimized_matrix <- data.frame(optimized_matrix)
    optimized_matrix$y <- truth
    
    optimized_matrices[[fold]] <- optimized_matrix
    weights_used[[fold]] <- fold_weights_used
  }
  
  # Return both matrices and weights used
  list(
    matrices = optimized_matrices,
    weights_used = weights_used
  )
}

#' Calculate multiclass performance from One-vs-Rest ensemble matrices
#' @param ovr_ensemble_result Result from generate_ovr_optimized_ensemble_matrices containing matrices and weights
#' @param type Type of analysis ("cv" or "loso")
#' @return Data frame with multiclass performance metrics for each fold
analyze_ovr_ensemble_multiclass_performance <- function(ovr_ensemble_result, type = "cv") {
  cat("Analyzing One-vs-Rest ensemble multiclass performance...\n")
  
  # Extract optimized matrices from the result structure
  optimized_matrices <- ovr_ensemble_result$matrices
  
  performance_results <- list()
  
  for (fold_name in names(optimized_matrices)) {
    cat(sprintf("  Analyzing fold %s...\n", fold_name))
    
    # Get the optimized matrix for this fold
    optimized_matrix <- optimized_matrices[[fold_name]]
    
    # Extract true labels and remove from probability matrix
    truth <- optimized_matrix$y
    prob_matrix <- optimized_matrix[, !colnames(optimized_matrix) %in% c("y", "inner_fold", "outer_fold"), drop = FALSE]
    
    # Get predictions (class with highest probability)
    preds <- colnames(prob_matrix)[apply(prob_matrix, 1, which.max)]
    
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
    
    performance_results[[fold_name]] <- cm
  }
  
  # Combine all fold results
  performance_results
}

#' Perform global ensemble optimization using overall kappa
#' @param results Analysis results containing probability matrices
#' @param weights Weight configurations for ensemble
#' @param type Type of analysis ("cv" or "loso")
#' @return List of performance metrics for each fold and weight configuration
perform_global_ensemble_analysis <- function(results, weights, type = "cv") {
  cat("Performing global ensemble analysis...\n")
  
  folds <- names(results$probability_matrices$svm[[type]])
  df_list <- list()
  
  for (fold in folds) {
    cat(sprintf("  Processing fold %s...\n", fold))
    df_list[[fold]] <- list()
    
    # Align probability matrices for this fold
    aligned_matrices <- align_probability_matrices(results$probability_matrices, fold, type)
    if (is.null(aligned_matrices)) {
      cat(sprintf("    Skipping fold %s - unable to align matrices\n", fold))
      next
    }
    
    for (i in seq_along(weights)) {
      weight_i <- names(weights)[i]
      
      # Extract aligned probability data frames
      prob_df_SVM <- aligned_matrices$svm
      prob_df_XGB <- aligned_matrices$xgboost
      prob_df_NN <- aligned_matrices$neural_net
      truth <- aligned_matrices$truth
      
      # Calculate weighted ensemble probabilities
      prob_df <- prob_df_SVM * weights[[weight_i]]$SVM +
                 prob_df_XGB * weights[[weight_i]]$XGB +
                 prob_df_NN * weights[[weight_i]]$NN
      
      # Normalize probabilities to sum to 1 for each sample
      prob_df <- prob_df / rowSums(prob_df)
      
      # Get predictions
      preds <- colnames(prob_df)[apply(prob_df, 1, which.max)]
      
      # Clean class labels
      truth <- make.names(gsub("Class. ", "", truth))
      preds <- make.names(gsub("Class. ", "", preds))
      
      # Ensure all classes are represented
      all_classes <- unique(c(truth, preds))
      truth <- factor(truth, levels = all_classes)
      preds <- factor(preds, levels = all_classes)
      
      # Calculate confusion matrix and metrics
      cm <- caret::confusionMatrix(preds, truth)
      
      # Extract overall performance metrics
      overall_kappa <- cm$overall["Kappa"]
      overall_accuracy <- cm$overall["Accuracy"]
      
      # Create results data frame
      df <- data.frame(
        weights = weight_i,
        type = type,
        kappa = overall_kappa,
        accuracy = overall_accuracy
      )
      
      df_list[[fold]][[weight_i]] <- df
    }
    
    # Combine results for this fold
    df_list[[fold]] <- do.call(rbind, df_list[[fold]])
  }
  
  df_list
}

#' Generate globally optimized ensemble probability matrices
#' @param results Analysis results containing probability matrices
#' @param weights Weight configurations for ensemble
#' @param type Type of analysis ("cv" or "loso")
#' @return List containing optimized probability matrices and weights used for each fold
generate_global_optimized_ensemble_matrices <- function(results, weights, type = "cv", ensemble_performance) {
  cat("Generating globally optimized ensemble probability matrices...\n")
  
  folds <- names(results$probability_matrices$svm[[type]])
  optimized_matrices <- list()
  weights_used <- list()  # Store weights used for each fold
  
  for (fold in folds) {
    cat(sprintf("  Creating globally optimized matrix for fold %s...\n", fold))
    
    # Align probability matrices for this fold
    aligned_matrices <- align_probability_matrices(results$probability_matrices, fold, type)
    if (is.null(aligned_matrices)) {
      cat(sprintf("    Skipping fold %s - unable to align matrices\n", fold))
      next
    }
    
    # Get best weight configuration for this fold (highest kappa)
    fold_performance <- ensemble_performance[[fold]]
    if (nrow(fold_performance) == 0) {
      cat(sprintf("    Skipping fold %s - no performance data available\n", fold))
      next
    }
    
    best_weight_name <- fold_performance$weights[which.max(fold_performance$kappa)]
    best_weights <- weights[[best_weight_name]]
    best_kappa <- max(fold_performance$kappa)
    
    # Store the weight configuration used for this fold
    weights_used[[fold]] <- list(
      weight_name = best_weight_name,
      weights = best_weights,
      kappa = best_kappa
    )
    
    cat(sprintf("    Using globally optimized weights (%s) for fold %s (kappa = %.4f)\n", 
                best_weight_name, fold, best_kappa))
    
    # Extract aligned probability data frames
    prob_df_SVM <- aligned_matrices$svm
    prob_df_XGB <- aligned_matrices$xgboost
    prob_df_NN <- aligned_matrices$neural_net
    truth <- aligned_matrices$truth
    
    # Calculate weighted ensemble probabilities using best global weights
    optimized_matrix <- prob_df_SVM * best_weights$SVM +
                       prob_df_XGB * best_weights$XGB +
                       prob_df_NN * best_weights$NN
    
    # Normalize probabilities to sum to 1 for each sample
    optimized_matrix <- optimized_matrix / rowSums(optimized_matrix)
    
    # Convert to data frame and add true labels
    optimized_matrix <- data.frame(optimized_matrix)
    optimized_matrix$y <- truth
    
    optimized_matrices[[fold]] <- optimized_matrix
  }
  
  # Return both matrices and weights used
  list(
    matrices = optimized_matrices,
    weights_used = weights_used
  )
}

# =============================================================================
# Matrix Alignment Functions
# =============================================================================

#' Align probability matrices from different models for ensemble analysis
#' @param prob_matrices List of probability matrices from different models
#' @param fold_name Name of the fold being processed
#' @param type Type of analysis ("cv" or "loso")
#' @return List of aligned probability matrices
align_probability_matrices <- function(prob_matrices, fold_name, type) {
  # Extract matrices for this fold
  svm_matrix <- prob_matrices$svm[[type]][[fold_name]]
  xgb_matrix <- prob_matrices$xgboost[[type]][[fold_name]]
  nn_matrix <- prob_matrices$neural_net[[type]][[fold_name]]
  
  # Check if all matrices exist
  if (is.null(svm_matrix) || is.null(xgb_matrix) || is.null(nn_matrix)) {
    warning(sprintf("Missing probability matrix for fold %s in %s analysis", fold_name, type))
    return(NULL)
  }
  
  # Extract true labels
  truth_svm <- make.names(svm_matrix$y)
  truth_xgb <- make.names(xgb_matrix$y)
  truth_nn <- make.names(nn_matrix$y)
  
  # Remove non-probability columns from matrices (y, inner_fold, outer_fold)
  # Use proper column selection instead of setting to NULL
  svm_matrix <- svm_matrix[, !colnames(svm_matrix) %in% c("y", "inner_fold", "outer_fold"), drop = FALSE]
  xgb_matrix <- xgb_matrix[, !colnames(xgb_matrix) %in% c("y", "inner_fold", "outer_fold"), drop = FALSE]
  nn_matrix <- nn_matrix[, !colnames(nn_matrix) %in% c("y", "inner_fold", "outer_fold"), drop = FALSE]
  
  # Get all unique class names across all models
  all_classes <- unique(c(
    colnames(svm_matrix),
    colnames(xgb_matrix),
    colnames(nn_matrix)
  ))
  
  # Get the minimum number of samples across all models
  min_samples <- min(nrow(svm_matrix), nrow(xgb_matrix), nrow(nn_matrix))
  
  # Align matrices by ensuring they have the same columns and sample size
  aligned_matrices <- list()
  
  for (model_name in c("svm", "xgboost", "neural_net")) {
    matrix_data <- switch(model_name,
      "svm" = svm_matrix,
      "xgboost" = xgb_matrix,
      "neural_net" = nn_matrix
    )
    
    # Ensure all required columns exist (add 0s for missing classes)
    missing_cols <- setdiff(all_classes, colnames(matrix_data))
    for (col in missing_cols) {
      matrix_data[[col]] <- 0
    }
    
    # Reorder columns to match all_classes
    matrix_data <- matrix_data[, all_classes, drop = FALSE]
    
    # Truncate to minimum sample size if necessary
    if (nrow(matrix_data) > min_samples) {
      matrix_data <- matrix_data[1:min_samples, , drop = FALSE]
    }
    
    aligned_matrices[[model_name]] <- matrix_data
  }
  
  # Use the truth from SVM as reference (or the one with minimum samples)
  reference_truth <- if (length(truth_svm) >= min_samples) {
    truth_svm[1:min_samples]
  } else if (length(truth_xgb) >= min_samples) {
    truth_xgb[1:min_samples]
  } else {
    truth_nn[1:min_samples]
  }
  
  # Add aligned truth to the result
  aligned_matrices$truth <- reference_truth
  
  aligned_matrices
}

# =============================================================================
# Ensemble Analysis Functions
# =============================================================================

#' Perform ensemble analysis with different weight configurations
#' @param results Analysis results containing probability matrices
#' @param weights Weight configurations for ensemble
#' @param type Type of analysis ("cv" or "loso")
#' @return List of performance metrics for each fold and weight configuration
perform_ensemble_analysis <- function(results, weights, type = "cv") {
  cat("Performing ensemble analysis...\n")
  
  folds <- names(results$probability_matrices$svm[[type]])
  df_list <- list()
  
  for (fold in folds) {
    cat(sprintf("  Processing fold %s...\n", fold))
    df_list[[fold]] <- list()
    
    # Align probability matrices for this fold
    aligned_matrices <- align_probability_matrices(results$probability_matrices, fold, type)
    if (is.null(aligned_matrices)) {
      cat(sprintf("    Skipping fold %s - unable to align matrices\n", fold))
      next
    }
    
    for (i in seq_along(weights)) {
      weight_i <- names(weights)[i]
      
      # Extract aligned probability data frames
      prob_df_SVM <- aligned_matrices$svm
      prob_df_XGB <- aligned_matrices$xgboost
      prob_df_NN <- aligned_matrices$neural_net
      truth <- aligned_matrices$truth
      
      # Calculate weighted ensemble probabilities
      prob_df <- prob_df_SVM * weights[[weight_i]]$SVM +
                 prob_df_XGB * weights[[weight_i]]$XGB +
                 prob_df_NN * weights[[weight_i]]$NN
      
      # Get predictions
      preds <- colnames(prob_df)[apply(prob_df, 1, which.max)]
      
      # Clean class labels
      truth <- make.names(gsub("Class. ", "", truth))
      preds <- make.names(gsub("Class. ", "", preds))
      
      # Ensure all classes are represented
      all_classes <- unique(c(truth, preds))
      truth <- factor(truth, levels = all_classes)
      preds <- factor(preds, levels = all_classes)
      
      # Calculate confusion matrix and metrics
      cm <- caret::confusionMatrix(preds, truth)
      
      # Extract performance metrics
      balanced_accuracy <- t(data.frame(cm$byClass)["Balanced.Accuracy"])
      F1 <- t(data.frame(cm$byClass)["F1"])
      
      # Create results data frame
      df <- t(rbind(balanced_accuracy, F1))
      df <- data.frame(df)
      df$weights <- weight_i
      df$type <- type
      df$class <- make.names(gsub("Class. ", "", rownames(df)))
      df_list[[fold]][[weight_i]] <- df
    }
    
    # Combine results for this fold
    df_list[[fold]] <- do.call(rbind, df_list[[fold]])
  }
  
  # Select best performing weight configuration per class per fold
  df_list <- lapply(df_list, function(x) {
    if (nrow(x) > 0) {
      x %>% group_by(class) %>% slice_max(F1, with_ties = FALSE)
    } else {
      x
    }
  })
  
  df_list
}

#' Generate optimized ensemble probability matrices using best predictors per class
#' @param results Analysis results containing probability matrices
#' @param weights Weight configurations for ensemble
#' @param type Type of analysis ("cv" or "loso")
#' @return List of optimized probability matrices for each fold
generate_optimized_ensemble_matrices <- function(results, weights, type = "cv") {
  cat("Generating optimized ensemble probability matrices...\n")
  
  # First, perform ensemble analysis to get best weights per class
  ensemble_performance <- perform_ensemble_analysis(results, weights, type)
  
  folds <- names(results$probability_matrices$svm[[type]])
  optimized_matrices <- list()
  
  for (fold in folds) {
    cat(sprintf("  Creating optimized matrix for fold %s...\n", fold))
    
    # Align probability matrices for this fold
    aligned_matrices <- align_probability_matrices(results$probability_matrices, fold, type)
    if (is.null(aligned_matrices)) {
      cat(sprintf("    Skipping fold %s - unable to align matrices\n", fold))
      next
    }
    
    # Get best weight configuration for each class in this fold
    fold_performance <- ensemble_performance[[fold]]
    if (nrow(fold_performance) == 0) {
      cat(sprintf("    Skipping fold %s - no performance data available\n", fold))
      next
    }
    
    # Create a more robust lookup for best weights per class
    best_weights_per_class <- setNames(
      fold_performance$weights, 
      fold_performance$class
    )
    
    # Extract aligned probability data frames
    prob_df_SVM <- aligned_matrices$svm
    prob_df_XGB <- aligned_matrices$xgboost
    prob_df_NN <- aligned_matrices$neural_net
    truth <- aligned_matrices$truth
    
    # Get all class names from the probability matrices
    all_classes <- colnames(prob_df_SVM)
    
    # Get classes that actually have performance data (i.e., were present in this fold)
    available_classes <- names(best_weights_per_class)
    
    # Initialize optimized probability matrix
    optimized_matrix <- matrix(0, nrow = nrow(prob_df_SVM), ncol = length(all_classes))
    colnames(optimized_matrix) <- all_classes
    
    # For each class, use the best ensemble weights
    for (class_name in all_classes) {
      # Clean class name for matching
      clean_class_name <- gsub("Class.", "", class_name)
      clean_class_name_no_dots <- gsub("\\.", "", clean_class_name)
      
      # Check if this class has performance data (was present in this fold)
      class_has_data <- FALSE
      best_weight_name <- NULL
      
      # Try to find the best weight configuration for this class
      if (clean_class_name %in% available_classes) {
        best_weight_name <- best_weights_per_class[[clean_class_name]]
        class_has_data <- TRUE
      } else if (clean_class_name_no_dots %in% available_classes) {
        best_weight_name <- best_weights_per_class[[clean_class_name_no_dots]]
        class_has_data <- TRUE
      } else {
        # Try partial matching
        matching_classes <- available_classes[
          grepl(clean_class_name, available_classes, ignore.case = TRUE) |
          grepl(clean_class_name_no_dots, available_classes, ignore.case = TRUE)
        ]
        if (length(matching_classes) > 0) {
          best_weight_name <- best_weights_per_class[[matching_classes[1]]]
          class_has_data <- TRUE
        }
      }
      
      if (class_has_data) {
        # Debug: Check if best_weight_name is valid
        if (is.null(best_weight_name) || is.na(best_weight_name) || length(best_weight_name) == 0 || best_weight_name == "") {
          cat(sprintf("    Warning: Invalid weight name for class %s, using default weights\n", clean_class_name))
          best_weights <- weights[["ALL"]]
        } else if (!best_weight_name %in% names(weights)) {
          cat(sprintf("    Warning: Weight '%s' not found in weights list for class %s, using default weights\n", best_weight_name, clean_class_name))
          best_weights <- weights[["ALL"]]
        } else {
          best_weights <- weights[[best_weight_name]]
          cat(sprintf("    Using optimized weights (%s) for class %s\n", best_weight_name, clean_class_name))
        }
      } else {
        # This class wasn't present in the performance data, so use default weights
        cat(sprintf("    Using default weights for class %s (not present in this fold)\n", clean_class_name))
        best_weights <- weights[["ALL"]]
      }
      
      # Calculate weighted ensemble probabilities for this class
      class_probs <- prob_df_SVM[[class_name]] * best_weights$SVM +
                    prob_df_XGB[[class_name]] * best_weights$XGB +
                    prob_df_NN[[class_name]] * best_weights$NN
      
      optimized_matrix[, class_name] <- class_probs
    }
    
    # Convert to data frame and add true labels
    optimized_matrix <- data.frame(optimized_matrix)
    optimized_matrix$y <- truth
    
    optimized_matrices[[fold]] <- optimized_matrix
  }
  
  optimized_matrices
}

#' Calculate performance metrics for optimized ensemble matrices
#' @param ensemble_result Result containing optimized ensemble probability matrices and weights (for global optimization) or just matrices (for other methods)
#' @param type Type of analysis ("cv" or "loso")
#' @return Data frame with performance metrics for each fold
analyze_optimized_ensemble_performance <- function(ensemble_result, type = "cv") {
  cat("Analyzing optimized ensemble performance...\n")
  
  # Handle different input structures - if it's the new structure with matrices and weights, extract matrices
  if (is.list(ensemble_result) && "matrices" %in% names(ensemble_result)) {
    optimized_matrices <- ensemble_result$matrices
  } else {
    # For backward compatibility with old structure
    optimized_matrices <- ensemble_result
  }
  
  performance_results <- list()
  
  for (fold_name in names(optimized_matrices)) {
    cat(sprintf("  Analyzing fold %s...\n", fold_name))
    
    # Get the optimized matrix for this fold
    optimized_matrix <- optimized_matrices[[fold_name]]
    
    # Extract true labels and remove from probability matrix
    truth <- optimized_matrix$y
    prob_matrix <- optimized_matrix[, !colnames(optimized_matrix) %in% c("y", "inner_fold", "outer_fold"), drop = FALSE]
    
    # Get predictions
    preds <- colnames(prob_matrix)[apply(prob_matrix, 1, which.max)]
    
    # Clean class labels
    truth <- gsub("Class. ", "", truth)
    preds <- gsub("Class. ", "", preds)
    truth <- modify_classes(truth)
    preds <- modify_classes(preds)
    # Ensure all classes are represented
    all_classes <- unique(c(truth, preds))
    truth <- factor(truth, levels = all_classes)
    preds <- factor(preds, levels = all_classes)
    
    # Calculate confusion matrix and metrics
    cm <- caret::confusionMatrix(preds, truth)
    
    performance_results[[fold_name]] <- cm
  }
  
  # Combine all fold results
  performance_results
}

# =============================================================================
# Performance Comparison Functions
# =============================================================================

#' Calculate and display mean kappa across folds for all ensemble methods
#' @param results Analysis results containing all ensemble performance metrics
#' @param type Type of analysis ("cv" or "loso")
#' @return Data frame with mean kappa for each ensemble method
compare_ensemble_performance <- function(results, type = "cv") {
  cat("Comparing ensemble performance across folds...\n")
  
  folds <- names(results$probability_matrices$svm[[type]])
  performance_summary <- list()
  
  # Individual model performance
  cat("\n=== Individual Model Performance ===\n")
  for (model_name in c("svm", "xgboost", "neural_net")) {
    model_kappas <- numeric(length(folds))
    
    for (i in seq_along(folds)) {
      fold <- folds[i]
      optimized_matrix <- results$probability_matrices[[model_name]][[type]][[fold]]
      
      # Extract true labels and remove from probability matrix
      truth <- make.names(optimized_matrix$y)
      prob_matrix <- optimized_matrix[, !colnames(optimized_matrix) %in% c("y", "inner_fold", "outer_fold"), drop = FALSE]
      
      # Get predictions
      preds <- colnames(prob_matrix)[apply(prob_matrix, 1, which.max)]
      
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
      model_kappas[i] <- cm$overall["Kappa"]
    }
    
    mean_kappa <- mean(model_kappas, na.rm = TRUE)
    sd_kappa <- sd(model_kappas, na.rm = TRUE)
    cat(sprintf("%s: Mean Kappa = %.4f ± %.4f\n", toupper(model_name), mean_kappa, sd_kappa))
    
    performance_summary[[toupper(model_name)]] <- list(
      mean_kappa = mean_kappa,
      sd_kappa = sd_kappa,
      fold_kappas = model_kappas
    )
  }
  
  # Ensemble method performance
  cat("\n=== Ensemble Method Performance ===\n")
  ensemble_methods <- list(
    "OvR_Ensemble" = results$ovr_ensemble_multiclass_performance,
    # "Per_Class_Optimized" = results$optimized_ensemble_performance,
    "Global_Optimized" = results$global_optimized_ensemble_performance
  )
  
  for (method_name in names(ensemble_methods)) {
    method_performance <- ensemble_methods[[method_name]]
    method_kappas <- numeric(length(folds))
    
    for (i in seq_along(folds)) {
      fold <- folds[i]
      if (fold %in% names(method_performance)) {
        cm <- method_performance[[fold]]
        method_kappas[i] <- cm$overall["Kappa"]
      }
    }
    
    mean_kappa <- mean(method_kappas, na.rm = TRUE)
    sd_kappa <- sd(method_kappas, na.rm = TRUE)
    cat(sprintf("%s: Mean Kappa = %.4f ± %.4f\n", method_name, mean_kappa, sd_kappa))
    
    performance_summary[[method_name]] <- list(
      mean_kappa = mean_kappa,
      sd_kappa = sd_kappa,
      fold_kappas = method_kappas
    )
  }
  
  # Create summary data frame
  summary_df <- data.frame(
    Method = names(performance_summary),
    Mean_Kappa = sapply(performance_summary, function(x) x$mean_kappa),
    SD_Kappa = sapply(performance_summary, function(x) x$sd_kappa),
    stringsAsFactors = FALSE
  )
  
  # Sort by mean kappa (descending)
  summary_df <- summary_df[order(summary_df$Mean_Kappa, decreasing = TRUE), ]
  
  cat("\n=== Performance Summary (Ranked by Mean Kappa) ===\n")
  print(summary_df)
  
  # Detailed fold-by-fold comparison
  cat("\n=== Detailed Fold-by-Fold Comparison ===\n")
  for (fold in folds) {
    cat(sprintf("\nFold %s:\n", fold))
    
    # Individual models
    for (model_name in c("svm", "xgboost", "neural_net")) {
      optimized_matrix <- results$probability_matrices[[model_name]][[type]][[fold]]
      
      # Extract true labels and remove from probability matrix
      truth <- make.names(optimized_matrix$y)
      prob_matrix <- optimized_matrix[, !colnames(optimized_matrix) %in% c("y", "inner_fold", "outer_fold"), drop = FALSE]
      
      # Get predictions
      preds <- colnames(prob_matrix)[apply(prob_matrix, 1, which.max)]
      
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
      
      cat(sprintf("  %s: Kappa = %.4f\n", toupper(model_name), cm$overall["Kappa"]))
    }
    
    # Ensemble methods
    if (fold %in% names(results$ovr_ensemble_multiclass_performance)) {
      cat(sprintf("  OvR_Ensemble: Kappa = %.4f\n", 
                  results$ovr_ensemble_multiclass_performance[[fold]]$overall["Kappa"]))
    }
    if (fold %in% names(results$global_optimized_ensemble_performance)) {
      cat(sprintf("  Global_Optimized: Kappa = %.4f\n", 
                  results$global_optimized_ensemble_performance[[fold]]$overall["Kappa"]))
    }
  }
  
  return(summary_df)
}

# =============================================================================
# Rejection Analysis Functions
# =============================================================================

#' Evaluate nested CV kappa with rejection for a single probability matrix
#' @param prob_matrix Probability matrix with class probabilities and true labels
#' @param fold_name Name of the fold being analyzed
#' @param model_name Name of the model being analyzed
#' @param type Type of analysis ("cv" or "loso")
#' @return Data frame with rejection analysis results
evaluate_single_matrix_with_rejection <- function(prob_matrix, fold_name, model_name, type) {
  # Extract true labels and remove from probability matrix
  truth <- prob_matrix$y
  prob_matrix_clean <- prob_matrix[, !colnames(prob_matrix) %in% c("y", "inner_fold", "outer_fold"), drop = FALSE]
  
  # Clean class labels
  truth <- gsub("Class. ", "", truth)
  truth <- modify_classes(truth)
  
  # Get predictions (class with highest probability)
  pred_indices <- apply(prob_matrix_clean, 1, which.max)
  preds <- colnames(prob_matrix_clean)[pred_indices]
  preds <- gsub("Class. ", "", preds)
  preds <- modify_classes(preds)
  
  # Get max probabilities for each sample
  max_probs <- apply(prob_matrix_clean, 1, max)
  
  # Ensure all classes are represented
  all_classes <- unique(c(truth, preds))
  truth <- factor(truth, levels = all_classes)
  preds <- factor(preds, levels = all_classes)
  
  # Test probability cutoffs 
  prob_cutoffs <- seq(0.00, 1.00, by = 0.01)
  all_results <- data.frame()
  
  for (cutoff in prob_cutoffs) {
    # Identify samples to reject (max probability below cutoff)
    rejected_indices <- which(max_probs < cutoff)
    accepted_indices <- which(max_probs >= cutoff)
    
    if (length(accepted_indices) == 0) {
      # If all samples are rejected, skip this cutoff
      next
    }
    
    # Calculate accuracy for rejected samples (if any)
    rejected_accuracy <- NA
    if (length(rejected_indices) > 0) {
      rejected_truth <- truth[rejected_indices]
      rejected_preds <- preds[rejected_indices]
      rejected_accuracy <- sum(rejected_truth == rejected_preds) / length(rejected_indices)
    }
    # Only proceed if rejected samples have accuracy < 50% (or if no samples are rejected)
      # Use only accepted samples for kappa calculation
      accepted_truth <- truth[accepted_indices]
      accepted_preds <- preds[accepted_indices]
      
      # Calculate kappa for accepted samples
      res <- caret::confusionMatrix(accepted_preds, accepted_truth)
      kappa <- as.numeric(res$overall["Kappa"])
      accuracy <- as.numeric(res$overall["Accuracy"])
      
      # Store results
      all_results <- rbind(
        all_results,
        data.frame(
          model = model_name,
          type = type,
          fold = fold_name,
          prob_cutoff = cutoff,
          kappa = kappa,
          accuracy = accuracy,
          n_accepted = length(accepted_indices),
          n_rejected = length(rejected_indices),
          perc_rejected = length(rejected_indices) / (length(accepted_indices) + length(rejected_indices)),
          rejected_accuracy = rejected_accuracy,
          total_samples = nrow(prob_matrix),
          stringsAsFactors = FALSE
        )
      )
  }
  
  return(all_results)
}

#' Evaluate rejection analysis for all probability matrices
#' @param probability_matrices List of probability matrices for all models
#' @param ensemble_matrices List of ensemble probability matrices
#' @param type Type of analysis ("cv" or "loso")
#' @return Data frame with rejection analysis results for all models and ensembles
evaluate_all_matrices_with_rejection <- function(probability_matrices, ensemble_matrices, type = "cv") {
  cat("Performing rejection analysis for all probability matrices...\n")
  
  all_rejection_results <- data.frame()
  
  # Analyze individual models
  cat("  Analyzing individual models...\n")
  for (model_name in names(probability_matrices)) {
    cat(sprintf("    Processing %s...\n", toupper(model_name)))
    
    if (type %in% names(probability_matrices[[model_name]])) {
      fold_matrices <- probability_matrices[[model_name]][[type]]
      
      for (fold_name in names(fold_matrices)) {
        prob_matrix <- fold_matrices[[fold_name]]
        
        if (!is.null(prob_matrix) && nrow(prob_matrix) > 0) {
          rejection_results <- evaluate_single_matrix_with_rejection(
            prob_matrix, fold_name, model_name, type
          )
          all_rejection_results <- rbind(all_rejection_results, rejection_results)
        }
      }
    }
  }
  
  # Analyze ensemble methods
  cat("  Analyzing ensemble methods...\n")
  ensemble_methods <- list(
    "OvR_Ensemble" = ensemble_matrices$ovr_optimized_ensemble_matrices,
    #"Per_Class_Optimized" = ensemble_matrices$optimized_ensemble_matrices,
    "Global_Optimized" = ensemble_matrices$global_optimized_ensemble_matrices
  )
  
  for (ensemble_name in names(ensemble_methods)) {
    cat(sprintf("    Processing %s...\n", ensemble_name))
    
    ensemble_matrices_fold <- ensemble_methods[[ensemble_name]]
    
    for (fold_name in names(ensemble_matrices_fold)) {
      prob_matrix <- ensemble_matrices_fold[[fold_name]]
      
      if (!is.null(prob_matrix) && nrow(prob_matrix) > 0) {
        rejection_results <- evaluate_single_matrix_with_rejection(
          prob_matrix, fold_name, ensemble_name, type
        )
        all_rejection_results <- rbind(all_rejection_results, rejection_results)
      }
    }
  }
  
  return(all_rejection_results)
}

#' Find optimal probability cutoff for each model/ensemble
#' @param rejection_results Data frame with rejection analysis results
#' @param optimization_metric Metric to optimize ("kappa" or "accuracy")
#' @return Data frame with optimal cutoffs for each model/ensemble
find_optimal_cutoffs <- function(rejection_results, optimization_metric = "kappa") {
  cat("Finding optimal probability cutoffs...\n")
  
  # Group by model and fold, then find the cutoff that maximizes the optimization metric
  optimal_cutoffs <- rejection_results %>%
    group_by(model, fold) %>%
    filter(!is.na(!!sym(optimization_metric))) %>%
    filter( (is.na(rejected_accuracy) | rejected_accuracy < 0.5) & (perc_rejected < 0.10) ) %>%
    slice_max(!!sym(optimization_metric), with_ties = FALSE) %>%
    ungroup()
  
  # Calculate summary statistics across folds for each model
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

#' Generate rejection analysis plots
#' @param rejection_results Data frame with rejection analysis results
#' @param output_dir Directory to save plots
#' @param type Type of analysis ("cv" or "loso")
generate_rejection_plots <- function(rejection_results, output_dir, type = "cv") {
  cat("Generating rejection analysis plots...\n")
  
  # Load plotting libraries
  load_library_quietly("ggplot2")
  load_library_quietly("gridExtra")
  
  # Create output directory
  create_directory_safely(output_dir)
  
  # Plot 1: Kappa vs Probability Cutoff for each model
  p1 <- ggplot(rejection_results, aes(x = prob_cutoff, y = kappa, color = model)) +
    geom_line(alpha = 0.7) +
    facet_wrap(~model, scales = "free_y") +
    labs(title = sprintf("Kappa vs Probability Cutoff (%s)", toupper(type)),
         x = "Probability Cutoff",
         y = "Kappa") +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Plot 2: Percentage Rejected vs Probability Cutoff
  p2 <- ggplot(rejection_results, aes(x = prob_cutoff, y = perc_rejected * 100, color = model)) +
    geom_line(alpha = 0.7) +
    facet_wrap(~model, scales = "free_y") +
    labs(title = sprintf("Percentage Rejected vs Probability Cutoff (%s)", toupper(type)),
         x = "Probability Cutoff",
         y = "Percentage Rejected (%)") +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Plot 3: Kappa vs Percentage Rejected (trade-off analysis)
  p3 <- ggplot(rejection_results, aes(x = perc_rejected * 100, y = kappa, color = model)) +
    geom_point(alpha = 0.6) +
    facet_wrap(~model, scales = "free") +
    labs(title = sprintf("Kappa vs Percentage Rejected Trade-off (%s)", toupper(type)),
         x = "Percentage Rejected (%)",
         y = "Kappa") +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Plot 4: Accuracy vs Probability Cutoff
  p4 <- ggplot(rejection_results, aes(x = prob_cutoff, y = accuracy, color = model)) +
    geom_line(alpha = 0.7) +
    facet_wrap(~model, scales = "free_y") +
    labs(title = sprintf("Accuracy vs Probability Cutoff (%s)", toupper(type)),
         x = "Probability Cutoff",
         y = "Accuracy") +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Save individual plots
  ggsave(file.path(output_dir, sprintf("kappa_vs_cutoff_%s.png", type)), p1, width = 12, height = 8)
  ggsave(file.path(output_dir, sprintf("perc_rejected_vs_cutoff_%s.png", type)), p2, width = 12, height = 8)
  ggsave(file.path(output_dir, sprintf("kappa_vs_perc_rejected_%s.png", type)), p3, width = 12, height = 8)
  ggsave(file.path(output_dir, sprintf("accuracy_vs_cutoff_%s.png", type)), p4, width = 12, height = 8)
  
  # Create combined plot
  combined_plot <- grid.arrange(p1, p2, p3, p4, ncol = 2)
  ggsave(file.path(output_dir, sprintf("rejection_analysis_combined_%s.png", type)), 
         combined_plot, width = 16, height = 12)
  
  cat(sprintf("  Plots saved to: %s\n", output_dir))
}

#' Run complete rejection analysis for both CV and LOSO
#' @param probability_matrices Probability matrices for all models
#' @param ensemble_results Ensemble analysis results
#' @param output_base_dir Base directory for output files
#' @return List of rejection analysis results
run_complete_rejection_analysis <- function(probability_matrices, ensemble_results, output_base_dir) {
  cat("Running complete rejection analysis...\n")
  
  rejection_results <- list()
  
  for (analysis_type in c("cv", "loso")) {
    cat(sprintf("\n=== Running rejection analysis for %s ===\n", toupper(analysis_type)))
    
    # Check if we have data for this analysis type
    if (!analysis_type %in% names(ensemble_results)) {
      cat(sprintf("Skipping %s rejection analysis - missing ensemble results\n", toupper(analysis_type)))
      next
    }
    
    # Extract ensemble matrices for this analysis type
    ensemble_matrices <- list(
      ovr_optimized_ensemble_matrices = ensemble_results[[analysis_type]]$ovr_optimized_ensemble_matrices,
      # optimized_ensemble_matrices = ensemble_results[[analysis_type]]$optimized_ensemble_matrices,
      global_optimized_ensemble_matrices = ensemble_results[[analysis_type]]$global_optimized_ensemble_matrices$matrices
    )
    
    # Perform rejection analysis
    rejection_results[[analysis_type]] <- evaluate_all_matrices_with_rejection(
      probability_matrices, ensemble_matrices, analysis_type
    )
    
    # Find optimal cutoffs
    optimal_results <- find_optimal_cutoffs(rejection_results[[analysis_type]], "kappa")
    
    # Save results
    output_dir <- file.path(output_base_dir, "rejection_analysis", analysis_type)
    create_directory_safely(output_dir)
    
    # Save detailed rejection results
    write.csv(rejection_results[[analysis_type]], 
              file.path(output_dir, "detailed_rejection_results.csv"), 
              row.names = FALSE)
    
    # Save optimal cutoffs
    write.csv(optimal_results$optimal_cutoffs, 
              file.path(output_dir, "optimal_cutoffs.csv"), 
              row.names = FALSE)
    
    # Save summary statistics
    write.csv(optimal_results$summary_stats, 
              file.path(output_dir, "summary_statistics.csv"), 
              row.names = FALSE)
    
    # Generate plots
    generate_rejection_plots(rejection_results[[analysis_type]], output_dir, analysis_type)
    
    # Display summary
    cat(sprintf("\n=== Rejection Analysis Summary for %s ===\n", toupper(analysis_type)))
    print(optimal_results$summary_stats)
  }
  
  return(rejection_results)
}

# =============================================================================
# Analysis Runner Functions
# =============================================================================

#' Run ensemble analysis for both CV and LOSO types
#' @param probability_matrices Probability matrices for all models
#' @param weights Weight configurations for ensemble
#' @return List of results for both CV and LOSO
run_ensemble_analysis_for_both_types <- function(probability_matrices, weights) {
  cat("Running ensemble analysis for both CV and LOSO...\n")
  
  results <- list()
  
  for (analysis_type in c("cv", "loso")) {
    cat(sprintf("\n=== Running %s analysis ===\n", toupper(analysis_type)))
    
    # Check if we have data for this analysis type
    if (!all(sapply(probability_matrices, function(x) analysis_type %in% names(x)))) {
      cat(sprintf("Skipping %s analysis - missing data\n", toupper(analysis_type)))
      next
    }
    
    # # Perform ensemble analysis
    # ensemble_results <- perform_ensemble_analysis(
    #   list(probability_matrices = probability_matrices), 
    #   weights, 
    #   analysis_type
    # )
    # 
    # # Generate optimized ensemble matrices (per-class optimization)
    # optimized_ensemble_matrices <- generate_optimized_ensemble_matrices(
    #   list(probability_matrices = probability_matrices), 
    #   weights, 
    #   analysis_type
    # )
    # 
    # # Analyze optimized ensemble performance (per-class optimization)
    # optimized_ensemble_performance <- analyze_optimized_ensemble_performance(optimized_ensemble_matrices, analysis_type)
    
    # Perform global ensemble analysis
    global_ensemble_results <- perform_global_ensemble_analysis(
      list(probability_matrices = probability_matrices), 
      weights, 
      analysis_type
    )
    
    # Generate globally optimized ensemble matrices
    global_optimized_ensemble_matrices <- generate_global_optimized_ensemble_matrices(
      list(probability_matrices = probability_matrices), 
      weights, 
      analysis_type,
      global_ensemble_results
    )
    
    # Analyze globally optimized ensemble performance
    global_optimized_ensemble_performance <- analyze_optimized_ensemble_performance(global_optimized_ensemble_matrices, analysis_type)
    
    # Perform One-vs-Rest ensemble analysis (properly handles OvR classification)
    ovr_ensemble_results <- perform_ovr_ensemble_analysis(
      list(probability_matrices = probability_matrices), 
      weights, 
      analysis_type
    )
    
    # Generate One-vs-Rest optimized ensemble matrices
    ovr_optimized_result <- generate_ovr_optimized_ensemble_matrices(
      list(probability_matrices = probability_matrices), 
      weights, 
      analysis_type,
      ovr_ensemble_results
    )
    
    # Analyze One-vs-Rest ensemble multiclass performance
    ovr_ensemble_multiclass_performance <- analyze_ovr_ensemble_multiclass_performance(ovr_optimized_result, analysis_type)
    
    # Store results for this analysis type
    results[[analysis_type]] <- list(
      # ensemble_results = ensemble_results,
      # optimized_ensemble_matrices = optimized_ensemble_matrices,
      # optimized_ensemble_performance = optimized_ensemble_performance,
      global_ensemble_results = global_ensemble_results,
      global_optimized_ensemble_matrices = global_optimized_ensemble_matrices,
      global_optimized_ensemble_performance = global_optimized_ensemble_performance,
      global_ensemble_weights_used = global_optimized_ensemble_matrices$weights_used,
      ovr_ensemble_results = ovr_ensemble_results,
      ovr_optimized_ensemble_matrices = ovr_optimized_result$matrices,
      ovr_ensemble_multiclass_performance = ovr_ensemble_multiclass_performance,
      ovr_ensemble_weights_used = ovr_optimized_result$weights_used
    )
  }
  
  results
}

#' Compare ensemble performance for both CV and LOSO
#' @param results Analysis results containing all ensemble performance metrics
#' @return List of performance comparisons for both CV and LOSO
compare_ensemble_performance_for_both_types <- function(results) {
  cat("Comparing ensemble performance for both CV and LOSO...\n")
  
  performance_comparisons <- list()
  
  for (analysis_type in c("cv", "loso")) {
    cat(sprintf("\n=== Performance Comparison for %s ===\n", toupper(analysis_type)))
    
    # Check if we have results for this analysis type
    if (!analysis_type %in% names(results)) {
      cat(sprintf("Skipping %s performance comparison - missing results\n", toupper(analysis_type)))
      next
    }
    
    # Create results list for performance comparison
    comparison_results <- list(
      probability_matrices = results$probability_matrices,
      ovr_ensemble_multiclass_performance = results[[analysis_type]]$ovr_ensemble_multiclass_performance,
      # optimized_ensemble_performance = results[[analysis_type]]$optimized_ensemble_performance,
      global_optimized_ensemble_performance = results[[analysis_type]]$global_optimized_ensemble_performance
    )
    
    # Compare all ensemble methods and display mean kappa across folds
    performance_comparison <- compare_ensemble_performance(comparison_results, analysis_type)
    performance_comparisons[[analysis_type]] <- performance_comparison
  }
  
  performance_comparisons
}

# =============================================================================
# Data Loading and Processing
# =============================================================================

#' Load and process all model data
#' @param model_configs Model configurations
#' @return List of processed model results
load_all_model_data <- function(model_configs) {
  cat("Loading model data...\n")
  
  model_results <- list()
  
  for (model_name in names(model_configs)) {
    config <- model_configs[[model_name]]
    cat(sprintf("Loading %s data...\n", toupper(model_name)))
    
    model_results[[model_name]] <- list()
    
    for (fold_type in names(config$file_paths)) {
      file_path <- config$file_paths[[fold_type]]
      
      if (model_name == "neural_net") {
        # Neural networks use directory of CSV files
        results <- safe_read_file(file_path, combine_csv_files)
        if (!is.null(results)) {
          results <- process_neural_net_results(results)
        }
      } else {
        # SVM and XGBoost use single CSV files
        results <- safe_read_file(file_path, function(f) read_and_process_csv(f, FALSE))
      }
      
      if (!is.null(results)) {
        model_results[[model_name]][[fold_type]] <- results
      } else {
        warning(sprintf("Failed to load %s %s data", model_name, fold_type))
      }
    }
  }
  
  model_results
}

#' Extract best parameters for all models
#' @param model_results Processed model results
#' @param model_configs Model configurations
#' @return List of best parameters for each model
extract_all_best_parameters <- function(model_results, model_configs) {
  cat("Extracting best parameters...\n")
  
  best_parameters <- list()
  
  for (model_name in names(model_results)) {
    config <- model_configs[[model_name]]
    cat(sprintf("Extracting best parameters for %s...\n", toupper(model_name)))
    
    best_parameters[[model_name]] <- list()
    
    for (fold_type in names(model_results[[model_name]])) {
      results <- model_results[[model_name]][[fold_type]]
      if (!is.null(results)) {
        best_params <- extract_best_hyperparameters(results, config$classification_type)
        best_parameters[[model_name]][[fold_type]] <- best_params
      }
    }
  }
  
  best_parameters
}

#' Save best parameters for all models
#' @param best_parameters Best parameters for each model
#' @param model_configs Model configurations
save_all_best_parameters <- function(best_parameters, model_configs) {
  cat("Saving best parameters...\n")
  
  for (model_name in names(best_parameters)) {
    config <- model_configs[[model_name]]
    output_dir <- config$output_dir
    
    cat(sprintf("Saving %s results...\n", toupper(model_name)))
    create_directory_safely(output_dir)
    
    for (fold_type in names(best_parameters[[model_name]])) {
      best_params <- best_parameters[[model_name]][[fold_type]]
      if (!is.null(best_params)) {
        filename <- sprintf("%s_best_param_%s.csv", toupper(model_name), fold_type)
        filepath <- file.path(output_dir, filename)
        write.csv(best_params, file = filepath, row.names = FALSE)
        cat(sprintf("  Saved: %s\n", filepath))
      }
    }
  }
}

# =============================================================================
# Weight Saving Functions
# =============================================================================

#' Save ensemble weights used for each fold and analysis type
#' @param ensemble_results Ensemble analysis results containing weights used
#' @param output_base_dir Base directory for output files
save_ensemble_weights <- function(ensemble_results, output_base_dir) {
  cat("Saving ensemble weights used...\n")
  
  for (analysis_type in names(ensemble_results)) {
    cat(sprintf("Saving weights for %s analysis...\n", toupper(analysis_type)))
    
    # Create output directory for this analysis type
    weights_output_dir <- file.path(output_base_dir, "ensemble_weights", analysis_type)
    create_directory_safely(weights_output_dir)
    
    # Save OvR ensemble weights
    if ("ovr_ensemble_weights_used" %in% names(ensemble_results[[analysis_type]])) {
      ovr_weights <- ensemble_results[[analysis_type]]$ovr_ensemble_weights_used
      
      # Convert to a more structured format for saving
      ovr_weights_df <- data.frame()
      
      for (fold_name in names(ovr_weights)) {
        fold_weights <- ovr_weights[[fold_name]]
        
        for (class_name in names(fold_weights)) {
          class_weight_info <- fold_weights[[class_name]]
          
          ovr_weights_df <- rbind(ovr_weights_df, data.frame(
            fold = fold_name,
            class = class_name,
            weight_name = class_weight_info$weight_name,
            svm_weight = class_weight_info$weights$SVM,
            xgb_weight = class_weight_info$weights$XGB,
            nn_weight = class_weight_info$weights$NN,
            f1_score = class_weight_info$f1_score,
            stringsAsFactors = FALSE
          ))
        }
      }
      
      # Save OvR weights
      ovr_weights_file <- file.path(weights_output_dir, "ovr_ensemble_weights_used.csv")
      write.csv(ovr_weights_df, ovr_weights_file, row.names = FALSE)
      cat(sprintf("  Saved OvR weights: %s\n", ovr_weights_file))
    }
    
    # Save global ensemble weights
    if ("global_ensemble_weights_used" %in% names(ensemble_results[[analysis_type]])) {
      global_weights <- ensemble_results[[analysis_type]]$global_ensemble_weights_used
      
      # Convert to data frame format
      global_weights_df <- data.frame()
      
      for (fold_name in names(global_weights)) {
        fold_weight_info <- global_weights[[fold_name]]
        
        global_weights_df <- rbind(global_weights_df, data.frame(
          fold = fold_name,
          weight_name = fold_weight_info$weight_name,
          svm_weight = fold_weight_info$weights$SVM,
          xgb_weight = fold_weight_info$weights$XGB,
          nn_weight = fold_weight_info$weights$NN,
          kappa = fold_weight_info$kappa,
          stringsAsFactors = FALSE
        ))
      }
      
      # Save global weights
      global_weights_file <- file.path(weights_output_dir, "global_ensemble_weights_used.csv")
      write.csv(global_weights_df, global_weights_file, row.names = FALSE)
      cat(sprintf("  Saved global weights: %s\n", global_weights_file))
    }
  }
}

# =============================================================================
# Main Execution
# =============================================================================

#' Main function to run the complete analysis
main <- function() {
  # Load required libraries
  load_library_quietly("plyr")
  load_library_quietly("dplyr")
  load_library_quietly("stringr")
  load_library_quietly("caret")
  
  # Load label mapping and data
  cat("Loading label mapping and data...\n")
  label_mapping <- safe_read_file("~/Documents/AML_PhD/leukem_ai/label_mapping_df_n10.csv", read.csv)
  if (is.null(label_mapping)) {
    stop("Failed to load label mapping file")
  }
  
  # Load leukemia subtype data
  leukemia_subtypes <- safe_read_file("../data/rgas.csv", function(f) read.csv(f)$ICC_Subtype)
  if (is.null(leukemia_subtypes)) {
    stop("Failed to load leukemia subtype data")
  }
  
  # Load study metadata
  study_names <- safe_read_file("../data/meta.csv", function(f) read.csv(f)$Studies)
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
  
  # Load and process model data
  model_results <- load_all_model_data(MODEL_CONFIGS)
  
  # Extract best parameters
  best_parameters <- extract_all_best_parameters(model_results, MODEL_CONFIGS)
  
  # Save best parameters
  save_all_best_parameters(best_parameters, MODEL_CONFIGS)
  
  # Generate probability matrices
  cat("Generating prediction probability matrices...\n")
  probability_matrices <- list()
  
  # Define output directory for saving results
  output_base_dir <- "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10"
  
  for (model_name in names(model_results)) {
    config <- MODEL_CONFIGS[[model_name]]
    cat(sprintf("Extracting %s probabilities...\n", toupper(model_name)))
    
    probability_matrices[[model_name]] <- list()
    
    for (fold_type in names(model_results[[model_name]])) {
      results <- model_results[[model_name]][[fold_type]]
      best_params <- best_parameters[[model_name]][[fold_type]]
      
      if (!is.null(results) && !is.null(best_params)) {
        if (config$classification_type == "OvR") {
          probs <- generate_ovr_probability_matrices(results, best_params, label_mapping)
        } else {
          probs <- generate_standard_probability_matrices(results, best_params, label_mapping, filtered_leukemia_subtypes)
        }
        probability_matrices[[model_name]][[fold_type]] <- probs
      }
    }
  }
  
  # Run ensemble analysis for both CV and LOSO
  ensemble_results <- run_ensemble_analysis_for_both_types(probability_matrices, ENSEMBLE_WEIGHTS)
  
  # Save ensemble weights used for later analysis
  save_ensemble_weights(ensemble_results, output_base_dir)
  
  # Compare ensemble performance for both CV and LOSO
  performance_comparisons <- compare_ensemble_performance_for_both_types(
    list(probability_matrices = probability_matrices, cv = ensemble_results$cv, loso = ensemble_results$loso)
  )
  
  # Run rejection analysis for both CV and LOSO
  rejection_results <- run_complete_rejection_analysis(
    probability_matrices, ensemble_results, output_base_dir
  )
  
  cat("Analysis complete!\n")
  
  # Return results for potential further use
  list(
    model_results = model_results,
    best_parameters = best_parameters,
    probability_matrices = probability_matrices,
    ensemble_results = ensemble_results,
    performance_comparisons = performance_comparisons,
    rejection_results = rejection_results,
    filtered_subtypes = filtered_leukemia_subtypes
  )
}

# Run the analysis if this script is executed directly
if (!exists("SKIP_MAIN_EXECUTION")) {
  results <- main()
}

#' Load ensemble weights used for outer fold analysis
#' @param weights_base_dir Base directory containing saved weights
#' @param analysis_type Type of analysis ("cv" or "loso")
#' @return List containing OvR and global ensemble weights
load_ensemble_weights <- function(weights_base_dir, analysis_type = "cv") {
  cat(sprintf("Loading ensemble weights for %s analysis...\n", toupper(analysis_type)))
  
  weights_dir <- file.path(weights_base_dir, "ensemble_weights", analysis_type)
  
  if (!dir.exists(weights_dir)) {
    stop(sprintf("Weights directory does not exist: %s", weights_dir))
  }
  
  weights_data <- list()
  
  # Load OvR ensemble weights
  ovr_weights_file <- file.path(weights_dir, "ovr_ensemble_weights_used.csv")
  if (file.exists(ovr_weights_file)) {
    ovr_weights_df <- read.csv(ovr_weights_file, stringsAsFactors = FALSE)
    
    # Convert back to nested list structure
    ovr_weights <- list()
    for (i in 1:nrow(ovr_weights_df)) {
      row <- ovr_weights_df[i, ]
      fold <- row$fold
      class <- row$class
      
      if (!fold %in% names(ovr_weights)) {
        ovr_weights[[fold]] <- list()
      }
      
      ovr_weights[[fold]][[class]] <- list(
        weight_name = row$weight_name,
        weights = list(
          SVM = row$svm_weight,
          XGB = row$xgb_weight,
          NN = row$nn_weight
        ),
        f1_score = row$f1_score
      )
    }
    
    weights_data$ovr_weights <- ovr_weights
    cat(sprintf("  Loaded OvR weights from: %s\n", ovr_weights_file))
  } else {
    warning(sprintf("OvR weights file not found: %s", ovr_weights_file))
  }
  
  # Load global ensemble weights
  global_weights_file <- file.path(weights_dir, "global_ensemble_weights_used.csv")
  if (file.exists(global_weights_file)) {
    global_weights_df <- read.csv(global_weights_file, stringsAsFactors = FALSE)
    
    # Convert to nested list structure
    global_weights <- list()
    for (i in 1:nrow(global_weights_df)) {
      row <- global_weights_df[i, ]
      fold <- row$fold
      
      global_weights[[fold]] <- list(
        weight_name = row$weight_name,
        weights = list(
          SVM = row$svm_weight,
          XGB = row$xgb_weight,
          NN = row$nn_weight
        ),
        kappa = row$kappa
      )
    }
    
    weights_data$global_weights <- global_weights
    cat(sprintf("  Loaded global weights from: %s\n", global_weights_file))
  } else {
    warning(sprintf("Global weights file not found: %s", global_weights_file))
  }
  
  return(weights_data)
}