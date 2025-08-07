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
    stop("Parsed string does not result in a vector with numeric values")
  }
  
  numeric_values <- as.numeric(unlist(strsplit(cleaned_string, " ")))
  
  if (any(is.na(numeric_values))) {
    stop("Parsed string leads to NAs")
  }
  
  return(numeric_values)
}

#' Read CSV files and optionally process for One-vs-One classification
#' @param file_path Path to the CSV file
#' @return Data frame with processed data
read_and_process_csv <- function(file_path) {
  data_frame <- safe_read_file(file_path, function(f) data.frame(data.table::fread(f)))
  
  if (is.null(data_frame)) {
    stop(sprintf("Failed to read file: %s", file_path))
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

#' Load ensemble weights used for outer fold analysis
#' @param weights_base_dir Base directory containing saved weights
#' @param analysis_type Type of analysis ("cv" or "loso")
#' @return List containing OvR and global ensemble weights
load_ensemble_weights <- function(WEIGHTS_BASE_DIR, analysis_type = "cv") {
  cat(sprintf("Loading ensemble weights for %s analysis...\n", toupper(analysis_type)))
  
  weights_dir <- file.path(WEIGHTS_BASE_DIR, analysis_type)
  
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
      fold <- as.character(row$fold)
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
      fold <- as.character(row$fold)
      
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
  max_samples <- max(nrow(svm_matrix), nrow(xgb_matrix), nrow(nn_matrix))
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
    if (nrow(matrix_data) < max_samples) {
      print(sprintf("The prob matrix of %s has less samples than the rest!", model_name))
      #matrix_data <- matrix_data[1:min_samples, , drop = FALSE]
    }
    
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
