# =============================================================================
# Utility Functions for Cross-Validation Analysis
# =============================================================================
# This file contains shared utility functions used by:
# - inner_cv_analysis.R (nested inner cross-validation)
# - outer_cv_analysis.r (outer cross-validation with pre-computed weights)
# - train_test_analysis.R (final train/test split analysis)
# =============================================================================

# =============================================================================
# Library Loading Functions
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

# =============================================================================
# File I/O Functions
# =============================================================================

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
   data.frame(data.table::fread(file, sep = ",", drop = 1))
  })

  # Remove NULL results from failed reads
  combined_results <- combined_results[!sapply(combined_results, is.null)]

  if (length(combined_results) == 0) {
    stop("No files could be read successfully")
  }

  do.call(rbind, combined_results)
}

# =============================================================================
# Data Parsing Functions
# =============================================================================

#' Clean and parse string data consistently
#' @param input_string String to clean and parse
#' @param strict If TRUE, throw error on NAs; if FALSE, filter out NAs (default: FALSE)
#' @return Numeric vector
parse_numeric_string <- function(input_string, strict = FALSE) {
  if (is.null(input_string) || is.na(input_string) || input_string == "") {
    return(numeric(0))
  }

  cleaned_string <- input_string %>%
    str_replace_all(",|\\[|\\]|\\{|\\}|\\\n", "") %>%
    str_squish()

  if (cleaned_string == "") {
    if (strict) {
      stop("Parsed string does not result in a vector with numeric values")
    }
    return(numeric(0))
  }

  numeric_values <- as.numeric(unlist(strsplit(cleaned_string, " ")))

  if (strict && any(is.na(numeric_values))) {
    stop("Parsed string leads to NAs")
  }

  # Filter out NAs when not in strict mode
  numeric_values[!is.na(numeric_values)]
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

# =============================================================================
# Label and Class Functions
# =============================================================================

#' Add descriptive class labels to data frame
#' @param data_frame Input data frame
#' @param label_mapping Label mapping data frame
#' @return Data frame with added class labels
add_class_labels <- function(data_frame, label_mapping) {
  if (!"class" %in% colnames(data_frame)) {
    stop("Data frame must contain 'class' column")
  }

  data_frame$class_label <- label_mapping$Label[data_frame$class + 1]
  as.data.frame(data_frame)
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

# =============================================================================
# Neural Network Processing Functions
# =============================================================================

#' Process neural network results to clean epoch information
#' @param nn_results Neural network results data frame
#' @param group_by_outer_fold Whether to group by outer_fold in addition to params (TRUE for inner_cv, FALSE for train_test)
#' @return Processed data frame
process_neural_net_results <- function(nn_results, group_by_outer_fold = TRUE) {
  # Extract best_epoch as numeric
  nn_results$epochs <- str_match(nn_results$params, "best_epoch': np\\.int64\\((\\d+)\\)")[,2] |> as.integer()

  # Remove best_epoch from param string
  nn_results$params <- gsub(", 'best_epoch'.+", "", nn_results$params)

  # Add mean best_epoch per group back into param string
  if (group_by_outer_fold && "outer_fold" %in% colnames(nn_results)) {
    nn_results %>%
      group_by(outer_fold, params) %>%
      mutate(params = paste0(params, ", 'best_epoch': ", round(mean(epochs)), "}")) %>%
      ungroup()
  } else {
    nn_results %>%
      group_by(params) %>%
      mutate(params = paste0(params, ", 'best_epoch': ", round(mean(epochs)), "}")) %>%
      ungroup()
  }
}

# =============================================================================
# Hyperparameter Extraction Functions
# =============================================================================

#' Extract the best hyperparameters based on mean kappa
#' @param cv_results Data frame with cross-validation results
#' @param classification_type Classification type: "standard", "OvR", or "OvO"
#' @param include_outer_fold Whether to include outer_fold in grouping (TRUE for inner_cv, FALSE for train_test)
#' @return Data frame with best parameters
extract_best_hyperparameters <- function(cv_results, classification_type, include_outer_fold = TRUE) {
  # Determine required columns based on include_outer_fold
  if (include_outer_fold) {
    required_cols <- c("outer_fold", "params", "kappa", "accuracy")
  } else {
    required_cols <- c("params", "kappa", "accuracy")
  }

  # Validate inputs
  missing_cols <- setdiff(required_cols, colnames(cv_results))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
  }

  # Choose grouping variables based on classification type and include_outer_fold
  if (include_outer_fold) {
    grouping_vars <- if (classification_type == "standard") {
      c("outer_fold", "params")
    } else {
      c("outer_fold", "class", "params")
    }
  } else {
    grouping_vars <- if (classification_type == "standard") {
      c("params")
    } else {
      c("class", "params")
    }
  }

  # Compute mean kappa and accuracy across folds for each param set
  best_parameters <- cv_results %>%
    group_by(across(all_of(grouping_vars))) %>%
    summarise(
      mean_kappa = mean(kappa, na.rm = TRUE),
      sd_kappa = sd(kappa, na.rm = TRUE),
      mean_mcc = mean(mcc, na.rm = TRUE),
      sd_mcc = sd(mcc, na.rm = TRUE),
      mean_accuracy = mean(accuracy, na.rm = TRUE),
      sd_accuracy = sd(accuracy, na.rm = TRUE),
      across(any_of(c("class_0", "class_1")), first),
      .groups = "drop_last"
    )

  # For each group, retain the param set with the highest mean_kappa
  best_parameters %>%
    group_by(across(all_of(grouping_vars[-length(grouping_vars)]))) %>%
    filter(mean_kappa == max(mean_kappa, na.rm = TRUE)) %>%
    slice(1) %>%
    ungroup()
}

#' Extract best parameters for all models
#' @param model_results Model results list
#' @param model_configs Model configurations list
#' @param include_outer_fold Whether to include outer_fold in hyperparameter extraction
#' @return List of best parameters for each model
extract_all_best_parameters <- function(model_results, model_configs, include_outer_fold = TRUE) {
  cat("Extracting best parameters...\n")

  best_parameters <- list()

  for (model_name in names(model_results)) {
    config <- model_configs[[model_name]]
    cat(sprintf("Extracting best parameters for %s...\n", toupper(model_name)))

    best_parameters[[model_name]] <- list()

    for (fold_type in names(model_results[[model_name]])) {
      results <- model_results[[model_name]][[fold_type]]
      if (!is.null(results)) {
        best_params <- extract_best_hyperparameters(results, config$classification_type, include_outer_fold)
        best_parameters[[model_name]][[fold_type]] <- best_params
      }
    }
  }

  best_parameters
}

# =============================================================================
# Model Data Loading Functions
# =============================================================================

#' Load all model data from CSV files
#' @param model_configs Model configurations list
#' @param group_nn_by_outer_fold Whether to group neural network results by outer_fold
#' @return List of model results
load_all_model_data <- function(model_configs, group_nn_by_outer_fold = TRUE) {
  cat("Loading model data...\n")

  model_results <- list()

  for (model_name in names(model_configs)) {
    config <- model_configs[[model_name]]
    cat(sprintf("Loading %s data...\n", toupper(model_name)))

    model_results[[model_name]] <- list()

    for (fold_type in names(config$file_paths)) {
      file_path <- config$file_paths[[fold_type]]
      results <- combine_csv_files(file_path)
      if (model_name == "neural_net") {
        if (!is.null(results)) {
          results <- process_neural_net_results(results, group_nn_by_outer_fold)
        }
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
# Sample Filtering Functions
# =============================================================================

#' Filter samples to only include those with classes present in training
#' @param prob_matrix Probability matrix with y column and optional inner_fold, outer_fold, indices columns
#' @param training_classes Vector of class labels that were in the training set
#' @param fold_id Current fold identifier for logging
#' @param handle_na_labels Whether to filter out NA labels (TRUE for OvR inner_cv, FALSE for outer_cv)
#' @return List with filtered_matrix and stats
filter_samples_by_training_classes <- function(prob_matrix, training_classes, fold_id, handle_na_labels = TRUE) {
  if (is.null(prob_matrix) || nrow(prob_matrix) == 0) {
    return(list(filtered_matrix = prob_matrix, stats = NULL))
  }

  # Get true labels
  true_labels <- prob_matrix$y

  # Clean class names for comparison (make.names is applied to both)
  training_classes_clean <- make.names(training_classes)

  # Create mask for samples with classes in training
  valid_mask <- true_labels %in% training_classes_clean

  # Optionally handle NA labels (for OvR where some samples may not have training labels)
  if (handle_na_labels) {
    na_mask <- !is.na(true_labels)
    combined_mask <- na_mask & valid_mask
    unseen_classes <- unique(true_labels[!combined_mask & na_mask])
  } else {
    combined_mask <- valid_mask
    unseen_classes <- unique(true_labels[!combined_mask])
  }

  # Calculate statistics
  n_total <- nrow(prob_matrix)
  n_filtered <- sum(!combined_mask)
  n_kept <- sum(combined_mask)

  # Log filtering information
  if (n_filtered > 0) {
    cat(sprintf("    Fold %s: Filtered %d/%d samples (%.1f%%) with classes not in training\n",
                fold_id, n_filtered, n_total, 100 * n_filtered / n_total))
    if (length(unseen_classes) > 0) {
      cat(sprintf("      Classes in test but not in training: %s\n",
                  paste(unseen_classes, collapse = ", ")))
    }
  }

  # Filter the matrix
  filtered_matrix <- prob_matrix[combined_mask, , drop = FALSE]

  # Return filtered matrix and statistics
  stats <- data.frame(
    fold = fold_id,
    n_total = n_total,
    n_kept = n_kept,
    n_filtered = n_filtered,
    pct_filtered = 100 * n_filtered / n_total,
    unseen_classes = paste(unseen_classes, collapse = "; "),
    stringsAsFactors = FALSE
  )

  return(list(
    filtered_matrix = filtered_matrix,
    stats = stats
  ))
}

# =============================================================================
# Ensemble Weight Functions
# =============================================================================

#' Round to nearest base value
#' @param x Value to round
#' @param base Base value for rounding
#' @return Rounded value
round_to <- function(x, base = 0.05) {
  base * round(x / base)
}

#' Generate ensemble weight combinations
#' @param step Step size for weight generation (default 0.025)
#' @return Named list of weight configurations
generate_weights <- function(step = 0.025) {
  # Generate all combinations of weights from 0 to 1 in step increments
  steps <- seq(0, 1, by = step)
  grid <- expand.grid(SVM = steps, XGB = steps, NN = steps)
  grid <- grid[abs(rowSums(grid) - 1) < 1e-9, ]

  # Convert to a named list
  ENSEMBLE_WEIGHTS <- apply(grid, 1, function(row) {
    list(SVM = row["SVM"], XGB = row["XGB"], NN = row["NN"])
  })

  # Name the list elements for clarity
  names(ENSEMBLE_WEIGHTS) <- paste0("W", seq_along(ENSEMBLE_WEIGHTS))
  ENSEMBLE_WEIGHTS[["mix"]] <- list(SVM = 0.33, XGB = 0.33, NN = 0.33)
  # SVM as main fallback since this is in general the best working model
  ENSEMBLE_WEIGHTS[["ALL"]] <- list(SVM = 1, XGB = 0, NN = 0)

  return(ENSEMBLE_WEIGHTS)
}

#' Load ensemble weights used for outer fold analysis
#' @param weights_base_dir Base directory containing saved weights
#' @param analysis_type Type of analysis ("cv" or "loso")
#' @return List containing OvR and global ensemble weights
load_ensemble_weights <- function(weights_base_dir, analysis_type = "cv") {
  cat(sprintf("Loading ensemble weights for %s analysis...\n", toupper(analysis_type)))

  weights_dir <- file.path(weights_base_dir, analysis_type)

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
        f1_score = if ("f1_score" %in% names(row)) row$f1_score else row$mean_f1_score
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
        kappa = if ("kappa" %in% names(row)) row$kappa else row$mean_kappa
      )
    }

    weights_data$global_weights <- global_weights
    cat(sprintf("  Loaded global weights from: %s\n", global_weights_file))
  } else {
    warning(sprintf("Global weights file not found: %s", global_weights_file))
  }

  return(weights_data)
}

#' Save ensemble weights used for each fold and analysis type
#' @param ensemble_results Ensemble analysis results containing weights used
#' @param output_base_dir Base directory for output files
#' @param save_per_fold Whether to save per-fold weights (TRUE for inner_cv, FALSE for train_test)
save_ensemble_weights <- function(ensemble_results, output_base_dir, save_per_fold = TRUE) {
  cat("Saving ensemble weights used...\n")

  for (analysis_type in names(ensemble_results)) {
    cat(sprintf("Saving weights for %s analysis...\n", toupper(analysis_type)))

    # Create output directory for this analysis type
    # Check if output_base_dir already contains "ensemble_weights" in its name
    # (e.g., "ensemble_weights_merged" or "ensemble_weights_unmerged")
    # If so, don't add extra "ensemble_weights" subdirectory
    base_dir_name <- basename(output_base_dir)
    has_ensemble_weights_in_name <- grepl("ensemble_weights", base_dir_name, ignore.case = TRUE)

    if (save_per_fold && !has_ensemble_weights_in_name) {
      # For inner_cv with standard directory structure, add ensemble_weights subdirectory
      weights_output_dir <- file.path(output_base_dir, "ensemble_weights", analysis_type)
    } else {
      # For train_test or when directory already contains ensemble_weights, don't add extra folder
      weights_output_dir <- file.path(output_base_dir, analysis_type)
    }
    create_directory_safely(weights_output_dir)

    # Save OvR ensemble weights
    if ("ovr_ensemble_weights_used" %in% names(ensemble_results[[analysis_type]])) {
      ovr_weights <- ensemble_results[[analysis_type]]$ovr_ensemble_weights_used

      ovr_weights_df <- data.frame()

      if (save_per_fold) {
        # Save weights for each fold (inner_cv style)
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
              f1_score = if (!is.null(class_weight_info$mean_f1_score)) class_weight_info$mean_f1_score else class_weight_info$f1_score,
              stringsAsFactors = FALSE
            ))
          }
        }
      } else {
        # Save global weights per class (train_test style - use first fold as representative)
        first_fold_name <- names(ovr_weights)[1]
        if (!is.null(first_fold_name)) {
          fold_weights <- ovr_weights[[first_fold_name]]

          for (class_name in names(fold_weights)) {
            class_weight_info <- fold_weights[[class_name]]

            ovr_weights_df <- rbind(ovr_weights_df, data.frame(
              class = class_name,
              weight_name = class_weight_info$weight_name,
              svm_weight = class_weight_info$weights$SVM,
              xgb_weight = class_weight_info$weights$XGB,
              nn_weight = class_weight_info$weights$NN,
              mean_f1_score = if (!is.null(class_weight_info$f1_score)) class_weight_info$f1_score else class_weight_info$mean_f1_score,
              stringsAsFactors = FALSE
            ))
          }
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

      global_weights_df <- data.frame()

      if (save_per_fold) {
        # Save weights for each fold (inner_cv style)
        for (fold_name in names(global_weights)) {
          fold_weight_info <- global_weights[[fold_name]]

          global_weights_df <- rbind(global_weights_df, data.frame(
            fold = fold_name,
            weight_name = fold_weight_info$weight_name,
            svm_weight = fold_weight_info$weights$SVM,
            xgb_weight = fold_weight_info$weights$XGB,
            nn_weight = fold_weight_info$weights$NN,
            kappa = if (!is.null(fold_weight_info$mean_kappa)) fold_weight_info$mean_kappa else fold_weight_info$kappa,
            stringsAsFactors = FALSE
          ))
        }
      } else {
        # Save single global weight (train_test style - use first fold as representative)
        first_fold_name <- names(global_weights)[1]
        if (!is.null(first_fold_name)) {
          fold_weight_info <- global_weights[[first_fold_name]]

          global_weights_df <- data.frame(
            weight_name = fold_weight_info$weight_name,
            svm_weight = fold_weight_info$weights$SVM,
            xgb_weight = fold_weight_info$weights$XGB,
            nn_weight = fold_weight_info$weights$NN,
            mean_kappa = if (!is.null(fold_weight_info$kappa)) fold_weight_info$kappa else fold_weight_info$mean_kappa,
            stringsAsFactors = FALSE
          )
        }
      }

      # Save global weights
      global_weights_file <- file.path(weights_output_dir, "global_ensemble_weights_used.csv")
      write.csv(global_weights_df, global_weights_file, row.names = FALSE)
      cat(sprintf("  Saved global weights: %s\n", global_weights_file))
    }
  }
}

# =============================================================================
# Matrix Alignment Functions
# =============================================================================

#' Align probability matrices from different models for ensemble analysis
#' @param prob_matrices List of probability matrices from different models
#' @param outer_fold_name Name of the outer fold being processed
#' @param inner_fold_name Name of the inner fold being processed (NULL for train_test/outer_cv)
#' @param type Type of analysis ("cv" or "loso")
#' @return List of aligned probability matrices
align_probability_matrices <- function(prob_matrices, outer_fold_name, inner_fold_name = NULL, type) {
  # Extract matrices for this fold - handle both nested and flat structures
  if (!is.null(inner_fold_name)) {
    # Nested structure: [model][type][outer_fold][inner_fold]
    svm_matrix <- prob_matrices$svm[[type]][[outer_fold_name]][[inner_fold_name]]
    xgb_matrix <- prob_matrices$xgboost[[type]][[outer_fold_name]][[inner_fold_name]]
    nn_matrix <- prob_matrices$neural_net[[type]][[outer_fold_name]][[inner_fold_name]]
  } else {
    # Flat structure: [model][type][outer_fold]
    svm_matrix <- prob_matrices$svm[[type]][[outer_fold_name]]
    xgb_matrix <- prob_matrices$xgboost[[type]][[outer_fold_name]]
    nn_matrix <- prob_matrices$neural_net[[type]][[outer_fold_name]]
  }

  # Check if all matrices exist
  if (is.null(svm_matrix) || is.null(xgb_matrix) || is.null(nn_matrix)) {
    fold_desc <- if (!is.null(inner_fold_name)) paste(outer_fold_name, inner_fold_name, sep = "_") else outer_fold_name
    warning(sprintf("Missing probability matrix for fold %s in %s analysis", fold_desc, type))
    return(NULL)
  }

  # Align samples across all three models using indices (critical after filtering)
  if ("indices" %in% colnames(svm_matrix) &&
      "indices" %in% colnames(xgb_matrix) &&
      "indices" %in% colnames(nn_matrix)) {

    svm_samples <- svm_matrix$indices
    xgb_samples <- xgb_matrix$indices
    nn_samples <- nn_matrix$indices

    # Find common samples
    common_samples <- Reduce(intersect, list(svm_samples, xgb_samples, nn_samples))

    if (length(common_samples) == 0) {
      fold_desc <- if (!is.null(inner_fold_name)) paste(outer_fold_name, inner_fold_name, sep = "_") else outer_fold_name
      warning(sprintf("No common samples across models for fold %s, skipping", fold_desc))
      return(NULL)
    }

    # Get original counts
    n_svm_orig <- nrow(svm_matrix)
    n_xgb_orig <- nrow(xgb_matrix)
    n_nn_orig <- nrow(nn_matrix)

    # Filter to common samples
    svm_matrix <- svm_matrix[svm_matrix$indices %in% common_samples, ]
    xgb_matrix <- xgb_matrix[xgb_matrix$indices %in% common_samples, ]
    nn_matrix <- nn_matrix[nn_matrix$indices %in% common_samples, ]

    # Sort by indices to ensure alignment
    svm_matrix <- svm_matrix[order(svm_matrix$indices), ]
    xgb_matrix <- xgb_matrix[order(xgb_matrix$indices), ]
    nn_matrix <- nn_matrix[order(nn_matrix$indices), ]

    # Log if samples were dropped
    max_orig <- max(n_svm_orig, n_xgb_orig, n_nn_orig)
    n_dropped <- max_orig - length(common_samples)
    if (n_dropped > 0) {
      fold_desc <- if (!is.null(inner_fold_name)) paste(outer_fold_name, inner_fold_name, sep = "_") else outer_fold_name
      cat(sprintf("    Aligned samples for fold %s: dropped %d samples to match across models (SVM: %d, XGB: %d, NN: %d -> common: %d)\n",
                  fold_desc, n_dropped, n_svm_orig, n_xgb_orig, n_nn_orig, length(common_samples)))
    }
  } else {
    # If no indices column, check row counts match
    if (nrow(svm_matrix) != nrow(xgb_matrix) || nrow(svm_matrix) != nrow(nn_matrix)) {
      fold_desc <- if (!is.null(inner_fold_name)) paste(outer_fold_name, inner_fold_name, sep = "_") else outer_fold_name
      warning(sprintf("Sample counts don't match for fold %s (SVM: %d, XGB: %d, NN: %d), attempting to align by truncation",
                      fold_desc, nrow(svm_matrix), nrow(xgb_matrix), nrow(nn_matrix)))
    }
  }

  # Extract true labels
  truth_svm <- make.names(svm_matrix$y)
  truth_xgb <- make.names(xgb_matrix$y)
  truth_nn <- make.names(nn_matrix$y)

  # Store non_prob columns
  non_prob_cols <- svm_matrix[, colnames(svm_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study"), drop = FALSE]

  # Remove non-probability columns from matrices
  svm_matrix <- svm_matrix[, !colnames(svm_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study"), drop = FALSE]
  xgb_matrix <- xgb_matrix[, !colnames(xgb_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study"), drop = FALSE]
  nn_matrix <- nn_matrix[, !colnames(nn_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study"), drop = FALSE]

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

    if (nrow(matrix_data) < max_samples) {
      cat(sprintf("The probabilities for %s have less samples than max_samples\n", model_name))
    }

    # Truncate to minimum sample size if necessary
    if (nrow(matrix_data) > min_samples) {
      matrix_data <- matrix_data[1:min_samples, , drop = FALSE]
    }

    aligned_matrices[[model_name]] <- matrix_data
  }

  # Add aligned non_prob_cols to the result
  aligned_matrices$non_prob_cols <- non_prob_cols[1:min_samples, , drop = FALSE]
  aligned_matrices
}

# =============================================================================
# Fast Metric Calculation Functions
# =============================================================================

#' Fast kappa calculation without caret overhead
#' @param preds Factor of predicted labels
#' @param truth Factor of true labels
#' @return Numeric kappa value
fast_kappa <- function(preds, truth) {
  # Build confusion matrix
  cm <- table(preds, truth)

  # Calculate observed accuracy
  n <- sum(cm)
  if (n == 0) return(NA_real_)

  observed_accuracy <- sum(diag(cm)) / n

  # Calculate expected accuracy (chance agreement)
  row_sums <- rowSums(cm)
  col_sums <- colSums(cm)
  expected_accuracy <- sum(row_sums * col_sums) / (n * n)

  # Cohen's Kappa
  if (expected_accuracy == 1) return(1)
  (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
}

#' Fast accuracy calculation
#' @param preds Factor of predicted labels
#' @param truth Factor of true labels
#' @return Numeric accuracy value
fast_accuracy <- function(preds, truth) {
  sum(preds == truth) / length(truth)
}

#' Fast binary metrics calculation (sensitivity, specificity, F1, etc.)
#' @param preds Factor of binary predictions ("Class", "Not_Class")
#' @param truth Factor of binary truth ("Class", "Not_Class")
#' @return Named vector with sensitivity, specificity, balanced_accuracy, f1_score, prevalence
fast_binary_metrics <- function(preds, truth) {
  # Build 2x2 confusion matrix
  # Levels: Not_Class, Class (negative, positive)
  tp <- sum(preds == "Class" & truth == "Class")
  tn <- sum(preds == "Not_Class" & truth == "Not_Class")
  fp <- sum(preds == "Class" & truth == "Not_Class")
  fn <- sum(preds == "Not_Class" & truth == "Class")

  n <- tp + tn + fp + fn
  positives <- tp + fn
  negatives <- tn + fp

  sensitivity <- if (positives > 0) tp / positives else NA_real_
  specificity <- if (negatives > 0) tn / negatives else NA_real_
  balanced_accuracy <- if (!is.na(sensitivity) && !is.na(specificity)) (sensitivity + specificity) / 2 else NA_real_

  precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  f1_score <- if (!is.na(precision) && !is.na(sensitivity) && (precision + sensitivity) > 0) {
    2 * precision * sensitivity / (precision + sensitivity)
  } else {
    NA_real_
  }

  prevalence <- positives / n

  c(
    sensitivity = sensitivity,
    specificity = specificity,
    balanced_accuracy = balanced_accuracy,
    f1_score = f1_score,
    prevalence = prevalence
  )
}

# =============================================================================
# Optimized Analysis Functions
# =============================================================================

#' Evaluate a single cutoff for rejection analysis
#' @param cutoff Probability cutoff value
#' @param max_probs Vector of maximum probabilities
#' @param truth Factor of true labels
#' @param preds Factor of predicted labels
#' @param model_name Name of the model
#' @param type Type of analysis
#' @param fold_name Name of the fold
#' @param total_samples Total number of samples
#' @return Data frame row with rejection results for this cutoff
evaluate_single_cutoff <- function(cutoff, max_probs, truth, preds, model_name, type, fold_name, total_samples) {
  # Identify samples to reject (max probability below cutoff)
  accepted_mask <- max_probs >= cutoff

  n_accepted <- sum(accepted_mask)
  if (n_accepted == 0) {
    return(NULL)
  }

  n_rejected <- total_samples - n_accepted

  # Calculate accuracy for rejected samples (if any)
  rejected_accuracy <- NA
  if (n_rejected > 0) {
    rejected_mask <- !accepted_mask
    rejected_accuracy <- fast_accuracy(preds[rejected_mask], truth[rejected_mask])
  }

  # Calculate kappa and accuracy for accepted samples
  accepted_truth <- truth[accepted_mask]
  accepted_preds <- preds[accepted_mask]

  kappa <- fast_kappa(accepted_preds, accepted_truth)
  accuracy <- fast_accuracy(accepted_preds, accepted_truth)

  data.frame(
    model = model_name,
    type = type,
    fold = fold_name,
    prob_cutoff = cutoff,
    kappa = kappa,
    accuracy = accuracy,
    n_accepted = n_accepted,
    n_rejected = n_rejected,
    perc_rejected = n_rejected / total_samples,
    rejected_accuracy = rejected_accuracy,
    total_samples = total_samples,
    stringsAsFactors = FALSE
  )
}

#' Vectorized cutoff analysis - much faster than per-cutoff evaluation
#' Uses sorted probabilities and cumulative calculations
#' @param prob_matrix Probability matrix with class probabilities and true labels
#' @param fold_name Name of the fold being analyzed
#' @param model_name Name of the model being analyzed
#' @param type Type of analysis ("cv" or "loso")
#' @param cutoff_step Step size for probability cutoffs (default: 0.01)
#' @return Data frame with rejection analysis results
evaluate_single_matrix_with_rejection_vectorized <- function(prob_matrix, fold_name, model_name, type, cutoff_step = 0.01) {
  # Extract true labels and remove from probability matrix
  truth <- prob_matrix$y
  prob_matrix_clean <- prob_matrix[, !colnames(prob_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study"), drop = FALSE]

  # Clean class labels
  truth <- gsub("Class. ", "", truth)

  # Vectorized: Get predictions using max.col (much faster than apply)
  prob_mat <- as.matrix(prob_matrix_clean)
  pred_indices <- max.col(prob_mat, ties.method = "first")
  preds <- colnames(prob_matrix_clean)[pred_indices]
  preds <- gsub("Class. ", "", preds)

  # Vectorized: Get max probabilities
  max_probs <- prob_mat[cbind(seq_len(nrow(prob_mat)), pred_indices)]

  # Ensure all classes are represented
  all_classes <- unique(c(truth, preds))
  truth <- factor(truth, levels = all_classes)
  preds <- factor(preds, levels = all_classes)

  # Pre-compute correctness
  correct <- as.integer(truth == preds)
  total_samples <- length(truth)

  # Sort by max_probs for efficient cumulative processing
  sort_order <- order(max_probs)
  sorted_probs <- max_probs[sort_order]
  sorted_correct <- correct[sort_order]
  sorted_truth <- truth[sort_order]
  sorted_preds <- preds[sort_order]

  # Generate cutoffs
  prob_cutoffs <- seq(0.00, 1.00, by = cutoff_step)
  n_cutoffs <- length(prob_cutoffs)

  # Pre-allocate result vectors
  kappa_vec <- numeric(n_cutoffs)
  accuracy_vec <- numeric(n_cutoffs)
  n_accepted_vec <- integer(n_cutoffs)
  n_rejected_vec <- integer(n_cutoffs)
  rejected_accuracy_vec <- numeric(n_cutoffs)

  # For each cutoff, find the split point
  for (i in seq_len(n_cutoffs)) {
    cutoff <- prob_cutoffs[i]

    # Find first index where prob >= cutoff
    first_accepted <- which(sorted_probs >= cutoff)[1]

    if (is.na(first_accepted)) {
      # All samples rejected
      kappa_vec[i] <- NA
      accuracy_vec[i] <- NA
      n_accepted_vec[i] <- 0
      n_rejected_vec[i] <- total_samples
      rejected_accuracy_vec[i] <- mean(sorted_correct)
    } else {
      n_accepted <- total_samples - first_accepted + 1
      n_rejected <- first_accepted - 1

      n_accepted_vec[i] <- n_accepted
      n_rejected_vec[i] <- n_rejected

      # Accepted samples metrics
      accepted_idx <- first_accepted:total_samples
      accepted_truth <- sorted_truth[accepted_idx]
      accepted_preds <- sorted_preds[accepted_idx]

      kappa_vec[i] <- fast_kappa(accepted_preds, accepted_truth)
      accuracy_vec[i] <- mean(sorted_correct[accepted_idx])

      # Rejected samples accuracy
      if (n_rejected > 0) {
        rejected_accuracy_vec[i] <- mean(sorted_correct[1:(first_accepted - 1)])
      } else {
        rejected_accuracy_vec[i] <- NA
      }
    }
  }

  # Remove cutoffs where all samples are rejected
  valid_mask <- n_accepted_vec > 0

  data.frame(
    model = model_name,
    type = type,
    fold = fold_name,
    prob_cutoff = prob_cutoffs[valid_mask],
    kappa = kappa_vec[valid_mask],
    accuracy = accuracy_vec[valid_mask],
    n_accepted = n_accepted_vec[valid_mask],
    n_rejected = n_rejected_vec[valid_mask],
    perc_rejected = n_rejected_vec[valid_mask] / total_samples,
    rejected_accuracy = rejected_accuracy_vec[valid_mask],
    total_samples = total_samples,
    stringsAsFactors = FALSE
  )
}

#' Evaluate rejection analysis for a single probability matrix (optimized vectorized version)
#' @param prob_matrix Probability matrix with class probabilities and true labels
#' @param fold_name Name of the fold being analyzed
#' @param model_name Name of the model being analyzed
#' @param type Type of analysis ("cv" or "loso")
#' @param cutoff_step Step size for probability cutoffs (default: 0.01, use 0.05 for faster analysis)
#' @return Data frame with rejection analysis results
evaluate_single_matrix_with_rejection_parallel <- function(prob_matrix, fold_name, model_name, type, cutoff_step = 0.01) {
  # Use the faster vectorized implementation
  evaluate_single_matrix_with_rejection_vectorized(prob_matrix, fold_name, model_name, type, cutoff_step)
}

#' Evaluate a single weight configuration for global ensemble (matrix version)
#' @param weight_config Named list with SVM, XGB, NN weights
#' @param weight_name Name of this weight configuration
#' @param prob_mat_SVM SVM probability matrix
#' @param prob_mat_XGB XGBoost probability matrix
#' @param prob_mat_NN Neural net probability matrix
#' @param class_names Vector of class names
#' @param truth Factor of true labels
#' @param outer_fold Outer fold identifier
#' @param inner_fold Inner fold identifier
#' @param type Type of analysis
#' @return Data frame row with ensemble performance for this weight config
evaluate_single_weight_global_matrix <- function(weight_config, weight_name, prob_mat_SVM, prob_mat_XGB, prob_mat_NN, class_names, truth, outer_fold, inner_fold, type) {
  # Calculate weighted ensemble probabilities using matrix operations
  prob_mat <- prob_mat_SVM * weight_config$SVM +
    prob_mat_XGB * weight_config$XGB +
    prob_mat_NN * weight_config$NN

  # Normalize probabilities to sum to 1 for each sample
  row_sums <- rowSums(prob_mat)
  prob_mat <- prob_mat / row_sums

  # Vectorized: Get predictions using max.col (much faster than apply)
  pred_indices <- max.col(prob_mat, ties.method = "first")
  preds <- class_names[pred_indices]

  # Clean class labels
  truth_clean <- make.names(gsub("Class. ", "", truth))
  preds_clean <- make.names(gsub("Class. ", "", preds))

  # Ensure all classes are represented
  all_classes <- unique(c(truth_clean, preds_clean))
  truth_factor <- factor(truth_clean, levels = all_classes)
  preds_factor <- factor(preds_clean, levels = all_classes)

  # Use fast kappa and accuracy calculations
  kappa <- fast_kappa(preds_factor, truth_factor)
  accuracy <- fast_accuracy(preds_factor, truth_factor)

  data.frame(
    outer_fold = outer_fold,
    inner_fold = inner_fold,
    weights = weight_name,
    type = type,
    kappa = kappa,
    accuracy = accuracy,
    stringsAsFactors = FALSE
  )
}

#' Evaluate a single weight configuration for global ensemble (backward compatibility)
#' @param weight_config Named list with SVM, XGB, NN weights
#' @param weight_name Name of this weight configuration
#' @param prob_df_SVM SVM probability data frame
#' @param prob_df_XGB XGBoost probability data frame
#' @param prob_df_NN Neural net probability data frame
#' @param truth Factor of true labels
#' @param outer_fold Outer fold identifier
#' @param inner_fold Inner fold identifier
#' @param type Type of analysis
#' @return Data frame row with ensemble performance for this weight config
evaluate_single_weight_global <- function(weight_config, weight_name, prob_df_SVM, prob_df_XGB, prob_df_NN, truth, outer_fold, inner_fold, type) {
  # Convert to matrices and call matrix version
  prob_mat_SVM <- as.matrix(prob_df_SVM)
  prob_mat_XGB <- as.matrix(prob_df_XGB)
  prob_mat_NN <- as.matrix(prob_df_NN)
  class_names <- colnames(prob_df_SVM)

  evaluate_single_weight_global_matrix(
    weight_config, weight_name,
    prob_mat_SVM, prob_mat_XGB, prob_mat_NN, class_names, truth,
    outer_fold, inner_fold, type
  )
}

#' Evaluate a single weight configuration for OvR ensemble (matrix version)
#' @param weight_config Named list with SVM, XGB, NN weights
#' @param weight_name Name of this weight configuration
#' @param class_name Name of the class being evaluated
#' @param prob_mat_SVM SVM probability matrix
#' @param prob_mat_XGB XGBoost probability matrix
#' @param prob_mat_NN Neural net probability matrix
#' @param class_col_idx Column index for the class in the matrices
#' @param truth Factor of true labels
#' @param outer_fold Outer fold identifier
#' @param inner_fold Inner fold identifier
#' @param type Type of analysis
#' @return Data frame row with OvR ensemble performance for this weight/class combination
evaluate_single_weight_ovr_matrix <- function(weight_config, weight_name, class_name, prob_mat_SVM, prob_mat_XGB, prob_mat_NN, class_col_idx, truth, outer_fold, inner_fold, type) {
  # Calculate weighted ensemble probabilities for this class only using matrix operations
  class_probs <- prob_mat_SVM[, class_col_idx] * weight_config$SVM +
    prob_mat_XGB[, class_col_idx] * weight_config$XGB +
    prob_mat_NN[, class_col_idx] * weight_config$NN

  # Vectorized binary predictions: class vs not class
  binary_preds <- ifelse(class_probs > 0.5, "Class", "Not_Class")

  # Vectorized binary truth: class vs not class
  binary_truth <- ifelse(truth == class_name, "Class", "Not_Class")

  # Use fast binary metrics calculation
  metrics <- fast_binary_metrics(binary_preds, binary_truth)

  data.frame(
    outer_fold = outer_fold,
    inner_fold = inner_fold,
    weights = weight_name,
    type = type,
    class = gsub("Class.", "", class_name),
    sensitivity = metrics["sensitivity"],
    specificity = metrics["specificity"],
    balanced_accuracy = metrics["balanced_accuracy"],
    f1_score = metrics["f1_score"],
    prevalence = metrics["prevalence"],
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

#' Evaluate a single weight configuration for OvR ensemble (backward compatibility)
#' @param weight_config Named list with SVM, XGB, NN weights
#' @param weight_name Name of this weight configuration
#' @param class_name Name of the class being evaluated
#' @param prob_df_SVM SVM probability data frame
#' @param prob_df_XGB XGBoost probability data frame
#' @param prob_df_NN Neural net probability data frame
#' @param truth Factor of true labels
#' @param outer_fold Outer fold identifier
#' @param inner_fold Inner fold identifier
#' @param type Type of analysis
#' @return Data frame row with OvR ensemble performance for this weight/class combination
evaluate_single_weight_ovr <- function(weight_config, weight_name, class_name, prob_df_SVM, prob_df_XGB, prob_df_NN, truth, outer_fold, inner_fold, type) {
  # Convert to matrices and find class column index
  prob_mat_SVM <- as.matrix(prob_df_SVM)
  prob_mat_XGB <- as.matrix(prob_df_XGB)
  prob_mat_NN <- as.matrix(prob_df_NN)
  class_col_idx <- which(colnames(prob_df_SVM) == class_name)

  if (length(class_col_idx) == 0) {
    stop(sprintf("Class %s not found in probability matrices", class_name))
  }

  evaluate_single_weight_ovr_matrix(
    weight_config, weight_name, class_name,
    prob_mat_SVM, prob_mat_XGB, prob_mat_NN, class_col_idx, truth,
    outer_fold, inner_fold, type
  )
}

#' Create all weight-class combinations for OvR evaluation
#' @param weights List of weight configurations
#' @param all_classes Vector of class names
#' @return Data frame with all combinations
create_weight_class_combinations <- function(weights, all_classes) {
  expand.grid(
    weight_idx = seq_along(weights),
    class_name = all_classes,
    stringsAsFactors = FALSE
  )
}

# =============================================================================
# Batch Weight Evaluation Functions
# =============================================================================

#' Batch evaluate all weight configurations for global ensemble
#' Pre-computes all weighted probability matrices, then evaluates metrics
#' @param weights List of weight configurations
#' @param prob_mat_SVM SVM probability matrix
#' @param prob_mat_XGB XGBoost probability matrix
#' @param prob_mat_NN Neural net probability matrix
#' @param class_names Vector of class names
#' @param truth Factor of true labels
#' @param outer_fold Outer fold identifier
#' @param inner_fold Inner fold identifier
#' @param type Type of analysis
#' @return Data frame with ensemble performance for all weight configs
evaluate_batch_weights_global <- function(weights, prob_mat_SVM, prob_mat_XGB, prob_mat_NN, class_names, truth, outer_fold, inner_fold, type) {
  # Pre-compute all weighted probability matrices
  weighted_matrices <- lapply(weights, function(w) {
    prob_mat_SVM * w$SVM + prob_mat_XGB * w$XGB + prob_mat_NN * w$NN
  })

  # Batch normalize all matrices
  weighted_matrices <- lapply(weighted_matrices, function(mat) {
    row_sums <- rowSums(mat)
    mat / row_sums
  })

  # Pre-compute cleaned truth once (used for all weights)
  truth_clean <- make.names(gsub("Class. ", "", truth))
  all_classes <- unique(truth_clean)  # Will be expanded with predictions

  # Evaluate all weights
  results_list <- mapply(function(mat, w_name) {
    # Get predictions using max.col
    pred_indices <- max.col(mat, ties.method = "first")
    preds <- class_names[pred_indices]
    preds_clean <- make.names(gsub("Class. ", "", preds))

    # Ensure all classes are represented
    all_classes_combined <- unique(c(truth_clean, preds_clean))
    truth_factor <- factor(truth_clean, levels = all_classes_combined)
    preds_factor <- factor(preds_clean, levels = all_classes_combined)

    # Compute metrics
    kappa <- fast_kappa(preds_factor, truth_factor)
    accuracy <- fast_accuracy(preds_factor, truth_factor)

    data.frame(
      outer_fold = outer_fold,
      inner_fold = inner_fold,
      weights = w_name,
      type = type,
      kappa = kappa,
      accuracy = accuracy,
      stringsAsFactors = FALSE
    )
  }, weighted_matrices, names(weights), SIMPLIFY = FALSE)

  # Combine all results
  do.call(rbind, results_list)
}

#' Batch evaluate all weight-class combinations for OvR ensemble
#' Pre-computes all weighted probability matrices, then evaluates metrics
#' @param weights List of weight configurations
#' @param prob_mat_SVM SVM probability matrix
#' @param prob_mat_XGB XGBoost probability matrix
#' @param prob_mat_NN Neural net probability matrix
#' @param all_classes Vector of all class names
#' @param truth Factor of true labels
#' @param outer_fold Outer fold identifier
#' @param inner_fold Inner fold identifier
#' @param type Type of analysis
#' @return Data frame with OvR ensemble performance for all weight/class combinations
evaluate_batch_weights_ovr <- function(weights, prob_mat_SVM, prob_mat_XGB, prob_mat_NN, all_classes, truth, outer_fold, inner_fold, type) {
  # Create all weight-class combinations
  combinations <- create_weight_class_combinations(weights, all_classes)

  # Pre-compute class column indices for faster access
  class_col_indices <- match(combinations$class_name, all_classes)

  # Pre-compute all weighted probability matrices (one per weight)
  weighted_matrices <- lapply(weights, function(w) {
    prob_mat_SVM * w$SVM + prob_mat_XGB * w$XGB + prob_mat_NN * w$NN
  })

  # Evaluate all weight-class combinations
  results_list <- lapply(seq_len(nrow(combinations)), function(idx) {
    weight_idx <- combinations$weight_idx[idx]
    class_name <- combinations$class_name[idx]
    class_col_idx <- class_col_indices[idx]
    weight_name <- names(weights)[weight_idx]

    # Get weighted probabilities for this class from pre-computed matrix
    class_probs <- weighted_matrices[[weight_idx]][, class_col_idx]

    # Vectorized binary predictions: class vs not class
    binary_preds <- ifelse(class_probs > 0.5, "Class", "Not_Class")

    # Vectorized binary truth: class vs not class
    binary_truth <- ifelse(truth == class_name, "Class", "Not_Class")

    # Use fast binary metrics calculation
    metrics <- fast_binary_metrics(binary_preds, binary_truth)

    data.frame(
      outer_fold = outer_fold,
      inner_fold = inner_fold,
      weights = weight_name,
      type = type,
      class = gsub("Class.", "", class_name),
      sensitivity = metrics["sensitivity"],
      specificity = metrics["specificity"],
      balanced_accuracy = metrics["balanced_accuracy"],
      f1_score = metrics["f1_score"],
      prevalence = metrics["prevalence"],
      stringsAsFactors = FALSE,
      row.names = NULL
    )
  })

  # Combine all results
  do.call(rbind, results_list)
}

# =============================================================================
# Unified Ensemble Analysis Functions
# =============================================================================

#' Cached version of align_probability_matrices to avoid redundant alignment
#' @param prob_matrices List of probability matrices from different models
#' @param outer_fold_name Name of the outer fold being processed
#' @param inner_fold_name Name of the inner fold being processed (NULL for train_test/outer_cv)
#' @param type Type of analysis ("cv" or "loso")
#' @param cache Environment to store cached aligned matrices (created if NULL)
#' @return List of aligned probability matrices
align_probability_matrices_cached <- function(prob_matrices, outer_fold_name, inner_fold_name = NULL, type, cache = NULL) {
  # Create cache if not provided
  if (is.null(cache)) {
    cache <- new.env(hash = TRUE)
  }

  # Create cache key
  cache_key <- if (!is.null(inner_fold_name)) {
    paste(type, outer_fold_name, inner_fold_name, sep = "_")
  } else {
    paste(type, outer_fold_name, sep = "_")
  }

  # Check cache
  if (exists(cache_key, envir = cache)) {
    return(get(cache_key, envir = cache))
  }

  # Align matrices
  aligned <- align_probability_matrices(prob_matrices, outer_fold_name, inner_fold_name, type)

  # Store in cache
  if (!is.null(aligned)) {
    assign(cache_key, aligned, envir = cache)
  }

  return(aligned)
}

#' Perform global ensemble optimization (unified for both inner_cv and train_test)
#' @param results Analysis results containing probability matrices
#' @param weights Weight configurations for ensemble
#' @param type Type of analysis ("cv" or "loso")
#' @param has_inner_folds Whether data has inner fold nesting (TRUE for inner_cv, FALSE for train_test)
#' @return List of performance metrics for each outer fold and weight configuration
perform_global_ensemble_analysis_unified <- function(results, weights, type = "cv", has_inner_folds = TRUE) {
  cat(sprintf("Performing global ensemble analysis (%s)...\n", ifelse(has_inner_folds, "with inner folds", "train/test")))

  outer_folds <- names(results$probability_matrices$svm[[type]])

  # Create cache for aligned matrices
  alignment_cache <- new.env(hash = TRUE)

  # Helper function to process a single outer fold
  process_outer_fold <- function(outer_fold) {
    # Pre-allocate list to collect results (avoid rbind in loops)
    all_weight_results <- list()

    if (has_inner_folds) {
      # Inner CV: iterate over inner folds
      inner_folds <- names(results$probability_matrices$svm[[type]][[outer_fold]])

      for (inner_fold in inner_folds) {
        # Use cached alignment
        aligned_matrices <- align_probability_matrices_cached(
          results$probability_matrices, outer_fold, inner_fold, type, alignment_cache
        )
        if (is.null(aligned_matrices)) next

        # Convert to matrices once for efficiency
        prob_mat_SVM <- as.matrix(aligned_matrices$svm)
        prob_mat_XGB <- as.matrix(aligned_matrices$xgboost)
        prob_mat_NN <- as.matrix(aligned_matrices$neural_net)
        truth <- make.names(aligned_matrices$non_prob_cols$y)
        class_names <- colnames(aligned_matrices$svm)

        # Batch evaluate all weights at once
        fold_results <- evaluate_batch_weights_global(
          weights, prob_mat_SVM, prob_mat_XGB, prob_mat_NN, class_names, truth,
          outer_fold, inner_fold, type
        )
        all_weight_results[[length(all_weight_results) + 1]] <- fold_results
      }

      # Combine all results at once
      if (length(all_weight_results) > 0) {
        all_results_df <- do.call(rbind, all_weight_results)

        # Aggregate across inner folds
        aggregated_results <- all_results_df %>%
          dplyr::group_by(outer_fold, weights, type) %>%
          dplyr::summarise(
            mean_kappa = mean(kappa, na.rm = TRUE),
            mean_accuracy = mean(accuracy, na.rm = TRUE),
            sd_kappa = sd(kappa, na.rm = TRUE),
            sd_accuracy = sd(accuracy, na.rm = TRUE),
            n_inner_folds = dplyr::n(),
            .groups = "drop"
          )
        return(aggregated_results)
      }
    } else {
      # Train/test: no inner folds
      aligned_matrices <- align_probability_matrices_cached(
        results$probability_matrices, outer_fold, NULL, type, alignment_cache
      )
      if (is.null(aligned_matrices)) return(NULL)

      # Convert to matrices once for efficiency
      prob_mat_SVM <- as.matrix(aligned_matrices$svm)
      prob_mat_XGB <- as.matrix(aligned_matrices$xgboost)
      prob_mat_NN <- as.matrix(aligned_matrices$neural_net)
      truth <- make.names(aligned_matrices$non_prob_cols$y)
      class_names <- colnames(aligned_matrices$svm)

      # Batch evaluate all weights at once
      weight_results <- evaluate_batch_weights_global(
        weights, prob_mat_SVM, prob_mat_XGB, prob_mat_NN, class_names, truth,
        outer_fold, NA, type
      )

      if (nrow(weight_results) > 0) {
        weight_results$inner_fold <- NULL
        return(weight_results)
      }
    }
    return(NULL)
  }

  # Process outer folds
  df_list <- list()
  for (outer_fold in outer_folds) {
    cat(sprintf("  Processing outer fold %s...\n", outer_fold))
    df_list[[outer_fold]] <- process_outer_fold(outer_fold)
  }

  df_list
}

#' Perform OvR ensemble analysis (unified for both inner_cv and train_test)
#' @param results Analysis results containing probability matrices
#' @param weights Weight configurations for ensemble
#' @param type Type of analysis ("cv" or "loso")
#' @param has_inner_folds Whether data has inner fold nesting (TRUE for inner_cv, FALSE for train_test)
#' @return List of performance metrics for each outer fold, weight, and class
perform_ovr_ensemble_analysis_unified <- function(results, weights, type = "cv", has_inner_folds = TRUE) {
  cat(sprintf("Performing OvR ensemble analysis (%s)...\n", ifelse(has_inner_folds, "with inner folds", "train/test")))

  outer_folds <- names(results$probability_matrices$svm[[type]])

  # Create cache for aligned matrices
  alignment_cache <- new.env(hash = TRUE)

  # Helper function to process a single outer fold
  process_outer_fold <- function(outer_fold) {
    # Pre-allocate list to collect results (avoid rbind in loops)
    all_combo_results <- list()

    if (has_inner_folds) {
      # Inner CV: iterate over inner folds
      inner_folds <- names(results$probability_matrices$svm[[type]][[outer_fold]])

      for (inner_fold in inner_folds) {
        # Use cached alignment
        aligned_matrices <- align_probability_matrices_cached(
          results$probability_matrices, outer_fold, inner_fold, type, alignment_cache
        )
        if (is.null(aligned_matrices)) next

        # Convert to matrices once for efficiency
        prob_mat_SVM <- as.matrix(aligned_matrices$svm)
        prob_mat_XGB <- as.matrix(aligned_matrices$xgboost)
        prob_mat_NN <- as.matrix(aligned_matrices$neural_net)
        truth <- make.names(aligned_matrices$non_prob_cols$y)
        all_classes <- colnames(aligned_matrices$svm)

        # Batch evaluate all weight-class combinations at once
        fold_results <- evaluate_batch_weights_ovr(
          weights, prob_mat_SVM, prob_mat_XGB, prob_mat_NN, all_classes, truth,
          outer_fold, inner_fold, type
        )
        all_combo_results[[length(all_combo_results) + 1]] <- fold_results
      }

      # Combine all results at once
      if (length(all_combo_results) > 0) {
        all_results_df <- do.call(rbind, all_combo_results)

        # Aggregate across inner folds
        aggregated_results <- all_results_df %>%
          dplyr::group_by(outer_fold, weights, type, class) %>%
          dplyr::summarise(
            mean_sensitivity = mean(sensitivity, na.rm = TRUE),
            mean_specificity = mean(specificity, na.rm = TRUE),
            mean_balanced_accuracy = mean(balanced_accuracy, na.rm = TRUE),
            mean_f1_score = mean(f1_score, na.rm = TRUE),
            mean_prevalence = mean(prevalence, na.rm = TRUE),
            sd_sensitivity = sd(sensitivity, na.rm = TRUE),
            sd_specificity = sd(specificity, na.rm = TRUE),
            sd_balanced_accuracy = sd(balanced_accuracy, na.rm = TRUE),
            sd_f1_score = sd(f1_score, na.rm = TRUE),
            sd_prevalence = sd(prevalence, na.rm = TRUE),
            n_inner_folds = dplyr::n(),
            .groups = "drop"
          )
        return(aggregated_results)
      }
    } else {
      # Train/test: no inner folds
      aligned_matrices <- align_probability_matrices_cached(
        results$probability_matrices, outer_fold, NULL, type, alignment_cache
      )
      if (is.null(aligned_matrices)) return(NULL)

      # Convert to matrices once for efficiency
      prob_mat_SVM <- as.matrix(aligned_matrices$svm)
      prob_mat_XGB <- as.matrix(aligned_matrices$xgboost)
      prob_mat_NN <- as.matrix(aligned_matrices$neural_net)
      truth <- make.names(aligned_matrices$non_prob_cols$y)
      all_classes <- colnames(aligned_matrices$svm)

      # Batch evaluate all weight-class combinations at once
      combo_results <- evaluate_batch_weights_ovr(
        weights, prob_mat_SVM, prob_mat_XGB, prob_mat_NN, all_classes, truth,
        outer_fold, NA, type
      )

      if (nrow(combo_results) > 0) {
        combo_results$inner_fold <- NULL
        return(combo_results)
      }
    }
    return(NULL)
  }

  # Process outer folds
  df_list <- list()
  for (outer_fold in outer_folds) {
    cat(sprintf("  Processing outer fold %s...\n", outer_fold))
    df_list[[outer_fold]] <- process_outer_fold(outer_fold)
  }

  df_list
}

#' Evaluate rejection analysis for all probability matrices (unified)
#' @param probability_matrices List of probability matrices for all models
#' @param ensemble_matrices List of ensemble probability matrices
#' @param type Type of analysis ("cv" or "loso")
#' @param has_inner_folds Whether data has inner fold nesting
#' @return Data frame with rejection analysis results for all models and ensembles
evaluate_all_matrices_with_rejection_unified <- function(probability_matrices, ensemble_matrices, type = "cv", has_inner_folds = TRUE) {
  cat(sprintf("Performing rejection analysis (%s)...\n", ifelse(has_inner_folds, "with inner folds", "train/test")))

  # Build list of all tasks to process
  tasks <- list()

  # Collect individual model tasks
  for (model_name in names(probability_matrices)) {
    if (type %in% names(probability_matrices[[model_name]])) {
      outer_fold_matrices <- probability_matrices[[model_name]][[type]]

      for (outer_fold_name in names(outer_fold_matrices)) {
        if (has_inner_folds) {
          inner_fold_matrices <- outer_fold_matrices[[outer_fold_name]]
          for (inner_fold_name in names(inner_fold_matrices)) {
            prob_matrix <- inner_fold_matrices[[inner_fold_name]]
            if (!is.null(prob_matrix) && nrow(prob_matrix) > 0) {
              tasks[[length(tasks) + 1]] <- list(
                prob_matrix = prob_matrix,
                fold_name = paste(outer_fold_name, inner_fold_name, sep = "_"),
                model_name = model_name,
                outer_fold = outer_fold_name,
                inner_fold = inner_fold_name
              )
            }
          }
        } else {
          prob_matrix <- outer_fold_matrices[[outer_fold_name]]
          if (!is.null(prob_matrix) && nrow(prob_matrix) > 0) {
            tasks[[length(tasks) + 1]] <- list(
              prob_matrix = prob_matrix,
              fold_name = outer_fold_name,
              model_name = model_name,
              outer_fold = outer_fold_name,
              inner_fold = NA
            )
          }
        }
      }
    }
  }

  # Collect ensemble tasks
  ensemble_methods <- list(
    "OvR_Ensemble" = ensemble_matrices$ovr_optimized_ensemble_matrices,
    "Global_Optimized" = ensemble_matrices$global_optimized_ensemble_matrices
  )

  for (ensemble_name in names(ensemble_methods)) {
    ensemble_outer_fold_matrices <- ensemble_methods[[ensemble_name]]
    if (is.null(ensemble_outer_fold_matrices)) next

    for (outer_fold_name in names(ensemble_outer_fold_matrices)) {
      if (has_inner_folds) {
        inner_fold_matrices <- ensemble_outer_fold_matrices[[outer_fold_name]]
        for (inner_fold_name in names(inner_fold_matrices)) {
          prob_matrix <- inner_fold_matrices[[inner_fold_name]]
          if (!is.null(prob_matrix) && nrow(prob_matrix) > 0) {
            tasks[[length(tasks) + 1]] <- list(
              prob_matrix = prob_matrix,
              fold_name = paste(outer_fold_name, inner_fold_name, sep = "_"),
              model_name = ensemble_name,
              outer_fold = outer_fold_name,
              inner_fold = inner_fold_name
            )
          }
        }
      } else {
        prob_matrix <- ensemble_outer_fold_matrices[[outer_fold_name]]
        if (!is.null(prob_matrix) && nrow(prob_matrix) > 0) {
          tasks[[length(tasks) + 1]] <- list(
            prob_matrix = prob_matrix,
            fold_name = outer_fold_name,
            model_name = ensemble_name,
            outer_fold = outer_fold_name,
            inner_fold = NA
          )
        }
      }
    }
  }

  cat(sprintf("  Processing %d fold-model combinations...\n", length(tasks)))

  # Process all tasks
  results_list <- lapply(tasks, function(task) {
    result <- evaluate_single_matrix_with_rejection_parallel(
      task$prob_matrix, task$fold_name, task$model_name, type
    )
    result$outer_fold <- task$outer_fold
    result$inner_fold <- task$inner_fold
    result
  })

  # Combine all results
  do.call(rbind, results_list)
}

# =============================================================================
# Class Merging Functions
# =============================================================================

#' Merge classes in probability matrix by using max probability (not sum)
#' Merges:
#' 1. All classes containing "MDS" or "TP53" (case-insensitive) -> "MDS.r"
#' 2. If merge_mds_only is FALSE: All classes containing "KMT2A" but not "MLLT3" (case-insensitive) -> "other.KMT2A"
#' Uses max probability among merged classes instead of summing them
#' @param prob_matrix Probability matrix data frame with class columns
#' @param non_prob_cols Vector of column names that are not probability columns (e.g., "y", "outer_fold", etc.)
#' @param merge_mds_only If TRUE, only merge MDS classes and keep KMT2A classes separate (default: FALSE)
#' @return Modified probability matrix with merged classes
merge_probability_matrix_classes <- function(prob_matrix, non_prob_cols = c("y", "inner_fold", "outer_fold", "indices", "study", "sample_indices"), merge_mds_only = FALSE) {
  # Get all column names
  all_cols <- colnames(prob_matrix)

  # Identify probability columns (exclude non-probability columns)
  prob_cols <- all_cols[!all_cols %in% non_prob_cols]

  # Identify classes to merge for MDS/TP53
  mds_classes <- character(0)
  for (col in prob_cols) {
    col_lower <- tolower(col)
    if ((grepl("mds", col_lower) || grepl("tp53", col_lower)) ) {
      mds_classes <- c(mds_classes, col)
    }
  }

  # Identify classes to merge for other KMT2A (excluding MLLT3)
  other_kmt2a_classes <- character(0)
  for (col in prob_cols) {
    col_lower <- tolower(col)
    if (grepl("kmt2a", col_lower) && !grepl("mllt3", col_lower)) {
      other_kmt2a_classes <- c(other_kmt2a_classes, col)
    }
  }

  # Create a copy of the matrix
  merged_matrix <- prob_matrix

  # Merge MDS/TP53 classes - use max probability instead of sum
  if (length(mds_classes) > 0) {
    cat(sprintf("    Merging %d classes to MDS.r (max prob method): %s\n",
                length(mds_classes),
                paste(mds_classes, collapse = ", ")))

    # Use max probability for MDS/TP53 classes (not sum)
    merged_matrix$MDS.r <- apply(merged_matrix[, mds_classes, drop = FALSE], 1, max, na.rm = TRUE)

    # Remove individual classes
    merged_matrix <- merged_matrix[, !colnames(merged_matrix) %in% mds_classes, drop = FALSE]
  }

  # Merge other KMT2A classes (only if merge_mds_only is FALSE) - use max probability instead of sum
  if (!merge_mds_only && length(other_kmt2a_classes) > 0) {
    cat(sprintf("    Merging %d classes to other.KMT2A (max prob method): %s\n",
                length(other_kmt2a_classes),
                paste(other_kmt2a_classes, collapse = ", ")))

    # Use max probability for other KMT2A classes (not sum)
    merged_matrix$other.KMT2A <- apply(merged_matrix[, other_kmt2a_classes, drop = FALSE], 1, max, na.rm = TRUE)

    # Remove individual classes
    merged_matrix <- merged_matrix[, !colnames(merged_matrix) %in% other_kmt2a_classes, drop = FALSE]
  }

  # Normalize probabilities to sum to 1 for each sample (only probability columns)
  prob_cols_merged <- colnames(merged_matrix)[!colnames(merged_matrix) %in% non_prob_cols]
  if (length(prob_cols_merged) > 0) {
    prob_sums <- rowSums(merged_matrix[, prob_cols_merged, drop = FALSE], na.rm = TRUE)
    prob_sums[prob_sums == 0] <- 1  # Avoid division by zero
    for (col in prob_cols_merged) {
      merged_matrix[[col]] <- merged_matrix[[col]] / prob_sums
    }
  }

  return(merged_matrix)
}

#' Merge true labels to match merged class structure
#' @param true_labels Vector of true labels (character or factor)
#' @param merge_mds_only If TRUE, only merge MDS classes and keep KMT2A classes separate (default: FALSE)
#' @return Vector of merged true labels
merge_true_labels <- function(true_labels, merge_mds_only = FALSE) {
  # Convert to character if factor
  if (is.factor(true_labels)) {
    true_labels <- as.character(true_labels)
  }

  # Create a copy
  merged_labels <- true_labels

  # Merge MDS/TP53 labels
  merged_labels[grepl("MDS|TP53", merged_labels, ignore.case = TRUE)] <- "MDS.r"

  # Merge other KMT2A labels (excluding MLLT3) only if merge_mds_only is FALSE
  if (!merge_mds_only) {
    merged_labels[grepl("KMT2A", merged_labels, ignore.case = TRUE) &
                  !grepl("MLLT3", merged_labels, ignore.case = TRUE)] <- "other.KMT2A"
  }

  # Convert to make.names format for consistency
  merged_labels <- make.names(merged_labels)

  return(merged_labels)
}

#' Apply class merging to a probability matrix and its true labels
#' @param prob_matrix Probability matrix data frame
#' @param non_prob_cols Vector of column names that are not probability columns
#' @param merge_mds_only If TRUE, only merge MDS classes and keep KMT2A classes separate (default: FALSE)
#' @return Modified probability matrix with merged classes and updated true labels
merge_classes_in_matrix <- function(prob_matrix, non_prob_cols = c("y", "inner_fold", "outer_fold", "indices", "study", "sample_indices"), merge_mds_only = FALSE) {
  # Merge probability matrix classes
  merged_matrix <- merge_probability_matrix_classes(prob_matrix, non_prob_cols, merge_mds_only)

  # Merge true labels if present
  if ("y" %in% colnames(merged_matrix)) {
    merged_matrix$y <- merge_true_labels(merged_matrix$y, merge_mds_only)
  }

  return(merged_matrix)
}
