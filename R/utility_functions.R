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

KNN_DISTANCE_COLUMNS <- c(
  "knn10_mean_d", "knn10_min_d", "knn10_q90_d",
  "knn20_mean_d", "knn20_min_d", "knn20_q90_d"
)

REJECT_OPTION_EXTRA_FEATURE_COLUMNS <- c(
  "trust_ratio_knn10",
  "conformal_set_size_90"
)

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
  # DNN as main fallback (best overall performer)
  ENSEMBLE_WEIGHTS[["ALL"]] <- list(SVM = 0, XGB = 0, NN = 1)

  return(ENSEMBLE_WEIGHTS)
}

#' Product-of-experts ensemble: p(class) ∝ Π_m p_m(class)^{w_m}
#' @param weights Named list with SVM, XGB, NN exponents (sum to 1 on the weight grid)
product_of_experts_probs <- function(prob_mat_SVM, prob_mat_XGB, prob_mat_NN, weights, eps = 1e-12) {
  poe <- (pmax(prob_mat_SVM, eps) ^ weights$SVM) *
    (pmax(prob_mat_XGB, eps) ^ weights$XGB) *
    (pmax(prob_mat_NN, eps) ^ weights$NN)
  row_sums <- rowSums(poe)
  row_sums[row_sums == 0] <- 1
  poe / row_sums
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

    # Diagnose index consistency before alignment.
    svm_unique <- unique(svm_samples)
    xgb_unique <- unique(xgb_samples)
    nn_unique <- unique(nn_samples)
    common_samples <- Reduce(intersect, list(svm_unique, xgb_unique, nn_unique))
    union_samples <- Reduce(union, list(svm_unique, xgb_unique, nn_unique))

    svm_dup_n <- sum(duplicated(svm_samples))
    xgb_dup_n <- sum(duplicated(xgb_samples))
    nn_dup_n <- sum(duplicated(nn_samples))

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

    # Log if samples were dropped, and explain whether this is due to
    # duplicates, set mismatch across models, or both.
    max_orig <- max(n_svm_orig, n_xgb_orig, n_nn_orig)
    n_dropped <- max_orig - length(common_samples)
    if (n_dropped > 0) {
      fold_desc <- if (!is.null(inner_fold_name)) paste(outer_fold_name, inner_fold_name, sep = "_") else outer_fold_name
      cat(sprintf("    Aligned samples for fold %s: dropped %d samples to match across models (SVM: %d, XGB: %d, NN: %d -> common: %d)\n",
                  fold_desc, n_dropped, n_svm_orig, n_xgb_orig, n_nn_orig, length(common_samples)))

      non_common_svm <- setdiff(svm_unique, common_samples)
      non_common_xgb <- setdiff(xgb_unique, common_samples)
      non_common_nn <- setdiff(nn_unique, common_samples)
      cat(sprintf(
        "      Index diagnostics for fold %s: unique(SVM/XGB/NN)=(%d/%d/%d), duplicates(SVM/XGB/NN)=(%d/%d/%d), non-common unique indices (SVM/XGB/NN)=(%d/%d/%d), union=%d\n",
        fold_desc,
        length(svm_unique), length(xgb_unique), length(nn_unique),
        svm_dup_n, xgb_dup_n, nn_dup_n,
        length(non_common_svm), length(non_common_xgb), length(non_common_nn),
        length(union_samples)
      ))

      if (length(non_common_svm) > 0) {
        cat(sprintf("      Example SVM-only/non-common indices: %s\n",
                    paste(head(sort(non_common_svm), 10), collapse = ", ")))
      }
      if (length(non_common_xgb) > 0) {
        cat(sprintf("      Example XGB-only/non-common indices: %s\n",
                    paste(head(sort(non_common_xgb), 10), collapse = ", ")))
      }
      if (length(non_common_nn) > 0) {
        cat(sprintf("      Example NN-only/non-common indices: %s\n",
                    paste(head(sort(non_common_nn), 10), collapse = ", ")))
      }
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

  # Sample-level metadata columns we want to preserve through alignment.
  # is_leftout is included so left-out-aware pipelines can distinguish
  # known vs left-out rows after alignment.
  # Keep all sample-level metadata out of probability columns.
  # `sample_indices` is critical in left-out-aware paths and must never be
  # interpreted as a class probability feature.
  meta_col_names <- c(
    "y", "inner_fold", "outer_fold", "indices", "sample_indices", "study", "is_leftout",
    "n_models_agree", "top1_prob_variance_across_models",
    KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS
  )

  non_prob_cols <- svm_matrix[, colnames(svm_matrix) %in% meta_col_names, drop = FALSE]

  svm_matrix <- svm_matrix[, !colnames(svm_matrix) %in% meta_col_names, drop = FALSE]
  xgb_matrix <- xgb_matrix[, !colnames(xgb_matrix) %in% meta_col_names, drop = FALSE]
  nn_matrix <- nn_matrix[, !colnames(nn_matrix) %in% meta_col_names, drop = FALSE]

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

#' Compute correctness for rejection analysis.
#' Left-out/OOD rows are never granted subtype-collapse correctness exceptions.
#' @param prob_matrix Probability matrix with metadata columns.
#' @param truth Character vector of true labels (cleaned).
#' @param preds Character vector of predicted labels (cleaned).
#' @param prob_cols Character vector of probability column names used for prediction.
#' @return Integer vector (0/1) indicating correctness.
compute_rejection_correctness <- function(prob_matrix, truth, preds, prob_cols) {
  as.integer(truth == preds)
}

#' Vectorized cutoff analysis - much faster than per-cutoff evaluation
#' Uses sorted probabilities and cumulative calculations
#' @param prob_matrix Probability matrix with class probabilities and true labels
#' @param fold_name Name of the fold being analyzed
#' @param model_name Name of the model being analyzed
#' @param type Type of analysis ("cv" or "loso")
#' @param cutoff_step Step size for probability cutoffs
#' @return Data frame with rejection analysis results
evaluate_single_matrix_with_rejection_vectorized <- function(prob_matrix, fold_name, model_name, type, cutoff_step = 0.01) {
  # Non-probability columns to exclude (sample_indices would otherwise be used as "prob" and break rejection)
  meta_cols <- c("y", "inner_fold", "outer_fold", "indices", "study", "sample_indices",
                 "confidence_multivariate",
                 "confidence_id", "confidence_correct", "confidence_two_head",
                 "confidence_seen_new_cohort", "confidence_unseen", "confidence_three_head",
                 "confidence_two_head_postcal",
                 "confidence_two_head_min_gate", "confidence_two_head_id_veto",
                 "is_leftout", "n_models_agree",
                 "top1_prob_variance_across_models", KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS)
  prob_matrix_clean <- prob_matrix[, !colnames(prob_matrix) %in% meta_cols, drop = FALSE]

  # Extract true labels
  truth <- prob_matrix$y

  # Clean class labels
  truth <- gsub("Class. ", "", truth)

  # Vectorized: Get predictions using max.col (much faster than apply)
  prob_mat <- as.matrix(prob_matrix_clean)
  pred_indices <- max.col(prob_mat, ties.method = "first")
  preds <- colnames(prob_matrix_clean)[pred_indices]
  preds <- gsub("Class. ", "", preds)

  # Use multivariate confidence when present, else max predicted class probability
  if ("confidence_multivariate" %in% colnames(prob_matrix)) {
    confidence_vals <- prob_matrix$confidence_multivariate
  } else {
    confidence_vals <- prob_mat[cbind(seq_len(nrow(prob_mat)), pred_indices)]
  }

  # Ensure all classes are represented
  all_classes <- unique(c(truth, preds))
  truth <- factor(truth, levels = all_classes)
  preds <- factor(preds, levels = all_classes)

  # Pre-compute correctness with strict label equality (no OOD overrides).
  correct <- compute_rejection_correctness(
    prob_matrix = prob_matrix,
    truth = as.character(truth),
    preds = as.character(preds),
    prob_cols = colnames(prob_matrix_clean)
  )
  total_samples <- length(truth)

  # Sort by confidence for efficient cumulative processing
  sort_order <- order(confidence_vals)
  sorted_probs <- confidence_vals[sort_order]
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
#' @param cutoff_step Step size for probability cutoffs (default: 0.05)
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
  prob_mat <- product_of_experts_probs(
    prob_mat_SVM, prob_mat_XGB, prob_mat_NN, weight_config
  )

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
  # Pre-compute PoE ensemble matrices for every weight grid point
  weighted_matrices <- lapply(weights, function(w) {
    product_of_experts_probs(prob_mat_SVM, prob_mat_XGB, prob_mat_NN, w)
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
  cat(sprintf(
    "Performing global ensemble analysis (product-of-experts, %s)...\n",
    ifelse(has_inner_folds, "with inner folds", "train/test")
  ))

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

# =============================================================================
# Nested fold pooling helpers (legacy single-head Platt was removed; no confidence_calibrated column)
# =============================================================================

#' Extract max probability and correctness per row from a probability matrix
#' @param prob_matrix Data frame with class prob columns and "y" (true label)
#' @return List with max_prob numeric vector and correct integer vector (0/1)
get_max_prob_and_correct_from_matrix <- function(prob_matrix) {
  meta_cols <- c("y", "inner_fold", "outer_fold", "indices", "study", "sample_indices",
                 "confidence_multivariate",
                 "confidence_id", "confidence_correct", "confidence_two_head",
                 "confidence_seen_new_cohort", "confidence_unseen", "confidence_three_head",
                 "confidence_two_head_postcal",
                 "confidence_two_head_min_gate", "confidence_two_head_id_veto",
                 "is_leftout", "n_models_agree",
                 "top1_prob_variance_across_models", KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS)
  prob_cols <- colnames(prob_matrix)[!colnames(prob_matrix) %in% meta_cols]
  prob_mat <- as.matrix(prob_matrix[, prob_cols, drop = FALSE])
  pred_indices <- max.col(prob_mat, ties.method = "first")
  max_prob <- prob_mat[cbind(seq_len(nrow(prob_mat)), pred_indices)]
  truth <- gsub("Class\\. ", "", prob_matrix$y)
  preds <- gsub("Class\\. ", "", prob_cols[pred_indices])
  correct <- compute_rejection_correctness(
    prob_matrix = prob_matrix,
    truth = as.character(truth),
    preds = as.character(preds),
    prob_cols = prob_cols
  )
  list(max_prob = max_prob, correct = correct)
}

#' Legacy hook: nested single-head Platt previously wrote `confidence_calibrated` (removed).
#' @return target_matrix unchanged
apply_platt_to_target_from_pool <- function(pool_matrices, target_matrix, use_logit = FALSE, eps = 1e-6) {
  target_matrix
}


#' Extract rich rejection features from a probability matrix (for multivariate calibration).
#' Returns max_prob, margin (top1 - top2), entropy, model top-1 variance, predicted class, and correctness.
#' @param prob_matrix Data frame with class prob columns and "y" (true label)
#' @return Data frame with one row per sample: max_prob, margin, entropy, top1_prob_variance_across_models, pred_class, correct
get_rejection_features_from_matrix <- function(prob_matrix) {
  meta_cols <- c("y", "inner_fold", "outer_fold", "indices", "study",
                 "sample_indices", "confidence_multivariate",
                 "confidence_id", "confidence_correct", "confidence_two_head",
                 "confidence_seen_new_cohort", "confidence_unseen", "confidence_three_head",
                 "confidence_two_head_postcal",
                 "confidence_two_head_min_gate", "confidence_two_head_id_veto",
                 "is_leftout", "n_models_agree",
                 "top1_prob_variance_across_models", KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS)
  prob_cols <- colnames(prob_matrix)[!colnames(prob_matrix) %in% meta_cols]
  prob_mat <- as.matrix(prob_matrix[, prob_cols, drop = FALSE])
  n <- nrow(prob_mat)

  # Top-1 prediction
  pred_indices <- max.col(prob_mat, ties.method = "first")
  max_prob <- prob_mat[cbind(seq_len(n), pred_indices)]
  pred_class <- gsub("Class\\. ", "", prob_cols[pred_indices])

  # Top-2 for margin: set top-1 to -Inf and find next max
  prob_mat_mod <- prob_mat
  prob_mat_mod[cbind(seq_len(n), pred_indices)] <- -Inf
  second_prob <- prob_mat_mod[cbind(seq_len(n), max.col(prob_mat_mod, ties.method = "first"))]
  margin <- max_prob - second_prob

  # Normalized entropy captures distributional uncertainty beyond top-1/top-2.
  prob_clipped <- pmax(prob_mat, 1e-12)
  n_classes <- ncol(prob_clipped)
  if (n_classes > 1) {
    entropy <- -rowSums(prob_clipped * log(prob_clipped)) / log(n_classes)
    entropy <- pmin(1, pmax(0, entropy))
  } else {
    entropy <- rep(0, nrow(prob_clipped))
  }

  # Ensemble disagreement feature from upstream alignment step.
  # If absent (non-ensemble path), keep as NA and formula builder will skip it.
  top1_prob_variance_across_models <- if ("top1_prob_variance_across_models" %in% colnames(prob_matrix)) {
    as.numeric(prob_matrix$top1_prob_variance_across_models)
  } else {
    rep(NA_real_, n)
  }

  # Correctness
  truth <- gsub("Class\\. ", "", prob_matrix$y)
  correct <- compute_rejection_correctness(
    prob_matrix = prob_matrix,
    truth = as.character(truth),
    preds = as.character(pred_class),
    prob_cols = prob_cols
  )

  data.frame(
    max_prob = max_prob,
    margin = margin,
    entropy = entropy,
    top1_prob_variance_across_models = top1_prob_variance_across_models,
    trust_ratio_knn10 = if ("trust_ratio_knn10" %in% colnames(prob_matrix)) as.numeric(prob_matrix$trust_ratio_knn10) else NA_real_,
    conformal_set_size_90 = if ("conformal_set_size_90" %in% colnames(prob_matrix)) as.numeric(prob_matrix$conformal_set_size_90) else NA_real_,
    knn10_mean_d = if ("knn10_mean_d" %in% colnames(prob_matrix)) as.numeric(prob_matrix$knn10_mean_d) else NA_real_,
    knn10_min_d = if ("knn10_min_d" %in% colnames(prob_matrix)) as.numeric(prob_matrix$knn10_min_d) else NA_real_,
    knn10_q90_d = if ("knn10_q90_d" %in% colnames(prob_matrix)) as.numeric(prob_matrix$knn10_q90_d) else NA_real_,
    knn20_mean_d = if ("knn20_mean_d" %in% colnames(prob_matrix)) as.numeric(prob_matrix$knn20_mean_d) else NA_real_,
    knn20_min_d = if ("knn20_min_d" %in% colnames(prob_matrix)) as.numeric(prob_matrix$knn20_min_d) else NA_real_,
    knn20_q90_d = if ("knn20_q90_d" %in% colnames(prob_matrix)) as.numeric(prob_matrix$knn20_q90_d) else NA_real_,
    pred_class = pred_class,
    correct = correct,
    stringsAsFactors = FALSE
  )
}

# Deep-copy fold data.frames. Multivariate / two-head calibrators assign
# confidence_* columns in place; without copying, successive regimes that share
# the same underlying list (e.g. outer CV augmented folds) alias and overwrite
# each other's outputs.
copy_fold_matrix_list <- function(fold_list) {
  if (!is.list(fold_list)) return(fold_list)
  nms <- names(fold_list)
  out <- lapply(fold_list, function(m) {
    if (inherits(m, "data.frame")) as.data.frame(m) else m
  })
  names(out) <- nms
  out
}


#' Fit multivariate Platt calibration on pool and apply to target (ensemble only).
#' Uses max_prob + margin + entropy + n_models_agree + top1_prob_variance_across_models as features.
#' Strict mode: all required predictors must be present and finite.
#' @param pool_matrices List of probability matrices to pool for fitting
#' @param target_matrix Single matrix to calibrate
#' @return Target matrix with confidence_multivariate column added
apply_multivariate_platt_to_target_from_pool <- function(pool_matrices, target_matrix,
                                                          use_logit_max_prob = FALSE,
                                                          knn_k = NULL,
                                                          eps = 1e-6) {
  # Pool features from all pool matrices
  pool_features <- list()
  for (m in pool_matrices) {
    if (is.null(m) || nrow(m) == 0) next
    feats <- get_rejection_features_from_matrix(m)
    # Add disagreement features if present on the matrix
    if ("n_models_agree" %in% colnames(m)) {
      feats$n_models_agree <- m$n_models_agree
    }
    pool_features[[length(pool_features) + 1]] <- feats
  }

  if (length(pool_features) == 0) {
    stop("Multivariate calibration failed: empty pooled feature set.")
  }
  pool_df <- do.call(rbind, pool_features)

  if (nrow(pool_df) < 10L || length(unique(pool_df$correct)) < 2L) {
    stop("Multivariate calibration failed: pooled data has <10 rows or no class variation in correctness.")
  }

  if (use_logit_max_prob) {
    pool_df$logit_max_prob <- qlogis(pmin(1 - eps, pmax(eps, pool_df$max_prob)))
  }
  required_terms <- c("max_prob", "margin", "entropy", "n_models_agree", "top1_prob_variance_across_models")
  if (!is.null(knn_k)) {
    required_terms <- c(
      required_terms,
      sprintf("knn%d_mean_d", knn_k),
      sprintf("knn%d_min_d", knn_k),
      sprintf("knn%d_q90_d", knn_k)
    )
  }
  if (use_logit_max_prob) {
    required_terms[required_terms == "max_prob"] <- "logit_max_prob"
  }
  missing_terms <- required_terms[!required_terms %in% colnames(pool_df)]
  if (length(missing_terms) > 0) {
    stop(sprintf("Multivariate calibration failed: missing pooled predictors: %s", paste(missing_terms, collapse = ", ")))
  }
  nonfinite_terms <- required_terms[!sapply(required_terms, function(v) all(is.finite(pool_df[[v]])))]
  if (length(nonfinite_terms) > 0) {
    stop(sprintf("Multivariate calibration failed: non-finite pooled predictor values in: %s", paste(nonfinite_terms, collapse = ", ")))
  }
  formula <- stats::as.formula(paste("correct ~", paste(required_terms, collapse = " + ")))

  fit <- tryCatch({
    stats::glm(formula, data = pool_df, family = stats::binomial)
  }, error = function(e) {
    stop(sprintf("Multivariate calibration failed during GLM fit: %s", conditionMessage(e)))
  })

  # Extract and validate target features.
  target_feats <- get_rejection_features_from_matrix(target_matrix)
  if ("n_models_agree" %in% colnames(target_matrix)) {
    target_feats$n_models_agree <- target_matrix$n_models_agree
  }
  if (use_logit_max_prob) {
    target_feats$logit_max_prob <- qlogis(pmin(1 - eps, pmax(eps, target_feats$max_prob)))
  }
  target_missing <- required_terms[!required_terms %in% colnames(target_feats)]
  if (length(target_missing) > 0) {
    stop(sprintf("Multivariate calibration failed: missing target predictors: %s", paste(target_missing, collapse = ", ")))
  }
  target_nonfinite <- required_terms[!sapply(required_terms, function(v) all(is.finite(target_feats[[v]])))]
  if (length(target_nonfinite) > 0) {
    stop(sprintf("Multivariate calibration failed: non-finite target predictor values in: %s", paste(target_nonfinite, collapse = ", ")))
  }

  calibrated <- tryCatch({
    as.numeric(stats::predict(fit, newdata = target_feats, type = "response"))
  }, error = function(e) {
    stop(sprintf("Multivariate calibration failed during prediction: %s", conditionMessage(e)))
  })

  target_matrix$confidence_multivariate <- calibrated
  target_matrix
}


#' Fit a two-head calibrator (OOD head + correctness head) on a leave-one-fold-out
#' pool and apply it to the target matrix.
#'
#' Two complementary logistic models are fit on the pooled rows:
#'   * correctness head — P(correct | features), fit on in-distribution rows only
#'     (is_leftout == FALSE). This is the same quantity as the existing
#'     multivariate Platt calibrator.
#'   * OOD head — P(is_id | features), fit on the full pool (known + leftout),
#'     optionally with inverse-class-frequency weights to counter the leftout
#'     minority. Captures "does this sample look in-distribution at all?"
#'
#' The target matrix gains three columns:
#'   * confidence_id         — P(is_id)
#'   * confidence_correct    — P(correct | ID)
#'   * confidence_two_head   — product (default scalar accept score)
#' When `confidence_col` differs from `confidence_two_head`, that slot is set to the same product
#' (e.g. multivariate bundles overwrite `confidence_multivariate`).
#'
#' @param pool_matrices List of probability matrices (must carry is_leftout)
#' @param target_matrix Single matrix to calibrate
#' @param use_multivariate If TRUE (default), both heads are fit with the full
#'   feature set (max_prob, margin, entropy, n_models_agree,
#'   top1_prob_variance). If FALSE, univariate with max_prob only — mirrors the
#'   simple Platt path for symmetry.
#' @param class_balanced_ood If TRUE (default), the OOD head is fit with inverse
#'   class-frequency weights to counter left-out scarcity.
#' @param confidence_col Which slot to overwrite with the scalar score (typically
#'   "confidence_multivariate"). Use "confidence_two_head" only when it must alias the product.
#' @return Target matrix with new columns.
apply_two_head_calibration_to_target_from_pool <- function(pool_matrices, target_matrix,
                                                             use_multivariate = TRUE,
                                                             knn_k = NULL,
                                                             class_balanced_ood = TRUE,
                                                             confidence_col = "confidence_multivariate") {
  # 1. Build pooled feature frame with both labels (correct and is_id).
  pool_features <- list()
  for (m in pool_matrices) {
    if (is.null(m) || nrow(m) == 0) next
    feats <- get_rejection_features_from_matrix(m)
    if ("n_models_agree" %in% colnames(m)) feats$n_models_agree <- m$n_models_agree
    feats$is_id <- if ("is_leftout" %in% colnames(m)) {
      as.integer(!as.logical(m$is_leftout))
    } else {
      rep(1L, nrow(m))
    }
    pool_features[[length(pool_features) + 1]] <- feats
  }
  if (length(pool_features) == 0) {
    stop("Two-head calibration failed: empty pooled feature set.")
  }
  pool_df <- do.call(rbind, pool_features)

  # 2. Pick feature set and validate strictly.
  required_terms <- if (use_multivariate) {
    c("max_prob", "margin", "entropy", "n_models_agree", "top1_prob_variance_across_models")
  } else {
    c("max_prob")
  }
  if (!is.null(knn_k)) {
    required_terms <- c(
      required_terms,
      sprintf("knn%d_mean_d", knn_k),
      sprintf("knn%d_min_d", knn_k),
      sprintf("knn%d_q90_d", knn_k)
    )
  }
  missing_terms <- required_terms[!required_terms %in% colnames(pool_df)]
  if (length(missing_terms) > 0) {
    stop(sprintf("Two-head calibration failed: missing pooled predictors: %s", paste(missing_terms, collapse = ", ")))
  }
  nonfinite_terms <- required_terms[!sapply(required_terms, function(v) all(is.finite(pool_df[[v]])))]
  if (length(nonfinite_terms) > 0) {
    stop(sprintf("Two-head calibration failed: non-finite pooled predictor values in: %s", paste(nonfinite_terms, collapse = ", ")))
  }
  rhs <- paste(required_terms, collapse = " + ")

  # 3. Correctness head — same target as standard multivariate Platt, fit on
  #    known-only rows so ID calibration isn't distorted by OOD.
  correct_pool <- pool_df[pool_df$is_id == 1L, , drop = FALSE]
  if (nrow(correct_pool) < 10L || length(unique(correct_pool$correct)) < 2L) {
    stop("Two-head calibration failed: correctness head pool has <10 rows or no class variation.")
  }
  fit_correct <- tryCatch({
    stats::glm(stats::as.formula(paste("correct ~", rhs)),
               data = correct_pool,
               family = stats::binomial(),
               control = stats::glm.control(maxit = 200, epsilon = 1e-8))
  }, error = function(e) {
    stop(sprintf("Two-head calibration failed during correctness-head fit: %s", conditionMessage(e)))
  })
  if (!isTRUE(fit_correct$converged)) {
    stop(
      sprintf(
        "Two-head calibration failed: correctness-head GLM did not converge (rows=%d, positives=%d, negatives=%d).",
        nrow(correct_pool),
        sum(correct_pool$correct == 1, na.rm = TRUE),
        sum(correct_pool$correct == 0, na.rm = TRUE)
      )
    )
  }

  # 4. OOD head — fit on full pool; needs both classes present.
  if (length(unique(pool_df$is_id)) < 2L) {
    stop("Two-head calibration failed: OOD head pool has no ID/OOD class variation.")
  }
  weights_vec <- NULL
  if (class_balanced_ood) {
    # Inverse class-frequency weights, rescaled so the total equals n. Each
    # class then contributes equal total weight to the loss regardless of
    # sample count.
    freq <- table(pool_df$is_id)
    per_row <- 1 / as.numeric(freq[as.character(pool_df$is_id)])
    weights_vec <- per_row * nrow(pool_df) / sum(per_row)
  }
  # Fractional weights are not valid "trial counts" for family=binomial (R warns:
  # non-integer successes). quasibinomial() fits the same mean / coefficients.
  ood_family <- if (is.null(weights_vec)) stats::binomial() else stats::quasibinomial()
  fit_ood <- tryCatch(
    if (is.null(weights_vec)) {
      stats::glm(stats::as.formula(paste("is_id ~", rhs)),
                 data = pool_df,
                 family = ood_family,
                 control = stats::glm.control(maxit = 200, epsilon = 1e-8))
    } else {
      stats::glm(stats::as.formula(paste("is_id ~", rhs)),
                 data = pool_df,
                 family = ood_family,
                 weights = weights_vec,
                 control = stats::glm.control(maxit = 200, epsilon = 1e-8))
    },
    error = function(e) stop(sprintf("Two-head calibration failed during OOD-head fit: %s", conditionMessage(e)))
  )
  if (!isTRUE(fit_ood$converged)) {
    stop(
      sprintf(
        "Two-head calibration failed: OOD-head GLM did not converge (rows=%d, ID=%d, OOD=%d).",
        nrow(pool_df),
        sum(pool_df$is_id == 1, na.rm = TRUE),
        sum(pool_df$is_id == 0, na.rm = TRUE)
      )
    )
  }

  # 5. Score target.
  target_feats <- get_rejection_features_from_matrix(target_matrix)
  if ("n_models_agree" %in% colnames(target_matrix)) {
    target_feats$n_models_agree <- target_matrix$n_models_agree
  }
  target_missing <- required_terms[!required_terms %in% colnames(target_feats)]
  if (length(target_missing) > 0) {
    stop(sprintf("Two-head calibration failed: missing target predictors: %s", paste(target_missing, collapse = ", ")))
  }
  target_nonfinite <- required_terms[!sapply(required_terms, function(v) all(is.finite(target_feats[[v]])))]
  if (length(target_nonfinite) > 0) {
    stop(sprintf("Two-head calibration failed: non-finite target predictor values in: %s", paste(target_nonfinite, collapse = ", ")))
  }
  p_correct <- tryCatch({
    as.numeric(stats::predict(fit_correct, newdata = target_feats, type = "response"))
  }, error = function(e) {
    stop(sprintf("Two-head calibration failed during correctness-head prediction: %s", conditionMessage(e)))
  })
  p_id <- tryCatch({
    as.numeric(stats::predict(fit_ood, newdata = target_feats, type = "response"))
  }, error = function(e) {
    stop(sprintf("Two-head calibration failed during OOD-head prediction: %s", conditionMessage(e)))
  })

  target_matrix$confidence_correct <- p_correct
  target_matrix$confidence_id <- p_id
  target_matrix$confidence_two_head <- p_correct * p_id
  if (confidence_col != "confidence_two_head") {
    target_matrix[[confidence_col]] <- target_matrix$confidence_two_head
  }
  target_matrix
}


#' Fit a three-head calibrator on pooled folds and apply to target.
#'
#' Heads:
#' - correctness: P(correct)
#' - seen_new_cohort proxy: P(known-but-uncertain) on ID rows (proxy target: incorrect among ID)
#' - unseen: P(unseen class / OOD) using is_leftout as supervision
#'
#' The default scalar score written to `confidence_col` is:
#'   confidence_three_head = p_correct * (1 - p_unseen)
apply_three_head_calibration_to_target_from_pool <- function(pool_matrices, target_matrix,
                                                             use_multivariate = TRUE,
                                                             knn_k = NULL,
                                                             confidence_col = "confidence_multivariate") {
  pool_features <- list()
  for (m in pool_matrices) {
    if (is.null(m) || nrow(m) == 0) next
    feats <- get_rejection_features_from_matrix(m)
    if ("n_models_agree" %in% colnames(m)) feats$n_models_agree <- m$n_models_agree
    feats$is_id <- if ("is_leftout" %in% colnames(m)) {
      as.integer(!as.logical(m$is_leftout))
    } else {
      rep(1L, nrow(m))
    }
    feats$seen_new_proxy <- as.integer(feats$is_id == 1L & feats$correct == 0L)
    feats$unseen_target <- as.integer(feats$is_id == 0L)
    pool_features[[length(pool_features) + 1]] <- feats
  }
  if (length(pool_features) == 0) {
    stop("Three-head calibration failed: empty pooled feature set.")
  }
  pool_df <- do.call(rbind, pool_features)

  required_terms <- if (use_multivariate) {
    c("max_prob", "margin", "entropy", "n_models_agree", "top1_prob_variance_across_models")
  } else {
    c("max_prob")
  }
  if (!is.null(knn_k)) {
    required_terms <- c(
      required_terms,
      sprintf("knn%d_mean_d", knn_k),
      sprintf("knn%d_min_d", knn_k),
      sprintf("knn%d_q90_d", knn_k)
    )
  }
  rhs <- paste(required_terms, collapse = " + ")

  correct_pool <- pool_df[pool_df$is_id == 1L, , drop = FALSE]
  if (nrow(correct_pool) < 10L || length(unique(correct_pool$correct)) < 2L) {
    stop("Three-head calibration failed: correctness head pool insufficient.")
  }
  fit_correct <- stats::glm(stats::as.formula(paste("correct ~", rhs)),
                            data = correct_pool,
                            family = stats::binomial(),
                            control = stats::glm.control(maxit = 200, epsilon = 1e-8))

  # Proxy head for known-but-shifted/uncertain behavior.
  if (nrow(correct_pool) < 10L || length(unique(correct_pool$seen_new_proxy)) < 2L) {
    # Fall back to a weak constant head when separation is impossible.
    fit_seen <- NULL
    seen_const <- mean(correct_pool$seen_new_proxy, na.rm = TRUE)
  } else {
    fit_seen <- stats::glm(stats::as.formula(paste("seen_new_proxy ~", rhs)),
                           data = correct_pool,
                           family = stats::binomial(),
                           control = stats::glm.control(maxit = 200, epsilon = 1e-8))
    seen_const <- NULL
  }

  if (length(unique(pool_df$unseen_target)) < 2L) {
    stop("Three-head calibration failed: unseen head has no class variation.")
  }
  fit_unseen <- stats::glm(stats::as.formula(paste("unseen_target ~", rhs)),
                           data = pool_df,
                           family = stats::binomial(),
                           control = stats::glm.control(maxit = 200, epsilon = 1e-8))

  target_feats <- get_rejection_features_from_matrix(target_matrix)
  if ("n_models_agree" %in% colnames(target_matrix)) {
    target_feats$n_models_agree <- target_matrix$n_models_agree
  }

  p_correct <- as.numeric(stats::predict(fit_correct, newdata = target_feats, type = "response"))
  p_unseen <- as.numeric(stats::predict(fit_unseen, newdata = target_feats, type = "response"))
  p_seen <- if (is.null(fit_seen)) {
    rep(seen_const, nrow(target_feats))
  } else {
    as.numeric(stats::predict(fit_seen, newdata = target_feats, type = "response"))
  }

  clamp01 <- function(x) pmax(0, pmin(1, as.numeric(x)))
  p_correct <- clamp01(p_correct)
  p_unseen <- clamp01(p_unseen)
  p_seen <- clamp01(p_seen)

  target_matrix$confidence_correct <- p_correct
  target_matrix$confidence_seen_new_cohort <- p_seen
  target_matrix$confidence_unseen <- p_unseen
  target_matrix$confidence_three_head <- p_correct * (1 - p_unseen)
  if (confidence_col != "confidence_three_head") {
    target_matrix[[confidence_col]] <- target_matrix$confidence_three_head
  }
  target_matrix
}


#' Leave-one-outer-fold-out Platt / two-head calibration on augmented (known + left-out) matrices.
#' For each held-out fold, fits only on other folds (and for `known_only`, only non-left-out rows
#' in those folds). Matches outer CV `with_leftout` / `with_leftout_ood_aware` / `with_leftout_two_head`
#' univariate paths.
#' @param calibration_mode known_only | ood_aware | two_head
apply_platt_to_augmented_fold_matrices <- function(fold_matrices,
                                                    calibration_mode = c("known_only", "known_only_logit",
                                                                         "ood_aware", "ood_aware_logit",
                                                                         "two_head", "three_head")) {
  calibration_mode <- match.arg(calibration_mode)
  if (!is.list(fold_matrices) || length(fold_matrices) < 2L) return(fold_matrices)
  fold_matrices <- copy_fold_matrix_list(fold_matrices)
  fold_names <- names(fold_matrices)
  result <- list()

  for (k in seq_along(fold_names)) {
    target <- fold_matrices[[fold_names[k]]]
    others <- fold_matrices[setdiff(fold_names, fold_names[k])]

    result[[fold_names[k]]] <- switch(calibration_mode,
      known_only = {
        pool <- lapply(others, function(m) {
          if ("is_leftout" %in% colnames(m)) {
            m[!as.logical(m$is_leftout), , drop = FALSE]
          } else {
            m
          }
        })
        apply_platt_to_target_from_pool(pool, target)
      },
      known_only_logit = {
        pool <- lapply(others, function(m) {
          if ("is_leftout" %in% colnames(m)) {
            m[!as.logical(m$is_leftout), , drop = FALSE]
          } else {
            m
          }
        })
        apply_platt_to_target_from_pool(pool, target, use_logit = TRUE)
      },
      ood_aware = apply_platt_to_target_from_pool(others, target),
      ood_aware_logit = apply_platt_to_target_from_pool(others, target, use_logit = TRUE),
      two_head = apply_two_head_calibration_to_target_from_pool(
        others, target,
        use_multivariate = FALSE,
        confidence_col = "confidence_two_head"
      ),
      three_head = apply_three_head_calibration_to_target_from_pool(
        others, target,
        use_multivariate = FALSE,
        confidence_col = "confidence_three_head"
      )
    )
  }
  result
}


#' Inner-fold list helper (legacy nested Platt no longer writes columns).
#' @return Same list, unchanged
apply_platt_to_inner_fold_matrices <- function(inner_fold_matrices) {
  inner_fold_matrices
}

#' Resample rows (with replacement) of each matrix in nested probability/ensemble structure.
#' Used for bootstrap CI on risk-coverage curve. Returns a deep copy with resampled matrices.
#' @param probability_matrices Nested list: [[model]][[type]][[outer_fold]][[inner_fold]] = matrix
#' @param ensemble_matrices List with global_optimized_ensemble_matrices: [[outer_fold]][[inner_fold]] = matrix
#' @param type Analysis type ("cv" or "loso")
#' @param seed Random seed for resampling (set before calling for reproducibility)
#' @return List with probability_matrices_resampled and ensemble_matrices_resampled (new copies)
resample_rejection_matrices <- function(probability_matrices, ensemble_matrices, type, seed) {
  set.seed(seed)
  resample_df <- function(m) {
    if (!is.data.frame(m) && !is.matrix(m)) return(m)
    n <- nrow(m)
    if (n == 0) return(m)
    idx <- sample.int(n, n, replace = TRUE)
    m[idx, , drop = FALSE]
  }
  # Deep copy and resample probability matrices
  out_prob <- list()
  for (model_name in names(probability_matrices)) {
    if (!type %in% names(probability_matrices[[model_name]])) next
    out_prob[[model_name]] <- list()
    out_prob[[model_name]][[type]] <- list()
    for (outer_fold_name in names(probability_matrices[[model_name]][[type]])) {
      inner_list <- probability_matrices[[model_name]][[type]][[outer_fold_name]]
      out_prob[[model_name]][[type]][[outer_fold_name]] <- lapply(inner_list, resample_df)
    }
  }
  # Deep copy and resample ensemble matrices
  out_ens <- list(global_optimized_ensemble_matrices = NULL)
  ens_outer <- ensemble_matrices$global_optimized_ensemble_matrices
  if (!is.null(ens_outer)) {
    out_ens$global_optimized_ensemble_matrices <- list()
    for (outer_fold_name in names(ens_outer)) {
      inner_list <- ens_outer[[outer_fold_name]]
      out_ens$global_optimized_ensemble_matrices[[outer_fold_name]] <- lapply(inner_list, resample_df)
    }
  }
  list(probability_matrices = out_prob, ensemble_matrices = out_ens)
}

#' Legacy no-op batch hook (nested single-head Platt removed).
apply_platt_to_all_rejection_matrices <- function(probability_matrices, ensemble_matrices, type, has_inner_folds) {
  if (!has_inner_folds) return(invisible(NULL))
  # Individual models: probability_matrices[[model]][[type]][[outer_fold]] = list(inner_fold -> matrix)
  for (model_name in names(probability_matrices)) {
    if (!type %in% names(probability_matrices[[model_name]])) next
    outer_fold_matrices <- probability_matrices[[model_name]][[type]]
    for (outer_fold_name in names(outer_fold_matrices)) {
      inner_list <- outer_fold_matrices[[outer_fold_name]]
      calibrated_list <- apply_platt_to_inner_fold_matrices(inner_list)
      for (inner_name in names(calibrated_list)) {
        probability_matrices[[model_name]][[type]][[outer_fold_name]][[inner_name]] <- calibrated_list[[inner_name]]
      }
    }
  }
  # Ensembles: global only (OvR removed); outer_fold -> inner_fold -> matrix
  outer_list <- ensemble_matrices$global_optimized_ensemble_matrices
  if (!is.null(outer_list)) {
    for (outer_fold_name in names(outer_list)) {
      inner_list <- outer_list[[outer_fold_name]]
      if (!is.list(inner_list) || length(inner_list) == 0) next
      calibrated_list <- apply_platt_to_inner_fold_matrices(inner_list)
      for (inner_name in names(calibrated_list)) {
        ensemble_matrices$global_optimized_ensemble_matrices[[outer_fold_name]][[inner_name]] <- calibrated_list[[inner_name]]
      }
    }
  }
  invisible(NULL)
}

#' Evaluate rejection analysis for all probability matrices (unified)
#' @param probability_matrices List of probability matrices for all models
#' @param ensemble_matrices List of ensemble probability matrices
#' @param type Type of analysis ("cv" or "loso")
#' @param has_inner_folds Whether data has inner fold nesting
#' @param apply_platt Whether to apply Platt scaling (set FALSE when matrices are already calibrated, e.g. bootstrap)
#' @return Data frame with rejection analysis results for all models and ensembles
evaluate_all_matrices_with_rejection_unified <- function(probability_matrices, ensemble_matrices, type = "cv", has_inner_folds = TRUE, apply_platt = TRUE) {
  if (apply_platt) {
    cat(sprintf("Performing rejection analysis (%s)...\n", ifelse(has_inner_folds, "with inner folds", "train/test")))
    if (has_inner_folds) {
      cat("  Applying Platt scaling for confidence (out-of-sample per inner fold)...\n")
      apply_platt_to_all_rejection_matrices(probability_matrices, ensemble_matrices, type, has_inner_folds)
    }
  }

  # Build list of all tasks to process
  tasks <- list()
  task_keys <- character(0)  # Track (model, fold) to detect duplicates

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
              fold_name <- paste(outer_fold_name, inner_fold_name, sep = "_")
              key <- paste(model_name, fold_name, sep = "||")
              if (key %in% task_keys) {
                warning(sprintf(
                  "Duplicate task skipped: model=%s, fold=%s (has_inner_folds=TRUE)",
                  model_name, fold_name
                ))
              } else {
                task_keys <- c(task_keys, key)
                tasks[[length(tasks) + 1]] <- list(
                  prob_matrix = prob_matrix,
                  fold_name = fold_name,
                  model_name = model_name,
                  outer_fold = outer_fold_name,
                  inner_fold = inner_fold_name
                )
              }
            }
          }
        } else {
          prob_matrix <- outer_fold_matrices[[outer_fold_name]]
          # Only add if prob_matrix is a single matrix/data frame (not a nested list)
          is_valid <- !is.null(prob_matrix) &&
            (is.data.frame(prob_matrix) || is.matrix(prob_matrix)) &&
            nrow(prob_matrix) > 0
          if (is_valid) {
            key <- paste(model_name, outer_fold_name, sep = "||")
            if (key %in% task_keys) {
              warning(sprintf(
                "Duplicate task skipped: model=%s, fold=%s (has_inner_folds=FALSE)",
                model_name, outer_fold_name
              ))
            } else {
              task_keys <- c(task_keys, key)
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
  }

  # Collect ensemble tasks (Global only; OvR removed)
  ensemble_outer_fold_matrices <- ensemble_matrices$global_optimized_ensemble_matrices
  if (!is.null(ensemble_outer_fold_matrices)) {
    ensemble_name <- "Global_Optimized"

    for (outer_fold_name in names(ensemble_outer_fold_matrices)) {
      if (has_inner_folds) {
        inner_fold_matrices <- ensemble_outer_fold_matrices[[outer_fold_name]]
        for (inner_fold_name in names(inner_fold_matrices)) {
          prob_matrix <- inner_fold_matrices[[inner_fold_name]]
          if (!is.null(prob_matrix) && nrow(prob_matrix) > 0) {
            fold_name <- paste(outer_fold_name, inner_fold_name, sep = "_")
            key <- paste(ensemble_name, fold_name, sep = "||")
            if (key %in% task_keys) {
              warning(sprintf(
                "Duplicate ensemble task skipped: model=%s, fold=%s",
                ensemble_name, fold_name
              ))
            } else {
              task_keys <- c(task_keys, key)
              tasks[[length(tasks) + 1]] <- list(
                prob_matrix = prob_matrix,
                fold_name = fold_name,
                model_name = ensemble_name,
                outer_fold = outer_fold_name,
                inner_fold = inner_fold_name
              )
            }
          }
        }
      } else {
        prob_matrix <- ensemble_outer_fold_matrices[[outer_fold_name]]
        if (!is.null(prob_matrix) && nrow(prob_matrix) > 0) {
          key <- paste(ensemble_name, outer_fold_name, sep = "||")
          if (key %in% task_keys) {
            warning(sprintf(
              "Duplicate ensemble task skipped: model=%s, fold=%s",
              ensemble_name, outer_fold_name
            ))
          } else {
            task_keys <- c(task_keys, key)
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

##' Merge classes in probability matrix by using max probability (not sum)
##' Merges:
##' 1. All classes containing "MDS" or "TP53" (case-insensitive) -> "MDS.r"
##' 2. All classes containing "KMT2A" but not "MLLT3" (case-insensitive) -> "other.KMT2A"
##' 3. Classes representing MECOM rearrangements (e.g. "GATA2;MECOM", "MECOM other") -> "MECOM"
##'    (In reporting/figures the merged class is displayed as "MECOM rearrangement", consistent with
##'    MDS.r -> "MDS-related" and other.KMT2A -> "Other KMT2A rearrangements".)
##' Uses max probability among merged classes instead of summing them.
##' @param prob_matrix Probability matrix data frame with class columns.
##' @param non_prob_cols Vector of column names that are not probability columns (e.g., "y", "outer_fold", etc.).
##' @param merge_prob_method "max" = max probability per row among merged classes; "sum" = sum probabilities (then renormalized).
##' @return Modified probability matrix with merged classes.
merge_probability_matrix_classes <- function(prob_matrix, non_prob_cols = c("y", "inner_fold", "outer_fold", "indices", "study", "sample_indices", "confidence_multivariate", "confidence_id", "confidence_correct", "confidence_two_head", "confidence_seen_new_cohort", "confidence_unseen", "confidence_three_head", "confidence_two_head_postcal", "confidence_two_head_min_gate", "confidence_two_head_id_veto", "is_leftout", "n_models_agree", "top1_prob_variance_across_models", KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS), merge_prob_method = c("max", "sum")) {
  merge_prob_method <- match.arg(merge_prob_method)
  method_label <- if (merge_prob_method == "max") "max prob" else "summed"
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

  # Identify classes to merge for MECOM rearrangements (e.g. GATA2;MECOM and MECOM other)
  mecom_classes <- character(0)
  for (col in prob_cols) {
    col_lower <- tolower(col)
    if (grepl("mecom", col_lower) && (grepl("gata2", col_lower) || grepl("other", col_lower))) {
      mecom_classes <- c(mecom_classes, col)
    }
  }

  # Create a copy of the matrix
  merged_matrix <- prob_matrix

  merge_fun <- if (merge_prob_method == "max") {
    function(x) apply(x, 1, max, na.rm = TRUE)
  } else {
    function(x) rowSums(x, na.rm = TRUE)
  }

  # Merge MDS/TP53 classes
  if (length(mds_classes) > 0) {
    cat(sprintf("    Merging %d classes to MDS.r (%s method): %s\n",
                length(mds_classes), method_label, paste(mds_classes, collapse = ", ")))
    merged_matrix$MDS.r <- merge_fun(merged_matrix[, mds_classes, drop = FALSE])
    merged_matrix <- merged_matrix[, !colnames(merged_matrix) %in% mds_classes, drop = FALSE]
  }

  # Merge other KMT2A classes
  if (length(other_kmt2a_classes) > 0) {
    cat(sprintf("    Merging %d classes to other.KMT2A (%s method): %s\n",
                length(other_kmt2a_classes), method_label, paste(other_kmt2a_classes, collapse = ", ")))
    merged_matrix$other.KMT2A <- merge_fun(merged_matrix[, other_kmt2a_classes, drop = FALSE])
    merged_matrix <- merged_matrix[, !colnames(merged_matrix) %in% other_kmt2a_classes, drop = FALSE]
  }

  # Merge MECOM rearrangement classes
  if (length(mecom_classes) > 0) {
    cat(sprintf("    Merging %d classes to MECOM (%s method): %s\n",
                length(mecom_classes), method_label, paste(mecom_classes, collapse = ", ")))
    merged_matrix$MECOM <- merge_fun(merged_matrix[, mecom_classes, drop = FALSE])
    merged_matrix <- merged_matrix[, !colnames(merged_matrix) %in% mecom_classes, drop = FALSE]
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
#' @return Vector of merged true labels
merge_true_labels <- function(true_labels) {
  # Convert to character if factor
  if (is.factor(true_labels)) {
    true_labels <- as.character(true_labels)
  }

  # Create a copy
  merged_labels <- true_labels

  # Merge MDS/TP53 labels
  merged_labels[grepl("MDS|TP53", merged_labels, ignore.case = TRUE)] <- "MDS.r"

  # Merge other KMT2A labels (excluding MLLT3)
  merged_labels[grepl("KMT2A", merged_labels, ignore.case = TRUE) &
                !grepl("MLLT3", merged_labels, ignore.case = TRUE)] <- "other.KMT2A"

  # Merge MECOM rearrangement labels (e.g. GATA2;MECOM and MECOM other) -> "MECOM"
  # (Display name "MECOM rearrangement" is applied in analyse_results.Rmd via fix_names mapping.)
  merged_labels[grepl("MECOM", merged_labels, ignore.case = TRUE)] <- "MECOM"

  # Convert to make.names format for consistency
  merged_labels <- make.names(merged_labels)

  return(merged_labels)
}

#' Apply class merging to a probability matrix and its true labels
#' @param prob_matrix Probability matrix data frame
#' @param non_prob_cols Vector of column names that are not probability columns
#' @param merge_prob_method "max" or "sum" for merging probabilities (see merge_probability_matrix_classes).
#' @return Modified probability matrix with merged classes and updated true labels
merge_classes_in_matrix <- function(prob_matrix, non_prob_cols = c("y", "inner_fold", "outer_fold", "indices", "study", "sample_indices", "confidence_multivariate", "confidence_id", "confidence_correct", "confidence_two_head", "confidence_seen_new_cohort", "confidence_unseen", "confidence_three_head", "confidence_two_head_postcal", "confidence_two_head_min_gate", "confidence_two_head_id_veto", "is_leftout", "n_models_agree", "top1_prob_variance_across_models", KNN_DISTANCE_COLUMNS, REJECT_OPTION_EXTRA_FEATURE_COLUMNS), merge_prob_method = c("max", "sum")) {
  merge_prob_method <- match.arg(merge_prob_method)
  # Merge probability matrix classes
  merged_matrix <- merge_probability_matrix_classes(prob_matrix, non_prob_cols, merge_prob_method)

  # Merge true labels if present. Keep left-out/OOD labels uncollapsed so they
  # remain truly out-of-distribution in merged-class analyses.
  if ("y" %in% colnames(merged_matrix)) {
    if ("is_leftout" %in% colnames(merged_matrix)) {
      is_leftout <- as.logical(merged_matrix$is_leftout)
      if (length(is_leftout) != nrow(merged_matrix)) {
        is_leftout <- rep(FALSE, nrow(merged_matrix))
      }
      merged_y <- merge_true_labels(merged_matrix$y)
      merged_y[is_leftout] <- as.character(merged_matrix$y[is_leftout])
      merged_matrix$y <- merged_y
    } else {
      merged_matrix$y <- merge_true_labels(merged_matrix$y)
    }
  }

  return(merged_matrix)
}
