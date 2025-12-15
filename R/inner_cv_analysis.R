main_inner_cv <- function(){

  #' Create directory safely
  #' @param dir_path Directory path to create
  create_directory_safely <- function(dir_path) {
    if (!dir.exists(dir_path)) {
      dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
    }
  }

  load_library_quietly <- function(package_name) {
    invisible(capture.output(
      suppressMessages(
        suppressWarnings(
          library(package_name, character.only = TRUE)
        )
      )
    ))
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

    # Compute mean kappa and accuracy across inner folds for each param set
    best_parameters <- inner_cv_results %>%
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

  load_all_model_data <- function(model_configs) {
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
            results <- process_neural_net_results(results)
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

  #' Modify class labels to group related subtypes
  #' @param vector Vector of class labels
  #' @return Modified vector with grouped classes
  modify_classes <- function(vector) {
    vector[grepl("MDS|TP53|MECOM", vector)] <- "MDS.r"
    vector[!grepl("MLLT3", vector) & grepl("KMT2A", vector)] <- "other.KMT2A"
    vector
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
      dir.create(output_dir)

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

  #' Filter samples to only include those with classes present in training
  #' @param prob_matrix Probability matrix with y, inner_fold, outer_fold, and indices columns
  #' @param training_classes Vector of class labels that were in the training set
  #' @param fold_id Current fold identifier for logging
  #' @return Filtered probability matrix and statistics
  filter_samples_by_training_classes <- function(prob_matrix, training_classes, fold_id) {
    if (is.null(prob_matrix) || nrow(prob_matrix) == 0) {
      return(list(filtered_matrix = prob_matrix, stats = NULL))
    }
    
    # Get true labels
    true_labels <- prob_matrix$y
    
    # Clean class names for comparison (make.names is applied to both)
    training_classes_clean <- make.names(training_classes)
    
    # For OvR, also filter out NA labels (samples with classes not in training)
    na_mask <- !is.na(true_labels)
    valid_mask <- true_labels %in% training_classes_clean
    
    # Combine masks
    combined_mask <- na_mask & valid_mask
    
    # Calculate statistics
    n_total <- nrow(prob_matrix)
    n_filtered <- sum(!combined_mask)
    n_kept <- sum(combined_mask)
    
    # Get classes that were in test but not in training (excluding NAs)
    unseen_classes <- unique(true_labels[!combined_mask & na_mask])
    
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

  #' Generate probability data frames for One-vs-Rest classification
  #' @param cv_results_df Cross-validation results data frame
  #' @param best_params_df Best parameters data frame
  #' @param label_mapping Label mapping data frame
  #' @param filter_unseen_classes Whether to filter samples with classes not in training (default: TRUE)
  #' @return List of probability data frames organized by outer fold (and filtering statistics)
  generate_ovr_probability_matrices <- function(cv_results_df, best_params_df, label_mapping, study_names, filter_unseen_classes = TRUE) {
    best_params_with_labels <- add_class_labels(best_params_df, label_mapping)
    outer_fold_ids <- unique(cv_results_df$outer_fold)

    probability_matrices <- list()
    filtering_stats_list <- list()
    
    if (filter_unseen_classes) {
      cat("  Filtering samples with classes not present in training set...\n")
    }

    for (outer_fold_id in outer_fold_ids) {
      outer_fold_data <- cv_results_df[cv_results_df$outer_fold == outer_fold_id, ]
      inner_fold_ids <- unique(outer_fold_data$inner_fold)

      fold_matrices <- list()

      for (inner_fold_id in inner_fold_ids) {
        inner_fold_data <- outer_fold_data[outer_fold_data$inner_fold == inner_fold_id, ]
        # class_labels contains the classes that were in the training set (OvR only creates models for training classes)
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
        probability_matrix$indices <- parse_numeric_string(inner_fold_data$sample_indices[1]) + 1
        probability_matrix$study <- study_names[probability_matrix$indices]
        
        # Apply filtering if requested
        if (filter_unseen_classes) {
          fold_id <- paste(outer_fold_id, inner_fold_id, sep = "_")
          filter_result <- filter_samples_by_training_classes(
            probability_matrix, 
            class_labels,  # class_labels are the training classes for OvR
            fold_id
          )
          probability_matrix <- filter_result$filtered_matrix
          if (!is.null(filter_result$stats)) {
            filtering_stats_list[[fold_id]] <- filter_result$stats
          }
        }
        
        fold_matrices[[as.character(inner_fold_id)]] <- probability_matrix
      }

      if (length(fold_matrices) > 0) {
        probability_matrices[[as.character(outer_fold_id)]] <- fold_matrices
        #probability_matrices[[as.character(outer_fold_id)]][is.na(probability_matrices[[as.character(outer_fold_id)]])] <- 0
      }
    }

    # Return both matrices and filtering statistics
    result <- list(matrices = probability_matrices)
    if (filter_unseen_classes && length(filtering_stats_list) > 0) {
      result$filtering_stats <- do.call(rbind, filtering_stats_list)
    }
    
    return(result)
  }

  get_per_model_performance <- function(probability_matrices) {
    results_list <- list()

    for (model in names(probability_matrices)) {
      for (method in names(probability_matrices[[model]])) {
        for (outer_fold in names(probability_matrices[[model]][[method]])) {
          for (inner_fold in names(probability_matrices[[model]][[method]][[outer_fold]])) {

            the_matrix <- probability_matrices[[model]][[method]][[outer_fold]][[inner_fold]]

            # Extract truth and probability matrix
            truth <- the_matrix$y
            prob_matrix <- the_matrix[, !colnames(the_matrix) %in%
                                        c("y", "inner_fold", "outer_fold", "indices", "study"),
                                      drop = FALSE]

            # Predictions
            preds <- colnames(prob_matrix)[apply(prob_matrix, 1, which.max)]

            # Clean class labels
            truth <- gsub("Class. ", "", truth)
            preds <- gsub("Class. ", "", preds)
            truth <- modify_classes(truth)
            preds <- modify_classes(preds)

            # Ensure consistent factor levels
            all_classes <- unique(c(truth, preds))
            truth <- factor(truth, levels = all_classes)
            preds <- factor(preds, levels = all_classes)

            # Metrics
            cm <- caret::confusionMatrix(preds, truth)
            mcc <- mltools::mcc(preds, truth)

            results_list[[length(results_list) + 1]] <- data.frame(
              model = model,
              method = method,
              outer_fold = outer_fold,
              inner_fold = inner_fold,
              kappa = cm[["overall"]][["Kappa"]],
              accuracy = cm[["overall"]][["Accuracy"]],
              mcc = mcc,
              confusion_matrix = I(list(cm))
            )
          }
        }
      }
    }

    do.call(rbind, results_list)
  }



  #' Generate probability data frames for standard multiclass classification
  #' @param cv_results_df Cross-validation results data frame
  #' @param best_params_df Best parameters data frame
  #' @param label_mapping Label mapping data frame
  #' @param filtered_subtypes Filtered leukemia subtypes
  #' @param filter_unseen_classes Whether to filter samples with classes not in training (default: TRUE)
  #' @return List of probability data frames organized by outer fold (and filtering statistics)

  generate_standard_probability_matrices <- function(cv_results_df, best_params_df, label_mapping, filtered_subtypes, study_names, filter_unseen_classes = TRUE) {
    outer_fold_ids <- unique(cv_results_df$outer_fold)
    probability_matrices <- list()
    filtering_stats_list <- list()
    
    if (filter_unseen_classes) {
      cat("  Filtering samples with classes not present in training set...\n")
    }

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


        # Extract class information (these are the training classes)
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
        probability_matrix$inner_fold <- inner_fold_id # this is the left out fold for the inner cv
        probability_matrix$outer_fold <- outer_fold_id # outer left out fold, more an id of the cv run
        probability_matrix$indices <- parse_numeric_string(inner_fold_data$sample_indices) + 1
        probability_matrix$study <- study_names[probability_matrix$indices]
        
        # Apply filtering if requested
        if (filter_unseen_classes) {
          fold_id <- paste(outer_fold_id, inner_fold_id, sep = "_")
          filter_result <- filter_samples_by_training_classes(
            probability_matrix, 
            class_labels,  # class_labels are the training classes
            fold_id
          )
          probability_matrix <- filter_result$filtered_matrix
          if (!is.null(filter_result$stats)) {
            filtering_stats_list[[fold_id]] <- filter_result$stats
          }
        }
        
        fold_matrices[[as.character(inner_fold_id)]] <- probability_matrix
      }

      probability_matrices[[as.character(outer_fold_id)]] <- fold_matrices
    }

    # Return both matrices and filtering statistics
    result <- list(matrices = probability_matrices)
    if (filter_unseen_classes && length(filtering_stats_list) > 0) {
      result$filtering_stats <- do.call(rbind, filtering_stats_list)
    }
    
    return(result)
  }

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
              f1_score = class_weight_info$mean_f1_score,
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
            kappa = fold_weight_info$mean_kappa,
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
  round_to <- function(x, base = 0.05) {
      base * round(x / base)
    }
  generate_weights <- function(step = 0.05){
    # Generate all combinations of weights from 0 to 1 in 0.1 steps
    steps <- seq(0, 1, by = step)
    grid <- expand.grid(SVM = steps, XGB = steps, NN = steps)

    # Remove rows where all the values are the same
    grid <- grid[!apply(grid, 1, function(x) length(unique(x)) == 1),]
    grid <- t(apply(grid, 1, function(row) {
      round_to(row / sum(row), 0.05)
    }))
    grid <- grid[!duplicated(grid), ]

    # Convert to a named list
    ENSEMBLE_WEIGHTS <- apply(grid, 1, function(row) {
      list(SVM = row["SVM"], XGB = row["XGB"], NN = row["NN"])
    })

    # Name the list elements for clarity (optional)
    names(ENSEMBLE_WEIGHTS) <- paste0("W", seq_along(ENSEMBLE_WEIGHTS))
    ENSEMBLE_WEIGHTS[["ALL"]] <- list(SVM = 1, XGB = 0, NN = 0) # SVM as main fallback since this is in general the best working model

    return(ENSEMBLE_WEIGHTS)
  }

  #' Perform One-vs-Rest ensemble analysis for each class separately
  #' @param results Analysis results containing probability matrices
  #' @param weights Weight configurations for ensemble
  #' @param type Type of analysis ("cv" or "loso")
  #' @return List of performance metrics aggregated across inner folds for each outer fold, weight configuration, and class
  perform_ovr_ensemble_analysis <- function(results, weights, type = "cv") {
    cat("Performing One-vs-Rest ensemble analysis...\n")

    outer_folds <- names(results$probability_matrices$svm[[type]])
    df_list <- list()

    for (outer_fold in outer_folds) {
      cat(sprintf("  Processing outer fold %s...\n", outer_fold))

      # Get inner folds for this outer fold
      inner_folds <- names(results$probability_matrices$svm[[type]][[outer_fold]])

      # Store results for all inner folds within this outer fold
      outer_fold_results <- list()

      for (inner_fold in inner_folds) {
        cat(sprintf("    Processing inner fold %s...\n", inner_fold))

        # Align probability matrices for this outer fold and inner fold
        aligned_matrices <- align_probability_matrices(results$probability_matrices, outer_fold, inner_fold, type)
        if (is.null(aligned_matrices)) {
          cat(sprintf("    Skipping outer fold %s, inner fold %s - unable to align matrices\n", outer_fold, inner_fold))
          next
        }

        for (i in seq_along(weights)) {
          weight_i <- names(weights)[i]

          # Extract aligned probability data frames
          prob_df_SVM <- aligned_matrices$svm
          prob_df_XGB <- aligned_matrices$xgboost
          prob_df_NN <- aligned_matrices$neural_net
          truth <- make.names(aligned_matrices$non_prob_cols$y)

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
            prevalence <- cm$byClass["Prevalence"]

            # Create results data frame for this class and inner fold
            df <- data.frame(
              outer_fold = outer_fold,
              inner_fold = inner_fold,
              weights = weight_i,
              type = type,
              class = gsub("Class.", "", class_name),
              sensitivity = sensitivity,
              specificity = specificity,
              balanced_accuracy = balanced_accuracy,
              f1_score = f1_score,
              prevalence = prevalence,
              stringsAsFactors = FALSE
            )

            key <- paste(weight_i, class_name, sep = "_")
            outer_fold_results[[key]] <- rbind(outer_fold_results[[key]], df)
          }
        }
      }

      # Aggregate results across inner folds for this outer fold
      if (length(outer_fold_results) > 0) {
        aggregated_results <- list()

        for (key in names(outer_fold_results)) {
          inner_fold_data <- outer_fold_results[[key]]

          # Aggregate across inner folds (average the metrics)
          aggregated_df <- inner_fold_data %>%
            group_by(outer_fold, weights, type, class) %>%
            summarise(
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
              n_inner_folds = n(),
              .groups = "drop"
            )

          aggregated_results[[key]] <- aggregated_df
        }

        # Combine all aggregated results for this outer fold
        df_list[[outer_fold]] <- do.call(rbind, aggregated_results)
      }
    }

    df_list
  }

  #' Generate One-vs-Rest optimized ensemble probability matrices
  #' @param results Analysis results containing probability matrices
  #' @param weights Weight configurations for ensemble
  #' @param type Type of analysis ("cv" or "loso")
  #' @param ensemble_performance Aggregated performance results from perform_ovr_ensemble_analysis
  #' @return List containing optimized probability matrices and weights used for each outer fold
  generate_ovr_optimized_ensemble_matrices <- function(results, weights, type = "cv", ensemble_performance) {
    cat("Generating One-vs-Rest optimized ensemble probability matrices...\n")

    outer_folds <- names(results$probability_matrices$svm[[type]])
    optimized_matrices <- list()
    weights_used <- list()  # Store weights used for each outer fold

    for (outer_fold in outer_folds) {
      cat(sprintf("  Creating OvR optimized matrices for outer fold %s...\n", outer_fold))

      # Get best weight configuration for each class in this outer fold based on aggregated performance
      fold_performance <- ensemble_performance[[outer_fold]]
      if (is.null(fold_performance) || nrow(fold_performance) == 0) {
        cat(sprintf("    Skipping outer fold %s - no performance data available\n", outer_fold))
        next
      }

      # Get classes that actually have performance data (i.e., were present in this outer fold)
      available_classes <- unique(fold_performance$class)

      # Store weights used for each class in this outer fold
      outer_fold_weights_used <- list()

      # For each class, find the best ensemble weights based on mean F1 score
      for (class_name in available_classes) {
        class_performance <- fold_performance[fold_performance$class == class_name, ]

        if (nrow(class_performance) > 0) {
          # Use the weights with the highest mean F1 score for this class
          best_weight_indices <- which.max(class_performance$mean_f1_score)
          best_weight_name <- class_performance$weights[best_weight_indices]

          # Ensure we have a single weight name (take first if multiple)
          if (length(best_weight_name) > 1) {
            best_weight_name <- best_weight_name[1]
            cat(sprintf("    Warning: Multiple best weights found for class %s, using first one\n", class_name))
          }

          # Validate weight name
          if (is.null(best_weight_name) || is.na(best_weight_name) || length(best_weight_name) == 0 || best_weight_name == "") {
            cat(sprintf("    Warning: Invalid weight name for class %s, using default weights\n", class_name))
            best_weights <- weights[["ALL"]]
            best_weight_name <- "ALL"
          } else if (!best_weight_name %in% names(weights)) {
            cat(sprintf("    Warning: Weight '%s' not found in weights list for class %s, using default weights\n", best_weight_name, class_name))
            best_weights <- weights[["ALL"]]
            best_weight_name <- "ALL"
          } else {
            best_weights <- weights[[best_weight_name]]
          }

          # Store the weight configuration used for this class
          outer_fold_weights_used[[class_name]] <- list(
            weight_name = best_weight_name,
            weights = best_weights,
            mean_f1_score = max(class_performance$mean_f1_score, na.rm = TRUE)
          )

          cat(sprintf("    Best weights for class %s: %s (F1=%.4f)\n",
                      class_name, best_weight_name, max(class_performance$mean_f1_score, na.rm = TRUE)))
        }
      }

      # Now generate optimized matrices for each inner fold using the selected weights
      inner_folds <- names(results$probability_matrices$svm[[type]][[outer_fold]])
      inner_fold_matrices <- list()

      for (inner_fold in inner_folds) {
        cat(sprintf("    Creating optimized matrix for inner fold %s...\n", inner_fold))

        # Align probability matrices for this outer fold and inner fold
        aligned_matrices <- align_probability_matrices(results$probability_matrices, outer_fold, inner_fold, type)
        if (is.null(aligned_matrices)) {
          cat(sprintf("      Skipping outer fold %s, inner fold %s - unable to align matrices\n", outer_fold, inner_fold))
          next
        }

        # Extract aligned probability data frames
        prob_df_SVM <- aligned_matrices$svm
        prob_df_XGB <- aligned_matrices$xgboost
        prob_df_NN <- aligned_matrices$neural_net
        non_prob_cols <- aligned_matrices$non_prob_cols

        # Get all class names
        all_classes <- colnames(prob_df_SVM)

        # Initialize optimized probability matrix
        optimized_matrix <- matrix(0, nrow = nrow(prob_df_SVM), ncol = length(all_classes))
        colnames(optimized_matrix) <- all_classes

        # For each class, use the selected best weights for this outer fold
        for (class_name in all_classes) {
          # Clean class name for matching
          clean_class_name <- gsub("Class.", "", class_name)
          clean_class_name_no_dots <- gsub("\\.", "", clean_class_name)

          # Find the weights to use for this class
          best_weights <- NULL
          if (clean_class_name %in% names(outer_fold_weights_used)) {
            best_weights <- outer_fold_weights_used[[clean_class_name]]$weights
          } else if (clean_class_name_no_dots %in% names(outer_fold_weights_used)) {
            best_weights <- outer_fold_weights_used[[clean_class_name_no_dots]]$weights
          } else {
            # Try partial matching
            matching_classes <- names(outer_fold_weights_used)[
              grepl(clean_class_name, names(outer_fold_weights_used), ignore.case = TRUE) |
                grepl(clean_class_name_no_dots, names(outer_fold_weights_used), ignore.case = TRUE)
            ]
            if (length(matching_classes) > 0) {
              best_weights <- outer_fold_weights_used[[matching_classes[1]]]$weights
            }
          }

          # Use default weights if no specific weights found for this class
          if (is.null(best_weights)) {
            cat(sprintf("      Using default weights for class %s (not present in outer fold performance)\n", clean_class_name))
            best_weights <- weights[["ALL"]]
          }

          # Calculate weighted ensemble probabilities for this class
          class_probs <- prob_df_SVM[[class_name]] * best_weights$SVM +
            prob_df_XGB[[class_name]] * best_weights$XGB +
            prob_df_NN[[class_name]] * best_weights$NN

          optimized_matrix[, class_name] <- class_probs
        }

        # Convert to data frame and normalize probabilities
        optimized_matrix <- data.frame(optimized_matrix)
        optimized_matrix <- t(apply(optimized_matrix, 1, function(row) row / sum(row)))
        optimized_matrix <- data.frame(optimized_matrix)
        optimized_matrix <- cbind(optimized_matrix, non_prob_cols)

        inner_fold_matrices[[inner_fold]] <- optimized_matrix
      }

      optimized_matrices[[outer_fold]] <- inner_fold_matrices
      weights_used[[outer_fold]] <- outer_fold_weights_used
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
  #' @return Data frame with multiclass performance metrics for each outer fold and inner fold
  analyze_ovr_ensemble_multiclass_performance <- function(ovr_ensemble_result, type = "cv") {
    cat("Analyzing One-vs-Rest ensemble multiclass performance...\n")

    # Extract optimized matrices from the result structure
    optimized_matrices <- ovr_ensemble_result$matrices

    performance_results <- list()

    for (outer_fold_name in names(optimized_matrices)) {
      cat(sprintf("  Analyzing outer fold %s...\n", outer_fold_name))

      # Check if this is the nested structure [outer_fold][inner_fold]
      outer_fold_data <- optimized_matrices[[outer_fold_name]]

      if (is.list(outer_fold_data) && !is.data.frame(outer_fold_data)) {
        # This is the new nested structure - iterate through inner folds
        inner_fold_results <- list()

        for (inner_fold_name in names(outer_fold_data)) {
          cat(sprintf("    Analyzing inner fold %s...\n", inner_fold_name))

          # Get the optimized matrix for this inner fold
          optimized_matrix <- outer_fold_data[[inner_fold_name]]

          # Extract true labels and remove from probability matrix
          truth <- optimized_matrix$y
          prob_matrix <- optimized_matrix[, !colnames(optimized_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study"), drop = FALSE]

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

          inner_fold_results[[inner_fold_name]] <- cm
        }

        performance_results[[outer_fold_name]] <- inner_fold_results

      } else {
        # This is the old flat structure - handle as before
        cat(sprintf("  Analyzing fold %s (old structure)...\n", outer_fold_name))

        # Get the optimized matrix for this fold
        optimized_matrix <- outer_fold_data

        # Extract true labels and remove from probability matrix
        truth <- optimized_matrix$y
        prob_matrix <- optimized_matrix[, !colnames(optimized_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study"), drop = FALSE]

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

        performance_results[[outer_fold_name]] <- cm
      }
    }

    # Return all fold results
    performance_results
  }

  #' Perform global ensemble optimization using overall kappa
  #' @param results Analysis results containing probability matrices
  #' @param weights Weight configurations for ensemble
  #' @param type Type of analysis ("cv" or "loso")
  #' @return List of performance metrics aggregated across inner folds for each outer fold and weight configuration
  perform_global_ensemble_analysis <- function(results, weights, type = "cv") {
    cat("Performing global ensemble analysis...\n")

    outer_folds <- names(results$probability_matrices$svm[[type]])
    df_list <- list()

    for (outer_fold in outer_folds) {
      cat(sprintf("  Processing outer fold %s...\n", outer_fold))

      # Get inner folds for this outer fold
      inner_folds <- names(results$probability_matrices$svm[[type]][[outer_fold]])

      # Store results for all inner folds within this outer fold
      outer_fold_results <- list()

      for (inner_fold in inner_folds) {
        cat(sprintf("    Processing inner fold %s...\n", inner_fold))

        # Align probability matrices for this outer fold and inner fold
        aligned_matrices <- align_probability_matrices(results$probability_matrices, outer_fold, inner_fold, type)
        if (is.null(aligned_matrices)) {
          cat(sprintf("    Skipping outer fold %s, inner fold %s - unable to align matrices\n", outer_fold, inner_fold))
          next
        }

        for (i in seq_along(weights)) {
          weight_i <- names(weights)[i]

          # Extract aligned probability data frames
          prob_df_SVM <- aligned_matrices$svm
          prob_df_XGB <- aligned_matrices$xgboost
          prob_df_NN <- aligned_matrices$neural_net
          truth <- make.names(aligned_matrices$non_prob_cols$y)

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

          # Create results data frame for this inner fold
          df <- data.frame(
            outer_fold = outer_fold,
            inner_fold = inner_fold,
            weights = weight_i,
            type = type,
            kappa = overall_kappa,
            accuracy = overall_accuracy,
            stringsAsFactors = FALSE
          )

          outer_fold_results[[weight_i]] <- rbind(outer_fold_results[[weight_i]], df)
        }
      }

      # Aggregate results across inner folds for this outer fold
      if (length(outer_fold_results) > 0) {
        aggregated_results <- list()

        for (weight_name in names(outer_fold_results)) {
          inner_fold_data <- outer_fold_results[[weight_name]]

          # Aggregate across inner folds (average the metrics)
          aggregated_df <- inner_fold_data %>%
            group_by(outer_fold, weights, type) %>%
            summarise(
              mean_kappa = mean(kappa, na.rm = TRUE),
              mean_accuracy = mean(accuracy, na.rm = TRUE),
              sd_kappa = sd(kappa, na.rm = TRUE),
              sd_accuracy = sd(accuracy, na.rm = TRUE),
              n_inner_folds = n(),
              .groups = "drop"
            )

          aggregated_results[[weight_name]] <- aggregated_df
        }

        # Combine all aggregated results for this outer fold
        df_list[[outer_fold]] <- do.call(rbind, aggregated_results)
      }
    }

    df_list
  }

  #' Generate globally optimized ensemble probability matrices
  #' @param results Analysis results containing probability matrices
  #' @param weights Weight configurations for ensemble
  #' @param type Type of analysis ("cv" or "loso")
  #' @param ensemble_performance Aggregated performance results from perform_global_ensemble_analysis
  #' @return List containing optimized probability matrices and weights used for each outer fold
  generate_global_optimized_ensemble_matrices <- function(results, weights, type = "cv", ensemble_performance) {
    cat("Generating globally optimized ensemble probability matrices...\n")

    outer_folds <- names(results$probability_matrices$svm[[type]])
    optimized_matrices <- list()
    weights_used <- list()  # Store weights used for each outer fold

    for (outer_fold in outer_folds) {
      cat(sprintf("  Creating globally optimized matrices for outer fold %s...\n", outer_fold))

      # Get best weight configuration for this outer fold (highest mean kappa)
      fold_performance <- ensemble_performance[[outer_fold]]
      if (is.null(fold_performance) || nrow(fold_performance) == 0) {
        cat(sprintf("    Skipping outer fold %s - no performance data available\n", outer_fold))
        next
      }

      best_weight_name <- fold_performance$weights[which.max(fold_performance$mean_kappa)]
      best_weights <- weights[[best_weight_name]]
      best_kappa <- max(fold_performance$mean_kappa, na.rm = TRUE)

      # Store the weight configuration used for this outer fold
      weights_used[[outer_fold]] <- list(
        weight_name = best_weight_name,
        weights = best_weights,
        mean_kappa = best_kappa
      )

      cat(sprintf("    Using globally optimized weights (%s) for outer fold %s (mean kappa = %.4f)\n",
                  best_weight_name, outer_fold, best_kappa))

      # Now generate optimized matrices for each inner fold using the selected weights
      inner_folds <- names(results$probability_matrices$svm[[type]][[outer_fold]])
      inner_fold_matrices <- list()

      for (inner_fold in inner_folds) {
        cat(sprintf("    Creating optimized matrix for inner fold %s...\n", inner_fold))

        # Align probability matrices for this outer fold and inner fold
        aligned_matrices <- align_probability_matrices(results$probability_matrices, outer_fold, inner_fold, type)
        if (is.null(aligned_matrices)) {
          cat(sprintf("      Skipping outer fold %s, inner fold %s - unable to align matrices\n", outer_fold, inner_fold))
          next
        }

        # Extract aligned probability data frames
        prob_df_SVM <- aligned_matrices$svm
        prob_df_XGB <- aligned_matrices$xgboost
        prob_df_NN <- aligned_matrices$neural_net
        non_prob_cols <- aligned_matrices$non_prob_cols

        # Calculate weighted ensemble probabilities using best global weights
        optimized_matrix <- prob_df_SVM * best_weights$SVM +
          prob_df_XGB * best_weights$XGB +
          prob_df_NN * best_weights$NN

        # Normalize probabilities to sum to 1 for each sample
        optimized_matrix <- optimized_matrix / rowSums(optimized_matrix)

        # Convert to data frame and add true labels
        optimized_matrix <- data.frame(optimized_matrix)
        optimized_matrix <- cbind(optimized_matrix, non_prob_cols)

        inner_fold_matrices[[inner_fold]] <- optimized_matrix
      }

      optimized_matrices[[outer_fold]] <- inner_fold_matrices
    }

    # Return both matrices and weights used
    list(
      matrices = optimized_matrices,
      weights_used = weights_used
    )
  }

  #' Calculate performance metrics for optimized ensemble matrices
  #' @param ensemble_result Result containing optimized ensemble probability matrices and weights (for global optimization) or just matrices (for other methods)
  #' @param type Type of analysis ("cv" or "loso")
  #' @return Data frame with performance metrics for each outer fold and inner fold
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

    for (outer_fold_name in names(optimized_matrices)) {
      cat(sprintf("  Analyzing outer fold %s...\n", outer_fold_name))

      # Check if this is the nested structure [outer_fold][inner_fold]
      outer_fold_data <- optimized_matrices[[outer_fold_name]]

      if (is.list(outer_fold_data) && !is.data.frame(outer_fold_data)) {
        # This is the new nested structure - iterate through inner folds
        inner_fold_results <- list()

        for (inner_fold_name in names(outer_fold_data)) {
          cat(sprintf("    Analyzing inner fold %s...\n", inner_fold_name))

          # Get the optimized matrix for this inner fold
          optimized_matrix <- outer_fold_data[[inner_fold_name]]

          # Extract true labels and remove from probability matrix
          truth <- optimized_matrix$y
          prob_matrix <- optimized_matrix[, !colnames(optimized_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study"), drop = FALSE]

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

          inner_fold_results[[inner_fold_name]] <- cm
        }

        performance_results[[outer_fold_name]] <- inner_fold_results

      } else {
        # This is the old flat structure - handle as before
        cat(sprintf("  Analyzing fold %s (old structure)...\n", outer_fold_name))

        # Get the optimized matrix for this fold
        optimized_matrix <- outer_fold_data

        # Extract true labels and remove from probability matrix
        truth <- optimized_matrix$y
        prob_matrix <- optimized_matrix[, !colnames(optimized_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study"), drop = FALSE]

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

        performance_results[[outer_fold_name]] <- cm
      }
    }

    # Return all fold results
    performance_results
  }
  # =============================================================================
  # Matrix Alignment Functions
  # =============================================================================

  #' Align probability matrices from different models for ensemble analysis
  #' @param prob_matrices List of probability matrices from different models
  #' @param outer_fold_name Name of the outer fold being processed
  #' @param inner_fold_name Name of the inner fold being processed
  #' @param type Type of analysis ("cv" or "loso")
  #' @return List of aligned probability matrices
  align_probability_matrices <- function(prob_matrices, outer_fold_name, inner_fold_name, type) {
    # Extract matrices for this outer fold and inner fold
    svm_matrix <- prob_matrices$svm[[type]][[outer_fold_name]][[inner_fold_name]]
    xgb_matrix <- prob_matrices$xgboost[[type]][[outer_fold_name]][[inner_fold_name]]
    nn_matrix <- prob_matrices$neural_net[[type]][[outer_fold_name]][[inner_fold_name]]

    # Check if all matrices exist
    if (is.null(svm_matrix) || is.null(xgb_matrix) || is.null(nn_matrix)) {
      warning(sprintf("Missing probability matrix for outer fold %s, inner fold %s in %s analysis", outer_fold_name, inner_fold_name, type))
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
        warning(sprintf("No common samples across models for outer fold %s, inner fold %s, skipping", 
                       outer_fold_name, inner_fold_name))
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
        cat(sprintf("    Aligned samples for outer %s, inner %s: dropped %d samples to match across models (SVM: %d, XGB: %d, NN: %d -> common: %d)\n", 
                   outer_fold_name, inner_fold_name, n_dropped, n_svm_orig, n_xgb_orig, n_nn_orig, length(common_samples)))
      }
    } else {
      # If no indices column, check row counts match
      if (nrow(svm_matrix) != nrow(xgb_matrix) || nrow(svm_matrix) != nrow(nn_matrix)) {
        warning(sprintf("Sample counts don't match for outer fold %s, inner fold %s (SVM: %d, XGB: %d, NN: %d), attempting to align by truncation",
                       outer_fold_name, inner_fold_name, nrow(svm_matrix), nrow(xgb_matrix), nrow(nn_matrix)))
        # Fall through to existing truncation logic
      }
    }

    # Extract true labels
    truth_svm <- make.names(svm_matrix$y)
    truth_xgb <- make.names(xgb_matrix$y)
    truth_nn <- make.names(nn_matrix$y)

    # store non_prob columns
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
        cat(sprintf("The probabilties for %s have less samples then max_samples\n", model_name))
      }

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
    aligned_matrices$non_prob_cols <- non_prob_cols
    aligned_matrices
  }


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

  #' Calculate and display mean kappa across folds for all ensemble methods
  #' @param results Analysis results containing all ensemble performance metrics
  #' @param type Type of analysis ("cv" or "loso")
  #' @return Data frame with mean kappa for each ensemble method
  compare_ensemble_performance <- function(results, type = "cv") {

    outer_folds <- names(results$probability_matrices$svm[[type]])
    performance_summary <- list()

    # Individual model performance - need to aggregate across all inner folds
    for (model_name in c("svm", "xgboost", "neural_net")) {
      all_kappas <- c()

      for (outer_fold in outer_folds) {
        # Get all inner folds for this outer fold
        inner_folds <- names(results$probability_matrices[[model_name]][[type]][[outer_fold]])

        for (inner_fold in inner_folds) {
          optimized_matrix <- results$probability_matrices[[model_name]][[type]][[outer_fold]][[inner_fold]]

          # Extract true labels and remove from probability matrix
          truth <- make.names(optimized_matrix$y)
          prob_matrix <- optimized_matrix[, !colnames(optimized_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study"), drop = FALSE]

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
          all_kappas <- c(all_kappas, cm$overall["Kappa"])
        }
      }

      mean_kappa <- mean(all_kappas, na.rm = TRUE)
      sd_kappa <- sd(all_kappas, na.rm = TRUE)

      performance_summary[[toupper(model_name)]] <- list(
        mean_kappa = mean_kappa,
        sd_kappa = sd_kappa,
        fold_kappas = all_kappas
      )
    }

    # Ensemble method performance - need to aggregate across nested structure
    ensemble_methods <- list(
      "OvR_Ensemble" = results$ovr_ensemble_multiclass_performance,
      "Global_Optimized" = results$global_optimized_ensemble_performance
    )

    for (method_name in names(ensemble_methods)) {
      method_performance <- ensemble_methods[[method_name]]
      all_kappas <- c()

      for (outer_fold in outer_folds) {
        if (outer_fold %in% names(method_performance)) {
          outer_fold_performance <- method_performance[[outer_fold]]

          # Check if this is nested structure
          if (is.list(outer_fold_performance) && !inherits(outer_fold_performance, "confusionMatrix")) {
            # This is nested [outer_fold][inner_fold] structure
            for (inner_fold in names(outer_fold_performance)) {
              cm <- outer_fold_performance[[inner_fold]]
              if (inherits(cm, "confusionMatrix")) {
                all_kappas <- c(all_kappas, cm$overall["Kappa"])
              }
            }
          } else {
            # This is flat structure
            if (inherits(outer_fold_performance, "confusionMatrix")) {
              all_kappas <- c(all_kappas, outer_fold_performance$overall["Kappa"])
            }
          }
        }
      }

      mean_kappa <- mean(all_kappas, na.rm = TRUE)
      sd_kappa <- sd(all_kappas, na.rm = TRUE)

      performance_summary[[method_name]] <- list(
        mean_kappa = mean_kappa,
        sd_kappa = sd_kappa,
        fold_kappas = all_kappas
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

    print(summary_df)

    return(summary_df)
  }

  #' Compare ensemble performance for both CV and LOSO
  #' @param results Analysis results containing all ensemble performance metrics
  #' @return List of performance comparisons for both CV and LOSO
  compare_ensemble_performance_for_both_types <- function(results) {

    performance_comparisons <- list()

    for (analysis_type in c("cv", "loso")) {

      # Check if we have results for this analysis type
      if (!analysis_type %in% names(results)) {
        next
      }

      # Create results list for performance comparison
      comparison_results <- list(
        probability_matrices = results$probability_matrices,
        ovr_ensemble_multiclass_performance = results[[analysis_type]]$ovr_ensemble_multiclass_performance,
        global_optimized_ensemble_performance = results[[analysis_type]]$global_optimized_ensemble_performance
      )

      # Compare all ensemble methods and display mean kappa across folds
      performance_comparison <- compare_ensemble_performance(comparison_results, analysis_type)
      performance_comparisons[[analysis_type]] <- performance_comparison
    }

    performance_comparisons
  }

  #' Evaluate nested CV kappa with rejection for a single probability matrix
  #' @param prob_matrix Probability matrix with class probabilities and true labels
  #' @param fold_name Name of the fold being analyzed
  #' @param model_name Name of the model being analyzed
  #' @param type Type of analysis ("cv" or "loso")
  #' @return Data frame with rejection analysis results
  evaluate_single_matrix_with_rejection <- function(prob_matrix, fold_name, model_name, type) {
    # Extract true labels and remove from probability matrix
    truth <- prob_matrix$y
    prob_matrix_clean <- prob_matrix[, !colnames(prob_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study"), drop = FALSE]

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
        outer_fold_matrices <- probability_matrices[[model_name]][[type]]

        for (outer_fold_name in names(outer_fold_matrices)) {
          inner_fold_matrices <- outer_fold_matrices[[outer_fold_name]]

          for (inner_fold_name in names(inner_fold_matrices)) {
            prob_matrix <- inner_fold_matrices[[inner_fold_name]]

            if (!is.null(prob_matrix) && nrow(prob_matrix) > 0) {
              # Create combined fold name for identification
              combined_fold_name <- paste(outer_fold_name, inner_fold_name, sep = "_")
              rejection_results <- evaluate_single_matrix_with_rejection(
                prob_matrix, combined_fold_name, model_name, type
              )
              # Add outer and inner fold information
              rejection_results$outer_fold <- outer_fold_name
              rejection_results$inner_fold <- inner_fold_name
              all_rejection_results <- rbind(all_rejection_results, rejection_results)
            }
          }
        }
      }
    }

    # Analyze ensemble methods
    cat("  Analyzing ensemble methods...\n")
    ensemble_methods <- list(
      "OvR_Ensemble" = ensemble_matrices$ovr_optimized_ensemble_matrices,
      "Global_Optimized" = ensemble_matrices$global_optimized_ensemble_matrices
    )

    for (ensemble_name in names(ensemble_methods)) {
      cat(sprintf("    Processing %s...\n", ensemble_name))

      ensemble_outer_fold_matrices <- ensemble_methods[[ensemble_name]]

      for (outer_fold_name in names(ensemble_outer_fold_matrices)) {
        inner_fold_matrices <- ensemble_outer_fold_matrices[[outer_fold_name]]

        for (inner_fold_name in names(inner_fold_matrices)) {
          prob_matrix <- inner_fold_matrices[[inner_fold_name]]

          if (!is.null(prob_matrix) && nrow(prob_matrix) > 0) {
            # Create combined fold name for identification
            combined_fold_name <- paste(outer_fold_name, inner_fold_name, sep = "_")
            rejection_results <- evaluate_single_matrix_with_rejection(
              prob_matrix, combined_fold_name, ensemble_name, type
            )
            # Add outer and inner fold information
            rejection_results$outer_fold <- outer_fold_name
            rejection_results$inner_fold <- inner_fold_name
            all_rejection_results <- rbind(all_rejection_results, rejection_results)
          }
        }
      }
    }

    return(all_rejection_results)
  }

  #' Find optimal probability cutoff for each model/ensemble per outer fold
  #' @param rejection_results Data frame with rejection analysis results
  #' @param optimization_metric Metric to optimize ("kappa" or "accuracy")
  #' @return List with optimal cutoffs per outer fold and summary statistics
  find_optimal_cutoffs <- function(rejection_results, optimization_metric = "kappa") {
    cat("Finding optimal probability cutoffs per outer fold...\n")

    # optimal_cutoffs_per_inner_fold <- rejection_results %>%
    #   filter( (is.na(rejected_accuracy) | rejected_accuracy < 0.5) & (perc_rejected < 0.05) ) %>% mutate(rejected_accuracy_mod = ifelse(is.na(rejected_accuracy), 0, rejected_accuracy)) %>%
    #   group_by(model, outer_fold, inner_fold) %>% slice_max(kappa) %>% slice_min(rejected_accuracy_mod, with_ties = F) %>% ungroup()


    optimal_cutoffs_per_outer_fold <- rejection_results %>%
      mutate(rejected_accuracy_mod = ifelse(is.na(rejected_accuracy), 0, rejected_accuracy)) %>%
      group_by(model, outer_fold, prob_cutoff) %>%
      summarise(
        mean_cutoff = mean(prob_cutoff, na.rm = TRUE),
        sd_cutoff = sd(prob_cutoff, na.rm = TRUE),
        mean_kappa = mean(kappa, na.rm = TRUE),
        sd_kappa = sd(kappa, na.rm = TRUE),
        mean_accuracy = mean(accuracy, na.rm = TRUE),
        sd_accuracy = sd(accuracy, na.rm = TRUE),
        mean_rejected_accuracy = mean(rejected_accuracy_mod, na.rm = TRUE),
        sd_rejected_accuracy = sd(rejected_accuracy_mod, na.rm = TRUE),
        mean_perc_rejected = mean(perc_rejected, na.rm = TRUE),
        sd_perc_rejected = sd(perc_rejected, na.rm = TRUE),
        n_outer_folds = n(),
        .groups = "drop"
      ) %>%
      # apply filters on the averages
      filter(mean_rejected_accuracy < 0.5, mean_perc_rejected < 0.05) %>%
      group_by(model, outer_fold) %>%
      slice_max(mean_kappa) %>%
      slice_min(mean_rejected_accuracy, with_ties = FALSE) %>%
      ungroup()

    # optimal_cutoffs_per_outer_fold <- optimal_cutoffs_per_inner_fold %>%
    # group_by(model, outer_fold)  %>%
    #   summarise(
    #     mean_cutoff = mean(prob_cutoff, na.rm = TRUE),
    #     sd_cutoff = sd(prob_cutoff, na.rm = TRUE),
    #     mean_kappa = mean(kappa, na.rm = TRUE),
    #     sd_kappa = sd(kappa, na.rm = TRUE),
    #     mean_accuracy = mean(accuracy, na.rm = TRUE),
    #     sd_accuracy = sd(accuracy, na.rm = TRUE),
    #     mean_perc_rejected = mean(perc_rejected, na.rm = TRUE),
    #     sd_perc_rejected = sd(perc_rejected, na.rm = TRUE),
    #     n_outer_folds = n(),
    #     .groups = "drop"
    #   )

    # Calculate summary statistics across outer folds for each model
    summary_stats <- optimal_cutoffs_per_outer_fold %>%
      group_by(model) %>%
      summarise(
        mean_cutoff = mean(mean_cutoff, na.rm = TRUE),
        sd_cutoff = sd(sd_cutoff, na.rm = TRUE),
        mean_kappa = mean(mean_kappa, na.rm = TRUE),
        sd_kappa = sd(sd_kappa, na.rm = TRUE),
        mean_accuracy = mean(mean_accuracy, na.rm = TRUE),
        sd_accuracy = sd(sd_accuracy, na.rm = TRUE),
        mean_rejected_accuracy = mean(mean_rejected_accuracy, na.rm = TRUE),
        sd_rejected_accuracy = sd(sd_rejected_accuracy, na.rm = TRUE),
        mean_perc_rejected = mean(mean_perc_rejected, na.rm = TRUE),
        sd_perc_rejected = sd(sd_perc_rejected, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      arrange(desc(mean_kappa))

    return(list(
      optimal_cutoffs_per_outer_fold = optimal_cutoffs_per_outer_fold,
      summary_stats = summary_stats
    ))
  }

  #' Run complete rejection analysis for both CV and LOSO
  #' @param probability_matrices Probability matrices for all models
  #' @param ensemble_results Ensemble analysis results
  #' @param output_base_dir Base directory for output files
  #' @return List of rejection analysis results
  run_complete_rejection_analysis <- function(probability_matrices, ensemble_results) {
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
        global_optimized_ensemble_matrices = ensemble_results[[analysis_type]]$global_optimized_ensemble_matrices$matrices
      )

      # Perform rejection analysis
      rejection_results[[analysis_type]][["all_results"]] <- evaluate_all_matrices_with_rejection(
        probability_matrices, ensemble_matrices, analysis_type
      )

      # Find optimal cutoffs
      rejection_results[[analysis_type]][["optimal_results"]] <- find_optimal_cutoffs(rejection_results[[analysis_type]][["all_results"]], "kappa")

    }

    return(rejection_results)
  }

  compare_all_results <- function(type, inner_cv_results){
    # Extract performance without rejection
    df_no_rejection <- inner_cv_results[["performance_comparisons"]][[type]] %>%
      as_tibble() %>%
      rename(model = Method,
             mean_kappa = Mean_Kappa) %>%
      select(model, mean_kappa) %>%
      mutate(model = str_to_lower(model))

    # Extract performance with rejection
    df_rejection <- inner_cv_results[["rejection_results"]][[type]][["optimal_results"]][["summary_stats"]] %>%
      select(model, mean_kappa_with_rejection = mean_kappa, mean_perc_rejected) %>%
      mutate(model = str_to_lower(model))

    # Combine
    combined_df <- df_no_rejection %>%
      left_join(df_rejection, by = "model")

    return(combined_df)
  }

  combine_all_results <- function(inner_cv_results){
    combined_results <- list()
    for (type in c("cv", "loso")){
      combined_results[[type]] <- compare_all_results(type, inner_cv_results)
    }
    return(combined_results)
  }

  load_library_quietly("plyr")
  load_library_quietly("dplyr")
  load_library_quietly("stringr")

  # Filters
  DATA_FILTERS <- list(
    min_samples_per_subtype = 10,
    excluded_subtypes = c("AML NOS", "Missing data", "Multi"),
    selected_studies = c(
      "TCGA-LAML",
      "LEUCEGENE",
      "BEATAML1.0-COHORT",
      "AAML0531",
      "AAML1031",
      "AAML03P1",
      "100LUMC"
    )
  )

  # Load mapping of class labels to numeric labels
  label_mapping <- read.csv("../data/label_mapping_all.csv")

  # Load leukemia subtype data
  leukemia_subtypes <- read.csv("../data/rgas_20aug25.csv")$ICC_Subtype

  # Load study metadata
  meta <- read.csv("../data/meta_20aug25.csv")
  study_names <- meta$Studies

  # Filter data based on criteria
  subtypes_with_sufficient_samples <- names(which(table(leukemia_subtypes) >= DATA_FILTERS$min_samples_per_subtype))
  filter <- which(
    leukemia_subtypes %in% subtypes_with_sufficient_samples &
      !leukemia_subtypes %in% DATA_FILTERS$excluded_subtypes &
      study_names %in% DATA_FILTERS$selected_studies
  )

  # Load leukemia subtype data
  filtered_leukemia_subtypes <- leukemia_subtypes[filter]

  # Load study metadata
  filtered_study_names <- study_names[filter]

  meta <- meta[filter, ]

  dir.create("../data/out/inner_cv/inner_cv_best_params")

  MODEL_CONFIGS <- list(
    svm = list(
      classification_type = "OvR",
      file_paths = list(
        cv = "../data/out/inner_cv/SVM_array/cv_20aug25_all/",
        loso = "../data/out/inner_cv/SVM_array/loso_20aug25_all/"
      ),
      output_dir = "../data/out/inner_cv/inner_cv_best_params/SVM_20aug25"
    ),
    xgboost = list(
      classification_type = "OvR",
      file_paths = list(
        cv = "../data/out/inner_cv/XGBOOST_array/cv_20aug25_all/",
        loso = "../data/out/inner_cv/XGBOOST_array/loso_20aug25_all/"
      ),
      output_dir = "../data/out/inner_cv/inner_cv_best_params/XGBOOST_20aug25"
    ),
    neural_net = list(
      classification_type = "standard",
      file_paths = list(
        cv = "../data/out/inner_cv/NN_array/cv_20aug25_all/",
        loso = "../data/out/inner_cv/NN_array/loso_20aug25_all/"
      ),
      output_dir = "../data/out/inner_cv/inner_cv_best_params/NN_20aug25"
    )
  )



  model_results <- load_all_model_data(MODEL_CONFIGS)

  # Extract best parameters
  best_parameters <- extract_all_best_parameters(model_results, MODEL_CONFIGS)

  # Save best parameters
  save_all_best_parameters(best_parameters, MODEL_CONFIGS)

  probability_matrices <- list()
  filtering_statistics <- list()  # Store filtering stats for reporting
  
  for (model_name in names(model_results)) {
    config <- MODEL_CONFIGS[[model_name]]
    cat(sprintf("Extracting %s probabilities...\n", toupper(model_name)))

    probability_matrices[[model_name]] <- list()

    for (fold_type in names(model_results[[model_name]])) {
      results <- model_results[[model_name]][[fold_type]]
      best_params <- best_parameters[[model_name]][[fold_type]]

      if (!is.null(results) && !is.null(best_params)) {
        # Generate probability matrices (with filtering)
        if (config$classification_type == "OvR") {
          result <- generate_ovr_probability_matrices(results, best_params, label_mapping, study_names, filter_unseen_classes = TRUE)
        } else {
          result <- generate_standard_probability_matrices(results, best_params, label_mapping, filtered_leukemia_subtypes, study_names, filter_unseen_classes = TRUE)
        }
        
        # Extract matrices and filtering stats
        probs <- result$matrices
        if (!is.null(result$filtering_stats)) {
          if (!model_name %in% names(filtering_statistics)) {
            filtering_statistics[[model_name]] <- list()
          }
          result$filtering_stats$model <- model_name
          result$filtering_stats$type <- fold_type
          filtering_statistics[[model_name]][[fold_type]] <- result$filtering_stats
        }

        probs <- lapply(probs, function(prob_outer) {
          lapply(prob_outer, function(prob) {
          MDS_cols <- grepl("MDS|TP53|MECOM", colnames(prob))
          MDS <- rowSums(prob[,MDS_cols])

          other_KMT2A_cols <- grepl("KMT2A", colnames(prob)) & !grepl("MLLT3", colnames(prob))
          other_KMT2A <- rowSums(prob[,other_KMT2A_cols])

          prob <- prob[, !(MDS_cols | other_KMT2A_cols)]
          prob$MDS.r <- MDS
          prob$other.KMT2A <- other_KMT2A
          prob$y <- modify_classes(prob$y)
          colnames(prob) <- modify_classes(colnames(prob))
          return(prob)
        })
          })

        probability_matrices[[model_name]][[fold_type]] <- probs
        remove(probs)
      }
    }
  }

  per_model_results <- get_per_model_performance(probability_matrices)

  per_class_results <- list()

  for (model in unique(per_model_results$model)) {
    for (outer_fold in unique(per_model_results$outer_fold)) {
      for (inner_fold in unique(per_model_results$inner_fold)) {

        subset <- per_model_results[
          per_model_results$model == model &
            per_model_results$outer_fold == outer_fold &
            per_model_results$inner_fold == inner_fold, ]

        if (nrow(subset) == 0) next

        cm <- subset$confusion_matrix[[1]]
        df <- as.data.frame(cm$byClass)
        df$class <- rownames(cm$byClass)
        df$model <- model
        df$outer_fold <- outer_fold
        df$inner_fold <- inner_fold

        per_class_results[[length(per_class_results) + 1]] <- df
      }
    }
  }

  per_class_results <- do.call(rbind, per_class_results)

  # Run ensemble analysis for both CV and LOSO
  ensemble_results <- run_ensemble_analysis_for_both_types(probability_matrices,  generate_weights(0.05))

  weights_dir <- "../data/out/inner_cv/ensemble_weights"
  save_ensemble_weights(ensemble_results,  weights_dir)

  performance_comparisons <- compare_ensemble_performance_for_both_types(
    list(probability_matrices = probability_matrices,  cv = ensemble_results$cv, loso = ensemble_results$loso)
  )

  # Run rejection analysiss
  rejection_results <- run_complete_rejection_analysis(
    probability_matrices,  ensemble_results
  )

  # Extract the optimal cutoffs dataframes
  cv_cutoffs <- NULL
  loso_cutoffs <- NULL

  if ("cv" %in% names(rejection_results) && "optimal_results" %in% names(rejection_results[["cv"]])) {
    cv_cutoffs <- rejection_results[["cv"]][["optimal_results"]][["optimal_cutoffs_per_outer_fold"]]
    if (!is.null(cv_cutoffs)) {
      cv_cutoffs$source <- "cv"
    }
  }

  if ("loso" %in% names(rejection_results) && "optimal_results" %in% names(rejection_results[["loso"]])) {
    loso_cutoffs <- rejection_results[["loso"]][["optimal_results"]][["optimal_cutoffs_per_outer_fold"]]
    if (!is.null(loso_cutoffs)) {
      loso_cutoffs$source <- "loso"
    }
  }

  cutoff_dir <- "../data/out/inner_cv/cutoffs"
  dir.create(cutoff_dir)

  # Bind the dataframes together if they exist
  combined_cutoffs <- data.frame()
  if (!is.null(cv_cutoffs)) {
    combined_cutoffs <- rbind(combined_cutoffs, cv_cutoffs)
  }
  if (!is.null(loso_cutoffs)) {
    combined_cutoffs <- rbind(combined_cutoffs, loso_cutoffs)
  }

  if (nrow(combined_cutoffs) > 0) {
    write.csv(combined_cutoffs, "../data/out/inner_cv/cutoffs/cutoffs.csv", row.names = FALSE)
  } else {
    cat("Warning: No optimal cutoffs found to save\n")
  }

  # Print summary of filtering statistics
  cat("\n=== Inner CV Sample Filtering Summary ===\n")
  if (length(filtering_statistics) > 0) {
    total_filtered <- 0
    total_samples <- 0
    for (model_name in names(filtering_statistics)) {
      for (fold_type in names(filtering_statistics[[model_name]])) {
        stats <- filtering_statistics[[model_name]][[fold_type]]
        model_total_filtered <- sum(stats$n_filtered)
        model_total_samples <- sum(stats$n_total)
        total_filtered <- total_filtered + model_total_filtered
        total_samples <- total_samples + model_total_samples
        cat(sprintf("%s (%s): Filtered %d/%d samples (%.1f%%)\n", 
                    toupper(model_name), toupper(fold_type), 
                    model_total_filtered, model_total_samples,
                    100 * model_total_filtered / model_total_samples))
      }
    }
    cat(sprintf("OVERALL: Filtered %d/%d samples (%.1f%%)\n", 
                total_filtered, total_samples,
                100 * total_filtered / total_samples))
  } else {
    cat("No samples were filtered (all validation samples had classes in training)\n")
  }
  cat("=========================================\n\n")

  inner_cv_results <- list(
    model_results = model_results,
    best_parameters = best_parameters,
    probability_matrices = probability_matrices,
    filtering_statistics = filtering_statistics,  # Add filtering statistics
    ensemble_results = ensemble_results,
    performance_comparisons = performance_comparisons,
    optimal_cutoffs = combined_cutoffs,
    rejection_results = rejection_results
  )

  inner_cv_results$final_results = combine_all_results(inner_cv_results)
  print(inner_cv_results$final_results)
  saveRDS(inner_cv_results, "../data/out/inner_cv/inner_cv_results.rds")
  
  # Save filtering statistics to CSV for easy inspection
  if (length(filtering_statistics) > 0) {
    all_filtering_stats <- do.call(rbind, lapply(names(filtering_statistics), function(model_name) {
      do.call(rbind, lapply(names(filtering_statistics[[model_name]]), function(fold_type) {
        stats <- filtering_statistics[[model_name]][[fold_type]]
        stats$model <- model_name
        stats$type <- fold_type
        return(stats)
      }))
    }))
    write.csv(all_filtering_stats, "../data/out/inner_cv/filtering_statistics.csv", row.names = FALSE)
    cat("Inner CV filtering statistics saved to: ../data/out/inner_cv/filtering_statistics.csv\n")
  }
  
  return(inner_cv_results)
}

inner_cv_results <- main_inner_cv()
