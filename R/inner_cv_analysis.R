# Source shared utility functions
source("utility_functions.R")

main_inner_cv <- function(merge_classes = FALSE, merge_mds_only = FALSE){

  #' Generate probability data frames for One-vs-Rest classification
  #' @param cv_results_df Cross-validation results data frame
  #' @param best_params_df Best parameters data frame
  #' @param label_mapping Label mapping data frame
  #' @param filter_unseen_classes Whether to filter samples with classes not in training (default: TRUE)
  #' @return List of probability data frames organized by outer fold (and filtering statistics)
  generate_ovr_probability_matrices <- function(cv_results_df, best_params_df, label_mapping, study_names, filter_unseen_classes = TRUE, merge_classes = FALSE, merge_mds_only = FALSE) {
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

        # Apply class merging if requested (before filtering)
        if (merge_classes) {
          probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_mds_only = merge_mds_only)
          # Update class_labels after merging for filtering
          class_labels <- colnames(probability_matrix)[!colnames(probability_matrix) %in%
                                                        c("y", "inner_fold", "outer_fold", "indices", "study")]
        }

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

  generate_standard_probability_matrices <- function(cv_results_df, best_params_df, label_mapping, filtered_subtypes, study_names, filter_unseen_classes = TRUE, merge_classes = FALSE, merge_mds_only = FALSE) {
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

        # Apply class merging if requested (before filtering)
        if (merge_classes) {
          probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_mds_only = merge_mds_only)
          # Update class_labels after merging for filtering
          class_labels <- colnames(probability_matrix)[!colnames(probability_matrix) %in%
                                                        c("y", "inner_fold", "outer_fold", "indices", "study")]
        }

        # Apply filtering if requested
        if (filter_unseen_classes) {
          fold_id <- paste(outer_fold_id, inner_fold_id, sep = "_")
          filter_result <- filter_samples_by_training_classes(
            probability_matrix,
            class_labels,  # class_labels are the training classes (may be merged)
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

  #' Perform One-vs-Rest ensemble analysis for each class separately (parallelized)
  #' Uses the unified function from utility_functions.R
  perform_ovr_ensemble_analysis <- function(results, weights, type = "cv") {
    perform_ovr_ensemble_analysis_unified(results, weights, type, has_inner_folds = TRUE)
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

    # Create cache for aligned matrices
    alignment_cache <- new.env(hash = TRUE)

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

        # Use cached alignment
        aligned_matrices <- align_probability_matrices_cached(
          results$probability_matrices, outer_fold, inner_fold, type, alignment_cache
        )
        if (is.null(aligned_matrices)) {
          cat(sprintf("      Skipping outer fold %s, inner fold %s - unable to align matrices\n", outer_fold, inner_fold))
          next
        }

        # Convert to matrices once for efficiency
        prob_mat_SVM <- as.matrix(aligned_matrices$svm)
        prob_mat_XGB <- as.matrix(aligned_matrices$xgboost)
        prob_mat_NN <- as.matrix(aligned_matrices$neural_net)
        non_prob_cols <- aligned_matrices$non_prob_cols

        # Get all class names
        all_classes <- colnames(aligned_matrices$svm)

        # Initialize optimized probability matrix
        optimized_matrix <- matrix(0, nrow = nrow(prob_mat_SVM), ncol = length(all_classes))
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

          # Calculate weighted ensemble probabilities for this class using matrix operations
          class_col_idx <- which(all_classes == class_name)
          optimized_matrix[, class_col_idx] <- prob_mat_SVM[, class_col_idx] * best_weights$SVM +
            prob_mat_XGB[, class_col_idx] * best_weights$XGB +
            prob_mat_NN[, class_col_idx] * best_weights$NN
        }

        # Normalize probabilities to sum to 1 for each sample
        row_sums <- rowSums(optimized_matrix)
        optimized_matrix <- optimized_matrix / row_sums

        # Convert to data frame and add true labels
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

  #' Perform global ensemble optimization using overall kappa (parallelized)
  #' Uses the unified function from utility_functions.R
  perform_global_ensemble_analysis <- function(results, weights, type = "cv") {
    perform_global_ensemble_analysis_unified(results, weights, type, has_inner_folds = TRUE)
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

    # Create cache for aligned matrices
    alignment_cache <- new.env(hash = TRUE)

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

        # Use cached alignment
        aligned_matrices <- align_probability_matrices_cached(
          results$probability_matrices, outer_fold, inner_fold, type, alignment_cache
        )
        if (is.null(aligned_matrices)) {
          cat(sprintf("      Skipping outer fold %s, inner fold %s - unable to align matrices\n", outer_fold, inner_fold))
          next
        }

        # Convert to matrices once for efficiency
        prob_mat_SVM <- as.matrix(aligned_matrices$svm)
        prob_mat_XGB <- as.matrix(aligned_matrices$xgboost)
        prob_mat_NN <- as.matrix(aligned_matrices$neural_net)
        non_prob_cols <- aligned_matrices$non_prob_cols

        # Calculate weighted ensemble probabilities using best global weights (matrix operations)
        optimized_matrix <- prob_mat_SVM * best_weights$SVM +
          prob_mat_XGB * best_weights$XGB +
          prob_mat_NN * best_weights$NN

        # Normalize probabilities to sum to 1 for each sample
        row_sums <- rowSums(optimized_matrix)
        optimized_matrix <- optimized_matrix / row_sums

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

  #' Evaluate nested CV kappa with rejection for a single probability matrix (parallelized cutoffs)
  #' @param prob_matrix Probability matrix with class probabilities and true labels
  #' @param fold_name Name of the fold being analyzed
  #' @param model_name Name of the model being analyzed
  #' @param type Type of analysis ("cv" or "loso")
  #' @return Data frame with rejection analysis results
  evaluate_single_matrix_with_rejection <- function(prob_matrix, fold_name, model_name, type) {
    # Use the parallelized version from utility_functions.R
    evaluate_single_matrix_with_rejection_parallel(prob_matrix, fold_name, model_name, type)
  }

  #' Evaluate rejection analysis for all probability matrices (parallelized)
  #' Uses the unified function from utility_functions.R
  evaluate_all_matrices_with_rejection <- function(probability_matrices, ensemble_matrices, type = "cv") {
    evaluate_all_matrices_with_rejection_unified(probability_matrices, ensemble_matrices, type, has_inner_folds = TRUE)
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
  leukemia_subtypes <- read.csv("../data/rgas_26jan26.csv")$ICC_Subtype

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
        cv = "../data/out/inner_cv/SVM_array/cv_26jan26_all/",
        loso = "../data/out/inner_cv/SVM_array/loso_26jan26_all/"
      ),
      output_dir = "../data/out/inner_cv/inner_cv_best_params/SVM_26jan26"
    ),
    xgboost = list(
      classification_type = "OvR",
      file_paths = list(
        cv = "../data/out/inner_cv/XGBOOST_array/cv_26jan26_all/",
        loso = "../data/out/inner_cv/XGBOOST_array/loso_26jan26_all/"
      ),
      output_dir = "../data/out/inner_cv/inner_cv_best_params/XGBOOST_26jan26"
    ),
    neural_net = list(
      classification_type = "standard",
      file_paths = list(
        cv = "../data/out/inner_cv/NN_array/cv_26jan26_all/",
        loso = "../data/out/inner_cv/NN_array/loso_26jan26_all/"
      ),
      output_dir = "../data/out/inner_cv/inner_cv_best_params/NN_26jan26"
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
        # Generate probability matrices (with filtering and optional merging)
        if (config$classification_type == "OvR") {
          result <- generate_ovr_probability_matrices(results, best_params, label_mapping, study_names, filter_unseen_classes = TRUE, merge_classes = merge_classes, merge_mds_only = merge_mds_only)
        } else {
          result <- generate_standard_probability_matrices(results, best_params, label_mapping, filtered_leukemia_subtypes, study_names, filter_unseen_classes = TRUE, merge_classes = merge_classes, merge_mds_only = merge_mds_only)
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

        # Keep individual classes for ensemble analysis (don't combine yet)
        probability_matrices[[model_name]][[fold_type]] <- probs
        remove(probs)
      }
    }
  }

  # Run ensemble analysis on all classes
  cat("Running ensemble analysis...\n")
  ensemble_results <- run_ensemble_analysis_for_both_types(probability_matrices, generate_weights())

  # Use probability matrices for per-model performance evaluation
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

  # Determine suffix for file paths (maxprob method - uses max probability instead of summing)
  if (!merge_classes) {
    merge_suffix <- "_unmerged_maxprob"
  } else if (merge_mds_only) {
    merge_suffix <- "_mds_only_maxprob"
  } else {
    merge_suffix <- "_merged_maxprob"
  }
  weights_dir <- paste0("../data/out/inner_cv/ensemble_weights", merge_suffix)
  save_ensemble_weights(ensemble_results,  weights_dir)

  performance_comparisons <- compare_ensemble_performance_for_both_types(
    list(probability_matrices = probability_matrices, cv = ensemble_results$cv, loso = ensemble_results$loso)
  )

  # Run rejection analysis
  rejection_results <- run_complete_rejection_analysis(
    probability_matrices, ensemble_results
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

  cutoff_dir <- paste0("../data/out/inner_cv/cutoffs", merge_suffix)
  dir.create(cutoff_dir, recursive = TRUE)

  # Bind the dataframes together if they exist
  combined_cutoffs <- data.frame()
  if (!is.null(cv_cutoffs)) {
    combined_cutoffs <- rbind(combined_cutoffs, cv_cutoffs)
  }
  if (!is.null(loso_cutoffs)) {
    combined_cutoffs <- rbind(combined_cutoffs, loso_cutoffs)
  }

  if (nrow(combined_cutoffs) > 0) {
    write.csv(combined_cutoffs, file.path(cutoff_dir, "cutoffs.csv"), row.names = FALSE)
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
    filtering_statistics = filtering_statistics,
    ensemble_results = ensemble_results,
    performance_comparisons = performance_comparisons,
    optimal_cutoffs = combined_cutoffs,
    rejection_results = rejection_results
  )

  inner_cv_results$final_results = combine_all_results(inner_cv_results)
  inner_cv_results$merge_classes <- merge_classes  # Store merge status in results
  inner_cv_results$merge_mds_only <- merge_mds_only  # Store merge_mds_only status in results
  print(inner_cv_results$final_results)
  saveRDS(inner_cv_results, paste0("../data/out/inner_cv/inner_cv_results_28jan2025", merge_suffix, ".rds"))

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
    write.csv(all_filtering_stats, paste0("../data/out/inner_cv/filtering_statistics", merge_suffix, ".csv"), row.names = FALSE)
    cat(sprintf("Inner CV filtering statistics saved to: ../data/out/inner_cv/filtering_statistics%s.csv\n", merge_suffix))
  }

  return(inner_cv_results)
}

# Run merged and MDS-only merged versions (maxprob method)
cat("=== Running Inner CV Analysis (Merged - MaxProb Method) ===\n")
inner_cv_results_unmerged <- main_inner_cv(merge_classes = FALSE, merge_mds_only = FALSE)

cat("=== Running Inner CV Analysis (MDS Only Merged - MaxProb Method) ===\n")
inner_cv_results_mds_only <- main_inner_cv(merge_classes = TRUE, merge_mds_only = TRUE)

cat("=== Running Inner CV Analysis (MDS Only Merged - MaxProb Method) ===\n")
inner_cv_results_merged <- main_inner_cv(merge_classes = TRUE, merge_mds_only = FALSE)
