# Source shared utility functions
source("utility_functions.R")

main_train_test_analysis <- function(merge_classes = FALSE, merge_mds_only = FALSE){

  #' Generate probability data frames for One-vs-Rest classification
  #' @param cv_results_df Cross-validation results data frame
  #' @param best_params_df Best parameters data frame
  #' @param label_mapping Label mapping data frame
  #' @return List of probability data frames organized by outer fold
  #'
  generate_ovr_probability_matrices <- function(cv_results_df, best_params_df, label_mapping, study_names, merge_classes = FALSE, merge_mds_only = FALSE) {
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

        # Apply class merging if requested
        if (merge_classes) {
          probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_mds_only = merge_mds_only)
        }

        fold_matrices[[as.character(inner_fold_id)]] <- probability_matrix
      }

      if (length(fold_matrices) > 0) {
        probability_matrices[[as.character(outer_fold_id)]] <- do.call(rbind, fold_matrices)
        probability_matrices[[as.character(outer_fold_id)]][is.na(probability_matrices[[as.character(outer_fold_id)]])] <- 0
      }
    }

    probability_matrices
  }

  get_per_model_performance <- function(probability_matrices){
    results_list <- list()
    for (model in names(probability_matrices)) {
      probability_matrices_model <- probability_matrices[[model]]

      for (method in names(probability_matrices_model)) {
        probability_matrices_method <- probability_matrices_model[[method]]

        for (fold_name in names(probability_matrices_method)){

          # Get matrix for this fold
          the_matrix <- probability_matrices_method[[fold_name]]

          # Extract true labels and remove from probability matrix
          truth <- the_matrix$y
          prob_matrix <- the_matrix[, !colnames(the_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study"), drop = FALSE]

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
          mcc <- mltools::mcc(preds, truth)

          results_list[[length(results_list) + 1]] <- data.frame(
            model = model,
            method = method,
            fold_name = fold_name,
            kappa = cm[["overall"]][["Kappa"]],
            accuracy = cm[["overall"]][["Accuracy"]],
            mcc = mcc,
            confusion_matrix = I(list(cm))  # store cm object in a column
          )
        }
      }
    }
    results_df <- do.call(rbind, results_list)
    results_df
  }


  #' Generate probability data frames for standard multiclass classification
  #' @param cv_results_df Cross-validation results data frame
  #' @param best_params_df Best parameters data frame
  #' @param label_mapping Label mapping data frame
  #' @param filtered_subtypes Filtered leukemia subtypes
  #' @return List of probability data frames organized by outer fold

  generate_standard_probability_matrices <- function(cv_results_df, best_params_df, label_mapping, filtered_subtypes, study_names, merge_classes = FALSE, merge_mds_only = FALSE) {
    outer_fold_ids <- unique(cv_results_df$outer_fold)
    probability_matrices <- list()

    for (outer_fold_id in outer_fold_ids) {
      outer_fold_data <- cv_results_df[cv_results_df$outer_fold == outer_fold_id, ]
      inner_fold_ids <- unique(outer_fold_data$inner_fold)

      fold_matrices <- list()

      for (inner_fold_id in inner_fold_ids) {
        best_param <- best_params_df$params[1]


        inner_fold_data <- outer_fold_data[
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
        probability_matrix$inner_fold <- inner_fold_id # this is the left out fold for the inner cv
        probability_matrix$outer_fold <- outer_fold_id # outer left out fold, more an id of the cv run
        probability_matrix$indices <- parse_numeric_string(inner_fold_data$sample_indices) + 1
        probability_matrix$study <- study_names[probability_matrix$indices]

        # Apply class merging if requested
        if (merge_classes) {
          probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_mds_only = merge_mds_only)
        }

        fold_matrices[[as.character(inner_fold_id)]] <- probability_matrix
      }

      probability_matrices[[as.character(outer_fold_id)]] <- do.call(rbind, fold_matrices)
    }

    probability_matrices
  }

  #' Align probability matrices from different models for ensemble analysis (train/test version)
  #' Uses the unified function from utility_functions.R
  align_probability_matrices_train_test <- function(prob_matrices, outer_fold_name, type) {
    align_probability_matrices(prob_matrices, outer_fold_name, inner_fold_name = NULL, type)
  }

  #' Perform One-vs-Rest ensemble analysis for train/test (parallelized)
  #' Uses the unified function from utility_functions.R
  perform_ovr_ensemble_analysis_train_test <- function(results, weights, type = "cv") {
    perform_ovr_ensemble_analysis_unified(results, weights, type, has_inner_folds = FALSE)
  }

  #' Perform global ensemble optimization using overall kappa for train/test (parallelized)
  #' Uses the unified function from utility_functions.R
  perform_global_ensemble_analysis_train_test <- function(results, weights, type = "cv") {
    perform_global_ensemble_analysis_unified(results, weights, type, has_inner_folds = FALSE)
  }

  #' Generate One-vs-Rest optimized ensemble probability matrices for train/test
  #' @param results Analysis results containing probability matrices
  #' @param weights Weight configurations for ensemble
  #' @param type Type of analysis ("cv" or "loso")
  #' @param ensemble_performance Performance results from perform_ovr_ensemble_analysis_train_test
  #' @return List containing optimized probability matrices and weights used for each outer fold
  generate_ovr_optimized_ensemble_matrices_train_test <- function(results, weights, type = "cv", ensemble_performance) {
    cat("Generating One-vs-Rest optimized ensemble probability matrices for train/test...\n")

    outer_folds <- names(results$probability_matrices$svm[[type]])
    optimized_matrices <- list()
    weights_used <- list()

    # Aggregate performance across all folds to find globally best weights for each class
    cat("  Aggregating performance across all folds to find globally best weights for each class...\n")
    all_performance <- list()
    for (outer_fold in outer_folds) {
      fold_performance <- ensemble_performance[[outer_fold]]
      if (!is.null(fold_performance) && nrow(fold_performance) > 0) {
        all_performance[[outer_fold]] <- fold_performance
      }
    }

    if (length(all_performance) == 0) {
      cat("    No performance data available across all folds\n")
      return(list(matrices = optimized_matrices, weights_used = weights_used))
    }

    # Combine all performance data
    combined_performance <- do.call(rbind, all_performance)

    # Get all classes that have performance data
    all_available_classes <- unique(combined_performance$class)

    # For each class, find globally best weights based on mean F1 score across all folds
    global_class_weights <- list()
    for (class_name in all_available_classes) {
      class_performance <- combined_performance[combined_performance$class == class_name, ]

      if (nrow(class_performance) > 0) {
        # Check if we have valid F1 scores to aggregate
        valid_f1_scores <- class_performance[!is.na(class_performance$f1_score) & !is.null(class_performance$f1_score), ]

        if (nrow(valid_f1_scores) == 0) {
          cat(sprintf("    Warning: No valid F1 scores for class %s, using default weights\n", class_name))
          global_class_weights[[class_name]] <- list(
            weight_name = "ALL",
            weights = weights[["ALL"]],
            f1_score = 0
          )
          next
        }

        # Calculate mean F1 score for each weight configuration for this class
        mean_performance <- aggregate(f1_score ~ weights, data = valid_f1_scores, FUN = mean, na.rm = TRUE)

        # Check if aggregation produced any results
        if (nrow(mean_performance) == 0) {
          cat(sprintf("    Warning: No aggregated performance data for class %s, using default weights\n", class_name))
          global_class_weights[[class_name]] <- list(
            weight_name = "ALL",
            weights = weights[["ALL"]],
            f1_score = 0
          )
          next
        }

        # Find best weight configuration for this class
        best_weight_indices <- which.max(mean_performance$f1_score)
        best_weight_name <- mean_performance$weights[best_weight_indices]

        # Ensure we have a single weight name
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

        # Store globally best weight configuration for this class
        global_class_weights[[class_name]] <- list(
          weight_name = best_weight_name,
          weights = best_weights,
          f1_score = max(mean_performance$f1_score, na.rm = TRUE)
        )

        cat(sprintf("  Globally best weights for class %s: %s (mean F1=%.4f)\n",
                    class_name, best_weight_name, max(mean_performance$f1_score, na.rm = TRUE)))
      } else {
        cat(sprintf("    Warning: No performance data for class %s, using default weights\n", class_name))
        global_class_weights[[class_name]] <- list(
          weight_name = "ALL",
          weights = weights[["ALL"]],
          f1_score = 0
        )
      }
    }

    for (outer_fold in outer_folds) {
      cat(sprintf("  Creating OvR optimized matrices for outer fold %s...\n", outer_fold))

      # Store the globally best weight configurations used for this outer fold
      outer_fold_weights_used <- global_class_weights

      # Generate optimized matrices using the selected weights (with caching)
      alignment_cache <- new.env(hash = TRUE)
      aligned_matrices <- align_probability_matrices_cached(
        results$probability_matrices, outer_fold, NULL, type, alignment_cache
      )
      if (is.null(aligned_matrices)) {
        cat(sprintf("      Skipping outer fold %s - unable to align matrices\n", outer_fold))
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

        # Find the globally best weights to use for this class
        best_weights <- NULL
        if (clean_class_name %in% names(global_class_weights)) {
          best_weights <- global_class_weights[[clean_class_name]]$weights
        } else if (clean_class_name_no_dots %in% names(global_class_weights)) {
          best_weights <- global_class_weights[[clean_class_name_no_dots]]$weights
        } else {
          # Try partial matching
          matching_classes <- names(global_class_weights)[
            grepl(clean_class_name, names(global_class_weights), ignore.case = TRUE) |
              grepl(clean_class_name_no_dots, names(global_class_weights), ignore.case = TRUE)
          ]
          if (length(matching_classes) > 0) {
            best_weights <- global_class_weights[[matching_classes[1]]]$weights
          }
        }

        # Use default weights if no specific weights found for this class
        if (is.null(best_weights)) {
          cat(sprintf("      Using default weights for class %s (not present in global performance)\n", clean_class_name))
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

      optimized_matrices[[outer_fold]] <- optimized_matrix
      weights_used[[outer_fold]] <- outer_fold_weights_used
    }

    # Return both matrices and weights used
    list(
      matrices = optimized_matrices,
      weights_used = weights_used
    )
  }

  #' Generate globally optimized ensemble probability matrices for train/test
  #' @param results Analysis results containing probability matrices
  #' @param weights Weight configurations for ensemble
  #' @param type Type of analysis ("cv" or "loso")
  #' @param ensemble_performance Performance results from perform_global_ensemble_analysis_train_test
  #' @return List containing optimized probability matrices and weights used for each outer fold
  generate_global_optimized_ensemble_matrices_train_test <- function(results, weights, type = "cv", ensemble_performance) {
    cat("Generating globally optimized ensemble probability matrices for train/test...\n")

    outer_folds <- names(results$probability_matrices$svm[[type]])
    optimized_matrices <- list()
    weights_used <- list()

    # Aggregate performance across all folds to find globally best weights
    cat("  Aggregating performance across all folds to find globally best weights...\n")
    all_performance <- list()
    for (outer_fold in outer_folds) {
      fold_performance <- ensemble_performance[[outer_fold]]
      if (!is.null(fold_performance) && nrow(fold_performance) > 0) {
        all_performance[[outer_fold]] <- fold_performance
      }
    }

    if (length(all_performance) == 0) {
      cat("    No performance data available across all folds\n")
      return(list(matrices = optimized_matrices, weights_used = weights_used))
    }

    # Combine all performance data
    combined_performance <- do.call(rbind, all_performance)

    # Calculate mean kappa for each weight configuration across all folds
    mean_performance <- aggregate(kappa ~ weights, data = combined_performance, FUN = mean, na.rm = TRUE)

    # Find globally best weight configuration
    global_best_weight_name <- mean_performance$weights[which.max(mean_performance$kappa)]
    global_best_weights <- weights[[global_best_weight_name]]
    global_best_kappa <- max(mean_performance$kappa, na.rm = TRUE)

    cat(sprintf("  Globally best weights: %s (mean kappa = %.4f)\n", global_best_weight_name, global_best_kappa))

    for (outer_fold in outer_folds) {
      cat(sprintf("  Creating globally optimized matrices for outer fold %s...\n", outer_fold))

      # Store the globally best weight configuration used for this outer fold
      weights_used[[outer_fold]] <- list(
        weight_name = global_best_weight_name,
        weights = global_best_weights,
        kappa = global_best_kappa
      )

      cat(sprintf("    Using globally optimized weights (%s) for outer fold %s (mean kappa = %.4f)\n",
                  global_best_weight_name, outer_fold, global_best_kappa))

      # Align probability matrices for this outer fold (with caching)
      alignment_cache <- new.env(hash = TRUE)
      aligned_matrices <- align_probability_matrices_cached(
        results$probability_matrices, outer_fold, NULL, type, alignment_cache
      )
      if (is.null(aligned_matrices)) {
        cat(sprintf("      Skipping outer fold %s - unable to align matrices\n", outer_fold))
        next
      }

      # Convert to matrices once for efficiency
      prob_mat_SVM <- as.matrix(aligned_matrices$svm)
      prob_mat_XGB <- as.matrix(aligned_matrices$xgboost)
      prob_mat_NN <- as.matrix(aligned_matrices$neural_net)
      non_prob_cols <- aligned_matrices$non_prob_cols

      # Calculate weighted ensemble probabilities using globally best weights (matrix operations)
      optimized_matrix <- prob_mat_SVM * global_best_weights$SVM +
        prob_mat_XGB * global_best_weights$XGB +
        prob_mat_NN * global_best_weights$NN

      # Normalize probabilities to sum to 1 for each sample
      row_sums <- rowSums(optimized_matrix)
      optimized_matrix <- optimized_matrix / row_sums

      # Convert to data frame and add true labels
      optimized_matrix <- data.frame(optimized_matrix)
      optimized_matrix <- cbind(optimized_matrix, non_prob_cols)

      optimized_matrices[[outer_fold]] <- optimized_matrix
    }

    # Return both matrices and weights used
    list(
      matrices = optimized_matrices,
      weights_used = weights_used
    )
  }

  #' Calculate performance metrics for optimized ensemble matrices (train/test version)
  #' @param ensemble_result Result containing optimized ensemble probability matrices and weights
  #' @param type Type of analysis ("cv" or "loso")
  #' @return Performance results for each outer fold
  analyze_optimized_ensemble_performance_train_test <- function(ensemble_result, type = "cv") {
    cat("Analyzing optimized ensemble performance for train/test...\n")

    # Handle different input structures
    if (is.list(ensemble_result) && "matrices" %in% names(ensemble_result)) {
      optimized_matrices <- ensemble_result$matrices
    } else {
      optimized_matrices <- ensemble_result
    }

    performance_results <- list()

    for (outer_fold_name in names(optimized_matrices)) {
      cat(sprintf("  Analyzing outer fold %s...\n", outer_fold_name))

      # Get the optimized matrix for this outer fold
      optimized_matrix <- optimized_matrices[[outer_fold_name]]

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

    performance_results
  }

  #' Evaluate nested CV kappa with rejection for a single probability matrix (train/test version)
  #' Uses the unified function from utility_functions.R
  evaluate_single_matrix_with_rejection_train_test <- function(prob_matrix, fold_name, model_name, type) {
    evaluate_single_matrix_with_rejection_parallel(prob_matrix, fold_name, model_name, type)
  }

  #' Evaluate rejection analysis for all probability matrices (train/test version, parallelized)
  #' Uses the unified function from utility_functions.R
  evaluate_all_matrices_with_rejection_train_test <- function(probability_matrices, ensemble_matrices, type = "cv") {
    evaluate_all_matrices_with_rejection_unified(probability_matrices, ensemble_matrices, type, has_inner_folds = FALSE)
  }

  #' Find optimal probability cutoff for each model/ensemble per outer fold (train/test version)
  #' @param rejection_results Data frame with rejection analysis results
  #' @param optimization_metric Metric to optimize ("kappa" or "accuracy")
  #' @return List with optimal cutoffs per outer fold and summary statistics
  find_optimal_cutoffs_train_test <- function(rejection_results, optimization_metric = "kappa") {
    cat("Finding optimal probability cutoffs per outer fold (train/test)...\n")

    # optimal_cutoffs_per_outer_fold <- rejection_results %>%
    #   filter((is.na(rejected_accuracy) | rejected_accuracy < 0.5) & (perc_rejected < 0.05)) %>%
    #   mutate(rejected_accuracy_mod = ifelse(is.na(rejected_accuracy), 0, rejected_accuracy)) %>%
    #   group_by(model, outer_fold) %>%
    #   slice_max(kappa) %>%
    #   slice_min(rejected_accuracy_mod, with_ties = F) %>%
    #   ungroup()

    optimal_cutoffs_per_outer_fold <- rejection_results %>%
      mutate(rejected_accuracy_mod = ifelse(is.na(rejected_accuracy), 0, rejected_accuracy)) %>%
      group_by(model, prob_cutoff) %>%
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
      group_by(model) %>%
      slice_max(mean_kappa) %>%
      slice_min(mean_rejected_accuracy, with_ties = FALSE) %>%
      ungroup()

    summary_stats <- optimal_cutoffs_per_outer_fold %>%
      group_by(model) %>%
      arrange(desc(mean_kappa))

    # Calculate summary statistics across outer folds for each model
    # summary_stats <- optimal_cutoffs_per_outer_fold %>%
    #   group_by(model) %>%
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
    #   ) %>%
    #   arrange(desc(mean_kappa))

    return(list(
      optimal_cutoffs_per_outer_fold = optimal_cutoffs_per_outer_fold,
      summary_stats = summary_stats
    ))
  }

  #' Compare performance with and without rejection for train/test analysis
  #' @param type Analysis type ("cv" or "loso")
  #' @param train_test_results Train/test results object
  #' @return Combined data frame with performance metrics with and without rejection
  compare_all_results_train_test <- function(type, train_test_results){
    # Extract performance without rejection
    df_no_rejection <- train_test_results[["performance_comparisons"]][[type]] %>%
      as_tibble() %>%
      rename(model = Method,
             mean_kappa = Mean_Kappa) %>%
      select(model, mean_kappa) %>%
      mutate(model = str_to_lower(model))

    # Extract performance with rejection
    df_rejection <- train_test_results[["rejection_results"]][[type]][["optimal_results"]][["summary_stats"]] %>%
      select(model, mean_kappa_with_rejection = mean_kappa, mean_perc_rejected) %>%
      mutate(model = str_to_lower(model))

    # Combine
    combined_df <- df_no_rejection %>%
      left_join(df_rejection, by = "model")

    return(combined_df)
  }

  #' Combine all results for both CV and LOSO train/test analysis
  #' @param train_test_results Train/test results object
  #' @return List with combined results for CV and LOSO
  combine_all_results_train_test <- function(train_test_results){
    combined_results <- list()
    for (type in c("cv", "loso")){
      if (type %in% names(train_test_results[["performance_comparisons"]])) {
        combined_results[[type]] <- compare_all_results_train_test(type, train_test_results)
      }
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
  leukemia_subtypes <- read.csv("../data/rgas_18dec25.csv")$ICC_Subtype

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

  dir.create("../data/out/final_train_test/best_params")

  MODEL_CONFIGS <- list(
    svm = list(
      classification_type = "OvR",
      file_paths = list(
        cv = "../data/out/final_train_test/SVM_final_selection/final_cv_svm/",
        loso = "../data/out/final_train_test/SVM_final_selection/final_loso_svm/"
      ),
      output_dir = "../data/out/final_train_test/best_params/SVM"
    ),
    xgboost = list(
      classification_type = "OvR",
      file_paths = list(
        cv = "../data/out/final_train_test/XGBOOST_final_selection/final_cv_xgb/",
        loso = "../data/out/final_train_test/XGBOOST_final_selection/final_loso_xgb/"
      ),
      output_dir = "../data/out/final_train_test/best_params/XGBOOST"
    ),
    neural_net = list(
      classification_type = "standard",
      file_paths = list(
        cv = "../data/out/final_train_test/NN_final_selection/final_cv_nn/",
        loso = "../data/out/final_train_test/NN_final_selection/final_loso_nn/"
      ),
      output_dir = "../data/out/final_train_test/best_params/NN"
    )
  )

  model_results <- load_all_model_data(MODEL_CONFIGS, group_nn_by_outer_fold = FALSE)

  # Extract best parameters
  best_parameters <- extract_all_best_parameters(model_results, MODEL_CONFIGS, include_outer_fold = FALSE)

  # Save best parameters
  save_all_best_parameters(best_parameters, MODEL_CONFIGS)

  probability_matrices <- list()
  for (model_name in names(model_results)) {
    config <- MODEL_CONFIGS[[model_name]]
    cat(sprintf("Extracting %s probabilities...\n", toupper(model_name)))

    probability_matrices[[model_name]] <- list()

    for (fold_type in names(model_results[[model_name]])) {
      results <- model_results[[model_name]][[fold_type]]
      best_params <- best_parameters[[model_name]][[fold_type]]

      if (!is.null(results) && !is.null(best_params)) {
        if (config$classification_type == "OvR") {
          probs <- generate_ovr_probability_matrices(results, best_params, label_mapping, study_names, merge_classes = merge_classes, merge_mds_only = merge_mds_only)
        } else {
          probs <- generate_standard_probability_matrices(results, best_params, label_mapping, filtered_leukemia_subtypes, study_names, merge_classes = merge_classes, merge_mds_only = merge_mds_only)
        }

        # No class grouping - keep original classes
        # probs remain unchanged

        probability_matrices[[model_name]][[fold_type]] <- probs
        remove(probs)
      }
    }
  }

  per_model_results <- get_per_model_performance(probability_matrices)

  per_class_results <- list()
  i <- 1
  for (model in unique(per_model_results$model)){
    for (fold in unique(per_model_results$fold_name)){
      subset <- per_model_results[per_model_results$model == model & per_model_results$fold_name==fold,]
      subset <- subset$confusion_matrix[[1]]
      subset <- subset$byClass
      subset <- data.frame(subset)
      subset$class <- rownames(subset)
      subset$model <- model
      subset$fold <- fold
      per_class_results[[i]] <- subset
      i = i +1
    }
  }

  per_class_results <- do.call(rbind, per_class_results)

  # Run ensemble analysis for both CV and LOSO
  cat("Running ensemble analysis for train/test...\n")
  ensemble_results <- list()

  for (analysis_type in c("cv", "loso")) {
    cat(sprintf("\n=== Running %s analysis ===\n", toupper(analysis_type)))

    # Check if we have data for this analysis type
    if (!all(sapply(probability_matrices, function(x) analysis_type %in% names(x)))) {
      cat(sprintf("Skipping %s analysis - missing data\n", toupper(analysis_type)))
      next
    }

    # Perform global ensemble analysis
    global_ensemble_results <- perform_global_ensemble_analysis_train_test(
      list(probability_matrices = probability_matrices),
      generate_weights(),
      analysis_type
    )

    # Generate globally optimized ensemble matrices
    global_optimized_ensemble_matrices <- generate_global_optimized_ensemble_matrices_train_test(
      list(probability_matrices = probability_matrices),
      generate_weights(),
      analysis_type,
      global_ensemble_results
    )

    # Analyze globally optimized ensemble performance
    global_optimized_ensemble_performance <- analyze_optimized_ensemble_performance_train_test(global_optimized_ensemble_matrices, analysis_type)

    # Perform One-vs-Rest ensemble analysis
    ovr_ensemble_results <- perform_ovr_ensemble_analysis_train_test(
      list(probability_matrices = probability_matrices),
      generate_weights(),
      analysis_type
    )

    # Generate One-vs-Rest optimized ensemble matrices
    ovr_optimized_result <- generate_ovr_optimized_ensemble_matrices_train_test(
      list(probability_matrices = probability_matrices),
      generate_weights(),
      analysis_type,
      ovr_ensemble_results
    )

    # Analyze One-vs-Rest ensemble multiclass performance
    ovr_ensemble_multiclass_performance <- analyze_optimized_ensemble_performance_train_test(ovr_optimized_result, analysis_type)

    # Store results for this analysis type
    ensemble_results[[analysis_type]] <- list(
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

  # Run rejection analysis
  cat("Running rejection analysis for train/test...\n")
  rejection_results <- list()

  for (analysis_type in c("cv", "loso")) {
    cat(sprintf("\n=== Running rejection analysis for %s ===\n", toupper(analysis_type)))

    # Check if we have ensemble results for this analysis type
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
    rejection_results[[analysis_type]][["all_results"]] <- evaluate_all_matrices_with_rejection_train_test(
      probability_matrices, ensemble_matrices, analysis_type
    )

    # Find optimal cutoffs
    rejection_results[[analysis_type]][["optimal_results"]] <- find_optimal_cutoffs_train_test(rejection_results[[analysis_type]][["all_results"]], "kappa")
  }

  # Determine suffix for file paths (maxprob method - uses max probability instead of summing)
  if (!merge_classes) {
    merge_suffix <- "_unmerged_maxprob"
  } else if (merge_mds_only) {
    merge_suffix <- "_mds_only_maxprob"
  } else {
    merge_suffix <- "_merged_maxprob"
  }

  # Save ensemble weights
  weights_dir <- paste0("../data/out/final_train_test/ensemble_weights", merge_suffix)
  dir.create(weights_dir, recursive = TRUE)
  save_ensemble_weights(ensemble_results, weights_dir, save_per_fold = FALSE)

  # Save cutoffs
  cutoff_dir <- paste0("../data/out/final_train_test/cutoffs", merge_suffix)
  dir.create(cutoff_dir, recursive = TRUE)

  # Extract and save the optimal cutoffs
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

  # Combine and save cutoffs
  combined_cutoffs <- data.frame()
  if (!is.null(cv_cutoffs)) {
    combined_cutoffs <- rbind(combined_cutoffs, cv_cutoffs)
  }
  if (!is.null(loso_cutoffs)) {
    combined_cutoffs <- rbind(combined_cutoffs, loso_cutoffs)
  }

  if (nrow(combined_cutoffs) > 0) {
    write.csv(combined_cutoffs, file.path(cutoff_dir, paste0("train_test_cutoffs", merge_suffix, ".csv")), row.names = FALSE)
  } else {
    cat("Warning: No optimal cutoffs found to save\n")
  }

  # Calculate performance comparisons
  cat("Calculating performance comparisons...\n")
  performance_comparisons <- list()

  for (analysis_type in c("cv", "loso")) {
    if (analysis_type %in% names(ensemble_results)) {

      # Individual model performance
      individual_performance <- list()
      for (model_name in c("svm", "xgboost", "neural_net")) {
        if (analysis_type %in% names(probability_matrices[[model_name]])) {
          all_kappas <- c()

          for (outer_fold in names(probability_matrices[[model_name]][[analysis_type]])) {
            optimized_matrix <- probability_matrices[[model_name]][[analysis_type]][[outer_fold]]

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

          individual_performance[[toupper(model_name)]] <- list(
            mean_kappa = mean(all_kappas, na.rm = TRUE),
            sd_kappa = sd(all_kappas, na.rm = TRUE),
            fold_kappas = all_kappas
          )
        }
      }

      # Ensemble performance
      ensemble_methods <- list(
        "OvR_Ensemble" = ensemble_results[[analysis_type]]$ovr_ensemble_multiclass_performance,
        "Global_Optimized" = ensemble_results[[analysis_type]]$global_optimized_ensemble_performance
      )

      for (method_name in names(ensemble_methods)) {
        method_performance <- ensemble_methods[[method_name]]
        all_kappas <- c()

        for (outer_fold in names(method_performance)) {
          cm <- method_performance[[outer_fold]]
          if (inherits(cm, "confusionMatrix")) {
            all_kappas <- c(all_kappas, cm$overall["Kappa"])
          }
        }

        individual_performance[[method_name]] <- list(
          mean_kappa = mean(all_kappas, na.rm = TRUE),
          sd_kappa = sd(all_kappas, na.rm = TRUE),
          fold_kappas = all_kappas
        )
      }

      # Create summary data frame
      summary_df <- data.frame(
        Method = names(individual_performance),
        Mean_Kappa = sapply(individual_performance, function(x) x$mean_kappa),
        SD_Kappa = sapply(individual_performance, function(x) x$sd_kappa),
        stringsAsFactors = FALSE
      )

      # Sort by mean kappa (descending)
      summary_df <- summary_df[order(summary_df$Mean_Kappa, decreasing = TRUE), ]

      performance_comparisons[[analysis_type]] <- summary_df

      cat(sprintf("\n=== %s Performance Summary ===\n", toupper(analysis_type)))
      print(summary_df)
    }
  }

  # Create comprehensive results structure for final model building
  train_test_results <- list(
    model_results = model_results,
    best_parameters = best_parameters,
    probability_matrices = probability_matrices,
    per_model_results = per_model_results,
    per_class_results = per_class_results,
    ensemble_results = ensemble_results,
    rejection_results = rejection_results,
    optimal_cutoffs = combined_cutoffs,
    performance_comparisons = performance_comparisons
  )

  # Add final results comparison (with and without rejection)
  train_test_results$final_results <- combine_all_results_train_test(train_test_results)
  train_test_results$merge_classes <- merge_classes  # Store merge status in results
  train_test_results$merge_mds_only <- merge_mds_only  # Store merge_mds_only status in results

  cat("\n=== Final Train/Test Results Summary ===\n")
  print(train_test_results$final_results)

  saveRDS(train_test_results, paste0("../data/out/final_train_test/final_train_test_results_5jan2025", merge_suffix, ".rds"))
  return(train_test_results)
}

# Run merged and MDS-only merged versions (maxprob method)
cat("=== Running Train/Test Analysis (Merged - MaxProb Method) ===\n")
train_test_results_merged <- main_train_test_analysis(merge_classes = TRUE, merge_mds_only = FALSE)

#cat("=== Running Train/Test Analysis (MDS Only Merged - MaxProb Method) ===\n")
#train_test_results_mds_only <- main_train_test_analysis(merge_classes = TRUE, merge_mds_only = TRUE)
