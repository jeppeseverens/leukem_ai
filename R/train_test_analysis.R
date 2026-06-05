# Source shared utility functions
source("utility_functions.R")

main_train_test_analysis <- function(merge_classes = FALSE){

  #' Map filtered-dataset 0-based indices to full-dataset 1-based indices.
  #' Train/test CV files store sample_indices in filtered-index space.
  map_filtered_local_to_global_indices <- function(sample_indices_zero_based, filter_vec) {
    local_one_based <- sample_indices_zero_based + 1L
    if (length(local_one_based) == 0) return(integer(0))
    if (any(local_one_based < 1L | local_one_based > length(filter_vec))) {
      stop(
        sprintf(
          "Found sample_indices outside filtered index range [0, %d]. Example values: %s",
          length(filter_vec) - 1L,
          paste(head(sample_indices_zero_based, 15), collapse = ", ")
        )
      )
    }
    as.integer(filter_vec[local_one_based])
  }

  #' Generate probability data frames for One-vs-Rest classification
  #' @param cv_results_df Cross-validation results data frame
  #' @param best_params_df Best parameters data frame
  #' @param label_mapping Label mapping data frame
  #' @return List of probability data frames organized by outer fold
  #'
  generate_ovr_probability_matrices <- function(cv_results_df, best_params_df, label_mapping, study_names, merge_classes = FALSE) {
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
        local_idx <- parse_numeric_string(inner_fold_data$sample_indices[1])
        probability_matrix$indices <- map_filtered_local_to_global_indices(local_idx, filter)
        probability_matrix$study <- study_names[probability_matrix$indices]

        # Apply class merging if requested (summed method, consistent with inner CV)
        if (merge_classes) {
          probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_prob_method = "sum")
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

          # Extract true labels and remove non-probability / meta columns
          truth <- the_matrix$y
          prob_matrix <- the_matrix[, !colnames(the_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study", "confidence_multivariate"), drop = FALSE]

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

  generate_standard_probability_matrices <- function(cv_results_df, best_params_df, label_mapping, filtered_subtypes, study_names, merge_classes = FALSE) {
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
        local_idx <- parse_numeric_string(inner_fold_data$sample_indices)
        probability_matrix$indices <- map_filtered_local_to_global_indices(local_idx, filter)
        probability_matrix$study <- study_names[probability_matrix$indices]

        # Apply class merging if requested (summed method, consistent with inner CV)
        if (merge_classes) {
          probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_prob_method = "sum")
        }

        fold_matrices[[as.character(inner_fold_id)]] <- probability_matrix
      }

      probability_matrices[[as.character(outer_fold_id)]] <- do.call(rbind, fold_matrices)
    }

    probability_matrices
  }

  #' Find newest CSV matching regex pattern in a directory
  find_latest_csv <- function(dir_path, pattern) {
    if (!dir.exists(dir_path)) return(NULL)
    files <- list.files(dir_path, pattern = pattern, full.names = TRUE)
    if (length(files) == 0) return(NULL)
    files[which.max(file.info(files)$mtime)]
  }

  #' Build left-out probability matrix for one model from final model left-out CSV
  build_leftout_probability_matrix <- function(leftout_csv, model_type, label_mapping, all_subtypes, merge_classes = FALSE) {
    if (is.null(leftout_csv) || !file.exists(leftout_csv)) return(NULL)
    lo <- read.csv(leftout_csv, stringsAsFactors = FALSE)
    if (nrow(lo) == 0) return(NULL)

    if (!"sample_indices" %in% colnames(lo) || !"preds_prob" %in% colnames(lo)) return(NULL)
    sample_indices <- parse_numeric_string(lo$sample_indices[1])
    if (length(sample_indices) == 0) return(NULL)

    if (model_type %in% c("svm", "xgboost")) {
      classes <- unique(lo$class)
      prob_mat <- matrix(NA_real_, nrow = length(sample_indices), ncol = length(classes))
      colnames(prob_mat) <- make.names(classes)
      for (j in seq_along(classes)) {
        row_j <- lo[lo$class == classes[j], , drop = FALSE]
        if (nrow(row_j) == 0) next
        probs <- parse_numeric_string(row_j$preds_prob[1])
        if (length(probs) == length(sample_indices)) prob_mat[, j] <- probs
      }
      prob_df <- as.data.frame(prob_mat)
    } else {
      class_indices <- unique(parse_numeric_string(lo$classes[1]))
      class_labels <- label_mapping$Label[class_indices + 1]
      probs <- parse_numeric_string(lo$preds_prob[1])
      if (length(class_labels) == 0 || length(probs) == 0) return(NULL)
      prob_mat <- t(matrix(probs, ncol = length(sample_indices), nrow = length(class_labels)))
      colnames(prob_mat) <- make.names(class_labels)
      prob_df <- as.data.frame(prob_mat)
    }

    prob_df <- ensure_all_class_columns(prob_df, label_mapping)
    prob_df <- t(apply(prob_df, 1, function(row) {
      s <- sum(row, na.rm = TRUE)
      if (s > 0) row / s else row
    }))
    prob_df <- as.data.frame(prob_df)
    prob_df$y <- make.names(all_subtypes[sample_indices + 1])
    prob_df$indices <- sample_indices + 1
    # Keep a 0-based sample index in parallel so we can align to outer-CV
    # leftout fold assignments (which are written in 0-based form).
    prob_df$sample_indices <- sample_indices
    prob_df$is_leftout <- TRUE

    # Keep left-out rows fully uncollapsed so OOD subtypes are not folded into
    # merged classes such as other.KMT2A.
    prob_df
  }

  #' Load per-fold left-out sample assignment from an outer CV leftout CSV.
  #' The outer CV leftout CSV has one row per outer fold with the
  #' `sample_indices` (0-based) of the leftout samples assigned to that fold.
  #' Returns a data frame with columns `sample_index` (0-based) and `outer_fold`
  #' (character) or NULL if the file is missing / unreadable.
  load_leftout_fold_assignment <- function(outer_cv_leftout_csv) {
    if (is.null(outer_cv_leftout_csv) || !file.exists(outer_cv_leftout_csv)) return(NULL)
    df <- read.csv(outer_cv_leftout_csv, stringsAsFactors = FALSE)
    if (nrow(df) == 0 || !all(c("sample_indices", "outer_fold") %in% colnames(df))) return(NULL)
    rows <- lapply(seq_len(nrow(df)), function(i) {
      idx <- parse_numeric_string(df$sample_indices[i])
      if (length(idx) == 0) return(NULL)
      data.frame(
        sample_index = idx,
        outer_fold = as.character(df$outer_fold[i]),
        stringsAsFactors = FALSE
      )
    })
    rows <- Filter(Negate(is.null), rows)
    if (length(rows) == 0) return(NULL)
    do.call(rbind, rows)
  }

  #' Append left-out samples to each outer fold matrix for a model.
  #' If `fold_assignment` is supplied, each fold only receives the leftout
  #' samples assigned to that fold (mirrors `assign_leftout_to_cv_folds` used
  #' by outer CV). When NULL, the full leftout matrix is appended to every
  #' fold (legacy behaviour).
  augment_model_with_leftout <- function(model_fold_matrices, leftout_matrix,
                                         fold_assignment = NULL) {
    if (is.null(leftout_matrix) || !is.list(model_fold_matrices)) return(model_fold_matrices)
    out <- list()
    can_partition <- !is.null(fold_assignment) &&
      is.data.frame(fold_assignment) &&
      all(c("sample_index", "outer_fold") %in% colnames(fold_assignment)) &&
      "sample_indices" %in% colnames(leftout_matrix)

    for (fold_name in names(model_fold_matrices)) {
      known <- model_fold_matrices[[fold_name]]
      known$is_leftout <- FALSE

      if (can_partition) {
        allowed_idx <- fold_assignment$sample_index[
          as.character(fold_assignment$outer_fold) == as.character(fold_name)
        ]
        lo <- leftout_matrix[leftout_matrix$sample_indices %in% allowed_idx, , drop = FALSE]
      } else {
        lo <- leftout_matrix
      }

      lo$outer_fold <- fold_name
      lo$inner_fold <- NA
      lo$study <- NA

      # Left-out augmentation must not duplicate indices already present in the
      # known fold matrix. Duplicates indicate index-space mismatch or data
      # leakage and must be fixed upstream, not silently collapsed downstream.
      if ("indices" %in% colnames(known) && "indices" %in% colnames(lo)) {
        known_idx <- known$indices[is.finite(known$indices)]
        lo_idx <- lo$indices[is.finite(lo$indices)]
        overlap_idx <- sort(intersect(unique(known_idx), unique(lo_idx)))
        if (length(overlap_idx) > 0) {
          stop(sprintf(
            paste0(
              "Left-out augmentation produced duplicate sample indices in fold %s: ",
              "%d overlapping indices between known and left-out rows. ",
              "Example overlapping indices: %s. ",
              "This indicates index-space mismatch or left-out leakage; ",
              "fix index mapping before augmentation."
            ),
            fold_name,
            length(overlap_idx),
            paste(head(overlap_idx, 15), collapse = ", ")
          ))
        }
      }

      common_cols <- union(colnames(known), colnames(lo))
      for (cc in setdiff(common_cols, colnames(known))) known[[cc]] <- NA
      for (cc in setdiff(common_cols, colnames(lo))) lo[[cc]] <- NA
      if (nrow(lo) == 0) {
        out[[fold_name]] <- known[, common_cols, drop = FALSE]
      } else {
        out[[fold_name]] <- rbind(known[, common_cols, drop = FALSE],
                                   lo[, common_cols, drop = FALSE])
      }
    }
    out
  }

  #' Align probability matrices from different models for ensemble analysis (train/test version)
  #' Uses the unified function from utility_functions.R
  align_probability_matrices_train_test <- function(prob_matrices, outer_fold_name, type) {
    align_probability_matrices(prob_matrices, outer_fold_name, inner_fold_name = NULL, type)
  }

  #' Perform global ensemble optimization using product-of-experts for train/test
  #' (final model path: evaluate PoE with different expert exponents).
  perform_global_ensemble_analysis_train_test <- function(results, weights, type = "cv") {
    cat("Performing global ensemble analysis for train/test (product-of-experts)...\n")
    outer_folds <- names(results$probability_matrices$svm[[type]])
    all_results <- list()

    for (outer_fold in outer_folds) {
      aligned_matrices <- align_probability_matrices_cached(
        results$probability_matrices, outer_fold, NULL, type, new.env(hash = TRUE)
      )
      if (is.null(aligned_matrices)) next

      prob_mat_SVM <- as.matrix(aligned_matrices$svm)
      prob_mat_XGB <- as.matrix(aligned_matrices$xgboost)
      prob_mat_NN <- as.matrix(aligned_matrices$neural_net)
      class_names <- colnames(prob_mat_SVM)
      truth <- make.names(gsub("Class. ", "", aligned_matrices$non_prob_cols$y))

      fold_weight_results <- lapply(names(weights), function(weight_name) {
        w <- weights[[weight_name]]
        eps <- 1e-12
        poe_matrix <- (pmax(prob_mat_SVM, eps) ^ w$SVM) *
          (pmax(prob_mat_XGB, eps) ^ w$XGB) *
          (pmax(prob_mat_NN, eps) ^ w$NN)
        row_sums <- rowSums(poe_matrix)
        row_sums[row_sums == 0] <- 1
        poe_matrix <- poe_matrix / row_sums

        preds <- class_names[max.col(poe_matrix, ties.method = "first")]
        preds <- make.names(gsub("Class. ", "", preds))
        all_classes <- unique(c(truth, preds))
        truth_factor <- factor(truth, levels = all_classes)
        preds_factor <- factor(preds, levels = all_classes)

        data.frame(
          outer_fold = outer_fold,
          weights = weight_name,
          type = type,
          kappa = fast_kappa(preds_factor, truth_factor),
          accuracy = fast_accuracy(preds_factor, truth_factor),
          stringsAsFactors = FALSE
        )
      })

      all_results[[outer_fold]] <- do.call(rbind, fold_weight_results)
    }

    all_results
  }

  #' Generate One-vs-Rest optimized ensemble probability matrices for train/test
  #' @param results Analysis results containing probability matrices
  #' @param weights Weight configurations for ensemble
  #' @param type Type of analysis ("cv" only; LOSO removed)
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
  #' @param type Type of analysis ("cv" only; LOSO removed)
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

      # Product-of-experts in probability space: p ∝ Π_m p_m^{w_m}
      eps <- 1e-12
      optimized_matrix <- (pmax(prob_mat_SVM, eps) ^ global_best_weights$SVM) *
        (pmax(prob_mat_XGB, eps) ^ global_best_weights$XGB) *
        (pmax(prob_mat_NN, eps) ^ global_best_weights$NN)

      # Normalize probabilities to sum to 1 for each sample
      row_sums <- rowSums(optimized_matrix)
      row_sums[row_sums == 0] <- 1
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
  #' @param type Type of analysis ("cv" only; LOSO removed)
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

      # Extract true labels and remove non-probability / meta columns
      truth <- optimized_matrix$y
      prob_matrix <- optimized_matrix[, !colnames(optimized_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study", "confidence_multivariate"), drop = FALSE]

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
  #' @param target_risk Maximum acceptable error rate on accepted samples (e.g. 0.05 for 5%)
  #' @return List with optimal cutoffs per outer fold and summary statistics
  find_optimal_cutoffs_train_test <- function(rejection_results,
                                              optimization_metric = "kappa",
                                              target_risk = 0.02) {
    cat("Finding optimal probability cutoffs per outer fold (train/test)...\n")

    # optimal_cutoffs_per_outer_fold <- rejection_results %>%
    #   filter((is.na(rejected_accuracy) | rejected_accuracy < 0.5) & (perc_rejected < 0.05)) %>%
    #   mutate(rejected_accuracy_mod = ifelse(is.na(rejected_accuracy), 0, rejected_accuracy)) %>%
    #   group_by(model, outer_fold) %>%
    #   slice_max(kappa) %>%
    #   slice_min(rejected_accuracy_mod, with_ties = F) %>%
    #   ungroup()

    # Aggregate metrics per model / cutoff to get risk–coverage curve
    summarised <- rejection_results %>%
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
        mean_risk = 1 - mean_accuracy,
        mean_coverage = 1 - mean_perc_rejected,
        .groups = "drop"
      )

    # Select operating cutoff per model: mean_risk <= target_risk, then maximise
    # mean_coverage. High cutoffs can yield NA metrics when no samples are
    # accepted in any fold; filter those out before comparing so logical
    # expressions do not short-circuit into NA. If every row is NA for a
    # model, fall back to the lowest-cutoff row so the pipeline keeps going.
    optimal_cutoffs_per_outer_fold <- summarised %>%
      group_by(model) %>%
      arrange(mean_risk, desc(mean_coverage), desc(mean_kappa)) %>%
      group_modify(~{
        df <- .x
        usable <- !is.na(df$mean_risk) & !is.na(df$mean_coverage)
        if (!any(usable)) {
          return(df[1, , drop = FALSE])
        }
        df_usable <- df[usable, , drop = FALSE]
        meets <- df_usable$mean_risk <= target_risk
        if (any(meets)) {
          df_ok <- df_usable[meets, , drop = FALSE]
          best_cov <- max(df_ok$mean_coverage, na.rm = TRUE)
          df_ok <- df_ok[df_ok$mean_coverage == best_cov, , drop = FALSE]
          best_kappa <- suppressWarnings(max(df_ok$mean_kappa, na.rm = TRUE))
          if (is.finite(best_kappa)) {
            df_ok <- df_ok[!is.na(df_ok$mean_kappa) & df_ok$mean_kappa == best_kappa, , drop = FALSE]
          }
          df_ok[1, , drop = FALSE]
        } else {
          best_risk <- min(df_usable$mean_risk, na.rm = TRUE)
          df_r <- df_usable[df_usable$mean_risk == best_risk, , drop = FALSE]
          best_cov <- max(df_r$mean_coverage, na.rm = TRUE)
          df_r <- df_r[df_r$mean_coverage == best_cov, , drop = FALSE]
          best_kappa <- suppressWarnings(max(df_r$mean_kappa, na.rm = TRUE))
          if (is.finite(best_kappa)) {
            df_r <- df_r[!is.na(df_r$mean_kappa) & df_r$mean_kappa == best_kappa, , drop = FALSE]
          }
          df_r[1, , drop = FALSE]
        }
      }) %>%
      ungroup()

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
        sd_rejected_accuracy = sd(mean_rejected_accuracy, na.rm = TRUE),
        mean_perc_rejected = mean(mean_perc_rejected, na.rm = TRUE),
        sd_perc_rejected = sd(mean_perc_rejected, na.rm = TRUE),
        mean_risk = mean(mean_risk, na.rm = TRUE),
        sd_risk = sd(mean_risk, na.rm = TRUE),
        mean_coverage = mean(mean_coverage, na.rm = TRUE),
        sd_coverage = sd(mean_coverage, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      arrange(desc(mean_kappa))

    return(list(
      optimal_cutoffs_per_outer_fold = optimal_cutoffs_per_outer_fold,
      summary_stats = summary_stats,
      risk_coverage = summarised
    ))
  }

  #' Compare performance with and without rejection for train/test analysis
  #' @param type Analysis type ("cv" only)
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

    # Extract performance with rejection. Final deployment no longer selects
    # cutoffs in this script, so this block can be absent; keep NA placeholders
    # so downstream summaries remain well-formed.
    df_rejection <- NULL
    if (!is.null(train_test_results[["rejection_results"]]) &&
        type %in% names(train_test_results[["rejection_results"]]) &&
        "optimal_results" %in% names(train_test_results[["rejection_results"]][[type]]) &&
        "summary_stats" %in% names(train_test_results[["rejection_results"]][[type]][["optimal_results"]])) {
      df_rejection <- train_test_results[["rejection_results"]][[type]][["optimal_results"]][["summary_stats"]] %>%
        select(model, mean_kappa_with_rejection = mean_kappa, mean_perc_rejected) %>%
        mutate(model = str_to_lower(model))
    } else {
      df_rejection <- df_no_rejection %>%
        mutate(
          mean_kappa_with_rejection = NA_real_,
          mean_perc_rejected = NA_real_
        )
    }

    # Combine
    combined_df <- df_no_rejection %>%
      left_join(df_rejection, by = "model")

    return(combined_df)
  }

  #' Combine results for CV train/test analysis
  #' @param train_test_results Train/test results object
  #' @return List with combined results for CV
  combine_all_results_train_test <- function(train_test_results){
    combined_results <- list()
    if ("cv" %in% names(train_test_results[["performance_comparisons"]])) {
      combined_results[["cv"]] <- compare_all_results_train_test("cv", train_test_results)
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
  leukemia_subtypes <- read.csv("../data/rgas_10feb26.csv")$ICC_Subtype

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

  # CV only (hyperparameters for final model are selected from standard CV)
  MODEL_CONFIGS <- list(
    svm = list(
      classification_type = "OvR",
      file_paths = list(cv = "../data/out/final_train_test/SVM_final_selection/final_cv_march26/"),
      output_dir = "../data/out/final_train_test/best_params/SVM"
    ),
    xgboost = list(
      classification_type = "OvR",
      file_paths = list(cv = "../data/out/final_train_test/XGBOOST_final_selection/final_cv_march26//"),
      output_dir = "../data/out/final_train_test/best_params/XGBOOST"
    ),
    neural_net = list(
      classification_type = "standard",
      file_paths = list(cv = "../data/out/final_train_test/NN_final_selection/final_cv_march26/"),
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
          probs <- generate_ovr_probability_matrices(results, best_params, label_mapping, study_names, merge_classes = merge_classes)
        } else {
          probs <- generate_standard_probability_matrices(results, best_params, label_mapping, filtered_leukemia_subtypes, study_names, merge_classes = merge_classes)
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

  # Run ensemble analysis for CV (final model uses CV-selected hyperparameters)
  cat("Running ensemble analysis for train/test (CV)...\n")
  ensemble_results <- list()

  for (analysis_type in "cv") {
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

    # Store results for this analysis type (Global ensemble only; OvR removed)
    ensemble_results[[analysis_type]] <- list(
      global_ensemble_results = global_ensemble_results,
      global_optimized_ensemble_matrices = global_optimized_ensemble_matrices,
      global_optimized_ensemble_performance = global_optimized_ensemble_performance,
      global_ensemble_weights_used = global_optimized_ensemble_matrices$weights_used
    )
  }

  # Final train/test deployment no longer uses this pre-leftout multivariate pass.
  # We keep global ensemble matrices uncalibrated here and calibrate only through
  # the selected left-out two-head multivariate path below.
  cat("Skipping standalone multivariate calibration on base global ensemble matrices (train/test)...\n")

  # Deployment-only script: do not run train/test cutoff sweeps here.
  # Operating cutoffs should come from nested CV analysis outputs.
  rejection_results <- list()

  # Determine suffix for file paths (maxprob method - uses max probability instead of summing)
  if (!merge_classes) {
    merge_suffix <- "_unmerged_maxprob"
  } else {
    merge_suffix <- "_merged_summed"
  }

  # Save ensemble weights
  weights_dir <- paste0("../data/out/final_train_test/ensemble_weights", merge_suffix)
  dir.create(weights_dir, recursive = TRUE)
  save_ensemble_weights(ensemble_results, weights_dir, save_per_fold = FALSE)

  # Save cutoff artifacts directory (used by deployment parameter exports).
  cutoff_dir <- paste0("../data/out/final_train_test/cutoffs", merge_suffix)
  dir.create(cutoff_dir, recursive = TRUE)
  combined_cutoffs <- data.frame()

  # -------------------------------------------------------------------------
  # Left-out-aware augmentation for deployment calibration fitting
  # -------------------------------------------------------------------------
  cat("Preparing left-out-aware augmented matrices for deployment calibration fitting...\n")
  leftout_file_configs <- list(
    svm = find_latest_csv("../data/out/final_models/SVM", "^SVM_final_CV_OvR_leftout.*\\.csv$"),
    xgboost = find_latest_csv("../data/out/final_models/XGBOOST", "^XGBOOST_final_CV_OvR_leftout.*\\.csv$"),
    neural_net = find_latest_csv("../data/out/final_models/NN", "^NN_final_CV_standard_leftout.*\\.csv$")
  )

  has_all_leftout <- all(sapply(leftout_file_configs, function(p) !is.null(p) && file.exists(p)))
  if (has_all_leftout && "cv" %in% names(ensemble_results)) {
    cat("  Found final-model left-out prediction files. Building augmented matrices...\n")

    lo_svm <- build_leftout_probability_matrix(leftout_file_configs$svm, "svm", label_mapping, leukemia_subtypes, merge_classes = merge_classes)
    lo_xgb <- build_leftout_probability_matrix(leftout_file_configs$xgboost, "xgboost", label_mapping, leukemia_subtypes, merge_classes = merge_classes)
    lo_nn <- build_leftout_probability_matrix(leftout_file_configs$neural_net, "neural_net", label_mapping, leukemia_subtypes, merge_classes = merge_classes)

    if (!is.null(lo_svm) && !is.null(lo_xgb) && !is.null(lo_nn)) {
      # Load per-fold leftout assignment from any available outer CV leftout CSV
      # so train/test mirrors the partitioning used by outer CV. If none is
      # available, fall back to duplicating leftouts in every fold (legacy path).
      outer_cv_leftout_candidates <- c(
        find_latest_csv("../data/out/outer_cv/NN_n10_fs_eta",
                        "^NN_outer_cv_CV_standard_leftout_fs_eta_\\d+_\\d+\\.csv$"),
        find_latest_csv("../data/out/outer_cv/SVM_n10_fs_eta",
                        "^SVM_outer_cv_CV_OvR_leftout_fs_eta_\\d+_\\d+\\.csv$"),
        find_latest_csv("../data/out/outer_cv/XGBOOST_n10_fs_eta",
                        "^XGBOOST_outer_cv_CV_OvR_leftout_fs_eta_\\d+_\\d+\\.csv$")
      )
      outer_cv_leftout_candidates <- outer_cv_leftout_candidates[
        !sapply(outer_cv_leftout_candidates, is.null)
      ]
      leftout_fold_assignment <- NULL
      for (cand in outer_cv_leftout_candidates) {
        leftout_fold_assignment <- load_leftout_fold_assignment(cand)
        if (!is.null(leftout_fold_assignment)) {
          cat(sprintf("  Using outer-CV leftout fold assignment from: %s\n", cand))
          break
        }
      }
      if (is.null(leftout_fold_assignment)) {
        cat("  WARNING: no outer-CV leftout CSV found; left-out rows will be duplicated into every fold (legacy behaviour).\n")
      }

      prob_aug <- list(
        svm = list(cv = augment_model_with_leftout(
          probability_matrices$svm$cv, lo_svm, leftout_fold_assignment)),
        xgboost = list(cv = augment_model_with_leftout(
          probability_matrices$xgboost$cv, lo_xgb, leftout_fold_assignment)),
        neural_net = list(cv = augment_model_with_leftout(
          probability_matrices$neural_net$cv, lo_nn, leftout_fold_assignment))
      )

      # LOFO isolation check: when leftouts are partitioned, a given
      # sample_index must appear in at most one fold across each model. If
      # this fails, the OOD-aware calibrator would leak target-fold OOD into
      # its fit pool.
      assert_leftout_fold_disjoint <- function(prob_aug_by_model) {
        for (model_name in names(prob_aug_by_model)) {
          folds <- prob_aug_by_model[[model_name]]$cv
          if (!is.list(folds) || length(folds) == 0) next
          per_fold_idx <- lapply(folds, function(m) {
            if (is.null(m) || !"sample_indices" %in% colnames(m) ||
                !"is_leftout" %in% colnames(m)) return(integer(0))
            vals <- m$sample_indices[as.logical(m$is_leftout)]
            vals[!is.na(vals)]
          })
          all_idx <- unlist(per_fold_idx, use.names = FALSE)
          if (length(all_idx) == 0) next
          dup <- duplicated(all_idx)
          if (any(dup)) {
            stop(sprintf(
              "Leftout partition is not disjoint for model %s: sample_index %s appears in multiple folds. Regenerate the outer CV leftout CSVs or clear the fold_assignment.",
              model_name, paste(unique(all_idx[dup]), collapse = ", ")
            ))
          }
        }
      }
      if (!is.null(leftout_fold_assignment)) assert_leftout_fold_disjoint(prob_aug)

      # Build augmented global product ensemble with already selected global weights.
      # Attach model-disagreement features expected by multivariate/two-head
      # confidence models so calibration has the full predictor set.
      add_disagreement_features_to_ensemble <- function(ensemble_df, aligned_probs) {
        svm_p <- as.matrix(aligned_probs$svm)
        xgb_p <- as.matrix(aligned_probs$xgboost)
        nn_p <- as.matrix(aligned_probs$neural_net)
        n <- nrow(ensemble_df)
        if (n == 0) return(ensemble_df)

        # Fail fast if any model has rows with no finite probabilities. This is
        # data corruption for disagreement/two-head features, not something to
        # silently skip.
        find_all_nonfinite_rows <- function(m) {
          which(rowSums(is.finite(m)) == 0)
        }
        bad_svm <- find_all_nonfinite_rows(svm_p)
        bad_xgb <- find_all_nonfinite_rows(xgb_p)
        bad_nn <- find_all_nonfinite_rows(nn_p)
        if (length(bad_svm) > 0 || length(bad_xgb) > 0 || length(bad_nn) > 0) {
          idx_vals <- if ("indices" %in% colnames(ensemble_df)) ensemble_df$indices else seq_len(n)
          bad_idx <- unique(c(
            idx_vals[bad_svm],
            idx_vals[bad_xgb],
            idx_vals[bad_nn]
          ))
          stop(sprintf(
            paste0(
              "Disagreement feature computation failed: rows with no finite class probabilities ",
              "detected (SVM=%d, XGB=%d, NN=%d). Example sample indices: %s. ",
              "Inspect left-out probability construction/alignment for these samples."
            ),
            length(bad_svm), length(bad_xgb), length(bad_nn),
            paste(head(bad_idx, 15), collapse = ", ")
          ))
        }

        # Top-1 class per expert and agreement count across experts.
        svm_top_idx <- max.col(svm_p, ties.method = "first")
        xgb_top_idx <- max.col(xgb_p, ties.method = "first")
        nn_top_idx <- max.col(nn_p, ties.method = "first")
        svm_pred <- colnames(svm_p)[svm_top_idx]
        xgb_pred <- colnames(xgb_p)[xgb_top_idx]
        nn_pred <- colnames(nn_p)[nn_top_idx]

        preds_mat <- cbind(svm_pred, xgb_pred, nn_pred)
        n_models_agree <- apply(preds_mat, 1, function(x) {
          x <- x[!is.na(x)]
          if (length(x) == 0) return(NA_real_)
          max(table(x))
        })
        if (any(!is.finite(n_models_agree))) {
          idx_vals <- if ("indices" %in% colnames(ensemble_df)) ensemble_df$indices else seq_len(n)
          bad_rows <- which(!is.finite(n_models_agree))
          stop(sprintf(
            "Disagreement feature computation failed: could not compute n_models_agree for %d rows. Example sample indices: %s",
            length(bad_rows), paste(head(idx_vals[bad_rows], 15), collapse = ", ")
          ))
        }

        # Variance of each expert's own top-1 probability.
        top1_prob_mat <- cbind(
          svm_p[cbind(seq_len(n), svm_top_idx)],
          xgb_p[cbind(seq_len(n), xgb_top_idx)],
          nn_p[cbind(seq_len(n), nn_top_idx)]
        )
        top1_prob_variance <- apply(top1_prob_mat, 1, var, na.rm = TRUE)
        top1_prob_variance[!is.finite(top1_prob_variance)] <- 0

        ensemble_df$n_models_agree <- as.numeric(n_models_agree)
        ensemble_df$top1_prob_variance_across_models <- pmax(0, as.numeric(top1_prob_variance))
        ensemble_df
      }

      aug_global <- list()
      weights_used <- ensemble_results$cv$global_optimized_ensemble_matrices$weights_used
      for (outer_fold in names(prob_aug$svm$cv)) {
        aligned <- align_probability_matrices_cached(
          prob_aug, outer_fold, NULL, "cv", new.env(hash = TRUE)
        )
        if (is.null(aligned) || !outer_fold %in% names(weights_used)) next
        w <- weights_used[[outer_fold]]$weights
        eps <- 1e-12
        poe <- (pmax(as.matrix(aligned$svm), eps) ^ w$SVM) *
          (pmax(as.matrix(aligned$xgboost), eps) ^ w$XGB) *
          (pmax(as.matrix(aligned$neural_net), eps) ^ w$NN)
        rs <- rowSums(poe)
        rs[rs == 0] <- 1
        poe <- as.data.frame(poe / rs)
        ens_with_meta <- cbind(poe, aligned$non_prob_cols)
        ens_with_meta <- add_disagreement_features_to_ensemble(ens_with_meta, aligned)
        aug_global[[outer_fold]] <- ens_with_meta
      }

      # Deployment-only behavior: do not run train/test cutoff sweeps or write
      # train/test risk-coverage artifacts. Cutoff selection is delegated to
      # nested CV outputs from outer_cv_analysis.R.
      cat("  Skipping left-out-aware train/test cutoff sweeps; using nested-CV cutoffs at inference.\n")
    } else {
      stop(
        "Could not parse one or more left-out prediction files. ",
        "Two-head deployment requires these files to train the OOD head on ",
        "known + left-out data."
      )
    }
  } else {
    stop(
      "Left-out prediction files were not found for all models. ",
      "Two-head deployment requires these files to train the OOD head on ",
      "known + left-out data."
    )
  }

  # Fit and save calibration parameters for deployment.
  # Exported calibration settings:
  #   * known_only  : P(correct | features) on known-only rows
  #   * known_only_logit: known_only with logit(max_prob) replacing max_prob
  #   * ood_aware   : P(correct_AND_id | features) on known + left-out rows
  #   * ood_aware_logit: ood_aware with logit(max_prob) replacing max_prob
  #   * two_head    : correctness head + dedicated OOD head (product at inference)
  # For each setting we export univariate (max_prob) and multivariate variants.
  cat("Fitting calibration model(s) for deployment (known-only, ood-aware, two-head; uni + multi)...\n")

  fit_and_save_single_head_params <- function(
    fold_list,
    out_dir,
    out_filename,
    description,
    candidate_terms,
    target_mode = c("correct", "correct_and_id"),
    use_logit_max_prob = FALSE,
    eps = 1e-6
  ) {
    target_mode <- match.arg(target_mode)
    if (!is.list(fold_list) || length(fold_list) == 0L) return(invisible(NULL))
    pooled_features <- lapply(fold_list, function(m) {
      if (is.null(m) || nrow(m) == 0) return(NULL)
      feats <- get_rejection_features_from_matrix(m)
      if ("n_models_agree" %in% colnames(m)) feats$n_models_agree <- m$n_models_agree
      feats$is_id <- if ("is_leftout" %in% colnames(m)) {
        as.integer(!as.logical(m$is_leftout))
      } else {
        rep(1L, nrow(m))
      }
      if (target_mode == "correct_and_id") {
        feats$target <- as.integer(feats$correct) * as.integer(feats$is_id)
      } else {
        feats$target <- as.integer(feats$correct)
      }
      feats
    })
    pooled_features <- pooled_features[sapply(pooled_features, nrow) > 0]
    if (length(pooled_features) == 0) {
      cat(sprintf("  Warning: No pooled feature rows for %s calibration model\n", description))
      return(invisible(NULL))
    }
    pooled_df <- do.call(rbind, pooled_features)
    if (nrow(pooled_df) < 10L || length(unique(pooled_df$target)) < 2L) {
      cat(sprintf("  Warning: Not enough pooled data for %s calibration model\n", description))
      return(invisible(NULL))
    }
    if (use_logit_max_prob) {
      pooled_df$logit_max_prob <- qlogis(pmin(1 - eps, pmax(eps, pooled_df$max_prob)))
      candidate_terms <- ifelse(candidate_terms == "max_prob", "logit_max_prob", candidate_terms)
    }
    usable_terms <- candidate_terms[sapply(candidate_terms, function(v) {
      v %in% colnames(pooled_df) && any(is.finite(pooled_df[[v]]))
    })]
    if (length(usable_terms) == 0L) {
      cat(sprintf("  Warning: No usable predictors for %s calibration model\n", description))
      return(invisible(NULL))
    }
    formula <- stats::as.formula(paste("target ~", paste(usable_terms, collapse = " + ")))
    fit <- tryCatch(
      stats::glm(
        formula,
        data = pooled_df,
        family = stats::binomial(),
        control = stats::glm.control(maxit = 200, epsilon = 1e-8)
      ),
      error = function(e) NULL
    )
    if (is.null(fit)) {
      cat(sprintf("  Warning: Could not fit %s calibration model for deployment\n", description))
      return(invisible(NULL))
    }
    coef_vec <- stats::coef(fit)
    params_df <- data.frame(
      model = "Global_Optimized",
      term = names(coef_vec),
      estimate = as.numeric(coef_vec),
      stringsAsFactors = FALSE
    )
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    out_path <- file.path(out_dir, out_filename)
    write.csv(params_df, out_path, row.names = FALSE)
    cat(sprintf("  Saved %s calibration parameters to: %s\n", description, out_path))
    invisible(params_df)
  }

  if ("cv" %in% names(ensemble_results) &&
      "global_optimized_ensemble_matrices" %in% names(ensemble_results[["cv"]]) &&
      "matrices" %in% names(ensemble_results[["cv"]]$global_optimized_ensemble_matrices)) {
    multivariate_dir <- paste0("../data/out/final_train_test/multivariate_params", merge_suffix)
    univariate_dir <- paste0("../data/out/final_train_test/univariate_params", merge_suffix)

    # Known-only single-head (existing behavior).
    fit_and_save_single_head_params(
      ensemble_results[["cv"]]$global_optimized_ensemble_matrices$matrices,
      multivariate_dir,
      paste0("multivariate_params", merge_suffix, ".csv"),
      "multivariate known-only",
      candidate_terms = c("max_prob", "margin", "entropy", "n_models_agree", "top1_prob_variance_across_models"),
      target_mode = "correct"
    )

    fit_and_save_single_head_params(
      ensemble_results[["cv"]]$global_optimized_ensemble_matrices$matrices,
      univariate_dir,
      paste0("univariate_params", merge_suffix, ".csv"),
      "univariate known-only",
      candidate_terms = c("max_prob"),
      target_mode = "correct"
    )

    # Explicit known-only filenames for unified predictor loading.
    fit_and_save_single_head_params(
      ensemble_results[["cv"]]$global_optimized_ensemble_matrices$matrices,
      multivariate_dir,
      paste0("multivariate_params_known_only", merge_suffix, ".csv"),
      "multivariate known-only (explicit)",
      candidate_terms = c("max_prob", "margin", "entropy", "n_models_agree", "top1_prob_variance_across_models"),
      target_mode = "correct"
    )
    fit_and_save_single_head_params(
      ensemble_results[["cv"]]$global_optimized_ensemble_matrices$matrices,
      univariate_dir,
      paste0("univariate_params_known_only", merge_suffix, ".csv"),
      "univariate known-only (explicit)",
      candidate_terms = c("max_prob"),
      target_mode = "correct"
    )

    # Logit(max_prob) variants for known-only single-head calibration.
    fit_and_save_single_head_params(
      ensemble_results[["cv"]]$global_optimized_ensemble_matrices$matrices,
      multivariate_dir,
      paste0("multivariate_params_known_only_logit", merge_suffix, ".csv"),
      "multivariate known-only logit(max_prob)",
      candidate_terms = c("max_prob", "margin", "entropy", "n_models_agree", "top1_prob_variance_across_models"),
      target_mode = "correct",
      use_logit_max_prob = TRUE
    )
    fit_and_save_single_head_params(
      ensemble_results[["cv"]]$global_optimized_ensemble_matrices$matrices,
      univariate_dir,
      paste0("univariate_params_known_only_logit", merge_suffix, ".csv"),
      "univariate known-only logit(max_prob)",
      candidate_terms = c("max_prob"),
      target_mode = "correct",
      use_logit_max_prob = TRUE
    )
  }

  # Two-head deployment: the correctness head is the existing known-only model
  # (multivariate_params). We additionally fit a dedicated OOD head on
  # is_id ~ features from the full augmented pool (known + leftout) with
  # inverse-class-frequency weights. At inference the composite accept score
  # is P_ID(x) * P_correct(x).
  fit_and_save_ood_head_params <- function(fold_list, out_dir, out_filename, description, candidate_terms) {
    if (!is.list(fold_list) || length(fold_list) == 0L) return(invisible(NULL))
    pool_features <- list()
    for (m in fold_list) {
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
      cat("  Warning: No pooled features available for OOD head deployment fit\n")
      return(invisible(NULL))
    }
    pool_df <- do.call(rbind, pool_features)
    if (nrow(pool_df) < 10L || length(unique(pool_df$is_id)) < 2L) {
      cat("  Warning: Not enough data / no OOD examples for OOD head deployment fit\n")
      return(invisible(NULL))
    }

    usable_terms <- candidate_terms[sapply(candidate_terms, function(v) {
      v %in% colnames(pool_df) && any(is.finite(pool_df[[v]]))
    })]
    if (length(usable_terms) == 0L) {
      cat(sprintf("  Warning: No usable predictors for %s OOD head deployment fit\n", description))
      return(invisible(NULL))
    }
    formula <- stats::as.formula(paste("is_id ~", paste(usable_terms, collapse = " + ")))
    freq <- table(pool_df$is_id)
    per_row <- 1 / as.numeric(freq[as.character(pool_df$is_id)])
    weights_vec <- per_row * nrow(pool_df) / sum(per_row)
    # Fractional class-balancing weights require quasibinomial to avoid the
    # non-integer-successes warning while keeping the same mean model.
    ood_family <- stats::quasibinomial()
    fit <- tryCatch(
      stats::glm(
        formula,
        data = pool_df,
        family = ood_family,
        weights = weights_vec,
        control = stats::glm.control(maxit = 200, epsilon = 1e-8)
      ),
      error = function(e) NULL
    )
    if (is.null(fit)) {
      cat("  Warning: Could not fit OOD head for deployment\n")
      return(invisible(NULL))
    }
    coef_vec <- stats::coef(fit)
    params_df <- data.frame(
      model = "Global_Optimized",
      term = names(coef_vec),
      estimate = as.numeric(coef_vec),
      stringsAsFactors = FALSE
    )
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    out_path <- file.path(out_dir, out_filename)
    write.csv(params_df, out_path, row.names = FALSE)
    cat(sprintf("  Saved %s OOD head (is_id ~ features, class-balanced) parameters to: %s\n", description, out_path))
    invisible(params_df)
  }

  fit_and_save_two_head_postcal_params <- function(
    fold_list,
    out_dir,
    out_filename,
    description,
    use_multivariate = TRUE
  ) {
    if (!is.list(fold_list) || length(fold_list) < 2L) return(invisible(NULL))
    fold_names <- names(fold_list)
    rows <- list()
    idx <- 1L
    eps <- 1e-6

    for (k in seq_along(fold_names)) {
      target <- fold_list[[fold_names[k]]]
      others <- fold_list[setdiff(fold_names, fold_names[k])]
      if (is.null(target) || nrow(target) == 0 || length(others) == 0) next

      confidence_col <- if (use_multivariate) "confidence_multivariate" else "confidence_two_head"
      scored <- tryCatch(
        apply_two_head_calibration_to_target_from_pool(
          others,
          target,
          use_multivariate = use_multivariate,
          confidence_col = confidence_col
        ),
        error = function(e) NULL
      )
      if (is.null(scored) || !confidence_col %in% colnames(scored)) next

      base <- get_max_prob_and_correct_from_matrix(scored)
      s <- as.numeric(scored[[confidence_col]])
      s <- pmin(1 - eps, pmax(eps, s))
      rows[[idx]] <- data.frame(
        correct = as.integer(base$correct),
        logit_score = qlogis(s),
        stringsAsFactors = FALSE
      )
      idx <- idx + 1L
    }

    if (length(rows) == 0) {
      cat(sprintf("  Warning: No cross-fitted rows for %s two_head_postcal fit\n", description))
      return(invisible(NULL))
    }
    df <- do.call(rbind, rows)
    if (nrow(df) < 10L || length(unique(df$correct)) < 2L) {
      cat(sprintf("  Warning: Not enough class variation for %s two_head_postcal fit\n", description))
      return(invisible(NULL))
    }

    fit <- tryCatch(
      stats::glm(
        correct ~ logit_score,
        data = df,
        family = stats::binomial(),
        control = stats::glm.control(maxit = 200, epsilon = 1e-8)
      ),
      error = function(e) NULL
    )
    if (is.null(fit)) {
      cat(sprintf("  Warning: Could not fit %s two_head_postcal model\n", description))
      return(invisible(NULL))
    }
    coef_vec <- stats::coef(fit)
    params_df <- data.frame(
      model = "Global_Optimized",
      term = names(coef_vec),
      estimate = as.numeric(coef_vec),
      stringsAsFactors = FALSE
    )
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    out_path <- file.path(out_dir, out_filename)
    write.csv(params_df, out_path, row.names = FALSE)
    cat(sprintf("  Saved %s two_head_postcal calibration parameters to: %s\n", description, out_path))
    invisible(params_df)
  }

  if (!exists("aug_global") || !is.list(aug_global) || length(aug_global) == 0L) {
    stop(
      "Missing augmented fold matrices for OOD head training. ",
      "Two-head deployment requires pooled known + left-out data."
    )
  }
  multivariate_dir <- paste0("../data/out/final_train_test/multivariate_params", merge_suffix)
  univariate_dir <- paste0("../data/out/final_train_test/univariate_params", merge_suffix)

  # OOD-aware single-head calibrators on full augmented pool.
  fit_and_save_single_head_params(
    aug_global,
    multivariate_dir,
    paste0("multivariate_params_ood_aware", merge_suffix, ".csv"),
    "multivariate ood-aware",
    candidate_terms = c("max_prob", "margin", "entropy", "n_models_agree", "top1_prob_variance_across_models"),
    target_mode = "correct_and_id"
  )
  fit_and_save_single_head_params(
    aug_global,
    univariate_dir,
    paste0("univariate_params_ood_aware", merge_suffix, ".csv"),
    "univariate ood-aware",
    candidate_terms = c("max_prob"),
    target_mode = "correct_and_id"
  )

  fit_and_save_single_head_params(
    aug_global,
    multivariate_dir,
    paste0("multivariate_params_ood_aware_logit", merge_suffix, ".csv"),
    "multivariate ood-aware logit(max_prob)",
    candidate_terms = c("max_prob", "margin", "entropy", "n_models_agree", "top1_prob_variance_across_models"),
    target_mode = "correct_and_id",
    use_logit_max_prob = TRUE
  )
  fit_and_save_single_head_params(
    aug_global,
    univariate_dir,
    paste0("univariate_params_ood_aware_logit", merge_suffix, ".csv"),
    "univariate ood-aware logit(max_prob)",
    candidate_terms = c("max_prob"),
    target_mode = "correct_and_id",
    use_logit_max_prob = TRUE
  )

  fit_and_save_ood_head_params(
    aug_global,
    multivariate_dir,
    paste0("ood_head_params", merge_suffix, ".csv"),
    description = "multivariate",
    candidate_terms = c("max_prob", "margin", "entropy", "n_models_agree", "top1_prob_variance_across_models")
  )
  fit_and_save_ood_head_params(
    aug_global,
    univariate_dir,
    paste0("ood_head_params_univariate", merge_suffix, ".csv"),
    description = "univariate",
    candidate_terms = c("max_prob")
  )

  # Post-calibration on top of two-head product score.
  fit_and_save_two_head_postcal_params(
    aug_global,
    multivariate_dir,
    paste0("two_head_postcal_params", merge_suffix, ".csv"),
    description = "multivariate",
    use_multivariate = TRUE
  )
  fit_and_save_two_head_postcal_params(
    aug_global,
    univariate_dir,
    paste0("two_head_postcal_params", merge_suffix, ".csv"),
    description = "univariate",
    use_multivariate = FALSE
  )

  # Calculate performance comparisons (CV only)
  cat("Calculating performance comparisons (CV)...\n")
  performance_comparisons <- list()

  for (analysis_type in "cv") {
    if (analysis_type %in% names(ensemble_results)) {

      # Individual model performance
      individual_performance <- list()
      for (model_name in c("svm", "xgboost", "neural_net")) {
        if (analysis_type %in% names(probability_matrices[[model_name]])) {
          all_kappas <- c()

          for (outer_fold in names(probability_matrices[[model_name]][[analysis_type]])) {
            optimized_matrix <- probability_matrices[[model_name]][[analysis_type]][[outer_fold]]

            # Extract true labels and remove non-probability / meta columns
            truth <- make.names(optimized_matrix$y)
            prob_matrix <- optimized_matrix[, !colnames(optimized_matrix) %in% c("y", "inner_fold", "outer_fold", "indices", "study", "confidence_multivariate"), drop = FALSE]

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

      # Ensemble performance (Global only; OvR removed)
      method_performance <- ensemble_results[[analysis_type]]$global_optimized_ensemble_performance
      all_kappas <- c()
      for (outer_fold in names(method_performance)) {
        cm <- method_performance[[outer_fold]]
        if (inherits(cm, "confusionMatrix")) {
          all_kappas <- c(all_kappas, cm$overall["Kappa"])
        }
      }
      individual_performance[["Global_Optimized"]] <- list(
        mean_kappa = mean(all_kappas, na.rm = TRUE),
        sd_kappa = sd(all_kappas, na.rm = TRUE),
        fold_kappas = all_kappas
      )

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

  cat("\n=== Final Train/Test Results Summary ===\n")
  print(train_test_results$final_results)

  saveRDS(train_test_results, paste0("../data/out/final_train_test/final_train_test_results_10feb2026", merge_suffix, ".rds"))
  return(train_test_results)
}

# Run unmerged and fully merged versions (MDS.r, other.KMT2A, MECOM; maxprob method)
cat("=== Running Train/Test Analysis (Unmerged - MaxProb Method) ===\n")
train_test_results_unmerged <- main_train_test_analysis(merge_classes = FALSE)

cat("=== Running Train/Test Analysis (Merged MDS/KMT2A/MECOM - MaxProb Method) ===\n")
train_test_results_merged <- main_train_test_analysis(merge_classes = TRUE)
