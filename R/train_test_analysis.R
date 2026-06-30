# Source shared utility functions
source("utility_functions.R")

# KNN + derived reject features (mirrors outer_cv_analysis.r).
add_optional_knn_columns <- function(probability_matrix, source_row, num_samples) {
  for (knn_col in KNN_DISTANCE_COLUMNS) {
    if (!knn_col %in% colnames(source_row)) next
    parsed <- parse_numeric_string(source_row[[knn_col]][1])
    if (length(parsed) == num_samples) {
      probability_matrix[[knn_col]] <- parsed
    }
  }
  probability_matrix
}

# Attach cohort KNN from bash/run_all_final.sh export (final_selection CSVs omit KNN).
cohort_knn_features_path <- function(fs_method = "eta2") {
  file.path("../data/out/final_train_test", paste0("cohort_knn_features_", fs_method, ".csv"))
}

attach_cohort_knn_to_probability_matrices <- function(probability_matrices, fs_method = "eta2") {
  knn_path <- cohort_knn_features_path(fs_method)
  if (!file.exists(knn_path)) {
    stop(sprintf(
      paste0(
        "Missing cohort KNN file: %s. ",
        "Run bash/run_all_final.sh first (exports cohort KNN before final model training)."
      ),
      knn_path
    ))
  }
  knn_df <- read.csv(knn_path, stringsAsFactors = FALSE)
  required_cols <- c("indices", KNN_DISTANCE_COLUMNS)
  if (!all(required_cols %in% colnames(knn_df))) {
    stop(sprintf("Invalid cohort KNN export (missing columns): %s", knn_path))
  }
  knn_lookup <- knn_df[, required_cols, drop = FALSE]
  rownames(knn_lookup) <- as.character(knn_lookup$indices)

  attach_to_matrix <- function(m) {
    if (is.null(m) || !"indices" %in% colnames(m)) return(m)
    idx_chr <- as.character(m$indices)
    missing_idx <- idx_chr[!idx_chr %in% rownames(knn_lookup)]
    if (length(missing_idx) > 0) {
      stop(sprintf(
        "Cohort KNN lookup missing %d sample indices. Example: %s",
        length(missing_idx), paste(head(unique(missing_idx), 10), collapse = ", ")
      ))
    }
    for (kcol in KNN_DISTANCE_COLUMNS) {
      vals <- as.numeric(knn_lookup[idx_chr, kcol])
      if (kcol %in% colnames(m) && any(is.finite(m[[kcol]]))) {
        replace <- !is.finite(m[[kcol]])
        m[[kcol]][replace] <- vals[replace]
      } else {
        m[[kcol]] <- vals
      }
    }
    m
  }

  for (model_name in names(probability_matrices)) {
    for (fold_type in names(probability_matrices[[model_name]])) {
      folds <- probability_matrices[[model_name]][[fold_type]]
      if (is.list(folds)) {
        probability_matrices[[model_name]][[fold_type]] <- lapply(folds, attach_to_matrix)
      }
    }
  }
  probability_matrices
}

assert_finite_knn_columns <- function(df, context) {
  for (kcol in KNN_DISTANCE_COLUMNS) {
    if (!kcol %in% colnames(df)) {
      stop(sprintf("%s: missing KNN column '%s'.", context, kcol))
    }
    kv <- suppressWarnings(as.numeric(df[[kcol]]))
    if (!all(is.finite(kv))) {
      stop(sprintf(
        "%s: KNN column '%s' has %d non-finite of %d rows.",
        context, kcol, sum(!is.finite(kv)), length(kv)
      ))
    }
  }
  invisible(TRUE)
}

add_roi_reject_features <- function(df, alpha = 0.10, eps = 1e-8) {
  if (is.null(df) || nrow(df) == 0) return(df)
  meta_cols <- PROB_MATRIX_META_COLUMNS
  prob_cols <- colnames(df)[!colnames(df) %in% meta_cols]
  if (length(prob_cols) == 0) {
    df$trust_ratio_knn10 <- NA_real_
    df$conformal_set_size_90 <- NA_real_
    return(df)
  }
  prob_df <- df[, prob_cols, drop = FALSE]
  prob_df[] <- lapply(prob_df, function(x) {
    v <- suppressWarnings(as.numeric(x))
    v[!is.finite(v)] <- 0
    v
  })
  prob_mat <- as.matrix(prob_df)
  mode(prob_mat) <- "numeric"
  prob_mat <- pmax(prob_mat, 0)
  rs <- rowSums(prob_mat)
  rs[!is.finite(rs) | rs <= 0] <- 1
  prob_mat <- prob_mat / rs

  if (all(c("knn10_min_d", "knn10_q90_d") %in% colnames(df))) {
    knn_min <- as.numeric(df$knn10_min_d)
    knn_q90 <- as.numeric(df$knn10_q90_d)
    df$trust_ratio_knn10 <- knn_min / pmax(knn_q90, eps)
  } else {
    df$trust_ratio_knn10 <- NA_real_
  }

  p_sorted <- t(apply(prob_mat, 1, sort, decreasing = TRUE))
  cum_sorted <- t(apply(p_sorted, 1, cumsum))
  threshold <- 1 - alpha
  df$conformal_set_size_90 <- apply(cum_sorted, 1, function(cs) {
    hit <- which(cs >= threshold)
    if (length(hit) == 0) ncol(prob_mat) else hit[1]
  })
  df
}

main_train_test_analysis <- function(merge_classes = FALSE, merge_prob_method = c("sum", "max")){
  merge_prob_method <- match.arg(merge_prob_method)

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
  generate_ovr_probability_matrices <- function(cv_results_df, best_params_df, label_mapping, study_names, merge_classes = FALSE, merge_prob_method = "sum") {
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
        # KNN vectors are identical across OvR class rows within a fold.
        probability_matrix <- add_optional_knn_columns(
          probability_matrix, inner_fold_data[1, , drop = FALSE], num_samples
        )

        # Apply class merging if requested (summed method, consistent with inner CV)
        if (merge_classes) {
          probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_prob_method = merge_prob_method)
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

  generate_standard_probability_matrices <- function(cv_results_df, best_params_df, label_mapping, filtered_subtypes, study_names, merge_classes = FALSE, merge_prob_method = "sum") {
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
        probability_matrix <- add_optional_knn_columns(
          probability_matrix, inner_fold_data[1, , drop = FALSE], num_samples
        )

        # Apply class merging if requested (summed method, consistent with inner CV)
        if (merge_classes) {
          probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_prob_method = merge_prob_method)
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
      # OvR left-out CSVs use integer class + class_label (see run_final_train.py restore_labels).
      if (!"class_label" %in% colnames(lo)) {
        if (!"class" %in% colnames(lo)) {
          stop("OvR left-out CSV missing class_label and class columns.")
        }
        lo$class_label <- label_mapping$Label[as.integer(lo$class) + 1L]
      }
      classes <- unique(lo$class_label)
      prob_mat <- matrix(NA_real_, nrow = length(sample_indices), ncol = length(classes))
      colnames(prob_mat) <- classes
      for (j in seq_along(classes)) {
        row_j <- lo[lo$class_label == classes[j], , drop = FALSE]
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
    prob_df <- add_optional_knn_columns(prob_df, lo[1, , drop = FALSE], length(sample_indices))

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
      prob_cols_union <- setdiff(common_cols, PROB_MATRIX_META_COLUMNS)
      for (cc in setdiff(common_cols, colnames(known))) {
        known[[cc]] <- if (cc %in% prob_cols_union) 0 else NA
      }
      for (cc in setdiff(common_cols, colnames(lo))) {
        lo[[cc]] <- if (cc %in% prob_cols_union) 0 else NA
      }
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

  #' Generate globally optimized ensemble probability matrices for train/test
  #' @param results Analysis results containing probability matrices
  #' @param weights Weight configurations for ensemble
  #' @param type Type of analysis ("loso" for final deployment)
  #' @param ensemble_performance Performance results from perform_global_ensemble_analysis_unified
  #' @param ensemble_rule "poe" (product-of-experts) or "simple" (linear weighted average)
  #' @return List containing optimized probability matrices and weights used for each outer fold
  generate_global_optimized_ensemble_matrices_train_test <- function(
      results, weights, type = "loso", ensemble_performance, ensemble_rule = c("poe", "simple")) {
    ensemble_rule <- match.arg(ensemble_rule)
    combine_probs <- global_ensemble_combine_fn(ensemble_rule)
    rule_label <- if (ensemble_rule == "simple") "simple weighted average" else "product-of-experts"
    cat(sprintf("Generating globally optimized ensemble probability matrices for train/test (%s)...\n", rule_label))

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

      optimized_matrix <- combine_probs(
        prob_mat_SVM, prob_mat_XGB, prob_mat_NN, global_best_weights
      )

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
  #' @param type Type of analysis ("loso" for final deployment)
  #' @return Performance results for each outer fold
  analyze_optimized_ensemble_performance_train_test <- function(ensemble_result, type = "loso") {
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

  # Must match bash/run_all_final.sh and predict_new_samples.py KNN reject features.
  FINAL_FS_METHOD <- "eta2"

  # LOSO only (hyperparameters for final deployment model are selected under LOSO)
  # Paths must match bash/final_selection_array_*_loso.sh --run_name and
  # python/run_final_selection_array.py output directory.
  MODEL_CONFIGS <- list(
    svm = list(
      classification_type = "OvR",
      file_paths = list(loso = "../data/out/final_train_test/SVM_final_selection/final_loso_svm/"),
      output_dir = "../data/out/final_train_test/best_params/SVM"
    ),
    xgboost = list(
      classification_type = "OvR",
      file_paths = list(loso = "../data/out/final_train_test/XGBOOST_final_selection/final_loso_xgb/"),
      output_dir = "../data/out/final_train_test/best_params/XGBOOST"
    ),
    neural_net = list(
      classification_type = "standard",
      file_paths = list(loso = "../data/out/final_train_test/NN_final_selection/final_loso_nn/"),
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
          probs <- generate_ovr_probability_matrices(results, best_params, label_mapping, study_names, merge_classes = merge_classes, merge_prob_method = merge_prob_method)
        } else {
          probs <- generate_standard_probability_matrices(results, best_params, label_mapping, filtered_leukemia_subtypes, study_names, merge_classes = merge_classes, merge_prob_method = merge_prob_method)
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

  # Run ensemble analysis for LOSO (final model uses LOSO-selected hyperparameters)
  cat("Running ensemble analysis for train/test (LOSO)...\n")
  ensemble_results <- list()

  for (analysis_type in "loso") {
    cat(sprintf("\n=== Running %s analysis ===\n", toupper(analysis_type)))

    # Check if we have data for this analysis type
    if (!all(sapply(probability_matrices, function(x) analysis_type %in% names(x)))) {
      cat(sprintf("Skipping %s analysis - missing data\n", toupper(analysis_type)))
      next
    }

    weight_grid <- generate_weights()
    results_for_ensemble <- list(probability_matrices = probability_matrices)

    # Product-of-experts (deployment default for calibration matrices)
    global_ensemble_results <- perform_global_ensemble_analysis_unified(
      results_for_ensemble, weight_grid, analysis_type,
      has_inner_folds = FALSE, ensemble_rule = "poe"
    )
    global_optimized_ensemble_matrices <- generate_global_optimized_ensemble_matrices_train_test(
      results_for_ensemble, weight_grid, analysis_type,
      global_ensemble_results, ensemble_rule = "poe"
    )
    global_optimized_ensemble_performance <- analyze_optimized_ensemble_performance_train_test(
      global_optimized_ensemble_matrices, analysis_type
    )

    # Simple weighted average (separate inner-CV-selected weights for MLL / comparisons)
    global_simple_ensemble_results <- perform_global_ensemble_analysis_unified(
      results_for_ensemble, weight_grid, analysis_type,
      has_inner_folds = FALSE, ensemble_rule = "simple"
    )
    global_simple_optimized_ensemble_matrices <- generate_global_optimized_ensemble_matrices_train_test(
      results_for_ensemble, weight_grid, analysis_type,
      global_simple_ensemble_results, ensemble_rule = "simple"
    )
    global_simple_optimized_ensemble_performance <- analyze_optimized_ensemble_performance_train_test(
      global_simple_optimized_ensemble_matrices, analysis_type
    )

    # Store results for this analysis type (Global ensemble only; OvR removed)
    ensemble_results[[analysis_type]] <- list(
      global_ensemble_results = global_ensemble_results,
      global_optimized_ensemble_matrices = global_optimized_ensemble_matrices,
      global_optimized_ensemble_performance = global_optimized_ensemble_performance,
      global_ensemble_weights_used = global_optimized_ensemble_matrices$weights_used,
      global_simple_ensemble_results = global_simple_ensemble_results,
      global_simple_optimized_ensemble_matrices = global_simple_optimized_ensemble_matrices,
      global_simple_optimized_ensemble_performance = global_simple_optimized_ensemble_performance,
      global_simple_ensemble_weights_used = global_simple_optimized_ensemble_matrices$weights_used
    )
  }

  # NOTE: Rejection / calibration feature selection moved to calibration_reject_models_final.R
  # (mirrors outer_cv_analysis.r + calibration_reject_models.R).

  # Determine suffix for file paths (collapsed vocabulary differs by sum vs max combine rule)
  if (!merge_classes) {
    merge_suffix <- "_unmerged_maxprob"
  } else if (merge_prob_method == "max") {
    merge_suffix <- "_merged_maxprob"
  } else {
    merge_suffix <- "_merged_summed"
  }

  # Save ensemble weights
  weights_dir <- paste0("../data/out/final_train_test/ensemble_weights", merge_suffix)
  dir.create(weights_dir, recursive = TRUE)
  save_ensemble_weights(ensemble_results, weights_dir, save_per_fold = FALSE)

  multivariate_results <- list(with_leftout_ood_aware = list())
  CALIBRATION_MV_BASE_MODEL <- "Global_Optimized"
  CALIBRATION_MV_MODEL_LABEL <- "Global_Optimized_augmented_disagreement_folds"
  has_leftout_augmentation <- FALSE

  # -------------------------------------------------------------------------
  # Left-out-aware augmented ensemble folds for calibration_reject_models_final.R
  # -------------------------------------------------------------------------
  cat("Preparing left-out-aware augmented matrices for final calibration analysis...\n")
  leftout_file_configs <- list(
    svm = find_latest_csv("../data/out/final_models/SVM", "^SVM_final_loso_OvR_leftout.*\\.csv$"),
    xgboost = find_latest_csv("../data/out/final_models/XGBOOST", "^XGBOOST_final_loso_OvR_leftout.*\\.csv$"),
    neural_net = find_latest_csv("../data/out/final_models/NN", "^NN_final_loso_standard_leftout.*\\.csv$")
  )

  has_all_leftout <- all(sapply(leftout_file_configs, function(p) !is.null(p) && file.exists(p)))
  if (has_all_leftout && "loso" %in% names(ensemble_results)) {
    cat("  Found final-model left-out prediction files. Building augmented matrices...\n")

    # Known cohort rows need KNN from run_all_final.sh; left-out rows carry KNN from leftout CSVs.
    cat(sprintf("  Attaching cohort KNN features (fs_method=%s)...\n", FINAL_FS_METHOD))
    probability_matrices <- attach_cohort_knn_to_probability_matrices(
      probability_matrices, fs_method = FINAL_FS_METHOD
    )

    lo_svm <- build_leftout_probability_matrix(leftout_file_configs$svm, "svm", label_mapping, leukemia_subtypes, merge_classes = merge_classes)
    lo_xgb <- build_leftout_probability_matrix(leftout_file_configs$xgboost, "xgboost", label_mapping, leukemia_subtypes, merge_classes = merge_classes)
    lo_nn <- build_leftout_probability_matrix(leftout_file_configs$neural_net, "neural_net", label_mapping, leukemia_subtypes, merge_classes = merge_classes)

    if (!is.null(lo_svm) && !is.null(lo_xgb) && !is.null(lo_nn)) {
      # Per-fold left-out partition for LOFO calibration (written by run_final_train.py).
      final_leftout_assignment_path <- "../data/out/final_train_test/leftout_fold_assignment_loso.csv"
      leftout_fold_assignment <- load_leftout_fold_assignment(final_leftout_assignment_path)
      if (!is.null(leftout_fold_assignment)) {
        cat(sprintf("  Using final-pipeline leftout fold assignment from: %s\n", final_leftout_assignment_path))
      } else {
        stop(sprintf(
          "Missing %s. Run run_final_train.py --include_leftout before the second train_test_analysis.R pass.",
          final_leftout_assignment_path
        ))
      }

      prob_aug <- list(
        svm = list(loso = augment_model_with_leftout(
          probability_matrices$svm$loso, lo_svm, leftout_fold_assignment)),
        xgboost = list(loso = augment_model_with_leftout(
          probability_matrices$xgboost$loso, lo_xgb, leftout_fold_assignment)),
        neural_net = list(loso = augment_model_with_leftout(
          probability_matrices$neural_net$loso, lo_nn, leftout_fold_assignment))
      )

      # LOFO isolation check: when leftouts are partitioned, a given
      # sample_index must appear in at most one fold across each model. If
      # this fails, the OOD-aware calibrator would leak target-fold OOD into
      # its fit pool.
      assert_leftout_fold_disjoint <- function(prob_aug_by_model) {
        for (model_name in names(prob_aug_by_model)) {
          folds <- prob_aug_by_model[[model_name]]$loso
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
        n <- nrow(ensemble_df)
        if (n == 0) return(ensemble_df)

        # Build numeric class-probability matrices only. Non-probability columns
        # (e.g. confidence_* filled with NA on left-out rows) must not enter
        # max.col(); otherwise left-out rows fail with NA top-1 indices.
        svm_prob_cols <- colnames(aligned_probs$svm)[!colnames(aligned_probs$svm) %in% PROB_MATRIX_META_COLUMNS]
        xgb_prob_cols <- colnames(aligned_probs$xgboost)[!colnames(aligned_probs$xgboost) %in% PROB_MATRIX_META_COLUMNS]
        nn_prob_cols <- colnames(aligned_probs$neural_net)[!colnames(aligned_probs$neural_net) %in% PROB_MATRIX_META_COLUMNS]
        all_prob_cols <- unique(c(svm_prob_cols, xgb_prob_cols, nn_prob_cols))
        if (length(all_prob_cols) == 0) {
          stop("Disagreement feature computation failed: no class-probability columns found after alignment.")
        }

        make_aligned <- function(mat, prob_cols) {
          m <- matrix(0, nrow = nrow(mat), ncol = length(all_prob_cols))
          colnames(m) <- all_prob_cols
          for (col in intersect(prob_cols, all_prob_cols)) {
            vals <- suppressWarnings(as.numeric(mat[[col]]))
            vals[!is.finite(vals)] <- 0
            m[, col] <- vals
          }
          rs <- rowSums(m)
          rs[rs == 0] <- 1
          m / rs
        }

        svm_p <- make_aligned(aligned_probs$svm, svm_prob_cols)
        xgb_p <- make_aligned(aligned_probs$xgboost, xgb_prob_cols)
        nn_p <- make_aligned(aligned_probs$neural_net, nn_prob_cols)

        find_all_nonfinite_rows <- function(m) which(rowSums(is.finite(m)) == 0)
        bad_svm <- find_all_nonfinite_rows(svm_p)
        bad_xgb <- find_all_nonfinite_rows(xgb_p)
        bad_nn <- find_all_nonfinite_rows(nn_p)
        if (length(bad_svm) > 0 || length(bad_xgb) > 0 || length(bad_nn) > 0) {
          idx_vals <- if ("indices" %in% colnames(ensemble_df)) ensemble_df$indices else seq_len(n)
          bad_idx <- unique(c(idx_vals[bad_svm], idx_vals[bad_xgb], idx_vals[bad_nn]))
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

        svm_top_idx <- max.col(svm_p, ties.method = "first")
        xgb_top_idx <- max.col(xgb_p, ties.method = "first")
        nn_top_idx <- max.col(nn_p, ties.method = "first")
        svm_pred <- all_prob_cols[svm_top_idx]
        xgb_pred <- all_prob_cols[xgb_top_idx]
        nn_pred <- all_prob_cols[nn_top_idx]

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
      weights_used <- ensemble_results$loso$global_optimized_ensemble_matrices$weights_used
      for (outer_fold in names(prob_aug$svm$loso)) {
        aligned <- align_probability_matrices_cached(
          prob_aug, outer_fold, NULL, "loso", new.env(hash = TRUE)
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
        ens_with_meta <- add_roi_reject_features(ens_with_meta)
        assert_finite_knn_columns(
          ens_with_meta,
          sprintf("augmented ensemble fold %s (Global_Optimized)", outer_fold)
        )
        aug_global[[outer_fold]] <- ens_with_meta
      }

      if (length(aug_global) >= 2L) {
        multivariate_results$with_leftout_ood_aware[[CALIBRATION_MV_BASE_MODEL]] <- list(
          loso = list(
            fold_matrices = copy_fold_matrix_list(aug_global),
            model_label = CALIBRATION_MV_MODEL_LABEL
          )
        )
        has_leftout_augmentation <- TRUE
        cat(sprintf(
          "  Saved %d augmented ensemble folds for %s (R/calibration_reject_models_final.R).\n",
          length(aug_global), CALIBRATION_MV_BASE_MODEL
        ))
      } else {
        warning("Augmented ensemble has fewer than 2 folds; skipping multivariate_results bundle.")
      }

      # SVM augmented folds for deployment calibration (in-model / KNN10 rejector features).
      aug_svm <- list()
      for (outer_fold in names(prob_aug$svm$loso)) {
        svm_df <- prob_aug$svm$loso[[outer_fold]]
        svm_df <- add_roi_reject_features(svm_df)
        assert_finite_knn_columns(
          svm_df,
          sprintf("augmented SVM fold %s", outer_fold)
        )
        aug_svm[[outer_fold]] <- svm_df
      }
      if (length(aug_svm) >= 2L) {
        multivariate_results$with_leftout_ood_aware[["svm"]] <- list(
          loso = list(
            fold_matrices = copy_fold_matrix_list(aug_svm),
            model_label = "svm_augmented_disagreement_folds"
          )
        )
        cat(sprintf(
          "  Saved %d augmented SVM folds for svm (R/calibration_reject_models_final.R).\n",
          length(aug_svm)
        ))
      } else {
        warning("Augmented SVM has fewer than 2 folds; skipping svm multivariate_results bundle.")
      }
    } else {
      warning("Could not parse one or more left-out prediction files; skipping augmented calibration folds.")
    }
  } else {
    warning(
      "Left-out prediction files were not found for all models. ",
      "Run run_final_train.py --include_leftout, then re-run this script before calibration_reject_models_final.R."
    )
  }

  if (!has_leftout_augmentation) {
    cat("  No with_leftout_ood_aware fold bundles written (left-out augmentation incomplete).\n")
  }
  # Calculate performance comparisons (LOSO only)
  cat("Calculating performance comparisons (LOSO)...\n")
  performance_comparisons <- list()

  for (analysis_type in "loso") {
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

  # Comprehensive results for final model building + calibration_reject_models_final.R
  train_test_results <- list(
    model_results = model_results,
    best_parameters = best_parameters,
    probability_matrices = probability_matrices,
    per_model_results = per_model_results,
    per_class_results = per_class_results,
    ensemble_results = ensemble_results,
    performance_comparisons = performance_comparisons,
    multivariate_results = multivariate_results
  )
  train_test_results$merge_classes <- merge_classes

  cat("\n=== Final Train/Test Performance Summary ===\n")
  if ("loso" %in% names(performance_comparisons)) {
    print(performance_comparisons$loso)
  }
  cat(sprintf(
    "  multivariate_results: with_leftout_ood_aware$%s + svm (calibration via R/calibration_reject_models_final.R)\n",
    CALIBRATION_MV_BASE_MODEL
  ))

  saveRDS(train_test_results, paste0("../data/out/final_train_test/final_train_test_results_10feb2026", merge_suffix, ".rds"))
  return(train_test_results)
}

# Run unmerged and fully merged versions (MDS.r, other.KMT2A, MECOM; maxprob method)
cat("=== Running Train/Test Analysis (Unmerged - MaxProb Method) ===\n")
train_test_results_unmerged <- main_train_test_analysis(merge_classes = FALSE)

cat("=== Running Train/Test Analysis (Merged MDS/KMT2A/MECOM - Sum Method) ===\n")
train_test_results_merged <- main_train_test_analysis(merge_classes = TRUE, merge_prob_method = "sum")

cat("=== Running Train/Test Analysis (Merged MDS/KMT2A/MECOM - Max Method) ===\n")
train_test_results_merged_maxprob <- main_train_test_analysis(merge_classes = TRUE, merge_prob_method = "max")
