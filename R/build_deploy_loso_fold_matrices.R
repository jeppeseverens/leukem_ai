# =============================================================================
# Build Option B deploy-loso ensemble fold matrices for rejector calibration
# =============================================================================
# Reads deploy-loso SVM CSVs (python/run_deploy_calibration_loso.py), augments each
# fold with left-out OOD rows (same leftout CSVs as Track A), and writes fold
# matrices for calibration_reject_models_deploy_b.R.
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
})

repo_root <- if (file.exists("R/utility_functions.R")) "." else if (file.exists("utility_functions.R")) ".." else "."
util_path <- file.path(repo_root, if (file.exists("R/utility_functions.R")) "R/utility_functions.R" else "utility_functions.R")
core_path <- file.path(repo_root, if (file.exists("R/calibration_reject_core.R")) "R/calibration_reject_core.R" else "calibration_reject_core.R")
source(util_path)
source(core_path)

DEPLOY_LOSO_ROOT <- file.path(repo_root, "data/out/final_train_test/deploy_loso")
DEPLOY_LOSO_RESULTS_TEMPLATE <- "data/out/final_train_test/deploy_loso_fold_results_10feb2026%s.rds"
FINAL_FS_METHOD <- "eta2"

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

find_latest_csv <- function(dir_path, pattern) {
  if (!dir.exists(dir_path)) return(NULL)
  files <- list.files(dir_path, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(NULL)
  files[which.max(file.info(files)$mtime)]
}

# Left-out probability matrix from final-model leftout CSV (Track A / outer CV).
build_leftout_probability_matrix <- function(
  leftout_csv, model_type, label_mapping, all_subtypes, merge_classes = FALSE
) {
  if (is.null(leftout_csv) || !file.exists(leftout_csv)) return(NULL)
  lo <- read.csv(leftout_csv, stringsAsFactors = FALSE)
  if (nrow(lo) == 0) return(NULL)
  if (!"sample_indices" %in% colnames(lo) || !"preds_prob" %in% colnames(lo)) return(NULL)

  sample_indices <- parse_numeric_string(lo$sample_indices[1])
  if (length(sample_indices) == 0) return(NULL)

  if (model_type %in% c("svm", "xgboost")) {
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
  prob_df$sample_indices <- sample_indices
  prob_df$is_leftout <- TRUE
  prob_df <- add_optional_knn_columns(prob_df, lo[1, , drop = FALSE], length(sample_indices))
  prob_df
}

load_leftout_fold_assignment <- function(leftout_assignment_csv) {
  if (is.null(leftout_assignment_csv) || !file.exists(leftout_assignment_csv)) return(NULL)
  df <- read.csv(leftout_assignment_csv, stringsAsFactors = FALSE)
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

augment_model_with_leftout <- function(model_fold_matrices, leftout_matrix, fold_assignment = NULL) {
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

    if ("indices" %in% colnames(known) && "indices" %in% colnames(lo)) {
      overlap_idx <- intersect(unique(known$indices[is.finite(known$indices)]),
                               unique(lo$indices[is.finite(lo$indices)]))
      if (length(overlap_idx) > 0) {
        stop(sprintf(
          "Left-out augmentation duplicate indices in fold %s: %s",
          fold_name, paste(head(overlap_idx, 15), collapse = ", ")
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
      out[[fold_name]] <- rbind(known[, common_cols, drop = FALSE], lo[, common_cols, drop = FALSE])
    }
  }
  out
}

assert_leftout_fold_disjoint <- function(prob_aug_by_model) {
  for (model_name in names(prob_aug_by_model)) {
    folds <- prob_aug_by_model[[model_name]]$loso
    if (!is.list(folds) || length(folds) == 0) next
    per_fold_idx <- lapply(folds, function(m) {
      if (is.null(m) || !"sample_indices" %in% colnames(m) || !"is_leftout" %in% colnames(m)) {
        return(integer(0))
      }
      vals <- m$sample_indices[as.logical(m$is_leftout)]
      vals[!is.na(vals)]
    })
    all_idx <- unlist(per_fold_idx, use.names = FALSE)
    if (length(all_idx) == 0) next
    dup <- duplicated(all_idx)
    if (any(dup)) {
      stop(sprintf(
        "Leftout partition not disjoint for model %s: sample_index %s in multiple folds.",
        model_name, paste(unique(all_idx[dup]), collapse = ", ")
      ))
    }
  }
}

build_deploy_loso_fold_matrices <- function(merge_classes = FALSE, merge_prob_method = c("sum", "max")) {
  merge_prob_method <- match.arg(merge_prob_method)
  merge_suffix <- if (!merge_classes) {
    "_unmerged_maxprob"
  } else if (merge_prob_method == "max") {
    "_merged_maxprob"
  } else {
    "_merged_summed"
  }
  cat(sprintf("\n=== Deploy-loso fold matrices (%s) ===\n", merge_suffix))

  label_mapping <- read.csv(file.path(repo_root, "data/label_mapping_all.csv"))
  leukemia_subtypes <- read.csv(file.path(repo_root, "data/rgas_10feb26.csv"))$ICC_Subtype
  meta <- read.csv(file.path(repo_root, "data/meta_20aug25.csv"))
  study_names <- meta$Studies

  DATA_FILTERS <- list(
    min_samples_per_subtype = 10,
    excluded_subtypes = c("AML NOS", "Missing data", "Multi"),
    selected_studies = c(
      "TCGA-LAML", "LEUCEGENE", "BEATAML1.0-COHORT",
      "AAML0531", "AAML1031", "AAML03P1", "100LUMC"
    )
  )

  subtypes_with_sufficient_samples <- names(which(table(leukemia_subtypes) >= DATA_FILTERS$min_samples_per_subtype))
  filter <- which(
    leukemia_subtypes %in% subtypes_with_sufficient_samples &
      !leukemia_subtypes %in% DATA_FILTERS$excluded_subtypes &
      study_names %in% DATA_FILTERS$selected_studies
  )
  filtered_leukemia_subtypes <- leukemia_subtypes[filter]
  filtered_study_names <- study_names[filter]

  map_filtered_local_to_global_indices <- function(sample_indices_zero_based, filter_vec) {
    local_one_based <- sample_indices_zero_based + 1L
    if (length(local_one_based) == 0) return(integer(0))
    if (any(local_one_based < 1L | local_one_based > length(filter_vec))) {
      stop(sprintf(
        "Found sample_indices outside filtered index range [0, %d]. Example values: %s",
        length(filter_vec) - 1L,
        paste(head(sample_indices_zero_based, 15), collapse = ", ")
      ))
    }
    as.integer(filter_vec[local_one_based])
  }

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
        if (nrow(inner_fold_data) == 0 ||
            is.null(inner_fold_data$preds_prob[1]) ||
            is.na(inner_fold_data$preds_prob[1])) next

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
          if (length(probs) == num_samples) probability_matrix[, j] <- probs

          target_values <- parse_numeric_string(best_param_row$y_val)
          true_labels_vector[target_values == 1] <- current_class_label
        }

        if (all(is.na(true_labels_vector))) next

        probability_matrix <- t(apply(probability_matrix, 1, function(row) row / sum(row)))
        probability_matrix <- data.frame(probability_matrix)
        probability_matrix <- ensure_all_class_columns(probability_matrix, label_mapping)
        probability_matrix$y <- make.names(true_labels_vector)
        probability_matrix$inner_fold <- inner_fold_id
        probability_matrix$outer_fold <- outer_fold_id
        local_idx <- parse_numeric_string(inner_fold_data$sample_indices[1])
        probability_matrix$indices <- map_filtered_local_to_global_indices(local_idx, filter)
        probability_matrix$study <- study_names[probability_matrix$indices]
        probability_matrix <- add_optional_knn_columns(
          probability_matrix, inner_fold_data[1, , drop = FALSE], num_samples
        )
        if (merge_classes) {
          probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_prob_method = merge_prob_method)
        }
        fold_matrices[[as.character(inner_fold_id)]] <- probability_matrix
      }

      if (length(fold_matrices) > 0) {
        probability_matrices[[as.character(outer_fold_id)]] <- do.call(rbind, fold_matrices)
        probability_matrices[[as.character(outer_fold_id)]][
          is.na(probability_matrices[[as.character(outer_fold_id)]])
        ] <- 0
      }
    }
    probability_matrices
  }

  generate_standard_probability_matrices <- function(cv_results_df, best_params_df, label_mapping, filtered_subtypes, study_names, merge_classes = FALSE, merge_prob_method = "sum") {
    outer_fold_ids <- unique(cv_results_df$outer_fold)
    probability_matrices <- list()

    for (outer_fold_id in outer_fold_ids) {
      outer_fold_data <- cv_results_df[cv_results_df$outer_fold == outer_fold_id, ]
      inner_fold_ids <- unique(outer_fold_data$inner_fold)
      fold_matrices <- list()

      for (inner_fold_id in inner_fold_ids) {
        best_param <- best_params_df$params[1]
        inner_fold_data <- outer_fold_data[outer_fold_data$params == best_param, ]
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
        probability_matrix$inner_fold <- inner_fold_id
        probability_matrix$outer_fold <- outer_fold_id
        local_idx <- parse_numeric_string(inner_fold_data$sample_indices)
        probability_matrix$indices <- map_filtered_local_to_global_indices(local_idx, filter)
        probability_matrix$study <- study_names[probability_matrix$indices]
        probability_matrix <- add_optional_knn_columns(
          probability_matrix, inner_fold_data[1, , drop = FALSE], num_samples
        )
        if (merge_classes) {
          probability_matrix <- merge_classes_in_matrix(probability_matrix, merge_prob_method = merge_prob_method)
        }
        fold_matrices[[as.character(inner_fold_id)]] <- probability_matrix
      }

      probability_matrices[[as.character(outer_fold_id)]] <- do.call(rbind, fold_matrices)
    }
    probability_matrices
  }

  MODEL_CONFIGS <- list(
    svm = list(
      classification_type = "OvR",
      file_paths = list(loso = file.path(DEPLOY_LOSO_ROOT, "svm")),
      output_dir = file.path(repo_root, "data/out/final_train_test/best_params/SVM")
    )
  )

  for (cfg in MODEL_CONFIGS) {
    if (!dir.exists(cfg$file_paths$loso)) {
      stop(sprintf(
        "Missing deploy-loso directory: %s. Run bash/run_deploy_calibration_loso.sh first.",
        cfg$file_paths$loso
      ))
    }
  }

  model_results <- load_all_model_data(MODEL_CONFIGS, group_nn_by_outer_fold = FALSE)
  best_parameters <- extract_all_best_parameters(model_results, MODEL_CONFIGS, include_outer_fold = FALSE)

  probability_matrices <- list()
  for (model_name in names(model_results)) {
    config <- MODEL_CONFIGS[[model_name]]
    probability_matrices[[model_name]] <- list()
    for (fold_type in names(model_results[[model_name]])) {
      results <- model_results[[model_name]][[fold_type]]
      best_params <- best_parameters[[model_name]][[fold_type]]
      if (is.null(results) || is.null(best_params)) next
      if (config$classification_type == "OvR") {
        probs <- generate_ovr_probability_matrices(
          results, best_params, label_mapping, filtered_study_names, merge_classes = merge_classes, merge_prob_method = merge_prob_method
        )
      } else {
        probs <- generate_standard_probability_matrices(
          results, best_params, label_mapping, filtered_leukemia_subtypes,
          filtered_study_names, merge_classes = merge_classes, merge_prob_method = merge_prob_method
        )
      }
      probability_matrices[[model_name]][[fold_type]] <- probs
    }
  }

  # Augment deploy-loso known-cohort folds with left-out OOD (mirrors Track A).
  cat("Augmenting deploy-loso folds with left-out OOD samples...\n")
  leftout_file_configs <- list(
    svm = find_latest_csv(
      file.path(repo_root, "data/out/final_models/SVM"), "^SVM_final_loso_OvR_leftout.*\\.csv$"
    )
  )
  if (is.null(leftout_file_configs$svm) || !file.exists(leftout_file_configs$svm)) {
    stop(
      "Missing SVM final-model leftout CSV. ",
      "Run run_final_train.py --include_leftout before build_deploy_loso_fold_matrices.R."
    )
  }

  leftout_assignment_path <- file.path(repo_root, "data/out/final_train_test/leftout_fold_assignment_loso.csv")
  leftout_fold_assignment <- load_leftout_fold_assignment(leftout_assignment_path)
  if (is.null(leftout_fold_assignment)) {
    stop(sprintf("Missing leftout fold assignment: %s", leftout_assignment_path))
  }
  cat(sprintf("  Using leftout fold assignment: %s\n", leftout_assignment_path))

  lo_svm <- build_leftout_probability_matrix(
    leftout_file_configs$svm, "svm", label_mapping, leukemia_subtypes, merge_classes = FALSE
  )
  if (is.null(lo_svm)) {
    stop("Could not parse SVM left-out prediction file.")
  }

  probability_matrices <- list(
    svm = list(loso = augment_model_with_leftout(
      probability_matrices$svm$loso, lo_svm, leftout_fold_assignment
    ))
  )
  assert_leftout_fold_disjoint(probability_matrices)

  aug_svm <- list()
  for (outer_fold in names(probability_matrices$svm$loso)) {
    svm_df <- probability_matrices$svm$loso[[outer_fold]]
    svm_df <- add_roi_reject_features(svm_df)
    assert_finite_knn_columns(
      svm_df,
      sprintf("deploy-loso SVM fold %s", outer_fold)
    )
    aug_svm[[outer_fold]] <- svm_df
  }

  if (length(aug_svm) < 2L) {
    stop(sprintf("Deploy-loso SVM has fewer than 2 folds (%d).", length(aug_svm)))
  }

  CALIBRATION_MV_BASE_MODEL <- "svm"
  deploy_loso_results <- list(
    multivariate_results = list(
      with_leftout_ood_aware = list(
        svm = list(
          loso = list(
            fold_matrices = copy_fold_matrix_list(aug_svm),
            model_label = "svm_deploy_loso_ood_aware"
          )
        )
      )
    ),
    calibration_source = "deploy_loso",
    fs_method = FINAL_FS_METHOD,
    merge_suffix = merge_suffix
  )

  out_path <- file.path(repo_root, sprintf(DEPLOY_LOSO_RESULTS_TEMPLATE, merge_suffix))
  saveRDS(deploy_loso_results, out_path)
  cat(sprintf(
    "Saved %d deploy-loso SVM fold matrices to %s\n",
    length(aug_svm), out_path
  ))
  invisible(deploy_loso_results)
}

cat("Building deploy-loso fold matrices (unmerged + merged)...\n")
if (sys.nframe() == 0L) {
  build_deploy_loso_fold_matrices(merge_classes = FALSE)
  build_deploy_loso_fold_matrices(merge_classes = TRUE, merge_prob_method = "sum")
  build_deploy_loso_fold_matrices(merge_classes = TRUE, merge_prob_method = "max")
  cat("Done.\n")
}
