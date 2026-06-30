# Backfill svm augmented LOSO folds into final_train_test_results*.rds for deployment calibration.
# Usage: Rscript R/backfill_svm_deploy_folds.R

suppressPackageStartupMessages(library(dplyr))

repo_root <- if (file.exists("R/utility_functions.R")) "." else ".."
source(file.path(repo_root, "R/utility_functions.R"))
source(file.path(repo_root, "R/calibration_reject_core.R"))
source(file.path(repo_root, "R/build_deploy_loso_fold_matrices.R"))

FINAL_FS_METHOD <- "eta2"
SCENARIO_KEY <- "with_leftout_ood_aware"

cohort_knn_features_path <- function(fs_method = "eta2") {
  file.path(repo_root, "data/out/final_train_test", paste0("cohort_knn_features_", fs_method, ".csv"))
}

attach_cohort_knn_to_probability_matrices <- function(probability_matrices, fs_method = "eta2") {
  knn_path <- cohort_knn_features_path(fs_method)
  if (!file.exists(knn_path)) {
    stop(sprintf("Missing cohort KNN file: %s", knn_path))
  }
  knn_df <- read.csv(knn_path, stringsAsFactors = FALSE)
  required_cols <- c("indices", KNN_DISTANCE_COLUMNS)
  knn_lookup <- knn_df[, required_cols, drop = FALSE]
  rownames(knn_lookup) <- as.character(knn_lookup$indices)
  attach_to_matrix <- function(m) {
    if (is.null(m) || !"indices" %in% colnames(m)) return(m)
    idx_chr <- as.character(m$indices)
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

backfill_one <- function(merge_classes) {
  merge_suffix <- if (merge_classes) "_merged_summed" else "_unmerged_maxprob"
  rds_path <- file.path(
    repo_root, sprintf("data/out/final_train_test/final_train_test_results_10feb2026%s.rds", merge_suffix)
  )
  if (!file.exists(rds_path)) stop("Missing ", rds_path)

  obj <- readRDS(rds_path)
  if ("svm" %in% names(obj$multivariate_results[[SCENARIO_KEY]])) {
    cat(sprintf("[%s] svm folds already present; skipping.\n", merge_suffix))
    return(invisible(rds_path))
  }

  label_mapping <- read.csv(file.path(repo_root, "data/label_mapping_all.csv"))
  leukemia_subtypes <- read.csv(file.path(repo_root, "data/rgas_10feb26.csv"))$ICC_Subtype

  leftout_file_configs <- list(
    svm = find_latest_csv(
      file.path(repo_root, "data/out/final_models/SVM"),
      "^SVM_final_loso_OvR_leftout.*\\.csv$"
    ),
    xgboost = find_latest_csv(
      file.path(repo_root, "data/out/final_models/XGBOOST"),
      "^XGBOOST_final_loso_OvR_leftout.*\\.csv$"
    ),
    neural_net = find_latest_csv(
      file.path(repo_root, "data/out/final_models/NN"),
      "^NN_final_loso_standard_leftout.*\\.csv$"
    )
  )
  if (!all(vapply(leftout_file_configs, function(p) !is.null(p) && file.exists(p), logical(1L)))) {
    stop("Missing left-out prediction CSVs for svm augmentation.")
  }

  probability_matrices <- attach_cohort_knn_to_probability_matrices(
    obj$probability_matrices, fs_method = FINAL_FS_METHOD
  )

  lo_svm <- build_leftout_probability_matrix(
    leftout_file_configs$svm, "svm", label_mapping, leukemia_subtypes, merge_classes = merge_classes
  )
  if (is.null(lo_svm)) stop("Could not parse SVM left-out file.")

  final_leftout_assignment_path <- file.path(
    repo_root, "data/out/final_train_test/leftout_fold_assignment_loso.csv"
  )
  leftout_fold_assignment <- load_leftout_fold_assignment(final_leftout_assignment_path)
  if (is.null(leftout_fold_assignment)) {
    stop("Missing ", final_leftout_assignment_path)
  }

  prob_aug <- list(
    svm = list(loso = augment_model_with_leftout(
      probability_matrices$svm$loso, lo_svm, leftout_fold_assignment
    ))
  )

  aug_svm <- list()
  for (outer_fold in names(prob_aug$svm$loso)) {
    svm_df <- prob_aug$svm$loso[[outer_fold]]
    svm_df <- add_roi_reject_features(svm_df)
    assert_finite_knn_columns(svm_df, sprintf("backfill svm fold %s", outer_fold))
    aug_svm[[outer_fold]] <- svm_df
  }
  if (length(aug_svm) < 2L) stop("Fewer than 2 svm augmented folds.")

  obj$multivariate_results[[SCENARIO_KEY]][["svm"]] <- list(
    loso = list(
      fold_matrices = copy_fold_matrix_list(aug_svm),
      model_label = "svm_augmented_disagreement_folds"
    )
  )
  saveRDS(obj, rds_path)
  cat(sprintf("[%s] Wrote %d svm folds to %s\n", merge_suffix, length(aug_svm), rds_path))
  invisible(rds_path)
}

cat("Backfilling svm deployment folds into final_train_test_results...\n")
backfill_one(merge_classes = FALSE)
backfill_one(merge_classes = TRUE)
cat("Done.\n")
