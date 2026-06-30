# MLL comparison tables: SVM deployment + SVM rejector recipes at 5% target risk.
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(caret)
})

repo_root <- if (file.exists("R/utility_functions.R")) "." else ".."
source(file.path(repo_root, "R/mll_label_standardization.R"))

mll_input_stem <- "STAR_AML_MLLlab"
mll_metadata_path <- file.path(repo_root, "data/MLL_lab/20231017_SampleMetadata.csv")
mll_pred_dir <- file.path(repo_root, "data/out/predictions", paste0(mll_input_stem, "_predictions"))

MLL_THRESHOLD_METHODS <- tibble::tribble(
  ~threshold_method, ~threshold_tag, ~threshold_label,
  "jackknife_adjusted", "", "Jackknife-adjusted",
  "pooled_oof", "_pooled_oof", "Pooled OOF",
  "ucb_95", "_ucb_95", "UCB (95%, one-sided)"
)

as_logical_flag <- function(x) {
  if (is.logical(x)) return(x)
  tolower(trimws(as.character(x))) %in% c("true", "t", "1", "yes", "y")
}

compute_kappa_safe <- function(truth, pred) {
  keep <- !is.na(truth) & !is.na(pred)
  truth <- as.character(truth[keep])
  pred <- as.character(pred[keep])
  if (length(truth) < 2L) return(NA_real_)
  lv <- sort(unique(c(truth, pred)))
  if (length(lv) < 2L) return(NA_real_)
  as.numeric(confusionMatrix(factor(pred, levels = lv), factor(truth, levels = lv))$overall["Kappa"])
}

mll_unseen_classes <- c("AML..NOS")

MLL_LABEL_SETS <- tibble::tribble(
  ~label_set_key, ~pred_suffix, ~label_display, ~target_risk_pct, ~risk_tag,
  "merged_summed", "merged_summed", "Collapsed", 5.0, "_risk5p0",
  "merged_maxprob", "merged_maxprob", "Collapsed (max)", 5.0, "_risk5p0",
  "unmerged_maxprob", "unmerged", "Full subtypes", 5.0, "_risk5p0"
)

MLL_REJECTOR_CONFIGS <- tibble::tribble(
  ~rejector_key, ~rejector_label,
  "svm_single_head", "SVM maxprob single-head",
  "svm_ridge_in_model", "SVM ridge in-model"
)

MLL_PRED_CONFIGS <- crossing(
  MLL_LABEL_SETS,
  MLL_REJECTOR_CONFIGS,
  tribble(
    ~cutoff_source, ~calibration_label, ~cutoff_tag,
    "selection_loso", "Selection LOSO", "",
    "deploy_loso", "Deploy LOSO", "_deploy_loso"
  ),
  MLL_THRESHOLD_METHODS
) %>%
  mutate(
    pred_path = file.path(
      mll_pred_dir,
      paste0(
        mll_input_stem, "_SVM_predictions_",
        pred_suffix, "_", rejector_key, cutoff_tag, threshold_tag, risk_tag, ".csv"
      )
    )
  )

missing <- MLL_PRED_CONFIGS %>% filter(!file.exists(pred_path)) %>% pull(pred_path)
if (length(missing)) {
  stop(
    "Missing prediction file(s). Run:\n",
    "  python python/predict_new_samples.py --cutoff_source both --threshold_method both ",
    "--rejector_mode all --input_file ... --output_dir data/out/predictions\n",
    paste0("  - ", missing, collapse = "\n")
  )
}

mll_meta <- read.csv(mll_metadata_path, sep = ";", stringsAsFactors = FALSE) %>%
  transmute(
    sample_name = as.character(MLL_ID),
    truth_raw = ICC_2022
  ) %>%
  filter(!is.na(truth_raw), nzchar(truth_raw))

read_metrics <- function(label_set_key, rejector_key, cutoff_source, threshold_method, target_risk_pct) {
  cfg <- MLL_PRED_CONFIGS %>%
    filter(
      label_set_key == !!label_set_key,
      rejector_key == !!rejector_key,
      cutoff_source == !!cutoff_source,
      threshold_method == !!threshold_method,
      target_risk_pct == !!target_risk_pct
    )
  pred_df <- read.csv(cfg$pred_path[[1]], stringsAsFactors = FALSE) %>%
    mutate(
      sample_name = as.character(sample_name),
      pred_label = standardize_mll_prediction(prediction, label_set_key),
      passed_cutoff = as_logical_flag(prediction_passed_cutoff)
    ) %>%
    left_join(mll_meta, by = "sample_name") %>%
    mutate(truth_label = standardize_mll_truth(truth_raw, label_set_key)) %>%
    filter(!is.na(truth_label), !is.na(pred_label))

  pred_acc <- pred_df %>% filter(passed_cutoff)
  seen_mask <- !(pred_df$truth_label %in% mll_unseen_classes)

  tibble(
    label_set_key = label_set_key,
    label_display = cfg$label_display[[1]],
    rejector_key = rejector_key,
    rejector_label = cfg$rejector_label[[1]],
    cutoff_source = cutoff_source,
    calibration_label = cfg$calibration_label[[1]],
    threshold_method = threshold_method,
    threshold_label = cfg$threshold_label[[1]],
    target_risk_pct = target_risk_pct,
    n_samples = nrow(pred_df),
    n_accepted = nrow(pred_acc),
    coverage_pct = 100 * mean(pred_df$passed_cutoff, na.rm = TRUE),
    seen_coverage_pct = if (any(seen_mask)) {
      100 * mean(pred_df$passed_cutoff[seen_mask], na.rm = TRUE)
    } else {
      NA_real_
    },
    accepted_error_rate_pct = if (nrow(pred_acc) > 0) {
      100 * mean(pred_acc$truth_label != pred_acc$pred_label, na.rm = TRUE)
    } else {
      NA_real_
    },
    kappa_without_rejection = compute_kappa_safe(pred_df$truth_label, pred_df$pred_label),
    kappa_with_rejection = compute_kappa_safe(pred_acc$truth_label, pred_acc$pred_label)
  )
}

mll_metrics <- pmap_dfr(
  MLL_PRED_CONFIGS %>% select(label_set_key, rejector_key, cutoff_source, threshold_method, target_risk_pct),
  read_metrics
)

mll_perf_table <- mll_metrics %>%
  mutate(across(where(is.numeric) & !target_risk_pct, ~round(.x, 3))) %>%
  select(
    threshold_label, calibration_label, label_display, target_risk_pct, rejector_label,
    accepted_error_rate_pct, coverage_pct, seen_coverage_pct,
    kappa_without_rejection, kappa_with_rejection, n_samples, n_accepted
  )

mll_calibration_delta <- mll_metrics %>%
  select(
    threshold_label, label_display, target_risk_pct, rejector_label, cutoff_source,
    accepted_error_rate_pct, coverage_pct, seen_coverage_pct,
    kappa_with_rejection, n_accepted
  ) %>%
  pivot_wider(
    names_from = cutoff_source,
    values_from = c(
      accepted_error_rate_pct, coverage_pct, seen_coverage_pct,
      kappa_with_rejection, n_accepted
    )
  ) %>%
  mutate(
    delta_accepted_error_pct = accepted_error_rate_pct_deploy_loso - accepted_error_rate_pct_selection_loso,
    delta_coverage_pct = coverage_pct_deploy_loso - coverage_pct_selection_loso,
    delta_kappa_accepted = kappa_with_rejection_deploy_loso - kappa_with_rejection_selection_loso,
    delta_n_accepted = n_accepted_deploy_loso - n_accepted_selection_loso
  )

out_dir <- file.path(repo_root, "writing/tables_new")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(mll_perf_table, file.path(out_dir, "mll_calibration_track_svm_comparison.csv"), row.names = FALSE)
write.csv(mll_calibration_delta, file.path(out_dir, "mll_calibration_track_svm_delta.csv"), row.names = FALSE)

cat("MLL calibration comparison [SVM] (deploy-loso minus selection-loso):\n")
print(
  mll_calibration_delta %>%
    select(threshold_label, label_display, rejector_label, delta_accepted_error_pct, delta_coverage_pct, delta_kappa_accepted, delta_n_accepted) %>%
    mutate(across(where(is.numeric), ~round(.x, 3)))
)
cat(sprintf("\nWrote %s\n", file.path(out_dir, "mll_calibration_track_svm_comparison.csv")))
cat(sprintf("Wrote %s\n", file.path(out_dir, "mll_calibration_track_svm_delta.csv")))
