# =============================================================================
# Final-model nested target-risk calibration (selection-safe feature grid)
# =============================================================================
# Reads final_train_test_results*.rds (built by R/train_test_analysis.R) and
# runs the same inner-CV feature selection / outer evaluation as
# calibration_reject_models.R, then exports deployment GLM coefficients and
# operating cutoffs for predict_new_samples.py.
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
})

if (file.exists("R/utility_functions.R")) {
  source("R/utility_functions.R")
} else if (file.exists("utility_functions.R")) {
  source("utility_functions.R")
} else {
  stop("Could not locate utility_functions.R. Run from repo root or R directory.")
}

repo_root <- if (file.exists("R/train_test_analysis.R")) "." else if (file.exists("train_test_analysis.R")) ".." else "."

ANALYSIS_INPUTS <- tibble::tribble(
  ~label_set_key,       ~label_set,            ~results_rel_path,
  "unmerged_maxprob",   "full_subtypes",       "data/out/final_train_test/final_train_test_results_10feb2026_unmerged_maxprob.rds",
  "merged_summed",      "collapsed_classes",   "data/out/final_train_test/final_train_test_results_10feb2026_merged_summed.rds"
)
ANALYSIS_INPUTS <- ANALYSIS_INPUTS %>% mutate(results_path = file.path(repo_root, results_rel_path))

OUTPUT_DIR <- file.path(repo_root, "data/out/final_train_test/calibration_feature_utility_selection_safe")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

PRIMARY_TABLE_TARGET_RISK <- 0.05
CALIBRATION_CURVE_TARGET_RISKS <- seq(20L, 100L, by = 5L) / 1000
TARGET_RISK <- PRIMARY_TABLE_TARGET_RISK
INNER_SELECTION_ANCHOR_RISKS <- c(0.03, 0.05, 0.10)
INNER_SELECTION_ANCHOR_LABELS <- c("p03", "p05", "p10")
INNER_RISK_BAND_LOW_OFFSET <- 0.005
SD_RISK_TIE_EPS <- 0.005

PARALLEL_MC_CORES <- suppressWarnings(as.integer(Sys.getenv("CALIBRATION_REJECT_MC_CORES", unset = NA_integer_)))
if (length(PARALLEL_MC_CORES) != 1L || is.na(PARALLEL_MC_CORES) || PARALLEL_MC_CORES < 1L) {
  dc <- suppressWarnings(parallel::detectCores())
  if (is.na(dc) || dc < 1L) dc <- 1L
  PARALLEL_MC_CORES <- max(1L, as.integer(dc) - 1L)
}
if (.Platform$OS.type != "unix") {
  PARALLEL_MC_CORES <- 1L
}

TARGET_BASE_MODEL <- "Global_Optimized"
BASELINE_TERMS <- c("max_prob")
ALL_FEATURE_TERMS <- c(
  "max_prob", "margin", "entropy", "top1_prob_variance_across_models",
  "knn10_mean_d", "knn10_min_d", "knn10_q90_d",
  "conformal_set_size_90"
)

SCENARIO_KEY <- "with_leftout_ood_aware"
SCENARIO_NAME <- "Single-head (OOD-aware train)"
POOL_RULE <- "all_rows"
TEST_RULE <- "all_rows"
CALIBRATION_RECIPE_INNER_BEST <- "inner_best_features"
CALIBRATION_RECIPE_MAX_PROB <- "max_prob_only"
BASELINE_ONLY_RHS_KEY <- "max_prob"
FULL_COVERAGE_THRESHOLD <- 0

cat("Starting final-model nested target-risk calibration...\n")
cat(sprintf("  Scenario: %s, label sets: %d\n", SCENARIO_KEY, nrow(ANALYSIS_INPUTS)))
cat(sprintf("  Base model: %s\n", TARGET_BASE_MODEL))
cat(sprintf("  Output directory: %s\n", OUTPUT_DIR))
cat(sprintf("  Inputs: %s\n", paste(ANALYSIS_INPUTS$results_path, collapse = "; ")))

source(if (file.exists("R/calibration_reject_core.R")) {
  "R/calibration_reject_core.R"
} else {
  "calibration_reject_core.R"
})

merge_suffix_from_key <- function(label_set_key) {
  if (label_set_key == "unmerged_maxprob") "_unmerged_maxprob" else "_merged_summed"
}

# Fit pooled deployment GLM and write coef + cutoff CSVs expected by predict_new_samples.py.
export_final_deployment_artifacts <- function(
  results_obj,
  nested_res,
  label_set_key,
  label_set,
  risk_target = PRIMARY_TABLE_TARGET_RISK
) {
  merge_suffix <- merge_suffix_from_key(label_set_key)
  base_dir <- file.path(repo_root, "data/out/final_train_test")
  params_dir <- file.path(base_dir, paste0("multivariate_params", merge_suffix))
  cutoffs_dir <- file.path(base_dir, paste0("cutoffs", merge_suffix))
  dir.create(params_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(cutoffs_dir, recursive = TRUE, showWarnings = FALSE)

  pf <- nested_res$per_fold_df
  if (is.null(pf) || nrow(pf) == 0L) {
    stop(sprintf("No per-fold nested results for %s; cannot export deployment artifacts.", label_set_key))
  }
  pf_loso <- pf[pf$split_type == "loso" & pf$label_set == label_set, , drop = FALSE]
  if (nrow(pf_loso) == 0L) {
    stop(sprintf("No LOSO per-fold rows for label_set=%s (%s).", label_set, label_set_key))
  }

  rhs_tab <- sort(table(pf_loso$inner_winner_rhs_key), decreasing = TRUE)
  rhs_key <- names(rhs_tab)[[1]]
  rhs_terms <- strsplit(rhs_key, ";", fixed = TRUE)[[1]]

  fam <- results_obj$multivariate_results[[SCENARIO_KEY]][[TARGET_BASE_MODEL]]$loso$fold_matrices
  if (is.null(fam) || length(fam) < 2L) {
    stop(sprintf(
      "Missing augmented fold matrices for %s. Run train_test_analysis.R with final left-out predictions.",
      label_set_key
    ))
  }

  fold_feats <- lapply(copy_fold_matrix_list(fam), extract_features)
  train_df <- bind_rows(fold_feats)
  fit_obj <- fit_binary_model(train_df, "accept_combined", rhs_terms)
  if (is.null(fit_obj)) {
    stop(sprintf("Could not fit deployment GLM for %s with RHS: %s", label_set_key, rhs_key))
  }

  coef_vec <- stats::coef(fit_obj$fit)
  params_df <- data.frame(
    model = TARGET_BASE_MODEL,
    term = names(coef_vec),
    estimate = as.numeric(coef_vec),
    stringsAsFactors = FALSE
  )
  params_path <- file.path(
    params_dir,
    paste0("multivariate_params_ood_aware", merge_suffix, ".csv")
  )
  write.csv(params_df, params_path, row.names = FALSE)
  cat(sprintf("  Exported deployment GLM: %s\n", params_path))

  thr <- threshold_from_oof_pool_singlehead(
    fold_feats,
    names(fold_feats),
    "accept_combined",
    fit_obj$rhs_terms,
    POOL_RULE,
    TEST_RULE,
    risk_target
  )
  if (!is.finite(thr)) {
    stop(sprintf("Could not derive deployment cutoff for %s at risk=%.2f%%", label_set_key, 100 * risk_target))
  }

  cutoffs_df <- data.frame(
    model = TARGET_BASE_MODEL,
    prob_cutoff = thr,
    source = "loso",
    requested_target_risk = risk_target,
    inner_winner_rhs_key = rhs_key,
    stringsAsFactors = FALSE
  )
  cutoffs_path <- file.path(cutoffs_dir, paste0("deploy_cutoffs", merge_suffix, ".csv"))
  write.csv(cutoffs_df, cutoffs_path, row.names = FALSE)
  cat(sprintf("  Exported deployment cutoff: %s (threshold=%.4f)\n", cutoffs_path, thr))

  invisible(list(params_path = params_path, cutoffs_path = cutoffs_path, rhs_key = rhs_key))
}

all_per_fold <- list()
all_rejection_stratum_per_fold <- list()
all_summary <- list()
all_max_prob_per_fold <- list()
all_summary_max_prob <- list()
all_heat <- list()
all_inner_ranked <- list()
calibration_curve_chunks <- list()
calibration_curve_per_fold_chunks <- list()
calibration_compare_chunks <- list()
full_coverage_per_fold_rows <- list()
classifier_only_per_fold_rows <- list()
p_idx <- rej_idx <- s_idx <- mpf_idx <- smp_idx <- h_idx <- in_idx <- cc_idx <- cc_pf_idx <- cc_cmp_idx <- fc_idx <- co_idx <- 1L

for (i in seq_len(nrow(ANALYSIS_INPUTS))) {
  row <- ANALYSIS_INPUTS[i, ]
  if (!file.exists(row$results_path)) {
    stop(sprintf("Missing final results file: %s. Run R/train_test_analysis.R first.", row$results_path))
  }
  cat(sprintf("Loading %s ...\n", row$results_path))
  obj <- readRDS(row$results_path)

  cat(sprintf(
    "  [%s] Nested target-risk selection (%s), primary risk = %.0f%% ...\n",
    row$label_set, SCENARIO_KEY, 100 * PRIMARY_TABLE_TARGET_RISK
  ))
  res <- run_nested_target_risk_analysis(obj, row$label_set, risk_target = PRIMARY_TABLE_TARGET_RISK)

  export_final_deployment_artifacts(
    obj, res, row$label_set_key, row$label_set, risk_target = PRIMARY_TABLE_TARGET_RISK
  )

  merge_suffix <- merge_suffix_from_key(row$label_set_key)
  cutoffs_dir <- file.path(repo_root, "data/out/final_train_test", paste0("cutoffs", merge_suffix))
  deploy_curve <- build_deploy_risk_coverage_curve_from_stubs(
    res$recipe_jobs,
    c(PRIMARY_TABLE_TARGET_RISK, CALIBRATION_CURVE_TARGET_RISKS),
    target_model = TARGET_BASE_MODEL
  )
  deploy_curve_loso <- deploy_curve %>%
    dplyr::filter(.data$split_type == "loso", .data$label_set == row$label_set)
  if (nrow(deploy_curve_loso) == 0L) {
    stop(sprintf("No LOSO deploy risk-coverage curve rows for %s.", row$label_set_key))
  }
  deploy_curve_path <- file.path(
    cutoffs_dir,
    paste0("deploy_risk_coverage_curve", merge_suffix, ".csv")
  )
  readr::write_csv(
    deploy_curve_loso %>%
      dplyr::select(
        model, prob_cutoff, mean_risk, mean_coverage, requested_target_risk_pct
      ),
    deploy_curve_path
  )
  cat(sprintf("  Exported deploy risk-coverage curve: %s\n", deploy_curve_path))

  curve_part <- build_calibration_curve_from_stubs(res$recipe_jobs, CALIBRATION_CURVE_TARGET_RISKS)
  if (nrow(curve_part) > 0L) {
    calibration_curve_chunks[[cc_idx]] <- curve_part
    cc_idx <- cc_idx + 1L
  }
  compare_part <- build_calibration_compare_curves(res$recipe_jobs, CALIBRATION_CURVE_TARGET_RISKS)
  if (nrow(compare_part) > 0L) {
    calibration_compare_chunks[[cc_cmp_idx]] <- compare_part
    cc_cmp_idx <- cc_cmp_idx + 1L
  }
  for (stub in res$recipe_jobs) {
    fc_row <- evaluate_full_coverage_from_stub(stub)
    if (!is.null(fc_row)) {
      full_coverage_per_fold_rows[[fc_idx]] <- fc_row
      fc_idx <- fc_idx + 1L
    }
    co_row <- evaluate_classifier_only_full_coverage(stub$target_df)
    if (!is.null(co_row)) {
      co_row$label_set <- stub$label_set
      co_row$split_type <- stub$split_type
      co_row$target_fold <- as.character(stub$fold_name)
      classifier_only_per_fold_rows[[co_idx]] <- co_row
      co_idx <- co_idx + 1L
    }
  }
  rej_pf <- build_rejection_stratum_per_fold(res$recipe_jobs, PRIMARY_TABLE_TARGET_RISK)
  if (nrow(rej_pf) > 0L) {
    all_rejection_stratum_per_fold[[rej_idx]] <- rej_pf
    rej_idx <- rej_idx + 1L
  }
  max_prob_pf <- build_max_prob_per_fold_primary(res$recipe_jobs, PRIMARY_TABLE_TARGET_RISK)
  if (nrow(max_prob_pf) > 0L) {
    all_max_prob_per_fold[[mpf_idx]] <- max_prob_pf
    mpf_idx <- mpf_idx + 1L
    max_prob_sum <- summarize_four_settings(max_prob_pf)
    if (nrow(max_prob_sum) > 0L) {
      all_summary_max_prob[[smp_idx]] <- max_prob_sum
      smp_idx <- smp_idx + 1L
    }
  }
  if (nrow(res$per_fold_df) > 0L) {
    all_per_fold[[p_idx]] <- res$per_fold_df
    p_idx <- p_idx + 1L
  }
  if (nrow(res$summary_4) > 0L) {
    all_summary[[s_idx]] <- res$summary_4
    s_idx <- s_idx + 1L
  }
  if (nrow(res$inner_scores_ranked) > 0L) {
    all_inner_ranked[[in_idx]] <- res$inner_scores_ranked
    in_idx <- in_idx + 1L
  }
  if (nrow(res$heatmap_long) > 0L) {
    all_heat[[h_idx]] <- res$heatmap_long
    h_idx <- h_idx + 1L
  }
}

per_fold_out <- if (length(all_per_fold) == 0L) data.frame() else bind_rows(all_per_fold)
summary_out <- if (length(all_summary) == 0L) data.frame() else bind_rows(all_summary)
heatmap_out <- if (length(all_heat) == 0L) data.frame() else bind_rows(all_heat)
inner_scores_out <- if (length(all_inner_ranked) == 0L) data.frame() else bind_rows(all_inner_ranked)
calibration_curve_out <- if (length(calibration_curve_chunks) == 0L) {
  data.frame()
} else {
  bind_rows(calibration_curve_chunks)
}
calibration_compare_out <- if (length(calibration_compare_chunks) == 0L) {
  data.frame()
} else {
  bind_rows(calibration_compare_chunks)
}

if (nrow(per_fold_out) > 0L) {
  write_csv(per_fold_out, file.path(OUTPUT_DIR, "final_nested_target_risk_per_fold.csv"))
}
if (nrow(summary_out) > 0L) {
  write_csv(summary_out, file.path(OUTPUT_DIR, "final_nested_target_risk_summary.csv"))
}
if (nrow(heatmap_out) > 0L) {
  write_csv(heatmap_out, file.path(OUTPUT_DIR, "final_nested_target_risk_feature_heatmap_long.csv"))
}
if (nrow(inner_scores_out) > 0L) {
  write_csv(inner_scores_out, file.path(OUTPUT_DIR, "final_nested_target_risk_inner_scores_ranked.csv"))
}
if (nrow(calibration_curve_out) > 0L) {
  write_csv(calibration_curve_out, file.path(OUTPUT_DIR, "final_nested_target_risk_calibration_curve.csv"))
}
if (nrow(calibration_compare_out) > 0L) {
  write_csv(calibration_compare_out, file.path(OUTPUT_DIR, "final_nested_target_risk_calibration_compare.csv"))
}

manifest <- data.frame(
  key = c("timestamp_utc", "output_dir", "inputs", "deployment_exports"),
  value = c(
    format(Sys.time(), tz = "UTC", usetz = TRUE),
    OUTPUT_DIR,
    paste(ANALYSIS_INPUTS$results_path, collapse = "; "),
    "multivariate_params_ood_aware{suffix}.csv + deploy_cutoffs{suffix}.csv per label set"
  ),
  stringsAsFactors = FALSE
)
write_csv(manifest, file.path(OUTPUT_DIR, "final_nested_target_risk_manifest.csv"))

cat("Final-model nested target-risk calibration complete.\n")
