# =============================================================================
# Selection-Safe Calibration Feature Utility Analysis (Strict Regime Design)
# =============================================================================
# OOD-aware single-head rejector (with_leftout_ood_aware): exhaustive feature grid;
# inner CV scores each recipe at 3%, 5%, 10% requested risk; fuse rankings (sum of ranks) for one RHS;
# inner winner evaluated on held-out outer fold.
# Inner CV: train pool minus val | threshold from LOSO-OOF on train | metrics on val.
# Across:
#   - split_type: CV / LOSO
#   - label_set: full_subtypes / collapsed_classes
# Outer folds run in parallel on macOS/Linux via parallel::mclapply (fork). Set env e.g.
#   CALIBRATION_REJECT_MC_CORES=8
# or use 1 to force sequential.
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

ANALYSIS_INPUTS <- tibble::tribble(
  ~label_set_key,       ~label_set,            ~results_rel_path,
  "unmerged_maxprob",   "full_subtypes",       "data/out/outer_cv/outer_cv_analysis_outputs_unmerged_maxprob/outer_cv_results.rds",
  "merged_summed",      "collapsed_classes",   "data/out/outer_cv/outer_cv_analysis_outputs_merged_summed/outer_cv_results.rds"
)

repo_root <- if (file.exists("R/outer_cv_analysis.R")) "." else if (file.exists("outer_cv_analysis.R")) ".." else "."
ANALYSIS_INPUTS <- ANALYSIS_INPUTS %>% mutate(results_path = file.path(repo_root, results_rel_path))

OUTPUT_DIR <- file.path(repo_root, "data/out/outer_cv/calibration_feature_utility_selection_safe")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
# Primary CSVs (per-fold summary, heatmap, inner grid): this risk only.
PRIMARY_TABLE_TARGET_RISK <- 0.05
# Calibration curve: outer threshold refit per requested risk (1% … 10% step 0.5%); RHS fixed to primary inner winner.
CALIBRATION_CURVE_TARGET_RISKS <- seq(20L, 100L, by = 5L) / 1000 # e.g. 0.010, 0.015, …, 0.100
TARGET_RISK <- PRIMARY_TABLE_TARGET_RISK
# Inner fusion: score + rank grid at each anchor; RHS minimizing sum(inner_rank); ties — max rank, p05 / p03 / p10 rank, rhs_key.
INNER_SELECTION_ANCHOR_RISKS <- c(0.03, 0.05, 0.10)
INNER_SELECTION_ANCHOR_LABELS <- c("p03", "p05", "p10")
# Within-anchor ranking (`rank_inner_scores`): risk band + sd_risk tier + coverage + complexity.
INNER_RISK_BAND_LOW_OFFSET <- 0.005
SD_RISK_TIE_EPS <- 0.005

# Parallel outer folds: fork workers via parallel::mclapply (macOS/Linux, incl. Apple Silicon).
# Windows falls back to sequential (mc.cores > 1 is not fork-based there).
PARALLEL_MC_CORES <- suppressWarnings(as.integer(Sys.getenv("CALIBRATION_REJECT_MC_CORES", unset = NA_integer_)))
if (length(PARALLEL_MC_CORES) != 1L || is.na(PARALLEL_MC_CORES) || PARALLEL_MC_CORES < 1L) {
  dc <- suppressWarnings(parallel::detectCores())
  if (is.na(dc) || dc < 1L) dc <- 1L
  PARALLEL_MC_CORES <- max(1L, as.integer(dc) - 1L)
}
if (.Platform$OS.type != "unix") {
  PARALLEL_MC_CORES <- 1L
}
PARALLEL_MC_CORES <- 4
TARGET_BASE_MODEL <- "Global_Product_Optimized"
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
# Calibration-curve comparison: fused inner winner vs probability-only rejector.
CALIBRATION_RECIPE_INNER_BEST <- "inner_best_features"
CALIBRATION_RECIPE_MAX_PROB <- "max_prob_only"
BASELINE_ONLY_RHS_KEY <- "max_prob"
# Accept-all threshold for full seen-class coverage baseline (risk–coverage curve origin).
FULL_COVERAGE_THRESHOLD <- 0

cat("Starting nested target-risk calibration (strict inner-CV thresholding)...\n")
cat(sprintf("  Scenario: %s, splits: 2 (CV/LOSO), label sets: %d\n", SCENARIO_KEY, nrow(ANALYSIS_INPUTS)))
cat(sprintf("  Base model: %s\n", TARGET_BASE_MODEL))
cat(sprintf("  Feature set: %s\n", paste(ALL_FEATURE_TERMS, collapse = ", ")))
cat(sprintf(
  "  Inner fusion anchors: %s%% (rank each recipe grid per anchor; winner minimizes sum of ranks).\n",
  paste(100 * INNER_SELECTION_ANCHOR_RISKS, collapse = ", ")
))
cat(sprintf(
  "  Per-anchor rank bands: low = anchor - %.1f%% (else fallback); sd_risk within %.3f of tier min, etc.\n",
  100 * INNER_RISK_BAND_LOW_OFFSET,
  SD_RISK_TIE_EPS
))
cat(sprintf(
  "  Primary table / heatmap / inner-grid outputs: target risk = %.0f%%.\n",
  100 * PRIMARY_TABLE_TARGET_RISK
))
cat(sprintf(
  "  Calibration curve CSV: risks %s%% using fixed inner winner from %.0f%% (threshold refit per target only).\n",
  paste(format(100 * CALIBRATION_CURVE_TARGET_RISKS, trim = TRUE), collapse = ", "),
  100 * PRIMARY_TABLE_TARGET_RISK
))
cat(sprintf(
  "  Calibration compare CSV: %s vs %s (same risk sweep; threshold refit per target).\n",
  CALIBRATION_RECIPE_INNER_BEST,
  CALIBRATION_RECIPE_MAX_PROB
))
cat("  Inner: per val fold, train on pool\\val, threshold from LOSO-OOF on train, evaluate on val.\n")
cat("  Outer: threshold from LOSO-OOF on full pool; final model on pool; evaluate on held-out target.\n")
cat(sprintf(
  "  Parallel outer folds: mc.cores=%d (%s); override with CALIBRATION_REJECT_MC_CORES\n",
  PARALLEL_MC_CORES,
  if (.Platform$OS.type == "unix") "fork via parallel::mclapply" else "sequential (non-Unix)"
))
cat(sprintf("  Output directory: %s\n", OUTPUT_DIR))


source("calibration_reject_core.R")

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
p_idx <- 1L
rej_idx <- 1L
s_idx <- 1L
mpf_idx <- 1L
smp_idx <- 1L
h_idx <- 1L
in_idx <- 1L
cc_idx <- 1L
cc_pf_idx <- 1L
cc_cmp_idx <- 1L
fc_idx <- 1L
co_idx <- 1L

for (i in seq_len(nrow(ANALYSIS_INPUTS))) {
  row <- ANALYSIS_INPUTS[i, ]
  if (!file.exists(row$results_path)) {
    warning(sprintf("Skipping missing results file: %s", row$results_path))
    next
  }
  cat(sprintf("Loading %s ...\n", row$results_path))
  obj <- readRDS(row$results_path)

  cat(sprintf(
    "  [%s] Nested target-risk selection (%s), primary inner/outer risk = %.0f%% ...\n",
    row$label_set, SCENARIO_KEY, 100 * PRIMARY_TABLE_TARGET_RISK
  ))
  res <- run_nested_target_risk_analysis(obj, row$label_set, risk_target = PRIMARY_TABLE_TARGET_RISK)
  cat(sprintf(
    "    rows=%d summary=%d heatmap_long=%d inner-ranked=%d recipe_stubs=%d\n",
    nrow(res$per_fold_df),
    nrow(res$summary_4),
    nrow(res$heatmap_long),
    nrow(res$inner_scores_ranked),
    length(res$recipe_jobs)
  ))

  curve_part <- build_calibration_curve_from_stubs(res$recipe_jobs, CALIBRATION_CURVE_TARGET_RISKS)
  if (nrow(curve_part) > 0L) {
    calibration_curve_chunks[[cc_idx]] <- curve_part
    cc_idx <- cc_idx + 1L
  }
  curve_pf_part <- build_calibration_curve_per_fold_from_stubs(res$recipe_jobs, CALIBRATION_CURVE_TARGET_RISKS)
  if (nrow(curve_pf_part) > 0L) {
    calibration_curve_per_fold_chunks[[cc_pf_idx]] <- curve_pf_part
    cc_pf_idx <- cc_pf_idx + 1L
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
rejection_stratum_per_fold_out <- if (length(all_rejection_stratum_per_fold) == 0L) {
  data.frame()
} else {
  bind_rows(all_rejection_stratum_per_fold)
}
rejection_stratum_summary_out <- summarize_rejection_stratum(rejection_stratum_per_fold_out)
rejection_stratum_loso_avg_out <- summarize_rejection_stratum_loso_labels_averaged(rejection_stratum_summary_out)
rejection_stratum_pooled_out <- summarize_rejection_stratum_pooled(rejection_stratum_per_fold_out)
summary_out <- if (length(all_summary) == 0L) data.frame() else bind_rows(all_summary)
per_fold_max_prob_out <- if (length(all_max_prob_per_fold) == 0L) {
  data.frame()
} else {
  bind_rows(all_max_prob_per_fold)
}
summary_max_prob_out <- if (length(all_summary_max_prob) == 0L) {
  data.frame()
} else {
  bind_rows(all_summary_max_prob)
}
summarize_full_coverage_baseline <- function(per_fold_df, baseline_kind) {
  if (nrow(per_fold_df) == 0L) return(data.frame())
  per_fold_df %>%
    mutate(setting_col = setting_column_label(split_type, label_set)) %>%
    group_by(label_set, split_type, setting_col) %>%
    summarise(
      baseline_kind = baseline_kind,
      n_outer_folds = n(),
      mean_outer_risk_all_accepted = mean(outer_risk_all_accepted, na.rm = TRUE),
      sd_outer_risk_all_accepted = stats::sd(outer_risk_all_accepted, na.rm = TRUE),
      mean_outer_coverage_seen = mean(outer_coverage_seen, na.rm = TRUE),
      mean_outer_kappa_accepted = mean(outer_kappa_accepted, na.rm = TRUE),
      sd_outer_kappa_accepted = stats::sd(outer_kappa_accepted, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(label_set, split_type)
}

full_coverage_per_fold_out <- if (length(full_coverage_per_fold_rows) == 0L) {
  data.frame()
} else {
  bind_rows(full_coverage_per_fold_rows)
}
classifier_only_per_fold_out <- if (length(classifier_only_per_fold_rows) == 0L) {
  data.frame()
} else {
  bind_rows(classifier_only_per_fold_rows)
}
full_coverage_summary_out <- dplyr::bind_rows(
  summarize_full_coverage_baseline(full_coverage_per_fold_out, "multivariate_accept_all"),
  summarize_full_coverage_baseline(classifier_only_per_fold_out, "classifier_only")
)

summary_combined_out <- if (nrow(summary_out) == 0L && nrow(summary_max_prob_out) == 0L) {
  data.frame()
} else {
  bind_rows(
    if (nrow(summary_out) > 0L) {
      summary_out %>% mutate(calibration_recipe = CALIBRATION_RECIPE_INNER_BEST)
    },
    if (nrow(summary_max_prob_out) > 0L) {
      summary_max_prob_out %>% mutate(calibration_recipe = CALIBRATION_RECIPE_MAX_PROB)
    }
  ) %>%
    arrange(label_set, split_type, calibration_recipe)
}
heatmap_out <- if (length(all_heat) == 0L) data.frame() else bind_rows(all_heat)
inner_scores_out <- if (length(all_inner_ranked) == 0L) data.frame() else bind_rows(all_inner_ranked)
calibration_curve_out <- if (length(calibration_curve_chunks) == 0L) {
  data.frame()
} else {
  bind_rows(calibration_curve_chunks) %>% arrange(label_set, split_type, requested_target_risk_pct)
}
calibration_curve_per_fold_out <- if (length(calibration_curve_per_fold_chunks) == 0L) {
  data.frame()
} else {
  bind_rows(calibration_curve_per_fold_chunks) %>%
    arrange(label_set, split_type, target_fold, requested_target_risk_pct)
}
calibration_compare_out <- if (length(calibration_compare_chunks) == 0L) {
  data.frame()
} else {
  bind_rows(calibration_compare_chunks) %>%
    arrange(label_set, split_type, calibration_recipe, requested_target_risk_pct)
}
if (nrow(calibration_curve_out) > 0L) {
  cat("Writing nested_target_risk_calibration_curve.csv ...\n")
  write_csv(calibration_curve_out, file.path(OUTPUT_DIR, "nested_target_risk_calibration_curve.csv"))
}
if (nrow(calibration_curve_per_fold_out) > 0L) {
  cat("Writing nested_target_risk_calibration_curve_per_fold.csv ...\n")
  write_csv(
    calibration_curve_per_fold_out,
    file.path(OUTPUT_DIR, "nested_target_risk_calibration_curve_per_fold.csv")
  )
}
if (nrow(calibration_compare_out) > 0L) {
  cat("Writing nested_target_risk_calibration_compare.csv ...\n")
  write_csv(
    calibration_compare_out,
    file.path(OUTPUT_DIR, "nested_target_risk_calibration_compare.csv")
  )
}
if (nrow(full_coverage_per_fold_out) > 0L) {
  cat("Writing nested_target_risk_full_coverage_per_fold.csv ...\n")
  write_csv(
    full_coverage_per_fold_out,
    file.path(OUTPUT_DIR, "nested_target_risk_full_coverage_per_fold.csv")
  )
}
if (nrow(full_coverage_summary_out) > 0L) {
  cat("Writing nested_target_risk_full_coverage_summary.csv ...\n")
  write_csv(
    full_coverage_summary_out,
    file.path(OUTPUT_DIR, "nested_target_risk_full_coverage_summary.csv")
  )
}
if (nrow(rejection_stratum_per_fold_out) > 0L) {
  cat("Writing nested_target_risk_rejection_stratum_per_fold.csv ...\n")
  write_csv(
    rejection_stratum_per_fold_out,
    file.path(OUTPUT_DIR, "nested_target_risk_rejection_stratum_per_fold.csv")
  )
}
if (nrow(rejection_stratum_summary_out) > 0L) {
  cat("Writing nested_target_risk_rejection_stratum_summary.csv ...\n")
  write_csv(
    rejection_stratum_summary_out,
    file.path(OUTPUT_DIR, "nested_target_risk_rejection_stratum_summary.csv")
  )
}
if (nrow(rejection_stratum_loso_avg_out) > 0L) {
  cat("Writing nested_target_risk_rejection_stratum_loso_labels_averaged.csv ...\n")
  write_csv(
    rejection_stratum_loso_avg_out,
    file.path(OUTPUT_DIR, "nested_target_risk_rejection_stratum_loso_labels_averaged.csv")
  )
}
if (nrow(rejection_stratum_pooled_out) > 0L) {
  cat("Writing nested_target_risk_rejection_stratum_pooled.csv ...\n")
  write_csv(
    rejection_stratum_pooled_out,
    file.path(OUTPUT_DIR, "nested_target_risk_rejection_stratum_pooled.csv")
  )
}
if (nrow(per_fold_out) > 0L) {
  cat("Writing nested_target_risk_per_fold.csv ...\n")
  write_csv(per_fold_out, file.path(OUTPUT_DIR, "nested_target_risk_per_fold.csv"))
}
if (nrow(summary_out) > 0L) {
  cat("Writing nested_target_risk_summary_four_settings.csv ...\n")
  write_csv(summary_out, file.path(OUTPUT_DIR, "nested_target_risk_summary_four_settings.csv"))
}
if (nrow(summary_max_prob_out) > 0L) {
  cat("Writing nested_target_risk_summary_max_prob.csv ...\n")
  write_csv(summary_max_prob_out, file.path(OUTPUT_DIR, "nested_target_risk_summary_max_prob.csv"))
}
if (nrow(per_fold_max_prob_out) > 0L) {
  cat("Writing nested_target_risk_per_fold_max_prob.csv ...\n")
  write_csv(per_fold_max_prob_out, file.path(OUTPUT_DIR, "nested_target_risk_per_fold_max_prob.csv"))
}
if (nrow(summary_combined_out) > 0L) {
  cat("Writing nested_target_risk_summary_combined.csv ...\n")
  write_csv(summary_combined_out, file.path(OUTPUT_DIR, "nested_target_risk_summary_combined.csv"))
}
if (nrow(heatmap_out) > 0L) {
  cat("Writing nested_target_risk_feature_heatmap_long.csv ...\n")
  write_csv(heatmap_out, file.path(OUTPUT_DIR, "nested_target_risk_feature_heatmap_long.csv"))
}

if (nrow(inner_scores_out) > 0L) {
  cat("Writing nested_target_risk_inner_scores_ranked.csv ...\n")
  write_csv(inner_scores_out, file.path(OUTPUT_DIR, "nested_target_risk_inner_scores_ranked.csv"))
}

manifest <- data.frame(
  key = c(
    "timestamp_utc", "output_dir", "inputs", "base_model", "scenarios", "baseline_terms", "all_feature_terms",
    "selection_metric", "selection_strategy", "canonical_family_order",     "per_fold_csv", "summary_four_csv", "summary_max_prob_csv", "summary_combined_csv",
    "per_fold_max_prob_csv",
    "heatmap_long_csv", "inner_scores_ranked_csv", "calibration_curve_csv",
    "calibration_curve_per_fold_csv",
    "calibration_compare_csv",
    "full_coverage_per_fold_csv",
    "full_coverage_summary_csv",
    "rejection_stratum_per_fold_csv",
    "rejection_stratum_summary_csv",
    "rejection_stratum_loso_labels_averaged_csv",
    "rejection_stratum_pooled_csv",
    "risk_targets_primary_outputs_fraction",
    "risk_targets_calibration_curve_fraction",
    "calibration_curve_inner_recipe_policy",
    "calibration_compare_recipes",
    "parallel_mc_cores", "parallel_outer_folds_backend"
  ),
  value = c(
    format(Sys.time(), tz = "UTC", usetz = TRUE),
    OUTPUT_DIR,
    paste(ANALYSIS_INPUTS$results_path, collapse = "; "),
    TARGET_BASE_MODEL,
    SCENARIO_KEY,
    paste(BASELINE_TERMS, collapse = ", "),
    paste(ALL_FEATURE_TERMS, collapse = ", "),
    sprintf(
      "rank_inner_scores at anchors %s (fraction); fusion=min sum(inner_rank)",
      paste(INNER_SELECTION_ANCHOR_RISKS, collapse = ",")
    ),
    "triple_inner_cv_per_rhs;_fuse_sum_ranks;_tie_maxrank_p05_p03_p10_rhskey;_outer_at_primary_risk",
    SCENARIO_KEY,
    file.path(OUTPUT_DIR, "nested_target_risk_per_fold.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_summary_four_settings.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_summary_max_prob.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_summary_combined.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_per_fold_max_prob.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_feature_heatmap_long.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_inner_scores_ranked.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_calibration_curve.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_calibration_curve_per_fold.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_calibration_compare.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_full_coverage_per_fold.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_full_coverage_summary.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_rejection_stratum_per_fold.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_rejection_stratum_summary.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_rejection_stratum_loso_labels_averaged.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_rejection_stratum_pooled.csv"),
    as.character(PRIMARY_TABLE_TARGET_RISK),
    paste(as.character(CALIBRATION_CURVE_TARGET_RISKS), collapse = ", "),
    "fixed_rhs_from_primary_target;_outer_threshold_refit_only",
    paste(CALIBRATION_RECIPE_INNER_BEST, CALIBRATION_RECIPE_MAX_PROB, sep = ";"),
    as.character(PARALLEL_MC_CORES),
    if (.Platform$OS.type == "unix") "parallel::mclapply fork" else "sequential"
  ),
  stringsAsFactors = FALSE
)
write_csv(manifest, file.path(OUTPUT_DIR, "nested_target_risk_manifest.csv"))

cat("Nested target-risk calibration analysis complete.\n")
cat(sprintf("Output directory: %s\n", OUTPUT_DIR))

