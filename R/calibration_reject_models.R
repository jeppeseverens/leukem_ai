# =============================================================================
# Selection-Safe Calibration Feature Utility Analysis (Model Comparison)
# =============================================================================
# SVM + DNN: single-head max-prob and single-head ridge (in-model features only).
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
  "merged_summed",      "collapsed_classes",   "data/out/outer_cv/outer_cv_analysis_outputs_merged_summed/outer_cv_results.rds",
  "merged_maxprob",     "collapsed_maxprob",   "data/out/outer_cv/outer_cv_analysis_outputs_merged_maxprob/outer_cv_results.rds"
)

repo_root <- if (file.exists("R/outer_cv_analysis.R")) "." else if (file.exists("outer_cv_analysis.R")) ".." else "."
ANALYSIS_INPUTS <- ANALYSIS_INPUTS %>% mutate(results_path = file.path(repo_root, results_rel_path))

OUTPUT_DIR <- file.path(repo_root, "data/out/outer_cv/calibration_feature_utility_selection_safe")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

INNER_RANK_TARGET_RISK_BY_LABEL_SET <- c(
  full_subtypes = 0.075,
  collapsed_classes = 0.05,
  collapsed_maxprob = 0.05
)
OUTER_OPERATING_TARGET_RISKS <- c(0.05, 0.075, 0.10)
PRIMARY_TABLE_TARGET_RISK <- INNER_RANK_TARGET_RISK_BY_LABEL_SET[["collapsed_classes"]]
CURVE_TARGET_RISK_PCT_GRID <- seq(10L, 150L, by = 1L) / 1000
CALIBRATION_PLOT_TARGET_RISKS <- CURVE_TARGET_RISK_PCT_GRID
MARCE_CURVE_TARGET_RISKS <- CURVE_TARGET_RISK_PCT_GRID
TARGET_RISK <- PRIMARY_TABLE_TARGET_RISK

PARALLEL_MC_CORES <- suppressWarnings(as.integer(Sys.getenv("CALIBRATION_REJECT_MC_CORES", unset = NA_integer_)))
if (length(PARALLEL_MC_CORES) != 1L || is.na(PARALLEL_MC_CORES) || PARALLEL_MC_CORES < 1L) {
  PARALLEL_MC_CORES <- 4L
}
if (.Platform$OS.type != "unix") {
  PARALLEL_MC_CORES <- 1L
}
DEFAULT_CACHE_MC_CORES <- 3L
OUTER_SPLIT_TYPES <- c("loso")

BASELINE_TERMS <- c("max_prob")
BASELINE_ONLY_RHS_KEY <- "max_prob"
IN_MODEL_FEATURE_TERMS <- c("max_prob", "margin", "entropy", "conformal_set_size_90")
KNN10_FEATURE_TERMS <- c("knn10_mean_d", "knn10_min_d", "knn10_q90_d")
IN_MODEL_KNN10_FEATURE_TERMS <- c(IN_MODEL_FEATURE_TERMS, KNN10_FEATURE_TERMS)
ALL_FEATURE_TERMS <- IN_MODEL_KNN10_FEATURE_TERMS
NESTED_MAXPROB_ONLY <- TRUE
INNER_SELECTION_METHOD <- "maxprob"
ENET_ALPHA_GRID <- c(0)
CLASS_BALANCED_OOD <- TRUE

TWO_HEAD_EXPORT_SPECS <- tibble::tribble(
  ~rejector_key,        ~rejector_mode,       ~rejector_label,                                      ~two_head_combine,
  "two_head_min",       "two_head_min",       "Two-head min(P(correct|ID), P(ID))",               "min"
)

parse_feature_terms <- function(feature_terms_key) {
  if (is.na(feature_terms_key) || !nzchar(feature_terms_key)) {
    IN_MODEL_KNN10_FEATURE_TERMS
  } else {
    strsplit(feature_terms_key, ";", fixed = TRUE)[[1]]
  }
}

# One row per base-model rejector recipe (single-head max-prob, single-head ridge in-model).
RIDGE_IN_MODEL_FEATURE_TERMS_KEY <- paste(IN_MODEL_FEATURE_TERMS, collapse = ";")

rejector_recipe_rows <- function(ensemble_key, ensemble_label, base_model) {
  tibble::tribble(
    ~recipe_key,      ~rejector_method, ~feature_terms_key,                              ~recipe_rejector_key, ~recipe_label_single,
    "maxprob",        "maxprob",        NA_character_,                                   "single_head",        "Single-head max-prob",
    "ridge_in_model", "ridge",          RIDGE_IN_MODEL_FEATURE_TERMS_KEY,              "ridge_in_model",     "Single-head ridge (in-model)"
  ) %>%
    mutate(
      ensemble_key = ensemble_key,
      ensemble_label = ensemble_label,
      base_model = base_model,
      feature_terms = lapply(feature_terms_key, parse_feature_terms)
    )
}

# Fail fast if job grid drifts from the intended SVM/DNN single-head recipes.
validate_nested_jobs_config <- function(jobs) {
  expected_recipes <- c("maxprob", "ridge_in_model")
  expected_ensembles <- c("svm", "dnn")
  expected_base_models <- c("svm", "neural_net")
  bad_recipe <- setdiff(unique(jobs$recipe_key), expected_recipes)
  if (length(bad_recipe) > 0L) {
    stop(sprintf("Unexpected recipe_key values: %s", paste(bad_recipe, collapse = ", ")))
  }
  bad_mode <- setdiff(unique(jobs$nested_rejector_mode), "single_head")
  if (length(bad_mode) > 0L) {
    stop(sprintf("Only single_head nested runs are supported (found: %s).", paste(bad_mode, collapse = ", ")))
  }
  bad_ensemble <- setdiff(unique(jobs$ensemble_key), expected_ensembles)
  if (length(bad_ensemble) > 0L) {
    stop(sprintf("Unexpected ensemble_key values: %s", paste(bad_ensemble, collapse = ", ")))
  }
  bad_base <- setdiff(unique(jobs$base_model), expected_base_models)
  if (length(bad_base) > 0L) {
    stop(sprintf("Unexpected base_model values: %s", paste(bad_base, collapse = ", ")))
  }
  ridge_rows <- jobs %>% filter(.data$recipe_key == "ridge_in_model")
  bad_ridge_terms <- ridge_rows %>%
    filter(.data$feature_terms_key != RIDGE_IN_MODEL_FEATURE_TERMS_KEY)
  if (nrow(bad_ridge_terms) > 0L) {
    stop("ridge_in_model jobs must use in-model feature terms only (no KNN10).")
  }
  maxprob_rows <- jobs %>% filter(.data$recipe_key == "maxprob")
  bad_maxprob_terms <- maxprob_rows %>% filter(!is.na(.data$feature_terms_key))
  if (nrow(bad_maxprob_terms) > 0L) {
    stop("maxprob jobs must not specify feature_terms_key.")
  }
  expected_n <- length(expected_ensembles) * length(expected_recipes)
  if (nrow(jobs) != expected_n) {
    stop(sprintf("Expected %d nested jobs, got %d.", expected_n, nrow(jobs)))
  }
  invisible(jobs)
}

export_config_key <- function(ensemble_key, recipe_rejector_key, export_rejector_key) {
  if (identical(recipe_rejector_key, "single_head") && identical(export_rejector_key, "single_head")) {
    paste(ensemble_key, "single_head", sep = "_")
  } else if (identical(recipe_rejector_key, "single_head")) {
    paste(ensemble_key, export_rejector_key, sep = "_")
  } else if (identical(export_rejector_key, "single_head")) {
    paste(ensemble_key, recipe_rejector_key, sep = "_")
  } else {
    paste(ensemble_key, recipe_rejector_key, export_rejector_key, sep = "_")
  }
}

export_config_label <- function(ensemble_label, recipe_label_single, export_spec) {
  if (identical(export_spec$rejector_key, "single_head")) {
    paste(ensemble_label, recipe_label_single, sep = " | ")
  } else {
    paste(ensemble_label, gsub("^Single-head ", "", recipe_label_single), export_spec$rejector_label, sep = " | ")
  }
}

export_specs_for_nested_job <- function(job_row) {
  recipe_rejector_key <- job_row$recipe_rejector_key[[1]]
  if (identical(job_row$nested_rejector_mode[[1]], "single_head")) {
    list(tibble::tibble(
      rejector_key = "single_head",
      rejector_mode = "single_head",
      rejector_label = job_row$recipe_label_single[[1]],
      two_head_combine = NA_character_,
      config_key = export_config_key(job_row$ensemble_key[[1]], recipe_rejector_key, "single_head"),
      config_label = export_config_label(
        job_row$ensemble_label[[1]], job_row$recipe_label_single[[1]],
        list(rejector_key = "single_head")
      )
    ))
  } else {
    lapply(seq_len(nrow(TWO_HEAD_EXPORT_SPECS)), function(i) {
      es <- TWO_HEAD_EXPORT_SPECS[i, , drop = FALSE]
      tibble::tibble(
        rejector_key = es$rejector_key,
        rejector_mode = es$rejector_mode,
        rejector_label = es$rejector_label,
        two_head_combine = es$two_head_combine,
        config_key = export_config_key(
          job_row$ensemble_key[[1]], recipe_rejector_key, es$rejector_key[[1]]
        ),
        config_label = export_config_label(job_row$ensemble_label[[1]], job_row$recipe_label_single[[1]], es)
      )
    })
  }
}

# Nested jobs: SVM + DNN recipes (single-head max-prob and ridge in-model only).
NESTED_JOBS <- bind_rows(
  rejector_recipe_rows("svm", "SVM", "svm"),
  rejector_recipe_rows("dnn", "DNN (neural net)", "neural_net")
) %>%
  crossing(nested_rejector_mode = c("single_head")) %>%
  mutate(
    feature_terms = lapply(feature_terms_key, parse_feature_terms),
    job_key = paste(ensemble_key, recipe_key, nested_rejector_mode, sep = "|")
  )
validate_nested_jobs_config(NESTED_JOBS)

# Flat export manifest for startup log / CSV manifest.
CALIBRATION_EXPORTS <- bind_rows(lapply(seq_len(nrow(NESTED_JOBS)), function(i) {
  job <- NESTED_JOBS[i, , drop = FALSE]
  bind_rows(export_specs_for_nested_job(job)) %>%
    mutate(
      ensemble_key = job$ensemble_key,
      ensemble_label = job$ensemble_label,
      base_model = job$base_model,
      recipe_key = job$recipe_key,
      rejector_method = job$rejector_method,
      feature_terms_key = job$feature_terms_key,
      nested_rejector_mode = job$nested_rejector_mode,
      job_key = job$job_key
    )
}))

format_risk_grid_label <- function(risks) {
  if (length(risks) < 2L) {
    sprintf("%.1f%%", 100 * risks[[1L]])
  } else {
    sprintf(
      "%.1f–%.1f%% (%.1f%% steps, %d points)",
      100 * min(risks), 100 * max(risks),
      100 * (risks[[2L]] - risks[[1L]]),
      length(risks)
    )
  }
}

apply_nested_job_settings <- function(job_row) {
  terms <- job_row$feature_terms[[1]]
  assign("ALL_FEATURE_TERMS", terms, envir = .GlobalEnv)
  if (identical(job_row$rejector_method[[1]], "ridge")) {
    assign("NESTED_MAXPROB_ONLY", FALSE, envir = .GlobalEnv)
    assign("INNER_SELECTION_METHOD", "elasticnet", envir = .GlobalEnv)
    assign("ENET_ALPHA_GRID", c(0), envir = .GlobalEnv)
  } else {
    assign("NESTED_MAXPROB_ONLY", TRUE, envir = .GlobalEnv)
    assign("INNER_SELECTION_METHOD", "maxprob", envir = .GlobalEnv)
  }
  invisible(terms)
}

nested_job_settings_label <- function(job_row) {
  head_lab <- if (identical(job_row$nested_rejector_mode[[1]], "two_head_min")) {
    "two-head nested (min)"
  } else {
    "single-head"
  }
  if (identical(job_row$rejector_method[[1]], "ridge")) {
    sprintf(
      "%s; ridge alpha=0 (%s; lambda.min study-blocked cv.glmnet)",
      head_lab, paste(job_row$feature_terms[[1]], collapse = ", ")
    )
  } else {
    sprintf("%s; max-prob logistic (%s)", head_lab, BASELINE_ONLY_RHS_KEY)
  }
}

SCENARIO_KEY <- "with_leftout_ood_aware"
POOL_RULE <- "all_rows"
TEST_RULE <- "all_rows"
FULL_COVERAGE_THRESHOLD <- 0

cat("Starting nested target-risk calibration analysis...\n")
cat(sprintf("  Scenario: %s, outer splits: %s, label sets: %d\n", SCENARIO_KEY, paste(OUTER_SPLIT_TYPES, collapse = "/"), nrow(ANALYSIS_INPUTS)))
cat(sprintf(
  "  Nested jobs: %d (exports: %d config keys)\n",
  nrow(NESTED_JOBS), nrow(CALIBRATION_EXPORTS)
))
cat(sprintf("  Export config keys: %s\n", paste(CALIBRATION_EXPORTS$config_key, collapse = ", ")))
source(if (file.exists("R/calibration_reject_core.R")) {
  "R/calibration_reject_core.R"
} else {
  "calibration_reject_core.R"
})
cat(sprintf(
  "  Two-head OOD head class weights: %s\n",
  class_balanced_ood_label(CLASS_BALANCED_OOD)
))
cat(sprintf(
  "  Inner-report risks (sample decisions / primary per-fold): %s\n",
  paste(
    sprintf("%s=%.1f%%", names(INNER_RANK_TARGET_RISK_BY_LABEL_SET), 100 * INNER_RANK_TARGET_RISK_BY_LABEL_SET),
    collapse = ", "
  )
))
cat(sprintf(
  "  Outer operating points: %s%%\n",
  paste(format(100 * OUTER_OPERATING_TARGET_RISKS, trim = TRUE), collapse = ", ")
))
cat(sprintf(
  "  Calibration / MARCE grid: %s (score once per fold; all outer folds required per point)\n",
  format_risk_grid_label(CALIBRATION_PLOT_TARGET_RISKS)
))
cat(sprintf("  Parallel outer folds: mc.cores=%d\n", PARALLEL_MC_CORES))
cat(sprintf("  Output directory: %s\n", OUTPUT_DIR))
pin_blas_threads(1L)

CACHE_PARALLEL_MC_CORES <- cache_parallel_mc_cores(1L)
cat(sprintf(
  "  Outer eval cache parallelism: mc.cores=%d (default %d; set CALIBRATION_REJECT_CACHE_MC_CORES=1 for sequential / low memory)\n",
  CACHE_PARALLEL_MC_CORES, DEFAULT_CACHE_MC_CORES
))

ALL_EVAL_CACHE_RISKS <- unique(c(
  CALIBRATION_PLOT_TARGET_RISKS,
  OUTER_OPERATING_TARGET_RISKS
))

tag_calibration_run_export <- function(df, job_row, export_spec) {
  if (nrow(df) == 0L) return(df)
  df %>%
    mutate(
      ensemble_key = job_row$ensemble_key,
      ensemble_label = job_row$ensemble_label,
      rejector_key = export_spec$rejector_key,
      rejector_label = export_spec$rejector_label,
      rejector_mode = export_spec$rejector_mode,
      rejector_method = job_row$rejector_method,
      config_key = export_spec$config_key,
      config_label = export_spec$config_label,
      ensemble_base = job_row$base_model,
      feature_terms_key = job_row$feature_terms_key,
      recipe_key = job_row$recipe_key,
      nested_rejector_mode = job_row$nested_rejector_mode,
      job_key = job_row$job_key
    )
}

# Per outer-fold inner winners + cross-fold summary for inner CV log-loss / AUROC comparison.
build_inner_logloss_tables <- function(inner_scores_df) {
  empty_summary <- data.frame(
    job_key = character(),
    config_key = character(),
    config_label = character(),
    ensemble_key = character(),
    ensemble_label = character(),
    recipe_key = character(),
    rejector_method = character(),
    nested_rejector_mode = character(),
    label_set = character(),
    split_type = character(),
    n_outer_folds = integer(),
    mean_inner_logloss = numeric(),
    sd_inner_logloss = numeric(),
    mean_inner_sd_logloss = numeric(),
    mean_inner_auroc = numeric(),
    sd_inner_auroc = numeric(),
    mean_inner_sd_auroc = numeric(),
    inner_winner_alpha = numeric(),
    rank_mean_logloss = integer(),
    rank_mean_auroc = integer(),
    stringsAsFactors = FALSE
  )
  empty_per_fold <- data.frame(
    job_key = character(),
    config_key = character(),
    config_label = character(),
    ensemble_key = character(),
    recipe_key = character(),
    rejector_method = character(),
    nested_rejector_mode = character(),
    label_set = character(),
    split_type = character(),
    target_fold = character(),
    mean_logloss = numeric(),
    sd_logloss = numeric(),
    mean_auroc = numeric(),
    sd_auroc = numeric(),
    alpha = numeric(),
    feature_terms_key = character(),
    stringsAsFactors = FALSE
  )
  if (nrow(inner_scores_df) == 0L) {
    return(list(summary = empty_summary, per_fold = empty_per_fold))
  }
  if (!all(c("inner_rank", "mean_logloss", "mean_auroc") %in% names(inner_scores_df))) {
    stop("inner_scores_df missing inner_rank, mean_logloss, or mean_auroc columns.")
  }

  per_fold <- inner_scores_df %>%
    filter(.data$inner_rank == 1L, is.finite(.data$mean_logloss), is.finite(.data$mean_auroc)) %>%
    transmute(
      job_key = if ("job_key" %in% names(.)) as.character(.data$job_key) else NA_character_,
      config_key = as.character(.data$config_key),
      config_label = if ("config_label" %in% names(.)) as.character(.data$config_label) else NA_character_,
      ensemble_key = as.character(.data$ensemble_key),
      recipe_key = if ("recipe_key" %in% names(.)) as.character(.data$recipe_key) else NA_character_,
      rejector_method = as.character(.data$rejector_method),
      nested_rejector_mode = as.character(.data$nested_rejector_mode),
      label_set = as.character(.data$label_set),
      split_type = as.character(.data$split_type),
      target_fold = as.character(.data$target_fold),
      mean_logloss = as.numeric(.data$mean_logloss),
      sd_logloss = if ("sd_logloss" %in% names(.)) as.numeric(.data$sd_logloss) else NA_real_,
      mean_auroc = as.numeric(.data$mean_auroc),
      sd_auroc = if ("sd_auroc" %in% names(.)) as.numeric(.data$sd_auroc) else NA_real_,
      alpha = if ("alpha" %in% names(.)) as.numeric(.data$alpha) else NA_real_,
      feature_terms_key = if ("feature_terms_key" %in% names(.)) as.character(.data$feature_terms_key) else NA_character_
    )

  if (nrow(per_fold) == 0L) {
    return(list(summary = empty_summary, per_fold = empty_per_fold))
  }

  summary <- per_fold %>%
    group_by(
      job_key, config_key, config_label, ensemble_key, recipe_key,
      rejector_method, nested_rejector_mode, label_set, split_type
    ) %>%
    summarise(
      n_outer_folds = dplyr::n(),
      mean_inner_logloss = mean(.data$mean_logloss, na.rm = TRUE),
      sd_inner_logloss = stats::sd(.data$mean_logloss, na.rm = TRUE),
      mean_inner_sd_logloss = mean(.data$sd_logloss, na.rm = TRUE),
      mean_inner_auroc = mean(.data$mean_auroc, na.rm = TRUE),
      sd_inner_auroc = stats::sd(.data$mean_auroc, na.rm = TRUE),
      mean_inner_sd_auroc = mean(.data$sd_auroc, na.rm = TRUE),
      inner_winner_alpha = stats::median(.data$alpha, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    group_by(.data$label_set, .data$split_type, .data$nested_rejector_mode) %>%
    arrange(.data$mean_inner_logloss, .data$config_key) %>%
    mutate(rank_mean_logloss = dplyr::row_number()) %>%
    ungroup() %>%
    group_by(.data$label_set, .data$split_type, .data$nested_rejector_mode) %>%
    arrange(dplyr::desc(.data$mean_inner_auroc), .data$config_key) %>%
    mutate(rank_mean_auroc = dplyr::row_number()) %>%
    ungroup() %>%
    arrange(.data$label_set, .data$split_type, .data$nested_rejector_mode, .data$rank_mean_logloss)

  list(summary = summary, per_fold = per_fold %>% arrange(.data$config_key, .data$label_set, .data$target_fold))
}

tag_classifier_only_run <- function(df, job_row) {
  if (nrow(df) == 0L) return(df)
  df %>%
    mutate(
      ensemble_key = job_row$ensemble_key,
      ensemble_label = job_row$ensemble_label,
      ensemble_base = job_row$base_model
    )
}

append_export_artifacts <- function(
    export_spec, job_row, res, label_inner_risk,
    op_idx, sum_idx, cc_idx, cc_pf_idx, fc_idx, marce_idx, prob_idx, sd_idx,
    all_per_fold_operating, all_summary_operating, all_calibration_curve,
    all_calibration_curve_per_fold, all_calibration_fold_coverage, all_marce,
    all_probability_samples, all_sample_decisions) {
  tag_fn <- function(df) tag_calibration_run_export(df, job_row, export_spec)
  combine <- export_spec$two_head_combine[[1]]
  if (is.na(combine)) combine <- NULL
  is_two_head_export <- !is.null(combine)
  n_loso_folds_expected <- length(res$recipe_jobs)
  export_rejector_mode <- export_spec$rejector_mode[[1]]

  eval_cache_df <- build_outer_eval_cache(
    res$recipe_jobs, ALL_EVAL_CACHE_RISKS, THRESHOLD_METHODS,
    seed_operating_df = if (identical(combine, "product")) res$per_fold_operating_df else NULL,
    two_head_combine = combine
  )
  pf_operating <- if (is_two_head_export) {
    build_operating_from_eval_cache(
      eval_cache_df, OUTER_OPERATING_TARGET_RISKS, res$recipe_jobs, export_rejector_mode
    )
  } else {
    res$per_fold_operating_df
  }
  if (nrow(pf_operating) > 0L) {
    all_per_fold_operating[[op_idx]] <- tag_fn(pf_operating)
    op_idx <- op_idx + 1L
    summary_part <- tag_fn(
      summarize_operating_points(pf_operating, n_outer_folds_expected = n_loso_folds_expected)
    )
    if (nrow(summary_part) > 0L) {
      all_summary_operating[[sum_idx]] <- summary_part
      sum_idx <- sum_idx + 1L
    }
  }

  curve_raw <- build_calibration_curve_from_eval_cache(
    eval_cache_df, CALIBRATION_PLOT_TARGET_RISKS,
    fold_completeness = "per_risk_point",
    n_outer_folds_expected = n_loso_folds_expected
  )
  curve_part <- tag_fn(curve_raw)
  if (nrow(curve_part) > 0L) {
    all_calibration_curve[[cc_idx]] <- curve_part
    cc_idx <- cc_idx + 1L
  }
  marce_part <- tag_fn(compute_marce_from_curve(
    curve_raw, n_loso_folds_expected = n_loso_folds_expected
  ))
  if (nrow(marce_part) > 0L) {
    all_marce[[marce_idx]] <- marce_part
    marce_idx <- marce_idx + 1L
  }
  curve_pf_part <- tag_fn(
    build_calibration_curve_per_fold_from_eval_cache(
      eval_cache_df, CALIBRATION_PLOT_TARGET_RISKS,
      fold_completeness = "per_risk_point",
      n_outer_folds_expected = n_loso_folds_expected
    )
  )
  if (nrow(curve_pf_part) > 0L) {
    all_calibration_curve_per_fold[[cc_pf_idx]] <- curve_pf_part
    cc_pf_idx <- cc_pf_idx + 1L
  }
  fold_cov_part <- tag_fn(
    build_calibration_fold_coverage_from_eval_cache(eval_cache_df, CALIBRATION_PLOT_TARGET_RISKS)
  )
  if (nrow(fold_cov_part) > 0L) {
    all_calibration_fold_coverage[[fc_idx]] <- fold_cov_part
    fc_idx <- fc_idx + 1L
  }
  prob_part <- tag_fn(
    build_probability_samples_from_stubs(res$recipe_jobs, two_head_combine = combine)
  )
  if (nrow(prob_part) > 0L) {
    all_probability_samples[[prob_idx]] <- prob_part
    prob_idx <- prob_idx + 1L
  }
  dec_part <- tag_fn(
    build_sample_decisions_from_eval_cache(
      res$recipe_jobs, label_inner_risk, eval_cache_df, two_head_combine = combine
    )
  )
  if (nrow(dec_part) > 0L) {
    all_sample_decisions[[sd_idx]] <- dec_part
    sd_idx <- sd_idx + 1L
  }

  list(
    op_idx = op_idx, sum_idx = sum_idx, cc_idx = cc_idx, cc_pf_idx = cc_pf_idx,
    fc_idx = fc_idx, marce_idx = marce_idx, prob_idx = prob_idx, sd_idx = sd_idx,
    all_per_fold_operating = all_per_fold_operating,
    all_summary_operating = all_summary_operating,
    all_calibration_curve = all_calibration_curve,
    all_calibration_curve_per_fold = all_calibration_curve_per_fold,
    all_calibration_fold_coverage = all_calibration_fold_coverage,
    all_marce = all_marce,
    all_probability_samples = all_probability_samples,
    all_sample_decisions = all_sample_decisions
  )
}

all_per_fold_operating <- list()
all_summary_operating <- list()
all_calibration_curve <- list()
all_calibration_curve_per_fold <- list()
all_calibration_fold_coverage <- list()
all_marce <- list()
all_probability_samples <- list()
all_inner_ranked <- list()
all_inner_winner_coef <- list()
all_classifier_only_per_fold <- list()
all_sample_decisions <- list()
op_idx <- 1L
sum_idx <- 1L
cc_idx <- 1L
cc_pf_idx <- 1L
fc_idx <- 1L
marce_idx <- 1L
prob_idx <- 1L
in_idx <- 1L
coef_idx <- 1L
co_idx <- 1L
sd_idx <- 1L

for (jr in seq_len(nrow(NESTED_JOBS))) {
  job_row <- NESTED_JOBS[jr, ]
  TARGET_BASE_MODEL <- job_row$base_model
  TARGET_BASE_MODELS <- TARGET_BASE_MODEL
  apply_nested_job_settings(job_row)

  cat(sprintf(
    "\n=== [%s] %s | %s ===\n  %s\n",
    job_row$job_key, job_row$ensemble_label, job_row$recipe_key, nested_job_settings_label(job_row)
  ))

  for (i in seq_len(nrow(ANALYSIS_INPUTS))) {
    row <- ANALYSIS_INPUTS[i, ]
    if (!file.exists(row$results_path)) {
      stop(sprintf("Missing required results file: %s", row$results_path))
    }
    label_inner_risk <- INNER_RANK_TARGET_RISK_BY_LABEL_SET[[row$label_set]]
    if (length(label_inner_risk) != 1L || !is.finite(label_inner_risk)) {
      stop(sprintf("Missing inner ranking target risk for label_set=%s", row$label_set))
    }
    PRIMARY_TABLE_TARGET_RISK <- label_inner_risk
    TARGET_RISK <- label_inner_risk

    cat(sprintf("Loading %s ...\n", row$results_path))
    obj <- readRDS(row$results_path)
    cat(sprintf(
      "  [%s | %s | report=%.1f%%] Nested LOSO ...\n",
      job_row$job_key, row$label_set, 100 * label_inner_risk
    ))
    res <- run_nested_target_risk_analysis(
      obj, row$label_set, risk_target = label_inner_risk,
      rejector_mode = job_row$nested_rejector_mode
    )
    cat(sprintf(
      "    operating_rows=%d summary_operating=%d inner-ranked=%d recipe_stubs=%d\n",
      nrow(res$per_fold_operating_df),
      nrow(res$summary_operating),
      nrow(res$inner_scores_ranked),
      length(res$recipe_jobs)
    ))

    export_specs <- export_specs_for_nested_job(job_row)
    inner_export_spec <- if (identical(job_row$nested_rejector_mode[[1]], "two_head_min")) {
      dplyr::bind_rows(export_specs) %>%
        filter(.data$two_head_combine == "min") %>%
        slice(1)
    } else {
      export_specs[[1L]]
    }
    if (nrow(res$inner_scores_ranked) > 0L) {
      all_inner_ranked[[in_idx]] <- tag_calibration_run_export(
        res$inner_scores_ranked, job_row, inner_export_spec
      )
      in_idx <- in_idx + 1L
    }
    coef_part <- tag_calibration_run_export(
      build_inner_winner_outer_pool_coef_long(res$recipe_jobs), job_row, inner_export_spec
    )
    if (nrow(coef_part) > 0L) {
      all_inner_winner_coef[[coef_idx]] <- coef_part
      coef_idx <- coef_idx + 1L
    }

    for (es in export_specs) {
      if (identical(job_row$nested_rejector_mode[[1]], "two_head_min")) {
        cat(sprintf(
          "  Export two-head combine=%s -> %s\n",
          es$two_head_combine[[1]], es$config_key[[1]]
        ))
      }
      out <- append_export_artifacts(
        es, job_row, res, label_inner_risk,
        op_idx, sum_idx, cc_idx, cc_pf_idx, fc_idx, marce_idx, prob_idx, sd_idx,
        all_per_fold_operating, all_summary_operating, all_calibration_curve,
        all_calibration_curve_per_fold, all_calibration_fold_coverage, all_marce,
        all_probability_samples, all_sample_decisions
      )
      op_idx <- out$op_idx
      sum_idx <- out$sum_idx
      cc_idx <- out$cc_idx
      cc_pf_idx <- out$cc_pf_idx
      fc_idx <- out$fc_idx
      marce_idx <- out$marce_idx
      prob_idx <- out$prob_idx
      sd_idx <- out$sd_idx
      all_per_fold_operating <- out$all_per_fold_operating
      all_summary_operating <- out$all_summary_operating
      all_calibration_curve <- out$all_calibration_curve
      all_calibration_curve_per_fold <- out$all_calibration_curve_per_fold
      all_calibration_fold_coverage <- out$all_calibration_fold_coverage
      all_marce <- out$all_marce
      all_probability_samples <- out$all_probability_samples
      all_sample_decisions <- out$all_sample_decisions
    }

    if (identical(job_row$recipe_key[[1]], "maxprob") &&
        identical(job_row$nested_rejector_mode[[1]], "single_head")) {
      co_part <- tag_classifier_only_run(
        build_classifier_only_per_fold_from_stubs(res$recipe_jobs),
        job_row
      )
      if (nrow(co_part) > 0L) {
        all_classifier_only_per_fold[[co_idx]] <- co_part
        co_idx <- co_idx + 1L
      }
    }
  }
}

per_fold_operating_out <- if (length(all_per_fold_operating) == 0L) {
  data.frame()
} else {
  bind_rows(all_per_fold_operating)
}
summary_operating_out <- if (length(all_summary_operating) == 0L) {
  data.frame()
} else {
  bind_rows(all_summary_operating) %>%
    arrange(config_key, label_set, split_type, threshold_method, requested_target_risk_pct)
}
calibration_curve_out <- if (length(all_calibration_curve) == 0L) {
  data.frame()
} else {
  bind_rows(all_calibration_curve) %>%
    arrange(config_key, label_set, split_type, threshold_method, requested_target_risk_pct)
}
calibration_curve_per_fold_out <- if (length(all_calibration_curve_per_fold) == 0L) {
  data.frame()
} else {
  bind_rows(all_calibration_curve_per_fold)
}
calibration_fold_coverage_out <- if (length(all_calibration_fold_coverage) == 0L) {
  data.frame()
} else {
  bind_rows(all_calibration_fold_coverage) %>%
    arrange(config_key, label_set, split_type, threshold_method, requested_target_risk_pct)
}
marce_out <- if (length(all_marce) == 0L) {
  data.frame()
} else {
  bind_rows(all_marce) %>%
    arrange(config_key, label_set, split_type, threshold_method)
}
probability_samples_out <- if (length(all_probability_samples) == 0L) {
  data.frame()
} else {
  bind_rows(all_probability_samples)
}
inner_scores_out <- if (length(all_inner_ranked) == 0L) {
  data.frame()
} else {
  bind_rows(all_inner_ranked)
}
inner_logloss_tables <- build_inner_logloss_tables(inner_scores_out)
inner_logloss_summary_out <- inner_logloss_tables$summary
inner_logloss_per_fold_out <- inner_logloss_tables$per_fold
inner_winner_coef_out <- if (length(all_inner_winner_coef) == 0L) {
  data.frame()
} else {
  bind_rows(all_inner_winner_coef) %>%
    arrange(config_key, label_set, split_type, target_fold, head, feature)
}
classifier_only_per_fold_out <- if (length(all_classifier_only_per_fold) == 0L) {
  data.frame()
} else {
  bind_rows(all_classifier_only_per_fold) %>%
    arrange(ensemble_key, label_set, split_type, target_fold)
}
classifier_only_summary_out <- summarize_classifier_only_full_coverage(classifier_only_per_fold_out)
sample_decisions_out <- if (length(all_sample_decisions) == 0L) {
  data.frame()
} else {
  bind_rows(all_sample_decisions) %>%
    arrange(config_key, label_set, split_type, target_fold)
}

if (nrow(per_fold_operating_out) > 0L) {
  cat("Writing nested_target_risk_per_fold_operating.csv ...\n")
  write_csv(per_fold_operating_out, file.path(OUTPUT_DIR, "nested_target_risk_per_fold_operating.csv"))
}
if (nrow(summary_operating_out) > 0L) {
  cat("Writing nested_target_risk_summary_operating.csv ...\n")
  write_csv(summary_operating_out, file.path(OUTPUT_DIR, "nested_target_risk_summary_operating.csv"))
}
if (nrow(calibration_curve_out) > 0L) {
  cat("Writing nested_target_risk_calibration_curve.csv ...\n")
  write_csv(calibration_curve_out, file.path(OUTPUT_DIR, "nested_target_risk_calibration_curve.csv"))
}
if (nrow(calibration_curve_per_fold_out) > 0L) {
  cat("Writing nested_target_risk_calibration_curve_per_fold.csv ...\n")
  write_csv(calibration_curve_per_fold_out, file.path(OUTPUT_DIR, "nested_target_risk_calibration_curve_per_fold.csv"))
}
if (nrow(calibration_fold_coverage_out) > 0L) {
  cat("Writing nested_target_risk_calibration_fold_coverage.csv ...\n")
  write_csv(calibration_fold_coverage_out, file.path(OUTPUT_DIR, "nested_target_risk_calibration_fold_coverage.csv"))
}
if (nrow(marce_out) > 0L) {
  cat("Writing nested_target_risk_marce.csv ...\n")
  write_csv(marce_out, file.path(OUTPUT_DIR, "nested_target_risk_marce.csv"))
}
if (nrow(probability_samples_out) > 0L) {
  cat("Writing nested_target_risk_probability_samples.csv ...\n")
  write_csv(probability_samples_out, file.path(OUTPUT_DIR, "nested_target_risk_probability_samples.csv"))
}
if (nrow(inner_scores_out) > 0L) {
  cat("Writing nested_target_risk_inner_scores_ranked.csv ...\n")
  write_csv(inner_scores_out, file.path(OUTPUT_DIR, "nested_target_risk_inner_scores_ranked.csv"))
}
if (nrow(inner_logloss_summary_out) > 0L) {
  cat("Writing nested_target_risk_inner_logloss_summary.csv ...\n")
  write_csv(inner_logloss_summary_out, file.path(OUTPUT_DIR, "nested_target_risk_inner_logloss_summary.csv"))
}
if (nrow(inner_logloss_per_fold_out) > 0L) {
  cat("Writing nested_target_risk_inner_logloss_per_fold.csv ...\n")
  write_csv(inner_logloss_per_fold_out, file.path(OUTPUT_DIR, "nested_target_risk_inner_logloss_per_fold.csv"))
}
if (nrow(inner_winner_coef_out) > 0L) {
  cat("Writing nested_target_risk_inner_winner_outer_pool_coefs.csv ...\n")
  write_csv(inner_winner_coef_out, file.path(OUTPUT_DIR, "nested_target_risk_inner_winner_outer_pool_coefs.csv"))
}
if (nrow(classifier_only_per_fold_out) > 0L) {
  cat("Writing nested_target_risk_classifier_only_per_fold.csv ...\n")
  write_csv(classifier_only_per_fold_out, file.path(OUTPUT_DIR, "nested_target_risk_classifier_only_per_fold.csv"))
}
if (nrow(classifier_only_summary_out) > 0L) {
  cat("Writing nested_target_risk_classifier_only_summary.csv ...\n")
  write_csv(classifier_only_summary_out, file.path(OUTPUT_DIR, "nested_target_risk_classifier_only_summary.csv"))
}
if (nrow(sample_decisions_out) > 0L) {
  cat("Writing nested_target_risk_sample_decisions_inner_rank_risk.csv ...\n")
  write_csv(
    sample_decisions_out,
    file.path(OUTPUT_DIR, "nested_target_risk_sample_decisions_inner_rank_risk.csv")
  )
}

calibration_run_manifest <- CALIBRATION_EXPORTS %>%
  transmute(
    config_key,
    rejector_method,
    feature_terms_key = ifelse(is.na(feature_terms_key), "max_prob_only", feature_terms_key),
    nested_rejector_mode
  ) %>%
  mutate(entry = paste(config_key, rejector_method, feature_terms_key, nested_rejector_mode, sep = ":")) %>%
  pull(entry) %>%
  paste(collapse = ";")

manifest <- data.frame(
  key = c(
    "timestamp_utc", "output_dir", "inputs", "scenarios", "baseline_terms",
    "in_model_feature_terms", "knn10_feature_terms",
    "inner_rank_target_risks", "outer_operating_target_risks", "calibration_plot_target_risks", "marce_curve_target_risks", "threshold_methods",
    "outer_split_types", "nested_jobs", "calibration_exports", "class_balanced_ood",
    "per_fold_operating_csv", "summary_operating_csv",
    "calibration_curve_csv", "calibration_curve_per_fold_csv", "calibration_fold_coverage_csv", "marce_csv",
    "probability_samples_csv", "inner_scores_ranked_csv", "inner_logloss_summary_csv", "inner_logloss_per_fold_csv",
    "inner_winner_outer_pool_coefs_csv",
    "classifier_only_per_fold_csv", "classifier_only_summary_csv",
    "sample_decisions_inner_rank_risk_csv",
    "parallel_mc_cores", "cache_parallel_mc_cores", "blas_threads_pinned"
  ),
  value = c(
    format(Sys.time(), tz = "UTC", usetz = TRUE),
    OUTPUT_DIR,
    paste(ANALYSIS_INPUTS$results_path, collapse = "; "),
    SCENARIO_KEY,
    paste(BASELINE_TERMS, collapse = ", "),
    paste(IN_MODEL_FEATURE_TERMS, collapse = ", "),
    paste(KNN10_FEATURE_TERMS, collapse = ", "),
    paste(
      sprintf("%s=%s", names(INNER_RANK_TARGET_RISK_BY_LABEL_SET), INNER_RANK_TARGET_RISK_BY_LABEL_SET),
      collapse = "; "
    ),
    paste(as.character(OUTER_OPERATING_TARGET_RISKS), collapse = ", "),
    paste(as.character(CALIBRATION_PLOT_TARGET_RISKS), collapse = ", "),
    paste(as.character(MARCE_CURVE_TARGET_RISKS), collapse = ", "),
    paste(THRESHOLD_METHODS, collapse = ", "),
    paste(OUTER_SPLIT_TYPES, collapse = ", "),
    paste(NESTED_JOBS$job_key, collapse = ";"),
    calibration_run_manifest,
    class_balanced_ood_label(CLASS_BALANCED_OOD),
    file.path(OUTPUT_DIR, "nested_target_risk_per_fold_operating.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_summary_operating.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_calibration_curve.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_calibration_curve_per_fold.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_calibration_fold_coverage.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_marce.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_probability_samples.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_inner_scores_ranked.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_inner_logloss_summary.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_inner_logloss_per_fold.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_inner_winner_outer_pool_coefs.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_classifier_only_per_fold.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_classifier_only_summary.csv"),
    file.path(OUTPUT_DIR, "nested_target_risk_sample_decisions_inner_rank_risk.csv"),
    as.character(PARALLEL_MC_CORES),
    as.character(CACHE_PARALLEL_MC_CORES),
    "1"
  ),
  stringsAsFactors = FALSE
)
write_csv(manifest, file.path(OUTPUT_DIR, "nested_target_risk_manifest.csv"))

cat("Nested target-risk calibration analysis complete.\n")
cat(sprintf("Output directory: %s\n", OUTPUT_DIR))
