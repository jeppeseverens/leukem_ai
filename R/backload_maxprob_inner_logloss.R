# Temporary backfill: max-prob inner CV log-loss rows into existing calibration CSVs.
# Does not re-run outer calibration curves. Delete after use.
# Usage (repo root): Rscript R/backload_maxprob_inner_logloss.R

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
})

repo_root <- if (file.exists("R/utility_functions.R")) "." else if (file.exists("utility_functions.R")) ".." else "."
source(file.path(repo_root, "R/utility_functions.R"))

OUTPUT_DIR <- file.path(repo_root, "data/out/outer_cv/calibration_feature_utility_selection_safe")
INNER_SCORES_PATH <- file.path(OUTPUT_DIR, "nested_target_risk_inner_scores_ranked.csv")
SUMMARY_PATH <- file.path(OUTPUT_DIR, "nested_target_risk_inner_logloss_summary.csv")
PER_FOLD_PATH <- file.path(OUTPUT_DIR, "nested_target_risk_inner_logloss_per_fold.csv")

ANALYSIS_INPUTS <- tibble::tribble(
  ~label_set_key,     ~label_set,          ~results_rel_path,
  "unmerged_maxprob", "full_subtypes",     "data/out/outer_cv/outer_cv_analysis_outputs_unmerged_maxprob/outer_cv_results.rds",
  "merged_summed",    "collapsed_classes", "data/out/outer_cv/outer_cv_analysis_outputs_merged_summed/outer_cv_results.rds"
) %>%
  mutate(results_path = file.path(repo_root, results_rel_path))

BASELINE_ONLY_RHS_KEY <- "max_prob"
BASELINE_TERMS <- c("max_prob")
ALL_FEATURE_TERMS <- c("max_prob", "margin", "entropy", "conformal_set_size_90")
SCENARIO_KEY <- "with_leftout_ood_aware"
POOL_RULE <- "all_rows"
TEST_RULE <- "all_rows"
OUTER_SPLIT_TYPES <- "loso"
CLASS_BALANCED_OOD <- TRUE

PARALLEL_MC_CORES <- suppressWarnings(as.integer(Sys.getenv("CALIBRATION_REJECT_MC_CORES", unset = NA_integer_)))
if (length(PARALLEL_MC_CORES) != 1L || is.na(PARALLEL_MC_CORES) || PARALLEL_MC_CORES < 1L) {
  PARALLEL_MC_CORES <- 4L
}
if (.Platform$OS.type != "unix") {
  PARALLEL_MC_CORES <- 1L
}

TWO_HEAD_EXPORT_SPECS <- tibble::tribble(
  ~rejector_key,      ~rejector_mode,     ~rejector_label,                        ~two_head_combine,
  "two_head_min",     "two_head_min",     "Two-head min(P(correct|ID), P(ID))", "min",
  "two_head_product", "two_head_product", "Two-head P(correct|ID) × P(ID)",     "product"
)

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

MAXPROB_JOBS <- tibble::tribble(
  ~ensemble_key, ~ensemble_label,            ~base_model,                ~recipe_key, ~rejector_method, ~feature_terms_key, ~recipe_rejector_key, ~recipe_label_single,     ~nested_rejector_mode,
  "poe",         "PoE (product of experts)", "Global_Product_Optimized", "maxprob",   "maxprob",        NA_character_,    "single_head",        "Single-head max-prob",   "single_head",
  "simple",      "Simple weighted average",  "Global_Simple_Optimized",  "maxprob",   "maxprob",        NA_character_,    "single_head",        "Single-head max-prob",   "single_head",
  "svm",         "SVM",                      "svm",                      "maxprob",   "maxprob",        NA_character_,    "single_head",        "Single-head max-prob",   "single_head",
  "svm",         "SVM",                      "svm",                      "maxprob",   "maxprob",        NA_character_,    "single_head",        "Single-head max-prob",   "two_head_product",
  "dnn",         "DNN (neural net)",         "neural_net",               "maxprob",   "maxprob",        NA_character_,    "single_head",        "Single-head max-prob",   "single_head",
  "dnn",         "DNN (neural net)",         "neural_net",               "maxprob",   "maxprob",        NA_character_,    "single_head",        "Single-head max-prob",   "two_head_product"
) %>%
  mutate(job_key = paste(ensemble_key, recipe_key, nested_rejector_mode, sep = "|"))

source(file.path(repo_root, "R/calibration_reject_core.R"))
pin_blas_threads(1L)

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

build_inner_logloss_tables <- function(inner_scores_df) {
  if (nrow(inner_scores_df) == 0L) {
    stop("inner_scores_df is empty.")
  }
  if (!all(c("inner_rank", "mean_logloss") %in% names(inner_scores_df))) {
    stop("inner_scores_df missing inner_rank or mean_logloss columns.")
  }

  per_fold <- inner_scores_df %>%
    filter(.data$inner_rank == 1L, is.finite(.data$mean_logloss)) %>%
    transmute(
      job_key = as.character(.data$job_key),
      config_key = as.character(.data$config_key),
      config_label = as.character(.data$config_label),
      ensemble_key = as.character(.data$ensemble_key),
      recipe_key = as.character(.data$recipe_key),
      rejector_method = as.character(.data$rejector_method),
      nested_rejector_mode = as.character(.data$nested_rejector_mode),
      label_set = as.character(.data$label_set),
      split_type = as.character(.data$split_type),
      target_fold = as.character(.data$target_fold),
      mean_logloss = as.numeric(.data$mean_logloss),
      sd_logloss = as.numeric(.data$sd_logloss),
      alpha = if ("alpha" %in% names(.)) as.numeric(.data$alpha) else NA_real_,
      feature_terms_key = as.character(.data$feature_terms_key)
    )

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
      inner_winner_alpha = stats::median(.data$alpha, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    group_by(.data$label_set, .data$split_type, .data$nested_rejector_mode) %>%
    arrange(.data$mean_inner_logloss, .data$config_key) %>%
    mutate(rank_mean_logloss = dplyr::row_number()) %>%
    ungroup() %>%
    arrange(.data$label_set, .data$split_type, .data$nested_rejector_mode, .data$rank_mean_logloss)

  list(summary = summary, per_fold = per_fold %>% arrange(.data$config_key, .data$label_set, .data$target_fold))
}

# Inner CV log-loss only (no outer threshold / calibration re-eval).
score_maxprob_outer_fold_inner_logloss <- function(job) {
  base_model <- TARGET_BASE_MODELS[[1L]]
  pool_fold_dfs <- job$pool_fold_dfs_by_model[[base_model]]
  rhs_terms <- strsplit(BASELINE_ONLY_RHS_KEY, ";", fixed = TRUE)[[1L]]
  rejector_mode <- job$rejector_mode
  ll <- inner_cv_logloss_rejector(
    pool_fold_dfs, "accept_combined", rhs_terms, POOL_RULE, TEST_RULE,
    rejector_mode = rejector_mode
  )
  if (!isTRUE(ll$ok)) {
    stop(sprintf(
      "maxprob inner CV log-loss failed [%s | %s | fold %s | %s].",
      job$label_set, job$split_type, job$fold_name, rejector_mode
    ))
  }
  data.frame(
    label_set = job$label_set,
    split_type = job$split_type,
    target_fold = as.character(job$fold_name),
    scenario_name = rejector_scenario_name(rejector_mode),
    rejector_mode = rejector_mode,
    inner_selection_fusion_rule = "max_prob_only_fixed",
    class_balanced_ood = if (is_two_head_rejector(rejector_mode)) class_balanced_ood_setting() else NA,
    scenario_key = SCENARIO_KEY,
    base_model = base_model,
    ensemble_rule = ensemble_rule_from_base_model(base_model),
    rhs_key = BASELINE_ONLY_RHS_KEY,
    recipe_key = "maxprob",
    mean_logloss = ll$mean_logloss,
    median_logloss = ll$median_logloss,
    sd_logloss = ll$sd_logloss,
    inner_rank = 1L,
    inner_selection_tier = "fixed",
    recipe_optional_count = NA_integer_,
    stringsAsFactors = FALSE
  )
}

build_outer_fold_jobs <- function(results_obj, label_set, rejector_mode) {
  assert_all_target_base_models_in_multivariate_results(results_obj, label_set)
  fam0 <- results_obj$multivariate_results[[SCENARIO_KEY]]
  jobs <- list()
  j_idx <- 1L
  for (split_type in OUTER_SPLIT_TYPES) {
    base_model_fold_feats <- list()
    for (base_model in TARGET_BASE_MODELS) {
      if (!base_model %in% names(fam0) || !split_type %in% names(fam0[[base_model]])) {
        next
      }
      ff <- extract_ood_aware_fold_feats(results_obj, base_model, split_type, label_set)
      if (is.null(ff)) {
        stop(sprintf(
          "Fewer than 4 augmented folds for %s | %s | %s.",
          SCENARIO_KEY, base_model, label_set
        ))
      }
      base_model_fold_feats[[base_model]] <- ff
    }
    fold_names <- Reduce(intersect, lapply(base_model_fold_feats, names))
    if (length(fold_names) < 4L) {
      stop(sprintf("Fewer than 4 common outer folds for %s | %s.", label_set, split_type))
    }
    for (fold_name in fold_names) {
      jobs[[j_idx]] <- list(
        pool_fold_dfs_by_model = lapply(
          base_model_fold_feats,
          function(ff) ff[setdiff(fold_names, fold_name)]
        ),
        target_df_by_model = lapply(base_model_fold_feats, function(ff) ff[[fold_name]]),
        available_base_models = TARGET_BASE_MODELS,
        split_type = split_type,
        label_set = label_set,
        fold_name = fold_name,
        rejector_mode = rejector_mode
      )
      j_idx <- j_idx + 1L
    }
  }
  jobs
}

collect_maxprob_inner_scores <- function(results_obj, label_set, job_row) {
  rejector_mode <- job_row$nested_rejector_mode[[1]]
  jobs <- build_outer_fold_jobs(results_obj, label_set, rejector_mode)
  cat(sprintf(
    "    %d outer folds (mc.cores=%d)\n",
    length(jobs), PARALLEL_MC_CORES
  ))
  rows <- if (PARALLEL_MC_CORES > 1L && length(jobs) > 1L) {
    parallel::mclapply(jobs, score_maxprob_outer_fold_inner_logloss, mc.cores = PARALLEL_MC_CORES)
  } else {
    lapply(jobs, score_maxprob_outer_fold_inner_logloss)
  }
  bind_rows(rows)
}

if (!file.exists(INNER_SCORES_PATH)) {
  stop("Missing ", INNER_SCORES_PATH, ". Run ridge calibration first.")
}

cat("Backloading max-prob inner CV log-loss into:\n  ", OUTPUT_DIR, "\n", sep = "")
existing_scores <- read_csv(INNER_SCORES_PATH, show_col_types = FALSE)
maxprob_job_keys <- MAXPROB_JOBS$job_key

new_parts <- list()
part_i <- 1L
for (jr in seq_len(nrow(MAXPROB_JOBS))) {
  job_row <- MAXPROB_JOBS[jr, ]
  TARGET_BASE_MODEL <- job_row$base_model
  TARGET_BASE_MODELS <- TARGET_BASE_MODEL
  cat(sprintf("\n=== [%s] %s ===\n", job_row$job_key, job_row$ensemble_label))

  inner_export_spec <- if (identical(job_row$nested_rejector_mode[[1]], "two_head_product")) {
    dplyr::bind_rows(export_specs_for_nested_job(job_row)) %>%
      filter(.data$two_head_combine == "product") %>%
      slice(1)
  } else {
    export_specs_for_nested_job(job_row)[[1L]]
  }

  for (i in seq_len(nrow(ANALYSIS_INPUTS))) {
    row <- ANALYSIS_INPUTS[i, ]
    if (!file.exists(row$results_path)) {
      stop("Missing results file: ", row$results_path)
    }
    cat(sprintf("  Loading %s | %s ...\n", job_row$job_key, row$label_set))
    obj <- readRDS(row$results_path)
    scored <- collect_maxprob_inner_scores(obj, row$label_set, job_row)
    new_parts[[part_i]] <- tag_calibration_run_export(scored, job_row, inner_export_spec)
    part_i <- part_i + 1L
  }
}

new_scores <- bind_rows(new_parts)
cat(sprintf("\nNew max-prob inner-score rows: %d\n", nrow(new_scores)))

merged_scores <- existing_scores %>%
  filter(!.data$job_key %in% maxprob_job_keys) %>%
  bind_rows(new_scores) %>%
  arrange(.data$job_key, .data$label_set, .data$target_fold)

tables <- build_inner_logloss_tables(merged_scores)

cat("Writing updated inner log-loss tables ...\n")
write_csv(merged_scores, INNER_SCORES_PATH)
write_csv(tables$summary, SUMMARY_PATH)
write_csv(tables$per_fold, PER_FOLD_PATH)

cat(sprintf(
  "Done. Summary rows: %d (max-prob jobs: %d).\n",
  nrow(tables$summary),
  sum(tables$summary$recipe_key == "maxprob")
))
