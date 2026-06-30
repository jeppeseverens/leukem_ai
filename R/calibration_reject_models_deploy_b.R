# =============================================================================
# Option B deployment calibration: deploy-loso fold matrices (SVM + SVM rejectors)
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
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
  "unmerged_maxprob",   "full_subtypes",       "data/out/final_train_test/deploy_loso_fold_results_10feb2026_unmerged_maxprob.rds",
  "merged_summed",      "collapsed_classes",   "data/out/final_train_test/deploy_loso_fold_results_10feb2026_merged_summed.rds",
  "merged_maxprob",     "collapsed_maxprob",   "data/out/final_train_test/deploy_loso_fold_results_10feb2026_merged_maxprob.rds"
) %>%
  mutate(results_path = file.path(repo_root, results_rel_path))

OUTPUT_DIR <- file.path(repo_root, "data/out/final_train_test/calibration_feature_utility_deploy_loso")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

source(file.path(repo_root, "R/calibration_reject_deploy_config.R"))

SCENARIO_KEY <- "with_leftout_ood_aware"
POOL_RULE <- "all_rows"
TEST_RULE <- "all_rows"
CUTOFF_SOURCE <- "deploy_loso"
ARTIFACT_TAG <- "_deploy_loso"

cat("Option B deploy-loso calibration: SVM + SVM rejector recipes\n")
cat(sprintf(
  "  Base model: %s, rejectors: %d, label sets: %d, target risk: %.1f%%, tag: %s\n",
  DEPLOY_BASE_MODEL, nrow(DEPLOY_REJECTORS), nrow(ANALYSIS_INPUTS),
  100 * DEPLOY_RISK_TARGET, ARTIFACT_TAG
))

source(if (file.exists("R/calibration_reject_core.R")) {
  "R/calibration_reject_core.R"
} else {
  "calibration_reject_core.R"
})
pin_blas_threads(1L)

merge_suffix_from_key <- function(label_set_key) {
  # label_set_key is the filesystem suffix base (unmerged_maxprob / merged_summed / merged_maxprob).
  paste0("_", label_set_key, ARTIFACT_TAG)
}

load_pool_fold_dfs <- function(results_obj, label_set_key) {
  fam <- results_obj$multivariate_results[[SCENARIO_KEY]][[DEPLOY_BASE_MODEL]]$loso$fold_matrices
  if (is.null(fam) || length(fam) < 3L) {
    stop(sprintf(
      "Need >=3 deploy-loso fold matrices for %s. Run R/build_deploy_loso_fold_matrices.R first.",
      label_set_key
    ))
  }
  pool_fold_dfs <- lapply(copy_fold_matrix_list(fam), extract_features)
  assert_calibration_terms_available(
    bind_rows(pool_fold_dfs),
    sprintf("deploy-loso pool fold features (%s | %s)", label_set_key, DEPLOY_BASE_MODEL)
  )
  pool_fold_dfs
}

export_deploy_cutoffs_both_methods <- function(
    cutoff_builder, label_set_key, rejector_key, rejector_mode, two_head_combine,
    rhs_key, merge_suffix, cutoffs_dir, is_two_head, cutoff_source = CUTOFF_SOURCE) {
  cutoff_rows <- lapply(THRESHOLD_METHODS, function(tm) {
    cutoff_out <- cutoff_builder(tm)
    make_deploy_cutoff_row(
      cutoff_out, cutoff_out$requested_target_risk, rejector_key, rejector_mode,
      two_head_combine, rhs_key, tm, is_two_head,
      cutoff_source = cutoff_source, base_model = DEPLOY_BASE_MODEL
    )
  })
  write_deploy_cutoffs_all_methods(
    cutoff_rows, label_set_key, rejector_key, merge_suffix, cutoffs_dir
  )
}

export_maxprob_glm_deployment <- function(
  pool_fold_dfs, label_set_key, label_set, risk_target, rej_row
) {
  rejector_key <- rej_row$rejector_key[[1]]
  rejector_mode <- rej_row$rejector_mode[[1]]
  two_head_combine <- rej_row$two_head_combine[[1]]
  merge_suffix <- merge_suffix_from_key(label_set_key)
  base_dir <- file.path(repo_root, "data/out/final_train_test")
  params_dir <- file.path(base_dir, paste0("multivariate_params_", label_set_key))
  cutoffs_dir <- file.path(base_dir, paste0("cutoffs_", label_set_key))
  dir.create(params_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(cutoffs_dir, recursive = TRUE, showWarnings = FALSE)

  train_u <- apply_row_rule(bind_rows(pool_fold_dfs), POOL_RULE)
  is_two_head <- is_two_head_rejector(rejector_mode)

  if (is_two_head) {
    fit_obj <- fit_twohead_models(train_u, DEPLOY_RHS_TERMS, min_rows = 20L)
    if (is.null(fit_obj)) stop(sprintf("Could not fit two-head maxprob GLM for %s.", rejector_key))
    params_df <- export_glm_twohead_coef_df(fit_obj, DEPLOY_BASE_MODEL)
    cutoff_builder <- function(tm) {
      out <- derive_glm_twohead_deploy_cutoff(
        pool_fold_dfs, DEPLOY_RHS_TERMS, risk_target,
        combine = two_head_combine, rejector_mode = rejector_mode, threshold_method = tm
      )
      out$requested_target_risk <- risk_target
      out
    }
  } else {
    fit_obj <- fit_binary_model(train_u, "accept_combined", DEPLOY_RHS_TERMS, min_rows = 20L)
    if (is.null(fit_obj)) stop(sprintf("Could not fit single-head maxprob GLM for %s.", rejector_key))
    params_df <- export_glm_coef_df(fit_obj$fit, "accept_combined", DEPLOY_BASE_MODEL)
    cutoff_builder <- function(tm) {
      out <- derive_glm_singlehead_deploy_cutoff(
        pool_fold_dfs, DEPLOY_RHS_TERMS, risk_target,
        rejector_mode = rejector_mode, threshold_method = tm
      )
      out$requested_target_risk <- risk_target
      out
    }
  }

  params_path <- file.path(
    params_dir,
    sprintf("multivariate_params_%s%s.csv", rej_row$params_file_key[[1]], merge_suffix)
  )
  write.csv(params_df, params_path, row.names = FALSE)
  cat(sprintf("  [%s] Exported GLM params (%s): %s\n", label_set_key, rejector_key, params_path))

  cutoffs_path <- export_deploy_cutoffs_both_methods(
    cutoff_builder, label_set_key, rejector_key, rejector_mode, two_head_combine,
    DEPLOY_RHS_TERMS, merge_suffix, cutoffs_dir, is_two_head
  )

  curve_out <- export_final_deploy_risk_curves_all_methods(
    pool_fold_dfs, label_set, label_set_key, rejector_key, rejector_mode,
    rejector_spec = NULL, two_head_combine = if (is_two_head) two_head_combine else NA_character_,
    merge_suffix, cutoffs_dir,
    base_model = DEPLOY_BASE_MODEL, risk_grid = DEPLOY_RISK_GRID
  )

  invisible(list(
    params_path = params_path,
    cutoffs_path = cutoffs_path,
    calibration_curve = curve_out$calibration_curve,
    calibration_per_fold = curve_out$calibration_per_fold
  ))
}

export_enet_rejector_deployment <- function(
  pool_fold_dfs, label_set_key, label_set, risk_target, rej_row
) {
  rejector_key <- rej_row$rejector_key[[1]]
  rejector_mode <- rej_row$rejector_mode[[1]]
  two_head_combine <- rej_row$two_head_combine[[1]]
  feature_terms <- rej_row$feature_terms[[1]]
  params_file_key <- rej_row$params_file_key[[1]]
  merge_suffix <- merge_suffix_from_key(label_set_key)
  base_dir <- file.path(repo_root, "data/out/final_train_test")
  params_dir <- file.path(base_dir, paste0("multivariate_params_", label_set_key))
  cutoffs_dir <- file.path(base_dir, paste0("cutoffs_", label_set_key))
  dir.create(params_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(cutoffs_dir, recursive = TRUE, showWarnings = FALSE)

  alpha <- enet_alpha_grid_values()[[1L]]
  is_two_head <- is_two_head_rejector(rejector_mode)
  fit_rejector_mode <- rejector_mode
  cache_id <- paste(label_set_key, merge_suffix, params_file_key, sep = "|")

  if (exists(cache_id, envir = .enet_deploy_fit_cache)) {
    ena_fit <- get(cache_id, envir = .enet_deploy_fit_cache)
  } else {
    ena_fit <- fit_enet_rejector_on_pool(
      pool_fold_dfs, "accept_combined", feature_terms, alpha, POOL_RULE,
      rejector_mode = fit_rejector_mode
    )
    if (is.null(ena_fit)) {
      stop(sprintf("Could not fit elastic-net for %s (%s).", label_set_key, rejector_key))
    }
    assign(cache_id, ena_fit, envir = .enet_deploy_fit_cache)
  }

  rejector_spec <- rejector_spec_elasticnet(alpha, feature_terms, rejector_mode)
  rejector_spec$lambda <- if (is_two_head) {
    c(correct = ena_fit$fit_correct$lambda, ood = ena_fit$fit_ood$lambda)
  } else {
    ena_fit$lambda
  }
  rhs_key <- rejector_spec_rhs_key(rejector_spec)
  write_params <- identical(rejector_key, params_file_key)
  params_path <- file.path(
    params_dir,
    sprintf("multivariate_params_%s%s.csv", params_file_key, merge_suffix)
  )

  if (write_params) {
    params_df <- export_enet_rejector_coef_df(ena_fit, DEPLOY_BASE_MODEL, fit_rejector_mode)
    write.csv(params_df, params_path, row.names = FALSE)
    cat(sprintf("  [%s] Exported elastic-net params (%s): %s\n", label_set_key, params_file_key, params_path))
  }

  cutoff_builder <- function(tm) {
    out <- derive_enet_deploy_cutoff(
      pool_fold_dfs, rejector_spec, risk_target,
      two_head_combine = if (is_two_head) two_head_combine else NULL,
      threshold_method = tm
    )
    out$requested_target_risk <- risk_target
    out
  }

  cutoffs_path <- export_deploy_cutoffs_both_methods(
    cutoff_builder, label_set_key, rejector_key, rejector_mode, two_head_combine,
    rhs_key, merge_suffix, cutoffs_dir, is_two_head
  )

  curve_out <- export_final_deploy_risk_curves_all_methods(
    pool_fold_dfs, label_set, label_set_key, rejector_key, rejector_mode,
    rejector_spec = rejector_spec,
    two_head_combine = if (is_two_head) two_head_combine else NA_character_,
    merge_suffix, cutoffs_dir,
    base_model = DEPLOY_BASE_MODEL, risk_grid = DEPLOY_RISK_GRID
  )

  invisible(list(
    params_path = params_path,
    cutoffs_path = cutoffs_path,
    calibration_curve = curve_out$calibration_curve,
    calibration_per_fold = curve_out$calibration_per_fold
  ))
}

exports <- list()
all_calibration_curves <- list()
all_calibration_per_fold <- list()
cci <- 1L
ccpf <- 1L
for (i in seq_len(nrow(ANALYSIS_INPUTS))) {
  row <- ANALYSIS_INPUTS[i, ]
  if (!file.exists(row$results_path)) {
    stop(sprintf("Missing deploy-loso results: %s", row$results_path))
  }
  risk_target <- INNER_RANK_TARGET_RISK_BY_LABEL_SET[[row$label_set]]
  cat(sprintf("\n=== %s (target risk %.1f%%) ===\n", row$label_set_key, 100 * risk_target))
  reset_enet_deploy_fit_cache()
  obj <- readRDS(row$results_path)
  pool_fold_dfs <- load_pool_fold_dfs(obj, row$label_set_key)

  for (j in seq_len(nrow(DEPLOY_REJECTORS))) {
    rej <- DEPLOY_REJECTORS[j, ]
    export_key <- paste(row$label_set_key, rej$rejector_key, sep = "__")
    if (identical(rej$rejector_family[[1]], "maxprob")) {
      exports[[export_key]] <- export_maxprob_glm_deployment(
        pool_fold_dfs, row$label_set_key, row$label_set, risk_target, rej
      )
    } else {
      exports[[export_key]] <- export_enet_rejector_deployment(
        pool_fold_dfs, row$label_set_key, row$label_set, risk_target, rej
      )
    }
    if (nrow(exports[[export_key]]$calibration_curve) > 0L) {
      all_calibration_curves[[cci]] <- exports[[export_key]]$calibration_curve
      cci <- cci + 1L
    }
    if (nrow(exports[[export_key]]$calibration_per_fold) > 0L) {
      all_calibration_per_fold[[ccpf]] <- exports[[export_key]]$calibration_per_fold
      ccpf <- ccpf + 1L
    }
  }
}

if (length(all_calibration_curves) > 0L) {
  write_csv(
    dplyr::bind_rows(all_calibration_curves),
    file.path(OUTPUT_DIR, "deploy_loso_calibration_curve.csv")
  )
}
if (length(all_calibration_per_fold) > 0L) {
  write_csv(
    dplyr::bind_rows(all_calibration_per_fold),
    file.path(OUTPUT_DIR, "deploy_loso_calibration_curve_per_fold.csv")
  )
}

write_csv(
  data.frame(
    key = c("timestamp_utc", "calibration_source", "artifact_tag", "base_model", "rejector_keys"),
    value = c(
      format(Sys.time(), tz = "UTC", usetz = TRUE),
      CUTOFF_SOURCE, ARTIFACT_TAG, DEPLOY_BASE_MODEL,
      paste(DEPLOY_REJECTORS$rejector_key, collapse = "; ")
    ),
    stringsAsFactors = FALSE
  ),
  file.path(OUTPUT_DIR, "deploy_loso_manifest.csv")
)

cat("\nOption B deploy-loso calibration complete.\n")
