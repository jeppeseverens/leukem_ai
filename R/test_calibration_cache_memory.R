# Memory probe for calibration reject cache parallelism.
# Phase 1 (once): build and save recipe stubs
#   CALIBRATION_REJECT_MC_CORES=1 Rscript R/test_calibration_cache_memory.R save
# Phase 2 (repeat per cache core count):
#   CALIBRATION_REJECT_CACHE_MC_CORES=1 Rscript R/test_calibration_cache_memory.R cache
#   CALIBRATION_REJECT_CACHE_MC_CORES=2 Rscript R/test_calibration_cache_memory.R cache
#   CALIBRATION_REJECT_CACHE_MC_CORES=3 Rscript R/test_calibration_cache_memory.R cache

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
phase <- if (length(args) >= 1L) args[[1]] else "all"
if (!phase %in% c("all", "save", "cache")) {
  stop("Usage: Rscript R/test_calibration_cache_memory.R [save|cache|all]")
}

repo_root <- if (file.exists("R/calibration_reject_core.R")) "." else ".."
STUB_PATH <- file.path(repo_root, "data/out/outer_cv/calibration_feature_utility_selection_safe/_memtest_recipe_stubs.rds")

source(file.path(repo_root, "R/utility_functions.R"))

PARALLEL_MC_CORES <- suppressWarnings(as.integer(Sys.getenv("CALIBRATION_REJECT_MC_CORES", unset = "1")))
CACHE_MC_CORES <- suppressWarnings(as.integer(Sys.getenv("CALIBRATION_REJECT_CACHE_MC_CORES", unset = "1")))
LABEL_SET <- Sys.getenv("CALIBRATION_MEMTEST_LABEL_SET", unset = "collapsed_classes")
RESULTS_PATH <- file.path(
  repo_root,
  if (LABEL_SET == "collapsed_classes") {
    "data/out/outer_cv/outer_cv_analysis_outputs_merged_summed/outer_cv_results.rds"
  } else {
    "data/out/outer_cv/outer_cv_analysis_outputs_unmerged_maxprob/outer_cv_results.rds"
  }
)

OPERATING_TARGET_RISK_BY_LABEL_SET <- c(
  full_subtypes = 0.075,
  collapsed_classes = 0.05
)
OUTER_OPERATING_TARGET_RISKS <- c(0.05, 0.075, 0.10)
PRIMARY_TABLE_TARGET_RISK <- OPERATING_TARGET_RISK_BY_LABEL_SET[[LABEL_SET]]
PROBE_CACHE_RISKS <- unique(c(
  OUTER_OPERATING_TARGET_RISKS,
  seq(25L, 150L, by = 25L) / 1000,
  seq(50L, 100L, by = 25L) / 1000
))
INNER_SELECTION_METHOD <- "elasticnet"
ENET_ALPHA_GRID <- c(0)
SCENARIO_KEY <- "with_leftout_ood_aware"
POOL_RULE <- "all_rows"
TEST_RULE <- "all_rows"
TARGET_BASE_MODEL <- "Global_Product_Optimized"
TARGET_BASE_MODELS <- TARGET_BASE_MODEL
ALL_FEATURE_TERMS <- c(
  "max_prob", "margin", "entropy", "top1_prob_variance_across_models",
  "knn10_mean_d", "knn10_min_d", "knn10_q90_d",
  "conformal_set_size_90"
)
BASELINE_TERMS <- c("max_prob")
FULL_COVERAGE_THRESHOLD <- 0

source(file.path(repo_root, "R/calibration_reject_core.R"))
pin_blas_threads(1L)

rss_mb <- function() {
  kb <- suppressWarnings(as.numeric(system(sprintf("ps -o rss= -p %d", Sys.getpid()), intern = TRUE)))
  if (length(kb) != 1L || !is.finite(kb)) NA_real_ else kb / 1024
}

peak_mb <- new.env(parent = emptyenv())
peak_mb$val <- rss_mb()
track_peak <- function(label) {
  cur <- rss_mb()
  if (is.finite(cur) && cur > peak_mb$val) peak_mb$val <- cur
  cat(sprintf("  [%s] RSS=%.0f MB (peak=%.0f MB)\n", label, cur, peak_mb$val))
  invisible(cur)
}

run_save <- function() {
  cat("=== Phase: save recipe stubs ===\n")
  cat(sprintf("  label_set=%s nested_mc=%d\n", LABEL_SET, PARALLEL_MC_CORES))
  if (!file.exists(RESULTS_PATH)) stop("Missing: ", RESULTS_PATH)
  track_peak("start")
  obj <- readRDS(RESULTS_PATH)
  track_peak("after readRDS")
  res <- run_nested_target_risk_analysis(
    obj, LABEL_SET, risk_target = PRIMARY_TABLE_TARGET_RISK,
    rejector_mode = "single_head"
  )
  track_peak(sprintf("after nested (%d stubs)", length(res$recipe_jobs)))
  dir.create(dirname(STUB_PATH), recursive = TRUE, showWarnings = FALSE)
  saveRDS(
    list(
      recipe_jobs = res$recipe_jobs,
      per_fold_operating_df = res$per_fold_operating_df,
      label_set = LABEL_SET,
      probe_cache_risks = PROBE_CACHE_RISKS
    ),
    STUB_PATH
  )
  cat(sprintf("  saved: %s\n", STUB_PATH))
  cat(sprintf("PEAK_RSS_MB=%.0f\n", peak_mb$val))
}

run_cache <- function() {
  cat("=== Phase: cache build ===\n")
  cat(sprintf("  cache_mc=%d probe_risks=%d\n", CACHE_MC_CORES, length(PROBE_CACHE_RISKS)))
  if (!file.exists(STUB_PATH)) stop("Run save phase first: ", STUB_PATH)
  track_peak("start")
  saved <- readRDS(STUB_PATH)
  recipe_jobs <- saved$recipe_jobs
  pf_op <- saved$per_fold_operating_df
  risks <- saved$probe_cache_risks
  track_peak(sprintf("after load stubs (%d)", length(recipe_jobs)))

  Sys.setenv(CALIBRATION_REJECT_CACHE_MC_CORES = as.character(CACHE_MC_CORES))
  t0 <- Sys.time()
  eval_cache_df <- build_outer_eval_cache(
    recipe_jobs, risks, THRESHOLD_METHODS, seed_operating_df = pf_op
  )
  t_inner <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  track_peak(sprintf("after inner cache (%d rows, %.1fs)", nrow(eval_cache_df), t_inner))

  t0 <- Sys.time()
  mp_stubs <- stubs_with_fixed_rhs(recipe_jobs, "max_prob")
  mp_cache_df <- build_outer_eval_cache(mp_stubs, risks, THRESHOLD_METHODS)
  t_mp <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  track_peak(sprintf("after maxprob cache (%d rows, %.1fs)", nrow(mp_cache_df), t_mp))

  curve_df <- build_calibration_curve_from_eval_cache(eval_cache_df, risks)
  track_peak(sprintf("after curve (%d rows)", nrow(curve_df)))

  cat(sprintf("\nPEAK_RSS_MB=%.0f\n", peak_mb$val))
  cat(sprintf("CACHE_MC_CORES=%d INNER_SEC=%.1f MP_SEC=%.1f\n", CACHE_MC_CORES, t_inner, t_mp))
}

if (phase == "save") {
  run_save()
} else if (phase == "cache") {
  run_cache()
} else {
  run_save()
  run_cache()
}
