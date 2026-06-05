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

fit_logistic <- function(train_df, y_col, rhs_terms) {
  f <- stats::as.formula(paste(y_col, "~", paste(rhs_terms, collapse = " + ")))
  glm_fit <- tryCatch(
    stats::glm(f, data = train_df, family = stats::binomial(), control = stats::glm.control(maxit = 200, epsilon = 1e-8)),
    error = function(e) NULL
  )
  if (is.null(glm_fit) || !isTRUE(glm_fit$converged)) return(NULL)
  glm_fit
}

fit_binary_model <- function(train_df, y_col, rhs_terms, min_rows = 20L) {
  if (length(rhs_terms) == 0 || !y_col %in% colnames(train_df)) return(NULL)
  rhs_terms <- rhs_terms[rhs_terms %in% colnames(train_df)]
  if (length(rhs_terms) == 0) return(NULL)
  tr <- train_df[, c(y_col, rhs_terms), drop = FALSE] %>% filter(complete.cases(.))
  if (nrow(tr) < min_rows || length(unique(tr[[y_col]])) < 2L) return(NULL)
  rhs_terms <- rhs_terms[sapply(rhs_terms, function(v) length(unique(tr[[v]])) > 1L)]
  if (length(rhs_terms) == 0) return(NULL)
  fit <- fit_logistic(tr, y_col, rhs_terms)
  if (is.null(fit)) return(NULL)
  list(fit = fit, rhs_terms = rhs_terms)
}

predict_binary_model <- function(fit_obj, test_df) {
  if (is.null(fit_obj) || nrow(test_df) == 0) return(NULL)
  rhs_terms <- fit_obj$rhs_terms
  if (length(rhs_terms) == 0 || !all(rhs_terms %in% colnames(test_df))) return(NULL)
  te_rhs <- test_df[, rhs_terms, drop = FALSE]
  keep <- complete.cases(te_rhs)
  if (sum(keep) == 0) return(NULL)
  te <- te_rhs[keep, , drop = FALSE]
  p_hat <- tryCatch(
    as.numeric(stats::predict(fit_obj$fit, newdata = te, type = "response")),
    error = function(e) NULL
  )
  if (is.null(p_hat)) return(NULL)
  data.frame(row_id = which(keep), p_hat = p_hat)
}

calc_binary_auprc <- function(y_true, p_hat) {
  keep <- is.finite(y_true) & is.finite(p_hat)
  if (sum(keep) < 10L) return(NA_real_)
  y <- as.integer(y_true[keep] > 0)
  p <- as.numeric(p_hat[keep])
  n_pos <- sum(y == 1L)
  n_neg <- sum(y == 0L)
  if (n_pos == 0L || n_neg == 0L) return(NA_real_)
  ord <- order(p, decreasing = TRUE)
  y_ord <- y[ord]
  tp <- cumsum(y_ord == 1L)
  fp <- cumsum(y_ord == 0L)
  precision <- tp / pmax(1L, tp + fp)
  recall <- tp / n_pos
  precision <- c(1, precision)
  recall <- c(0, recall)
  sum((recall[-1] - recall[-length(recall)]) * precision[-1], na.rm = TRUE)
}

calc_binary_metrics <- function(y_true, p_hat) {
  keep <- is.finite(y_true) & is.finite(p_hat)
  if (sum(keep) < 10L) return(data.frame(auprc = NA_real_))
  y <- as.numeric(y_true[keep]); p <- as.numeric(p_hat[keep])
  data.frame(auprc = calc_binary_auprc(y, p))
}

select_threshold_with_target_risk <- function(y_true, p_hat, is_seen, risk_target = TARGET_RISK) {
  keep <- is.finite(y_true) & is.finite(p_hat) & is.finite(is_seen)
  if (sum(keep) < 10L) {
    return(data.frame(
      threshold = NA_real_, risk_all_accepted = NA_real_,
      coverage_seen = NA_real_, n_accepted = NA_integer_, n_seen_total = NA_integer_
    ))
  }
  y <- as.integer(y_true[keep] > 0)
  p <- as.numeric(p_hat[keep])
  seen <- as.integer(is_seen[keep] > 0)
  n_seen_total <- sum(seen == 1L)
  if (n_seen_total <= 0L) {
    return(data.frame(
      threshold = NA_real_, risk_all_accepted = NA_real_,
      coverage_seen = NA_real_, n_accepted = NA_integer_, n_seen_total = n_seen_total
    ))
  }

  # Evaluate all unique thresholds in O(n log n): sort once, then use cumulative counts.
  ord <- order(p, decreasing = TRUE)
  p_ord <- p[ord]
  y_ord <- y[ord]
  seen_ord <- seen[ord]
  n_acc_cum <- seq_along(p_ord)
  err_cum <- cumsum(y_ord == 0L)
  seen_acc_cum <- cumsum(seen_ord == 1L)

  run <- rle(p_ord)
  idx_last <- cumsum(run$lengths)
  if (length(idx_last) == 0L) {
    return(data.frame(
      threshold = NA_real_, risk_all_accepted = NA_real_,
      coverage_seen = NA_real_, n_accepted = 0L, n_seen_total = n_seen_total
    ))
  }
  eval_df <- data.frame(
    threshold = as.numeric(p_ord[idx_last]),
    risk_all_accepted = as.numeric(err_cum[idx_last] / pmax(1L, n_acc_cum[idx_last])),
    coverage_seen = as.numeric(seen_acc_cum[idx_last] / n_seen_total),
    n_accepted = as.integer(n_acc_cum[idx_last]),
    n_seen_total = as.integer(n_seen_total),
    stringsAsFactors = FALSE
  )
  in_band <- eval_df %>% filter(risk_all_accepted <= risk_target)
  if (nrow(in_band) > 0L) {
    in_band %>% arrange(desc(coverage_seen), risk_all_accepted, desc(n_accepted), threshold) %>% slice(1L)
  } else {
    eval_df %>% arrange(risk_all_accepted, desc(coverage_seen), desc(n_accepted), threshold) %>% slice(1L)
  }
}

extract_features <- function(prob_matrix) {
  feats <- get_rejection_features_from_matrix(prob_matrix)
  feats$true_class <- gsub("Class\\. ", "", prob_matrix$y)
  feats$is_seen <- if ("is_leftout" %in% colnames(prob_matrix)) as.integer(!as.logical(prob_matrix$is_leftout)) else 1L
  feats$is_unseen <- 1L - feats$is_seen
  feats$accept_combined <- as.integer(feats$correct == 1L & feats$is_seen == 1L)
  feats
}

# Cohen's kappa on accepted rows (pred vs truth); NA if too few accepts or one class only.
kappa_accepted_at_threshold <- function(truth_chr, pred_chr, p_hat, threshold) {
  keep <- is.finite(p_hat) & is.finite(threshold)
  if (sum(keep) < 2L) return(NA_real_)
  acc <- p_hat[keep] >= threshold
  if (sum(acc) < 2L) return(NA_real_)
  truth_a <- as.character(truth_chr[keep][acc])
  pred_a <- as.character(pred_chr[keep][acc])
  lvls <- sort(unique(c(truth_a, pred_a)))
  if (length(lvls) < 2L) return(NA_real_)
  fast_kappa(factor(pred_a, levels = lvls), factor(truth_a, levels = lvls))
}

apply_row_rule <- function(df, rule) {
  if (is.null(df) || nrow(df) == 0) return(df)
  if (rule == "known_only") return(df[df$is_seen == 1L, , drop = FALSE])
  df
}

# LOSO-within-pool OOF scores for target-risk threshold (no dedicated cal cohort).
# If glm/predict fails on any hold-one-out study in this pool, return NULL (caller drops whole RHS recipe).
pool_oof_singlehead <- function(fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, min_rows = 20L) {
  if (length(oof_ids) < 2L) return(NULL)
  y_all <- numeric(0)
  p_all <- numeric(0)
  seen_all <- integer(0)
  for (hold_id in oof_ids) {
    train_ids <- setdiff(oof_ids, hold_id)
    train_df <- bind_rows(fold_dfs[train_ids])
    hold_df <- fold_dfs[[hold_id]]
    tr_u <- apply_row_rule(train_df, pool_rule)
    hold_u <- apply_row_rule(hold_df, score_rule)
    if (nrow(tr_u) < min_rows || nrow(hold_u) < 10L) return(NULL)
    if (!y_col %in% colnames(tr_u) || !y_col %in% colnames(hold_u)) return(NULL)
    fit_obj <- fit_binary_model(tr_u, y_col, rhs_terms)
    if (is.null(fit_obj)) return(NULL)
    rhs <- fit_obj$rhs_terms
    hold2 <- hold_u[, unique(c(y_col, rhs)), drop = FALSE]
    keep <- complete.cases(hold2)
    if (sum(keep) < 10L) return(NULL)
    pred <- predict_binary_model(fit_obj, hold2[keep, , drop = FALSE])
    if (is.null(pred)) return(NULL)
    row_h <- which(keep)[pred$row_id]
    y_all <- c(y_all, as.numeric(hold_u[[y_col]][row_h]))
    p_all <- c(p_all, pred$p_hat)
    seen_all <- c(
      seen_all,
      if ("is_seen" %in% colnames(hold_u)) as.integer(hold_u$is_seen[row_h] > 0) else rep(1L, length(row_h))
    )
  }
  if (length(y_all) < 10L) return(NULL)
  list(y = y_all, p_hat = p_all, is_seen = seen_all)
}

threshold_from_oof_pool_singlehead <- function(
  fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, risk_target, min_rows = 20L
) {
  oof <- pool_oof_singlehead(fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, min_rows)
  if (is.null(oof)) return(NA_real_)
  thr_df <- select_threshold_with_target_risk(oof$y, oof$p_hat, oof$is_seen, risk_target)
  as.numeric(thr_df$threshold[[1]])
}

# Risk (error rate) and seen coverage among rows with p_hat >= threshold (same notion as select_threshold_with_target_risk).
metrics_at_fixed_threshold <- function(y_true, p_hat, is_seen, threshold) {
  keep <- is.finite(y_true) & is.finite(p_hat) & is.finite(is_seen) & is.finite(threshold)
  if (sum(keep) < 10L) {
    return(list(
      risk_all_accepted = NA_real_, coverage_seen = NA_real_,
      n_accepted = NA_integer_, n_seen_total = NA_integer_
    ))
  }
  y <- as.numeric(y_true[keep])
  p <- as.numeric(p_hat[keep])
  seen <- as.integer(is_seen[keep] > 0)
  acc <- p >= threshold
  n_seen_total <- sum(seen == 1L)
  if (n_seen_total <= 0L || sum(acc) < 1L) {
    return(list(
      risk_all_accepted = NA_real_, coverage_seen = NA_real_,
      n_accepted = as.integer(sum(acc)), n_seen_total = n_seen_total
    ))
  }
  risk_all_accepted <- mean(y[acc] == 0)
  coverage_seen <- sum(acc & seen == 1L) / n_seen_total
  list(
    risk_all_accepted = as.numeric(risk_all_accepted),
    coverage_seen = as.numeric(coverage_seen),
    n_accepted = as.integer(sum(acc)),
    n_seen_total = as.integer(n_seen_total)
  )
}

# Inner CV: train on pool\\val; threshold from LOSO-OOF on train; evaluate on val.
# Any inner validation fold where OOF thresholding fails or glm/predict/metrics fail drops the whole RHS recipe (no partial CV).
inner_cv_strict_singlehead <- function(pool_fold_dfs, y_col, rhs_terms, pool_rule, test_rule, risk_target, min_rows = 20L) {
  ids <- names(pool_fold_dfs)
  if (length(ids) < 3L) return(list(ok = FALSE))
  covs <- numeric(length(ids))
  risks <- numeric(length(ids))
  for (iv in seq_along(ids)) {
    val_id <- ids[[iv]]
    train_ids <- setdiff(ids, val_id)
    if (length(train_ids) < 2L) return(list(ok = FALSE))
    thr <- threshold_from_oof_pool_singlehead(
      pool_fold_dfs, train_ids, y_col, rhs_terms, pool_rule, test_rule, risk_target, min_rows
    )
    if (!is.finite(thr)) return(list(ok = FALSE))
    tr <- bind_rows(pool_fold_dfs[train_ids])
    te <- pool_fold_dfs[[val_id]]
    tr_u <- apply_row_rule(tr, pool_rule)
    te_u <- apply_row_rule(te, test_rule)
    if (nrow(tr_u) < min_rows || nrow(te_u) < 10L) return(list(ok = FALSE))
    if (!y_col %in% colnames(tr_u) || !y_col %in% colnames(te_u)) return(list(ok = FALSE))
    fit_obj <- fit_binary_model(tr_u, y_col, rhs_terms)
    if (is.null(fit_obj)) return(list(ok = FALSE))
    rhs <- fit_obj$rhs_terms
    te2 <- te_u[, unique(c(y_col, rhs)), drop = FALSE]
    keep_te <- complete.cases(te2)
    if (sum(keep_te) < 10L) return(list(ok = FALSE))
    pred_te <- predict_binary_model(fit_obj, te2[keep_te, , drop = FALSE])
    if (is.null(pred_te)) return(list(ok = FALSE))
    row_te <- which(keep_te)[pred_te$row_id]
    y_te <- as.numeric(te_u[[y_col]][row_te])
    p_te <- pred_te$p_hat
    seen_te <- if ("is_seen" %in% colnames(te_u)) as.integer(te_u$is_seen[row_te] > 0) else rep(1L, length(row_te))
    m <- metrics_at_fixed_threshold(y_te, p_te, seen_te, thr)
    if (!is.finite(m$risk_all_accepted) || !is.finite(m$coverage_seen)) return(list(ok = FALSE))
    covs[iv] <- m$coverage_seen
    risks[iv] <- m$risk_all_accepted
  }
  if (!all(is.finite(risks)) || !all(is.finite(covs))) return(list(ok = FALSE))
  mean_risk <- mean(risks)
  mean_cov <- mean(covs)
  median_risk <- stats::median(risks)
  median_cov <- stats::median(covs)
  sd_cov <- stats::sd(covs)
  sd_risk <- stats::sd(risks)
  list(
    ok = TRUE,
    mean_coverage = mean_cov,
    mean_risk = mean_risk,
    median_coverage = median_cov,
    median_risk = median_risk,
    sd_coverage = sd_cov,
    sd_risk = sd_risk
  )
}

candidate_terms_from_df <- function(df) {
  ALL_FEATURE_TERMS[vapply(ALL_FEATURE_TERMS, function(v) {
    v %in% colnames(df) && all(is.finite(df[[v]])) && length(unique(df[[v]])) > 1L
  }, logical(1))]
}

# Outer: fit on pool; evaluate on target. Threshold from pool LOSO-OOF unless fixed_threshold is set.
outer_eval_singlehead <- function(
  pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule, risk_target,
  min_rows = 20L, fixed_threshold = NULL
) {
  pool_ids <- names(pool_fold_dfs)
  if (length(pool_ids) < 2L) return(list(ok = FALSE))
  tgt_u <- apply_row_rule(target_df, test_rule)
  if (nrow(tgt_u) < 10L) return(list(ok = FALSE))

  thr <- if (!is.null(fixed_threshold)) {
    as.numeric(fixed_threshold)
  } else {
    threshold_from_oof_pool_singlehead(
      pool_fold_dfs, pool_ids, y_col, rhs_terms, pool_rule, test_rule, risk_target, min_rows
    )
  }
  if (!is.finite(thr)) return(list(ok = FALSE))

  train_df <- bind_rows(pool_fold_dfs)
  train_u <- apply_row_rule(train_df, pool_rule)
  if (nrow(train_u) < min_rows) return(list(ok = FALSE))
  fit_obj <- fit_binary_model(train_u, y_col, rhs_terms)
  if (is.null(fit_obj)) return(list(ok = FALSE))
  rhs <- fit_obj$rhs_terms
  te2 <- tgt_u[, unique(c(y_col, rhs)), drop = FALSE]
  keep_te <- complete.cases(te2)
  if (sum(keep_te) < 10L) return(list(ok = FALSE))
  pred_te <- predict_binary_model(fit_obj, te2[keep_te, , drop = FALSE])
  if (is.null(pred_te)) return(list(ok = FALSE))
  row_te <- which(keep_te)[pred_te$row_id]
  y_te <- as.numeric(tgt_u[[y_col]][row_te])
  p_te <- pred_te$p_hat
  seen_te <- if ("is_seen" %in% colnames(tgt_u)) as.integer(tgt_u$is_seen[row_te] > 0) else rep(1L, length(row_te))
  m <- metrics_at_fixed_threshold(y_te, p_te, seen_te, thr)
  if (!is.finite(m$risk_all_accepted) || !is.finite(m$coverage_seen)) return(list(ok = FALSE))
  aup <- calc_binary_metrics(y_te, p_te)$auprc
  if (!all(c("true_class", "pred_class") %in% colnames(tgt_u))) {
    return(list(ok = FALSE))
  }
  kappa_acc <- kappa_accepted_at_threshold(
    tgt_u$true_class[row_te], tgt_u$pred_class[row_te], p_te, thr
  )
  list(
    ok = TRUE,
    threshold = thr,
    threshold_median = thr,
    risk_all_accepted = m$risk_all_accepted,
    risk_all_accepted_median = m$risk_all_accepted,
    coverage_seen = m$coverage_seen,
    coverage_seen_median = m$coverage_seen,
    kappa_accepted = kappa_acc,
    kappa_accepted_median = kappa_acc,
    auprc_outer = aup,
    auprc_outer_median = aup
  )
}

# Rejection rates by outcome stratum on the outer target fold (rejected = p_hat < threshold).
rejection_stratum_counts_singlehead <- function(
  pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule, risk_target,
  min_rows = 20L
) {
  pool_ids <- names(pool_fold_dfs)
  if (length(pool_ids) < 2L) return(list(ok = FALSE))
  tgt_u <- apply_row_rule(target_df, test_rule)
  if (nrow(tgt_u) < 10L) return(list(ok = FALSE))
  if (!all(c("correct", "is_seen") %in% colnames(tgt_u))) return(list(ok = FALSE))

  thr <- threshold_from_oof_pool_singlehead(
    pool_fold_dfs, pool_ids, y_col, rhs_terms, pool_rule, test_rule, risk_target, min_rows
  )
  if (!is.finite(thr)) return(list(ok = FALSE))

  train_df <- bind_rows(pool_fold_dfs)
  train_u <- apply_row_rule(train_df, pool_rule)
  if (nrow(train_u) < min_rows) return(list(ok = FALSE))
  fit_obj <- fit_binary_model(train_u, y_col, rhs_terms)
  if (is.null(fit_obj)) return(list(ok = FALSE))
  rhs <- fit_obj$rhs_terms
  te2 <- tgt_u[, unique(c(y_col, "correct", "is_seen", rhs)), drop = FALSE]
  keep_te <- complete.cases(te2)
  if (sum(keep_te) < 10L) return(list(ok = FALSE))
  pred_te <- predict_binary_model(fit_obj, te2[keep_te, , drop = FALSE])
  if (is.null(pred_te)) return(list(ok = FALSE))

  row_te <- which(keep_te)[pred_te$row_id]
  seen <- as.integer(tgt_u$is_seen[row_te] > 0)
  correct <- as.integer(tgt_u$correct[row_te] > 0)
  p_te <- pred_te$p_hat
  rejected <- p_te < thr

  ood <- seen == 0L
  incorrect_seen <- seen == 1L & correct == 0L
  correct_seen <- seen == 1L & correct == 1L

  list(
    ok = TRUE,
    threshold = thr,
    n_ood = sum(ood),
    n_rejected_ood = sum(rejected & ood),
    n_incorrect_seen = sum(incorrect_seen),
    n_rejected_incorrect_seen = sum(rejected & incorrect_seen),
    n_correct_seen = sum(correct_seen),
    n_rejected_correct_seen = sum(rejected & correct_seen)
  )
}

pct_rejected <- function(n_rejected, n_total) {
  ifelse(!is.finite(n_total) | n_total <= 0, NA_real_, 100 * n_rejected / n_total)
}

build_rejection_stratum_per_fold <- function(recipe_jobs, risk_target) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  rows <- lapply(recipe_jobs, function(stub) {
    rhs_terms <- strsplit(stub$inner_winner_rhs_key, ";", fixed = TRUE)[[1]]
    cnt <- rejection_stratum_counts_singlehead(
      stub$pool_fold_dfs, stub$target_df, "accept_combined", rhs_terms,
      POOL_RULE, TEST_RULE, risk_target
    )
    if (is.null(cnt) || !isTRUE(cnt$ok)) return(NULL)
    data.frame(
      label_set = stub$label_set,
      split_type = stub$split_type,
      target_fold = as.character(stub$fold_name),
      scenario_key = SCENARIO_KEY,
      scenario_name = SCENARIO_NAME,
      requested_target_risk = risk_target,
      threshold_outer = cnt$threshold,
      n_ood = cnt$n_ood,
      n_rejected_ood = cnt$n_rejected_ood,
      pct_rejected_ood = pct_rejected(cnt$n_rejected_ood, cnt$n_ood),
      n_incorrect_seen = cnt$n_incorrect_seen,
      n_rejected_incorrect_seen = cnt$n_rejected_incorrect_seen,
      pct_rejected_incorrect_seen = pct_rejected(cnt$n_rejected_incorrect_seen, cnt$n_incorrect_seen),
      n_correct_seen = cnt$n_correct_seen,
      n_rejected_correct_seen = cnt$n_rejected_correct_seen,
      pct_rejected_correct_seen = pct_rejected(cnt$n_rejected_correct_seen, cnt$n_correct_seen),
      stringsAsFactors = FALSE
    )
  })
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) == 0L) data.frame() else bind_rows(rows)
}

summarize_rejection_stratum <- function(per_fold_df) {
  if (nrow(per_fold_df) == 0L) return(data.frame())
  per_fold_df %>%
    mutate(setting_col = setting_column_label(split_type, label_set)) %>%
    group_by(label_set, split_type, setting_col, requested_target_risk) %>%
    summarise(
      scenario_key = dplyr::first(scenario_key),
      scenario_name = dplyr::first(scenario_name),
      requested_target_risk_pct = 100 * dplyr::first(requested_target_risk),
      n_outer_folds = n(),
      n_ood = sum(n_ood, na.rm = TRUE),
      n_rejected_ood = sum(n_rejected_ood, na.rm = TRUE),
      n_incorrect_seen = sum(n_incorrect_seen, na.rm = TRUE),
      n_rejected_incorrect_seen = sum(n_rejected_incorrect_seen, na.rm = TRUE),
      n_correct_seen = sum(n_correct_seen, na.rm = TRUE),
      n_rejected_correct_seen = sum(n_rejected_correct_seen, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      pct_rejected_ood = pct_rejected(n_rejected_ood, n_ood),
      pct_rejected_incorrect_seen = pct_rejected(n_rejected_incorrect_seen, n_incorrect_seen),
      pct_rejected_correct_seen = pct_rejected(n_rejected_correct_seen, n_correct_seen)
    ) %>%
    arrange(label_set, split_type)
}

# LOSO only: unweighted mean of rejection % across full_subtypes and collapsed_classes.
summarize_rejection_stratum_loso_labels_averaged <- function(summary_df) {
  loso <- summary_df %>% filter(split_type == "loso")
  if (nrow(loso) != 2L) return(data.frame())
  loso %>%
    summarise(
      split_type = "loso",
      scenario_key = dplyr::first(scenario_key),
      scenario_name = dplyr::first(scenario_name),
      requested_target_risk = dplyr::first(requested_target_risk),
      requested_target_risk_pct = dplyr::first(requested_target_risk_pct),
      n_outer_folds = sum(n_outer_folds, na.rm = TRUE),
      n_label_sets_averaged = n(),
      pct_rejected_ood = mean(pct_rejected_ood, na.rm = TRUE),
      pct_rejected_incorrect_seen = mean(pct_rejected_incorrect_seen, na.rm = TRUE),
      pct_rejected_correct_seen = mean(pct_rejected_correct_seen, na.rm = TRUE),
      .groups = "drop"
    )
}

summarize_rejection_stratum_pooled <- function(per_fold_df) {
  if (nrow(per_fold_df) == 0L) return(data.frame())
  data.frame(
    scenario_key = per_fold_df$scenario_key[[1]],
    scenario_name = per_fold_df$scenario_name[[1]],
    requested_target_risk = per_fold_df$requested_target_risk[[1]],
    requested_target_risk_pct = 100 * per_fold_df$requested_target_risk[[1]],
    n_outer_folds = nrow(per_fold_df),
    n_ood = sum(per_fold_df$n_ood, na.rm = TRUE),
    n_rejected_ood = sum(per_fold_df$n_rejected_ood, na.rm = TRUE),
    n_incorrect_seen = sum(per_fold_df$n_incorrect_seen, na.rm = TRUE),
    n_rejected_incorrect_seen = sum(per_fold_df$n_rejected_incorrect_seen, na.rm = TRUE),
    n_correct_seen = sum(per_fold_df$n_correct_seen, na.rm = TRUE),
    n_rejected_correct_seen = sum(per_fold_df$n_rejected_correct_seen, na.rm = TRUE),
    stringsAsFactors = FALSE
  ) %>%
    mutate(
      pct_rejected_ood = pct_rejected(n_rejected_ood, n_ood),
      pct_rejected_incorrect_seen = pct_rejected(n_rejected_incorrect_seen, n_incorrect_seen),
      pct_rejected_correct_seen = pct_rejected(n_rejected_correct_seen, n_correct_seen)
    )
}

# Count RHS terms other than max_prob in one semicolon-separated key.
count_optional_rhs_str <- function(s) {
  if (length(s) != 1L) return(0L)
  x <- as.character(s)[[1]]
  if (is.na(x) || !nzchar(x)) return(0L)
  parts <- strsplit(x, ";", fixed = TRUE)[[1]]
  as.integer(sum(parts != "max_prob" & nzchar(parts)))
}

recipe_optional_count_df <- function(dat) {
  vapply(dat$rhs_key, count_optional_rhs_str, integer(1))
}

# Mark recipes tied on risk stability: sd_risk <= min(sd_risk in group) + SD_RISK_TIE_EPS.
flag_sd_risk_shortlist <- function(d, eps = SD_RISK_TIE_EPS) {
  if (nrow(d) == 0L) return(d)
  sd_star <- min(d$sd_risk, na.rm = TRUE)
  d %>% mutate(sd_risk_near_min = as.integer(sd_risk <= sd_star + eps))
}

# Sort keys after risk band: near-min sd_risk tier, then coverage, then finer sd_risk / complexity ties.
arrange_inner_rank_keys <- function(d, dist_first = FALSE) {
  if (dist_first) {
    d %>% arrange(
      .data$dist_to_target,
      desc(.data$sd_risk_near_min),
      desc(.data$mean_coverage),
      .data$sd_risk,
      .data$recipe_optional_count,
      .data$rhs_sort
    )
  } else {
    d %>% arrange(
      desc(.data$sd_risk_near_min),
      desc(.data$mean_coverage),
      .data$sd_risk,
      .data$recipe_optional_count,
      .data$rhs_sort
    )
  }
}

# Inner winner: mean risk in [target - low_offset, target] (one-sided, at or below target), else fallback by |mean-target|.
rank_inner_scores <- function(
  df,
  risk_target = TARGET_RISK,
  low_offset = INNER_RISK_BAND_LOW_OFFSET,
  sd_risk_eps = SD_RISK_TIE_EPS
) {
  if (nrow(df) == 0L) return(df)
  df <- df %>% mutate(.rid = row_number())
  ok_mask <- with(df, is.finite(mean_risk) & is.finite(sd_risk) & is.finite(mean_coverage) & is.finite(sd_coverage))
  ok <- df[ok_mask, , drop = FALSE]
  if (nrow(ok) == 0L) {
    return(df %>% mutate(inner_rank = NA_integer_, inner_selection_tier = NA_character_, recipe_optional_count = NA_integer_) %>% select(-.rid))
  }
  ok <- ok %>%
    mutate(
      recipe_optional_count = recipe_optional_count_df(ok),
      dist_to_target = abs(mean_risk - risk_target)
    )
  band_low <- risk_target - low_offset
  band_high <- risk_target
  in_band <- ok %>% filter(mean_risk >= band_low, mean_risk <= band_high)
  if (nrow(in_band) > 0L) {
    tier <- "target_band"
    in_p <- in_band
  } else {
    tier <- "fallback"
    in_p <- ok
  }
  out_p <- if (tier == "fallback") {
    ok[0L, , drop = FALSE]
  } else {
    dplyr::anti_join(ok, in_p, by = ".rid")
  }

  add_sort_cols <- function(d) {
    d %>% mutate(rhs_sort = dplyr::coalesce(rhs_key, ""))
  }
  use_dist_first <- (tier == "fallback")
  rin <- in_p %>%
    mutate(inner_selection_tier = tier) %>%
    flag_sd_risk_shortlist(eps = sd_risk_eps) %>%
    add_sort_cols()
  rin <- rin %>%
    arrange_inner_rank_keys(dist_first = use_dist_first) %>%
    select(-rhs_sort, -sd_risk_near_min)

  rout <- if (nrow(out_p) == 0L) {
    out_p
  } else {
    out_p %>%
      mutate(inner_selection_tier = "outside_band") %>%
      flag_sd_risk_shortlist(eps = sd_risk_eps) %>%
      add_sort_cols() %>%
      arrange_inner_rank_keys(dist_first = TRUE) %>%
      select(-rhs_sort, -sd_risk_near_min)
  }

  merged <- bind_rows(rin, rout) %>% mutate(inner_rank = row_number())
  out_full <- df %>%
    left_join(
      merged %>% select(.rid, inner_rank, inner_selection_tier, recipe_optional_count, dist_to_target),
      by = ".rid"
    ) %>%
    mutate(
      inner_selection_tier = dplyr::case_when(
        !ok_mask ~ "non_finite_metrics",
        is.na(inner_selection_tier) ~ "non_finite_metrics",
        TRUE ~ inner_selection_tier
      )
    ) %>%
    select(-.rid)
  out_full
}

# Safe for NA / empty keys from per-fold winner rows.
extras_from_rhs_key <- function(key, baseline_terms = BASELINE_TERMS) {
  if (length(key) != 1L || is.na(key) || !nzchar(as.character(key))) return(character(0))
  setdiff(strsplit(as.character(key), ";", fixed = TRUE)[[1]], baseline_terms)
}

# One outer fold: inner CV scores each recipe at 3%, 5%, 10%; fuse ranks (sum); outer eval at risk_target.
evaluate_outer_fold_joint <- function(pool_fold_dfs, target_df, split_type, label_set, fold_name, risk_target) {
  empty_pf <- data.frame()
  empty_in <- data.frame()
  pool_all <- bind_rows(pool_fold_dfs)
  cand_all <- candidate_terms_from_df(pool_all)
  if (!"max_prob" %in% cand_all || !"accept_combined" %in% colnames(pool_all)) {
    return(list(
      per_fold = empty_pf,
      inner_scores_ranked = empty_in,
      recipe_job_stub = NULL
    ))
  }
  rhs_list <- build_rhs_subsets(cand_all, BASELINE_TERMS)

  scores_by_anchor <- vector("list", length(INNER_SELECTION_ANCHOR_RISKS))
  ranked_by_anchor <- vector("list", length(INNER_SELECTION_ANCHOR_RISKS))
  names(scores_by_anchor) <- INNER_SELECTION_ANCHOR_LABELS
  names(ranked_by_anchor) <- INNER_SELECTION_ANCHOR_LABELS

  for (ki in seq_along(INNER_SELECTION_ANCHOR_RISKS)) {
    anchor_rt <- INNER_SELECTION_ANCHOR_RISKS[[ki]]
    lbl <- INNER_SELECTION_ANCHOR_LABELS[[ki]]
    score_rows <- list()
    si <- 1L
    for (rhs in rhs_list) {
      inn <- inner_cv_strict_singlehead(
        pool_fold_dfs, "accept_combined", rhs, POOL_RULE, TEST_RULE, anchor_rt
      )
      if (!isTRUE(inn$ok)) next
      score_rows[[si]] <- data.frame(
        scenario_key = SCENARIO_KEY,
        rhs_key = paste(rhs, collapse = ";"),
        mean_coverage = inn$mean_coverage,
        mean_risk = inn$mean_risk,
        median_coverage = inn$median_coverage,
        median_risk = inn$median_risk,
        sd_coverage = inn$sd_coverage,
        sd_risk = inn$sd_risk,
        stringsAsFactors = FALSE
      )
      si <- si + 1L
    }
    if (length(score_rows) == 0L) {
      return(list(
        per_fold = empty_pf,
        inner_scores_ranked = empty_in,
        recipe_job_stub = NULL
      ))
    }
    sdf <- bind_rows(score_rows)
    scores_by_anchor[[lbl]] <- sdf
    ranked_by_anchor[[lbl]] <- rank_inner_scores(sdf, anchor_rt)
  }

  fusion <- ranked_by_anchor[["p03"]] %>%
    dplyr::select(rhs_key, inner_rank) %>%
    dplyr::rename(inner_rank_p03 = inner_rank) %>%
    dplyr::inner_join(
      ranked_by_anchor[["p05"]] %>%
        dplyr::select(rhs_key, inner_rank) %>%
        dplyr::rename(inner_rank_p05 = inner_rank),
      by = "rhs_key"
    ) %>%
    dplyr::inner_join(
      ranked_by_anchor[["p10"]] %>%
        dplyr::select(rhs_key, inner_rank) %>%
        dplyr::rename(inner_rank_p10 = inner_rank),
      by = "rhs_key"
    )

  if (nrow(fusion) == 0L) {
    return(list(
      per_fold = empty_pf,
      inner_scores_ranked = empty_in,
      recipe_job_stub = NULL
    ))
  }

  fusion <- fusion %>%
    dplyr::mutate(
      fusion_rank_sum = inner_rank_p03 + inner_rank_p05 + inner_rank_p10,
      fusion_rank_max = pmax(inner_rank_p03, inner_rank_p05, inner_rank_p10)
    )

  enriched <- ranked_by_anchor[["p05"]] %>%
    dplyr::inner_join(
      fusion %>%
        dplyr::select(rhs_key, inner_rank_p03, inner_rank_p10, fusion_rank_sum, fusion_rank_max),
      by = "rhs_key"
    ) %>%
    dplyr::rename(inner_rank_p05 = inner_rank) %>%
    dplyr::arrange(
      fusion_rank_sum,
      fusion_rank_max,
      inner_rank_p05,
      inner_rank_p03,
      inner_rank_p10,
      rhs_key
    ) %>%
    dplyr::mutate(inner_rank = dplyr::row_number())

  inner_scores_ranked <- enriched %>%
    dplyr::mutate(
      label_set = label_set,
      split_type = split_type,
      target_fold = as.character(fold_name),
      scenario_name = SCENARIO_NAME,
      inner_selection_fusion_rule = "sum_ranks_at_3_5_10pct_then_maxrank_then_p05",
      .before = 1L
    )

  win <- enriched %>% dplyr::filter(inner_rank == 1L)
  if (nrow(win) != 1L) {
    stop(sprintf("Fusion inner winner not unique (inner_rank==1): n=%d", nrow(win)))
  }
  rhs_terms <- strsplit(win$rhs_key, ";", fixed = TRUE)[[1]]
  # Re-use this stub for calibration-curve points: same recipe, threshold re-fit per requested risk.
  recipe_job_stub <- list(
    pool_fold_dfs = pool_fold_dfs,
    target_df = target_df,
    split_type = split_type,
    label_set = label_set,
    fold_name = fold_name,
    inner_winner_rhs_key = as.character(win$rhs_key)
  )
  out <- outer_eval_singlehead(
    pool_fold_dfs, target_df, "accept_combined", rhs_terms,
    POOL_RULE, TEST_RULE, risk_target
  )
  if (is.null(out) || !isTRUE(out$ok)) {
    return(list(
      per_fold = empty_pf,
      inner_scores_ranked = inner_scores_ranked,
      recipe_job_stub = recipe_job_stub
    ))
  }

  feat_union <- extras_from_rhs_key(win$rhs_key)
  # Human-readable optional RHS terms for this fold only (baseline max_prob omitted).
  recipe_human <- if (length(feat_union) == 0L) "max_prob (baseline only)" else paste(feat_union, collapse = ";")
  per_fold <- data.frame(
    label_set = label_set,
    split_type = split_type,
    target_fold = as.character(fold_name),
    scenario_key = SCENARIO_KEY,
    scenario_name = SCENARIO_NAME,
    inner_mean_coverage_seen = win$mean_coverage,
    inner_mean_risk_all_accepted = win$mean_risk,
    inner_median_coverage_seen = win$median_coverage,
    inner_median_risk_all_accepted = win$median_risk,
    inner_sd_coverage_seen = win$sd_coverage,
    inner_sd_risk_all_accepted = win$sd_risk,
    inner_winner_inner_rank_p03 = win$inner_rank_p03,
    inner_winner_inner_rank_p05 = win$inner_rank_p05,
    inner_winner_inner_rank_p10 = win$inner_rank_p10,
    inner_winner_fusion_rank_sum = win$fusion_rank_sum,
    inner_winner_rhs_key = recipe_job_stub$inner_winner_rhs_key,
    inner_winner_optional_features = recipe_human,
    outer_n_cal_rotations = length(names(pool_fold_dfs)),
    threshold_outer_cal_mean = out$threshold,
    threshold_outer_cal_median = out$threshold_median,
    outer_risk_all_accepted = out$risk_all_accepted,
    outer_risk_all_accepted_median = out$risk_all_accepted_median,
    outer_coverage_seen = out$coverage_seen,
    outer_coverage_seen_median = out$coverage_seen_median,
    outer_kappa_accepted = out$kappa_accepted,
    outer_kappa_accepted_median = out$kappa_accepted_median,
    outer_auprc = out$auprc_outer,
    outer_auprc_median = out$auprc_outer_median,
    base_model = TARGET_BASE_MODEL,
    stringsAsFactors = FALSE
  )

  list(
    per_fold = per_fold,
    inner_scores_ranked = inner_scores_ranked,
    recipe_job_stub = recipe_job_stub
  )
}

# One outer-fold job for parallel::mclapply (fork on macOS/Linux).
worker_evaluate_outer_fold <- function(job) {
  evaluate_outer_fold_joint(
    pool_fold_dfs = job$pool_fold_dfs,
    target_df = job$target_df,
    split_type = job$split_type,
    label_set = job$label_set,
    fold_name = job$fold_name,
    risk_target = job$risk_target
  )
}

build_feature_subsets <- function(optional_terms) {
  subsets <- list(character(0))
  if (length(optional_terms) == 0) return(subsets)
  idx <- 2L
  for (k in seq_len(length(optional_terms))) {
    cmb <- combn(optional_terms, k, simplify = FALSE)
    for (s in cmb) {
      subsets[[idx]] <- s
      idx <- idx + 1L
    }
  }
  subsets
}

build_rhs_subsets <- function(candidate_terms, baseline_terms = BASELINE_TERMS) {
  baseline_terms <- unique(baseline_terms[baseline_terms %in% candidate_terms])
  if (length(baseline_terms) == 0) {
    stop(sprintf(
      "Baseline terms missing from candidate terms. baseline='%s' candidate='%s'",
      paste(baseline_terms, collapse = ","),
      paste(candidate_terms, collapse = ",")
    ))
  }
  optional_terms <- setdiff(unique(candidate_terms), baseline_terms)
  subsets <- build_feature_subsets(optional_terms)
  lapply(subsets, function(s) c(baseline_terms, s))
}

subset_key_from_terms <- function(rhs_terms, baseline_terms = BASELINE_TERMS) {
  paste(setdiff(rhs_terms, baseline_terms), collapse = ";")
}

ood_aware_results_available <- function(results_obj) {
  mr <- results_obj$multivariate_results
  if (is.null(mr)) return(FALSE)
  fam <- mr[[SCENARIO_KEY]]
  !is.null(fam) && TARGET_BASE_MODEL %in% names(fam)
}

# Strict nested CV: inner winner; threshold from LOSO-OOF (inner train pool + outer pool).
# Requires >=4 primary outer folds so the pool (all but target) has >=3 studies (>=2 for OOF + 1 val).
run_nested_target_risk_analysis <- function(results_obj, label_set, risk_target) {
  empty_out <- list(
    per_fold_df = data.frame(),
    summary_4 = data.frame(),
    heatmap_long = data.frame(),
    inner_scores_ranked = data.frame(),
    recipe_jobs = list()
  )
  if (!ood_aware_results_available(results_obj)) {
    return(empty_out)
  }
  fam0 <- results_obj$multivariate_results[[SCENARIO_KEY]]
  jobs <- list()
  j_idx <- 1L
  for (split_type in c("cv", "loso")) {
    if (!split_type %in% names(fam0[[TARGET_BASE_MODEL]])) next
    model_node <- fam0[[TARGET_BASE_MODEL]]
    bundle <- model_node[[split_type]]
    if (is.null(bundle) || is.null(bundle$fold_matrices) || length(bundle$fold_matrices) < 4L) next
    fold_feats <- lapply(bundle$fold_matrices, extract_features)
    fold_names <- names(fold_feats)
    cat(sprintf(
      "    [%s] %s (%s): inner winner + outer over %d outer folds (>=4 folds; threshold via pool LOSO-OOF)\n",
      label_set, toupper(split_type), SCENARIO_KEY, length(fold_names)
    ))
    for (fold_name in fold_names) {
      jobs[[j_idx]] <- list(
        pool_fold_dfs = fold_feats[setdiff(fold_names, fold_name)],
        target_df = fold_feats[[fold_name]],
        split_type = split_type,
        label_set = label_set,
        fold_name = fold_name,
        risk_target = risk_target
      )
      j_idx <- j_idx + 1L
    }
  }

  if (length(jobs) == 0L) {
    return(empty_out)
  }

  cat(sprintf(
    "    [%s] Dispatching %d outer-fold jobs (mc.cores=%d)\n",
    label_set, length(jobs), PARALLEL_MC_CORES
  ))

  ev_list <- if (PARALLEL_MC_CORES > 1L && length(jobs) > 1L) {
    parallel::mclapply(
      jobs,
      worker_evaluate_outer_fold,
      mc.cores = PARALLEL_MC_CORES,
      mc.preschedule = FALSE
    )
  } else {
    lapply(jobs, worker_evaluate_outer_fold)
  }

  per_fold_rows <- list()
  r_idx <- 1L
  inner_fold_rows <- list()
  ir2 <- 1L
  recipe_jobs <- list()
  rj <- 1L
  for (k in seq_along(ev_list)) {
    ev <- ev_list[[k]]
    if (nrow(ev$per_fold) > 0L) {
      per_fold_rows[[r_idx]] <- ev$per_fold
      r_idx <- r_idx + 1L
    }
    if (nrow(ev$inner_scores_ranked) > 0L) {
      inner_fold_rows[[ir2]] <- ev$inner_scores_ranked
      ir2 <- ir2 + 1L
    }
    if (!is.null(ev$recipe_job_stub)) {
      recipe_jobs[[rj]] <- ev$recipe_job_stub
      rj <- rj + 1L
    }
  }
  per_fold_df <- if (length(per_fold_rows) == 0L) data.frame() else bind_rows(per_fold_rows)
  inner_scores_ranked_df <- if (length(inner_fold_rows) == 0L) data.frame() else bind_rows(inner_fold_rows)
  summary_4 <- summarize_four_settings(per_fold_df)
  heatmap_long <- build_feature_heatmap_long(per_fold_df)
  list(
    per_fold_df = per_fold_df,
    summary_4 = summary_4,
    heatmap_long = heatmap_long,
    inner_scores_ranked = inner_scores_ranked_df,
    recipe_jobs = recipe_jobs
  )
}

# dplyr mutate passes whole columns; use vectorized logic (not scalar if/else).
setting_column_label <- function(split_type, label_set) {
  lab_chr <- as.character(label_set)
  ls_short <- case_when(
    lab_chr == "full_subtypes" ~ "Full",
    lab_chr == "collapsed_classes" ~ "Merged",
    TRUE ~ lab_chr
  )
  sprintf("%s | %s", toupper(as.character(split_type)), ls_short)
}

summarize_four_settings <- function(per_fold_df) {
  if (nrow(per_fold_df) == 0L) return(data.frame())
  modal_chr <- function(x) {
    x <- unique(as.character(x))
    x <- x[!is.na(x) & nzchar(x)]
    if (length(x) == 0L) return(NA_character_)
    tab <- sort(table(x), decreasing = TRUE)
    names(tab)[[1]]
  }
  per_fold_df %>%
    mutate(
      setting_col = setting_column_label(split_type, label_set)
    ) %>%
    group_by(label_set, split_type, setting_col) %>%
    summarise(
      n_outer_folds = n(),
      scenario_key = dplyr::first(scenario_key),
      scenario_name = dplyr::first(scenario_name),
      # Mode of the exact inner-winning optional-feature recipe across outer folds (not a feature union).
      modal_inner_winner_recipe = modal_chr(inner_winner_optional_features),
      mean_outer_coverage_seen = mean(outer_coverage_seen, na.rm = TRUE),
      sd_outer_coverage_seen = stats::sd(outer_coverage_seen, na.rm = TRUE),
      mean_outer_risk_all_accepted = mean(outer_risk_all_accepted, na.rm = TRUE),
      sd_outer_risk_all_accepted = stats::sd(outer_risk_all_accepted, na.rm = TRUE),
      mean_outer_kappa_accepted = mean(outer_kappa_accepted, na.rm = TRUE),
      sd_outer_kappa_accepted = stats::sd(outer_kappa_accepted, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(label_set, split_type)
}

build_feature_heatmap_long <- function(per_fold_df) {
  opt_feats <- setdiff(ALL_FEATURE_TERMS, BASELINE_TERMS)
  empty <- data.frame(
    setting_col = character(),
    label_set = character(),
    split_type = character(),
    feature = character(),
    frac_outer_folds_selected = numeric(),
    stringsAsFactors = FALSE
  )
  if (nrow(per_fold_df) == 0L) return(empty)
  per_fold_df %>%
    group_by(label_set, split_type) %>%
    group_modify(function(.g, .k) {
      n_f <- nrow(.g)
      data.frame(
        feature = opt_feats,
        frac_outer_folds_selected = vapply(opt_feats, function(f) {
          sum(vapply(seq_len(n_f), function(i) {
            sp <- strsplit(.g$inner_winner_optional_features[[i]], ";", fixed = TRUE)[[1]]
            f %in% sp
          }, logical(1))) / n_f
        }, numeric(1)),
        stringsAsFactors = FALSE
      )
    }) %>%
    ungroup() %>%
    mutate(setting_col = setting_column_label(split_type, label_set)) %>%
    select(setting_col, label_set, split_type, feature, frac_outer_folds_selected)
}

# Mean realized outer-fold metrics vs requested risk; SE/CIs across outer folds (same spirit as deployable curves in analyse_results.Rmd).
summarize_calibration_curve_metrics <- function(per_fold_df, risk_target) {
  if (nrow(per_fold_df) == 0L) return(data.frame())
  per_fold_df %>%
    mutate(setting_col = setting_column_label(split_type, label_set)) %>%
    group_by(label_set, split_type, setting_col) %>%
    summarise(
      requested_target_risk_pct = 100 * risk_target,
      n_outer_folds = n(),
      risk_mean = mean(outer_risk_all_accepted, na.rm = TRUE),
      risk_sd = if (n() > 1L) stats::sd(outer_risk_all_accepted, na.rm = TRUE) else NA_real_,
      cov_mean = mean(outer_coverage_seen, na.rm = TRUE),
      cov_sd = if (n() > 1L) stats::sd(outer_coverage_seen, na.rm = TRUE) else NA_real_,
      .groups = "drop"
    ) %>%
    mutate(
      realized_risk_pct = 100 * risk_mean,
      realized_risk_se_pct = if_else(
        n_outer_folds > 0L & is.finite(risk_sd),
        (100 * risk_sd) / sqrt(as.numeric(n_outer_folds)),
        NA_real_
      ),
      realized_risk_ci95_lo_pct = pmax(0, realized_risk_pct - 1.96 * realized_risk_se_pct),
      realized_risk_ci95_hi_pct = pmin(100, realized_risk_pct + 1.96 * realized_risk_se_pct),
      realized_coverage_seen_pct = 100 * cov_mean,
      realized_coverage_seen_se_pct = if_else(
        n_outer_folds > 0L & is.finite(cov_sd),
        (100 * cov_sd) / sqrt(as.numeric(n_outer_folds)),
        NA_real_
      ),
      realized_coverage_seen_ci95_lo_pct = pmax(0, realized_coverage_seen_pct - 1.96 * realized_coverage_seen_se_pct),
      realized_coverage_seen_ci95_hi_pct = pmin(100, realized_coverage_seen_pct + 1.96 * realized_coverage_seen_se_pct)
    ) %>%
    select(
      label_set, split_type, setting_col,
      requested_target_risk_pct, n_outer_folds,
      realized_risk_pct, realized_risk_ci95_lo_pct, realized_risk_ci95_hi_pct,
      realized_coverage_seen_pct, realized_coverage_seen_ci95_lo_pct, realized_coverage_seen_ci95_hi_pct
    )
}

# Threshold from pool LOSO-OOF at risk_target; RHS fixed to primary inner winner (semicolon key includes baseline terms).
run_fixed_recipe_outer_eval <- function(stub, risk_target) {
  rhs_terms <- strsplit(stub$inner_winner_rhs_key, ";", fixed = TRUE)[[1]]
  out <- outer_eval_singlehead(
    stub$pool_fold_dfs, stub$target_df, "accept_combined", rhs_terms,
    POOL_RULE, TEST_RULE, risk_target
  )
  if (is.null(out) || !isTRUE(out$ok)) return(NULL)
  data.frame(
    label_set = stub$label_set,
    split_type = stub$split_type,
    target_fold = as.character(stub$fold_name),
    outer_risk_all_accepted = out$risk_all_accepted,
    outer_coverage_seen = out$coverage_seen,
    outer_kappa_accepted = out$kappa_accepted,
    stringsAsFactors = FALSE
  )
}

# Curve rows: same inner-winning RHS as primary risk; only the outer threshold is refit per requested risk.
build_calibration_curve_from_stubs <- function(recipe_jobs, risk_targets) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  chunks <- list()
  ci <- 1L
  for (tr in risk_targets) {
    rows <- lapply(recipe_jobs, function(stub) run_fixed_recipe_outer_eval(stub, tr))
    rows <- rows[!vapply(rows, is.null, logical(1))]
    if (length(rows) == 0L) next
    pf <- bind_rows(rows)
    chunks[[ci]] <- summarize_calibration_curve_metrics(pf, tr)
    ci <- ci + 1L
  }
  if (ci == 1L) data.frame() else bind_rows(chunks)
}

# Clone outer-fold stubs with a fixed RHS (e.g. max_prob-only baseline).
stubs_with_fixed_rhs <- function(recipe_jobs, rhs_key) {
  if (length(recipe_jobs) == 0L) return(list())
  lapply(recipe_jobs, function(stub) {
    stub$inner_winner_rhs_key <- rhs_key
    stub
  })
}

# Full-coverage baseline: inner-winning multivariate recipe, accept all (threshold = 0).
evaluate_full_coverage_from_stub <- function(stub) {
  rhs_terms <- strsplit(stub$inner_winner_rhs_key, ";", fixed = TRUE)[[1]]
  out <- outer_eval_singlehead(
    stub$pool_fold_dfs, stub$target_df, "accept_combined", rhs_terms,
    POOL_RULE, TEST_RULE, risk_target = NULL, fixed_threshold = FULL_COVERAGE_THRESHOLD
  )
  if (is.null(out) || !isTRUE(out$ok)) return(NULL)
  data.frame(
    label_set = stub$label_set,
    split_type = stub$split_type,
    target_fold = as.character(stub$fold_name),
    scenario_key = SCENARIO_KEY,
    inner_winner_rhs_key = stub$inner_winner_rhs_key,
    outer_risk_all_accepted = out$risk_all_accepted,
    outer_coverage_seen = out$coverage_seen,
    outer_kappa_accepted = out$kappa_accepted,
    stringsAsFactors = FALSE
  )
}

# Classifier-only baseline (no rejector): error rate and kappa on all target-fold samples.
evaluate_classifier_only_full_coverage <- function(target_df) {
  tgt_u <- apply_row_rule(target_df, TEST_RULE)
  if (nrow(tgt_u) < 10L || !all(c("true_class", "pred_class", "correct") %in% colnames(tgt_u))) {
    return(NULL)
  }
  risk_all_accepted <- mean(as.integer(tgt_u$correct == 0L))
  lvls <- sort(unique(c(as.character(tgt_u$true_class), as.character(tgt_u$pred_class))))
  kappa_acc <- if (length(lvls) < 2L) {
    NA_real_
  } else {
    fast_kappa(
      factor(tgt_u$pred_class, levels = lvls),
      factor(tgt_u$true_class, levels = lvls)
    )
  }
  data.frame(
    outer_risk_all_accepted = risk_all_accepted,
    outer_coverage_seen = if ("is_seen" %in% colnames(tgt_u)) {
      mean(as.integer(tgt_u$is_seen > 0))
    } else {
      1
    },
    outer_kappa_accepted = kappa_acc,
    stringsAsFactors = FALSE
  )
}

# Outer-fold metrics at primary risk with max_prob-only RHS (no inner feature selection).
build_max_prob_per_fold_primary <- function(
  recipe_jobs,
  risk_target = PRIMARY_TABLE_TARGET_RISK
) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  rows <- lapply(recipe_jobs, function(stub) {
    stub_mp <- stubs_with_fixed_rhs(list(stub), BASELINE_ONLY_RHS_KEY)[[1]]
    ev <- run_fixed_recipe_outer_eval(stub_mp, risk_target)
    if (is.null(ev)) return(NULL)
    data.frame(
      label_set = stub$label_set,
      split_type = stub$split_type,
      target_fold = ev$target_fold,
      scenario_key = SCENARIO_KEY,
      scenario_name = SCENARIO_NAME,
      inner_winner_optional_features = "max_prob (baseline only)",
      inner_winner_rhs_key = BASELINE_ONLY_RHS_KEY,
      outer_risk_all_accepted = ev$outer_risk_all_accepted,
      outer_coverage_seen = ev$outer_coverage_seen,
      outer_kappa_accepted = ev$outer_kappa_accepted,
      stringsAsFactors = FALSE
    )
  })
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) == 0L) data.frame() else bind_rows(rows)
}

# Inner-best vs max_prob-only means across outer folds (for risk–coverage comparison plots).
build_calibration_compare_curves <- function(recipe_jobs, risk_targets) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  inner_best <- build_calibration_curve_from_stubs(recipe_jobs, risk_targets) %>%
    mutate(calibration_recipe = CALIBRATION_RECIPE_INNER_BEST)
  max_prob_only <- build_calibration_curve_from_stubs(
    stubs_with_fixed_rhs(recipe_jobs, BASELINE_ONLY_RHS_KEY),
    risk_targets
  ) %>%
    mutate(calibration_recipe = CALIBRATION_RECIPE_MAX_PROB)
  bind_rows(inner_best, max_prob_only) %>%
    arrange(label_set, split_type, calibration_recipe, requested_target_risk_pct)
}

# Per outer fold: realized risk/coverage vs requested target (no cross-fold aggregation).
build_calibration_curve_per_fold_from_stubs <- function(recipe_jobs, risk_targets) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  chunks <- list()
  ci <- 1L
  for (tr in risk_targets) {
    rows <- lapply(recipe_jobs, function(stub) run_fixed_recipe_outer_eval(stub, tr))
    rows <- rows[!vapply(rows, is.null, logical(1))]
    if (length(rows) == 0L) next
    chunks[[ci]] <- bind_rows(rows) %>%
      mutate(
        setting_col = setting_column_label(split_type, label_set),
        requested_target_risk_pct = 100 * tr,
        realized_risk_pct = 100 * outer_risk_all_accepted,
        realized_coverage_seen_pct = 100 * outer_coverage_seen
      ) %>%
      select(
        label_set, split_type, setting_col, target_fold,
        requested_target_risk_pct, realized_risk_pct, realized_coverage_seen_pct
      )
    ci <- ci + 1L
  }
  if (ci == 1L) {
    data.frame()
  } else {
    bind_rows(chunks) %>%
      arrange(label_set, split_type, target_fold, requested_target_risk_pct)
  }
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

