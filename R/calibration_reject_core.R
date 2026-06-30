# Shared nested target-risk calibration helpers (sourced by calibration_reject_models*.R).
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

# --- Elastic-net rejector (all features; alpha grid; lambda.min via study-blocked cv.glmnet) ---
enet_alpha_grid_values <- function() {
  if (exists("ENET_ALPHA_GRID", inherits = TRUE)) {
    as.numeric(get("ENET_ALPHA_GRID", inherits = TRUE))
  } else {
    c(0, 0.25, 0.5, 0.75, 1)
  }
}

enet_nfolds <- function(n) {
  as.integer(min(10L, max(3L, n %/% 5L)))
}

glmnet_maxit <- function() {
  if (exists("GLMNET_MAXIT", inherits = TRUE)) {
    as.integer(get("GLMNET_MAXIT", inherits = TRUE))
  } else {
    1000000L
  }
}

glmnet_control <- function() {
  glmnet::glmnet.control(maxit = glmnet_maxit())
}

ENET_STUDY_FOLDID_COL <- "._study_fold_id"
ENET_MIN_FEATURE_SD <- 1e-6
ENET_MIN_ABS_COEF <- 1e-6
ENET_EXPORT_PARITY_TOL <- 1e-5
ELASTICNET_TWO_HEAD_PARAMS_KEY <- "two_head"

# Shared two-head elastic-net fit cache (keyed by label_set + merge_suffix).
.enet_deploy_fit_cache <- new.env(parent = emptyenv())

reset_enet_deploy_fit_cache <- function() {
  rm(list = ls(envir = .enet_deploy_fit_cache), envir = .enet_deploy_fit_cache)
}

enet_params_export_key <- function(rejector_mode) {
  if (is_two_head_rejector(rejector_mode)) {
    return(ELASTICNET_TWO_HEAD_PARAMS_KEY)
  }
  as.character(rejector_mode)
}

# Drop features with near-zero variance on a head's training matrix.
enet_keep_feature_terms <- function(
  train_df, feature_terms, y_col = NULL, min_sd = ENET_MIN_FEATURE_SD
) {
  feature_terms <- feature_terms[feature_terms %in% colnames(train_df)]
  if (length(feature_terms) == 0L) return(character(0))
  des <- prep_enet_design(train_df, feature_terms, y_col)
  if (length(des$row_id) == 0L) return(character(0))
  sds <- apply(des$x, 2, stats::sd)
  names(sds) <- colnames(des$x)
  kept <- names(sds)[is.finite(sds) & sds >= min_sd]
  kept
}

# Raw-scale glmnet coefficients (glmnet 5.x: a0/beta apply to unscaled x).
extract_glmnet_raw_coef <- function(glmnet_obj, lambda) {
  if (is.null(glmnet_obj) || is.null(glmnet_obj$lambda)) return(NULL)
  idx <- which.min(abs(glmnet_obj$lambda - lambda))
  if (length(idx) != 1L || !is.finite(idx)) return(NULL)
  beta <- as.numeric(glmnet_obj$beta[, idx])
  list(
    intercept = as.numeric(glmnet_obj$a0[idx]),
    beta = beta,
    feature_terms = rownames(glmnet_obj$beta)
  )
}

extract_enet_head_raw_coef <- function(fit) {
  if (is.null(fit) || is.null(fit$cv_fit)) return(NULL)
  glmnet_obj <- fit$cv_fit
  if (!is.null(glmnet_obj$glmnet.fit)) {
    glmnet_obj <- glmnet_obj$glmnet.fit
  }
  raw <- extract_glmnet_raw_coef(glmnet_obj, fit$lambda)
  if (is.null(raw)) return(NULL)
  if (length(raw$feature_terms) == 0L || is.null(raw$feature_terms)) {
    raw$feature_terms <- fit$feature_terms
    names(raw$beta) <- fit$feature_terms
  }
  raw
}

# Reject null / all-zero exports only; glmnet may use large coefs at lambda.min.
enet_has_usable_coefs <- function(fit_obj) {
  if (is.null(fit_obj) || is.null(fit_obj$cv_fit)) return(FALSE)
  raw <- extract_enet_head_raw_coef(fit_obj)
  if (is.null(raw)) return(FALSE)
  if (!is.finite(raw$intercept)) return(FALSE)
  vals <- as.numeric(raw$beta)
  if (length(vals) == 0L) return(FALSE)
  if (!any(is.finite(vals) & abs(vals) >= ENET_MIN_ABS_COEF)) return(FALSE)
  TRUE
}

# Score one head using exported coef CSV layout (mirrors python/enet_rejector_scoring.py).
score_enet_head_from_export <- function(test_df, head_params_df) {
  if (is.null(head_params_df) || nrow(head_params_df) == 0L || nrow(test_df) == 0L) {
    return(numeric(0))
  }
  intercept <- as.numeric(head_params_df$estimate[head_params_df$term == "(Intercept)"])
  if (length(intercept) != 1L || !is.finite(intercept)) {
    stop("Exported elastic-net head missing finite (Intercept).")
  }
  feat_rows <- head_params_df[head_params_df$term != "(Intercept)", , drop = FALSE]
  linear <- rep(intercept, nrow(test_df))
  for (i in seq_len(nrow(feat_rows))) {
    term <- feat_rows$term[[i]]
    if (!term %in% colnames(test_df)) {
      stop(sprintf("Exported elastic-net head requires feature '%s' in test data.", term))
    }
    coef <- as.numeric(feat_rows$estimate[[i]])
    mean_x <- as.numeric(feat_rows$mean_x[[i]])
    sd_x <- as.numeric(feat_rows$sd_x[[i]])
    x_val <- as.numeric(test_df[[term]])
    use_raw <- is.na(mean_x) || is.na(sd_x) || (abs(mean_x) < 1e-12 && abs(sd_x - 1) < 1e-12)
    if (use_raw) {
      linear <- linear + coef * x_val
    } else {
      if (!is.finite(sd_x) || sd_x <= 0) {
        stop(sprintf("Invalid exported sd_x for feature '%s': %s", term, sd_x))
      }
      linear <- linear + coef * (x_val - mean_x) / sd_x
    }
  }
  1 / (1 + exp(-linear))
}

# Assert glmnet predict matches exported-coef scoring (Python deployment path).
validate_enet_export_parity <- function(
  ena_fit, test_df, params_df, rejector_mode,
  tol = ENET_EXPORT_PARITY_TOL, max_rows = 200L
) {
  if (nrow(test_df) == 0L) {
    stop("validate_enet_export_parity requires non-empty test_df.")
  }
  if (nrow(test_df) > max_rows) {
    test_df <- test_df[seq_len(max_rows), , drop = FALSE]
  }
  check_one <- function(fit, head_label) {
    glm_p <- predict_enet_cv_fit(fit, test_df)
    if (is.null(glm_p)) {
      stop(sprintf("glmnet predict failed for head '%s' during export parity check.", head_label))
    }
    head_params <- params_df[params_df$head == head_label, , drop = FALSE]
    export_p <- score_enet_head_from_export(test_df[glm_p$row_id, , drop = FALSE], head_params)
    diff <- abs(glm_p$p_hat - export_p)
    if (!all(is.finite(diff)) || max(diff) > tol) {
      stop(sprintf(
        "Elastic-net export parity failed for head '%s' (max |glmnet - export| = %.3g, tol = %.3g).",
        head_label, max(diff, na.rm = TRUE), tol
      ))
    }
  }
  if (is_two_head_rejector(rejector_mode)) {
    check_one(ena_fit$fit_correct, "correct_given_id")
    check_one(ena_fit$fit_ood, "id")
  } else {
    check_one(ena_fit, "accept_combined")
  }
  invisible(TRUE)
}

enet_scales_from_glmnet_fit <- function(glmnet_fit, x_colnames) {
  mean_src <- glmnet_fit$meanx
  sd_src <- glmnet_fit$sd
  if ((is.null(mean_src) || length(mean_src) == 0L) && !is.null(glmnet_fit$glmnet.fit)) {
    mean_src <- glmnet_fit$glmnet.fit$meanx
    sd_src <- glmnet_fit$glmnet.fit$sd
  }
  if (is.null(mean_src) || is.null(sd_src) || length(mean_src) != length(x_colnames)) {
    return(NULL)
  }
  list(
    mean_x = setNames(as.numeric(mean_src), x_colnames),
    sd_x = {
      sd_x <- setNames(as.numeric(sd_src), x_colnames)
      sd_x[!is.finite(sd_x) | sd_x <= 0] <- 1
      sd_x
    }
  )
}

prep_enet_design <- function(df, feature_terms, y_col = NULL) {
  feature_terms <- feature_terms[feature_terms %in% colnames(df)]
  if (length(feature_terms) == 0L) {
    return(list(x = matrix(numeric(0), nrow = 0L), row_id = integer(0), y = NULL))
  }
  mat <- as.matrix(df[, feature_terms, drop = FALSE])
  keep <- stats::complete.cases(mat)
  if (!is.null(y_col)) {
    keep <- keep & !is.na(df[[y_col]])
  }
  list(
    x = mat[keep, , drop = FALSE],
    row_id = which(keep),
    y = if (!is.null(y_col)) as.numeric(df[[y_col]][keep]) else NULL
  )
}

# glmnet 5.x no longer stores meanx/sd on the fit; keep training-column scales for Python export.
enet_feature_scales_from_matrix <- function(x_mat) {
  if (is.null(x_mat) || ncol(x_mat) == 0L) {
    return(list(mean_x = numeric(0), sd_x = numeric(0)))
  }
  mean_x <- colMeans(x_mat)
  sd_x <- apply(x_mat, 2, stats::sd)
  sd_x[!is.finite(sd_x) | sd_x <= 0] <- 1
  list(mean_x = mean_x, sd_x = sd_x)
}

enet_ood_weights <- function(is_id) {
  is_id <- as.integer(is_id > 0)
  freq <- table(is_id)
  per_row <- 1 / as.numeric(freq[as.character(is_id)])
  per_row * length(is_id) / sum(per_row)
}

# Two-head OOD head: inverse class-frequency weights (TRUE) vs unweighted (FALSE).
class_balanced_ood_setting <- function() {
  if (exists("CLASS_BALANCED_OOD", inherits = TRUE)) {
    isTRUE(get("CLASS_BALANCED_OOD", inherits = TRUE))
  } else {
    TRUE
  }
}

class_balanced_ood_label <- function(class_balanced_ood = class_balanced_ood_setting()) {
  if (isTRUE(class_balanced_ood)) "balanced" else "unweighted"
}

enet_study_foldid <- function(train_df, row_id, study_col = ENET_STUDY_FOLDID_COL) {
  if (!study_col %in% colnames(train_df)) return(NULL)
  foldid <- as.integer(factor(train_df[[study_col]]))
  foldid <- foldid[row_id]
  if (length(unique(foldid)) < 2L) return(NULL)
  foldid
}

cv_glmnet_binary <- function(
  train_df, y_col, feature_terms, alpha, min_rows = 20L, weights = NULL, nfolds = NULL
) {
  des <- prep_enet_design(train_df, feature_terms, y_col)
  if (length(des$y) < min_rows || length(unique(des$y)) < 2L) return(NULL)
  foldid <- enet_study_foldid(train_df, des$row_id)
  cv_args <- list(
    x = des$x,
    y = des$y,
    family = "binomial",
    alpha = alpha,
    type.measure = "deviance",
    control = glmnet_control()
  )
  if (!is.null(weights)) {
    if (length(weights) != nrow(train_df)) {
      stop("cv_glmnet_binary: weights length must match nrow(train_df).")
    }
    cv_args$weights <- weights[des$row_id]
  }
  if (!is.null(foldid)) {
    cv_args$foldid <- foldid
  } else {
    nf <- if (is.null(nfolds)) enet_nfolds(length(des$y)) else as.integer(nfolds)
    if (nf < 2L) return(NULL)
    cv_args$nfolds <- nf
  }
  cv_fit <- tryCatch(
    do.call(glmnet::cv.glmnet, cv_args),
    error = function(e) NULL
  )
  if (is.null(cv_fit)) return(NULL)
  scales <- enet_scales_from_glmnet_fit(cv_fit$glmnet.fit, colnames(des$x))
  if (is.null(scales)) {
    scales <- enet_feature_scales_from_matrix(des$x)
  }
  fit_obj <- list(
    cv_fit = cv_fit,
    lambda = cv_fit$lambda.min,
    alpha = alpha,
    feature_terms = feature_terms,
    mean_x = scales$mean_x,
    sd_x = scales$sd_x
  )
  if (!enet_has_usable_coefs(fit_obj)) return(NULL)
  fit_obj
}

predict_enet_cv_fit <- function(ena_fit, test_df) {
  if (is.null(ena_fit) || nrow(test_df) == 0L) return(NULL)
  des <- prep_enet_design(test_df, ena_fit$feature_terms)
  if (length(des$row_id) == 0L) return(NULL)
  p_hat <- as.numeric(stats::predict(ena_fit$cv_fit, newx = des$x, s = ena_fit$lambda, type = "response"))
  data.frame(row_id = des$row_id, p_hat = p_hat)
}

cv_glmnet_twohead <- function(
  train_df, feature_terms, alpha, min_rows = 20L, class_balanced_ood = NULL
) {
  if (is.null(class_balanced_ood)) class_balanced_ood <- class_balanced_ood_setting()
  if (!all(c("correct", "is_seen") %in% colnames(train_df))) return(NULL)
  correct_pool <- train_df[train_df$is_seen == 1L, , drop = FALSE]
  correct_terms <- enet_keep_feature_terms(correct_pool, feature_terms, "correct")
  if (length(correct_terms) == 0L) return(NULL)
  fit_correct <- cv_glmnet_binary(
    correct_pool, "correct", correct_terms, alpha, min_rows = min_rows
  )
  if (is.null(fit_correct)) return(NULL)
  tr <- train_df
  tr$is_id <- as.integer(tr$is_seen > 0)
  if (length(unique(tr$is_id)) < 2L) return(NULL)
  ood_terms <- enet_keep_feature_terms(tr, feature_terms, "is_id")
  if (length(ood_terms) == 0L) return(NULL)
  weights_vec <- if (isTRUE(class_balanced_ood)) enet_ood_weights(tr$is_id) else NULL
  fit_ood <- cv_glmnet_binary(
    tr, "is_id", ood_terms, alpha, min_rows = min_rows, weights = weights_vec
  )
  if (is.null(fit_ood)) return(NULL)
  list(
    fit_correct = fit_correct,
    fit_ood = fit_ood,
    alpha = alpha,
    feature_terms = feature_terms,
    rejector_mode = "two_head_product"
  )
}

predict_enet_twohead_heads <- function(two_fit, test_df) {
  if (is.null(two_fit) || nrow(test_df) == 0L) return(NULL)
  pred_c <- predict_enet_cv_fit(two_fit$fit_correct, test_df)
  pred_id <- predict_enet_cv_fit(two_fit$fit_ood, test_df)
  if (is.null(pred_c) || is.null(pred_id)) return(NULL)
  dplyr::inner_join(
    pred_c %>% dplyr::rename(p_correct = p_hat),
    pred_id %>% dplyr::rename(p_id = p_hat),
    by = "row_id"
  )
}

# Map rejector_mode to how the two head probabilities are combined.
two_head_combine_method <- function(rejector_mode, two_head_combine = NULL) {
  if (!is.null(two_head_combine)) {
    return(match.arg(as.character(two_head_combine), c("min", "product", "postcal")))
  }
  mode <- as.character(rejector_mode)
  if (mode == "two_head_min") return("min")
  if (mode %in% c("two_head_product", "two_head")) return("product")
  if (mode == "two_head_postcal") return("postcal")
  stop(sprintf("Unknown two-head rejector_mode for combine: %s", mode))
}

is_two_head_rejector <- function(rejector_mode) {
  as.character(rejector_mode) %in% c(
    "two_head_min", "two_head_product", "two_head_postcal", "two_head"
  )
}

# Platt-style recalibration on logit(P(correct|ID) * P(ID)) fit on pool OOF.
fit_twohead_postcal <- function(y, p_correct, p_id, eps = 1e-6) {
  keep <- is.finite(y) & is.finite(p_correct) & is.finite(p_id)
  if (sum(keep) < 10L) return(NULL)
  if (length(unique(as.integer(y[keep] > 0))) < 2L) return(NULL)
  raw <- pmax(eps, pmin(1 - eps, p_correct[keep] * p_id[keep]))
  df <- data.frame(
    y = as.integer(y[keep] > 0),
    logit_score = qlogis(raw)
  )
  fit <- tryCatch(
    stats::glm(
      y ~ logit_score, data = df, family = stats::binomial(),
      control = stats::glm.control(maxit = 200, epsilon = 1e-8)
    ),
    error = function(e) NULL
  )
  if (is.null(fit) || !isTRUE(fit$converged)) return(NULL)
  fit
}

apply_twohead_postcal <- function(p_correct, p_id, postcal_fit, eps = 1e-6) {
  if (is.null(postcal_fit)) return(rep(NA_real_, length(p_correct)))
  raw <- pmax(eps, pmin(1 - eps, p_correct * p_id))
  as.numeric(stats::predict(
    postcal_fit,
    newdata = data.frame(logit_score = qlogis(raw)),
    type = "response"
  ))
}

combine_twohead_scores <- function(
  p_correct, p_id, combine = c("min", "product", "postcal"), postcal_fit = NULL
) {
  combine <- match.arg(combine)
  if (combine == "min") return(pmin(p_correct, p_id))
  raw_product <- p_correct * p_id
  if (combine == "product") return(raw_product)
  apply_twohead_postcal(p_correct, p_id, postcal_fit)
}

predict_twohead_heads <- function(fit_obj, test_df) {
  if (is.null(fit_obj) || nrow(test_df) == 0L) return(NULL)
  rhs_terms <- fit_obj$rhs_terms
  if (length(rhs_terms) == 0L || !all(rhs_terms %in% colnames(test_df))) return(NULL)
  te_rhs <- test_df[, rhs_terms, drop = FALSE]
  keep <- stats::complete.cases(te_rhs)
  if (sum(keep) == 0L) return(NULL)
  te <- te_rhs[keep, , drop = FALSE]
  p_correct <- tryCatch(
    as.numeric(stats::predict(fit_obj$fit_correct, newdata = te, type = "response")),
    error = function(e) NULL
  )
  p_id <- tryCatch(
    as.numeric(stats::predict(fit_obj$fit_ood, newdata = te, type = "response")),
    error = function(e) NULL
  )
  if (is.null(p_correct) || is.null(p_id)) return(NULL)
  data.frame(
    row_id = which(keep),
    p_correct = p_correct,
    p_id = p_id
  )
}

predict_twohead_combined <- function(
  fit_obj, test_df, combine = c("min", "product", "postcal"), postcal_fit = NULL
) {
  heads <- predict_twohead_heads(fit_obj, test_df)
  if (is.null(heads)) return(NULL)
  heads$p_hat <- combine_twohead_scores(
    heads$p_correct, heads$p_id, combine = combine, postcal_fit = postcal_fit
  )
  heads
}

predict_twohead_min <- function(fit_obj, test_df) {
  predict_twohead_combined(fit_obj, test_df, combine = "min")
}

predict_enet_twohead_combined <- function(
  two_fit, test_df, combine = c("min", "product", "postcal"), postcal_fit = NULL
) {
  heads <- predict_enet_twohead_heads(two_fit, test_df)
  if (is.null(heads)) return(NULL)
  heads$p_hat <- combine_twohead_scores(
    heads$p_correct, heads$p_id, combine = combine, postcal_fit = postcal_fit
  )
  heads
}

predict_enet_twohead_min <- function(two_fit, test_df) {
  predict_enet_twohead_combined(two_fit, test_df, combine = "min")
}

# Bind named study folds and tag rows for study-blocked cv.glmnet foldid.
bind_pool_fold_dfs <- function(pool_fold_dfs, pool_rule) {
  if (is.null(pool_fold_dfs) || length(pool_fold_dfs) == 0L) return(NULL)
  pieces <- lapply(names(pool_fold_dfs), function(study_id) {
    df <- pool_fold_dfs[[study_id]]
    if (is.null(df) || nrow(df) == 0L) return(NULL)
    df <- apply_row_rule(df, pool_rule)
    if (nrow(df) == 0L) return(NULL)
    df[[ENET_STUDY_FOLDID_COL]] <- study_id
    df
  })
  pieces <- pieces[!vapply(pieces, is.null, logical(1L))]
  if (length(pieces) == 0L) return(NULL)
  bind_rows(pieces)
}

# Full pool fit: study-blocked cv.glmnet (LOSO folds as foldid) + lambda.min (matches inner CV).
fit_enet_rejector_on_pool <- function(
  pool_fold_dfs, y_col, feature_terms, alpha, pool_rule, min_rows = 20L,
  rejector_mode = "single_head", class_balanced_ood = NULL
) {
  if (is.null(class_balanced_ood)) class_balanced_ood <- class_balanced_ood_setting()
  train_u <- bind_pool_fold_dfs(pool_fold_dfs, pool_rule)
  if (is.null(train_u)) return(NULL)
  if (is_two_head_rejector(rejector_mode)) {
    cv_glmnet_twohead(train_u, feature_terms, alpha, min_rows = min_rows, class_balanced_ood = class_balanced_ood)
  } else {
    cv_glmnet_binary(train_u, y_col, feature_terms, alpha, min_rows = min_rows)
  }
}

predict_enet_rejector <- function(
  ena_fit, test_df, rejector_mode = "single_head",
  two_head_combine = NULL, postcal_fit = NULL
) {
  if (is_two_head_rejector(rejector_mode)) {
    combine <- two_head_combine_method(rejector_mode, two_head_combine)
    predict_enet_twohead_combined(
      ena_fit, test_df, combine = combine, postcal_fit = postcal_fit
    )
  } else {
    predict_enet_cv_fit(ena_fit, test_df)
  }
}

rejector_spec_elasticnet <- function(
  alpha, feature_terms, rejector_mode, lambda = NULL, class_balanced_ood = NULL
) {
  if (is.null(class_balanced_ood)) class_balanced_ood <- class_balanced_ood_setting()
  list(
    kind = "elasticnet",
    alpha = as.numeric(alpha),
    lambda = lambda,
    feature_terms = feature_terms,
    rejector_mode = as.character(rejector_mode),
    class_balanced_ood = isTRUE(class_balanced_ood)
  )
}

rejector_spec_rhs_key <- function(spec) {
  sprintf("elasticnet;alpha=%g", spec$alpha)
}

rejector_spec_from_stub <- function(stub) {
  if (!is.null(stub$rejector_spec)) return(stub$rejector_spec)
  rhs_terms <- strsplit(stub$inner_winner_rhs_key, ";", fixed = TRUE)[[1]]
  rejector_mode <- if (!is.null(stub$rejector_mode)) stub$rejector_mode else "single_head"
  if (length(rhs_terms) >= 1L && identical(rhs_terms[[1L]], "elasticnet")) {
    alpha <- stub$inner_winner_alpha
    if (is.null(alpha) && length(rhs_terms) >= 2L) {
      alpha <- as.numeric(sub("^alpha=", "", rhs_terms[[2L]]))
    }
    feature_terms <- if (!is.null(stub$inner_winner_feature_terms)) {
      stub$inner_winner_feature_terms
    } else {
      ALL_FEATURE_TERMS
    }
    return(rejector_spec_elasticnet(
      alpha, feature_terms, rejector_mode, lambda = stub$inner_winner_lambda
    ))
  }
  list(kind = "glm", rhs_terms = rhs_terms, rejector_mode = rejector_mode)
}

# Two-head rejector: P(correct|ID) on seen rows; P(ID) on all rows; accept score = min of both.
fit_twohead_models <- function(train_df, rhs_terms, min_rows = 20L, class_balanced_ood = NULL) {
  if (is.null(class_balanced_ood)) class_balanced_ood <- class_balanced_ood_setting()
  if (length(rhs_terms) == 0L) return(NULL)
  need_cols <- c("correct", "is_seen", rhs_terms)
  if (!all(need_cols %in% colnames(train_df))) return(NULL)
  rhs_terms <- rhs_terms[rhs_terms %in% colnames(train_df)]
  tr <- train_df[, unique(c("correct", "is_seen", rhs_terms)), drop = FALSE] %>%
    dplyr::filter(stats::complete.cases(.))
  if (nrow(tr) < min_rows) return(NULL)
  rhs_terms <- rhs_terms[vapply(rhs_terms, function(v) length(unique(tr[[v]])) > 1L, logical(1L))]
  if (length(rhs_terms) == 0L) return(NULL)

  correct_pool <- tr[tr$is_seen == 1L, , drop = FALSE]
  if (nrow(correct_pool) < min_rows || length(unique(correct_pool$correct)) < 2L) return(NULL)
  fit_correct <- fit_logistic(correct_pool, "correct", rhs_terms)
  if (is.null(fit_correct)) return(NULL)

  tr$is_id <- as.integer(tr$is_seen > 0)
  if (length(unique(tr$is_id)) < 2L) return(NULL)
  f_ood <- stats::as.formula(paste("is_id ~", paste(rhs_terms, collapse = " + ")))
  weights_vec <- NULL
  if (isTRUE(class_balanced_ood)) {
    freq <- table(tr$is_id)
    per_row <- 1 / as.numeric(freq[as.character(tr$is_id)])
    weights_vec <- per_row * nrow(tr) / sum(per_row)
  }
  ood_family <- if (is.null(weights_vec)) stats::binomial() else stats::quasibinomial()
  fit_ood <- tryCatch(
    if (is.null(weights_vec)) {
      stats::glm(
        f_ood, data = tr, family = ood_family,
        control = stats::glm.control(maxit = 200, epsilon = 1e-8)
      )
    } else {
      stats::glm(
        f_ood, data = tr, family = ood_family, weights = weights_vec,
        control = stats::glm.control(maxit = 200, epsilon = 1e-8)
      )
    },
    error = function(e) NULL
  )
  if (is.null(fit_ood) || !isTRUE(fit_ood$converged)) return(NULL)
  list(fit_correct = fit_correct, fit_ood = fit_ood, rhs_terms = rhs_terms)
}

rejector_scenario_name <- function(rejector_mode) {
  switch(
    as.character(rejector_mode),
    single_head = "Single-head (OOD-aware train)",
    two_head_min = "Two-head min(P(correct|ID), P(ID))",
    two_head_product = "Two-head product P(correct|ID) * P(ID)",
    two_head_postcal = "Two-head product + logit Platt recalibration",
    two_head = "Two-head product P(correct|ID) * P(ID)",
    maxprob_two_head_product = "Max-prob GLM two-head product",
    maxprob_single_head = "Max-prob GLM single-head (accept_combined)",
    stop(sprintf("Unknown rejector_mode: %s", rejector_mode))
  )
}

# Export glm coefficients (intercept + terms) for Python logistic scoring.
export_glm_coef_df <- function(glm_fit, head_label, model_name) {
  cm <- stats::coef(glm_fit)
  data.frame(
    model = model_name,
    head = head_label,
    term = names(cm),
    estimate = as.numeric(cm),
    stringsAsFactors = FALSE
  )
}

export_glm_twohead_coef_df <- function(fit_obj, model_name) {
  dplyr::bind_rows(
    export_glm_coef_df(fit_obj$fit_correct, "correct_given_id", model_name),
    export_glm_coef_df(fit_obj$fit_ood, "id", model_name)
  )
}

THRESHOLD_METHODS <- c("pooled_oof", "jackknife_adjusted", "ucb_95")
THRESHOLD_METHOD_LABELS <- c(
  pooled_oof = "Pooled out-of-fold",
  jackknife_adjusted = "Jackknife-adjusted",
  ucb_95 = "UCB (95%, one-sided)"
)
OOF_RISK_SELECTION_METHODS <- c("pooled_oof", "ucb_95")

is_jackknife_threshold_method <- function(threshold_method) {
  threshold_method == "jackknife_adjusted"
}

oof_threshold_selection_method <- function(threshold_method) {
  if (threshold_method == "jackknife_adjusted") return("pooled_oof")
  match.arg(threshold_method, OOF_RISK_SELECTION_METHODS)
}

merge_jackknife_cutoff_metadata <- function(threshold_method, risk_target, jk) {
  if (!is_jackknife_threshold_method(threshold_method)) {
    return(list(
      thr_risk = risk_target,
      jackknife_gap = NA_real_,
      jackknife_gap_sd = NA_real_,
      n_jackknife_rotations = NA_integer_,
      adjusted_threshold_risk = risk_target
    ))
  }
  if (is.null(jk) || !isTRUE(jk$ok)) return(NULL)
  list(
    thr_risk = jk$adjusted_risk_target,
    jackknife_gap = jk$jackknife_gap,
    jackknife_gap_sd = jk$jackknife_gap_sd,
    n_jackknife_rotations = jk$n_jackknife_rotations,
    adjusted_threshold_risk = jk$adjusted_risk_target
  )
}

# Deploy cutoff from pool LOSO-OOF (GLM two-head).
derive_glm_twohead_deploy_cutoff <- function(
  pool_fold_dfs, rhs_terms, risk_target, combine = "product",
  rejector_mode = "two_head_product", min_rows = 20L,
  threshold_method = THRESHOLD_METHODS
) {
  threshold_method <- match.arg(threshold_method, THRESHOLD_METHODS)
  pool_ids <- names(pool_fold_dfs)
  oof <- pool_oof_twohead_combined(
    pool_fold_dfs, pool_ids, "accept_combined", rhs_terms, POOL_RULE, TEST_RULE,
    min_rows, combine = combine
  )
  if (is.null(oof)) return(list(ok = FALSE))
  jk <- if (is_jackknife_threshold_method(threshold_method)) {
    jackknife_pool_risk_gap_rejector(
      pool_fold_dfs, "accept_combined", POOL_RULE, TEST_RULE, risk_target,
      min_rows = min_rows, rejector_mode = rejector_mode, rhs_terms = rhs_terms
    )
  } else {
    NULL
  }
  meta <- merge_jackknife_cutoff_metadata(threshold_method, risk_target, jk)
  if (is.null(meta)) return(list(ok = FALSE))
  tm <- oof_threshold_selection_method(threshold_method)
  thr <- threshold_from_oof_scores(
    list(y = oof$y, is_seen = oof$is_seen, p_hat = oof$p_hat),
    meta$thr_risk,
    threshold_method = tm
  )
  if (!is.finite(thr)) return(list(ok = FALSE))
  list(
    ok = TRUE,
    threshold = thr,
    jackknife_gap = meta$jackknife_gap,
    jackknife_gap_sd = meta$jackknife_gap_sd,
    n_jackknife_rotations = meta$n_jackknife_rotations,
    adjusted_threshold_risk = meta$adjusted_threshold_risk
  )
}

derive_glm_twohead_jackknife_cutoff <- function(...) {
  do.call(derive_glm_twohead_deploy_cutoff, c(list(...), list(threshold_method = "jackknife_adjusted")))
}

# Deploy cutoff from pool LOSO-OOF (GLM single-head).
derive_glm_singlehead_deploy_cutoff <- function(
  pool_fold_dfs, rhs_terms, risk_target, rejector_mode = "single_head", min_rows = 20L,
  threshold_method = THRESHOLD_METHODS
) {
  threshold_method <- match.arg(threshold_method, THRESHOLD_METHODS)
  pool_ids <- names(pool_fold_dfs)
  oof <- pool_oof_singlehead(
    pool_fold_dfs, pool_ids, "accept_combined", rhs_terms, POOL_RULE, TEST_RULE, min_rows
  )
  if (is.null(oof)) return(list(ok = FALSE))
  jk <- if (is_jackknife_threshold_method(threshold_method)) {
    jackknife_pool_risk_gap_rejector(
      pool_fold_dfs, "accept_combined", POOL_RULE, TEST_RULE, risk_target,
      min_rows = min_rows, rejector_mode = rejector_mode, rhs_terms = rhs_terms
    )
  } else {
    NULL
  }
  meta <- merge_jackknife_cutoff_metadata(threshold_method, risk_target, jk)
  if (is.null(meta)) return(list(ok = FALSE))
  tm <- oof_threshold_selection_method(threshold_method)
  thr <- threshold_from_oof_scores(
    list(y = oof$y, is_seen = oof$is_seen, p_hat = oof$p_hat),
    meta$thr_risk,
    threshold_method = tm
  )
  if (!is.finite(thr)) return(list(ok = FALSE))
  list(
    ok = TRUE,
    threshold = thr,
    jackknife_gap = meta$jackknife_gap,
    jackknife_gap_sd = meta$jackknife_gap_sd,
    n_jackknife_rotations = meta$n_jackknife_rotations,
    adjusted_threshold_risk = meta$adjusted_threshold_risk
  )
}

derive_glm_singlehead_jackknife_cutoff <- function(...) {
  do.call(derive_glm_singlehead_deploy_cutoff, c(list(...), list(threshold_method = "jackknife_adjusted")))
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

calc_binary_auroc <- function(y_true, p_hat) {
  keep <- is.finite(y_true) & is.finite(p_hat)
  if (sum(keep) < 10L) return(NA_real_)
  y <- as.integer(y_true[keep] > 0)
  p <- as.numeric(p_hat[keep])
  n_pos <- sum(y == 1L)
  n_neg <- sum(y == 0L)
  if (n_pos == 0L || n_neg == 0L) return(NA_real_)
  ranks <- rank(p, ties.method = "average")
  r_pos <- sum(ranks[y == 1L])
  (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

calc_binary_metrics <- function(y_true, p_hat) {
  keep <- is.finite(y_true) & is.finite(p_hat)
  if (sum(keep) < 10L) return(data.frame(auprc = NA_real_, auroc = NA_real_))
  y <- as.numeric(y_true[keep]); p <- as.numeric(p_hat[keep])
  data.frame(
    auprc = calc_binary_auprc(y, p),
    auroc = calc_binary_auroc(y, p)
  )
}

# Binary log-loss on accept_combined (high p for accept, low p for reject/OOD).
calc_binary_logloss <- function(y_true, p_hat, eps = 1e-15) {
  keep <- is.finite(y_true) & is.finite(p_hat)
  if (sum(keep) < 10L) return(NA_real_)
  y <- pmax(0, pmin(1, as.numeric(y_true[keep])))
  p <- pmax(eps, pmin(1 - eps, as.numeric(p_hat[keep])))
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

one_sided_wilson_upper <- function(k, n, conf_level = 0.95) {
  if (!is.finite(k) || !is.finite(n) || n <= 0) return(NA_real_)
  phat <- pmax(0, pmin(1, as.numeric(k) / as.numeric(n)))
  z <- stats::qnorm(conf_level)
  denom <- 1 + (z^2) / n
  center <- (phat + (z^2) / (2 * n)) / denom
  rad <- (z / denom) * sqrt((phat * (1 - phat) / n) + (z^2) / (4 * n^2))
  pmin(1, center + rad)
}

risk_stat_from_counts <- function(err_count, n_accepted, threshold_method) {
  if (!is.finite(err_count) || !is.finite(n_accepted) || n_accepted <= 0) return(NA_real_)
  if (threshold_method == "ucb_95") {
    return(one_sided_wilson_upper(err_count, n_accepted, conf_level = 0.95))
  }
  as.numeric(err_count) / as.numeric(n_accepted)
}

select_threshold_with_target_risk <- function(
    y_true, p_hat, is_seen, risk_target = TARGET_RISK,
    threshold_method = OOF_RISK_SELECTION_METHODS
) {
  threshold_method <- match.arg(threshold_method, OOF_RISK_SELECTION_METHODS)
  keep <- is.finite(y_true) & is.finite(p_hat) & is.finite(is_seen)
  if (sum(keep) < 10L) {
    return(data.frame(
      threshold = NA_real_, risk_all_accepted = NA_real_,
      coverage_seen = NA_real_, n_accepted = NA_integer_, n_seen_total = NA_integer_,
      risk_statistic = NA_real_
    ))
  }
  y <- as.integer(y_true[keep] > 0)
  p <- as.numeric(p_hat[keep])
  seen <- as.integer(is_seen[keep] > 0)
  n_seen_total <- sum(seen == 1L)
  if (n_seen_total <= 0L) {
    return(data.frame(
      threshold = NA_real_, risk_all_accepted = NA_real_,
      coverage_seen = NA_real_, n_accepted = NA_integer_, n_seen_total = n_seen_total,
      risk_statistic = NA_real_
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
      coverage_seen = NA_real_, n_accepted = 0L, n_seen_total = n_seen_total,
      risk_statistic = NA_real_
    ))
  }
  err_vals <- as.integer(err_cum[idx_last])
  n_vals <- as.integer(n_acc_cum[idx_last])
  eval_df <- data.frame(
    threshold = as.numeric(p_ord[idx_last]),
    risk_all_accepted = as.numeric(err_vals / pmax(1L, n_vals)),
    coverage_seen = as.numeric(seen_acc_cum[idx_last] / n_seen_total),
    n_accepted = n_vals,
    n_seen_total = as.integer(n_seen_total),
    risk_statistic = vapply(
      seq_along(err_vals),
      function(i) risk_stat_from_counts(err_vals[[i]], n_vals[[i]], threshold_method),
      numeric(1)
    ),
    stringsAsFactors = FALSE
  )
  in_band <- eval_df %>% filter(risk_statistic <= risk_target)
  if (nrow(in_band) > 0L) {
    in_band %>% arrange(desc(coverage_seen), risk_statistic, risk_all_accepted, desc(n_accepted), threshold) %>% slice(1L)
  } else {
    eval_df %>% arrange(risk_statistic, risk_all_accepted, desc(coverage_seen), desc(n_accepted), threshold) %>% slice(1L)
  }
}

SAMPLE_ID_COLUMNS <- c("sample_indices", "study", "outer_fold", "indices")

extract_features <- function(prob_matrix) {
  feats <- get_rejection_features_from_matrix(prob_matrix)
  feats$true_class <- gsub("Class\\. ", "", prob_matrix$y)
  feats$is_seen <- if ("is_leftout" %in% colnames(prob_matrix)) as.integer(!as.logical(prob_matrix$is_leftout)) else 1L
  feats$is_unseen <- 1L - feats$is_seen
  feats$accept_combined <- as.integer(feats$correct == 1L & feats$is_seen == 1L)
  id_cols <- intersect(SAMPLE_ID_COLUMNS, colnames(prob_matrix))
  for (col in id_cols) feats[[col]] <- prob_matrix[[col]]
  # CV uses numeric outer_fold; LOSO uses study names — unify for bind_rows downstream.
  if ("outer_fold" %in% id_cols) feats$outer_fold <- as.character(feats$outer_fold)
  feats
}

copy_sample_id_cols <- function(src_df, row_idx) {
  cols <- intersect(SAMPLE_ID_COLUMNS, colnames(src_df))
  if (length(cols) == 0L) return(data.frame())
  src_df[row_idx, cols, drop = FALSE]
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

# Two-head OOF pool: per-head scores (fit once per rotation, combine separately).
pool_oof_twohead_heads <- function(
  fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, min_rows = 20L
) {
  if (length(oof_ids) < 2L) return(NULL)
  y_all <- numeric(0)
  p_correct_all <- numeric(0)
  p_id_all <- numeric(0)
  seen_all <- integer(0)
  for (hold_id in oof_ids) {
    train_ids <- setdiff(oof_ids, hold_id)
    train_df <- bind_rows(fold_dfs[train_ids])
    hold_df <- fold_dfs[[hold_id]]
    tr_u <- apply_row_rule(train_df, pool_rule)
    hold_u <- apply_row_rule(hold_df, score_rule)
    if (nrow(tr_u) < min_rows || nrow(hold_u) < 10L) return(NULL)
    if (!y_col %in% colnames(tr_u) || !y_col %in% colnames(hold_u)) return(NULL)
    fit_obj <- fit_twohead_models(tr_u, rhs_terms, min_rows = min_rows)
    if (is.null(fit_obj)) return(NULL)
    rhs <- fit_obj$rhs_terms
    hold2 <- hold_u[, unique(c(y_col, "correct", "is_seen", rhs)), drop = FALSE]
    keep <- stats::complete.cases(hold2)
    if (sum(keep) < 10L) return(NULL)
    pred <- predict_twohead_heads(fit_obj, hold2[keep, , drop = FALSE])
    if (is.null(pred)) return(NULL)
    row_h <- pred$row_id
    y_all <- c(y_all, as.numeric(hold_u[[y_col]][row_h]))
    p_correct_all <- c(p_correct_all, pred$p_correct)
    p_id_all <- c(p_id_all, pred$p_id)
    seen_all <- c(
      seen_all,
      if ("is_seen" %in% colnames(hold_u)) as.integer(hold_u$is_seen[row_h] > 0) else rep(1L, length(row_h))
    )
  }
  if (length(y_all) < 10L) return(NULL)
  list(
    y = y_all, is_seen = seen_all,
    p_correct = p_correct_all, p_id = p_id_all
  )
}

pool_oof_twohead_combined <- function(
  fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, min_rows = 20L,
  combine = c("min", "product", "postcal"), postcal_fit = NULL
) {
  combine <- match.arg(combine)
  heads <- pool_oof_twohead_heads(
    fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, min_rows
  )
  if (is.null(heads)) return(NULL)
  fit <- if (combine == "postcal") {
    fit_twohead_postcal(heads$y, heads$p_correct, heads$p_id)
  } else {
    postcal_fit
  }
  if (combine == "postcal" && is.null(fit)) return(NULL)
  pf <- if (combine == "postcal") fit else postcal_fit
  p_hat <- combine_twohead_scores(
    heads$p_correct, heads$p_id, combine = combine, postcal_fit = pf
  )
  list(y = heads$y, is_seen = heads$is_seen, p_hat = p_hat, postcal_fit = fit)
}

pool_oof_twohead <- function(
  fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, min_rows = 20L,
  combine = "min", postcal_fit = NULL
) {
  out <- pool_oof_twohead_combined(
    fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, min_rows,
    combine = combine, postcal_fit = postcal_fit
  )
  if (is.null(out)) return(NULL)
  list(y = out$y, p_hat = out$p_hat, is_seen = out$is_seen)
}

threshold_from_oof_pool_twohead <- function(
  fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, risk_target, min_rows = 20L,
  combine = "min", postcal_fit = NULL
) {
  oof <- pool_oof_twohead(
    fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, min_rows,
    combine = combine, postcal_fit = postcal_fit
  )
  if (is.null(oof)) return(NA_real_)
  thr_df <- select_threshold_with_target_risk(oof$y, oof$p_hat, oof$is_seen, risk_target)
  as.numeric(thr_df$threshold[[1]])
}

threshold_from_oof_pool <- function(
  fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, risk_target,
  min_rows = 20L, rejector_mode = "single_head"
) {
  if (is_two_head_rejector(rejector_mode)) {
    threshold_from_oof_pool_twohead(
      fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, risk_target, min_rows
    )
  } else {
    threshold_from_oof_pool_singlehead(
      fold_dfs, oof_ids, y_col, rhs_terms, pool_rule, score_rule, risk_target, min_rows
    )
  }
}

# Elastic-net OOF: collect per-head scores for two-head, then combine once.
pool_oof_enet_twohead_heads <- function(
  fold_dfs, oof_ids, y_col, feature_terms, alpha, pool_rule, score_rule,
  min_rows = 20L, rejector_mode = "two_head_product"
) {
  if (length(oof_ids) < 2L) return(NULL)
  y_all <- numeric(0)
  p_correct_all <- numeric(0)
  p_id_all <- numeric(0)
  seen_all <- integer(0)
  for (hold_id in oof_ids) {
    train_ids <- setdiff(oof_ids, hold_id)
    train_df <- bind_rows(fold_dfs[train_ids])
    hold_df <- fold_dfs[[hold_id]]
    tr_u <- apply_row_rule(train_df, pool_rule)
    hold_u <- apply_row_rule(hold_df, score_rule)
    if (nrow(tr_u) < min_rows || nrow(hold_u) < 10L) return(NULL)
    train_pool <- fold_dfs[train_ids]
    ena_fit <- fit_enet_rejector_on_pool(
      train_pool, y_col, feature_terms, alpha, pool_rule,
      min_rows = min_rows, rejector_mode = rejector_mode
    )
    if (is.null(ena_fit)) return(NULL)
    pred <- predict_enet_twohead_heads(ena_fit, hold_u)
    if (is.null(pred)) return(NULL)
    row_h <- pred$row_id
    y_all <- c(y_all, as.numeric(hold_u[[y_col]][row_h]))
    p_correct_all <- c(p_correct_all, pred$p_correct)
    p_id_all <- c(p_id_all, pred$p_id)
    seen_all <- c(
      seen_all,
      if ("is_seen" %in% colnames(hold_u)) as.integer(hold_u$is_seen[row_h] > 0) else rep(1L, length(row_h))
    )
  }
  if (length(y_all) < 10L) return(NULL)
  list(y = y_all, is_seen = seen_all, p_correct = p_correct_all, p_id = p_id_all)
}

pool_oof_enet_combined <- function(
  fold_dfs, oof_ids, y_col, feature_terms, alpha, pool_rule, score_rule,
  min_rows = 20L, rejector_mode = "single_head",
  two_head_combine = NULL, postcal_fit = NULL
) {
  if (is_two_head_rejector(rejector_mode)) {
    combine <- two_head_combine_method(rejector_mode, two_head_combine)
    heads <- pool_oof_enet_twohead_heads(
      fold_dfs, oof_ids, y_col, feature_terms, alpha, pool_rule, score_rule,
      min_rows, rejector_mode = rejector_mode
    )
    if (is.null(heads)) return(NULL)
    fit <- if (combine == "postcal") {
      fit_twohead_postcal(heads$y, heads$p_correct, heads$p_id)
    } else {
      postcal_fit
    }
    if (combine == "postcal" && is.null(fit)) return(NULL)
    pf <- if (combine == "postcal") fit else postcal_fit
    p_hat <- combine_twohead_scores(
      heads$p_correct, heads$p_id, combine = combine, postcal_fit = pf
    )
    return(list(
      y = heads$y, is_seen = heads$is_seen, p_hat = p_hat, postcal_fit = fit
    ))
  }
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
    train_pool <- fold_dfs[train_ids]
    ena_fit <- fit_enet_rejector_on_pool(
      train_pool, y_col, feature_terms, alpha, pool_rule,
      min_rows = min_rows, rejector_mode = rejector_mode
    )
    if (is.null(ena_fit)) return(NULL)
    pred <- predict_enet_cv_fit(ena_fit, hold_u)
    if (is.null(pred)) return(NULL)
    row_h <- pred$row_id
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

pool_oof_enet <- function(
  fold_dfs, oof_ids, y_col, feature_terms, alpha, pool_rule, score_rule,
  min_rows = 20L, rejector_mode = "single_head",
  two_head_combine = NULL, postcal_fit = NULL
) {
  out <- pool_oof_enet_combined(
    fold_dfs, oof_ids, y_col, feature_terms, alpha, pool_rule, score_rule,
    min_rows, rejector_mode, two_head_combine = two_head_combine, postcal_fit = postcal_fit
  )
  if (is.null(out)) return(NULL)
  list(y = out$y, p_hat = out$p_hat, is_seen = out$is_seen)
}

threshold_from_oof_pool_enet <- function(
  fold_dfs, oof_ids, y_col, feature_terms, alpha, pool_rule, score_rule, risk_target,
  min_rows = 20L, rejector_mode = "single_head"
) {
  oof <- pool_oof_enet(
    fold_dfs, oof_ids, y_col, feature_terms, alpha, pool_rule, score_rule, min_rows, rejector_mode
  )
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

threshold_from_oof_scores <- function(
    oof, risk_target,
    threshold_method = OOF_RISK_SELECTION_METHODS
) {
  threshold_method <- match.arg(threshold_method, OOF_RISK_SELECTION_METHODS)
  if (is.null(oof)) return(NA_real_)
  thr_df <- select_threshold_with_target_risk(
    oof$y, oof$p_hat, oof$is_seen, risk_target,
    threshold_method = threshold_method
  )
  as.numeric(thr_df$threshold[[1]])
}

target_score_bundle_from_prediction <- function(tgt_u, y_col, pred_te, row_te) {
  y_te <- as.numeric(tgt_u[[y_col]][row_te])
  p_te <- pred_te$p_hat
  list(
    y = y_te,
    p_hat = p_te,
    row_te = row_te,
    correct = if ("correct" %in% colnames(tgt_u)) {
      as.integer(tgt_u$correct[row_te])
    } else {
      rep(NA_integer_, length(row_te))
    },
    is_seen = if ("is_seen" %in% colnames(tgt_u)) {
      as.integer(tgt_u$is_seen[row_te] > 0)
    } else {
      rep(1L, length(row_te))
    },
    true_class = as.character(tgt_u$true_class[row_te]),
    pred_class = as.character(tgt_u$pred_class[row_te]),
    auprc = calc_binary_metrics(y_te, p_te)$auprc
  )
}

jackknife_gap_from_rotation <- function(rotation, risk_target) {
  if (is.null(rotation)) return(NA_real_)
  thr <- threshold_from_oof_scores(rotation$calib_oof, risk_target, threshold_method = "pooled_oof")
  if (!is.finite(thr)) return(NA_real_)
  m <- metrics_at_fixed_threshold(
    rotation$hold$y, rotation$hold$p_hat, rotation$hold$is_seen, thr
  )
  if (!is.finite(m$risk_all_accepted)) return(NA_real_)
  m$risk_all_accepted - risk_target
}

score_jackknife_rotation_singlehead <- function(
  pool_fold_dfs, hold_id, y_col, rhs_terms, pool_rule, test_rule, min_rows = 20L
) {
  pool_ids <- names(pool_fold_dfs)
  train_ids <- setdiff(pool_ids, hold_id)
  if (length(train_ids) < 2L) return(NULL)
  train_pool <- pool_fold_dfs[train_ids]
  hold_u <- apply_row_rule(pool_fold_dfs[[hold_id]], test_rule)
  if (nrow(hold_u) < 10L) return(NULL)
  train_u <- apply_row_rule(bind_rows(train_pool), pool_rule)
  if (nrow(train_u) < min_rows) return(NULL)
  fit_obj <- fit_binary_model(train_u, y_col, rhs_terms)
  if (is.null(fit_obj)) return(NULL)
  rhs <- fit_obj$rhs_terms
  hold2 <- hold_u[, unique(c(y_col, rhs)), drop = FALSE]
  keep <- complete.cases(hold2)
  if (sum(keep) < 10L) return(NULL)
  pred <- predict_binary_model(fit_obj, hold2[keep, , drop = FALSE])
  if (is.null(pred)) return(NULL)
  row_h <- which(keep)[pred$row_id]
  calib_oof <- pool_oof_singlehead(
    train_pool, train_ids, y_col, rhs_terms, pool_rule, test_rule, min_rows
  )
  if (is.null(calib_oof)) return(NULL)
  list(
    hold = list(
      y = as.numeric(hold_u[[y_col]][row_h]),
      p_hat = pred$p_hat,
      is_seen = if ("is_seen" %in% colnames(hold_u)) {
        as.integer(hold_u$is_seen[row_h] > 0)
      } else {
        rep(1L, length(row_h))
      }
    ),
    calib_oof = calib_oof
  )
}

score_jackknife_rotation_twohead <- function(
  pool_fold_dfs, hold_id, y_col, rhs_terms, pool_rule, test_rule, min_rows = 20L,
  combine = "min", postcal_fit = NULL
) {
  pool_ids <- names(pool_fold_dfs)
  train_ids <- setdiff(pool_ids, hold_id)
  if (length(train_ids) < 2L) return(NULL)
  train_pool <- pool_fold_dfs[train_ids]
  hold_u <- apply_row_rule(pool_fold_dfs[[hold_id]], test_rule)
  if (nrow(hold_u) < 10L) return(NULL)
  train_u <- apply_row_rule(bind_rows(train_pool), pool_rule)
  if (nrow(train_u) < min_rows) return(NULL)
  fit_obj <- fit_twohead_models(train_u, rhs_terms, min_rows = min_rows)
  if (is.null(fit_obj)) return(NULL)
  rhs <- fit_obj$rhs_terms
  hold2 <- hold_u[, unique(c(y_col, "correct", "is_seen", rhs)), drop = FALSE]
  keep <- stats::complete.cases(hold2)
  if (sum(keep) < 10L) return(NULL)
  pred <- predict_twohead_combined(
    fit_obj, hold2[keep, , drop = FALSE], combine = combine, postcal_fit = postcal_fit
  )
  if (is.null(pred)) return(NULL)
  row_h <- pred$row_id
  calib_oof <- pool_oof_twohead(
    train_pool, train_ids, y_col, rhs_terms, pool_rule, test_rule, min_rows,
    combine = combine, postcal_fit = postcal_fit
  )
  if (is.null(calib_oof)) return(NULL)
  list(
    hold = list(
      y = as.numeric(hold_u[[y_col]][row_h]),
      p_hat = pred$p_hat,
      is_seen = if ("is_seen" %in% colnames(hold_u)) {
        as.integer(hold_u$is_seen[row_h] > 0)
      } else {
        rep(1L, length(row_h))
      }
    ),
    calib_oof = calib_oof
  )
}

score_jackknife_rotation_enet <- function(
  pool_fold_dfs, hold_id, y_col, rejector_spec, pool_rule, test_rule, min_rows = 20L,
  two_head_combine = NULL, postcal_fit = NULL
) {
  pool_ids <- names(pool_fold_dfs)
  train_ids <- setdiff(pool_ids, hold_id)
  if (length(train_ids) < 2L) return(NULL)
  train_pool <- pool_fold_dfs[train_ids]
  hold_u <- apply_row_rule(pool_fold_dfs[[hold_id]], test_rule)
  if (nrow(hold_u) < 10L) return(NULL)
  rejector_mode <- rejector_spec$rejector_mode
  ena_fit <- fit_enet_rejector_on_pool(
    train_pool, y_col, rejector_spec$feature_terms, rejector_spec$alpha,
    pool_rule, min_rows = min_rows, rejector_mode = rejector_mode
  )
  if (is.null(ena_fit)) return(NULL)
  pred <- predict_enet_rejector(
    ena_fit, hold_u, rejector_mode = rejector_mode,
    two_head_combine = two_head_combine, postcal_fit = postcal_fit
  )
  if (is.null(pred)) return(NULL)
  row_h <- pred$row_id
  calib_oof <- pool_oof_enet(
    train_pool, train_ids, y_col, rejector_spec$feature_terms, rejector_spec$alpha,
    pool_rule, test_rule, min_rows, rejector_mode,
    two_head_combine = two_head_combine, postcal_fit = postcal_fit
  )
  if (is.null(calib_oof)) return(NULL)
  list(
    hold = list(
      y = as.numeric(hold_u[[y_col]][row_h]),
      p_hat = pred$p_hat,
      is_seen = if ("is_seen" %in% colnames(hold_u)) {
        as.integer(hold_u$is_seen[row_h] > 0)
      } else {
        rep(1L, length(row_h))
      }
    ),
    calib_oof = calib_oof
  )
}

score_target_singlehead <- function(
  pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule, min_rows = 20L
) {
  pool_ids <- names(pool_fold_dfs)
  if (length(pool_ids) < 2L) return(list(ok = FALSE))
  tgt_u <- apply_row_rule(target_df, test_rule)
  if (nrow(tgt_u) < 10L) return(list(ok = FALSE))
  pool_oof <- pool_oof_singlehead(
    pool_fold_dfs, pool_ids, y_col, rhs_terms, pool_rule, test_rule, min_rows
  )
  if (is.null(pool_oof)) return(list(ok = FALSE))
  train_u <- apply_row_rule(bind_rows(pool_fold_dfs), pool_rule)
  if (nrow(train_u) < min_rows) return(list(ok = FALSE))
  fit_obj <- fit_binary_model(train_u, y_col, rhs_terms)
  if (is.null(fit_obj)) return(list(ok = FALSE))
  rhs <- fit_obj$rhs_terms
  te2 <- tgt_u[, unique(c(y_col, "true_class", "pred_class", rhs)), drop = FALSE]
  keep_te <- complete.cases(te2)
  if (sum(keep_te) < 10L) return(list(ok = FALSE))
  pred_te <- predict_binary_model(fit_obj, te2[keep_te, , drop = FALSE])
  if (is.null(pred_te)) return(list(ok = FALSE))
  row_te <- which(keep_te)[pred_te$row_id]
  if (!all(c("true_class", "pred_class") %in% colnames(tgt_u))) return(list(ok = FALSE))
  list(
    ok = TRUE,
    pool_oof = pool_oof,
    target = target_score_bundle_from_prediction(tgt_u, y_col, pred_te, row_te)
  )
}

score_target_twohead <- function(
  pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule, min_rows = 20L,
  combine = "min", postcal_fit = NULL
) {
  pool_ids <- names(pool_fold_dfs)
  if (length(pool_ids) < 2L) return(list(ok = FALSE))
  tgt_u <- apply_row_rule(target_df, test_rule)
  if (nrow(tgt_u) < 10L) return(list(ok = FALSE))
  oof_combined <- pool_oof_twohead_combined(
    pool_fold_dfs, pool_ids, y_col, rhs_terms, pool_rule, test_rule, min_rows,
    combine = combine, postcal_fit = postcal_fit
  )
  if (is.null(oof_combined)) return(list(ok = FALSE))
  pool_oof <- list(
    y = oof_combined$y, is_seen = oof_combined$is_seen, p_hat = oof_combined$p_hat
  )
  pf <- if (combine == "postcal") oof_combined$postcal_fit else postcal_fit
  train_u <- apply_row_rule(bind_rows(pool_fold_dfs), pool_rule)
  if (nrow(train_u) < min_rows) return(list(ok = FALSE))
  fit_obj <- fit_twohead_models(train_u, rhs_terms, min_rows = min_rows)
  if (is.null(fit_obj)) return(list(ok = FALSE))
  rhs <- fit_obj$rhs_terms
  te2 <- tgt_u[, unique(c(y_col, "true_class", "pred_class", "correct", "is_seen", rhs)), drop = FALSE]
  keep_te <- stats::complete.cases(te2)
  if (sum(keep_te) < 10L) return(list(ok = FALSE))
  pred_te <- predict_twohead_combined(
    fit_obj, te2[keep_te, , drop = FALSE], combine = combine, postcal_fit = pf
  )
  if (is.null(pred_te)) return(list(ok = FALSE))
  row_te <- pred_te$row_id
  if (!all(c("true_class", "pred_class") %in% colnames(tgt_u))) return(list(ok = FALSE))
  list(
    ok = TRUE,
    pool_oof = pool_oof,
    target = target_score_bundle_from_prediction(tgt_u, y_col, pred_te, row_te),
    postcal_fit = pf
  )
}

score_target_enet <- function(
  pool_fold_dfs, target_df, y_col, rejector_spec, pool_rule, test_rule, min_rows = 20L,
  two_head_combine = NULL, postcal_fit = NULL
) {
  pool_ids <- names(pool_fold_dfs)
  if (length(pool_ids) < 2L) return(list(ok = FALSE))
  rejector_mode <- rejector_spec$rejector_mode
  tgt_u <- apply_row_rule(target_df, test_rule)
  if (nrow(tgt_u) < 10L) return(list(ok = FALSE))
  oof_out <- pool_oof_enet_combined(
    pool_fold_dfs, pool_ids, y_col, rejector_spec$feature_terms, rejector_spec$alpha,
    pool_rule, test_rule, min_rows, rejector_mode,
    two_head_combine = two_head_combine, postcal_fit = postcal_fit
  )
  if (is.null(oof_out)) return(list(ok = FALSE))
  pool_oof <- list(y = oof_out$y, is_seen = oof_out$is_seen, p_hat = oof_out$p_hat)
  pf <- if (is_two_head_rejector(rejector_mode) && !is.null(oof_out$postcal_fit)) {
    oof_out$postcal_fit
  } else {
    postcal_fit
  }
  ena_fit <- fit_enet_rejector_on_pool(
    pool_fold_dfs, y_col, rejector_spec$feature_terms, rejector_spec$alpha,
    pool_rule, min_rows = min_rows, rejector_mode = rejector_mode
  )
  if (is.null(ena_fit)) return(list(ok = FALSE))
  pred_te <- predict_enet_rejector(
    ena_fit, tgt_u, rejector_mode = rejector_mode,
    two_head_combine = two_head_combine, postcal_fit = pf
  )
  if (is.null(pred_te)) return(list(ok = FALSE))
  row_te <- pred_te$row_id
  if (!all(c("true_class", "pred_class") %in% colnames(tgt_u))) return(list(ok = FALSE))
  list(
    ok = TRUE,
    pool_oof = pool_oof,
    target = target_score_bundle_from_prediction(tgt_u, y_col, pred_te, row_te),
    postcal_fit = pf
  )
}

# Fit rejector once per outer-fold stub; reuse scores for threshold sweeps.
score_rejector_stub <- function(
  stub, min_rows = 20L, y_col = "accept_combined", two_head_combine = NULL
) {
  pool_fold_dfs <- stub$pool_fold_dfs
  target_df <- stub$target_df
  pool_ids <- names(pool_fold_dfs)
  if (length(pool_ids) < 2L) return(list(ok = FALSE))
  rejector_mode <- if (!is.null(stub$rejector_mode)) stub$rejector_mode else "single_head"
  combine <- if (is_two_head_rejector(rejector_mode)) {
    two_head_combine_method(rejector_mode, two_head_combine)
  } else {
    NULL
  }
  spec <- rejector_spec_from_stub(stub)
  scored <- if (identical(spec$kind, "elasticnet")) {
    score_target_enet(
      pool_fold_dfs, target_df, y_col, spec, POOL_RULE, TEST_RULE, min_rows,
      two_head_combine = combine
    )
  } else if (is_two_head_rejector(rejector_mode)) {
    score_target_twohead(
      pool_fold_dfs, target_df, y_col, spec$rhs_terms, POOL_RULE, TEST_RULE, min_rows,
      combine = combine
    )
  } else {
    score_target_singlehead(
      pool_fold_dfs, target_df, y_col, spec$rhs_terms, POOL_RULE, TEST_RULE, min_rows
    )
  }
  if (!isTRUE(scored$ok)) return(scored)
  postcal_fit <- if (!is.null(scored$postcal_fit)) scored$postcal_fit else NULL
  jackknife <- list()
  if (length(pool_ids) >= 3L) {
    for (hold_id in pool_ids) {
      rot <- if (identical(spec$kind, "elasticnet")) {
        score_jackknife_rotation_enet(
          pool_fold_dfs, hold_id, y_col, spec, POOL_RULE, TEST_RULE, min_rows,
          two_head_combine = combine, postcal_fit = postcal_fit
        )
      } else if (is_two_head_rejector(rejector_mode)) {
        score_jackknife_rotation_twohead(
          pool_fold_dfs, hold_id, y_col, spec$rhs_terms, POOL_RULE, TEST_RULE, min_rows,
          combine = combine, postcal_fit = postcal_fit
        )
      } else {
        score_jackknife_rotation_singlehead(
          pool_fold_dfs, hold_id, y_col, spec$rhs_terms, POOL_RULE, TEST_RULE, min_rows
        )
      }
      if (!is.null(rot)) jackknife[[hold_id]] <- rot
    }
  }
  scored$jackknife <- jackknife
  scored
}

# Threshold + metrics at requested risk from precomputed pool OOF and target scores.
outer_eval_from_scored <- function(
  scored, risk_target, threshold_method = THRESHOLD_METHODS
) {
  threshold_method <- match.arg(threshold_method, THRESHOLD_METHODS)
  if (!isTRUE(scored$ok)) return(list(ok = FALSE))
  jk <- NULL
  if (is_jackknife_threshold_method(threshold_method)) {
    gaps <- vapply(scored$jackknife, jackknife_gap_from_rotation, numeric(1), risk_target = risk_target)
    gaps <- gaps[is.finite(gaps)]
    if (length(gaps) < 2L) return(list(ok = FALSE))
    jk <- list(
      ok = TRUE,
      jackknife_gap = mean(gaps),
      jackknife_gap_sd = stats::sd(gaps),
      n_jackknife_rotations = length(gaps),
      adjusted_risk_target = max(0, risk_target - mean(gaps))
    )
  }
  meta <- merge_jackknife_cutoff_metadata(threshold_method, risk_target, jk)
  if (is.null(meta)) return(list(ok = FALSE))
  tm <- oof_threshold_selection_method(threshold_method)
  thr <- threshold_from_oof_scores(scored$pool_oof, meta$thr_risk, threshold_method = tm)
  if (!is.finite(thr)) return(list(ok = FALSE))
  tgt <- scored$target
  m <- metrics_at_fixed_threshold(tgt$y, tgt$p_hat, tgt$is_seen, thr)
  if (!is.finite(m$risk_all_accepted) || !is.finite(m$coverage_seen)) return(list(ok = FALSE))
  kappa_acc <- kappa_accepted_at_threshold(tgt$true_class, tgt$pred_class, tgt$p_hat, thr)
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
    auprc_outer = tgt$auprc,
    auprc_outer_median = tgt$auprc,
    threshold_method = threshold_method,
    jackknife_gap = meta$jackknife_gap,
    jackknife_gap_sd = meta$jackknife_gap_sd,
    n_jackknife_rotations = meta$n_jackknife_rotations,
    adjusted_threshold_risk = meta$adjusted_threshold_risk
  )
}

# Inner CV: train on pool\\val; threshold from LOSO-OOF on train; evaluate on val at risk_target.
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
  list(
    ok = TRUE,
    mean_coverage = mean(covs),
    mean_risk = mean(risks),
    median_coverage = stats::median(covs),
    median_risk = stats::median(risks),
    sd_coverage = stats::sd(covs),
    sd_risk = stats::sd(risks)
  )
}

# Inner CV for two-head min-gate rejector (same thresholding protocol as single-head).
predict_twohead_for_mode <- function(
  fit_obj, test_df, rejector_mode = "two_head_min", postcal_fit = NULL
) {
  combine <- two_head_combine_method(rejector_mode)
  predict_twohead_combined(
    fit_obj, test_df, combine = combine, postcal_fit = postcal_fit
  )
}

inner_cv_strict_twohead <- function(
  pool_fold_dfs, y_col, rhs_terms, pool_rule, test_rule, risk_target, min_rows = 20L,
  rejector_mode = "two_head_min"
) {
  combine <- two_head_combine_method(rejector_mode)
  oof_pf <- NULL
  if (combine == "postcal") {
    ids <- names(pool_fold_dfs)
    oof_c <- pool_oof_twohead_combined(
      pool_fold_dfs, ids, y_col, rhs_terms, pool_rule, test_rule, min_rows, combine = "postcal"
    )
    if (!is.null(oof_c)) oof_pf <- oof_c$postcal_fit
  }
  ids <- names(pool_fold_dfs)
  if (length(ids) < 3L) return(list(ok = FALSE))
  covs <- numeric(length(ids))
  risks <- numeric(length(ids))
  for (iv in seq_along(ids)) {
    val_id <- ids[[iv]]
    train_ids <- setdiff(ids, val_id)
    if (length(train_ids) < 2L) return(list(ok = FALSE))
    thr <- threshold_from_oof_pool_twohead(
      pool_fold_dfs, train_ids, y_col, rhs_terms, pool_rule, test_rule, risk_target, min_rows,
      combine = combine, postcal_fit = oof_pf
    )
    if (!is.finite(thr)) return(list(ok = FALSE))
    tr <- bind_rows(pool_fold_dfs[train_ids])
    te <- pool_fold_dfs[[val_id]]
    tr_u <- apply_row_rule(tr, pool_rule)
    te_u <- apply_row_rule(te, test_rule)
    if (nrow(tr_u) < min_rows || nrow(te_u) < 10L) return(list(ok = FALSE))
    if (!y_col %in% colnames(tr_u) || !y_col %in% colnames(te_u)) return(list(ok = FALSE))
    fit_obj <- fit_twohead_models(tr_u, rhs_terms, min_rows = min_rows)
    if (is.null(fit_obj)) return(list(ok = FALSE))
    rhs <- fit_obj$rhs_terms
    te2 <- te_u[, unique(c(y_col, "correct", "is_seen", rhs)), drop = FALSE]
    keep_te <- stats::complete.cases(te2)
    if (sum(keep_te) < 10L) return(list(ok = FALSE))
    pred_te <- predict_twohead_for_mode(
      fit_obj, te2[keep_te, , drop = FALSE], rejector_mode = rejector_mode, postcal_fit = oof_pf
    )
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
  list(
    ok = TRUE,
    mean_coverage = mean(covs),
    mean_risk = mean(risks),
    median_coverage = stats::median(covs),
    median_risk = stats::median(risks),
    sd_coverage = stats::sd(covs),
    sd_risk = stats::sd(risks)
  )
}

inner_cv_strict_rejector <- function(
  pool_fold_dfs, y_col, rhs_terms, pool_rule, test_rule, risk_target,
  min_rows = 20L, rejector_mode = "single_head"
) {
  if (is_two_head_rejector(rejector_mode)) {
    inner_cv_strict_twohead(
      pool_fold_dfs, y_col, rhs_terms, pool_rule, test_rule, risk_target, min_rows, rejector_mode
    )
  } else {
    inner_cv_strict_singlehead(pool_fold_dfs, y_col, rhs_terms, pool_rule, test_rule, risk_target, min_rows)
  }
}

# Aggregate per-inner-fold log-loss and AUROC (fail if any fold is non-finite).
summarize_inner_cv_logloss_auroc <- function(ll, auc) {
  if (any(!is.finite(ll)) || any(!is.finite(auc))) return(list(ok = FALSE))
  list(
    ok = TRUE,
    mean_logloss = mean(ll),
    median_logloss = stats::median(ll),
    sd_logloss = stats::sd(ll),
    mean_auroc = mean(auc),
    median_auroc = stats::median(auc),
    sd_auroc = stats::sd(auc)
  )
}

# Inner CV log-loss + AUROC: fit on pool\\val, score val (no thresholding).
inner_cv_logloss_singlehead <- function(
  pool_fold_dfs, y_col, rhs_terms, pool_rule, test_rule, min_rows = 20L
) {
  ids <- names(pool_fold_dfs)
  if (length(ids) < 3L) return(list(ok = FALSE))
  ll <- numeric(length(ids))
  auc <- numeric(length(ids))
  for (iv in seq_along(ids)) {
    val_id <- ids[[iv]]
    train_ids <- setdiff(ids, val_id)
    if (length(train_ids) < 2L) return(list(ok = FALSE))
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
    ll[iv] <- calc_binary_logloss(y_te, p_te)
    auc[iv] <- calc_binary_auroc(y_te, p_te)
    if (!is.finite(ll[iv]) || !is.finite(auc[iv])) return(list(ok = FALSE))
  }
  summarize_inner_cv_logloss_auroc(ll, auc)
}

inner_cv_logloss_twohead <- function(
  pool_fold_dfs, y_col, rhs_terms, pool_rule, test_rule, min_rows = 20L,
  rejector_mode = "two_head_product"
) {
  combine <- two_head_combine_method(rejector_mode)
  oof_pf <- NULL
  if (combine == "postcal") {
    ids <- names(pool_fold_dfs)
    oof_c <- pool_oof_twohead_combined(
      pool_fold_dfs, ids, y_col, rhs_terms, pool_rule, test_rule, min_rows, combine = "postcal"
    )
    if (!is.null(oof_c)) oof_pf <- oof_c$postcal_fit
  }
  ids <- names(pool_fold_dfs)
  if (length(ids) < 3L) return(list(ok = FALSE))
  ll <- numeric(length(ids))
  auc <- numeric(length(ids))
  for (iv in seq_along(ids)) {
    val_id <- ids[[iv]]
    train_ids <- setdiff(ids, val_id)
    if (length(train_ids) < 2L) return(list(ok = FALSE))
    tr <- bind_rows(pool_fold_dfs[train_ids])
    te <- pool_fold_dfs[[val_id]]
    tr_u <- apply_row_rule(tr, pool_rule)
    te_u <- apply_row_rule(te, test_rule)
    if (nrow(tr_u) < min_rows || nrow(te_u) < 10L) return(list(ok = FALSE))
    if (!y_col %in% colnames(tr_u) || !y_col %in% colnames(te_u)) return(list(ok = FALSE))
    fit_obj <- fit_twohead_models(tr_u, rhs_terms, min_rows = min_rows)
    if (is.null(fit_obj)) return(list(ok = FALSE))
    rhs <- fit_obj$rhs_terms
    te2 <- te_u[, unique(c(y_col, "correct", "is_seen", rhs)), drop = FALSE]
    keep_te <- stats::complete.cases(te2)
    if (sum(keep_te) < 10L) return(list(ok = FALSE))
    pred_te <- predict_twohead_for_mode(
      fit_obj, te2[keep_te, , drop = FALSE], rejector_mode = rejector_mode, postcal_fit = oof_pf
    )
    if (is.null(pred_te)) return(list(ok = FALSE))
    row_te <- which(keep_te)[pred_te$row_id]
    y_te <- as.numeric(te_u[[y_col]][row_te])
    p_te <- pred_te$p_hat
    ll[iv] <- calc_binary_logloss(y_te, p_te)
    auc[iv] <- calc_binary_auroc(y_te, p_te)
    if (!is.finite(ll[iv]) || !is.finite(auc[iv])) return(list(ok = FALSE))
  }
  summarize_inner_cv_logloss_auroc(ll, auc)
}

inner_cv_logloss_rejector <- function(
  pool_fold_dfs, y_col, rhs_terms, pool_rule, test_rule,
  min_rows = 20L, rejector_mode = "single_head"
) {
  if (is_two_head_rejector(rejector_mode)) {
    inner_cv_logloss_twohead(
      pool_fold_dfs, y_col, rhs_terms, pool_rule, test_rule, min_rows, rejector_mode
    )
  } else {
    inner_cv_logloss_singlehead(pool_fold_dfs, y_col, rhs_terms, pool_rule, test_rule, min_rows)
  }
}

# Inner CV log-loss for one elastic-net alpha (lambda.min via study-blocked cv.glmnet).
inner_cv_logloss_enet <- function(
  pool_fold_dfs, y_col, feature_terms, alpha, pool_rule, test_rule,
  min_rows = 20L, rejector_mode = "single_head"
) {
  ids <- names(pool_fold_dfs)
  if (length(ids) < 3L) return(list(ok = FALSE))
  ll <- numeric(length(ids))
  auc <- numeric(length(ids))
  for (iv in seq_along(ids)) {
    val_id <- ids[[iv]]
    train_ids <- setdiff(ids, val_id)
    if (length(train_ids) < 2L) return(list(ok = FALSE))
    tr <- bind_rows(pool_fold_dfs[train_ids])
    te <- pool_fold_dfs[[val_id]]
    tr_u <- apply_row_rule(tr, pool_rule)
    te_u <- apply_row_rule(te, test_rule)
    if (nrow(tr_u) < min_rows || nrow(te_u) < 10L) return(list(ok = FALSE))
    if (!y_col %in% colnames(tr_u) || !y_col %in% colnames(te_u)) return(list(ok = FALSE))
    train_pool <- pool_fold_dfs[train_ids]
    ena_fit <- fit_enet_rejector_on_pool(
      train_pool, y_col, feature_terms, alpha, pool_rule,
      min_rows = min_rows, rejector_mode = rejector_mode
    )
    if (is.null(ena_fit)) return(list(ok = FALSE))
    pred_te <- predict_enet_rejector(ena_fit, te_u, rejector_mode = rejector_mode)
    if (is.null(pred_te)) return(list(ok = FALSE))
    row_te <- pred_te$row_id
    y_te <- as.numeric(te_u[[y_col]][row_te])
    p_te <- pred_te$p_hat
    ll[iv] <- calc_binary_logloss(y_te, p_te)
    auc[iv] <- calc_binary_auroc(y_te, p_te)
    if (!is.finite(ll[iv]) || !is.finite(auc[iv])) return(list(ok = FALSE))
  }
  summarize_inner_cv_logloss_auroc(ll, auc)
}

# Fail if any ALL_FEATURE_TERMS column is missing, non-finite, or constant (no silent grid reduction).
assert_calibration_terms_available <- function(df, context = "calibration data") {
  if (is.null(df) || nrow(df) == 0L) {
    stop(sprintf("%s: empty data frame.", context))
  }
  if (!"accept_combined" %in% colnames(df)) {
    stop(sprintf("%s: missing accept_combined column.", context))
  }
  missing_cols <- setdiff(ALL_FEATURE_TERMS, colnames(df))
  if (length(missing_cols) > 0L) {
    stop(sprintf(
      "%s: missing required reject feature column(s): %s.",
      context, paste(missing_cols, collapse = ", ")
    ))
  }
  invalid <- vapply(ALL_FEATURE_TERMS, function(term) {
    x <- df[[term]]
    if (!all(is.finite(x))) {
      sprintf("%s (%d non-finite of %d rows)", term, sum(!is.finite(x)), nrow(df))
    } else if (length(unique(x)) <= 1L) {
      sprintf("%s (constant)", term)
    } else {
      NA_character_
    }
  }, character(1))
  invalid <- invalid[!is.na(invalid)]
  if (length(invalid) > 0L) {
    stop(sprintf(
      "%s: invalid required reject feature(s): %s.",
      context, paste(invalid, collapse = "; ")
    ))
  }
  invisible(TRUE)
}

candidate_terms_from_df <- function(df) {
  assert_calibration_terms_available(df, "candidate_terms_from_df")
  ALL_FEATURE_TERMS
}

# Study-level jackknife within the outer train pool: calibrate on pool\{hold}, measure realized−requested on hold.
jackknife_pool_risk_gap_rejector <- function(
  pool_fold_dfs, y_col, pool_rule, test_rule, risk_target,
  min_rows = 20L, rejector_mode = "single_head",
  rejector_spec = NULL, rhs_terms = NULL
) {
  pool_ids <- names(pool_fold_dfs)
  if (length(pool_ids) < 3L) return(list(ok = FALSE))
  gaps <- numeric(0)
  for (hold_id in pool_ids) {
    train_ids <- setdiff(pool_ids, hold_id)
    if (length(train_ids) < 2L) next
    train_pool <- pool_fold_dfs[train_ids]
    hold_df <- pool_fold_dfs[[hold_id]]
    out <- outer_eval_rejector(
      train_pool, hold_df, y_col, rhs_terms, pool_rule, test_rule, risk_target,
      min_rows = min_rows, rejector_mode = rejector_mode, rejector_spec = rejector_spec
    )
    if (is.null(out) || !isTRUE(out$ok)) next
    gaps <- c(gaps, out$risk_all_accepted - risk_target)
  }
  if (length(gaps) < 2L) return(list(ok = FALSE))
  gap_mean <- mean(gaps)
  list(
    ok = TRUE,
    jackknife_gap = gap_mean,
    jackknife_gap_sd = stats::sd(gaps),
    n_jackknife_rotations = length(gaps),
    adjusted_risk_target = max(0, risk_target - gap_mean)
  )
}

jackknife_pool_risk_gap_from_stub <- function(stub, risk_target, eval_cache = NULL) {
  jk_key <- jk_gap_cache_key(stub, risk_target)
  if (!is.null(eval_cache) && !is.null(eval_cache$jk_gap[[jk_key]])) {
    return(eval_cache$jk_gap[[jk_key]])
  }
  rejector_mode <- if (!is.null(stub$rejector_mode)) stub$rejector_mode else "single_head"
  spec <- rejector_spec_from_stub(stub)
  result <- if (identical(spec$kind, "elasticnet")) {
    jackknife_pool_risk_gap_rejector(
      stub$pool_fold_dfs, "accept_combined", POOL_RULE, TEST_RULE, risk_target,
      rejector_mode = rejector_mode, rejector_spec = spec
    )
  } else {
    jackknife_pool_risk_gap_rejector(
      stub$pool_fold_dfs, "accept_combined", POOL_RULE, TEST_RULE, risk_target,
      rejector_mode = rejector_mode, rhs_terms = spec$rhs_terms
    )
  }
  if (!is.null(eval_cache)) {
    eval_cache$jk_gap[[jk_key]] <- result
  }
  result
}

# Outer: fit on pool; evaluate on target. Threshold from pool LOSO-OOF unless fixed_threshold is set.
outer_eval_singlehead <- function(
  pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule, risk_target,
  min_rows = 20L, fixed_threshold = NULL, threshold_risk_target = NULL
) {
  pool_ids <- names(pool_fold_dfs)
  if (length(pool_ids) < 2L) return(list(ok = FALSE))
  tgt_u <- apply_row_rule(target_df, test_rule)
  if (nrow(tgt_u) < 10L) return(list(ok = FALSE))
  thr_risk <- if (is.null(threshold_risk_target)) risk_target else threshold_risk_target

  thr <- if (!is.null(fixed_threshold)) {
    as.numeric(fixed_threshold)
  } else {
    threshold_from_oof_pool_singlehead(
      pool_fold_dfs, pool_ids, y_col, rhs_terms, pool_rule, test_rule, thr_risk, min_rows
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

# Outer evaluation for two-head min-gate rejector.
outer_eval_twohead <- function(
  pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule, risk_target,
  min_rows = 20L, fixed_threshold = NULL, threshold_risk_target = NULL,
  rejector_mode = "two_head_min"
) {
  combine <- two_head_combine_method(rejector_mode)
  oof_pf <- NULL
  if (combine == "postcal" && is.null(fixed_threshold)) {
    pool_ids <- names(pool_fold_dfs)
    oof_c <- pool_oof_twohead_combined(
      pool_fold_dfs, pool_ids, y_col, rhs_terms, pool_rule, test_rule, min_rows, combine = "postcal"
    )
    if (is.null(oof_c)) return(list(ok = FALSE))
    oof_pf <- oof_c$postcal_fit
  }
  pool_ids <- names(pool_fold_dfs)
  if (length(pool_ids) < 2L) return(list(ok = FALSE))
  tgt_u <- apply_row_rule(target_df, test_rule)
  if (nrow(tgt_u) < 10L) return(list(ok = FALSE))
  thr_risk <- if (is.null(threshold_risk_target)) risk_target else threshold_risk_target

  thr <- if (!is.null(fixed_threshold)) {
    as.numeric(fixed_threshold)
  } else {
    threshold_from_oof_pool_twohead(
      pool_fold_dfs, pool_ids, y_col, rhs_terms, pool_rule, test_rule, thr_risk, min_rows,
      combine = combine, postcal_fit = oof_pf
    )
  }
  if (!is.finite(thr)) return(list(ok = FALSE))

  train_df <- bind_rows(pool_fold_dfs)
  train_u <- apply_row_rule(train_df, pool_rule)
  if (nrow(train_u) < min_rows) return(list(ok = FALSE))
  fit_obj <- fit_twohead_models(train_u, rhs_terms, min_rows = min_rows)
  if (is.null(fit_obj)) return(list(ok = FALSE))
  rhs <- fit_obj$rhs_terms
  te2 <- tgt_u[, unique(c(y_col, "correct", "is_seen", rhs)), drop = FALSE]
  keep_te <- stats::complete.cases(te2)
  if (sum(keep_te) < 10L) return(list(ok = FALSE))
  pred_te <- predict_twohead_for_mode(
    fit_obj, te2[keep_te, , drop = FALSE], rejector_mode = rejector_mode, postcal_fit = oof_pf
  )
  if (is.null(pred_te)) return(list(ok = FALSE))
  row_te <- which(keep_te)[pred_te$row_id]
  y_te <- as.numeric(tgt_u[[y_col]][row_te])
  p_te <- pred_te$p_hat
  seen_te <- if ("is_seen" %in% colnames(tgt_u)) as.integer(tgt_u$is_seen[row_te] > 0) else rep(1L, length(row_te))
  m <- metrics_at_fixed_threshold(y_te, p_te, seen_te, thr)
  if (!is.finite(m$risk_all_accepted) || !is.finite(m$coverage_seen)) return(list(ok = FALSE))
  aup <- calc_binary_metrics(y_te, p_te)$auprc
  if (!all(c("true_class", "pred_class") %in% colnames(tgt_u))) return(list(ok = FALSE))
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

# Outer evaluation with elastic-net rejector (alpha fixed; lambda.min via study-blocked cv.glmnet).
outer_eval_enet <- function(
  pool_fold_dfs, target_df, y_col, rejector_spec, pool_rule, test_rule, risk_target,
  min_rows = 20L, fixed_threshold = NULL, threshold_risk_target = NULL
) {
  pool_ids <- names(pool_fold_dfs)
  if (length(pool_ids) < 2L) return(list(ok = FALSE))
  tgt_u <- apply_row_rule(target_df, test_rule)
  if (nrow(tgt_u) < 10L) return(list(ok = FALSE))
  alpha <- rejector_spec$alpha
  feature_terms <- rejector_spec$feature_terms
  rejector_mode <- rejector_spec$rejector_mode
  thr_risk <- if (is.null(threshold_risk_target)) risk_target else threshold_risk_target

  thr <- if (!is.null(fixed_threshold)) {
    as.numeric(fixed_threshold)
  } else {
    threshold_from_oof_pool_enet(
      pool_fold_dfs, pool_ids, y_col, feature_terms, alpha, pool_rule, test_rule, thr_risk,
      min_rows, rejector_mode
    )
  }
  if (!is.finite(thr)) return(list(ok = FALSE))

  ena_fit <- fit_enet_rejector_on_pool(
    pool_fold_dfs, y_col, feature_terms, alpha, pool_rule, min_rows, rejector_mode
  )
  if (is.null(ena_fit)) return(list(ok = FALSE))
  pred_te <- predict_enet_rejector(ena_fit, tgt_u, rejector_mode = rejector_mode)
  if (is.null(pred_te)) return(list(ok = FALSE))
  row_te <- pred_te$row_id
  y_te <- as.numeric(tgt_u[[y_col]][row_te])
  p_te <- pred_te$p_hat
  seen_te <- if ("is_seen" %in% colnames(tgt_u)) as.integer(tgt_u$is_seen[row_te] > 0) else rep(1L, length(row_te))
  m <- metrics_at_fixed_threshold(y_te, p_te, seen_te, thr)
  if (!is.finite(m$risk_all_accepted) || !is.finite(m$coverage_seen)) return(list(ok = FALSE))
  aup <- calc_binary_metrics(y_te, p_te)$auprc
  if (!all(c("true_class", "pred_class") %in% colnames(tgt_u))) return(list(ok = FALSE))
  kappa_acc <- kappa_accepted_at_threshold(
    tgt_u$true_class[row_te], tgt_u$pred_class[row_te], p_te, thr
  )
  pool_lambda <- if (is_two_head_rejector(rejector_mode)) {
    c(correct = ena_fit$fit_correct$lambda, ood = ena_fit$fit_ood$lambda)
  } else {
    ena_fit$lambda
  }
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
    auprc_outer_median = aup,
    pool_lambda = pool_lambda,
    ena_fit = ena_fit
  )
}

outer_eval_rejector <- function(
  pool_fold_dfs, target_df, y_col, rhs_terms = NULL, pool_rule, test_rule, risk_target,
  min_rows = 20L, fixed_threshold = NULL, rejector_mode = "single_head",
  rejector_spec = NULL, threshold_risk_target = NULL
) {
  if (!is.null(rejector_spec) && identical(rejector_spec$kind, "elasticnet")) {
    return(outer_eval_enet(
      pool_fold_dfs, target_df, y_col, rejector_spec, pool_rule, test_rule, risk_target,
      min_rows, fixed_threshold, threshold_risk_target
    ))
  }
  if (is.null(rhs_terms)) {
    stop("outer_eval_rejector: rhs_terms required when rejector_spec is not elasticnet.")
  }
  if (is_two_head_rejector(rejector_mode)) {
    outer_eval_twohead(
      pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule, risk_target,
      min_rows, fixed_threshold, threshold_risk_target, rejector_mode = rejector_mode
    )
  } else {
    outer_eval_singlehead(
      pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule, risk_target,
      min_rows, fixed_threshold, threshold_risk_target
    )
  }
}

# Pin BLAS/OpenMP threads so forked mclapply workers do not oversubscribe CPU.
pin_blas_threads <- function(threads = 1L) {
  val <- as.character(as.integer(threads))
  env_vars <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
  args <- stats::setNames(rep(list(val), length(env_vars)), env_vars)
  do.call(Sys.setenv, args)
  invisible(as.integer(threads))
}

# Post-nested cache parallelism: default 3; override with CALIBRATION_REJECT_CACHE_MC_CORES (use 1 for low memory).
cache_parallel_mc_cores <- function(n_stubs = 1L) {
  if (.Platform$OS.type != "unix") return(1L)
  env_val <- suppressWarnings(as.integer(Sys.getenv("CALIBRATION_REJECT_CACHE_MC_CORES", unset = NA_integer_)))
  cap <- if (length(env_val) == 1L && !is.na(env_val) && env_val >= 1L) {
    as.integer(env_val)
  } else if (exists("DEFAULT_CACHE_MC_CORES", inherits = TRUE)) {
    as.integer(get("DEFAULT_CACHE_MC_CORES", inherits = TRUE))
  } else {
    3L
  }
  max(1L, min(cap, as.integer(n_stubs)))
}

risk_targets_match <- function(x, targets, tol = 1e-12) {
  if (length(targets) == 0L) return(rep(FALSE, length(x)))
  vapply(x, function(r) any(abs(targets - r) < tol), logical(1))
}

stub_eval_cache_key <- function(stub) {
  rej <- if (!is.null(stub$rejector_mode)) as.character(stub$rejector_mode) else "single_head"
  rhs <- if (!is.null(stub$inner_winner_rhs_key)) as.character(stub$inner_winner_rhs_key) else ""
  paste(
    as.character(stub$label_set), as.character(stub$split_type),
    as.character(stub$fold_name), as.character(stub$base_model),
    rej, rhs,
    sep = "|"
  )
}

jk_gap_cache_key <- function(stub, risk_target) {
  paste(stub_eval_cache_key(stub), format(risk_target, digits = 17, scientific = TRUE), sep = "||")
}

empty_outer_eval_cache_df <- function() {
  data.frame(
    stub_key = character(),
    label_set = character(),
    split_type = character(),
    target_fold = character(),
    base_model = character(),
    rejector_mode = character(),
    inner_winner_rhs_key = character(),
    requested_target_risk = numeric(),
    requested_target_risk_pct = numeric(),
    threshold_method = character(),
    threshold_method_label = character(),
    threshold = numeric(),
    threshold_median = numeric(),
    outer_risk_all_accepted = numeric(),
    outer_coverage_seen = numeric(),
    outer_kappa_accepted = numeric(),
    adjusted_threshold_risk = numeric(),
    jackknife_gap = numeric(),
    jackknife_gap_sd = numeric(),
    n_jackknife_rotations = integer(),
    outer_auprc = numeric(),
    stringsAsFactors = FALSE
  )
}

outer_eval_out_to_cache_row <- function(stub, risk_target, threshold_method, out) {
  rej <- if (!is.null(stub$rejector_mode)) as.character(stub$rejector_mode) else "single_head"
  rhs <- if (!is.null(stub$inner_winner_rhs_key)) as.character(stub$inner_winner_rhs_key) else NA_character_
  data.frame(
    stub_key = stub_eval_cache_key(stub),
    label_set = as.character(stub$label_set),
    split_type = as.character(stub$split_type),
    target_fold = as.character(stub$fold_name),
    base_model = as.character(stub$base_model),
    rejector_mode = rej,
    inner_winner_rhs_key = rhs,
    requested_target_risk = risk_target,
    requested_target_risk_pct = 100 * risk_target,
    threshold_method = threshold_method,
    threshold_method_label = THRESHOLD_METHOD_LABELS[[threshold_method]],
    threshold = out$threshold,
    threshold_median = out$threshold_median,
    outer_risk_all_accepted = out$risk_all_accepted,
    outer_coverage_seen = out$coverage_seen,
    outer_kappa_accepted = out$kappa_accepted,
    adjusted_threshold_risk = out$adjusted_threshold_risk,
    jackknife_gap = out$jackknife_gap,
    jackknife_gap_sd = out$jackknife_gap_sd,
    n_jackknife_rotations = out$n_jackknife_rotations,
    outer_auprc = out$auprc_outer,
    stringsAsFactors = FALSE
  )
}

operating_df_to_cache_rows <- function(seed_operating_df, risk_targets, threshold_methods) {
  empty <- empty_outer_eval_cache_df()
  if (is.null(seed_operating_df) || nrow(seed_operating_df) == 0L) return(empty)
  if (!"threshold_method" %in% names(seed_operating_df)) return(empty)
  df <- seed_operating_df
  df <- df[risk_targets_match(df$requested_target_risk, risk_targets), , drop = FALSE]
  df <- df[as.character(df$threshold_method) %in% threshold_methods, , drop = FALSE]
  if (nrow(df) == 0L) return(empty)
  rej_col <- if ("rejector_mode" %in% names(df)) as.character(df$rejector_mode) else "single_head"
  rhs_col <- if ("inner_winner_rhs_key" %in% names(df)) as.character(df$inner_winner_rhs_key) else NA_character_
  data.frame(
    stub_key = paste(df$label_set, df$split_type, df$target_fold, df$base_model, rej_col, rhs_col, sep = "|"),
    label_set = as.character(df$label_set),
    split_type = as.character(df$split_type),
    target_fold = as.character(df$target_fold),
    base_model = as.character(df$base_model),
    rejector_mode = rej_col,
    inner_winner_rhs_key = rhs_col,
    requested_target_risk = as.numeric(df$requested_target_risk),
    requested_target_risk_pct = as.numeric(df$requested_target_risk_pct),
    threshold_method = as.character(df$threshold_method),
    threshold_method_label = if ("threshold_method_label" %in% names(df)) {
      as.character(df$threshold_method_label)
    } else {
      THRESHOLD_METHOD_LABELS[as.character(df$threshold_method)]
    },
    threshold = as.numeric(df$threshold_outer_cal_mean),
    threshold_median = as.numeric(df$threshold_outer_cal_median),
    outer_risk_all_accepted = as.numeric(df$outer_risk_all_accepted),
    outer_coverage_seen = as.numeric(df$outer_coverage_seen),
    outer_kappa_accepted = as.numeric(df$outer_kappa_accepted),
    adjusted_threshold_risk = if ("adjusted_threshold_risk" %in% names(df)) {
      as.numeric(df$adjusted_threshold_risk)
    } else {
      NA_real_
    },
    jackknife_gap = if ("jackknife_gap" %in% names(df)) as.numeric(df$jackknife_gap) else NA_real_,
    jackknife_gap_sd = if ("jackknife_gap_sd" %in% names(df)) as.numeric(df$jackknife_gap_sd) else NA_real_,
    n_jackknife_rotations = if ("n_jackknife_rotations" %in% names(df)) {
      as.integer(df$n_jackknife_rotations)
    } else {
      NA_integer_
    },
    outer_auprc = if ("outer_auprc" %in% names(df)) as.numeric(df$outer_auprc) else NA_real_,
    stringsAsFactors = FALSE
  )
}

cache_row_exists <- function(cache_df, stub_key, risk_target, threshold_method) {
  if (is.null(cache_df) || nrow(cache_df) == 0L) return(FALSE)
  any(
    cache_df$stub_key == stub_key &
      abs(cache_df$requested_target_risk - risk_target) < 1e-12 &
      cache_df$threshold_method == threshold_method,
    na.rm = TRUE
  )
}

# One stub: score once, then sweep (risk x method) without refitting.
fill_stub_eval_cache <- function(
  stub_idx, recipe_jobs, risk_targets, threshold_methods, seed_df,
  two_head_combine = NULL
) {
  stub <- recipe_jobs[[stub_idx]]
  stub_key <- stub_eval_cache_key(stub)
  scored <- score_rejector_stub(stub, two_head_combine = two_head_combine)
  if (!isTRUE(scored$ok)) return(empty_outer_eval_cache_df())
  rows <- list()
  ri <- 1L
  for (tm in threshold_methods) {
    for (tr in risk_targets) {
      if (cache_row_exists(seed_df, stub_key, tr, tm)) next
      out <- outer_eval_from_scored(scored, tr, threshold_method = tm)
      if (is.null(out) || !isTRUE(out$ok)) next
      rows[[ri]] <- outer_eval_out_to_cache_row(stub, tr, tm, out)
      ri <- ri + 1L
    }
  }
  if (length(rows) == 0L) return(empty_outer_eval_cache_df())
  dplyr::bind_rows(rows)
}

# Build (stub x risk x method) results once; seed from per_fold_operating to skip duplicate work.
build_outer_eval_cache <- function(
  recipe_jobs, risk_targets, threshold_methods = THRESHOLD_METHODS,
  seed_operating_df = NULL, two_head_combine = NULL
) {
  if (length(recipe_jobs) == 0L) return(empty_outer_eval_cache_df())
  risk_targets <- unique(as.numeric(risk_targets))
  threshold_methods <- as.character(threshold_methods)
  seed_df <- operating_df_to_cache_rows(seed_operating_df, risk_targets, threshold_methods)
  n_stubs <- length(recipe_jobs)
  mc <- cache_parallel_mc_cores(n_stubs)
  cat(sprintf(
    "    Outer eval cache: %d stubs, %d risks, %d methods, %d seeded rows, score-once sweep, cache mc.cores=%d\n",
    n_stubs, length(risk_targets), length(threshold_methods), nrow(seed_df), mc
  ))
  if (mc > 1L && n_stubs > 1L) {
    stub_parts <- parallel::mclapply(
      seq_len(n_stubs),
      fill_stub_eval_cache,
      recipe_jobs = recipe_jobs,
      risk_targets = risk_targets,
      threshold_methods = threshold_methods,
      seed_df = seed_df,
      two_head_combine = two_head_combine,
      mc.cores = mc,
      mc.preschedule = FALSE
    )
  } else {
    stub_parts <- lapply(
      seq_len(n_stubs),
      fill_stub_eval_cache,
      recipe_jobs = recipe_jobs,
      risk_targets = risk_targets,
      threshold_methods = threshold_methods,
      seed_df = seed_df,
      two_head_combine = two_head_combine
    )
  }
  stub_parts <- stub_parts[vapply(stub_parts, function(x) nrow(x) > 0L, logical(1))]
  if (nrow(seed_df) == 0L && length(stub_parts) == 0L) return(empty_outer_eval_cache_df())
  dplyr::bind_rows(c(list(seed_df), stub_parts))
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

rejection_stratum_counts_twohead <- function(
  pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule, risk_target,
  min_rows = 20L, rejector_mode = "two_head_min"
) {
  combine <- two_head_combine_method(rejector_mode)
  oof_pf <- NULL
  pool_ids <- names(pool_fold_dfs)
  if (combine == "postcal") {
    oof_c <- pool_oof_twohead_combined(
      pool_fold_dfs, pool_ids, y_col, rhs_terms, pool_rule, test_rule, min_rows, combine = "postcal"
    )
    if (is.null(oof_c)) return(list(ok = FALSE))
    oof_pf <- oof_c$postcal_fit
  }
  if (length(pool_ids) < 2L) return(list(ok = FALSE))
  tgt_u <- apply_row_rule(target_df, test_rule)
  if (nrow(tgt_u) < 10L) return(list(ok = FALSE))
  if (!all(c("correct", "is_seen") %in% colnames(tgt_u))) return(list(ok = FALSE))

  thr <- threshold_from_oof_pool_twohead(
    pool_fold_dfs, pool_ids, y_col, rhs_terms, pool_rule, test_rule, risk_target, min_rows,
    combine = combine, postcal_fit = oof_pf
  )
  if (!is.finite(thr)) return(list(ok = FALSE))

  train_df <- bind_rows(pool_fold_dfs)
  train_u <- apply_row_rule(train_df, pool_rule)
  if (nrow(train_u) < min_rows) return(list(ok = FALSE))
  fit_obj <- fit_twohead_models(train_u, rhs_terms, min_rows = min_rows)
  if (is.null(fit_obj)) return(list(ok = FALSE))
  rhs <- fit_obj$rhs_terms
  te2 <- tgt_u[, unique(c(y_col, "correct", "is_seen", rhs)), drop = FALSE]
  keep_te <- stats::complete.cases(te2)
  if (sum(keep_te) < 10L) return(list(ok = FALSE))
  pred_te <- predict_twohead_for_mode(
    fit_obj, te2[keep_te, , drop = FALSE], rejector_mode = rejector_mode, postcal_fit = oof_pf
  )
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

rejection_stratum_counts_rejector <- function(
  pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule, risk_target,
  min_rows = 20L, rejector_mode = "single_head"
) {
  if (is_two_head_rejector(rejector_mode)) {
    rejection_stratum_counts_twohead(
      pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule, risk_target, min_rows,
      rejector_mode
    )
  } else {
    rejection_stratum_counts_singlehead(
      pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule, risk_target, min_rows
    )
  }
}

pct_rejected <- function(n_rejected, n_total) {
  ifelse(!is.finite(n_total) | n_total <= 0, NA_real_, 100 * n_rejected / n_total)
}

build_rejection_stratum_per_fold <- function(recipe_jobs, risk_target) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  rows <- lapply(recipe_jobs, function(stub) {
    rhs_terms <- strsplit(stub$inner_winner_rhs_key, ";", fixed = TRUE)[[1]]
    rejector_mode <- if (!is.null(stub$rejector_mode)) stub$rejector_mode else "single_head"
    cnt <- rejection_stratum_counts_rejector(
      stub$pool_fold_dfs, stub$target_df, "accept_combined", rhs_terms,
      POOL_RULE, TEST_RULE, risk_target, rejector_mode = rejector_mode
    )
    if (is.null(cnt) || !isTRUE(cnt$ok)) return(NULL)
    data.frame(
      label_set = stub$label_set,
      split_type = stub$split_type,
      target_fold = as.character(stub$fold_name),
      scenario_key = SCENARIO_KEY,
      scenario_name = rejector_scenario_name(rejector_mode),
      rejector_mode = rejector_mode,
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
  group_cols <- c("label_set", "split_type", "setting_col", "requested_target_risk")
  if ("rejector_key" %in% names(per_fold_df)) {
    group_cols <- c("label_set", "split_type", "rejector_key", "rejector_label", "setting_col", "requested_target_risk")
  }
  per_fold_df %>%
    mutate(setting_col = setting_column_label(split_type, label_set)) %>%
    group_by(dplyr::across(dplyr::all_of(group_cols))) %>%
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

# Inner winner: (1) mean risk in [target - low_offset, target], else [target, target + high_offset];
# (2) keep top_n_sd lowest sd_risk; (3) max mean_coverage (ties: sd_risk, optional count, rhs_key).
rank_inner_scores <- function(
  df,
  risk_target = TARGET_RISK,
  low_offset = INNER_RISK_BAND_LOW_OFFSET,
  high_offset = INNER_RISK_BAND_HIGH_OFFSET,
  top_n_sd = INNER_RISK_BAND_TOP_N_SD
) {
  if (nrow(df) == 0L) return(df)
  top_n_sd <- as.integer(top_n_sd)
  if (length(top_n_sd) != 1L || top_n_sd < 1L) {
    stop("INNER_RISK_BAND_TOP_N_SD must be a positive integer.")
  }
  df <- df %>% mutate(.rid = row_number())
  ok_mask <- with(df, is.finite(mean_risk) & is.finite(sd_risk) & is.finite(mean_coverage))
  ok <- df[ok_mask, , drop = FALSE]
  if (nrow(ok) == 0L) {
    return(df %>% mutate(inner_rank = NA_integer_, inner_selection_tier = NA_character_, recipe_optional_count = NA_integer_) %>% select(-.rid))
  }
  ok <- ok %>%
    mutate(
      recipe_optional_count = recipe_optional_count_df(ok),
      dist_to_target = abs(mean_risk - risk_target),
      rhs_sort = dplyr::coalesce(rhs_key, "")
    )
  band_low <- risk_target - low_offset
  band_high <- risk_target + high_offset
  in_band <- ok %>% dplyr::filter(mean_risk >= band_low, mean_risk <= risk_target)
  band_tier <- "target_band"
  if (nrow(in_band) == 0L) {
    in_band <- ok %>% dplyr::filter(mean_risk >= risk_target, mean_risk <= band_high)
    band_tier <- "target_band_high_fallback"
  }
  # No recipe in either one-sided band (e.g. all inner risks well above target): use all scored recipes.
  if (nrow(in_band) == 0L) {
    in_band <- ok
    band_tier <- "all_recipes_fallback"
  }

  sd_shortlist <- in_band %>%
    dplyr::arrange(.data$sd_risk, .data$rhs_sort) %>%
    dplyr::slice_head(n = top_n_sd)
  shortlist_ids <- sd_shortlist$.rid
  winner_id <- sd_shortlist %>%
    dplyr::arrange(
      dplyr::desc(.data$mean_coverage),
      .data$sd_risk,
      .data$recipe_optional_count,
      .data$rhs_sort
    ) %>%
    dplyr::slice(1L) %>%
    dplyr::pull(.rid)

  winner_tier <- switch(
    band_tier,
    target_band = "winner",
    target_band_high_fallback = "winner_high_fallback",
    all_recipes_fallback = "winner_all_fallback",
    "winner"
  )
  shortlist_tier <- switch(
    band_tier,
    target_band = "sd_shortlist",
    target_band_high_fallback = "sd_shortlist_high_fallback",
    all_recipes_fallback = "sd_shortlist_all_fallback",
    "sd_shortlist"
  )
  band_rest_tier <- switch(
    band_tier,
    target_band = "target_band",
    target_band_high_fallback = "target_band_high_fallback",
    all_recipes_fallback = "all_recipes_fallback",
    "target_band"
  )
  rin_winner <- sd_shortlist %>%
    dplyr::filter(.rid == winner_id) %>%
    dplyr::mutate(inner_selection_tier = winner_tier)
  rin_shortlist <- sd_shortlist %>%
    dplyr::filter(.rid != winner_id) %>%
    dplyr::mutate(inner_selection_tier = shortlist_tier) %>%
    dplyr::arrange(
      dplyr::desc(.data$mean_coverage),
      .data$sd_risk,
      .data$recipe_optional_count,
      .data$rhs_sort
    )
  rin_band_rest <- in_band %>%
    dplyr::filter(!.rid %in% shortlist_ids) %>%
    dplyr::mutate(inner_selection_tier = band_rest_tier) %>%
    dplyr::arrange(
      .data$sd_risk,
      dplyr::desc(.data$mean_coverage),
      .data$recipe_optional_count,
      .data$rhs_sort
    )
  rout <- ok %>%
    dplyr::filter(!.rid %in% in_band$.rid) %>%
    dplyr::mutate(inner_selection_tier = "outside_band") %>%
    dplyr::arrange(
      .data$dist_to_target,
      .data$sd_risk,
      .data$rhs_sort
    )

  merged <- dplyr::bind_rows(rin_winner, rin_shortlist, rin_band_rest, rout) %>%
    dplyr::mutate(inner_rank = dplyr::row_number()) %>%
    dplyr::select(.rid, inner_rank, inner_selection_tier, recipe_optional_count, dist_to_target)

  df %>%
    dplyr::left_join(merged, by = ".rid") %>%
    dplyr::mutate(
      inner_selection_tier = dplyr::case_when(
        !ok_mask ~ "non_finite_metrics",
        is.na(inner_selection_tier) ~ "non_finite_metrics",
        TRUE ~ inner_selection_tier
      )
    ) %>%
    dplyr::select(-.rid)
}

# Inner winner: lowest mean log-loss (ties: sd_logloss, optional count, rhs_key).
rank_inner_scores_logloss <- function(df) {
  if (nrow(df) == 0L) return(df)
  df <- df %>% mutate(.rid = row_number())
  ok_mask <- with(df, is.finite(mean_logloss))
  ok <- df[ok_mask, , drop = FALSE]
  if (nrow(ok) == 0L) {
    return(df %>% mutate(
      inner_rank = NA_integer_,
      inner_selection_tier = "non_finite_metrics",
      recipe_optional_count = NA_integer_
    ) %>% select(-.rid))
  }
  ok <- ok %>%
    mutate(
      recipe_optional_count = if ("alpha" %in% names(ok)) NA_integer_ else recipe_optional_count_df(ok),
      alpha_sort = if ("alpha" %in% names(ok)) alpha else NA_real_,
      rhs_sort = dplyr::coalesce(rhs_key, "")
    ) %>%
    arrange(mean_logloss, sd_logloss, alpha_sort, recipe_optional_count, rhs_sort) %>%
    mutate(
      inner_rank = row_number(),
      inner_selection_tier = if_else(inner_rank == 1L, "winner", "runner_up")
    ) %>%
    select(.rid, inner_rank, inner_selection_tier, recipe_optional_count)
  df %>%
    left_join(ok, by = ".rid") %>%
    mutate(
      inner_selection_tier = dplyr::case_when(
        !ok_mask ~ "non_finite_metrics",
        is.na(inner_selection_tier) ~ "non_finite_metrics",
        TRUE ~ inner_selection_tier
      )
    ) %>%
    select(-.rid)
}

# Safe for NA / empty keys from per-fold winner rows.
extras_from_rhs_key <- function(key, baseline_terms = BASELINE_TERMS) {
  if (length(key) != 1L || is.na(key) || !nzchar(as.character(key))) return(character(0))
  setdiff(strsplit(as.character(key), ";", fixed = TRUE)[[1]], baseline_terms)
}

# Map augmented ensemble base to combine rule used in outer_cv_analysis.R.
# Final train/test pipeline uses Global_Optimized for the same PoE ensemble.
ensemble_rule_from_base_model <- function(base_model) {
  bm <- as.character(base_model)
  if (identical(bm, "Global_Product_Optimized") || identical(bm, "Global_Optimized")) return("poe")
  if (identical(bm, "Global_Simple_Optimized")) return("simple")
  if (identical(bm, "svm")) return("svm")
  if (identical(bm, "neural_net")) return("dnn")
  stop(sprintf(
    paste0(
      "Unsupported base model: %s (expected Global_Optimized, Global_Product_Optimized, ",
      "Global_Simple_Optimized, svm, or neural_net)."
    ),
    bm
  ))
}

ensemble_rule_feature_name <- function(base_model) {
  rule <- ensemble_rule_from_base_model(base_model)
  if (identical(rule, "poe")) {
    "ensemble_product_of_experts"
  } else if (identical(rule, "simple")) {
    "ensemble_simple_weighted_average"
  } else if (identical(rule, "svm")) {
    "svm"
  } else if (identical(rule, "dnn")) {
    "neural_net"
  } else {
    stop(sprintf("Unsupported ensemble rule: %s", rule))
  }
}

recipe_key_from_parts <- function(base_model, rhs_key) {
  paste(as.character(base_model), as.character(rhs_key), sep = "||")
}

# Fuse per-anchor inner ranks (sum of ranks; ties via max rank then tiebreak anchor).
fuse_ranked_anchor_recipes <- function(ranked_by_anchor, anchor_labels) {
  anchor_labels <- as.character(anchor_labels)
  if (length(anchor_labels) < 1L) {
    stop("INNER_SELECTION_ANCHOR_LABELS must have at least one entry.")
  }
  tiebreak_lbl <- if ("p05" %in% anchor_labels) "p05" else anchor_labels[[1L]]
  rank_cols <- paste0("inner_rank_", anchor_labels)

  fusion <- ranked_by_anchor[[anchor_labels[[1L]]]] %>%
    dplyr::select(recipe_key, base_model, ensemble_rule, rhs_key, inner_rank) %>%
    dplyr::rename(!!rank_cols[[1L]] := inner_rank)
  if (length(anchor_labels) > 1L) {
    for (i in seq_along(anchor_labels)[-1]) {
      lbl <- anchor_labels[[i]]
      fusion <- fusion %>%
        dplyr::inner_join(
          ranked_by_anchor[[lbl]] %>%
            dplyr::select(recipe_key, inner_rank) %>%
            dplyr::rename(!!rank_cols[[i]] := inner_rank),
          by = "recipe_key"
        )
    }
  }
  fusion <- fusion %>%
    dplyr::mutate(
      fusion_rank_sum = rowSums(dplyr::across(dplyr::all_of(rank_cols))),
      fusion_rank_max = do.call(pmax, as.data.frame(dplyr::across(dplyr::all_of(rank_cols))))
    )

  tiebreak_rank_col <- paste0("inner_rank_", tiebreak_lbl)
  fusion_join_cols <- unique(c(
    "recipe_key", setdiff(rank_cols, tiebreak_rank_col), "fusion_rank_sum", "fusion_rank_max"
  ))
  arrange_order <- unique(c(
    "fusion_rank_sum", "fusion_rank_max", tiebreak_rank_col,
    setdiff(rank_cols, tiebreak_rank_col), "base_model", "rhs_key"
  ))

  enriched <- ranked_by_anchor[[tiebreak_lbl]] %>%
    dplyr::inner_join(
      fusion %>% dplyr::select(dplyr::all_of(intersect(fusion_join_cols, names(fusion)))),
      by = "recipe_key"
    ) %>%
    dplyr::rename(!!tiebreak_rank_col := inner_rank) %>%
    dplyr::arrange(dplyr::across(dplyr::all_of(intersect(arrange_order, names(.))))) %>%
    dplyr::mutate(inner_rank = dplyr::row_number())

  list(fusion = fusion, enriched = enriched, rank_cols = rank_cols)
}

# Shared RHS recipe grid for inner scoring (same candidates for all selection methods).
collect_inner_recipe_score_rows <- function(
    pool_fold_dfs_by_model, available_base_models, score_one_recipe) {
  score_rows <- list()
  si <- 1L
  for (base_model in available_base_models) {
    pool_fold_dfs <- pool_fold_dfs_by_model[[base_model]]
    if (is.null(pool_fold_dfs) || length(pool_fold_dfs) == 0L) next
    pool_all <- bind_rows(pool_fold_dfs)
    cand_all <- candidate_terms_from_df(pool_all)
    grid_cap <- if (exists("INNER_RHS_GRID_MAX_RECIPES", inherits = TRUE)) {
      get("INNER_RHS_GRID_MAX_RECIPES", inherits = TRUE)
    } else {
      NULL
    }
    rhs_list <- build_rhs_subsets(cand_all, BASELINE_TERMS, max_recipes = grid_cap)
    for (rhs in rhs_list) {
      inn <- score_one_recipe(pool_fold_dfs, base_model, rhs)
      if (!isTRUE(inn$ok)) next
      rhs_key <- paste(rhs, collapse = ";")
      score_rows[[si]] <- c(
        list(
          scenario_key = SCENARIO_KEY,
          base_model = base_model,
          ensemble_rule = ensemble_rule_from_base_model(base_model),
          rhs_key = rhs_key,
          recipe_key = recipe_key_from_parts(base_model, rhs_key)
        ),
        inn[setdiff(names(inn), "ok")]
      )
      si <- si + 1L
    }
  }
  if (length(score_rows) == 0L) {
    data.frame()
  } else {
    bind_rows(score_rows)
  }
}

# Inner scoring over elastic-net alpha grid (all features; lambda.min via study-blocked cv.glmnet).
collect_inner_enet_alpha_scores <- function(
    pool_fold_dfs_by_model, available_base_models, score_one_alpha, rejector_mode) {
  alphas <- enet_alpha_grid_values()
  score_rows <- list()
  si <- 1L
  for (base_model in available_base_models) {
    pool_fold_dfs <- pool_fold_dfs_by_model[[base_model]]
    if (is.null(pool_fold_dfs) || length(pool_fold_dfs) == 0L) next
    feature_terms <- candidate_terms_from_df(bind_rows(pool_fold_dfs))
    for (alpha in alphas) {
      inn <- score_one_alpha(pool_fold_dfs, base_model, alpha, feature_terms)
      if (!isTRUE(inn$ok)) next
      rk <- sprintf("elasticnet;alpha=%g", alpha)
      score_rows[[si]] <- c(
        list(
          scenario_key = SCENARIO_KEY,
          base_model = base_model,
          ensemble_rule = ensemble_rule_from_base_model(base_model),
          alpha = alpha,
          feature_terms_key = paste(feature_terms, collapse = ";"),
          rhs_key = rk,
          recipe_key = recipe_key_from_parts(base_model, rk)
        ),
        inn[setdiff(names(inn), "ok")]
      )
      si <- si + 1L
    }
  }
  if (length(score_rows) == 0L) data.frame() else bind_rows(score_rows)
}

finalize_outer_fold_winner <- function(
    win, enriched, pool_fold_dfs_by_model, target_df_by_model,
    split_type, label_set, fold_name, risk_target, inner_selection_rule_label,
    rank_cols = character(0), rejector_mode = "single_head",
    operating_risks = NULL) {
  empty_pf <- data.frame()
  winner_base_model <- as.character(win$base_model)
  pool_fold_dfs <- pool_fold_dfs_by_model[[winner_base_model]]
  target_df <- target_df_by_model[[winner_base_model]]

  use_enet <- "alpha" %in% names(win) && is.finite(win$alpha[[1]])
  if (use_enet) {
    feature_terms <- if ("feature_terms_key" %in% names(win)) {
      strsplit(as.character(win$feature_terms_key[[1]]), ";", fixed = TRUE)[[1]]
    } else {
      candidate_terms_from_df(bind_rows(pool_fold_dfs))
    }
    rejector_spec <- rejector_spec_elasticnet(
      win$alpha[[1]], feature_terms, rejector_mode
    )
    pool_fit <- fit_enet_rejector_on_pool(
      pool_fold_dfs, "accept_combined", feature_terms, win$alpha[[1]],
      POOL_RULE, rejector_mode = rejector_mode
    )
    if (is.null(pool_fit)) {
      stop(sprintf(
        "Could not fit elastic-net on outer pool [%s | %s | fold %s | alpha=%g].",
        label_set, split_type, fold_name, win$alpha[[1]]
      ))
    }
    rejector_spec$lambda <- if (is_two_head_rejector(rejector_mode)) {
      c(correct = pool_fit$fit_correct$lambda, ood = pool_fit$fit_ood$lambda)
    } else {
      pool_fit$lambda
    }
    rhs_key <- rejector_spec_rhs_key(rejector_spec)
    recipe_human <- sprintf(
      "%s; elasticnet alpha=%g",
      ensemble_rule_feature_name(winner_base_model),
      win$alpha[[1]]
    )
  } else {
    rejector_spec <- NULL
    rhs_key <- as.character(win$rhs_key)
    rhs_terms <- strsplit(rhs_key, ";", fixed = TRUE)[[1]]
    feat_union <- extras_from_rhs_key(rhs_key)
    ensemble_feat <- ensemble_rule_feature_name(winner_base_model)
    recipe_human <- paste(c(ensemble_feat, feat_union), collapse = ";")
  }

  recipe_job_stub <- list(
    pool_fold_dfs = pool_fold_dfs,
    target_df = target_df,
    pool_fold_dfs_by_model = pool_fold_dfs_by_model,
    target_df_by_model = target_df_by_model,
    base_model = winner_base_model,
    split_type = split_type,
    label_set = label_set,
    fold_name = fold_name,
    inner_winner_rhs_key = rhs_key,
    inner_winner_alpha = if (use_enet) win$alpha[[1]] else NA_real_,
    inner_winner_lambda = if (use_enet) rejector_spec$lambda else NA,
    inner_winner_feature_terms = if (use_enet) feature_terms else NA,
    rejector_spec = rejector_spec,
    rejector_mode = rejector_mode,
    class_balanced_ood = if (is_two_head_rejector(rejector_mode)) {
      class_balanced_ood_setting()
    } else {
      NA
    }
  )
  inner_scores_ranked <- if (nrow(enriched) == 0L) {
    enriched
  } else {
    rej_mode <- rejector_mode
    enriched %>%
      dplyr::mutate(
        label_set = label_set,
        split_type = split_type,
        target_fold = as.character(fold_name),
        scenario_name = rejector_scenario_name(rej_mode),
        rejector_mode = rej_mode,
        inner_selection_fusion_rule = inner_selection_rule_label,
        class_balanced_ood = if (is_two_head_rejector(rej_mode)) {
          class_balanced_ood_setting()
        } else {
          NA
        },
        .before = 1L
      )
  }

  if (is.null(operating_risks)) {
    operating_risks <- risk_target
  }
  operating_risks <- unique(as.numeric(operating_risks))

  per_fold_operating_rows <- list()
  opi <- 1L
  stub_eval_cache <- list()
  for (rt in operating_risks) {
    for (tm in THRESHOLD_METHODS) {
      out <- outer_eval_from_stub(
        recipe_job_stub, rt, threshold_method = tm, eval_cache = stub_eval_cache
      )
      if (is.null(out) || !isTRUE(out$ok)) next
      row <- data.frame(
        label_set = label_set,
        split_type = split_type,
        target_fold = as.character(fold_name),
        scenario_key = SCENARIO_KEY,
        scenario_name = rejector_scenario_name(rejector_mode),
        rejector_mode = rejector_mode,
        threshold_method = tm,
        threshold_method_label = THRESHOLD_METHOD_LABELS[[tm]],
        requested_target_risk = rt,
        requested_target_risk_pct = 100 * rt,
        adjusted_threshold_risk = out$adjusted_threshold_risk,
        jackknife_gap = out$jackknife_gap,
        jackknife_gap_sd = out$jackknife_gap_sd,
        n_jackknife_rotations = out$n_jackknife_rotations,
        inner_winner_rhs_key = rhs_key,
        inner_winner_optional_features = recipe_human,
        inner_winner_ensemble_rule = ensemble_rule_from_base_model(winner_base_model),
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
        abs_risk_gap = abs(out$risk_all_accepted - rt),
        base_model = winner_base_model,
        inner_winner_alpha = if (use_enet) win$alpha[[1]] else NA_real_,
        stringsAsFactors = FALSE
      )
      if (all(c("mean_logloss") %in% names(win))) {
        row$inner_mean_logloss <- win$mean_logloss
        row$inner_sd_logloss <- win$sd_logloss
      }
      if (all(c("mean_auroc") %in% names(win))) {
        row$inner_mean_auroc <- win$mean_auroc
        row$inner_sd_auroc <- win$sd_auroc
      }
      if (all(c("mean_coverage", "mean_risk") %in% names(win))) {
        row$inner_mean_coverage_seen <- win$mean_coverage
        row$inner_mean_risk_all_accepted <- win$mean_risk
        row$inner_sd_coverage_seen <- win$sd_coverage
        row$inner_sd_risk_all_accepted <- win$sd_risk
      }
      per_fold_operating_rows[[opi]] <- row
      opi <- opi + 1L
    }
  }

  per_fold_operating <- if (length(per_fold_operating_rows) == 0L) {
    empty_pf
  } else {
    bind_rows(per_fold_operating_rows)
  }
  per_fold <- if (nrow(per_fold_operating) == 0L) {
    empty_pf
  } else {
    per_fold_operating %>%
      filter(
        abs(requested_target_risk - risk_target) < 1e-12,
        threshold_method == "pooled_oof"
      ) %>%
      select(-abs_risk_gap)
  }

  list(
    per_fold = per_fold,
    per_fold_operating = per_fold_operating,
    inner_scores_ranked = inner_scores_ranked,
    recipe_job_stub = recipe_job_stub
  )
}

# Fixed max-prob logistic rejector; no inner feature or regularization selection.
evaluate_outer_fold_maxprob_only <- function(
    pool_fold_dfs_by_model, target_df_by_model, available_base_models,
    split_type, label_set, fold_name, risk_target, rejector_mode = "single_head") {
  empty_pf <- data.frame()
  empty_in <- data.frame()
  available_base_models <- as.character(available_base_models)
  missing_models <- setdiff(TARGET_BASE_MODELS, available_base_models)
  if (length(missing_models) > 0L) {
    stop(sprintf(
      "Outer-fold job [%s | %s | fold %s] missing ensemble base(s): %s. Expected: %s.",
      label_set, split_type, fold_name,
      paste(missing_models, collapse = ", "),
      paste(TARGET_BASE_MODELS, collapse = ", ")
    ))
  }
  if (length(TARGET_BASE_MODELS) != 1L) {
    stop(sprintf(
      "maxprob-only nested runs require exactly one TARGET_BASE_MODEL (got %s).",
      paste(TARGET_BASE_MODELS, collapse = ", ")
    ))
  }
  base_model <- TARGET_BASE_MODELS[[1L]]
  pool_fold_dfs <- pool_fold_dfs_by_model[[base_model]]
  rhs_terms <- strsplit(BASELINE_ONLY_RHS_KEY, ";", fixed = TRUE)[[1L]]
  ll <- inner_cv_logloss_rejector(
    pool_fold_dfs, "accept_combined", rhs_terms, POOL_RULE, TEST_RULE,
    rejector_mode = rejector_mode
  )
  if (!isTRUE(ll$ok)) {
    stop(sprintf(
      "maxprob inner CV log-loss failed [%s | %s | fold %s | %s].",
      label_set, split_type, fold_name, rejector_mode
    ))
  }
  enriched <- data.frame(
    scenario_key = SCENARIO_KEY,
    base_model = base_model,
    ensemble_rule = ensemble_rule_from_base_model(base_model),
    rhs_key = BASELINE_ONLY_RHS_KEY,
    recipe_key = "maxprob",
    mean_logloss = ll$mean_logloss,
    median_logloss = ll$median_logloss,
    sd_logloss = ll$sd_logloss,
    mean_auroc = ll$mean_auroc,
    median_auroc = ll$median_auroc,
    sd_auroc = ll$sd_auroc,
    inner_rank = 1L,
    inner_selection_tier = "fixed",
    recipe_optional_count = NA_integer_,
    stringsAsFactors = FALSE
  )
  win <- enriched
  operating_risks <- if (exists("OUTER_OPERATING_TARGET_RISKS", inherits = TRUE)) {
    get("OUTER_OPERATING_TARGET_RISKS", inherits = TRUE)
  } else {
    risk_target
  }
  finalize_outer_fold_winner(
    win, enriched, pool_fold_dfs_by_model, target_df_by_model,
    split_type, label_set, fold_name, risk_target,
    inner_selection_rule_label = "max_prob_only_fixed",
    rejector_mode = rejector_mode,
    operating_risks = operating_risks
  )
}

nested_maxprob_only <- function() {
  exists("NESTED_MAXPROB_ONLY", inherits = TRUE) && isTRUE(get("NESTED_MAXPROB_ONLY", inherits = TRUE))
}

# Inner CV risk-band ranking at primary target risk (default 5%).
evaluate_outer_fold_risk_ranking <- function(
    pool_fold_dfs_by_model, target_df_by_model, available_base_models,
    split_type, label_set, fold_name, risk_target, rejector_mode = "single_head") {
  empty_pf <- data.frame()
  empty_in <- data.frame()
  available_base_models <- as.character(available_base_models)
  missing_models <- setdiff(TARGET_BASE_MODELS, available_base_models)
  if (length(missing_models) > 0L) {
    stop(sprintf(
      "Outer-fold job [%s | %s | fold %s] missing ensemble base(s): %s. Expected: %s.",
      label_set, split_type, fold_name,
      paste(missing_models, collapse = ", "),
      paste(TARGET_BASE_MODELS, collapse = ", ")
    ))
  }

  inner_rank_risk <- if (exists("PRIMARY_TABLE_TARGET_RISK", inherits = TRUE)) {
    get("PRIMARY_TABLE_TARGET_RISK", inherits = TRUE)
  } else {
    risk_target
  }

  score_one <- function(pool_fold_dfs, base_model, rhs) {
    inner_cv_strict_rejector(
      pool_fold_dfs, "accept_combined", rhs, POOL_RULE, TEST_RULE, inner_rank_risk,
      rejector_mode = rejector_mode
    )
  }
  sdf <- collect_inner_recipe_score_rows(
    pool_fold_dfs_by_model, available_base_models, score_one
  )
  if (nrow(sdf) == 0L) {
    return(list(
      per_fold = empty_pf,
      per_fold_operating = empty_pf,
      inner_scores_ranked = empty_in,
      recipe_job_stub = NULL
    ))
  }

  enriched <- rank_inner_scores(sdf, inner_rank_risk)
  win <- enriched %>% dplyr::filter(inner_rank == 1L)
  if (nrow(win) == 0L) {
    stop(sprintf(
      "No inner winner for [%s | %s | fold %s | %s]: rank_inner_scores returned no inner_rank==1 (check inner CV scores).",
      label_set, split_type, fold_name, rejector_mode
    ))
  }
  if (nrow(win) != 1L) {
    stop(sprintf(
      "Inner winner not unique for [%s | %s | fold %s | %s]: inner_rank==1 count=%d",
      label_set, split_type, fold_name, rejector_mode, nrow(win)
    ))
  }
  operating_risks <- if (exists("OUTER_OPERATING_TARGET_RISKS", inherits = TRUE)) {
    get("OUTER_OPERATING_TARGET_RISKS", inherits = TRUE)
  } else {
    risk_target
  }
  finalize_outer_fold_winner(
    win, enriched, pool_fold_dfs_by_model, target_df_by_model,
    split_type, label_set, fold_name, risk_target,
    inner_selection_rule_label = sprintf(
      "risk_band_top%d_sd_max_cov_at_%.0f%%",
      INNER_RISK_BAND_TOP_N_SD, 100 * inner_rank_risk
    ),
    rejector_mode = rejector_mode,
    operating_risks = operating_risks
  )
}

# Inner CV: pick elastic-net alpha with lowest mean log-loss (lambda.min via study-blocked cv.glmnet).
evaluate_outer_fold_elasticnet <- function(
    pool_fold_dfs_by_model, target_df_by_model, available_base_models,
    split_type, label_set, fold_name, risk_target, rejector_mode = "single_head") {
  empty_pf <- data.frame()
  empty_in <- data.frame()
  available_base_models <- as.character(available_base_models)
  missing_models <- setdiff(TARGET_BASE_MODELS, available_base_models)
  if (length(missing_models) > 0L) {
    stop(sprintf(
      "Outer-fold job [%s | %s | fold %s] missing ensemble base(s): %s. Expected: %s.",
      label_set, split_type, fold_name,
      paste(missing_models, collapse = ", "),
      paste(TARGET_BASE_MODELS, collapse = ", ")
    ))
  }

  score_one_alpha <- function(pool_fold_dfs, base_model, alpha, feature_terms) {
    inner_cv_logloss_enet(
      pool_fold_dfs, "accept_combined", feature_terms, alpha, POOL_RULE, TEST_RULE,
      rejector_mode = rejector_mode
    )
  }
  sdf <- collect_inner_enet_alpha_scores(
    pool_fold_dfs_by_model, available_base_models, score_one_alpha, rejector_mode
  )
  if (nrow(sdf) == 0L) {
    return(list(
      per_fold = empty_pf,
      per_fold_operating = empty_pf,
      inner_scores_ranked = empty_in,
      recipe_job_stub = NULL
    ))
  }

  enriched <- rank_inner_scores_logloss(sdf)
  win <- enriched %>% dplyr::filter(inner_rank == 1L)
  if (nrow(win) == 0L) {
    stop(sprintf(
      "No inner winner for [%s | %s | fold %s | %s]: rank_inner_scores_logloss returned no inner_rank==1.",
      label_set, split_type, fold_name, rejector_mode
    ))
  }
  if (nrow(win) != 1L) {
    stop(sprintf(
      "Inner winner not unique for [%s | %s | fold %s | %s]: inner_rank==1 count=%d",
      label_set, split_type, fold_name, rejector_mode, nrow(win)
    ))
  }
  operating_risks <- if (exists("OUTER_OPERATING_TARGET_RISKS", inherits = TRUE)) {
    get("OUTER_OPERATING_TARGET_RISKS", inherits = TRUE)
  } else {
    risk_target
  }
  finalize_outer_fold_winner(
    win, enriched, pool_fold_dfs_by_model, target_df_by_model,
    split_type, label_set, fold_name, risk_target,
    inner_selection_rule_label = "elasticnet_lowest_mean_logloss_lambda_min_study_cv",
    rejector_mode = rejector_mode,
    operating_risks = operating_risks
  )
}

inner_selection_method <- function() {
  if (exists("INNER_SELECTION_METHOD", inherits = TRUE)) {
    as.character(get("INNER_SELECTION_METHOD", inherits = TRUE))
  } else {
    "risk_ranking"
  }
}

uses_elasticnet_inner_selection <- function() {
  method <- inner_selection_method()
  method %in% c("elasticnet", "logloss", "logloss_elasticnet")
}

# One outer-fold job for parallel::mclapply (fork on macOS/Linux).
worker_evaluate_outer_fold <- function(job) {
  args <- list(
    pool_fold_dfs_by_model = job$pool_fold_dfs_by_model,
    target_df_by_model = job$target_df_by_model,
    available_base_models = job$available_base_models,
    split_type = job$split_type,
    label_set = job$label_set,
    fold_name = job$fold_name,
    risk_target = job$risk_target,
    rejector_mode = job$rejector_mode
  )
  if (nested_maxprob_only()) {
    do.call(evaluate_outer_fold_maxprob_only, args)
  } else if (uses_elasticnet_inner_selection()) {
    do.call(evaluate_outer_fold_elasticnet, args)
  } else {
    do.call(evaluate_outer_fold_risk_ranking, args)
  }
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

build_rhs_subsets <- function(
    candidate_terms, baseline_terms = BASELINE_TERMS, max_recipes = NULL) {
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
  rhs_list <- lapply(subsets, function(s) c(baseline_terms, s))
  if (!is.null(max_recipes) && length(rhs_list) > as.integer(max_recipes)) {
    rhs_list <- rhs_list[seq_len(as.integer(max_recipes))]
  }
  rhs_list
}

subset_key_from_terms <- function(rhs_terms, baseline_terms = BASELINE_TERMS) {
  paste(setdiff(rhs_terms, baseline_terms), collapse = ";")
}

# Fail fast if any configured ensemble base is missing from augmented fold bundles.
assert_all_target_base_models_in_multivariate_results <- function(results_obj, label_set) {
  mr <- results_obj$multivariate_results
  if (is.null(mr) || is.null(mr[[SCENARIO_KEY]])) {
    stop(sprintf(
      "multivariate_results$%s is NULL (label_set=%s). Run R/outer_cv_analysis.R with left-out augmented folds.",
      SCENARIO_KEY, label_set
    ))
  }
  fam <- mr[[SCENARIO_KEY]]
  missing <- setdiff(TARGET_BASE_MODELS, names(fam))
  if (length(missing) > 0L) {
    stop(sprintf(
      paste0(
        "multivariate_results$%s missing augmented ensemble base(s): %s (label_set=%s). ",
        "Present: %s. Re-run R/outer_cv_analysis.R so Global_Product_Optimized, ",
        "Global_Simple_Optimized, svm, and neural_net are stored under ",
        "multivariate_results$with_leftout_ood_aware."
      ),
      SCENARIO_KEY,
      paste(missing, collapse = ", "),
      label_set,
      if (length(names(fam)) == 0L) "(none)" else paste(names(fam), collapse = ", ")
    ))
  }
  invisible(TRUE)
}

extract_ood_aware_fold_feats <- function(results_obj, base_model, split_type, label_set) {
  bundle <- results_obj$multivariate_results[[SCENARIO_KEY]][[base_model]][[split_type]]
  if (is.null(bundle) || is.null(bundle$fold_matrices) || length(bundle$fold_matrices) < 4L) {
    return(NULL)
  }
  fold_feats <- lapply(bundle$fold_matrices, extract_features)
  for (fn in names(fold_feats)) {
    assert_calibration_terms_available(
      fold_feats[[fn]],
      sprintf(
        "augmented fold %s (%s | %s | %s | %s)",
        fn, label_set, split_type, SCENARIO_KEY, base_model
      )
    )
  }
  fold_feats
}

# Strict nested CV: inner winner; threshold from LOSO-OOF (inner train pool + outer pool).
# Requires >=4 primary outer folds so the pool (all but target) has >=3 studies (>=2 for OOF + 1 val).
run_nested_target_risk_analysis <- function(
  results_obj, label_set, risk_target, rejector_mode = "single_head"
) {
  assert_all_target_base_models_in_multivariate_results(results_obj, label_set)
  available_models <- TARGET_BASE_MODELS
  fam0 <- results_obj$multivariate_results[[SCENARIO_KEY]]
  split_types <- if (exists("OUTER_SPLIT_TYPES", inherits = TRUE)) {
    unique(as.character(get("OUTER_SPLIT_TYPES", inherits = TRUE)))
  } else {
    c("cv", "loso")
  }
  split_types <- split_types[split_types %in% c("cv", "loso")]
  if (length(split_types) == 0L) {
    stop("OUTER_SPLIT_TYPES must include at least one of: cv, loso")
  }
  jobs <- list()
  j_idx <- 1L
  for (split_type in split_types) {
    base_model_fold_feats <- list()
    for (base_model in available_models) {
      if (!base_model %in% names(fam0) || !split_type %in% names(fam0[[base_model]])) {
        next
      }
      ff <- extract_ood_aware_fold_feats(results_obj, base_model, split_type, label_set)
      if (is.null(ff)) {
        stop(sprintf(
          "multivariate_results$%s$%s$%s has fewer than 4 augmented folds (label_set=%s). Re-run R/outer_cv_analysis.R.",
          SCENARIO_KEY, base_model, split_type, label_set
        ))
      }
      base_model_fold_feats[[base_model]] <- ff
    }
    if (length(base_model_fold_feats) == 0L) next
    fold_names <- Reduce(intersect, lapply(base_model_fold_feats, names))
    if (length(fold_names) < 4L) {
      stop(sprintf(
        "Fewer than 4 common outer folds across ensemble bases (%s) for %s | %s (found %d: %s).",
        paste(available_models, collapse = ", "),
        label_set, split_type, length(fold_names),
        if (length(fold_names) == 0L) "(none)" else paste(fold_names, collapse = ", ")
      ))
    }
    cat(sprintf(
      "    [%s] %s (%s | %s): inner winner + outer over %d outer folds (%s; threshold via pool LOSO-OOF)\n",
      label_set, toupper(split_type), rejector_scenario_name(rejector_mode), SCENARIO_KEY,
      length(fold_names), paste(names(base_model_fold_feats), collapse = ", ")
    ))
    for (fold_name in fold_names) {
      jobs[[j_idx]] <- list(
        pool_fold_dfs_by_model = lapply(
          base_model_fold_feats,
          function(ff) ff[setdiff(fold_names, fold_name)]
        ),
        target_df_by_model = lapply(base_model_fold_feats, function(ff) ff[[fold_name]]),
        available_base_models = available_models,
        split_type = split_type,
        label_set = label_set,
        fold_name = fold_name,
        risk_target = risk_target,
        rejector_mode = rejector_mode
      )
      j_idx <- j_idx + 1L
    }
  }

  if (length(jobs) == 0L) {
    stop(sprintf(
      "No reject feature-selection jobs for label_set=%s (need >=4 folds with %s bundles for %s).",
      label_set, SCENARIO_KEY, paste(available_models, collapse = ", ")
    ))
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
  per_fold_operating_rows <- list()
  op_idx <- 1L
  inner_fold_rows <- list()
  ir2 <- 1L
  recipe_jobs <- list()
  rj <- 1L
  for (k in seq_along(ev_list)) {
    ev <- ev_list[[k]]
    if (inherits(ev, "try-error")) {
      stop(sprintf("Outer-fold job %d/%d failed: %s", k, length(ev_list), as.character(ev)))
    }
    if (!is.list(ev) || is.null(ev$per_fold)) {
      stop(sprintf(
        "Outer-fold job %d/%d returned invalid result (expected list with per_fold).",
        k, length(ev_list)
      ))
    }
    if (nrow(ev$per_fold) > 0L) {
      per_fold_rows[[r_idx]] <- ev$per_fold
      r_idx <- r_idx + 1L
    }
    op_df <- if (!is.null(ev$per_fold_operating)) ev$per_fold_operating else data.frame()
    if (nrow(op_df) > 0L) {
      per_fold_operating_rows[[op_idx]] <- op_df
      op_idx <- op_idx + 1L
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
  per_fold_operating_df <- if (length(per_fold_operating_rows) == 0L) {
    data.frame()
  } else {
    bind_rows(per_fold_operating_rows)
  }
  inner_scores_ranked_df <- if (length(inner_fold_rows) == 0L) data.frame() else bind_rows(inner_fold_rows)
  summary_4 <- summarize_four_settings(per_fold_df)
  summary_operating <- summarize_operating_points(
    per_fold_operating_df, n_outer_folds_expected = length(jobs)
  )
  heatmap_long <- build_feature_heatmap_long(per_fold_df)
  list(
    per_fold_df = per_fold_df,
    per_fold_operating_df = per_fold_operating_df,
    summary_4 = summary_4,
    summary_operating = summary_operating,
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

modal_inner_winner_recipe_chr <- function(x) {
  x <- unique(as.character(x))
  x <- x[!is.na(x) & nzchar(x)]
  if (length(x) == 0L) return(NA_character_)
  tab <- sort(table(x), decreasing = TRUE)
  names(tab)[[1]]
}

summarize_operating_points <- function(per_fold_operating_df, n_outer_folds_expected = NULL) {
  if (nrow(per_fold_operating_df) == 0L) return(data.frame())
  group_cols <- c(
    "label_set", "split_type", "requested_target_risk", "requested_target_risk_pct",
    "threshold_method", "threshold_method_label"
  )
  extra_cols <- intersect(
    c("rejector_key", "rejector_label", "rejector_mode", "ensemble_key", "ensemble_label", "config_key"),
    names(per_fold_operating_df)
  )
  group_cols <- c(group_cols, extra_cols)
  if (!"threshold_method" %in% names(per_fold_operating_df)) {
    per_fold_operating_df$threshold_method <- "pooled_oof"
    per_fold_operating_df$threshold_method_label <- THRESHOLD_METHOD_LABELS[["pooled_oof"]]
  }
  n_expected <- if (length(n_outer_folds_expected) == 1L && is.finite(n_outer_folds_expected)) {
    as.integer(n_outer_folds_expected)
  } else {
    NA_integer_
  }
  per_fold_operating_df %>%
    mutate(setting_col = setting_column_label(split_type, label_set)) %>%
    group_by(dplyr::across(dplyr::all_of(c(group_cols, "setting_col")))) %>%
    summarise(
      n_outer_folds = n(),
      n_loso_folds_total = n_expected,
      all_outer_folds = is.na(n_expected) | (n() == n_expected),
      scenario_key = dplyr::first(scenario_key),
      scenario_name = dplyr::first(scenario_name),
      modal_inner_winner_recipe = modal_inner_winner_recipe_chr(inner_winner_optional_features),
      mean_jackknife_gap = if (all_outer_folds) mean(jackknife_gap, na.rm = TRUE) else NA_real_,
      mean_adjusted_threshold_risk = if (all_outer_folds) mean(adjusted_threshold_risk, na.rm = TRUE) else NA_real_,
      mean_abs_risk_gap = if (all_outer_folds) mean(abs_risk_gap, na.rm = TRUE) else NA_real_,
      sd_abs_risk_gap = if (all_outer_folds) stats::sd(abs_risk_gap, na.rm = TRUE) else NA_real_,
      mean_outer_coverage_seen = if (all_outer_folds) mean(outer_coverage_seen, na.rm = TRUE) else NA_real_,
      sd_outer_coverage_seen = if (all_outer_folds) stats::sd(outer_coverage_seen, na.rm = TRUE) else NA_real_,
      mean_outer_risk_all_accepted = if (all_outer_folds) mean(outer_risk_all_accepted, na.rm = TRUE) else NA_real_,
      sd_outer_risk_all_accepted = if (all_outer_folds) stats::sd(outer_risk_all_accepted, na.rm = TRUE) else NA_real_,
      mean_outer_kappa_accepted = if (all_outer_folds) mean(outer_kappa_accepted, na.rm = TRUE) else NA_real_,
      sd_outer_kappa_accepted = if (all_outer_folds) stats::sd(outer_kappa_accepted, na.rm = TRUE) else NA_real_,
      .groups = "drop"
    ) %>%
    arrange(label_set, split_type, threshold_method, requested_target_risk_pct)
}

# Mean absolute risk calibration error across a requested-risk grid.
compute_marce_from_curve <- function(curve_df, n_loso_folds_expected = NULL) {
  if (nrow(curve_df) == 0L) return(data.frame())
  if (!"threshold_method" %in% names(curve_df)) {
    curve_df$threshold_method <- "pooled_oof"
    curve_df$threshold_method_label <- THRESHOLD_METHOD_LABELS[["pooled_oof"]]
  }
  curve_df %>%
    mutate(
      abs_risk_gap_pct = abs(realized_risk_pct - requested_target_risk_pct),
      setting_col = if ("setting_col" %in% names(.)) setting_col else setting_column_label(split_type, label_set)
    ) %>%
    group_by(dplyr::across(dplyr::any_of(c(
      "label_set", "split_type", "setting_col", "threshold_method", "threshold_method_label",
      "rejector_key", "rejector_label", "ensemble_key", "ensemble_label", "config_key", "config_label"
    )))) %>%
    summarise(
      marce_risk_pct_lo = min(requested_target_risk_pct, na.rm = TRUE),
      marce_risk_pct_hi = max(requested_target_risk_pct, na.rm = TRUE),
      n_risk_grid_points = dplyr::n_distinct(requested_target_risk_pct),
      n_outer_folds_marce = {
        nf <- unique(stats::na.omit(.data$n_outer_folds))
        if (length(nf) == 0L) NA_integer_ else as.integer(nf[[1L]])
      },
      n_loso_folds_total = if (!is.null(n_loso_folds_expected)) {
        as.integer(n_loso_folds_expected)
      } else if ("n_loso_folds_total" %in% names(curve_df)) {
        as.integer(dplyr::first(stats::na.omit(.data$n_loso_folds_total)))
      } else {
        NA_integer_
      },
      marce_pct = mean(abs_risk_gap_pct, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      n_outer_folds_curve = .data$n_outer_folds_marce,
      pct_loso_folds_on_marce = dplyr::if_else(
        is.finite(.data$n_loso_folds_total) & .data$n_loso_folds_total > 0L,
        100 * .data$n_outer_folds_marce / .data$n_loso_folds_total,
        NA_real_
      )
    ) %>%
    arrange(label_set, split_type, threshold_method)
}

summarize_four_settings <- function(per_fold_df) {
  if (nrow(per_fold_df) == 0L) return(data.frame())
  group_cols <- c("label_set", "split_type")
  if ("rejector_key" %in% names(per_fold_df)) group_cols <- c(group_cols, "rejector_key", "rejector_label", "rejector_mode")
  per_fold_df %>%
    mutate(
      setting_col = setting_column_label(split_type, label_set)
    ) %>%
    group_by(dplyr::across(dplyr::all_of(c(group_cols, "setting_col")))) %>%
    summarise(
      n_outer_folds = n(),
      scenario_key = dplyr::first(scenario_key),
      scenario_name = dplyr::first(scenario_name),
      # Mode of the exact inner-winning optional-feature recipe across outer folds (not a feature union).
      modal_inner_winner_recipe = modal_inner_winner_recipe_chr(inner_winner_optional_features),
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
  if (!"threshold_method" %in% names(per_fold_df)) {
    per_fold_df$threshold_method <- "pooled_oof"
    per_fold_df$threshold_method_label <- THRESHOLD_METHOD_LABELS[["pooled_oof"]]
  }
  per_fold_df %>%
    mutate(setting_col = setting_column_label(split_type, label_set)) %>%
    group_by(label_set, split_type, setting_col, threshold_method, threshold_method_label) %>%
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
      label_set, split_type, setting_col, threshold_method, threshold_method_label,
      requested_target_risk_pct, n_outer_folds,
      realized_risk_pct, realized_risk_ci95_lo_pct, realized_risk_ci95_hi_pct,
      realized_coverage_seen_pct, realized_coverage_seen_ci95_lo_pct, realized_coverage_seen_ci95_hi_pct
    )
}

outer_eval_from_stub <- function(
  stub, risk_target, fixed_threshold = NULL,
  threshold_method = THRESHOLD_METHODS,
  eval_cache = NULL
) {
  threshold_method <- match.arg(threshold_method, THRESHOLD_METHODS)
  if (!is.null(fixed_threshold)) {
    threshold_risk_target <- NULL
    jackknife_gap <- NA_real_
    jackknife_gap_sd <- NA_real_
    n_jackknife_rotations <- NA_integer_
    adjusted_threshold_risk <- NA_real_
    if (is_jackknife_threshold_method(threshold_method)) {
      jk <- jackknife_pool_risk_gap_from_stub(stub, risk_target, eval_cache = eval_cache)
      if (!isTRUE(jk$ok)) return(list(ok = FALSE))
      threshold_risk_target <- jk$adjusted_risk_target
      jackknife_gap <- jk$jackknife_gap
      jackknife_gap_sd <- jk$jackknife_gap_sd
      n_jackknife_rotations <- jk$n_jackknife_rotations
      adjusted_threshold_risk <- jk$adjusted_risk_target
    }
    rejector_mode <- if (!is.null(stub$rejector_mode)) stub$rejector_mode else "single_head"
    spec <- rejector_spec_from_stub(stub)
    out <- if (identical(spec$kind, "elasticnet")) {
      outer_eval_rejector(
        stub$pool_fold_dfs, stub$target_df, "accept_combined",
        pool_rule = POOL_RULE, test_rule = TEST_RULE, risk_target = risk_target,
        fixed_threshold = fixed_threshold, rejector_mode = rejector_mode, rejector_spec = spec,
        threshold_risk_target = threshold_risk_target
      )
    } else {
      outer_eval_rejector(
        stub$pool_fold_dfs, stub$target_df, "accept_combined", spec$rhs_terms,
        POOL_RULE, TEST_RULE, risk_target, rejector_mode = rejector_mode,
        fixed_threshold = fixed_threshold, threshold_risk_target = threshold_risk_target
      )
    }
    if (is.null(out) || !isTRUE(out$ok)) return(out)
    out$threshold_method <- threshold_method
    out$jackknife_gap <- jackknife_gap
    out$jackknife_gap_sd <- jackknife_gap_sd
    out$n_jackknife_rotations <- n_jackknife_rotations
    out$adjusted_threshold_risk <- adjusted_threshold_risk
    return(out)
  }
  if (is.null(eval_cache)) eval_cache <- list()
  if (is.null(eval_cache$scored)) {
    eval_cache$scored <- score_rejector_stub(stub)
  }
  out <- outer_eval_from_scored(eval_cache$scored, risk_target, threshold_method = threshold_method)
  if (is.null(out) || !isTRUE(out$ok)) return(out)
  out
}

# Threshold from pool LOSO-OOF at risk_target; inner winner fixed per stub.
run_fixed_recipe_outer_eval <- function(
  stub, risk_target, threshold_method = "pooled_oof", eval_cache = NULL
) {
  out <- outer_eval_from_stub(stub, risk_target, threshold_method = threshold_method, eval_cache = eval_cache)
  if (is.null(out) || !isTRUE(out$ok)) return(NULL)
  data.frame(
    label_set = stub$label_set,
    split_type = stub$split_type,
    target_fold = as.character(stub$fold_name),
    threshold_method = threshold_method,
    threshold_method_label = THRESHOLD_METHOD_LABELS[[threshold_method]],
    outer_risk_all_accepted = out$risk_all_accepted,
    outer_coverage_seen = out$coverage_seen,
    outer_kappa_accepted = out$kappa_accepted,
    stringsAsFactors = FALSE
  )
}

cache_rows_for_risk_method <- function(eval_cache_df, risk_target, threshold_method) {
  if (is.null(eval_cache_df) || nrow(eval_cache_df) == 0L) return(eval_cache_df)
  eval_cache_df[
    abs(eval_cache_df$requested_target_risk - risk_target) < 1e-12 &
      as.character(eval_cache_df$threshold_method) == threshold_method,
    ,
    drop = FALSE
  ]
}

# How many requested-risk targets in risk_targets are present in observed_risks (tol match).
count_matched_risk_targets <- function(observed_risks, risk_targets, tol = 1e-12) {
  observed_risks <- unique(as.numeric(observed_risks))
  risk_targets <- unique(as.numeric(risk_targets))
  if (length(risk_targets) == 0L) return(0L)
  sum(vapply(
    risk_targets,
    function(rt) any(abs(observed_risks - rt) < tol),
    logical(1)
  ))
}

outer_fold_group_cols <- function(df) {
  intersect(
    c(
      "label_set", "split_type", "base_model", "rejector_mode", "inner_winner_rhs_key",
      "threshold_method", "config_key", "ensemble_key", "rejector_key"
    ),
    names(df)
  )
}

# Keep only outer folds with a successful eval at every risk target (avoids zigzag means).
filter_to_complete_outer_folds <- function(
  df, risk_targets, risk_col = "requested_target_risk", tol = 1e-12, log_prefix = NULL
) {
  if (is.null(df) || nrow(df) == 0L) return(df)
  if (!risk_col %in% names(df)) {
    stop(sprintf("filter_to_complete_outer_folds: missing column '%s'.", risk_col))
  }
  risk_targets <- unique(as.numeric(risk_targets))
  n_need <- length(risk_targets)
  if (n_need == 0L) return(df)
  group_cols <- outer_fold_group_cols(df)
  fold_cols <- c(group_cols, "target_fold")
  n_folds_before <- dplyr::n_distinct(df[, fold_cols, drop = FALSE])
  fold_ok <- df %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(fold_cols))) %>%
    dplyr::summarise(
      n_risk_hits = count_matched_risk_targets(.data[[risk_col]], risk_targets, tol = tol),
      .groups = "drop"
    ) %>%
    dplyr::filter(.data$n_risk_hits == n_need)
  out <- dplyr::inner_join(
    df,
    fold_ok %>% dplyr::select(dplyr::all_of(fold_cols)),
    by = fold_cols
  )
  n_folds_after <- if (nrow(out) == 0L) 0L else dplyr::n_distinct(out[, fold_cols, drop = FALSE])
  if (!is.null(log_prefix) && n_folds_after < n_folds_before) {
    cat(sprintf(
      "    %s: using %d/%d outer-fold series with eval on all %d risk grid points\n",
      log_prefix, n_folds_after, n_folds_before, n_need
    ))
  }
  out
}

# Rebuild aggregated calibration curve from per-fold rows (e.g. after fixing complete-fold logic).
rebuild_calibration_curve_from_per_fold_df <- function(
  per_fold_df, risk_targets, threshold_methods = THRESHOLD_METHODS
) {
  if (is.null(per_fold_df) || nrow(per_fold_df) == 0L) return(data.frame())
  if (!"requested_target_risk_pct" %in% names(per_fold_df)) {
    stop("rebuild_calibration_curve_from_per_fold_df: expected requested_target_risk_pct.")
  }
  risk_targets <- unique(as.numeric(risk_targets))
  tag_cols <- intersect(
    c(
      "ensemble_key", "ensemble_label", "rejector_key", "rejector_label", "rejector_mode",
      "config_key", "config_label", "ensemble_base"
    ),
    names(per_fold_df)
  )
  group_cols <- intersect(
    c("label_set", "split_type", "config_key", "threshold_method"),
    names(per_fold_df)
  )
  if (length(group_cols) == 0L) {
    stop("rebuild_calibration_curve_from_per_fold_df: missing grouping columns.")
  }
  chunks <- list()
  ci <- 1L
  groups <- per_fold_df %>%
    dplyr::distinct(dplyr::across(dplyr::all_of(group_cols)))
  for (gi in seq_len(nrow(groups))) {
    g <- groups[gi, , drop = FALSE]
    gdf <- per_fold_df
    for (col in group_cols) {
      gdf <- gdf[gdf[[col]] == g[[col, drop = TRUE]], , drop = FALSE]
    }
    if (nrow(gdf) == 0L) next
    tm <- as.character(g$threshold_method[[1]])
    if (!(tm %in% threshold_methods)) next
    gdf <- gdf %>% dplyr::mutate(requested_target_risk = requested_target_risk_pct / 100)
    gdf <- filter_to_complete_outer_folds(
      gdf, risk_targets, risk_col = "requested_target_risk",
      log_prefix = sprintf(
        "Calibration curve rebuild (%s | %s | %s)",
        g$label_set[[1]], g$split_type[[1]], g$config_key[[1]]
      )
    )
    if (nrow(gdf) == 0L) next
    tags <- gdf %>% dplyr::distinct(dplyr::across(dplyr::all_of(tag_cols))) %>% dplyr::slice(1L)
    for (tr in risk_targets) {
      fold_df <- gdf[abs(gdf$requested_target_risk - tr) < 1e-12, , drop = FALSE]
      if (nrow(fold_df) == 0L) next
      sum_df <- summarize_calibration_curve_metrics(
        fold_df %>% dplyr::transmute(
          label_set = .data$label_set,
          split_type = .data$split_type,
          target_fold = .data$target_fold,
          threshold_method = .data$threshold_method,
          threshold_method_label = .data$threshold_method_label,
          outer_risk_all_accepted = .data$realized_risk_pct / 100,
          outer_coverage_seen = .data$realized_coverage_seen_pct / 100,
          outer_kappa_accepted = NA_real_
        ),
        tr
      )
      if (nrow(sum_df) == 0L) next
      for (tc in tag_cols) {
        sum_df[[tc]] <- tags[[tc]][[1]]
      }
      chunks[[ci]] <- sum_df
      ci <- ci + 1L
    }
  }
  if (ci == 1L) data.frame() else dplyr::bind_rows(chunks)
}

filter_per_fold_calibration_to_complete_outer_folds <- function(
  per_fold_df, risk_targets, threshold_methods = THRESHOLD_METHODS
) {
  if (is.null(per_fold_df) || nrow(per_fold_df) == 0L) return(per_fold_df)
  group_cols <- intersect(
    c("label_set", "split_type", "config_key", "threshold_method"),
    names(per_fold_df)
  )
  out <- list()
  oi <- 1L
  groups <- per_fold_df %>% dplyr::distinct(dplyr::across(dplyr::all_of(group_cols)))
  for (gi in seq_len(nrow(groups))) {
    g <- groups[gi, , drop = FALSE]
    gdf <- per_fold_df
    for (col in group_cols) {
      gdf <- gdf[gdf[[col]] == g[[col, drop = TRUE]], , drop = FALSE]
    }
    if (nrow(gdf) == 0L) next
  tm <- as.character(g$threshold_method[[1]])
    if (!(tm %in% threshold_methods)) next
    gdf <- gdf %>% dplyr::mutate(requested_target_risk = requested_target_risk_pct / 100)
    gdf <- filter_to_complete_outer_folds(gdf, risk_targets, risk_col = "requested_target_risk")
    if (nrow(gdf) == 0L) next
    out[[oi]] <- gdf %>% dplyr::select(-dplyr::all_of("requested_target_risk"))
    oi <- oi + 1L
  }
  if (oi == 1L) per_fold_df[0L, , drop = FALSE] else dplyr::bind_rows(out)
}

# Per risk grid point: how many outer folds had a successful eval (before complete-fold filtering).
build_calibration_fold_coverage_from_eval_cache <- function(
  eval_cache_df, risk_targets, threshold_methods = THRESHOLD_METHODS
) {
  if (is.null(eval_cache_df) || nrow(eval_cache_df) == 0L) return(data.frame())
  risk_targets <- unique(as.numeric(risk_targets))
  group_cols <- intersect(
    c(
      "label_set", "split_type", "base_model", "rejector_mode", "inner_winner_rhs_key",
      "threshold_method", "threshold_method_label"
    ),
    names(eval_cache_df)
  )
  chunks <- list()
  ci <- 1L
  for (tm in threshold_methods) {
    for (tr in risk_targets) {
      pf <- cache_rows_for_risk_method(eval_cache_df, tr, tm)
      if (nrow(pf) == 0L) next
      chunks[[ci]] <- pf %>%
        dplyr::group_by(dplyr::across(dplyr::all_of(group_cols))) %>%
        dplyr::summarise(
          requested_target_risk_pct = 100 * tr,
          n_outer_folds_available = dplyr::n_distinct(.data$target_fold),
          .groups = "drop"
        )
      ci <- ci + 1L
    }
  }
  if (ci == 1L) return(data.frame())
  out <- dplyr::bind_rows(chunks)
  out %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(group_cols))) %>%
    dplyr::mutate(
      n_loso_folds_total = max(.data$n_outer_folds_available, na.rm = TRUE),
      pct_folds_available = 100 * .data$n_outer_folds_available / .data$n_loso_folds_total
    ) %>%
    dplyr::ungroup() %>%
    dplyr::arrange(
      .data$label_set, .data$split_type, .data$threshold_method,
      .data$requested_target_risk_pct
    )
}

build_calibration_curve_from_eval_cache <- function(
  eval_cache_df, risk_targets, threshold_methods = THRESHOLD_METHODS,
  fold_completeness = c("across_grid", "per_risk_point"),
  n_outer_folds_expected = NULL
) {
  fold_completeness <- match.arg(fold_completeness)
  if (is.null(eval_cache_df) || nrow(eval_cache_df) == 0L) return(data.frame())
  risk_targets <- unique(as.numeric(risk_targets))
  chunks <- list()
  ci <- 1L
  for (tm in threshold_methods) {
    tm_all <- eval_cache_df[as.character(eval_cache_df$threshold_method) == tm, , drop = FALSE]
    tm_df <- if (fold_completeness == "across_grid") {
      filter_to_complete_outer_folds(
        tm_all, risk_targets, risk_col = "requested_target_risk",
        log_prefix = sprintf("Calibration curve (%s)", tm)
      )
    } else {
      tm_all
    }
    n_expected <- n_outer_folds_expected
    if (is.null(n_expected) && nrow(tm_all) > 0L) {
      n_expected <- dplyr::n_distinct(tm_all$target_fold)
    }
    for (tr in risk_targets) {
      pf <- cache_rows_for_risk_method(
        if (fold_completeness == "across_grid") tm_df else tm_all, tr, tm
      ) %>%
        transmute(
          label_set = .data$label_set,
          split_type = .data$split_type,
          target_fold = .data$target_fold,
          threshold_method = .data$threshold_method,
          threshold_method_label = .data$threshold_method_label,
          outer_risk_all_accepted = .data$outer_risk_all_accepted,
          outer_coverage_seen = .data$outer_coverage_seen,
          outer_kappa_accepted = .data$outer_kappa_accepted
        )
      if (nrow(pf) == 0L) next
      if (fold_completeness == "per_risk_point" && !is.null(n_expected)) {
        if (dplyr::n_distinct(pf$target_fold) < as.integer(n_expected)) next
      }
      sum_df <- summarize_calibration_curve_metrics(pf, tr)
      if (!is.null(n_expected)) {
        sum_df$n_loso_folds_total <- as.integer(n_expected)
      }
      chunks[[ci]] <- sum_df
      ci <- ci + 1L
    }
  }
  if (ci == 1L) data.frame() else dplyr::bind_rows(chunks)
}

build_calibration_curve_per_fold_from_eval_cache <- function(
  eval_cache_df, risk_targets, threshold_methods = THRESHOLD_METHODS,
  fold_completeness = c("across_grid", "per_risk_point"),
  n_outer_folds_expected = NULL
) {
  fold_completeness <- match.arg(fold_completeness)
  if (is.null(eval_cache_df) || nrow(eval_cache_df) == 0L) return(data.frame())
  risk_targets <- unique(as.numeric(risk_targets))
  chunks <- list()
  ci <- 1L
  for (tm in threshold_methods) {
    tm_all <- eval_cache_df[as.character(eval_cache_df$threshold_method) == tm, , drop = FALSE]
    tm_df <- if (fold_completeness == "across_grid") {
      filter_to_complete_outer_folds(
        tm_all, risk_targets, risk_col = "requested_target_risk",
        log_prefix = sprintf("Calibration curve per-fold (%s)", tm)
      )
    } else {
      tm_all
    }
    n_expected <- n_outer_folds_expected
    if (is.null(n_expected) && nrow(tm_all) > 0L) {
      n_expected <- dplyr::n_distinct(tm_all$target_fold)
    }
    for (tr in risk_targets) {
      pf <- cache_rows_for_risk_method(
        if (fold_completeness == "across_grid") tm_df else tm_all, tr, tm
      )
      if (fold_completeness == "per_risk_point" && !is.null(n_expected)) {
        if (dplyr::n_distinct(pf$target_fold) < as.integer(n_expected)) next
      }
      if (nrow(pf) == 0L) next
      chunks[[ci]] <- pf %>%
        mutate(
          setting_col = setting_column_label(split_type, label_set),
          requested_target_risk_pct = 100 * tr,
          realized_risk_pct = 100 * outer_risk_all_accepted,
          realized_coverage_seen_pct = 100 * outer_coverage_seen
        ) %>%
        select(
          label_set, split_type, setting_col, target_fold,
          threshold_method, threshold_method_label,
          requested_target_risk_pct, realized_risk_pct, realized_coverage_seen_pct
        )
      ci <- ci + 1L
    }
  }
  if (ci == 1L) {
    data.frame()
  } else {
    dplyr::bind_rows(chunks) %>%
      arrange(label_set, split_type, threshold_method, target_fold, requested_target_risk_pct)
  }
}

build_operating_from_eval_cache <- function(
  eval_cache_df, operating_risks, recipe_jobs = list(), export_rejector_mode = NULL
) {
  if (is.null(eval_cache_df) || nrow(eval_cache_df) == 0L) return(data.frame())
  stub_lookup <- if (length(recipe_jobs) > 0L) {
    keys <- vapply(recipe_jobs, stub_eval_cache_key, character(1))
    stats::setNames(recipe_jobs, keys)
  } else {
    list()
  }
  rows <- list()
  ri <- 1L
  for (tr in operating_risks) {
    for (tm in THRESHOLD_METHODS) {
      pf <- cache_rows_for_risk_method(eval_cache_df, tr, tm)
      if (nrow(pf) == 0L) next
      for (k in seq_len(nrow(pf))) {
        row <- pf[k, , drop = FALSE]
        stub <- stub_lookup[[row$stub_key[[1]]]]
        rejector_mode <- if (!is.null(export_rejector_mode)) {
          as.character(export_rejector_mode)
        } else {
          row$rejector_mode[[1]]
        }
        rhs_key <- row$inner_winner_rhs_key[[1]]
        recipe_human <- if (!is.null(stub)) recipe_human_from_stub(stub) else rhs_key
        rows[[ri]] <- data.frame(
          label_set = row$label_set,
          split_type = row$split_type,
          target_fold = row$target_fold,
          scenario_key = SCENARIO_KEY,
          scenario_name = rejector_scenario_name(rejector_mode),
          rejector_mode = rejector_mode,
          threshold_method = row$threshold_method,
          threshold_method_label = row$threshold_method_label,
          requested_target_risk = tr,
          requested_target_risk_pct = 100 * tr,
          adjusted_threshold_risk = row$adjusted_threshold_risk,
          jackknife_gap = row$jackknife_gap,
          jackknife_gap_sd = row$jackknife_gap_sd,
          n_jackknife_rotations = row$n_jackknife_rotations,
          inner_winner_rhs_key = rhs_key,
          inner_winner_optional_features = recipe_human,
          inner_winner_ensemble_rule = ensemble_rule_from_base_model(row$base_model),
          outer_n_cal_rotations = if (!is.null(stub)) length(names(stub$pool_fold_dfs)) else NA_integer_,
          threshold_outer_cal_mean = row$threshold,
          threshold_outer_cal_median = row$threshold_median,
          outer_risk_all_accepted = row$outer_risk_all_accepted,
          outer_risk_all_accepted_median = row$outer_risk_all_accepted,
          outer_coverage_seen = row$outer_coverage_seen,
          outer_coverage_seen_median = row$outer_coverage_seen,
          outer_kappa_accepted = row$outer_kappa_accepted,
          outer_kappa_accepted_median = row$outer_kappa_accepted,
          outer_auprc = row$outer_auprc,
          outer_auprc_median = row$outer_auprc,
          abs_risk_gap = abs(row$outer_risk_all_accepted - tr),
          base_model = row$base_model,
          inner_winner_alpha = if (!is.null(stub)) stub$inner_winner_alpha else NA_real_,
          stringsAsFactors = FALSE
        )
        ri <- ri + 1L
      }
    }
  }
  if (length(rows) == 0L) data.frame() else dplyr::bind_rows(rows)
}

recipe_human_from_stub <- function(stub) {
  rhs_key <- as.character(stub$inner_winner_rhs_key)
  if (length(rhs_key) != 1L || is.na(rhs_key) || !nzchar(rhs_key)) return(NA_character_)
  feat_union <- if (grepl("^elasticnet", rhs_key) && !is.null(stub$inner_winner_feature_terms)) {
    stub$inner_winner_feature_terms
  } else {
    extras_from_rhs_key(rhs_key)
  }
  ensemble_feat <- ensemble_rule_feature_name(stub$base_model)
  paste(c(ensemble_feat, feat_union), collapse = ";")
}

build_fixed_rhs_operating_from_eval_cache <- function(
  eval_cache_df, operating_risks, rhs_key
) {
  if (is.null(eval_cache_df) || nrow(eval_cache_df) == 0L) return(data.frame())
  rows <- list()
  ri <- 1L
  for (tr in operating_risks) {
    for (tm in THRESHOLD_METHODS) {
      pf <- cache_rows_for_risk_method(eval_cache_df, tr, tm)
      if (nrow(pf) == 0L) next
      for (k in seq_len(nrow(pf))) {
        row <- pf[k, , drop = FALSE]
        rejector_mode <- row$rejector_mode[[1]]
        rows[[ri]] <- data.frame(
          label_set = row$label_set,
          split_type = row$split_type,
          target_fold = row$target_fold,
          scenario_key = SCENARIO_KEY,
          scenario_name = rejector_scenario_name(rejector_mode),
          rejector_mode = rejector_mode,
          threshold_method = row$threshold_method,
          threshold_method_label = row$threshold_method_label,
          requested_target_risk = tr,
          requested_target_risk_pct = 100 * tr,
          adjusted_threshold_risk = row$adjusted_threshold_risk,
          jackknife_gap = row$jackknife_gap,
          jackknife_gap_sd = row$jackknife_gap_sd,
          n_jackknife_rotations = row$n_jackknife_rotations,
          inner_winner_rhs_key = rhs_key,
          inner_winner_optional_features = sprintf("%s (fixed baseline)", rhs_key),
          inner_winner_ensemble_rule = ensemble_rule_from_base_model(row$base_model),
          outer_n_cal_rotations = NA_integer_,
          threshold_outer_cal_mean = row$threshold,
          threshold_outer_cal_median = row$threshold_median,
          outer_risk_all_accepted = row$outer_risk_all_accepted,
          outer_risk_all_accepted_median = row$outer_risk_all_accepted,
          outer_coverage_seen = row$outer_coverage_seen,
          outer_coverage_seen_median = row$outer_coverage_seen,
          outer_kappa_accepted = row$outer_kappa_accepted,
          outer_kappa_accepted_median = row$outer_kappa_accepted,
          outer_auprc = row$outer_auprc,
          outer_auprc_median = row$outer_auprc,
          abs_risk_gap = abs(row$outer_risk_all_accepted - tr),
          base_model = row$base_model,
          inner_winner_alpha = NA_real_,
          stringsAsFactors = FALSE
        )
        ri <- ri + 1L
      }
    }
  }
  if (length(rows) == 0L) data.frame() else dplyr::bind_rows(rows)
}

build_sample_decisions_from_eval_cache <- function(
  recipe_jobs, risk_target, eval_cache_df, threshold_method = "pooled_oof",
  two_head_combine = NULL
) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  rows <- lapply(recipe_jobs, function(stub) {
    stub_key <- stub_eval_cache_key(stub)
    cache_hit <- eval_cache_df[
      eval_cache_df$stub_key == stub_key &
        abs(eval_cache_df$requested_target_risk - risk_target) < 1e-12 &
        eval_cache_df$threshold_method == threshold_method,
      ,
      drop = FALSE
    ]
    if (nrow(cache_hit) != 1L) return(NULL)
    scored <- score_target_fold_probs_from_stub(stub, two_head_combine = two_head_combine)
    if (is.null(scored)) return(NULL)
    thr <- cache_hit$threshold[[1]]
    scored$threshold <- thr
    scored$requested_target_risk <- risk_target
    scored$requested_target_risk_pct <- 100 * risk_target
    scored$accepted <- as.integer(scored$p_hat >= thr)
    scored$decision <- ifelse(scored$accepted == 1L, "accepted", "rejected")
    scored$base_model <- stub$base_model
    scored$rejector_mode <- if (!is.null(stub$rejector_mode)) stub$rejector_mode else "single_head"
    scored$label_set <- stub$label_set
    scored$split_type <- stub$split_type
    scored$target_fold <- as.character(stub$fold_name)
    scored
  })
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) == 0L) {
    data.frame()
  } else {
    dplyr::bind_rows(rows) %>%
      select(
        label_set, split_type, target_fold, base_model, rejector_mode,
        dplyr::any_of(SAMPLE_ID_COLUMNS),
        true_class, pred_class, correct, is_seen,
        p_hat, threshold, requested_target_risk, requested_target_risk_pct,
        accepted, decision
      )
  }
}

# Curve rows: same inner-winning RHS; threshold method selects pooled OOF vs jackknife-adjusted cutoff.
build_calibration_curve_from_stubs <- function(
  recipe_jobs, risk_targets, threshold_methods = THRESHOLD_METHODS,
  seed_operating_df = NULL
) {
  cache_df <- build_outer_eval_cache(
    recipe_jobs, risk_targets, threshold_methods, seed_operating_df = seed_operating_df
  )
  build_calibration_curve_from_eval_cache(cache_df, risk_targets, threshold_methods)
}

# Clone outer-fold stubs with a fixed RHS (e.g. max_prob-only baseline).
stubs_with_fixed_rhs <- function(recipe_jobs, rhs_key) {
  if (length(recipe_jobs) == 0L) return(list())
  lapply(recipe_jobs, function(stub) {
    stub$inner_winner_rhs_key <- rhs_key
    stub$rejector_spec <- NULL
    stub$inner_winner_alpha <- NA_real_
    stub$inner_winner_lambda <- NA
    stub$inner_winner_feature_terms <- NA
    stub
  })
}

# Outer operating metrics with a fixed GLM RHS (no inner feature selection).
build_fixed_rhs_operating_from_stubs <- function(recipe_jobs, rhs_key, operating_risks) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  stubs <- stubs_with_fixed_rhs(recipe_jobs, rhs_key)
  cache_df <- build_outer_eval_cache(stubs, operating_risks, THRESHOLD_METHODS)
  build_fixed_rhs_operating_from_eval_cache(cache_df, operating_risks, rhs_key)
}

# glmnet coefficients at chosen lambda (intercept excluded).
extract_glmnet_coef_long <- function(ena_fit, head_label) {
  if (is.null(ena_fit) || is.null(ena_fit$cv_fit)) {
    return(data.frame(
      head = character(),
      feature = character(),
      coefficient = numeric(),
      stringsAsFactors = FALSE
    ))
  }
  cm <- as.matrix(stats::coef(ena_fit$cv_fit, s = ena_fit$lambda))
  feat <- rownames(cm)[-1L]
  data.frame(
    head = head_label,
    feature = feat,
    coefficient = as.numeric(cm[-1L, 1L]),
    stringsAsFactors = FALSE
  )
}

# Coefficients from the inner-winner elastic-net refit on each outer-fold train pool.
build_inner_winner_outer_pool_coef_long <- function(recipe_jobs) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  rows <- list()
  ri <- 1L
  for (stub in recipe_jobs) {
    spec <- rejector_spec_from_stub(stub)
    if (is.null(spec) || !identical(spec$kind, "elasticnet")) next
    rejector_mode <- spec$rejector_mode
    feature_terms <- as.character(spec$feature_terms)
    alpha <- spec$alpha
    ena_fit <- fit_enet_rejector_on_pool(
      stub$pool_fold_dfs, "accept_combined", feature_terms, alpha,
      POOL_RULE, rejector_mode = rejector_mode
    )
    if (is.null(ena_fit)) next
    if (is_two_head_rejector(rejector_mode)) {
      coef_df <- bind_rows(
        extract_glmnet_coef_long(ena_fit$fit_correct, "correct_given_id"),
        extract_glmnet_coef_long(ena_fit$fit_ood, "id")
      )
    } else {
      coef_df <- extract_glmnet_coef_long(ena_fit, "accept_combined")
    }
    if (nrow(coef_df) == 0L) next
    coef_df$label_set <- stub$label_set
    coef_df$split_type <- stub$split_type
    coef_df$target_fold <- as.character(stub$fold_name)
    coef_df$base_model <- stub$base_model
    coef_df$rejector_mode <- rejector_mode
    coef_df$inner_winner_alpha <- alpha
    coef_df$is_nonzero <- coef_df$coefficient != 0
    rows[[ri]] <- coef_df
    ri <- ri + 1L
  }
  if (length(rows) == 0L) {
    data.frame()
  } else {
    bind_rows(rows) %>%
      select(
        label_set, split_type, target_fold, base_model, rejector_mode, inner_winner_alpha,
        head, feature, coefficient, is_nonzero
      )
  }
}

# Full-coverage baseline: inner-winning multivariate recipe, accept all (threshold = 0).
evaluate_full_coverage_from_stub <- function(stub) {
  out <- outer_eval_from_stub(stub, risk_target = NULL, fixed_threshold = FULL_COVERAGE_THRESHOLD)
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

# Per outer fold: classifier-only error rate and kappa on all target-fold samples (no rejector).
build_classifier_only_per_fold_from_stubs <- function(recipe_jobs) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  rows <- lapply(recipe_jobs, function(stub) {
    co_row <- evaluate_classifier_only_full_coverage(stub$target_df)
    if (is.null(co_row)) return(NULL)
    co_row$label_set <- stub$label_set
    co_row$split_type <- stub$split_type
    co_row$target_fold <- as.character(stub$fold_name)
    co_row$scenario_key <- SCENARIO_KEY
    co_row
  })
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) == 0L) {
    data.frame()
  } else {
    bind_rows(rows) %>%
      select(
        label_set, split_type, target_fold, scenario_key,
        outer_risk_all_accepted, outer_coverage_seen, outer_kappa_accepted
      )
  }
}

# Mean classifier-only full-coverage metrics across outer folds per ensemble and label set.
summarize_classifier_only_full_coverage <- function(per_fold_df) {
  if (nrow(per_fold_df) == 0L) return(data.frame())
  group_cols <- c("label_set", "split_type")
  ensemble_cols <- intersect(
    c("ensemble_key", "ensemble_label", "ensemble_base"),
    names(per_fold_df)
  )
  group_cols <- c(ensemble_cols, group_cols)
  per_fold_df %>%
    mutate(setting_col = setting_column_label(split_type, label_set)) %>%
    group_by(dplyr::across(dplyr::all_of(c(group_cols, "setting_col")))) %>%
    summarise(
      n_outer_folds = n(),
      scenario_key = dplyr::first(scenario_key),
      mean_outer_coverage_seen = mean(outer_coverage_seen, na.rm = TRUE),
      sd_outer_coverage_seen = stats::sd(outer_coverage_seen, na.rm = TRUE),
      mean_outer_risk_all_accepted = mean(outer_risk_all_accepted, na.rm = TRUE),
      sd_outer_risk_all_accepted = stats::sd(outer_risk_all_accepted, na.rm = TRUE),
      mean_outer_kappa_accepted = mean(outer_kappa_accepted, na.rm = TRUE),
      sd_outer_kappa_accepted = stats::sd(outer_kappa_accepted, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    {
      if (length(ensemble_cols) > 0L) {
        arrange(., dplyr::across(dplyr::all_of(ensemble_cols)), label_set, split_type)
      } else {
        arrange(., label_set, split_type)
      }
    }
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
      scenario_name = if (!is.null(stub$rejector_mode)) {
        rejector_scenario_name(stub$rejector_mode)
      } else {
        SCENARIO_NAME
      },
      rejector_mode = if (!is.null(stub$rejector_mode)) stub$rejector_mode else "single_head",
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
# Score all target-fold samples with pool-fitted rejector (for probability distribution plots).
score_target_fold_probs_rejector <- function(
  pool_fold_dfs, target_df, y_col, rhs_terms, pool_rule, test_rule,
  min_rows = 20L, rejector_mode = "single_head"
) {
  pool_ids <- names(pool_fold_dfs)
  if (length(pool_ids) < 2L) return(NULL)
  tgt_u <- apply_row_rule(target_df, test_rule)
  if (nrow(tgt_u) < 10L) return(NULL)
  if (!all(c("correct", "is_seen") %in% colnames(tgt_u))) return(NULL)

  train_df <- bind_rows(pool_fold_dfs)
  train_u <- apply_row_rule(train_df, pool_rule)
  if (nrow(train_u) < min_rows) return(NULL)

  if (is_two_head_rejector(rejector_mode)) {
    combine <- two_head_combine_method(rejector_mode)
    oof_pf <- NULL
    if (combine == "postcal") {
      oof_c <- pool_oof_twohead_combined(
        pool_fold_dfs, pool_ids, y_col, rhs_terms, pool_rule, test_rule, min_rows, combine = "postcal"
      )
      if (is.null(oof_c)) return(NULL)
      oof_pf <- oof_c$postcal_fit
    }
    fit_obj <- fit_twohead_models(train_u, rhs_terms, min_rows = min_rows)
    if (is.null(fit_obj)) return(NULL)
    rhs <- fit_obj$rhs_terms
    te2 <- tgt_u[, unique(c(y_col, "correct", "is_seen", "true_class", "pred_class", rhs)), drop = FALSE]
    keep_te <- stats::complete.cases(te2)
    if (sum(keep_te) < 10L) return(NULL)
    pred_te <- predict_twohead_for_mode(
      fit_obj, te2[keep_te, , drop = FALSE], rejector_mode = rejector_mode, postcal_fit = oof_pf
    )
  } else {
    fit_obj <- fit_binary_model(train_u, y_col, rhs_terms)
    if (is.null(fit_obj)) return(NULL)
    rhs <- fit_obj$rhs_terms
    te2 <- tgt_u[, unique(c(y_col, "correct", "is_seen", "true_class", "pred_class", rhs)), drop = FALSE]
    keep_te <- complete.cases(te2)
    if (sum(keep_te) < 10L) return(NULL)
    pred_te <- predict_binary_model(fit_obj, te2[keep_te, , drop = FALSE])
  }
  if (is.null(pred_te)) return(NULL)
  row_te <- which(keep_te)[pred_te$row_id]
  out <- data.frame(
    p_hat = pred_te$p_hat,
    correct = as.integer(tgt_u$correct[row_te]),
    is_seen = as.integer(tgt_u$is_seen[row_te]),
    true_class = if ("true_class" %in% colnames(tgt_u)) as.character(tgt_u$true_class[row_te]) else NA_character_,
    pred_class = if ("pred_class" %in% colnames(tgt_u)) as.character(tgt_u$pred_class[row_te]) else NA_character_,
    stringsAsFactors = FALSE
  )
  id_df <- copy_sample_id_cols(tgt_u, row_te)
  if (ncol(id_df) > 0L) out <- bind_cols(out, id_df)
  out
}

score_target_fold_probs_from_stub <- function(stub, two_head_combine = NULL) {
  rejector_mode <- if (!is.null(stub$rejector_mode)) stub$rejector_mode else "single_head"
  scored <- score_rejector_stub(stub, two_head_combine = two_head_combine)
  if (!isTRUE(scored$ok)) return(NULL)
  tgt <- scored$target
  tgt_u <- apply_row_rule(stub$target_df, TEST_RULE)
  out <- data.frame(
    p_hat = tgt$p_hat,
    correct = tgt$correct,
    is_seen = tgt$is_seen,
    true_class = tgt$true_class,
    pred_class = tgt$pred_class,
    stringsAsFactors = FALSE
  )
  id_df <- copy_sample_id_cols(tgt_u, tgt$row_te)
  if (ncol(id_df) > 0L) out <- bind_cols(out, id_df)
  out
}

build_probability_samples_from_stubs <- function(recipe_jobs, two_head_combine = NULL) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  rows <- lapply(recipe_jobs, function(stub) {
    rejector_mode <- if (!is.null(stub$rejector_mode)) stub$rejector_mode else "single_head"
    scored <- score_target_fold_probs_from_stub(stub, two_head_combine = two_head_combine)
    if (is.null(scored)) return(NULL)
    scored$label_set <- stub$label_set
    scored$split_type <- stub$split_type
    scored$target_fold <- as.character(stub$fold_name)
    scored$base_model <- stub$base_model
    scored$rejector_mode <- rejector_mode
    scored
  })
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) == 0L) data.frame() else bind_rows(rows)
}

# Per-sample accept/reject at a requested target risk (threshold from pool LOSO-OOF).
score_target_fold_decisions_from_stub <- function(stub, risk_target) {
  out <- outer_eval_from_stub(stub, risk_target)
  if (is.null(out) || !isTRUE(out$ok)) return(NULL)
  scored <- score_target_fold_probs_from_stub(stub)
  if (is.null(scored)) return(NULL)
  thr <- out$threshold
  scored$threshold <- thr
  scored$requested_target_risk <- risk_target
  scored$requested_target_risk_pct <- 100 * risk_target
  scored$accepted <- as.integer(scored$p_hat >= thr)
  scored$decision <- ifelse(scored$accepted == 1L, "accepted", "rejected")
  scored$base_model <- stub$base_model
  scored$rejector_mode <- if (!is.null(stub$rejector_mode)) stub$rejector_mode else "single_head"
  scored
}

build_sample_decisions_from_stubs <- function(
  recipe_jobs, risk_target, eval_cache_df = NULL, threshold_method = "pooled_oof"
) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  if (!is.null(eval_cache_df) && nrow(eval_cache_df) > 0L) {
    return(build_sample_decisions_from_eval_cache(
      recipe_jobs, risk_target, eval_cache_df, threshold_method = threshold_method
    ))
  }
  rows <- lapply(recipe_jobs, function(stub) {
    scored <- score_target_fold_decisions_from_stub(stub, risk_target)
    if (is.null(scored)) return(NULL)
    scored$label_set <- stub$label_set
    scored$split_type <- stub$split_type
    scored$target_fold <- as.character(stub$fold_name)
    scored
  })
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) == 0L) {
    data.frame()
  } else {
    bind_rows(rows) %>%
      select(
        label_set, split_type, target_fold, base_model, rejector_mode,
        dplyr::any_of(SAMPLE_ID_COLUMNS),
        true_class, pred_class, correct, is_seen,
        p_hat, threshold, requested_target_risk, requested_target_risk_pct,
        accepted, decision
      )
  }
}

build_calibration_curve_per_fold_from_stubs <- function(
  recipe_jobs, risk_targets, threshold_methods = THRESHOLD_METHODS,
  seed_operating_df = NULL, eval_cache_df = NULL
) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  cache_df <- if (!is.null(eval_cache_df)) {
    eval_cache_df
  } else {
    build_outer_eval_cache(
      recipe_jobs, risk_targets, threshold_methods, seed_operating_df = seed_operating_df
    )
  }
  build_calibration_curve_per_fold_from_eval_cache(cache_df, risk_targets, threshold_methods)
}

# LOSO stubs for final deployment: each study fold held out, pool = all other folds.
build_final_deploy_stubs_from_pool_folds <- function(
  pool_fold_dfs, label_set, rejector_mode, base_model = "Global_Optimized",
  rejector_spec = NULL
) {
  fold_names <- names(pool_fold_dfs)
  if (length(fold_names) < 3L) return(list())
  rhs_key <- if (!is.null(rejector_spec)) {
    rejector_spec_rhs_key(rejector_spec)
  } else {
    "max_prob"
  }
  lapply(fold_names, function(hold_id) {
    train_ids <- setdiff(fold_names, hold_id)
    list(
      pool_fold_dfs = pool_fold_dfs[train_ids],
      target_df = pool_fold_dfs[[hold_id]],
      base_model = base_model,
      split_type = "loso",
      label_set = label_set,
      fold_name = hold_id,
      inner_winner_rhs_key = rhs_key,
      inner_winner_alpha = if (!is.null(rejector_spec)) rejector_spec$alpha else NA_real_,
      inner_winner_lambda = if (!is.null(rejector_spec)) rejector_spec$lambda else NA,
      inner_winner_feature_terms = if (!is.null(rejector_spec)) {
        rejector_spec$feature_terms
      } else {
        NA
      },
      rejector_spec = rejector_spec,
      rejector_mode = rejector_mode
    )
  })
}

# Pooled glmnet coef export for python/predict_new_samples.py (model, head, term, estimate, mean_x, sd_x).
export_enet_rejector_coef_df <- function(ena_fit, model_name, rejector_mode) {
  export_one <- function(fit, head_label) {
    if (is.null(fit) || is.null(fit$cv_fit)) {
      return(data.frame(
        model = character(),
        head = character(),
        term = character(),
        estimate = numeric(),
        mean_x = numeric(),
        sd_x = numeric(),
        stringsAsFactors = FALSE
      ))
    }
    raw <- extract_enet_head_raw_coef(fit)
    if (is.null(raw)) {
      stop(sprintf("Could not extract raw glmnet coefficients for head '%s'.", head_label))
    }
    feat_terms <- raw$feature_terms
    if (is.null(feat_terms) || length(feat_terms) == 0L) {
      feat_terms <- fit$feature_terms
      names(raw$beta) <- feat_terms
    }
    terms <- c("(Intercept)", feat_terms)
    data.frame(
      model = model_name,
      head = head_label,
      term = terms,
      estimate = c(raw$intercept, as.numeric(raw$beta)),
      mean_x = c(NA_real_, rep(0, length(feat_terms))),
      sd_x = c(NA_real_, rep(1, length(feat_terms))),
      stringsAsFactors = FALSE
    )
  }
  if (is_two_head_rejector(rejector_mode)) {
    dplyr::bind_rows(
      export_one(ena_fit$fit_correct, "correct_given_id"),
      export_one(ena_fit$fit_ood, "id")
    )
  } else {
    export_one(ena_fit, "accept_combined")
  }
}

# Deploy rejector threshold for pooled elastic-net.
derive_enet_deploy_cutoff <- function(
  pool_fold_dfs, rejector_spec, risk_target,
  two_head_combine = NULL, min_rows = 20L,
  threshold_method = THRESHOLD_METHODS
) {
  threshold_method <- match.arg(threshold_method, THRESHOLD_METHODS)
  rejector_mode <- rejector_spec$rejector_mode
  pool_ids <- names(pool_fold_dfs)
  oof <- pool_oof_enet(
    pool_fold_dfs, pool_ids, "accept_combined",
    rejector_spec$feature_terms, rejector_spec$alpha,
    POOL_RULE, TEST_RULE, min_rows, rejector_mode,
    two_head_combine = two_head_combine
  )
  if (is.null(oof)) return(list(ok = FALSE))
  jk <- if (is_jackknife_threshold_method(threshold_method)) {
    jackknife_pool_risk_gap_rejector(
      pool_fold_dfs, "accept_combined", POOL_RULE, TEST_RULE, risk_target,
      min_rows = min_rows, rejector_mode = rejector_mode, rejector_spec = rejector_spec
    )
  } else {
    NULL
  }
  meta <- merge_jackknife_cutoff_metadata(threshold_method, risk_target, jk)
  if (is.null(meta)) return(list(ok = FALSE))
  tm <- oof_threshold_selection_method(threshold_method)
  thr <- threshold_from_oof_scores(oof, meta$thr_risk, threshold_method = tm)
  if (!is.finite(thr)) return(list(ok = FALSE))
  list(
    ok = TRUE,
    threshold = thr,
    jackknife_gap = meta$jackknife_gap,
    jackknife_gap_sd = meta$jackknife_gap_sd,
    n_jackknife_rotations = meta$n_jackknife_rotations,
    adjusted_threshold_risk = meta$adjusted_threshold_risk
  )
}

derive_enet_jackknife_cutoff <- function(...) {
  do.call(derive_enet_deploy_cutoff, c(list(...), list(threshold_method = "jackknife_adjusted")))
}

# Shared LOSO curve export for final deployment rejectors.
export_final_deploy_risk_curves <- function(
  pool_fold_dfs, label_set, label_set_key, rejector_key, rejector_mode,
  rejector_spec, two_head_combine, merge_suffix, cutoffs_dir,
  base_model = "Global_Optimized", risk_grid, threshold_method, write_curve = TRUE
) {
  stubs <- build_final_deploy_stubs_from_pool_folds(
    pool_fold_dfs, label_set, rejector_mode,
    base_model = base_model, rejector_spec = rejector_spec
  )
  if (length(stubs) < 3L) {
    stop(sprintf("Need >=3 LOSO stubs for risk curves (%s, %s).", label_set_key, rejector_key))
  }
  combine_arg <- if (is_two_head_rejector(rejector_mode)) {
    if (is.null(two_head_combine)) two_head_combine_method(rejector_mode, NULL) else two_head_combine
  } else {
    NULL
  }
  eval_cache <- build_outer_eval_cache(
    stubs, risk_grid, threshold_method, two_head_combine = combine_arg
  )
  deploy_curve <- build_deploy_risk_coverage_curve_from_stubs(
    stubs, risk_grid, target_model = base_model,
    eval_cache_df = eval_cache, threshold_method = threshold_method
  )
  if (nrow(deploy_curve) == 0L) {
    stop(sprintf("Empty deploy risk-coverage curve for %s (%s).", label_set_key, rejector_key))
  }
  deploy_curve$rejector_key <- rejector_key
  deploy_curve$rejector_mode <- rejector_mode
  deploy_curve$label_set_key <- label_set_key
  deploy_curve$threshold_method <- threshold_method
  deploy_curve_path <- file.path(
    cutoffs_dir,
    sprintf("deploy_risk_coverage_curve_%s%s.csv", rejector_key, merge_suffix)
  )
  if (isTRUE(write_curve)) {
    write.csv(deploy_curve, deploy_curve_path, row.names = FALSE)
    cat(sprintf("  [%s] Exported deploy risk-coverage curve (%s): %s\n", label_set_key, rejector_key, deploy_curve_path))
  }

  calibration_curve <- build_calibration_curve_from_eval_cache(
    eval_cache, risk_grid, threshold_method,
    fold_completeness = "per_risk_point",
    n_outer_folds_expected = length(stubs)
  )
  calibration_per_fold <- build_calibration_curve_per_fold_from_eval_cache(
    eval_cache, risk_grid, threshold_method,
    fold_completeness = "per_risk_point",
    n_outer_folds_expected = length(stubs)
  )
  if (nrow(calibration_curve) > 0L) {
    calibration_curve$rejector_key <- rejector_key
    calibration_curve$rejector_mode <- rejector_mode
    calibration_curve$label_set_key <- label_set_key
  }
  if (nrow(calibration_per_fold) > 0L) {
    calibration_per_fold$rejector_key <- rejector_key
    calibration_per_fold$rejector_mode <- rejector_mode
    calibration_per_fold$label_set_key <- label_set_key
  }
  invisible(list(
    deploy_curve_path = deploy_curve_path,
    deploy_curve = deploy_curve,
    calibration_curve = calibration_curve,
    calibration_per_fold = calibration_per_fold
  ))
}

make_deploy_cutoff_row <- function(
    cutoff_out, risk_target, rejector_key, rejector_mode, two_head_combine, rhs_key,
    threshold_method, is_two_head, cutoff_source = "selection_loso",
    base_model = "Global_Optimized") {
  if (!isTRUE(cutoff_out$ok)) {
    stop(sprintf("Could not derive %s cutoff for rejector %s.", threshold_method, rejector_key))
  }
  data.frame(
    model = base_model,
    prob_cutoff = cutoff_out$threshold,
    source = cutoff_source,
    requested_target_risk = risk_target,
    threshold_method = threshold_method,
    adjusted_threshold_risk = cutoff_out$adjusted_threshold_risk,
    jackknife_gap = cutoff_out$jackknife_gap,
    jackknife_gap_sd = cutoff_out$jackknife_gap_sd,
    n_jackknife_rotations = cutoff_out$n_jackknife_rotations,
    rejector_key = rejector_key,
    rejector_mode = rejector_mode,
    two_head_combine = if (is_two_head) two_head_combine else NA_character_,
    rhs_key = rhs_key,
    stringsAsFactors = FALSE
  )
}

write_deploy_cutoffs_all_methods <- function(
    cutoff_rows, label_set_key, rejector_key, merge_suffix, cutoffs_dir) {
  cutoffs_df <- dplyr::bind_rows(cutoff_rows)
  cutoffs_path <- file.path(
    cutoffs_dir,
    sprintf("deploy_cutoffs_%s%s.csv", rejector_key, merge_suffix)
  )
  write.csv(cutoffs_df, cutoffs_path, row.names = FALSE)
  for (i in seq_len(nrow(cutoffs_df))) {
    row <- cutoffs_df[i, ]
    cat(sprintf(
      "  [%s] Exported %s cutoff (%s): %s (threshold=%.4f)\n",
      label_set_key, row$threshold_method, rejector_key, cutoffs_path, row$prob_cutoff
    ))
  }
  cutoffs_path
}

export_final_deploy_risk_curves_all_methods <- function(
    pool_fold_dfs, label_set, label_set_key, rejector_key, rejector_mode,
    rejector_spec, two_head_combine, merge_suffix, cutoffs_dir,
    base_model = "Global_Optimized", risk_grid) {
  deploy_curves <- list()
  calibration_curves <- list()
  calibration_per_fold <- list()
  deploy_curve_path <- file.path(
    cutoffs_dir,
    sprintf("deploy_risk_coverage_curve_%s%s.csv", rejector_key, merge_suffix)
  )
  for (tm in THRESHOLD_METHODS) {
    out <- export_final_deploy_risk_curves(
      pool_fold_dfs, label_set, label_set_key, rejector_key, rejector_mode,
      rejector_spec = rejector_spec,
      two_head_combine = two_head_combine,
      merge_suffix = merge_suffix,
      cutoffs_dir = cutoffs_dir,
      base_model = base_model,
      risk_grid = risk_grid,
      threshold_method = tm,
      write_curve = FALSE
    )
    deploy_curves[[tm]] <- out$deploy_curve
    if (nrow(out$calibration_curve) > 0L) {
      calibration_curves[[tm]] <- out$calibration_curve
    }
    if (nrow(out$calibration_per_fold) > 0L) {
      calibration_per_fold[[tm]] <- out$calibration_per_fold
    }
  }
  deploy_curve <- dplyr::bind_rows(deploy_curves)
  write.csv(deploy_curve, deploy_curve_path, row.names = FALSE)
  cat(sprintf(
    "  [%s] Exported deploy risk-coverage curve (%s, both threshold methods): %s\n",
    label_set_key, rejector_key, deploy_curve_path
  ))
  invisible(list(
    deploy_curve_path = deploy_curve_path,
    calibration_curve = if (length(calibration_curves)) dplyr::bind_rows(calibration_curves) else data.frame(),
    calibration_per_fold = if (length(calibration_per_fold)) dplyr::bind_rows(calibration_per_fold) else data.frame()
  ))
}

# Deployable risk–coverage curve with prob_cutoff per requested risk (for predict_new_samples.py).
build_deploy_risk_coverage_curve_from_stubs <- function(
  recipe_jobs,
  risk_targets,
  target_model = "Global_Optimized",
  eval_cache_df = NULL,
  threshold_method = "pooled_oof"
) {
  if (length(recipe_jobs) == 0L) return(data.frame())
  threshold_method <- match.arg(threshold_method, THRESHOLD_METHODS)
  chunks <- list()
  ci <- 1L
  for (tr in risk_targets) {
    rows <- if (!is.null(eval_cache_df) && nrow(eval_cache_df) > 0L) {
      pf <- cache_rows_for_risk_method(eval_cache_df, tr, threshold_method)
      if (nrow(pf) == 0L) {
        list()
      } else {
        lapply(seq_len(nrow(pf)), function(k) {
          row <- pf[k, , drop = FALSE]
          data.frame(
            label_set = row$label_set,
            split_type = row$split_type,
            target_fold = row$target_fold,
            base_model = row$base_model,
            threshold_outer = row$threshold,
            outer_risk_all_accepted = row$outer_risk_all_accepted,
            outer_coverage_seen = row$outer_coverage_seen,
            stringsAsFactors = FALSE
          )
        })
      }
    } else {
      lapply(recipe_jobs, function(stub) {
        out <- outer_eval_from_stub(stub, tr, threshold_method = threshold_method)
        if (is.null(out) || !isTRUE(out$ok)) return(NULL)
        data.frame(
          label_set = stub$label_set,
          split_type = stub$split_type,
          target_fold = as.character(stub$fold_name),
          base_model = if (!is.null(stub$base_model)) stub$base_model else target_model,
          threshold_outer = out$threshold,
          outer_risk_all_accepted = out$risk_all_accepted,
          outer_coverage_seen = out$coverage_seen,
          stringsAsFactors = FALSE
        )
      })
    }
    rows <- rows[!vapply(rows, is.null, logical(1))]
    if (length(rows) == 0L) next
    pf <- dplyr::bind_rows(rows)
    chunks[[ci]] <- pf %>%
      group_by(.data$label_set, .data$split_type) %>%
      summarise(
        model = dplyr::first(.data$base_model),
        requested_target_risk_pct = 100 * tr,
        prob_cutoff = stats::median(.data$threshold_outer, na.rm = TRUE),
        mean_risk = mean(.data$outer_risk_all_accepted, na.rm = TRUE),
        mean_coverage = mean(.data$outer_coverage_seen, na.rm = TRUE),
        n_outer_folds = dplyr::n(),
        .groups = "drop"
      )
    ci <- ci + 1L
  }
  if (ci == 1L) data.frame() else bind_rows(chunks)
}
