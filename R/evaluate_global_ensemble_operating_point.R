# Evaluate selected Global Ensemble operating points on prediction-level exports.
# This script is called from analyse_results.Rmd.

suppressPackageStartupMessages({
  library(dplyr)
})

cohen_kappa_binary <- function(truth, pred) {
  if (length(truth) == 0) return(NA_real_)
  labs <- sort(unique(c(truth, pred)))
  if (length(labs) < 2) return(NA_real_)
  tab <- table(factor(truth, levels = labs), factor(pred, levels = labs))
  n <- sum(tab)
  if (n == 0) return(NA_real_)
  po <- sum(diag(tab)) / n
  pe <- sum(rowSums(tab) * colSums(tab)) / (n * n)
  if (abs(1 - pe) < 1e-12) return(NA_real_)
  (po - pe) / (1 - pe)
}

resolve_decision_path <- function(analysis_output_root, method, regime, split_type) {
  if (method == "platt_simple") {
    file.path(
      analysis_output_root,
      if (regime == "with_leftout") "with_leftout" else "standard",
      paste0(
        "global_ensemble_prediction_decisions_",
        if (regime == "with_leftout") "with_leftout" else "standard",
        "_", split_type, ".rds"
      )
    )
  } else if (method == "multivariate") {
    regime_dir <- if (regime == "with_leftout") "with_leftout" else "standard"
    file.path(
      analysis_output_root,
      "ensemble_multivariate",
      regime_dir,
      paste0("global_ensemble_prediction_decisions_", regime_dir, "_", split_type, ".rds")
    )
  } else {
    stop("Unknown method. Use 'platt_simple' or 'multivariate'.")
  }
}

resolve_per_class_paths <- function(analysis_output_root, method, regime, split_type) {
  if (method != "multivariate") {
    return(list(cutoffs = NULL, per_class_rc = NULL))
  }
  regime_dir <- if (regime == "with_leftout") "with_leftout" else "standard"
  base_dir <- file.path(analysis_output_root, "ensemble_multivariate", regime_dir)
  list(
    cutoffs = file.path(base_dir, paste0("per_class_cutoffs_", split_type, ".csv")),
    per_class_rc = file.path(base_dir, paste0("per_class_risk_coverage_", split_type, ".csv"))
  )
}

select_cutoff_from_target <- function(candidate_cutoff_table, method, regime, split_type, target_risk) {
  if (is.null(candidate_cutoff_table) || nrow(candidate_cutoff_table) == 0) {
    stop("candidate_cutoff_table is required when prob_cutoff is not provided.")
  }
  cand <- candidate_cutoff_table %>%
    dplyr::filter(
      method == !!method,
      regime == !!regime,
      split_type == !!split_type,
      curve_source == "heldout"
    ) %>%
    dplyr::mutate(target_distance = abs(target_risk - !!target_risk)) %>%
    dplyr::arrange(target_distance, dplyr::desc(coverage_mono), risk, coverage_sd)
  if (nrow(cand) == 0) {
    stop("No candidate rows found for requested method/regime/split_type.")
  }
  cand$prob_cutoff[1]
}

derive_base_class_cutoffs <- function(analysis_output_root, method, regime, split_type,
                                      target_class_risk = 0.02, model_label = NULL) {
  paths <- resolve_per_class_paths(analysis_output_root, method, regime, split_type)
  if (!is.null(paths$cutoffs) && file.exists(paths$cutoffs)) {
    df <- read.csv(paths$cutoffs, stringsAsFactors = FALSE)
    if (!is.null(model_label) && "model" %in% colnames(df)) {
      df <- df[df$model == model_label, , drop = FALSE]
    }
    if (nrow(df) > 0 && all(c("pred_class", "prob_cutoff") %in% colnames(df))) {
      return(df %>% group_by(pred_class) %>% summarise(base_cutoff = mean(prob_cutoff, na.rm = TRUE), .groups = "drop"))
    }
  }
  if (!is.null(paths$per_class_rc) && file.exists(paths$per_class_rc)) {
    rc <- read.csv(paths$per_class_rc, stringsAsFactors = FALSE)
    if (!is.null(model_label) && "model" %in% colnames(rc)) {
      rc <- rc[rc$model == model_label, , drop = FALSE]
    }
    if (nrow(rc) > 0) {
      rc2 <- rc %>%
        mutate(class_risk = 1 - accuracy, class_coverage = 1 - perc_rejected) %>%
        group_by(pred_class, prob_cutoff) %>%
        summarise(
          class_risk = mean(class_risk, na.rm = TRUE),
          class_coverage = mean(class_coverage, na.rm = TRUE),
          .groups = "drop"
        )
      classes <- unique(rc2$pred_class)
      out <- lapply(classes, function(cls) {
        d <- rc2[rc2$pred_class == cls, , drop = FALSE]
        ok <- d[d$class_risk <= target_class_risk, , drop = FALSE]
        chosen <- if (nrow(ok) > 0) {
          ok[order(-ok$class_coverage, ok$class_risk), , drop = FALSE][1, , drop = FALSE]
        } else {
          d[order(d$class_risk, -d$class_coverage), , drop = FALSE][1, , drop = FALSE]
        }
        data.frame(pred_class = cls, base_cutoff = chosen$prob_cutoff, stringsAsFactors = FALSE)
      })
      return(bind_rows(out))
    }
  }
  stop("Could not derive per-class base cutoffs. Missing per_class_cutoffs/per_class_risk_coverage exports.")
}

evaluate_global_policy <- function(decisions, accepted, policy_label, alpha = NA_real_) {
  decisions <- decisions %>% mutate(accepted = accepted, correct = as.integer(truth == pred))
  acc <- decisions[decisions$accepted, , drop = FALSE]
  # Known classes are those seen in training; left-out rows are intentionally unseen.
  known_mask <- if ("is_leftout" %in% colnames(decisions)) !as.logical(decisions$is_leftout) else rep(TRUE, nrow(decisions))
  unknown_mask <- !known_mask
  n_total <- nrow(decisions)
  n_acc <- nrow(acc)
  n_rej <- n_total - n_acc
  n_known_total <- sum(known_mask, na.rm = TRUE)
  n_known_accepted <- sum(decisions$accepted & known_mask, na.rm = TRUE)
  n_unknown_total <- sum(unknown_mask, na.rm = TRUE)
  n_unknown_rejected <- sum((!decisions$accepted) & unknown_mask, na.rm = TRUE)
  acc_acc <- if (n_acc > 0) mean(acc$correct, na.rm = TRUE) else NA_real_
  data.frame(
    policy = policy_label,
    alpha = alpha,
    n_total = n_total,
    n_accepted = n_acc,
    n_rejected = n_rej,
    coverage = n_acc / pmax(1, n_total),
    n_known_total = n_known_total,
    n_known_accepted = n_known_accepted,
    n_known_rejected = n_known_total - n_known_accepted,
    coverage_known = ifelse(n_known_total > 0, n_known_accepted / n_known_total, NA_real_),
    n_unknown_total = n_unknown_total,
    n_unknown_rejected = n_unknown_rejected,
    pct_rejected_unknown = ifelse(n_rej > 0, n_unknown_rejected / n_rej, NA_real_),
    accepted_accuracy = acc_acc,
    accepted_risk = 1 - acc_acc,
    accepted_kappa = if (n_acc > 1) cohen_kappa_binary(acc$truth, acc$pred) else NA_real_,
    stringsAsFactors = FALSE
  )
}

evaluate_global_candidates <- function(df, cutoff_grid) {
  rows <- lapply(cutoff_grid, function(co) {
    accepted <- df$confidence >= co
    evaluate_global_policy(df, accepted, "global", alpha = NA_real_) %>%
      mutate(prob_cutoff = co)
  })
  bind_rows(rows)
}

select_best_under_risk <- function(candidate_df, target_risk) {
  eligible <- candidate_df %>% filter(accepted_risk <= target_risk)
  if (nrow(eligible) > 0) {
    eligible %>% arrange(desc(coverage), desc(accepted_kappa), accepted_risk) %>% slice(1)
  } else {
    candidate_df %>% arrange(accepted_risk, desc(coverage), desc(accepted_kappa)) %>% slice(1)
  }
}

derive_fold_class_cutoff_maps <- function(analysis_output_root, method, regime, split_type, model_label = NULL) {
  paths <- resolve_per_class_paths(analysis_output_root, method, regime, split_type)
  if (is.null(paths$cutoffs) || !file.exists(paths$cutoffs)) return(NULL)
  df <- read.csv(paths$cutoffs, stringsAsFactors = FALSE)
  if (!is.null(model_label) && "model" %in% colnames(df)) {
    df <- df[df$model == model_label, , drop = FALSE]
  }
  if (nrow(df) == 0 || !"fold" %in% colnames(df) || !"pred_class" %in% colnames(df)) return(NULL)
  fold_maps <- split(df, df$fold)
  lapply(fold_maps, function(x) {
    x %>%
      group_by(pred_class) %>%
      summarise(base_cutoff = mean(prob_cutoff, na.rm = TRUE), .groups = "drop")
  })
}

evaluate_global_ensemble_operating_point <- function(
    analysis_output_root,
    method = c("platt_simple", "multivariate"),
    regime = c("test_only", "with_leftout"),
    split_type = c("cv", "loso"),
    policy = c("global", "class_based"),
    target_risk = NULL,
    prob_cutoff = NULL,
    alpha = 1,
    alpha_grid = seq(0.5, 1.5, by = 0.01),
    risk_cap_grid = seq(0.005, 0.20, by = 0.0025),
    cutoff_grid = seq(0, 1, by = 0.01),
    candidate_cutoff_table = NULL,
    write_outputs = TRUE) {
  method <- match.arg(method)
  regime <- match.arg(regime)
  split_type <- match.arg(split_type)
  policy <- match.arg(policy)

  if (policy == "global" && is.null(prob_cutoff) && is.null(target_risk)) {
    stop("Provide either prob_cutoff or target_risk.")
  }
  # For global policy:
  # - explicit prob_cutoff => fixed cutoff across folds
  # - target_risk only     => cross-fitted per-fold cutoff selection from other folds

  decision_path <- resolve_decision_path(analysis_output_root, method, regime, split_type)
  if (!file.exists(decision_path)) {
    stop(sprintf("Decision export not found: %s", decision_path))
  }
  decisions <- readRDS(decision_path)
  if (nrow(decisions) == 0) stop("Decision table is empty.")
  if (!"pred_class" %in% colnames(decisions)) {
    decisions$pred_class <- decisions$pred
  }

  if (policy == "global") {
    # If explicit cutoff is provided, use it directly. Otherwise do per-fold cross-fitted selection.
    if (!is.null(prob_cutoff)) {
      decisions <- decisions %>%
        mutate(
          accepted = confidence >= prob_cutoff,
          correct = as.integer(truth == pred),
          is_known_class = if ("is_leftout" %in% colnames(.)) !as.logical(is_leftout) else TRUE
        )
      fold_metrics <- decisions %>%
        group_by(fold) %>%
        summarise(
          method = first(method),
          regime = first(regime),
          split_type = first(split_type),
          model = first(model),
          policy = "global",
          prob_cutoff = prob_cutoff,
          n_total = n(),
          n_accepted = sum(accepted, na.rm = TRUE),
          n_rejected = n_total - n_accepted,
          coverage = n_accepted / pmax(1, n_total),
          n_known_total = sum(is_known_class, na.rm = TRUE),
          n_known_accepted = sum(accepted & is_known_class, na.rm = TRUE),
          n_known_rejected = n_known_total - n_known_accepted,
          coverage_known = ifelse(n_known_total > 0, n_known_accepted / n_known_total, NA_real_),
          n_unknown_total = sum(!is_known_class, na.rm = TRUE),
          n_unknown_rejected = sum((!accepted) & (!is_known_class), na.rm = TRUE),
          pct_rejected_unknown = ifelse(n_rejected > 0, n_unknown_rejected / n_rejected, NA_real_),
          accepted_accuracy = ifelse(n_accepted > 0, mean(correct[accepted], na.rm = TRUE), NA_real_),
          accepted_risk = 1 - accepted_accuracy,
          accepted_kappa = ifelse(n_accepted > 1, cohen_kappa_binary(truth[accepted], pred[accepted]), NA_real_),
          .groups = "drop"
        )
      pooled <- evaluate_global_policy(decisions, decisions$accepted, "global", alpha = NA_real_) %>%
        mutate(
          method = method,
          regime = regime,
          split_type = split_type,
          model = unique(decisions$model)[1],
          prob_cutoff = prob_cutoff
        )
      out <- list(selected_cutoff = prob_cutoff, fold_metrics = fold_metrics, pooled_metrics = pooled)
    } else {
      # Cross-fitted selection: per held-out fold, choose cutoff from other folds only.
      fold_ids <- sort(unique(decisions$fold))
      fold_rows <- list()
      pooled_decisions <- list()
      sel_rows <- list()
      idx <- 1L
      for (f in fold_ids) {
        train_df <- decisions[decisions$fold != f, , drop = FALSE]
        test_df <- decisions[decisions$fold == f, , drop = FALSE]
        cand <- evaluate_global_candidates(train_df, cutoff_grid)
        chosen <- select_best_under_risk(cand, target_risk)
        co <- chosen$prob_cutoff[1]

        test_df$accepted <- test_df$confidence >= co
        test_df$correct <- as.integer(test_df$truth == test_df$pred)
        ev <- evaluate_global_policy(test_df, test_df$accepted, "global", alpha = NA_real_) %>%
          mutate(
            method = method,
            regime = regime,
            split_type = split_type,
            model = unique(decisions$model)[1],
            fold = f,
            selected_prob_cutoff = co,
            target_risk = target_risk
          )
        fold_rows[[idx]] <- ev
        pooled_decisions[[idx]] <- test_df
        sel_rows[[idx]] <- chosen %>% mutate(fold = f, target_risk = target_risk)
        idx <- idx + 1L
      }
      fold_metrics <- bind_rows(fold_rows)
      pooled_df <- bind_rows(pooled_decisions)
      pooled <- evaluate_global_policy(pooled_df, pooled_df$accepted, "global", alpha = NA_real_) %>%
        mutate(
          method = method,
          regime = regime,
          split_type = split_type,
          model = unique(decisions$model)[1],
          target_risk = target_risk
        )
      out <- list(
        selection_by_fold = bind_rows(sel_rows),
        fold_metrics = fold_metrics,
        pooled_metrics = pooled
      )
    }
  } else {
    fold_maps <- derive_fold_class_cutoff_maps(
      analysis_output_root = analysis_output_root,
      method = method,
      regime = regime,
      split_type = split_type,
      model_label = unique(decisions$model)[1]
    )
    base_map_global <- derive_base_class_cutoffs(
      analysis_output_root = analysis_output_root,
      method = method,
      regime = regime,
      split_type = split_type,
      model_label = unique(decisions$model)[1]
    )

    if (!is.null(target_risk)) {
      # Cross-fitted alpha selection per held-out fold using other folds.
      fold_ids <- sort(unique(decisions$fold))
      fold_rows <- list()
      pooled_decisions <- list()
      cand_rows <- list()
      best_rows <- list()
      idx <- 1L
      for (f in fold_ids) {
        train_df <- decisions[decisions$fold != f, , drop = FALSE]
        test_df <- decisions[decisions$fold == f, , drop = FALSE]
        map_f <- if (!is.null(fold_maps) && as.character(f) %in% names(fold_maps)) fold_maps[[as.character(f)]] else base_map_global
        train_df <- train_df %>% left_join(map_f, by = c("pred_class"))
        test_df <- test_df %>% left_join(map_f, by = c("pred_class"))
        train_df$base_cutoff[is.na(train_df$base_cutoff)] <- 0.5
        test_df$base_cutoff[is.na(test_df$base_cutoff)] <- 0.5

        fold_candidates <- lapply(alpha_grid, function(a) {
          co <- pmin(1, pmax(0, a * train_df$base_cutoff))
          evaluate_global_policy(train_df, train_df$confidence >= co, "class_based", a) %>%
            mutate(fold = f, method = method, regime = regime, split_type = split_type, model = unique(decisions$model)[1])
        }) %>% bind_rows()
        chosen <- select_best_under_risk(fold_candidates, target_risk)
        a_star <- chosen$alpha[1]

        test_cut <- pmin(1, pmax(0, a_star * test_df$base_cutoff))
        test_df$accepted <- test_df$confidence >= test_cut
        test_df$correct <- as.integer(test_df$truth == test_df$pred)
        ev <- evaluate_global_policy(test_df, test_df$accepted, "class_based", a_star) %>%
          mutate(
            method = method,
            regime = regime,
            split_type = split_type,
            model = unique(decisions$model)[1],
            fold = f,
            target_risk = target_risk
          )
        fold_rows[[idx]] <- ev
        pooled_decisions[[idx]] <- test_df
        cand_rows[[idx]] <- fold_candidates
        best_rows[[idx]] <- chosen %>% mutate(target_risk = target_risk)
        idx <- idx + 1L
      }
      fold_metrics <- bind_rows(fold_rows)
      pooled_df <- bind_rows(pooled_decisions)
      pooled <- evaluate_global_policy(pooled_df, pooled_df$accepted, "class_based", alpha = NA_real_) %>%
        mutate(
          method = method,
          regime = regime,
          split_type = split_type,
          model = unique(decisions$model)[1],
          target_risk = target_risk
        )
      candidate_points <- bind_rows(cand_rows)
      best_points <- bind_rows(best_rows)
      out <- list(
        base_class_cutoffs = base_map_global,
        candidate_points = candidate_points,
        selected_points_by_risk_cap = best_points,
        fold_metrics = fold_metrics,
        pooled_metrics = pooled
      )
    } else {
      decisions <- decisions %>% left_join(base_map_global, by = c("pred_class"))
      decisions$base_cutoff[is.na(decisions$base_cutoff)] <- 0.5
      per_row_cutoff <- pmin(1, pmax(0, alpha * decisions$base_cutoff))
      decisions$accepted <- decisions$confidence >= per_row_cutoff
      decisions$correct <- as.integer(decisions$truth == decisions$pred)
      decisions$is_known_class <- if ("is_leftout" %in% colnames(decisions)) !as.logical(decisions$is_leftout) else TRUE
      fold_metrics <- decisions %>%
        group_by(fold) %>%
        summarise(
          method = first(method),
          regime = first(regime),
          split_type = first(split_type),
          model = first(model),
          policy = "class_based",
          alpha = alpha,
          n_total = n(),
          n_accepted = sum(accepted, na.rm = TRUE),
          n_rejected = n_total - n_accepted,
          coverage = n_accepted / pmax(1, n_total),
          n_known_total = sum(is_known_class, na.rm = TRUE),
          n_known_accepted = sum(accepted & is_known_class, na.rm = TRUE),
          n_known_rejected = n_known_total - n_known_accepted,
          coverage_known = ifelse(n_known_total > 0, n_known_accepted / n_known_total, NA_real_),
          n_unknown_total = sum(!is_known_class, na.rm = TRUE),
          n_unknown_rejected = sum((!accepted) & (!is_known_class), na.rm = TRUE),
          pct_rejected_unknown = ifelse(n_rejected > 0, n_unknown_rejected / n_rejected, NA_real_),
          accepted_accuracy = ifelse(n_accepted > 0, mean(correct[accepted], na.rm = TRUE), NA_real_),
          accepted_risk = 1 - accepted_accuracy,
          accepted_kappa = ifelse(n_accepted > 1, cohen_kappa_binary(truth[accepted], pred[accepted]), NA_real_),
          .groups = "drop"
        )
      pooled <- evaluate_global_policy(decisions, decisions$accepted, "class_based", alpha) %>%
        mutate(
          method = method,
          regime = regime,
          split_type = split_type,
          model = unique(decisions$model)[1]
        )
      out <- list(selected_alpha = alpha, base_class_cutoffs = base_map_global, fold_metrics = fold_metrics, pooled_metrics = pooled)
    }
  }

  if (write_outputs) {
    out_dir <- file.path(analysis_output_root, "global_ensemble_selection")
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    if (policy == "global") {
      suffix <- if (!is.null(prob_cutoff)) {
        paste(method, regime, split_type, "global", sprintf("cutoff_%.3f", prob_cutoff), sep = "_")
      } else {
        paste(method, regime, split_type, "global_crossfit", sprintf("target_%.3f", target_risk), sep = "_")
      }
      write.csv(out$fold_metrics, file.path(out_dir, paste0("operating_point_eval_fold_", suffix, ".csv")), row.names = FALSE)
      write.csv(out$pooled_metrics, file.path(out_dir, paste0("operating_point_eval_pooled_", suffix, ".csv")), row.names = FALSE)
      if (!is.null(out$selection_by_fold)) {
        write.csv(out$selection_by_fold, file.path(out_dir, paste0("operating_point_selection_by_fold_", suffix, ".csv")), row.names = FALSE)
      }
    } else {
      suffix <- if (!is.null(out$selected_alpha)) {
        paste(method, regime, split_type, "class_based", sprintf("alpha_%.3f", out$selected_alpha), sep = "_")
      } else {
        paste(method, regime, split_type, "class_based_crossfit", sprintf("target_%.3f", target_risk), sep = "_")
      }
      write.csv(out$fold_metrics, file.path(out_dir, paste0("operating_point_eval_fold_", suffix, ".csv")), row.names = FALSE)
      write.csv(out$pooled_metrics, file.path(out_dir, paste0("operating_point_eval_pooled_", suffix, ".csv")), row.names = FALSE)
      write.csv(out$base_class_cutoffs, file.path(out_dir, paste0("class_based_base_cutoffs_", method, "_", regime, "_", split_type, ".csv")), row.names = FALSE)
      if (!is.null(out$candidate_points)) {
        write.csv(out$candidate_points, file.path(out_dir, paste0("class_based_candidate_points_", method, "_", regime, "_", split_type, ".csv")), row.names = FALSE)
      }
      if (!is.null(out$selected_points_by_risk_cap)) {
        write.csv(out$selected_points_by_risk_cap, file.path(out_dir, paste0("class_based_selected_points_by_risk_cap_", method, "_", regime, "_", split_type, ".csv")), row.names = FALSE)
      }
    }
  }

  out
}
