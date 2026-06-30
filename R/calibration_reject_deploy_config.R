# Shared final-deployment config: SVM classifier + SVM rejector recipes (matches calibration_reject_models.R).

DEPLOY_BASE_MODEL <- "svm"
DEPLOY_RHS_TERMS <- "max_prob"
DEPLOY_RISK_TARGET <- 0.05
DEPLOY_RISK_GRID <- DEPLOY_RISK_TARGET

IN_MODEL_FEATURE_TERMS <- c("max_prob", "margin", "entropy", "conformal_set_size_90")
KNN10_FEATURE_TERMS <- c("knn10_mean_d", "knn10_min_d", "knn10_q90_d")
IN_MODEL_KNN10_FEATURE_TERMS <- c(IN_MODEL_FEATURE_TERMS, KNN10_FEATURE_TERMS)
ALL_FEATURE_TERMS <- IN_MODEL_KNN10_FEATURE_TERMS

parse_deploy_feature_terms <- function(feature_terms_key) {
  if (is.na(feature_terms_key) || !nzchar(feature_terms_key)) {
    character(0)
  } else {
    strsplit(feature_terms_key, ";", fixed = TRUE)[[1]]
  }
}

# SVM rejector recipes aligned with calibration_reject_models.R export keys.
DEPLOY_REJECTORS <- tibble::tribble(
  ~rejector_key,        ~rejector_mode,  ~two_head_combine, ~rejector_family, ~feature_terms_key,                                ~params_file_key,
  "svm_single_head",    "single_head",   NA_character_,     "maxprob",        NA_character_,                                     "svm_single_head",
  "svm_ridge_in_model", "single_head",   NA_character_,     "ridge",          "max_prob;margin;entropy;conformal_set_size_90",   "svm_ridge_in_model"
) %>%
  mutate(
    feature_terms = lapply(feature_terms_key, parse_deploy_feature_terms),
    needs_knn10 = vapply(feature_terms, function(ft) any(ft %in% KNN10_FEATURE_TERMS), logical(1L))
  )

INNER_RANK_TARGET_RISK_BY_LABEL_SET <- c(
  full_subtypes = DEPLOY_RISK_TARGET,
  collapsed_classes = DEPLOY_RISK_TARGET,
  collapsed_maxprob = DEPLOY_RISK_TARGET
)

INNER_SELECTION_METHOD <- "elasticnet"
ENET_ALPHA_GRID <- c(0)
GLMNET_MAXIT <- 1000000L
CLASS_BALANCED_OOD <- TRUE
