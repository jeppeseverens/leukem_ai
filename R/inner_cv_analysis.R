# =============================================================================
# Inner Cross-Validation Analysis Script
# 
# This script processes inner cross-validation results from SVM and XGBOOST
# models, extracts the best hyperparameters for each outer fold, and saves
# the results to CSV files.
# =============================================================================

# Load required libraries
library(dplyr)

# =============================================================================
# Helper Functions
# =============================================================================

#' Read CSV files and optionally create 'class' column for OvO classification
#' @param path Path to the CSV file
#' @param OvO Logical indicating if this is One-vs-One classification
#' @return Data frame with processed data
read_and_process <- function(path, OvO = FALSE) {
  df <- read.csv(path)
  if (OvO) {
    df$class <- paste(df$class_0, df$class_1, sep = "_")
    df$kappa <- abs(df$kappa)
  }
  return(df)
}

#' Extract the best hyperparameter(s) per outer fold based on mean kappa
#' @param inner_res Data frame with inner cross-validation results
#' @param type Classification type: "standard", "OvR", or "OvO"
#' @return Data frame with best parameters for each outer fold
get_best_param <- function(inner_res, type) {
  
  # Choose grouping variables based on classification type
  if (type == "standard") {
    # In standard multiclass: one model per outer_fold and params combo
    group <- c("outer_fold", "params")
  } else if (type == "OvR" | type == "OvO") {
    # In One-vs-Rest/One-vs-One: separate model per outer_fold, class, and params
    group <- c("outer_fold", "class", "params")
  }

  # Step 1: Compute mean kappa and accuracy across inner folds for each param set
  best_param <- inner_res %>%
    group_by(across(all_of(group))) %>%
    summarise(mean_kappa = mean(kappa),
              mean_acc = mean(accuracy),
              across(any_of(c("class_0", "class_1")), first),
              .groups = "drop_last")  # maintain grouping for the next filter

  # Step 2: For each (outer_fold, [class]), retain the param set with the highest mean_kappa
  best_param <- best_param %>%
    group_by(across(all_of(group[-length(group)]))) %>%  # drop 'params' from grouping
    filter(mean_kappa == max(mean_kappa)) %>% # keep only best-performing param(s)
    slice(1) # Keep only the first

  # Return the best parameters
  return(best_param)
}

# =============================================================================
# File Paths Configuration
# =============================================================================

# Define file paths for all models and classification types
file_paths <- list(
  # SVM
  inner_res_standard      = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250605_0001/SVM_inner_cv_standard_20250605_0001.csv",
  inner_res_standard_loso = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250603_1642/SVM_inner_cv_loso_standard_20250603_1642.csv",
  inner_res_OvR           = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250605_0001/SVM_inner_cv_OvR_20250605_0001.csv",
  inner_res_OvR_loso      = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250603_1642/SVM_inner_cv_loso_OvR_20250603_1642.csv",
  inner_res_OvO           = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250605_0001/SVM_inner_cv_OvO_20250605_0001.csv",
  inner_res_OvO_loso      = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250603_1642/SVM_inner_cv_loso_OvO_20250603_1642.csv",
  
  # XGBOOST
  inner_res_xgb_standard  = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250605_0252/XGBOOST_inner_cv_standard_20250605_0252.csv",
  inner_res_xgb_OvR       = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250605_0252/XGBOOST_inner_cv_OvR_20250605_0252.csv",
  inner_res_xgb_OvO       = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250605_0252/XGBOOST_inner_cv_OvO_20250605_0252.csv",
  
  inner_res_xgb_standard_loso  = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250604_0242/XGBOOST_inner_cv_loso_standard_20250604_0242.csv",
  inner_res_xgb_OvR_loso       = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250604_0242/XGBOOST_inner_cv_loso_OvR_20250604_0242.csv",
  inner_res_xgb_OvO_loso       = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250604_0242/XGBOOST_inner_cv_loso_OvO_20250604_0242.csv"
)

# =============================================================================
# Data Loading
# =============================================================================

cat("Loading SVM data...\n")
# Read files (SVM)
inner_res_standard      <- read_and_process(file_paths$inner_res_standard)
inner_res_standard_loso <- read_and_process(file_paths$inner_res_standard_loso)
inner_res_OvR           <- read_and_process(file_paths$inner_res_OvR)
inner_res_OvR_loso      <- read_and_process(file_paths$inner_res_OvR_loso)
inner_res_OvO           <- read_and_process(file_paths$inner_res_OvO, OvO = TRUE)
inner_res_OvO_loso      <- read_and_process(file_paths$inner_res_OvO_loso, OvO = TRUE)

cat("Loading XGBOOST data...\n")
# Read files (XGBOOST)
inner_res_xgb_standard      <- read_and_process(file_paths$inner_res_xgb_standard)
inner_res_xgb_OvR           <- read_and_process(file_paths$inner_res_xgb_OvR)
inner_res_xgb_OvO           <- read_and_process(file_paths$inner_res_xgb_OvO, OvO = TRUE)
inner_res_xgb_standard_loso <- read_and_process(file_paths$inner_res_xgb_standard_loso)
inner_res_xgb_OvR_loso      <- read_and_process(file_paths$inner_res_xgb_OvR_loso)
inner_res_xgb_OvO_loso      <- read_and_process(file_paths$inner_res_xgb_OvO_loso, OvO = TRUE)

# =============================================================================
# Extract Best Parameters
# =============================================================================

cat("Extracting best parameters for SVM...\n")
# Get best params (SVM)
best_param_OvO           <- get_best_param(inner_res_OvO, type = "OvO")
best_param_OvR           <- get_best_param(inner_res_OvR, type = "OvR")
best_param_standard      <- get_best_param(inner_res_standard, type = "standard")
best_param_OvO_loso      <- get_best_param(inner_res_OvO_loso, type = "OvO")
best_param_OvR_loso      <- get_best_param(inner_res_OvR_loso, type = "OvR")
best_param_standard_loso <- get_best_param(inner_res_standard_loso, type = "standard")

cat("Extracting best parameters for XGBOOST...\n")
# Get best params (XGBOOST)
best_param_xgb_OvO           <- get_best_param(inner_res_xgb_OvO, type = "OvO")
best_param_xgb_OvR           <- get_best_param(inner_res_xgb_OvR, type = "OvR")
best_param_xgb_standard      <- get_best_param(inner_res_xgb_standard, type = "standard")
best_param_xgb_OvO_loso      <- get_best_param(inner_res_xgb_OvO_loso, type = "OvO")
best_param_xgb_OvR_loso      <- get_best_param(inner_res_xgb_OvR_loso, type = "OvR")
best_param_xgb_standard_loso <- get_best_param(inner_res_xgb_standard_loso, type = "standard")

# =============================================================================
# Save Results
# =============================================================================

cat("Saving SVM results...\n")
# Write to CSV (SVM)
out_dir_svm <- "inner_cv_best_params/SVM"
dir.create(out_dir_svm, recursive = TRUE, showWarnings = FALSE)

write.csv(best_param_OvO,           file = file.path(out_dir_svm, "SVM_best_param_OvO.csv"), row.names = FALSE)
write.csv(best_param_OvR,           file = file.path(out_dir_svm, "SVM_best_param_OvR.csv"), row.names = FALSE)
write.csv(best_param_standard,      file = file.path(out_dir_svm, "SVM_best_param_standard.csv"), row.names = FALSE)
write.csv(best_param_OvO_loso,      file = file.path(out_dir_svm, "SVM_best_param_OvO_loso.csv"), row.names = FALSE)
write.csv(best_param_OvR_loso,      file = file.path(out_dir_svm, "SVM_best_param_OvR_loso.csv"), row.names = FALSE)
write.csv(best_param_standard_loso, file = file.path(out_dir_svm, "SVM_best_param_standard_loso.csv"), row.names = FALSE)

cat("Saving XGBOOST results...\n")
# Write to CSV (XGBOOST)
out_dir_xgb <- "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params/XGBOOST"
dir.create(out_dir_xgb, recursive = TRUE, showWarnings = FALSE)

write.csv(best_param_xgb_OvO,           file = file.path(out_dir_xgb, "XGBOOST_best_param_OvO.csv"), row.names = FALSE)
write.csv(best_param_xgb_OvR,           file = file.path(out_dir_xgb, "XGBOOST_best_param_OvR.csv"), row.names = FALSE)
write.csv(best_param_xgb_standard,      file = file.path(out_dir_xgb, "XGBOOST_best_param_standard.csv"), row.names = FALSE)
write.csv(best_param_xgb_OvO_loso,      file = file.path(out_dir_xgb, "XGBOOST_best_param_OvO_loso.csv"), row.names = FALSE)
write.csv(best_param_xgb_OvR_loso,      file = file.path(out_dir_xgb, "XGBOOST_best_param_OvR_loso.csv"), row.names = FALSE)
write.csv(best_param_xgb_standard_loso, file = file.path(out_dir_xgb, "XGBOOST_best_param_standard_loso.csv"), row.names = FALSE)

cat("Analysis complete! Results saved to:\n")
cat("- SVM: ", out_dir_svm, "\n")
cat("- XGBOOST: ", out_dir_xgb, "\n") 