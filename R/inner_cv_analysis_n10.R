# Inner Cross-Validation Analysis Script
# 
# This script processes inner cross-validation results from SVM and XGBOOST
# models, extracts the best hyperparameters for each outer fold, and saves
# the results to CSV files.
# =============================================================================

# Load required libraries
library(dplyr)
library(stringr)
# =============================================================================
# Helper Functions
# =============================================================================

merge_classes <- function(x){
  x[grepl("MDS|TP53|GATA2", x)] <- "AML, MDSr"
  x[grepl("KMT2A", x) & !grepl("MLLT3", x)] <- "other KMT2A"
  #x[grepl("RBM15|PICALM|ETV6|KAT6A|FUS..ERG|CBFA", x)] <- "rare"
  x
}

clean_probs_str <- function(probs_str) {
  probs_str <- probs_str %>%
    str_replace_all(",|\\[|\\]|\\{|\\}|\\\n", "") %>%  # remove brackets and line breaks
    str_squish() %>%                                 # collapse multiple spaces
    str_split(" ") %>%                               # split by space
    unlist()
  as.numeric(probs_str)
}

clean_y_str <- function(probs_str) {
  probs_str %>%
    str_replace_all(",|\\[|\\]|\\{|\\}|\\\n", "") %>%  # remove brackets and line breaks
    str_squish() %>%                                 # collapse multiple spaces
    str_split(" ") %>%                               # split by space
    unlist() %>%
    as.numeric()
}

text_to_matrix <- function(raw_text, ncol) {
  cleaned_text <- raw_text %>%
    str_replace_all(",|\\[|\\]|\\{|\\}|\\\n", "") %>% str_squish()
  
  num_vec <- as.numeric(unlist(strsplit(cleaned_text, " ")))
  mat <- matrix(num_vec, ncol = ncol, byrow = TRUE)
  mat
}

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

add_labels <- function(df){
  df$class_label <- NA
  for (i in 1:nrow(df)){
    class_i <- df$class[i]
    label <- label_mapping$Label[class_i + 1]
    df$class_label[i] <- label
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
  inner_res_svm_OvR           = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250611_1318/SVM_inner_cv_OvR_20250611_1318.csv",
  inner_res_svm_OvR_loso      = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250611_1927/SVM_inner_cv_loso_OvR_20250611_1927.csv",
  inner_res_xgb_OvR       = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250612_0024/XGBOOST_inner_cv_OvR_20250612_0024.csv",
  inner_res_xgb_OvR_loso       = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250612_0516/XGBOOST_inner_cv_loso_OvR_20250612_0516.csv" # ,
  # inner_res_nn_standard       = "/Users/jsevere2/Documents/AML_PhD/predictor_out/NN/20250612_1000/NN_inner_cv_standard_20250612_1000.csv",
  # inner_res_nn_standard_loso       = "/Users/jsevere2/Documents/AML_PhD/predictor_out/NN/20250613_0048/NN_inner_cv_loso_standard_20250613_0048.csv"
)

# =============================================================================
# Data Loading
# =============================================================================

cat("Loading SVM data...\n")
# Read files (SVM)
inner_res_svm_OvR           <- read_and_process(file_paths$inner_res_svm_OvR)
inner_res_svm_OvR_loso      <- read_and_process(file_paths$inner_res_svm_OvR_loso)

cat("Loading XGBOOST data...\n")
# Read files (XGBOOST)
inner_res_xgb_OvR           <- read_and_process(file_paths$inner_res_xgb_OvR)
inner_res_xgb_OvR_loso      <- read_and_process(file_paths$inner_res_xgb_OvR_loso)

cat("Loading NN data...\n")
# Read files (NN)
# inner_res_nn_standard           <- read_and_process(file_paths$inner_res_nn_standard)
# inner_res_nn_standard_loso      <- read_and_process(file_paths$inner_res_nn_standard_loso)

# inner_res_nn_standard$params <- gsub("np.int64.+","",inner_res_nn_standard$params)
# inner_res_nn_standard_loso$params <- gsub("np.int64.+","",inner_res_nn_standard_loso$params)

# =============================================================================
# Extract Best Parameters
# =============================================================================

cat("Extracting best parameters for SVM...\n")
# Get best params (SVM)
best_param_svm_OvR           <- get_best_param(inner_res_svm_OvR, type = "OvR")
best_param_svm_OvR_loso      <- get_best_param(inner_res_svm_OvR_loso, type = "OvR")

cat("Extracting best parameters for XGBOOST...\n")
# Get best params (XGBOOST)
best_param_xgb_OvR           <- get_best_param(inner_res_xgb_OvR, type = "OvR")
best_param_xgb_OvR_loso      <- get_best_param(inner_res_xgb_OvR_loso, type = "OvR")

# cat("Extracting best parameters for NN...\n")
# # Get best params (NN)
# best_param_nn_standard           <- get_best_param(inner_res_nn_standard, type = "standard")
# best_param_nn_standard_loso      <- get_best_param(inner_res_nn_standard_loso, type = "standard")

# =============================================================================
# Save Best Parameters
# =============================================================================

cat("Saving SVM results...\n")
# Write to CSV (SVM)
out_dir_svm <- "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/SVM"
dir.create(out_dir_svm, recursive = TRUE, showWarnings = FALSE)

write.csv(best_param_svm_OvR,           file = file.path(out_dir_svm, "SVM_best_param_OvR.csv"), row.names = FALSE)
write.csv(best_param_svm_OvR_loso,      file = file.path(out_dir_svm, "SVM_best_param_OvR_loso.csv"), row.names = FALSE)

cat("Saving XGBOOST results...\n")
# Write to CSV (XGBOOST)
out_dir_xgb <- "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/XGBOOST"
dir.create(out_dir_xgb, recursive = TRUE, showWarnings = FALSE)

write.csv(best_param_xgb_OvR,           file = file.path(out_dir_xgb, "XGBOOST_best_param_OvR.csv"), row.names = FALSE)
write.csv(best_param_xgb_OvR_loso,      file = file.path(out_dir_xgb, "XGBOOST_best_param_OvR_loso.csv"), row.names = FALSE)

# cat("Saving NN results...\n")
# # Write to CSV (NN)
# out_dir_nn <- "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/NN"
# dir.create(out_dir_nn, recursive = TRUE, showWarnings = FALSE)

# write.csv(best_param_nn_standard,           file = file.path(out_dir_nn, "NN_best_param_standard.csv"), row.names = FALSE)
# write.csv(best_param_nn_standard_loso,      file = file.path(out_dir_nn, "NN_best_param_standard_loso.csv"), row.names = FALSE)

cat("Analysis complete! Results saved to:\n")
cat("- SVM: ", out_dir_svm, "\n")
cat("- XGBOOST: ", out_dir_xgb, "\n")
# cat("- NN: ", out_dir_nn, "\n")


# Labels 
# label_mapping <- read.csv("/Users/jsevere2/Documents/AML_PhD/leukem_ai/label_mapping_df_n10.csv")

# evaluate_nested_cv_kappa <- function(results_df, best_params_df) {
#   best_params_df <- add_labels(best_params_df)
#   outer_folds <- unique(results_df$outer_fold)
#   per_inner_fold_kappa <- list()
#   per_inner_fold_class_metrics <- list()
#   idx <- 1
#   idx_class <- 1
  
#   for (outer_fold_id in outer_folds) {
#     outer_fold_data <- results_df[results_df$outer_fold == outer_fold_id, ]
#     inner_folds <- unique(outer_fold_data$inner_fold)
    
#     for (inner_fold_id in inner_folds) {
#       inner_fold_data <- outer_fold_data[outer_fold_data$inner_fold == inner_fold_id, ]
#       class_labels <- unique(inner_fold_data$class_label)
#       n_classes <- length(class_labels)
#       if (nrow(inner_fold_data) == 0 || is.null(inner_fold_data$preds_prob[1]) || is.na(inner_fold_data$preds_prob[1])) next
#       n_samples <- length(clean_probs_str(inner_fold_data$preds_prob[1]))
      
#       prob_matrix <- matrix(NA, nrow = n_samples, ncol = n_classes)
#       colnames(prob_matrix) <- class_labels
#       truth_vector <- rep(NA, n_samples)
      
#       for (j in seq_along(class_labels)) {
#         class_label <- class_labels[j]
#         best_param <- best_params_df[
#           best_params_df$outer_fold == outer_fold_id & best_params_df$class_label == class_label, 
#         ]$params
        
#         best_param_row <- inner_fold_data[
#           inner_fold_data$class_label == class_label & inner_fold_data$params == best_param, 
#         ]
#         if (nrow(best_param_row) == 0) next
        
#         prob_matrix[, j] <- clean_probs_str(best_param_row$preds_prob)
#         y_val <- clean_y_str(best_param_row$y_val)
#         truth_vector[y_val == 1] <- class_label
#       }
      
#       if (all(is.na(truth_vector))) next
      
#       merged_truth <- merge_classes(truth_vector)
#       predicted_indices <- apply(prob_matrix, 1, which.max)
#       merged_preds <- merge_classes(class_labels[predicted_indices])
      
#       all_levels <- unique(c(merged_truth, merged_preds))
#       merged_truth <- factor(merged_truth, levels = all_levels)
#       merged_preds <- factor(merged_preds, levels = all_levels)
      
#       confusion <- caret::confusionMatrix(merged_preds, merged_truth)
#       kappa_value <- as.numeric(confusion$overall["Kappa"])
      
#       # Store overall kappa
#       per_inner_fold_kappa[[idx]] <- data.frame(
#         outer_fold = outer_fold_id, inner_fold = inner_fold_id, kappa = kappa_value
#       )
#       idx <- idx + 1
      
#       # Store per-class metrics
#       by_class <- as.data.frame(confusion$byClass)
#       if (ncol(by_class) == 1) {
#         # binary case: byClass is a vector, convert to 1-row data.frame
#         by_class <- as.data.frame(t(confusion$byClass))
#         rownames(by_class) <- levels(merged_truth)[2]  # positive class
#       }
#       by_class$class_label <- rownames(by_class)
#       by_class$outer_fold <- outer_fold_id
#       by_class$inner_fold <- inner_fold_id
#       per_inner_fold_class_metrics[[idx_class]] <- by_class
#       idx_class <- idx_class + 1
#     }
#   }
  
#   kappa_scores_df <- do.call(rbind, per_inner_fold_kappa)
#   per_outer_fold_kappa <- kappa_scores_df %>%
#     group_by(outer_fold) %>%
#     summarize(mean_kappa = mean(kappa, na.rm = TRUE), .groups = "drop")
  
#   # Aggregate per-class metrics per outer fold
#   class_metrics_df <- do.call(rbind, per_inner_fold_class_metrics)
#   per_outer_fold_class_metrics <- class_metrics_df %>%
#     group_by(outer_fold, class_label) %>%
#     summarize(F1 = mean(F1, na.rm = TRUE), .groups = "drop")
  
#   # Reshape per_outer_fold_class_metrics to wide format: one column per class_label, value = F1
#   per_outer_fold_class_metrics_wide <- per_outer_fold_class_metrics %>%
#     tidyr::pivot_wider(names_from = class_label, values_from = F1, names_prefix = "", names_sep = "")
  
#   # Rename columns to {class_label}_F1
#   f1_colnames <- setdiff(colnames(per_outer_fold_class_metrics_wide), "outer_fold")
#   colnames(per_outer_fold_class_metrics_wide)[colnames(per_outer_fold_class_metrics_wide) != "outer_fold"] <-
#     paste0(f1_colnames, "_F1")
  
#   # Merge with per_outer_fold_kappa
#   merged_df <- dplyr::left_join(per_outer_fold_kappa, per_outer_fold_class_metrics_wide, by = "outer_fold")
  
#   return(merged_df)
# }

# kappa_svm_results <- evaluate_nested_cv_kappa(results_df = inner_res_OvR_loso, best_params_df = best_param_OvR_loso)
# kappa_xgb_results <- evaluate_nested_cv_kappa(results_df = inner_res_xgb_OvR_loso, best_params_df = best_param_xgb_OvR_loso)
# #kappa_nn_results <- best_param_nn_standard_loso

# # Merge by outer_fold
# comparison_df <- merge(
#   kappa_svm_results, 
#   kappa_xgb_results, 
#   by = "outer_fold", 
#   suffixes = c("_svm", "_xgb")
# )

# # Identify class F1 columns (assumes columns end with _F1)
# f1_cols_svm <- grep("_F1_svm$", names(comparison_df), value = TRUE)
# class_names <- gsub("_F1_svm$", "", f1_cols_svm)

# # Calculate per-class F1 differences and add as new columns
# for (class in class_names) {
#   svm_col <- paste0(class, "_F1_svm")
#   xgb_col <- paste0(class, "_F1_xgb")
#   diff_col <- paste0(class, "_F1_diff")
#   comparison_df[[diff_col]] <- comparison_df[[svm_col]] - comparison_df[[xgb_col]]
# }

# # Summarize: mean absolute difference for each class
# diff_summary <- sapply(class_names, function(class) {
#   diff_col <- paste0(class, "_F1_diff")
#   mean(abs(comparison_df[[diff_col]]), na.rm = TRUE)
# })

# # Sort and print the biggest differences
# diff_summary <- sort(diff_summary, decreasing = TRUE)
# print(diff_summary)

# # Optional: visualize with boxplot
# library(reshape2)
# melted <- melt(comparison_df, id.vars = "outer_fold", 
#                measure.vars = paste0(class_names, "_F1_diff"),
#                variable.name = "class", value.name = "F1_diff")
# melted$class <- gsub("_F1_diff", "", melted$class)

# library(ggplot2)
# ggplot(melted, aes(x = class, y = F1_diff)) +
#   geom_boxplot() +
#   labs(title = "Per-class F1 difference (SVM - XGB) across folds",
#        y = "F1 difference (SVM - XGB)", x = "Class") +
#   theme_minimal()
# # results_df_svm  <- inner_res_OvR_loso
# # results_df_xgb  <- inner_res_xgb_OvR_loso
# # results_df_nn   <- inner_res_nn_standard_loso

# # best_params_svm  <- best_param_OvR_loso
# # best_params_xgb  <- best_param_xgb_OvR_loso
# # best_params_nn   <- best_param_nn_standard_loso

# evaluate_nested_cv_ensembl <- function(results_df_svm, results_df_xgb, results_df_nn,
#                                    best_params_svm, best_params_xgb, best_params_nn
#                                    ) {
#   best_params_svm <- add_labels(best_params_svm)
#   best_params_xgb <- add_labels(best_params_xgb)
  
#   outer_folds <- unique(results_df_svm$outer_fold)
#   kappa_scores <- data.frame()
  
#   for (outer_fold in outer_folds) {
#     df_outer_svm <- results_df_svm[results_df_svm$outer_fold == outer_fold, ]
#     df_outer_xgb <- results_df_xgb[results_df_xgb$outer_fold == outer_fold, ]
#     df_outer_nn <- results_df_nn[results_df_nn$outer_fold == outer_fold, ]
    
#     inner_folds <- unique(df_outer_svm$inner_fold)
    
#     for (inner_fold in inner_folds) {
#       probs_svm <- df_outer_svm[df_outer_svm$inner_fold == inner_fold, ]
#       probs_xgb <- df_outer_xgb[df_outer_xgb$inner_fold == inner_fold, ]
      
#       best_param_nn_of_fold <- best_params_nn[best_params_nn$outer_fold == outer_fold,]$params
#       probs_nn_of_fold <- df_outer_nn[df_outer_nn$inner_fold == inner_fold & df_outer_nn$params == best_param_nn_of_fold, ]
      
#       n_samples <- length(clean_y_str(probs_svm$y_val[1]))
#       class_ints <- sort(unique(probs_svm$class))
#       class_labels <- label_mapping$Label[class_ints + 1]

#       # if (nrow(probs_nn_of_fold)>0){
#       #   n_col_nn <- test <- probs_nn_of_fold$preds_prob  %>%
#       #     str_replace_all(",|\\[|\\]|\\{|\\}|\\\n", "") %>% str_squish()
#       #   n_col_nn <- length(as.numeric(unlist(strsplit(test, " "))))/n_samples
#       #   class_ints_nn <- sort(unique(clean_y_str(probs_nn_of_fold$y_val[1])))
#       #   probs_df_nn <- text_to_matrix(probs_nn_of_fold$preds_prob, ncol = n_col_nn)
#       #   probs_df_nn <- data.frame(probs_df_nn)
#       #   colnames(probs_df_nn) <- class_labels
#       # }
#       probs_df <- data.frame(matrix(ncol = length(class_ints), nrow = n_samples))
#       colnames(probs_df) <- class_labels
#       probs_df$truth <- NA
      
#       for (cl in class_labels) {
#         best_param_svm <- best_params_svm[
#           best_params_svm$outer_fold == outer_fold & best_params_svm$class_label == cl, 
#         ]
#         svm_kappa <- best_param_svm$mean_kappa
        
#         best_param_xgb <- best_params_xgb[
#           best_params_xgb$outer_fold == outer_fold & best_params_xgb$class_label == cl, 
#         ]
#         xgb_kappa <- best_param_xgb$mean_kappa
        
#         probs_svm_best <- probs_svm[
#           probs_svm$class_label == cl & probs_svm$params == best_param_svm$params, 
#         ]
        
#         probs_xgb_best <- probs_xgb[
#           probs_xgb$class_label == cl & probs_xgb$params == best_param_xgb$params, 
#         ]
        
#         probs_svm_clean <- clean_probs_str(probs_svm_best$preds_prob)
#         probs_xgb_clean <- clean_probs_str(probs_xgb_best$preds_prob)
#         #probs_nn_clean <- probs_df_nn[[as.character(cl)]]
        
#         probs_df[[as.character(cl)]] <- (probs_svm_clean + probs_xgb_clean)# + probs_nn_clean)
        
#         y_val <- clean_y_str(probs_svm_best$y_val)  # Assume both models used the same y_val
#         probs_df$truth[y_val == 1] <- cl
#       }
      
#       truth <- merge_classes(probs_df$truth)
#       pred_indices <- apply(probs_df[, class_labels], 1, which.max)
#       preds <- merge_classes(class_labels[pred_indices])
      
#       levels_labels <- unique(c(truth, preds))
#       truth <- factor(truth, levels = levels_labels)
#       preds <- factor(preds, levels = levels_labels)
      
#       res <- caret::confusionMatrix(preds, truth)
#       kappa <- as.numeric(res$overall["Kappa"])
      
#       kappa_scores <- rbind(
#         kappa_scores,
#         data.frame(outer_fold = outer_fold, inner_fold = inner_fold, kappa = kappa)
#       )
#     }
#   }
  
#   return(kappa_scores)
# }


# kappas_combined <- evaluate_nested_cv_ensembl(inner_res_OvR_loso, inner_res_xgb_OvR_loso, inner_res_nn_standard_loso,
#                                  best_param_OvR_loso, best_param_xgb_OvR_loso, best_param_nn_standard_loso)
# kappas_combined <- kappas_combined %>% group_by(outer_fold) %>% summarize(mean_kappa = mean(kappa))
# mean(kappas_combined$mean_kappa)

# evaluate_nested_cv_select <-function(results_df_svm, results_df_xgb, best_params_svm, best_params_xgb) {
#   best_params_svm <- add_labels(best_params_svm)
#   best_params_xgb <- add_labels(best_params_xgb)
  
#   outer_folds <- unique(results_df_svm$outer_fold)
#   kappa_scores <- data.frame()
  
#   for (fold in outer_folds) {
#     df_outer_svm <- results_df_svm[results_df_svm$outer_fold == fold, ]
#     df_outer_xgb <- results_df_xgb[results_df_xgb$outer_fold == fold, ]
    
#     inner_folds <- unique(df_outer_svm$inner_fold)
    
#     for (inner_fold in inner_folds) {
#       probs_svm <- df_outer_svm[df_outer_svm$inner_fold == inner_fold, ]
#       probs_xgb <- df_outer_xgb[df_outer_xgb$inner_fold == inner_fold, ]
      
#       class_labels <- unique(probs_svm$class_label)
#       n_classes <- length(class_labels)
#       n_samples <- length(clean_probs_str(probs_svm$preds_prob[1]))
      
#       probs_df <- data.frame(matrix(ncol = n_classes, nrow = n_samples))
#       colnames(probs_df) <- class_labels
#       probs_df$truth <- NA
      
#       for (cl in class_labels) {
#         # Get best params and metric for both models
#         svm_row <- best_params_svm[best_params_svm$outer_fold == fold & best_params_svm$class_label == cl, ]
#         xgb_row <- best_params_xgb[best_params_xgb$outer_fold == fold & best_params_xgb$class_label == cl, ]
        
#         # Choose the better model (based on higher metric_value)
#         if (svm_row$mean_kappa >= xgb_row$mean_kappa) {
#           chosen_model <- "svm"
#           best_params <- svm_row$params
#           probs_model <- probs_svm[probs_svm$class_label == cl & probs_svm$params == best_params, ]
#           probs_clean <- clean_probs_str(probs_model$preds_prob)
#         } else {
#           chosen_model <- "xgb"
#           best_params <- xgb_row$params
#           probs_model <- probs_xgb[probs_xgb$class_label == cl & probs_xgb$params == best_params, ]
#           probs_clean <- clean_probs_str(probs_model$preds_prob)
#         }
        
#         probs_df[[as.character(cl)]] <- probs_clean
#         y_val <- clean_y_str(probs_model$y_val)
#         probs_df$truth[y_val == 1] <- cl
#       }
      
#       truth <- merge_classes(probs_df$truth)
#       pred_indices <- apply(probs_df[, class_labels], 1, which.max)
#       preds <- merge_classes(class_labels[pred_indices])
      
#       levels_labels <- unique(c(truth, preds))
#       truth <- factor(truth, levels = levels_labels)
#       preds <- factor(preds, levels = levels_labels)
      
#       res <- caret::confusionMatrix(preds, truth)
#       kappa <- as.numeric(res$overall["Kappa"])
      
#       kappa_scores <- rbind(
#         kappa_scores,
#         data.frame(outer_fold = fold, inner_fold = inner_fold, kappa = kappa)
#       )
#     }
#   }
  
#   return(kappa_scores)
# }
# kappas <- evaluate3(inner_res_OvR_loso, inner_res_xgb_OvR_loso, best_param_OvR_loso, best_param_xgb_OvR_loso) %>% group_by(outer_fold) %>% summarise(mean_kappa = mean(kappa))

# mean(kappa_svm_results$mean_kappa)
# mean(kappas_combined$mean_kappa)
# mean(kappas$mean_kappa)


# evaluate_nested_cv_kappa_with_rejection <- function(results_df_svm, results_df_xgb, best_params_svm, best_params_xgb) {
#   best_params_svm <- add_labels(best_params_svm)
#   best_params_xgb <- add_labels(best_params_xgb)
  
#   outer_folds <- unique(results_df_svm$outer_fold)
#   all_results <- data.frame()
  
#   # Test probability cutoffs from 0.01 to 1.00 in steps of 0.01
#   prob_cutoffs <- seq(0.00, 1.00, by = 0.01)
  
#   for (fold in outer_folds) {
#     df_outer_svm <- results_df_svm[results_df_svm$outer_fold == fold, ]
#     df_outer_xgb <- results_df_xgb[results_df_xgb$outer_fold == fold, ]
    
#     inner_folds <- unique(df_outer_svm$inner_fold)
    
#     for (inner_fold in inner_folds) {
#       probs_svm <- df_outer_svm[df_outer_svm$inner_fold == inner_fold, ]
#       probs_xgb <- df_outer_xgb[df_outer_xgb$inner_fold == inner_fold, ]
      
#       class_labels <- unique(probs_svm$class_label)
#       n_classes <- length(class_labels)
#       n_samples <- length(clean_probs_str(probs_svm$preds_prob[1]))
      
#       probs_df <- data.frame(matrix(ncol = n_classes, nrow = n_samples))
#       colnames(probs_df) <- class_labels
#       probs_df$truth <- NA
      
#       for (cl in class_labels) {
#         best_param_svm <- best_params_svm[
#           best_params_svm$outer_fold == fold & best_params_svm$class_label == cl, 
#         ]$params
        
#         best_param_xgb <- best_params_xgb[
#           best_params_xgb$outer_fold == fold & best_params_xgb$class_label == cl, 
#         ]$params
        
#         probs_svm_best <- probs_svm[
#           probs_svm$class_label == cl & probs_svm$params == best_param_svm, 
#         ]
        
#         probs_xgb_best <- probs_xgb[
#           probs_xgb$class_label == cl & probs_xgb$params == best_param_xgb, 
#         ]
        
#         probs_svm_clean <- clean_probs_str(probs_svm_best$preds_prob)
#         probs_xgb_clean <- clean_probs_str(probs_xgb_best$preds_prob)
        
#         # Average probabilities
#         probs_df[[as.character(cl)]] <- (probs_svm_clean + probs_xgb_clean)/2
        
#         y_val <- clean_y_str(probs_svm_best$y_val)  # Assume both models used the same y_val
#         probs_df$truth[y_val == 1] <- cl
#       }
      
#       truth <- merge_classes(probs_df$truth)
#       pred_indices <- apply(probs_df[, class_labels], 1, which.max)
#       preds <- merge_classes(class_labels[pred_indices])
      
#       # Get max probabilities for each sample
#       max_probs <- apply(probs_df[, class_labels], 1, max)
      
#       levels_labels <- unique(c(truth, preds))
#       truth <- factor(truth, levels = levels_labels)
#       preds <- factor(preds, levels = levels_labels)
      
#       # Test each probability cutoff
#       for (cutoff in prob_cutoffs) {
#         # Identify samples to reject (max probability below cutoff)
#         rejected_indices <- which(max_probs < cutoff)
#         accepted_indices <- which(max_probs >= cutoff)
        
#         if (length(accepted_indices) == 0) {
#           # If all samples are rejected, skip this cutoff
#           next
#         }
        
#         # Calculate accuracy for rejected samples (if any)
#         rejected_accuracy <- NA
#         if (length(rejected_indices) > 0) {
#           rejected_truth <- truth[rejected_indices]
#           rejected_preds <- preds[rejected_indices]
#           rejected_accuracy <- sum(rejected_truth == rejected_preds) / length(rejected_indices)
#         }
        
#         # Only proceed if rejected samples have accuracy < 50% (or if no samples are rejected)
#         if (is.na(rejected_accuracy) || rejected_accuracy < 0.5) {
#           # Use only accepted samples for kappa calculation
#           accepted_truth <- truth[accepted_indices]
#           accepted_preds <- preds[accepted_indices]
          
#           # Calculate kappa for accepted samples
#           res <- caret::confusionMatrix(accepted_preds, accepted_truth)
#           kappa <- as.numeric(res$overall["Kappa"])
          
#           # Store results
#           all_results <- rbind(
#             all_results,
#             data.frame(
#               outer_fold = fold, 
#               inner_fold = inner_fold, 
#               prob_cutoff = cutoff,
#               kappa = kappa,
#               n_accepted = length(accepted_indices),
#               n_rejected = length(rejected_indices),
#               perc_rejected = length(rejected_indices) / (length(accepted_indices) + length(rejected_indices)),
#               rejected_accuracy = rejected_accuracy,
#               total_samples = n_samples
#             )
#           )
#         }
#       }
#     }
#   }
  
#   return(all_results)
# }

# kappas <- evaluate_nested_cv_kappa_with_rejection(inner_res_OvR_loso, inner_res_xgb_OvR_loso, best_param_OvR_loso, best_param_xgb_OvR_loso)
# kappas_0 <- kappas %>% group_by(outer_fold, inner_fold) %>%filter(prob_cutoff == 0.00) %>% slice(1)
# kappas_filtered <- kappas %>% group_by(outer_fold, inner_fold) %>% filter(kappa == max(kappa)) %>% slice(1)

# mean(kappas_0$kappa)
# mean(kappas_filtered$kappa)
# mean(kappas_filtered$perc_rejected)

# evaluate_nested_cv_per_class_f1_cutoff <- function(results_df_svm, results_df_xgb, best_params_svm, best_params_xgb, cutoffs = seq(0, 1, by = 0.01)) {
#   best_params_svm <- add_labels(best_params_svm)
#   best_params_xgb <- add_labels(best_params_xgb)
  
#   outer_folds <- unique(results_df_svm$outer_fold)
#   class_labels <- unique(results_df_svm$class_label)
  
#   # Store F1 for each class and cutoff
#   f1_results <- data.frame()
  
#   for (cl in class_labels) {
#     for (cutoff in cutoffs) {
#       all_truth <- c()
#       all_pred <- c()
#       all_rejected_truth <- c()
#       all_rejected_pred <- c()
      
#       for (fold in outer_folds) {
#         df_outer_svm <- results_df_svm[results_df_svm$outer_fold == fold, ]
#         df_outer_xgb <- results_df_xgb[results_df_xgb$outer_fold == fold, ]
#         inner_folds <- unique(df_outer_svm$inner_fold)
        
#         for (inner_fold in inner_folds) {
#           probs_svm <- df_outer_svm[df_outer_svm$inner_fold == inner_fold, ]
#           probs_xgb <- df_outer_xgb[df_outer_xgb$inner_fold == inner_fold, ]
#           n_samples <- length(clean_probs_str(probs_svm$preds_prob[1]))
          
#           # Build ensemble probability matrix
#           probs_df <- data.frame(matrix(ncol = length(class_labels), nrow = n_samples))
#           colnames(probs_df) <- class_labels
#           probs_df$truth <- NA
          
#           for (cl2 in class_labels) {
#             best_param_svm <- best_params_svm[
#               best_params_svm$outer_fold == fold & best_params_svm$class_label == cl2, 
#             ]$params
#             best_param_xgb <- best_params_xgb[
#               best_params_xgb$outer_fold == fold & best_params_xgb$class_label == cl2, 
#             ]$params
#             probs_svm_best <- probs_svm[
#               probs_svm$class_label == cl2 & probs_svm$params == best_param_svm, 
#             ]
#             probs_xgb_best <- probs_xgb[
#               probs_xgb$class_label == cl2 & probs_xgb$params == best_param_xgb, 
#             ]
#             if (nrow(probs_svm_best) == 0 || nrow(probs_xgb_best) == 0) {
#               probs_df[[as.character(cl2)]] <- rep(NA, n_samples)
#               next
#             }
#             probs_svm_clean <- clean_probs_str(probs_svm_best$preds_prob)
#             probs_xgb_clean <- clean_probs_str(probs_xgb_best$preds_prob)
#             probs_df[[as.character(cl2)]] <- (probs_svm_clean + probs_xgb_clean)/2
#             y_val <- clean_y_str(probs_svm_best$y_val)
#             probs_df$truth[y_val == 1] <- cl2
#           }
          
#           # Get predicted class and probability for each sample
#           pred_indices <- apply(probs_df[, class_labels], 1, which.max)
#           pred_class <- class_labels[pred_indices]
#           pred_prob <- mapply(function(i, j) probs_df[i, j], 1:n_samples, pred_class)
#           truth_class <- merge_classes(probs_df$truth)
#           pred_class <- merge_classes(pred_class)
          
#           # For this class, mask predictions below cutoff
#           is_class <- pred_class == cl
#           pred_class_cut <- pred_class
#           pred_class_cut[is_class & pred_prob < cutoff] <- NA
          
#           # For F1, treat only samples where truth or pred is this class
#           relevant <- (truth_class == cl) | (pred_class_cut == cl)
#           y_true <- factor(truth_class[relevant] == cl, levels = c(TRUE, FALSE))
#           y_pred <- factor(pred_class_cut[relevant] == cl, levels = c(TRUE, FALSE))
          
#           # For rejected samples (for this class): predicted class was cl, but was masked (set to NA)
#           rejected <- is_class & pred_prob < cutoff
#           if (any(rejected)) {
#             all_rejected_truth <- c(all_rejected_truth, truth_class[rejected] == cl)
#             all_rejected_pred <- c(all_rejected_pred, rep(FALSE, sum(rejected)))
#           }
          
#           # If there are no relevant samples, skip
#           if (length(y_true) == 0 || all(is.na(y_pred))) next
          
#           # Compute F1 for this class at this cutoff
#           cm <- caret::confusionMatrix(y_pred, y_true, mode = "prec_recall")
#           f1 <- as.numeric(cm$byClass["F1"])
          
#           all_truth <- c(all_truth, as.character(y_true))
#           all_pred <- c(all_pred, as.character(y_pred))
#         }
#       }
#       # Compute overall F1 for this class and cutoff across all folds
#       if (length(all_truth) > 0 && any(!is.na(all_pred))) {
#         cm_all <- caret::confusionMatrix(factor(all_pred, levels = c("TRUE", "FALSE")),
#                                          factor(all_truth, levels = c("TRUE", "FALSE")),
#                                          mode = "prec_recall")
#         f1_all <- as.numeric(cm_all$byClass["F1"])
#       } else {
#         f1_all <- NA
#       }
#       # Compute accuracy of rejected samples for this class and cutoff
#       if (length(all_rejected_truth) > 0) {
#         rejected_acc <- mean(all_rejected_truth == all_rejected_pred, na.rm = TRUE)
#       } else {
#         rejected_acc <- NA
#       }
#       # Only keep F1 if rejected accuracy is < 0.5 (or if no rejected samples)
#       if (!is.na(rejected_acc) && rejected_acc >= 0.5) {
#         f1_all <- NA
#       }
#       f1_results <- rbind(f1_results, data.frame(class = cl, cutoff = cutoff, F1 = f1_all, rejected_acc = rejected_acc))
#     }
#   }
#   # For each class, select the cutoff with the highest F1
#   best_cutoffs <- f1_results %>% group_by(class) %>% filter(F1 == max(F1, na.rm = TRUE)) %>% slice(1)
#   return(list(f1_results = f1_results, best_cutoffs = best_cutoffs))
# }

# kappas <- evaluate_nested_cv_per_class_f1_cutoff(inner_res_OvR_loso, inner_res_xgb_OvR_loso, best_param_OvR_loso, best_param_xgb_OvR_loso)
# kappas_0 <- kappas %>% group_by(outer_fold, inner_fold) %>%filter(prob_cutoff == 0.00) %>% slice(1)
# kappas_filtered <- kappas %>% group_by(outer_fold, inner_fold) %>% filter(kappa == max(kappa)) %>% slice(1)