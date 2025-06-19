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
  # SVM
  inner_res_OvR           = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250611_1318/SVM_inner_cv_OvR_20250611_1318.csv",
  inner_res_OvR_loso      = "/Users/jsevere2/Documents/AML_PhD/predictor_out/SVM/20250611_1927/SVM_inner_cv_loso_OvR_20250611_1927.csv",
  inner_res_xgb_OvR       = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250612_0024/XGBOOST_inner_cv_OvR_20250612_0024.csv",
  inner_res_xgb_OvR_loso       = "/Users/jsevere2/Documents/AML_PhD/predictor_out/XGBOOST/20250612_0516/XGBOOST_inner_cv_loso_OvR_20250612_0516.csv",
  inner_res_nn_standard       = "/Users/jsevere2/Documents/AML_PhD/predictor_out/NN/20250612_1000/NN_inner_cv_standard_20250612_1000.csv",
  inner_res_nn_standard_loso       = "/Users/jsevere2/Documents/AML_PhD/predictor_out/NN/20250613_0048/NN_inner_cv_loso_standard_20250613_0048.csv"
)

# =============================================================================
# Data Loading
# =============================================================================

cat("Loading SVM data...\n")
# Read files (SVM)
inner_res_OvR           <- read_and_process(file_paths$inner_res_OvR)
inner_res_OvR_loso      <- read_and_process(file_paths$inner_res_OvR_loso)

cat("Loading XGBOOST data...\n")
# Read files (XGBOOST)
inner_res_xgb_OvR           <- read_and_process(file_paths$inner_res_xgb_OvR)
inner_res_xgb_OvR_loso      <- read_and_process(file_paths$inner_res_xgb_OvR_loso)

cat("Loading NN data...\n")
# Read files (NN)
inner_res_nn_standard           <- read_and_process(file_paths$inner_res_nn_standard)
inner_res_nn_standard_loso      <- read_and_process(file_paths$inner_res_nn_standard_loso)

inner_res_nn_standard$params <- gsub("np.int64.+","",inner_res_nn_standard$params)
inner_res_nn_standard_loso$params <- gsub("np.int64.+","",inner_res_nn_standard_loso$params)

# =============================================================================
# Extract Best Parameters
# =============================================================================

cat("Extracting best parameters for SVM...\n")
# Get best params (SVM)
best_param_OvR           <- get_best_param(inner_res_OvR, type = "OvR")
best_param_OvR_loso      <- get_best_param(inner_res_OvR_loso, type = "OvR")

cat("Extracting best parameters for XGBOOST...\n")
# Get best params (XGBOOST)
best_param_xgb_OvR           <- get_best_param(inner_res_xgb_OvR, type = "OvR")
best_param_xgb_OvR_loso      <- get_best_param(inner_res_xgb_OvR_loso, type = "OvR")

cat("Extracting best parameters for NN...\n")
# Get best params (NN)
best_param_nn_standard           <- get_best_param(inner_res_nn_standard, type = "standard")
best_param_nn_standard_loso      <- get_best_param(inner_res_nn_standard_loso, type = "standard")

# =============================================================================
# Save Best Parameters
# =============================================================================

cat("Saving SVM results...\n")
# Write to CSV (SVM)
out_dir_svm <- "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/SVM"
dir.create(out_dir_svm, recursive = TRUE, showWarnings = FALSE)

write.csv(best_param_OvR,           file = file.path(out_dir_svm, "SVM_best_param_OvR.csv"), row.names = FALSE)
write.csv(best_param_OvR_loso,      file = file.path(out_dir_svm, "SVM_best_param_OvR_loso.csv"), row.names = FALSE)

cat("Saving XGBOOST results...\n")
# Write to CSV (XGBOOST)
out_dir_xgb <- "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/XGBOOST"
dir.create(out_dir_xgb, recursive = TRUE, showWarnings = FALSE)

write.csv(best_param_xgb_OvR,           file = file.path(out_dir_xgb, "XGBOOST_best_param_OvR.csv"), row.names = FALSE)
write.csv(best_param_xgb_OvR_loso,      file = file.path(out_dir_xgb, "XGBOOST_best_param_OvR_loso.csv"), row.names = FALSE)

cat("Saving NN results...\n")
# Write to CSV (NN)
out_dir_nn <- "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params_n10/NN"
dir.create(out_dir_nn, recursive = TRUE, showWarnings = FALSE)

write.csv(best_param_nn_standard,           file = file.path(out_dir_nn, "NN_best_param_standard.csv"), row.names = FALSE)
write.csv(best_param_nn_standard_loso,      file = file.path(out_dir_nn, "NN_best_param_standard_loso.csv"), row.names = FALSE)

cat("Analysis complete! Results saved to:\n")
cat("- SVM: ", out_dir_svm, "\n")
cat("- XGBOOST: ", out_dir_xgb, "\n")
cat("- NN: ", out_dir_nn, "\n")


# Labels 
label_mapping <- read.csv("/Users/jsevere2/Documents/AML_PhD/leukem_ai/label_mapping_df_n10.csv")

evaluate_nested_cv_kappa <- function(results_df, best_params_df) {
  best_params_df <- add_labels(best_params_df)
  outer_folds <- unique(results_df$outer_fold)
  per_inner_fold_kappa <- list()
  idx <- 1
  
  for (outer_fold_id in outer_folds) {
    outer_fold_data <- results_df[results_df$outer_fold == outer_fold_id, ]
    inner_folds <- unique(outer_fold_data$inner_fold)
    
    for (inner_fold_id in inner_folds) {
      inner_fold_data <- outer_fold_data[outer_fold_data$inner_fold == inner_fold_id, ]
      class_labels <- unique(inner_fold_data$class_label)
      n_classes <- length(class_labels)
      if (nrow(inner_fold_data) == 0 || is.null(inner_fold_data$preds_prob[1]) || is.na(inner_fold_data$preds_prob[1])) next
      n_samples <- length(clean_probs_str(inner_fold_data$preds_prob[1]))
      
      prob_matrix <- matrix(NA, nrow = n_samples, ncol = n_classes)
      colnames(prob_matrix) <- class_labels
      truth_vector <- rep(NA, n_samples)
      
      for (j in seq_along(class_labels)) {
        class_label <- class_labels[j]
        best_param <- best_params_df[
          best_params_df$outer_fold == outer_fold_id & best_params_df$class_label == class_label, 
        ]$params
        
        best_param_row <- inner_fold_data[
          inner_fold_data$class_label == class_label & inner_fold_data$params == best_param, 
        ]
        if (nrow(best_param_row) == 0) next
        
        prob_matrix[, j] <- clean_probs_str(best_param_row$preds_prob)
        y_val <- clean_y_str(best_param_row$y_val)
        truth_vector[y_val == 1] <- class_label
      }
      
      if (all(is.na(truth_vector))) next
      
      merged_truth <- merge_classes(truth_vector)
      predicted_indices <- apply(prob_matrix, 1, which.max)
      merged_preds <- merge_classes(class_labels[predicted_indices])
      
      all_levels <- unique(c(merged_truth, merged_preds))
      merged_truth <- factor(merged_truth, levels = all_levels)
      merged_preds <- factor(merged_preds, levels = all_levels)
      
      confusion <- caret::confusionMatrix(merged_preds, merged_truth)
      kappa_value <- as.numeric(confusion$overall["Kappa"])
      
      per_inner_fold_kappa[[idx]] <- data.frame(
        outer_fold = outer_fold_id, inner_fold = inner_fold_id, kappa = kappa_value
      )
      idx <- idx + 1
    }
  }
  
  kappa_scores_df <- do.call(rbind, per_inner_fold_kappa)
  per_outer_fold_kappa <- kappa_scores_df %>%
    group_by(outer_fold) %>%
    summarize(mean_kappa = mean(kappa, na.rm = TRUE), .groups = "drop")
  
  return(list(
    per_outer_fold = per_outer_fold_kappa,
    per_inner_fold = kappa_scores_df
  ))
}

kappa_svm_results <- evaluate_nested_cv_kappa(results_df = inner_res_OvR_loso, best_params_df = best_param_OvR_loso)
kappa_xgb_results <- evaluate_nested_cv_kappa(results_df = inner_res_xgb_OvR_loso, best_params_df = best_param_xgb_OvR_loso)
kappa_nn_results <- best_param_nn_standard_loso

# results_df_svm  <- inner_res_OvR_loso
# results_df_xgb  <- inner_res_xgb_OvR_loso
# results_df_nn   <- inner_res_nn_standard_loso

# best_params_svm  <- best_param_OvR_loso
# best_params_xgb  <- best_param_xgb_OvR_loso
# best_params_nn   <- best_param_nn_standard_loso

evaluate_nested_cv_ensembl <- function(results_df_svm, results_df_xgb, results_df_nn,
                                   best_params_svm, best_params_xgb, best_params_nn
                                   ) {
  best_params_svm <- add_labels(best_params_svm)
  best_params_xgb <- add_labels(best_params_xgb)
  
  outer_folds <- unique(results_df_svm$outer_fold)
  kappa_scores <- data.frame()
  
  for (outer_fold in outer_folds) {
    df_outer_svm <- results_df_svm[results_df_svm$outer_fold == outer_fold, ]
    df_outer_xgb <- results_df_xgb[results_df_xgb$outer_fold == outer_fold, ]
    df_outer_nn <- results_df_nn[results_df_nn$outer_fold == outer_fold, ]
    
    inner_folds <- unique(df_outer_svm$inner_fold)
    
    for (inner_fold in inner_folds) {
      probs_svm <- df_outer_svm[df_outer_svm$inner_fold == inner_fold, ]
      probs_xgb <- df_outer_xgb[df_outer_xgb$inner_fold == inner_fold, ]
      
      best_param_nn_of_fold <- best_params_nn[best_params_nn$outer_fold == outer_fold,]$params
      probs_nn_of_fold <- df_outer_nn[df_outer_nn$inner_fold == inner_fold & df_outer_nn$params == best_param_nn_of_fold, ]
      
      n_samples <- length(clean_y_str(probs_svm$y_val[1]))
      class_ints <- sort(unique(probs_svm$class))
      class_labels <- label_mapping$Label[class_ints + 1]

      # if (nrow(probs_nn_of_fold)>0){
      #   n_col_nn <- test <- probs_nn_of_fold$preds_prob  %>%
      #     str_replace_all(",|\\[|\\]|\\{|\\}|\\\n", "") %>% str_squish()
      #   n_col_nn <- length(as.numeric(unlist(strsplit(test, " "))))/n_samples
      #   class_ints_nn <- sort(unique(clean_y_str(probs_nn_of_fold$y_val[1])))
      #   probs_df_nn <- text_to_matrix(probs_nn_of_fold$preds_prob, ncol = n_col_nn)
      #   probs_df_nn <- data.frame(probs_df_nn)
      #   colnames(probs_df_nn) <- class_labels
      # }
      probs_df <- data.frame(matrix(ncol = length(class_ints), nrow = n_samples))
      colnames(probs_df) <- class_labels
      probs_df$truth <- NA
      
      for (cl in class_labels) {
        best_param_svm <- best_params_svm[
          best_params_svm$outer_fold == outer_fold & best_params_svm$class_label == cl, 
        ]
        svm_kappa <- best_param_svm$mean_kappa
        
        best_param_xgb <- best_params_xgb[
          best_params_xgb$outer_fold == outer_fold & best_params_xgb$class_label == cl, 
        ]
        xgb_kappa <- best_param_xgb$mean_kappa
        
        probs_svm_best <- probs_svm[
          probs_svm$class_label == cl & probs_svm$params == best_param_svm$params, 
        ]
        
        probs_xgb_best <- probs_xgb[
          probs_xgb$class_label == cl & probs_xgb$params == best_param_xgb$params, 
        ]
        
        probs_svm_clean <- clean_probs_str(probs_svm_best$preds_prob)
        probs_xgb_clean <- clean_probs_str(probs_xgb_best$preds_prob)
        #probs_nn_clean <- probs_df_nn[[as.character(cl)]]
        
        probs_df[[as.character(cl)]] <- (probs_svm_clean + probs_xgb_clean)# + probs_nn_clean)
        
        y_val <- clean_y_str(probs_svm_best$y_val)  # Assume both models used the same y_val
        probs_df$truth[y_val == 1] <- cl
      }
      
      truth <- merge_classes(probs_df$truth)
      pred_indices <- apply(probs_df[, class_labels], 1, which.max)
      preds <- merge_classes(class_labels[pred_indices])
      
      levels_labels <- unique(c(truth, preds))
      truth <- factor(truth, levels = levels_labels)
      preds <- factor(preds, levels = levels_labels)
      
      res <- caret::confusionMatrix(preds, truth)
      kappa <- as.numeric(res$overall["Kappa"])
      
      kappa_scores <- rbind(
        kappa_scores,
        data.frame(outer_fold = outer_fold, inner_fold = inner_fold, kappa = kappa)
      )
    }
  }
  
  return(kappa_scores)
}


kappas_combined <- evaluate_nested_cv_ensembl(inner_res_OvR_loso, inner_res_xgb_OvR_loso, inner_res_nn_standard_loso,
                                 best_param_OvR_loso, best_param_xgb_OvR_loso, best_param_nn_standard_loso)
kappas_combined <- kappas_combined %>% group_by(outer_fold) %>% summarize(mean_kappa = mean(kappa))
mean(kappas_combined$mean_kappa)

evaluate_nested_cv_select <- function(results_df_svm, results_df_xgb, results_df_nn,
                                   best_params_svm, best_params_xgb, best_params_nn
) {
  best_params_svm <- add_labels(best_params_svm)
  best_params_xgb <- add_labels(best_params_xgb)
  
  outer_folds <- unique(results_df_svm$outer_fold)
  kappa_scores <- data.frame()
  
  for (outer_fold in outer_folds) {
    df_outer_svm <- results_df_svm[results_df_svm$outer_fold == outer_fold, ]
    df_outer_xgb <- results_df_xgb[results_df_xgb$outer_fold == outer_fold, ]
    df_outer_nn <- results_df_nn[results_df_nn$outer_fold == outer_fold, ]
    
    inner_folds <- unique(df_outer_svm$inner_fold)
    
    for (inner_fold in inner_folds) {
      probs_svm <- df_outer_svm[df_outer_svm$inner_fold == inner_fold, ]
      probs_xgb <- df_outer_xgb[df_outer_xgb$inner_fold == inner_fold, ]
      
      #best_param_nn_of_fold <- best_params_nn[best_params_nn$outer_fold == outer_fold,]$params
      #probs_nn_of_fold <- df_outer_nn[df_outer_nn$inner_fold == inner_fold & df_outer_nn$params == best_param_nn_of_fold, ]
      
      #nn_y_val <- clean_y_str(probs_nn_of_fold$y_val)
      #nn_preds <- clean_y_str(probs_nn_of_fold$preds)
      
      class_ints <- label_mapping$Encoded
      class_labels <- label_mapping$Label
      n_samples <- length(clean_probs_str(probs_svm$preds_prob[1]))
      
      #probs_df_nn <- text_to_matrix(probs_nn_of_fold$preds_prob, ncol = 23)
      #probs_df_nn <- data.frame(probs_df_nn)
      #colnames(probs_df_nn) <- label_mapping$Label
      
      probs_df <- data.frame(matrix(ncol = length(label_mapping$Label), nrow = n_samples))
      colnames(probs_df) <- class_labels
      probs_df$truth <- NA
      
      for (cl in class_ints) {
        # SVM
        best_param_fold_svm <- best_params_svm[
          best_params_svm$outer_fold == outer_fold & best_params_svm$class == cl, 
        ]
        svm_kappa <- best_param_fold_svm$mean_kappa
        
        # XGB
        best_param_fold_xgb <- best_params_xgb[
          best_params_xgb$outer_fold == outer_fold & best_params_xgb$class == cl, 
        ]
        xgb_kappa <- best_param_fold_xgb$mean_kappa
        
        # NN
        #nn_y_val_bi <- factor(ifelse(nn_y_val == cl, 1, 0), c(1,0))
        #nn_preds_bi <- factor(ifelse(nn_preds == cl, 1, 0), c(1,0))
        
        #res <- caret::confusionMatrix(nn_preds_bi, nn_y_val_bi)
        #nn_kappa <- res$overall["Kappa"]
        all_kappas <- c(svm_kappa, xgb_kappa) #, nn_kappa)
        all_kappas[is.na(all_kappas)] <- 0
        best_model <- c("SVM", "XGB", "NN")[which.max(all_kappas)]
        
        if (best_model == "SVM"){
          probs_best <- probs_svm[
            probs_svm$class == cl & probs_svm$params == best_param_fold_svm$params, 
          ]$preds_prob
          probs_best <- clean_probs_str(probs_best)
        }
        
        if (best_model == "XGB"){
          probs_best <- probs_xgb[
            probs_xgb$class == cl & probs_xgb$params == best_param_fold_xgb$params, 
          ]$preds_prob
          probs_best <- clean_probs_str(probs_best)
        }
        
        #if (best_model == "NN"){
        #  probs_best <- probs_df_nn[[as.character(cl)]]
        #}
        print(class_labels[cl+1])
        probs_df[[class_labels[cl+1]]] <- probs_best
        # probs_df$truth[nn_y_val_bi == 1] <- class_labels[cl+1]
      }
      
      truth <- merge_classes(probs_df$truth)
      pred_indices <- apply(probs_df[, class_labels], 1, which.max)
      preds <- merge_classes(class_labels[pred_indices])
      
      levels_labels <- unique(c(truth, preds))
      truth <- factor(truth, levels = levels_labels)
      preds <- factor(preds, levels = levels_labels)
      
      res <- caret::confusionMatrix(preds, truth)
      kappa <- as.numeric(res$overall["Kappa"])
      
      kappa_scores <- rbind(
        kappa_scores,
        data.frame(outer_fold = outer_fold, inner_fold = inner_fold, kappa = kappa)
      )
    }
  }
  
  return(kappa_scores)
}


kappas_combined <- evaluate_nested_cv_select(inner_res_OvR, inner_res_xgb_OvR, inner_res_nn_standard,
                                          best_param_OvR, best_param_xgb_OvR, best_param_nn_standard)
kappas_combined %>% group_by(outer_fold) %>% summarize(mean_kappa = mean(kappa))




evaluate_nested_cv_kappa_with_rejection <- function(results_df_svm, results_df_xgb, best_params_svm, best_params_xgb) {
  best_params_svm <- add_labels(best_params_svm)
  best_params_xgb <- add_labels(best_params_xgb)
  
  outer_folds <- unique(results_df_svm$outer_fold)
  all_results <- data.frame()
  
  # Test probability cutoffs from 0.01 to 1.00 in steps of 0.01
  prob_cutoffs <- seq(0.00, 1.00, by = 0.01)
  
  for (fold in outer_folds) {
    df_outer_svm <- results_df_svm[results_df_svm$outer_fold == fold, ]
    df_outer_xgb <- results_df_xgb[results_df_xgb$outer_fold == fold, ]
    
    inner_folds <- unique(df_outer_svm$inner_fold)
    
    for (inner_fold in inner_folds) {
      probs_svm <- df_outer_svm[df_outer_svm$inner_fold == inner_fold, ]
      probs_xgb <- df_outer_xgb[df_outer_xgb$inner_fold == inner_fold, ]
      
      class_labels <- unique(probs_svm$class_label)
      n_classes <- length(class_labels)
      n_samples <- length(clean_probs_str(probs_svm$preds_prob[1]))
      
      probs_df <- data.frame(matrix(ncol = n_classes, nrow = n_samples))
      colnames(probs_df) <- class_labels
      probs_df$truth <- NA
      
      for (cl in class_labels) {
        best_param_svm <- best_params_svm[
          best_params_svm$outer_fold == fold & best_params_svm$class_label == cl, 
        ]$params
        
        best_param_xgb <- best_params_xgb[
          best_params_xgb$outer_fold == fold & best_params_xgb$class_label == cl, 
        ]$params
        
        probs_svm_best <- probs_svm[
          probs_svm$class_label == cl & probs_svm$params == best_param_svm, 
        ]
        
        probs_xgb_best <- probs_xgb[
          probs_xgb$class_label == cl & probs_xgb$params == best_param_xgb, 
        ]
        
        probs_svm_clean <- clean_probs_str(probs_svm_best$preds_prob)
        probs_xgb_clean <- clean_probs_str(probs_xgb_best$preds_prob)
        
        # Average probabilities
        probs_df[[as.character(cl)]] <- (probs_svm_clean + probs_xgb_clean)/2
        
        y_val <- clean_y_str(probs_svm_best$y_val)  # Assume both models used the same y_val
        probs_df$truth[y_val == 1] <- cl
      }
      
      truth <- merge_classes(probs_df$truth)
      pred_indices <- apply(probs_df[, class_labels], 1, which.max)
      preds <- merge_classes(class_labels[pred_indices])
      
      # Get max probabilities for each sample
      max_probs <- apply(probs_df[, class_labels], 1, max)
      
      levels_labels <- unique(c(truth, preds))
      truth <- factor(truth, levels = levels_labels)
      preds <- factor(preds, levels = levels_labels)
      
      # Test each probability cutoff
      for (cutoff in prob_cutoffs) {
        # Identify samples to reject (max probability below cutoff)
        rejected_indices <- which(max_probs < cutoff)
        accepted_indices <- which(max_probs >= cutoff)
        
        if (length(accepted_indices) == 0) {
          # If all samples are rejected, skip this cutoff
          next
        }
        
        # Calculate accuracy for rejected samples (if any)
        rejected_accuracy <- NA
        if (length(rejected_indices) > 0) {
          rejected_truth <- truth[rejected_indices]
          rejected_preds <- preds[rejected_indices]
          rejected_accuracy <- sum(rejected_truth == rejected_preds) / length(rejected_indices)
        }
        
        # Only proceed if rejected samples have accuracy < 50% (or if no samples are rejected)
        if (is.na(rejected_accuracy) || rejected_accuracy < 0.5) {
          # Use only accepted samples for kappa calculation
          accepted_truth <- truth[accepted_indices]
          accepted_preds <- preds[accepted_indices]
          
          # Calculate kappa for accepted samples
          res <- caret::confusionMatrix(accepted_preds, accepted_truth)
          kappa <- as.numeric(res$overall["Kappa"])
          
          # Store results
          all_results <- rbind(
            all_results,
            data.frame(
              outer_fold = fold, 
              inner_fold = inner_fold, 
              prob_cutoff = cutoff,
              kappa = kappa,
              n_accepted = length(accepted_indices),
              n_rejected = length(rejected_indices),
              perc_rejected = length(rejected_indices) / (length(accepted_indices) + length(rejected_indices)),
              rejected_accuracy = rejected_accuracy,
              total_samples = n_samples
            )
          )
        }
      }
    }
  }
  
  return(all_results)
}

kappas <- evaluate_nested_cv_kappa_with_rejection(inner_res_OvR, inner_res_xgb_OvR, best_param_OvR, best_param_xgb_OvR)
kappas_0 <- kappas %>% group_by(outer_fold, inner_fold) %>%filter(prob_cutoff == 0.00) %>% slice(1)
kappas_filtered <- kappas %>% group_by(outer_fold, inner_fold) %>% filter(kappa == max(kappa)) %>% slice(1)

mean(kappas_0$kappa)
mean(kappas_filtered$kappa)



kappas <- evaluate_nested_cv_kappa2(inner_res_OvR, inner_res_xgb_OvR, best_param_OvR, best_param_xgb_OvR)
mean(kappas$kappa)

evaluate3 <-function(results_df_svm, results_df_xgb, best_params_svm, best_params_xgb) {
  best_params_svm <- add_labels(best_params_svm)
  best_params_xgb <- add_labels(best_params_xgb)
  
  outer_folds <- unique(results_df_svm$outer_fold)
  kappa_scores <- data.frame()
  
  for (fold in outer_folds) {
    df_outer_svm <- results_df_svm[results_df_svm$outer_fold == fold, ]
    df_outer_xgb <- results_df_xgb[results_df_xgb$outer_fold == fold, ]
    
    inner_folds <- unique(df_outer_svm$inner_fold)
    
    for (inner_fold in inner_folds) {
      probs_svm <- df_outer_svm[df_outer_svm$inner_fold == inner_fold, ]
      probs_xgb <- df_outer_xgb[df_outer_xgb$inner_fold == inner_fold, ]
      
      class_labels <- unique(probs_svm$class_label)
      n_classes <- length(class_labels)
      n_samples <- length(clean_probs_str(probs_svm$preds_prob[1]))
      
      probs_df <- data.frame(matrix(ncol = n_classes, nrow = n_samples))
      colnames(probs_df) <- class_labels
      probs_df$truth <- NA
      
      for (cl in class_labels) {
        # Get best params and metric for both models
        svm_row <- best_params_svm[best_params_svm$outer_fold == fold & best_params_svm$class_label == cl, ]
        xgb_row <- best_params_xgb[best_params_xgb$outer_fold == fold & best_params_xgb$class_label == cl, ]
        
        # Choose the better model (based on higher metric_value)
        if (svm_row$mean_kappa >= xgb_row$mean_kappa) {
          chosen_model <- "svm"
          best_params <- svm_row$params
          probs_model <- probs_svm[probs_svm$class_label == cl & probs_svm$params == best_params, ]
          probs_clean <- clean_probs_str(probs_model$preds_prob)
        } else {
          chosen_model <- "xgb"
          best_params <- xgb_row$params
          probs_model <- probs_xgb[probs_xgb$class_label == cl & probs_xgb$params == best_params, ]
          probs_clean <- clean_probs_str(probs_model$preds_prob)
        }
        
        probs_df[[as.character(cl)]] <- probs_clean
        y_val <- clean_y_str(probs_model$y_val)
        probs_df$truth[y_val == 1] <- cl
      }
      
      truth <- merge_classes(probs_df$truth)
      pred_indices <- apply(probs_df[, class_labels], 1, which.max)
      preds <- merge_classes(class_labels[pred_indices])
      
      levels_labels <- unique(c(truth, preds))
      truth <- factor(truth, levels = levels_labels)
      preds <- factor(preds, levels = levels_labels)
      
      res <- caret::confusionMatrix(preds, truth)
      kappa <- as.numeric(res$overall["Kappa"])
      
      kappa_scores <- rbind(
        kappa_scores,
        data.frame(outer_fold = fold, inner_fold = inner_fold, kappa = kappa)
      )
    }
  }
  
  return(kappa_scores)
}
kappas <- evaluate3(inner_res_OvR, inner_res_xgb_OvR, best_param_OvR, best_param_xgb_OvR)
mean(kappas$kappa)

# Helper function to analyze rejection results
analyze_rejection_results <- function(rejection_results) {
  # Aggregate results across all folds for each cutoff
  summary_results <- rejection_results %>%
    group_by(prob_cutoff) %>%
    summarise(
      mean_kappa = mean(kappa, na.rm = TRUE),
      sd_kappa = sd(kappa, na.rm = TRUE),
      mean_n_accepted = mean(n_accepted),
      mean_n_rejected = mean(n_rejected),
      mean_rejected_accuracy = mean(rejected_accuracy, na.rm = TRUE),
      rejection_rate = mean(n_rejected) / mean(total_samples),
      n_folds = n(),
      .groups = "drop"
    )
  
  # Find optimal cutoff (highest mean kappa)
  optimal_cutoff <- summary_results$prob_cutoff[which.max(summary_results$mean_kappa)]
  
  # Get results for optimal cutoff
  optimal_results <- summary_results[summary_results$prob_cutoff == optimal_cutoff, ]
  
  cat("Optimal probability cutoff:", optimal_cutoff, "\n")
  cat("Mean kappa at optimal cutoff:", round(optimal_results$mean_kappa, 4), "\n")
  cat("Rejection rate at optimal cutoff:", round(optimal_results$rejection_rate * 100, 2), "%\n")
  cat("Mean rejected accuracy at optimal cutoff:", round(optimal_results$mean_rejected_accuracy, 4), "\n")
  
  return(list(
    summary = summary_results,
    optimal_cutoff = optimal_cutoff,
    optimal_results = optimal_results
  ))
}

# Example usage:
# rejection_results <- evaluate_nested_cv_kappa_with_rejection(
#   inner_res_OvR_loso, inner_res_xgb_OvR_loso, 
#   best_param_OvR_loso, best_param_xgb_OvR_loso
# )
# 
# analysis <- analyze_rejection_results(rejection_results)
# 
# # Plot results
# library(ggplot2)
# ggplot(analysis$summary, aes(x = prob_cutoff, y = mean_kappa)) +
#   geom_line() +
#   geom_point() +
#   geom_vline(xintercept = analysis$optimal_cutoff, color = "red", linetype = "dashed") +
#   labs(title = "Kappa vs Probability Cutoff", 
#        x = "Probability Cutoff", 
#        y = "Mean Kappa") +
#   theme_minimal()

# =============================================================================
# Save Results
# =============================================================================


# Alternative approach: evaluate different cutoffs for each class independently
evaluate_nested_cv_kappa_with_class_specific_cutoffs <- function(results_df_svm, results_df_xgb, best_params_svm, best_params_xgb) {
  best_params_svm <- add_labels(best_params_svm)
  best_params_xgb <- add_labels(best_params_xgb)
  
  outer_folds <- unique(results_df_svm$outer_fold)
  all_results <- data.frame()
  
  # Test probability cutoffs from 0.01 to 1.00 in steps of 0.01
  prob_cutoffs <- seq(0.00, 1.00, by = 0.1)
  
  for (fold in outer_folds) {
    print(fold)
    df_outer_svm <- results_df_svm[results_df_svm$outer_fold == fold, ]
    df_outer_xgb <- results_df_xgb[results_df_xgb$outer_fold == fold, ]
    
    inner_folds <- unique(df_outer_svm$inner_fold)
    
    for (inner_fold in inner_folds) {
      print(inner_fold)
      probs_svm <- df_outer_svm[df_outer_svm$inner_fold == inner_fold, ]
      probs_xgb <- df_outer_xgb[df_outer_xgb$inner_fold == inner_fold, ]
      
      class_labels <- unique(probs_svm$class_label)
      n_classes <- length(class_labels)
      n_samples <- length(clean_probs_str(probs_svm$preds_prob[1]))
      
      probs_df <- data.frame(matrix(ncol = n_classes, nrow = n_samples))
      colnames(probs_df) <- class_labels
      probs_df$truth <- NA
      
      for (cl in class_labels) {
        best_param_svm <- best_params_svm[
          best_params_svm$outer_fold == fold & best_params_svm$class_label == cl, 
        ]$params
        
        best_param_xgb <- best_params_xgb[
          best_params_xgb$outer_fold == fold & best_params_xgb$class_label == cl, 
        ]$params
        
        probs_svm_best <- probs_svm[
          probs_svm$class_label == cl & probs_svm$params == best_param_svm, 
        ]
        
        probs_xgb_best <- probs_xgb[
          probs_xgb$class_label == cl & probs_xgb$params == best_param_xgb, 
        ]
        
        probs_svm_clean <- clean_probs_str(probs_svm_best$preds_prob)
        probs_xgb_clean <- clean_probs_str(probs_xgb_best$preds_prob)
        
        # Average probabilities
        probs_df[[as.character(cl)]] <- (probs_svm_clean + probs_xgb_clean)/2
        
        y_val <- clean_y_str(probs_svm_best$y_val)  # Assume both models used the same y_val
        probs_df$truth[y_val == 1] <- cl
      }
      
      truth <- (probs_df$truth)
      truth_original <- probs_df$truth
      pred_indices <- apply(probs_df[, class_labels], 1, which.max)
      preds_original <- class_labels[pred_indices]
      
      levels_labels <- unique(c(truth, preds))
      truth <- factor(truth, levels = levels_labels)
      preds <- factor(preds, levels = levels_labels)
      
      # Evaluate each class independently with different cutoffs
      for (cl in class_labels) {
        for (cutoff in prob_cutoffs) {
          # Reject samples where probability for this specific class is below cutoff
          class_probs <- probs_df[[as.character(cl)]]
          rejected_indices <- (class_probs < cutoff) & (preds_original == cl)
          accepted_indices = !rejected_indices
          if (length(accepted_indices) == 0) {
            # If all samples are rejected, skip this cutoff
            next
          }
          
          # Calculate accuracy for rejected samples (if any)
          rejected_accuracy <- NA
          if (length(rejected_indices) > 0) {
            rejected_truth <- truth[rejected_indices]
            rejected_preds <- preds[rejected_indices]
            rejected_accuracy <- sum(rejected_truth == rejected_preds, na.rm = TRUE) / sum(rejected_indices,na.rm = TRUE)
          }
          
          # Only proceed if rejected samples have accuracy < 50% (or if no samples are rejected)
          if (is.na(rejected_accuracy) || rejected_accuracy < 0.5) {
            # Use only accepted samples for kappa calculation
            accepted_truth <- truth_original[accepted_indices]
            accepted_preds <- preds_original[accepted_indices]
            accepted_truth <- factor(accepted_truth, levels=levels_labels)
            accepted_preds <- factor(accepted_preds, levels=levels_labels)
            # Calculate kappa for accepted samples (one-vs-rest)
            res <- caret::confusionMatrix(accepted_preds, accepted_truth)
            kappa <- as.numeric(res$overall["Kappa"])
            
            # Store results
            all_results <- rbind(
              all_results,
              data.frame(
                outer_fold = fold, 
                inner_fold = inner_fold, 
                class = cl,
                prob_cutoff = cutoff,
                kappa = kappa,
                n_accepted = sum(accepted_indices,na.rm = TRUE),
                n_rejected = sum(rejected_indices,na.rm = TRUE),
                rejected_accuracy = rejected_accuracy,
                total_samples = n_samples,
                rejection_type = "class_specific_ovr"
              )
            )
          }
        }
      }
    }
  }
  
  return(all_results)
}

# Helper function to analyze class-specific rejection results
analyze_class_specific_rejection_results <- function(rejection_results) {
  # For the per-class approach (same cutoff for all classes)
  if ("prob_cutoff" %in% colnames(rejection_results) && !("class" %in% colnames(rejection_results))) {
    summary_results <- rejection_results %>%
      group_by(prob_cutoff) %>%
      summarise(
        mean_kappa = mean(kappa, na.rm = TRUE),
        sd_kappa = sd(kappa, na.rm = TRUE),
        mean_n_accepted = mean(n_accepted),
        mean_n_rejected = mean(n_rejected),
        mean_rejected_accuracy = mean(rejected_accuracy, na.rm = TRUE),
        rejection_rate = mean(n_rejected) / mean(total_samples),
        n_folds = n(),
        .groups = "drop"
      )
    
    optimal_cutoff <- summary_results$prob_cutoff[which.max(summary_results$mean_kappa)]
    optimal_results <- summary_results[summary_results$prob_cutoff == optimal_cutoff, ]
    
    cat("Optimal probability cutoff (per class):", optimal_cutoff, "\n")
    cat("Mean kappa at optimal cutoff:", round(optimal_results$mean_kappa, 4), "\n")
    cat("Rejection rate at optimal cutoff:", round(optimal_results$rejection_rate * 100, 2), "%\n")
    cat("Mean rejected accuracy at optimal cutoff:", round(optimal_results$mean_rejected_accuracy, 4), "\n")
    
    return(list(
      summary = summary_results,
      optimal_cutoff = optimal_cutoff,
      optimal_results = optimal_results
    ))
  } else if ("class" %in% colnames(rejection_results)) {
    # For class-specific cutoffs approach (one-vs-rest per class)
    # Find optimal cutoff for each class
    class_summaries <- rejection_results %>%
      group_by(class, prob_cutoff) %>%
      summarise(
        mean_kappa = mean(kappa, na.rm = TRUE),
        sd_kappa = sd(kappa, na.rm = TRUE),
        mean_n_accepted = mean(n_accepted),
        mean_n_rejected = mean(n_rejected),
        mean_rejected_accuracy = mean(rejected_accuracy, na.rm = TRUE),
        rejection_rate = mean(n_rejected) / mean(total_samples),
        n_folds = n(),
        .groups = "drop"
      )
    
    # Find optimal cutoff for each class
    optimal_cutoffs <- class_summaries %>%
      group_by(class) %>%
      summarise(
        optimal_cutoff = prob_cutoff[which.max(mean_kappa)],
        max_kappa = max(mean_kappa),
        .groups = "drop"
      )
    
    cat("Optimal probability cutoffs per class (one-vs-rest):\n")
    for (i in 1:nrow(optimal_cutoffs)) {
      cat("  ", optimal_cutoffs$class[i], ": ", optimal_cutoffs$optimal_cutoff[i], 
          " (kappa: ", round(optimal_cutoffs$max_kappa[i], 4), ")\n", sep = "")
    }
    
    # Calculate overall performance with optimal cutoffs
    overall_performance <- rejection_results %>%
      left_join(optimal_cutoffs, by = "class") %>%
      filter(prob_cutoff == optimal_cutoff) %>%
      summarise(
        mean_kappa = mean(kappa, na.rm = TRUE),
        sd_kappa = sd(kappa, na.rm = TRUE),
        mean_accuracy_rejects = mean(rejected_accuracy, na.rm = TRUE),
        mean_rejection_rate = mean(n_rejected) / mean(total_samples)
      )
    
    cat("Overall mean kappa with optimal class-specific cutoffs:", 
        round(overall_performance$mean_kappa, 4), "\n")
    cat("Overall mean rejection rate:", 
        round(overall_performance$mean_rejection_rate * 100, 2), "%\n")
    
    return(list(
      class_summaries = class_summaries,
      optimal_cutoffs = optimal_cutoffs,
      overall_performance = overall_performance,
      all_results = rejection_results
    ))
  } else {
    # For the old class-specific cutoffs approach
    # Find the best combination of cutoffs
    best_idx <- which.max(rejection_results$kappa)
    best_result <- rejection_results[best_idx, ]
    
    cat("Best class-specific cutoffs:\n")
    cutoff_cols <- grep("^cutoff_", colnames(best_result), value = TRUE)
    for (col in cutoff_cols) {
      class_name <- gsub("cutoff_", "", col)
      cat("  ", class_name, ":", best_result[[col]], "\n")
    }
    cat("Mean kappa with best cutoffs:", round(mean(rejection_results$kappa), 4), "\n")
    cat("Best individual kappa:", round(best_result$kappa, 4), "\n")
    
    return(list(
      best_result = best_result,
      all_results = rejection_results
    ))
  }
}

kappas <- evaluate_nested_cv_kappa_with_class_specific_cutoffs(inner_res_OvR, inner_res_xgb_OvR, best_param_OvR, best_param_xgb_OvR)
#analysis <- analyze_class_specific_rejection_results(kappas)
mean(kappas$kappa)

#####

# Function to apply best class-specific cutoffs for filtering predictions
apply_class_specific_filtering <- function(results_df_svm, results_df_xgb, best_params_svm, best_params_xgb, optimal_cutoffs) {
  best_params_svm <- add_labels(best_params_svm)
  best_params_xgb <- add_labels(best_params_xgb)
  
  outer_folds <- unique(results_df_svm$outer_fold)
  filtered_results <- data.frame()
  
  for (fold in outer_folds) {
    print(paste("Processing outer fold:", fold))
    df_outer_svm <- results_df_svm[results_df_svm$outer_fold == fold, ]
    df_outer_xgb <- results_df_xgb[results_df_xgb$outer_fold == fold, ]
    
    inner_folds <- unique(df_outer_svm$inner_fold)
    
    for (inner_fold in inner_folds) {
      probs_svm <- df_outer_svm[df_outer_svm$inner_fold == inner_fold, ]
      probs_xgb <- df_outer_xgb[df_outer_xgb$inner_fold == inner_fold, ]
      
      class_labels <- unique(probs_svm$class_label)
      n_classes <- length(class_labels)
      n_samples <- length(clean_probs_str(probs_svm$preds_prob[1]))
      
      probs_df <- data.frame(matrix(ncol = n_classes, nrow = n_samples))
      colnames(probs_df) <- class_labels
      probs_df$truth <- NA
      
      for (cl in class_labels) {
        best_param_svm <- best_params_svm[
          best_params_svm$outer_fold == fold & best_params_svm$class_label == cl, 
        ]$params
        
        best_param_xgb <- best_params_xgb[
          best_params_xgb$outer_fold == fold & best_params_xgb$class_label == cl, 
        ]$params
        
        probs_svm_best <- probs_svm[
          probs_svm$class_label == cl & probs_svm$params == best_param_svm, 
        ]
        
        probs_xgb_best <- probs_xgb[
          probs_xgb$class_label == cl & probs_xgb$params == best_param_xgb, 
        ]
        
        probs_svm_clean <- clean_probs_str(probs_svm_best$preds_prob)
        probs_xgb_clean <- clean_probs_str(probs_xgb_best$preds_prob)
        
        # Average probabilities
        probs_df[[as.character(cl)]] <- (probs_svm_clean + probs_xgb_clean)/2
        
        y_val <- clean_y_str(probs_svm_best$y_val)
        probs_df$truth[y_val == 1] <- cl
      }
      
      truth_original <- probs_df$truth
      pred_indices <- apply(probs_df[, class_labels], 1, which.max)
      preds_original <- class_labels[pred_indices]
      
      # Apply class-specific filtering using optimal cutoffs
      rejection_mask <- rep(FALSE, n_samples)
      
      for (cl in class_labels) {
        # Get optimal cutoff for this class
        optimal_cutoff <- optimal_cutoffs$optimal_cutoff[optimal_cutoffs$class == cl]
        
        # Reject samples where this class was predicted but probability is below cutoff
        class_probs <- probs_df[[as.character(cl)]]
        class_rejection <- class_probs < optimal_cutoff & preds_original == cl
        rejection_mask <- rejection_mask | class_rejection
      }
      
      # Separate accepted and rejected samples
      accepted_indices <- !rejection_mask
      rejected_indices <- rejection_mask
      
      # Calculate metrics for accepted samples
      accepted_truth <- truth_original[accepted_indices]
      accepted_preds <- preds_original[accepted_indices]
      
      # Calculate metrics for rejected samples
      rejected_accuracy <- NA
      if (sum(rejected_indices) > 0) {
        rejected_truth <- truth_original[rejected_indices]
        rejected_preds <- preds_original[rejected_indices]
        rejected_accuracy <- sum(rejected_truth == rejected_preds) / sum(rejected_indices, na.rm = TRUE)
      }
      
      # Calculate kappa for accepted samples
      levels_labels <- unique(c(accepted_truth, accepted_preds))
      if (length(levels_labels) > 1) {
        accepted_truth_factored <- factor(accepted_truth, levels = levels_labels)
        accepted_preds_factored <- factor(accepted_preds, levels = levels_labels)
        
        res <- caret::confusionMatrix(accepted_preds_factored, accepted_truth_factored)
        kappa <- as.numeric(res$overall["Kappa"])
        accuracy <- as.numeric(res$overall["Accuracy"])
      } else {
        kappa <- NA
        accuracy <- NA
      }
      
      # Store filtered results
      filtered_results <- rbind(
        filtered_results,
        data.frame(
          outer_fold = fold,
          inner_fold = inner_fold,
          kappa = kappa,
          accuracy = accuracy,
          n_accepted = sum(accepted_indices, na.rm = TRUE),
          n_rejected = sum(rejected_indices, na.rm = TRUE),
          rejected_accuracy = rejected_accuracy,
          total_samples = n_samples,
          rejection_rate = sum(rejected_indices, na.rm = TRUE) / n_samples
        )
      )
    }
  }
  
  return(filtered_results)
}

# Alternative function that applies filtering and returns detailed per-sample results
apply_class_specific_filtering_detailed <- function(results_df_svm, results_df_xgb, best_params_svm, best_params_xgb, optimal_cutoffs) {
  best_params_svm <- add_labels(best_params_svm)
  best_params_xgb <- add_labels(best_params_xgb)
  
  outer_folds <- unique(results_df_svm$outer_fold)
  all_sample_results <- data.frame()
  
  for (fold in outer_folds) {
    print(paste("Processing outer fold:", fold))
    df_outer_svm <- results_df_svm[results_df_svm$outer_fold == fold, ]
    df_outer_xgb <- results_df_xgb[results_df_xgb$outer_fold == fold, ]
    
    inner_folds <- unique(df_outer_svm$inner_fold)
    
    for (inner_fold in inner_folds) {
      probs_svm <- df_outer_svm[df_outer_svm$inner_fold == inner_fold, ]
      probs_xgb <- df_outer_xgb[df_outer_xgb$inner_fold == inner_fold, ]
      
      class_labels <- unique(probs_svm$class_label)
      n_classes <- length(class_labels)
      n_samples <- length(clean_probs_str(probs_svm$preds_prob[1]))
      
      probs_df <- data.frame(matrix(ncol = n_classes, nrow = n_samples))
      colnames(probs_df) <- class_labels
      probs_df$truth <- NA
      
      for (cl in class_labels) {
        best_param_svm <- best_params_svm[
          best_params_svm$outer_fold == fold & best_params_svm$class_label == cl, 
        ]$params
        
        best_param_xgb <- best_params_xgb[
          best_params_xgb$outer_fold == fold & best_params_xgb$class_label == cl, 
        ]$params
        
        probs_svm_best <- probs_svm[
          probs_svm$class_label == cl & probs_svm$params == best_param_svm, 
        ]
        
        probs_xgb_best <- probs_xgb[
          probs_xgb$class_label == cl & probs_xgb$params == best_param_xgb, 
        ]
        
        probs_svm_clean <- clean_probs_str(probs_svm_best$preds_prob)
        probs_xgb_clean <- clean_probs_str(probs_xgb_best$preds_prob)
        
        # Average probabilities
        probs_df[[as.character(cl)]] <- (probs_svm_clean + probs_xgb_clean)/2
        
        y_val <- clean_y_str(probs_svm_best$y_val)
        probs_df$truth[y_val == 1] <- cl
      }
      
      truth_original <- probs_df$truth
      pred_indices <- apply(probs_df[, class_labels], 1, which.max)
      preds_original <- class_labels[pred_indices]
      max_probs <- apply(probs_df[, class_labels], 1, max)
      
      # Apply class-specific filtering using optimal cutoffs
      rejection_reason <- rep("accepted", n_samples)
      
      for (cl in class_labels) {
        # Get optimal cutoff for this class
        optimal_cutoff <- optimal_cutoffs$optimal_cutoff[optimal_cutoffs$class == cl]
        
        # Mark samples for rejection
        class_rejection <- max_probs < optimal_cutoff & preds_original == cl
        rejection_reason[class_rejection] <- paste("rejected_", cl, "_below_", optimal_cutoff, sep = "")
      }
      
      # Create per-sample results
      sample_results <- data.frame(
        outer_fold = fold,
        inner_fold = inner_fold,
        sample_id = 1:n_samples,
        truth = truth_original,
        prediction = preds_original,
        max_probability = max_probs,
        rejected = rejection_reason != "accepted",
        rejection_reason = rejection_reason,
        correct_prediction = truth_original == preds_original
      )
      
      all_sample_results <- rbind(all_sample_results, sample_results)
    }
  }
  
  return(all_sample_results)
}


analysis <- analyze_class_specific_rejection_results(kappas)
optimal_cutoffs <- analysis$optimal_cutoffs

filtered_results <- apply_class_specific_filtering(
  inner_res_OvR, inner_res_xgb_OvR, 
  best_param_OvR, best_param_xgb_OvR,
  optimal_cutoffs
)

# Get detailed results for each sample
detailed_results <- apply_class_specific_filtering_detailed(
  inner_res_OvR, inner_res_xgb_OvR, 
  best_param_OvR, best_param_xgb_OvR,
  optimal_cutoffs
)

# Analyze rejection patterns
rejection_summary <- detailed_results %>%
  group_by(rejection_reason) %>%
  summarise(
    n_samples = n(),
    accuracy = mean(correct_prediction),
    mean_probability = mean(max_probability)
  )
rejection_summary
# =============================================================================
# Save Results
# =============================================================================


# Soft voting ensemble: sum probabilities from all predictors (SVM, XGB, NN)
ensemble_soft_voting_nested_cv <- function(results_df_svm, results_df_xgb, results_df_nn,
                                           best_params_svm, best_params_xgb, best_params_nn,
                                           label_mapping) {
  best_params_svm <- add_labels(best_params_svm)
  best_params_xgb <- add_labels(best_params_xgb)
  
  outer_folds <- unique(results_df_svm$outer_fold)
  kappa_scores <- data.frame()
  
  for (fold in outer_folds) {
    df_outer_svm <- results_df_svm[results_df_svm$outer_fold == fold, ]
    df_outer_xgb <- results_df_xgb[results_df_xgb$outer_fold == fold, ]
    df_outer_nn  <- results_df_nn[results_df_nn$outer_fold == fold, ]
    inner_folds <- unique(df_outer_svm$inner_fold)
    
    for (inner_fold in inner_folds) {
      probs_svm <- df_outer_svm[df_outer_svm$inner_fold == inner_fold, ]
      probs_xgb <- df_outer_xgb[df_outer_xgb$inner_fold == inner_fold, ]
      best_param_nn <- best_params_nn[best_params_nn$outer_fold == fold, ]$params
      probs_nn <- df_outer_nn[df_outer_nn$inner_fold == inner_fold & df_outer_nn$params == best_param_nn, ]
      
      class_labels <- label_mapping$Label
      n_classes <- length(class_labels)
      n_samples <- length(clean_probs_str(probs_svm$preds_prob[1]))
      
      probs_nn_matrix <- text_to_matrix(probs_nn$preds_prob, ncol = n_classes)
      colnames(probs_nn_matrix) <- class_labels
      
      probs_df <- data.frame(matrix(0, nrow = n_samples, ncol = n_classes))
      colnames(probs_df) <- class_labels
      probs_df$truth <- NA
      
      for (cl in class_labels) {
        # SVM
        best_param_svm <- best_params_svm[
          best_params_svm$outer_fold == fold & best_params_svm$class_label == cl, 
        ]$params
        probs_svm_best <- probs_svm[
          probs_svm$class_label == cl & probs_svm$params == best_param_svm, 
        ]
        probs_svm_clean <- clean_probs_str(probs_svm_best$preds_prob)
        
        # XGB
        best_param_xgb <- best_params_xgb[
          best_params_xgb$outer_fold == fold & best_params_xgb$class_label == cl, 
        ]$params
        probs_xgb_best <- probs_xgb[
          probs_xgb$class_label == cl & probs_xgb$params == best_param_xgb, 
        ]
        probs_xgb_clean <- clean_probs_str(probs_xgb_best$preds_prob)
        
        # NN
        probs_nn_clean <- probs_nn_matrix[, cl]
        
        # Sum probabilities (soft voting)
        probs_df[[cl]] <- probs_svm_clean + probs_xgb_clean + probs_nn_clean
        
        # Assign truth labels (use SVM y_val, assumed consistent)
        y_val <- clean_y_str(probs_svm_best$y_val)
        probs_df$truth[y_val == 1] <- cl
      }
      
      truth <- merge_classes(probs_df$truth)
      pred_indices <- apply(probs_df[, class_labels], 1, which.max)
      preds <- merge_classes(class_labels[pred_indices])
      
      levels_labels <- unique(c(truth, preds))
      truth <- factor(truth, levels = levels_labels)
      preds <- factor(preds, levels = levels_labels)
      
      res <- caret::confusionMatrix(preds, truth)
      kappa <- as.numeric(res$overall["Kappa"])
      
      kappa_scores <- rbind(
        kappa_scores,
        data.frame(outer_fold = fold, inner_fold = inner_fold, kappa = kappa)
      )
    }
  }
  return(kappa_scores)
}

# Example usage for soft voting ensemble:
kappas_ensemble_soft_voting <- ensemble_soft_voting_nested_cv(
  inner_res_OvR, inner_res_xgb_OvR, inner_res_nn_standard,
  best_param_OvR, best_param_xgb_OvR, best_param_nn_standard,
  label_mapping
)
kappas_ensemble_soft_voting %>% group_by(outer_fold) %>% summarize(mean_kappa = mean(kappa))

# cat("Saving SVM results...\n")
# # Write to CSV (SVM)
# out_dir_svm <- "inner_cv_best_params/SVM"
# dir.create(out_dir_svm, recursive = TRUE, showWarnings = FALSE)
# 
# write.csv(best_param_OvO,           file = file.path(out_dir_svm, "SVM_best_param_OvO.csv"), row.names = FALSE)
# write.csv(best_param_OvR,           file = file.path(out_dir_svm, "SVM_best_param_OvR.csv"), row.names = FALSE)
# write.csv(best_param_standard,      file = file.path(out_dir_svm, "SVM_best_param_standard.csv"), row.names = FALSE)
# write.csv(best_param_OvO_loso,      file = file.path(out_dir_svm, "SVM_best_param_OvO_loso.csv"), row.names = FALSE)
# write.csv(best_param_OvR_loso,      file = file.path(out_dir_svm, "SVM_best_param_OvR_loso.csv"), row.names = FALSE)
# write.csv(best_param_standard_loso, file = file.path(out_dir_svm, "SVM_best_param_standard_loso.csv"), row.names = FALSE)
# 
# cat("Saving XGBOOST results...\n")
# # Write to CSV (XGBOOST)
# out_dir_xgb <- "/Users/jsevere2/Documents/AML_PhD/leukem_ai/inner_cv_best_params/XGBOOST"
# dir.create(out_dir_xgb, recursive = TRUE, showWarnings = FALSE)
# 
# write.csv(best_param_xgb_OvO,           file = file.path(out_dir_xgb, "XGBOOST_best_param_OvO.csv"), row.names = FALSE)
# write.csv(best_param_xgb_OvR,           file = file.path(out_dir_xgb, "XGBOOST_best_param_OvR.csv"), row.names = FALSE)
# write.csv(best_param_xgb_standard,      file = file.path(out_dir_xgb, "XGBOOST_best_param_standard.csv"), row.names = FALSE)
# write.csv(best_param_xgb_OvO_loso,      file = file.path(out_dir_xgb, "XGBOOST_best_param_OvO_loso.csv"), row.names = FALSE)
# write.csv(best_param_xgb_OvR_loso,      file = file.path(out_dir_xgb, "XGBOOST_best_param_OvR_loso.csv"), row.names = FALSE)
# write.csv(best_param_xgb_standard_loso, file = file.path(out_dir_xgb, "XGBOOST_best_param_standard_loso.csv"), row.names = FALSE)
# 
# cat("Analysis complete! Results saved to:\n")
# cat("- SVM: ", out_dir_svm, "\n")
# cat("- XGBOOST: ", out_dir_xgb, "\n") 