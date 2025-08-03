
library(dplyr)
library(stringr)

clean_probs_str <- function(probs_str) {
  probs_str %>%
    str_replace_all("\\[|\\]|\\{|\\}|\\\n|,", "") %>%  # remove brackets and line breaks
    str_squish() %>%                                 # collapse multiple spaces
    str_split(" ") %>%                               # split by space
    unlist() %>%
    as.numeric()
}

modify_classes <- function(vector){
  vector[grepl("MDS|TP53", vector)] <- "MDS-r"
  vector[!grepl("MLLT3", vector) & grepl("KMT2A", vector)] <- "other KMT2A"
  vector
}

rgas_path <- "../data/rgas.csv"
y <- read.csv(rgas_path)$ICC_Subtype

studies_path <- "../data/meta.csv"
studies <- read.csv(studies_path)$Studies

atleast_10 <- names(which(table(y) >=10))
exclude <- c("AML NOS", "Missing data")
selected_studies <- c(        
  "TCGA-LAML",
  "LEUCEGENE",
  "BEATAML1.0-COHORT",
  "AAML0531",
  "AAML1031"
)


y <- y[y %in% atleast_10 & !y %in% exclude & studies %in% selected_studies]

class_mapping <- read.csv("~/Documents/AML_PhD/leukem_ai/label_mapping_df_n10.csv")



cms_from_standard <- function(results_df){
  outer_folds <- unique(results_df$outer_fold)
  
  cm_list <- list()
  for (outer_fold in outer_folds){
    results_df_fold <- results_df[results_df$outer_fold == outer_fold,]
    
    test_indices <- clean_probs_str(results_df_fold$sample_indices[1]) + 1
    n_samples <- length(test_indices)
    prob_df_fold <- data.frame(t(matrix(clean_probs_str(results_df_fold$preds_prob), ncol = n_samples)))
    
    y_test_fold <- y[test_indices]
    
    classes <- clean_probs_str(results_df_fold$classes)
    classes <- class_mapping$Label[classes+1]
    colnames(prob_df_fold) <- classes
    preds <- colnames(prob_df_fold)[apply(prob_df_fold, 1, which.max)]
    
    y_mod <- modify_classes(y_test_fold)
    preds_mod <- modify_classes(preds)
    
    all_classes <- unique(c(y_mod,preds_mod))
    y_mod <- factor(y_mod, levels = all_classes)
    preds_mod <- factor(preds_mod, levels = all_classes)
    
    cm_list[[as.character(outer_fold)]] <- caret::confusionMatrix(preds_mod, y_mod)
  }
  return(cm_list)
}

cms_from_ovr <- function(results_df){
  outer_folds <- unique(results_df$outer_fold)
  
  cm_list <- list()
  for (outer_fold in outer_folds){
    results_df_fold <- results_df[results_df$outer_fold == outer_fold,]
    
    test_indices <- clean_probs_str(results_df_fold$sample_indices[1]) + 1
    n_samples <- length(test_indices)
    prob_df_fold <- data.frame(matrix(nrow = n_samples, ncol = 0))
    
    y_test_fold <- y[test_indices]
    
    classes <- unique(results_df_fold$class_label)
    for (i in 1:nrow(results_df_fold)){
      class <- results_df_fold$class_label[i]
      prob_df_fold[class] <- clean_probs_str(results_df_fold$preds_prob[i])
    }
    
    preds <- colnames(prob_df_fold)[apply(prob_df_fold, 1, which.max)]
    
    y_mod <- modify_classes(y_test_fold)
    preds_mod <- modify_classes(preds)
    
    all_classes <- unique(y_mod)
    y_mod <- factor(y_mod, levels = all_classes)
    preds_mod <- factor(preds_mod, levels = all_classes)
    
    cm_list[[as.character(outer_fold)]] <- caret::confusionMatrix(preds_mod, y_mod)
  }
  return(cm_list)
}

cbind.fill<-function(...){
  nm <- list(...) 
  nm<-lapply(nm, as.matrix)
  n <- max(sapply(nm, nrow)) 
  do.call(cbind, lapply(nm, function (x) 
    rbind(x, matrix(, n-nrow(x), ncol(x))))) 
}

mean_overall <- function(cm_list, model, type){
  kappas <- sapply(cm_list, function(x) x$overall["Kappa"])
  mean_kappas <- mean(kappas)
  sd_kappas <- sd(kappas)
  
  acc <- sapply(cm_list, function(x) x$overall["Accuracy"])
  mean_acc <- mean(acc)
  sd_acc <- sd(acc)
  
  return(data.frame(model, type, mean_kappas, sd_kappas, mean_acc, sd_acc))
}

mean_per_class <- function(cm_list, model, type){
  byClass_Balanced.Accuracy <- do.call(cbind.fill,lapply(cm_list, function(x) data.frame(x$byClass)["Balanced.Accuracy"]))
  classes <- rownames(byClass_Balanced.Accuracy)
  mean_Balanced.Accuracy <- apply(byClass_Balanced.Accuracy, 1, mean, na.rm = TRUE)
  sd_Balanced.Accuracy <- apply(byClass_Balanced.Accuracy, 1, sd, na.rm = TRUE)
  
  byClass_F1 <- do.call(cbind.fill,lapply(cm_list, function(x) data.frame(x$byClass)["F1"]))
  mean_F1 <- apply(byClass_F1, 1, mean, na.rm = TRUE)
  sd_F1 <- apply(byClass_F1, 1, sd, na.rm = TRUE)
  
  byClass_Sensitivity <- do.call(cbind.fill,lapply(cm_list, function(x) data.frame(x$byClass)["Sensitivity"]))
  mean_Sensitivity <- apply(byClass_Sensitivity, 1, mean, na.rm = TRUE)
  sd_Sensitivity <- apply(byClass_Sensitivity, 1, sd, na.rm = TRUE)
  
  byClass_Specificity <- do.call(cbind.fill,lapply(cm_list, function(x) data.frame(x$byClass)["Specificity"]))
  mean_Specificity <- apply(byClass_Specificity, 1, mean, na.rm = TRUE)
  sd_Specificity <- apply(byClass_Specificity, 1, sd, na.rm = TRUE)
  
  data.frame(model, type, "class" = classes, mean_Balanced.Accuracy, sd_Balanced.Accuracy, mean_F1, sd_F1, mean_Sensitivity, sd_Sensitivity, mean_Specificity, sd_Specificity)
}

expand_cm <- function(cm_table, all_levels) {
  # Create a full zero matrix
  full_cm <- matrix(0, nrow = length(all_levels), ncol = length(all_levels),
                    dimnames = list(Reference = all_levels, Prediction = all_levels))
  
  # Fill in the known values
  full_cm[rownames(cm_table), colnames(cm_table)] <- cm_table
  return(full_cm)
}

combine_cms <- function(cms){
  cms_only <- lapply(cms, function(x) x$table)
  all_levels <- sort(unique(unlist(lapply(cms_only, function(x) union(rownames(x), colnames(x))))))
  expanded_cms <- lapply(cms_only, expand_cm, all_levels = all_levels)
  combined_cm <- Reduce(`+`, expanded_cms)
  combined_cm
}

analyse <- function(file, model, type, method, y){
  cat(sprintf("\nProcessing now: Model: %s  Type: %s  Method: %s\n", model, type, method))
  results_df <- read.csv(file)
  
  if (method == "ovr") {
    cms <- cms_from_ovr(results_df)
  } else if (method == "standard"){
    cms <- cms_from_standard(results_df)
  } else {
    stop(paste0("for file ", file, " with ", model, " and ", type,", method does not match 'ovr' or 'standard'"))
  }
  means_df <- mean_overall(cms, model, type)
  per_class_df <- mean_per_class(cms, model, type)
  combined_cm <- combine_cms(cms)
  
  out <- list(means_df, per_class_df, combined_cm)
  return(out)
}

analyse_all <- function(to_analyse){
  out <- lapply(to_analyse, function(x){
    analyse(x[1], x[2], x[3], x[4], y)
  })
  overall <- do.call(plyr::rbind.fill, lapply(out, function(x) x[[1]]))
  per_class <- do.call(plyr::rbind.fill, lapply(out, function(x) x[[2]]))
  combined_cm <- lapply(out, function(x) x[[3]])
  
  out <- list(overall, per_class, combined_cm)
  names(out) <- c("overall", "per_class", "cms")
  return(out)
}


to_analyse <- list(
  c("../out/outer_cv/SVM_n10/SVM_outer_cv_CV_OvR_20250703_1254.csv", "SVM", "CV", "ovr"),
  c("../out/outer_cv/SVM_n10/SVM_outer_cv_loso_OvR_20250703_1309.csv", "SVM", "LOSO", "ovr"),
  c("../out/outer_cv/XGBOOST_n10/XGBOOST_outer_cv_CV_OvR_20250703_1259.csv", "XGBOOST", "CV", "ovr"),
  c("../out/outer_cv/XGBOOST_n10/XGBOOST_outer_cv_loso_OvR_20250703_1312.csv", "XGBOOST", "LOSO", "ovr"),
  c("../out/outer_cv/NN_n10/NN_outer_cv_CV_standard_20250731_1756.csv", "NN", "CV", "standard"),
  c("../out/outer_cv/NN_n10/NN_outer_cv_loso_standard_20250731_1807.csv", "NN", "LOSO", "standard")
)

analyse_all(to_analyse)