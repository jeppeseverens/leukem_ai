source("inner_cv_analysis.R")

# Override model configs to point to eta2-based feature selection outputs

dir.create("../data/out/inner_cv/inner_cv_best_params_fs_eta", showWarnings = FALSE)

MODEL_CONFIGS_ETA <- list(
  svm = list(
    classification_type = "OvR",
    file_paths = list(
      cv = "../data/out/inner_cv/SVM_array/cv_28jan26_all_fs_eta/",
      loso = "../data/out/inner_cv/SVM_array/loso_28jan26_all_fs_eta/"
    ),
    output_dir = "../data/out/inner_cv/inner_cv_best_params_fs_eta/SVM_10feb26_fs_eta"
  ),
  xgboost = list(
    classification_type = "OvR",
    file_paths = list(
      cv = "../data/out/inner_cv/XGBOOST_array/cv_28jan26_all_fs_eta/",
      loso = "../data/out/inner_cv/XGBOOST_array/loso_28jan26_all_fs_eta/"
    ),
    output_dir = "../data/out/inner_cv/inner_cv_best_params_fs_eta/XGBOOST_10feb26_fs_eta"
  ),
  neural_net = list(
    classification_type = "standard",
    file_paths = list(
      cv = "../data/out/inner_cv/NN_array/cv_28jan26_all_fs_eta/",
      loso = "../data/out/inner_cv/NN_array/loso_28jan26_all_fs_eta/"
    ),
    output_dir = "../data/out/inner_cv/inner_cv_best_params_fs_eta/NN_10feb26_fs_eta"
  )
)

# Run the same analysis pipeline but on eta2-based feature selection outputs.
model_results_eta <- load_all_model_data(MODEL_CONFIGS_ETA)
best_parameters_eta <- extract_all_best_parameters(model_results_eta, MODEL_CONFIGS_ETA)
save_all_best_parameters(best_parameters_eta, MODEL_CONFIGS_ETA)

