"""
Option B deployment calibration: LOSO with 6-study final models scoring the held-out study.

Known cohort only (no left-out OOD rows). Outputs CSVs for R/build_deploy_loso_fold_matrices.R.
"""
import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import classifiers
import train_test
import transformers

MODEL_CONFIG = {
    "SVM": {
        "multi_type": "OvR",
        "model": None,  # set in main
        "best_params_rel": "data/out/final_train_test/best_params/SVM/SVM_best_param_loso.csv",
    },
    "XGBOOST": {
        "multi_type": "OvR",
        "model": None,
        "best_params_rel": "data/out/final_train_test/best_params/XGBOOST/XGBOOST_best_param_loso.csv",
    },
    "NN": {
        "multi_type": "standard",
        "model": None,
        "best_params_rel": "data/out/final_train_test/best_params/NN/NEURAL_NET_best_param_loso.csv",
    },
}


def build_pipe(fs_method: str) -> Pipeline:
    fs_method = fs_method.lower()
    if fs_method == "eta2":
        feature_selector = transformers.FeatureSelectionEta()
    elif fs_method == "mad":
        feature_selector = transformers.FeatureSelection2()
    else:
        raise ValueError(f"Unknown fs_method '{fs_method}'. Use 'mad' or 'eta2'.")
    return Pipeline(
        [
            ("DEseq2", transformers.DESeq2RatioNormalizer()),
            ("feature_selection", feature_selector),
            ("scaler", StandardScaler()),
        ]
    )


def resolve_model_class(model_type: str):
    if model_type == "XGBOOST":
        return classifiers.WeightedXGBClassifier
    if model_type == "SVM":
        from sklearn.svm import SVC

        return SVC
    if model_type == "NN":
        return classifiers.NeuralNet
    raise ValueError(f"Unsupported model_type '{model_type}'.")


def run_model_deploy_loso(
    model_type: str,
    fs_method: str,
    repo_root: Path,
    output_root: Path,
) -> Path:
    cfg = MODEL_CONFIG[model_type]
    multi_type = cfg["multi_type"]
    model_cls = resolve_model_class(model_type)
    pipe = build_pipe(fs_method)

    best_params_path = repo_root / cfg["best_params_rel"]
    if not best_params_path.exists():
        raise FileNotFoundError(f"Missing best-params file: {best_params_path}")
    best_params = pd.read_csv(best_params_path)
    if best_params.empty:
        raise ValueError(f"Best-params file is empty: {best_params_path}")

    data_path = repo_root / "data"
    X, y, study_labels = train_test.load_data(data_path)
    X, y, study_labels = train_test.filter_data(X, y, study_labels, min_n=10)
    y, label_mapping = train_test.encode_labels(y)

    pipelines_dir = train_test.get_pipeline_cache_dir()
    model_out_dir = output_root / model_type.lower()
    if model_type == "NN":
        model_out_dir = output_root / "neural_net"
    model_out_dir.mkdir(parents=True, exist_ok=True)

    all_fold_dfs = []
    for holdout_study in sorted(set(study_labels)):
        print(f"[{model_type}] deploy-loso fold: hold out {holdout_study}", flush=True)
        fold_df = train_test.run_deploy_calibration_loso_fold(
            X,
            y,
            study_labels,
            holdout_study,
            model_cls,
            pipe,
            best_params,
            multi_type=multi_type,
            model_type=model_type,
            label_mapping=label_mapping,
            fs_method=fs_method,
            pipelines_dir=pipelines_dir,
        )
        if fold_df is None or len(fold_df) == 0:
            raise ValueError(f"No deploy-loso rows for {model_type} fold {holdout_study}.")
        all_fold_dfs.append(fold_df)

    combined = pd.concat(all_fold_dfs, ignore_index=True)
    out_name = f"{model_type}_deploy_loso_{multi_type}.csv"
    out_path = model_out_dir / out_name
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} rows to {out_path}", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Run Option B deploy-loso calibration scoring (6-study models, 7 study folds)."
    )
    parser.add_argument(
        "--model_type",
        choices=["SVM", "XGBOOST", "NN"],
        default="SVM",
        help="Model to score (default: SVM for deployment calibration).",
    )
    parser.add_argument(
        "--fs_method",
        default="eta2",
        help='Feature selection method: "eta2" (default) or "mad".',
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output root (default: data/out/final_train_test/deploy_loso).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root / "data" / "out" / "final_train_test" / "deploy_loso"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    run_model_deploy_loso(args.model_type, args.fs_method, repo_root, output_root)
    print("Deploy-loso calibration scoring complete.", flush=True)


if __name__ == "__main__":
    main()
