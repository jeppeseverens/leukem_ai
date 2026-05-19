import argparse
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_test
import transformers


def main():
    parser = argparse.ArgumentParser(
        description="Backfill KNN reject-feature columns into an existing outer-CV CSV."
    )
    parser.add_argument("--outer_csv", required=True, help="Path to existing outer-CV CSV file.")
    parser.add_argument("--fold_type", required=True, choices=["CV", "loso"], help="Fold type used in the CSV.")
    parser.add_argument("--fs_method", default="eta2", choices=["eta2", "mad"], help="Feature selection method.")
    parser.add_argument("--knn_n_genes", type=int, default=500, help="n_genes for KNN feature space.")
    parser.add_argument("--cache_dir", default=None, help="Optional cache directory for preprocessing/KNN artifacts.")
    parser.add_argument("--output_csv", default=None, help="Optional explicit output path. Defaults to *_with_knn.csv.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input CSV in-place.")
    parser.add_argument(
        "--leftout_mode",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Whether the target CSV is a left-out prediction file. 'auto' infers from filename containing '_leftout_'.",
    )
    args = parser.parse_args()

    outer_csv = Path(args.outer_csv).resolve()
    if not outer_csv.exists():
        raise FileNotFoundError(f"outer_csv not found: {outer_csv}")

    print(f"Loading existing outer-CV CSV: {outer_csv}")
    outer_df = pd.read_csv(outer_csv)

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data"
    X_all, y_all, study_all = train_test.load_data(data_path)
    X, y, study_labels = train_test.filter_data(X_all, y_all, study_all, min_n=10)
    y, _ = train_test.encode_labels(y)
    X_leftout, y_leftout, study_leftout, leftout_global_idx = train_test.get_leftout_samples(
        X_all, y_all, study_all, min_n=10
    )

    if args.fs_method == "eta2":
        feature_selector = transformers.FeatureSelectionEta()
    else:
        feature_selector = transformers.FeatureSelection2()

    pipe = Pipeline(
        [
            ("DEseq2", transformers.DESeq2RatioNormalizer()),
            ("feature_selection", feature_selector),
            ("scaler", StandardScaler()),
        ]
    )

    is_leftout = (
        (args.leftout_mode == "yes")
        or (args.leftout_mode == "auto" and "_leftout_" in outer_csv.name)
    )
    print(f"Computing fold-matched KNN vectors and backfilling CSV rows (leftout={is_leftout})...")
    if is_leftout:
        out_df = train_test.backfill_knn_columns_in_outer_leftout_results(
            leftout_results_df=outer_df,
            X=X,
            y=y,
            study_labels=study_labels,
            X_leftout=X_leftout,
            y_leftout=y_leftout,
            study_leftout=study_leftout,
            leftout_global_idx=leftout_global_idx,
            pipe=pipe,
            fold_type=args.fold_type,
            fs_method=args.fs_method,
            knn_n_genes=args.knn_n_genes,
            cache_dir=args.cache_dir,
            strict=True,
        )
    else:
        out_df = train_test.backfill_knn_columns_in_outer_results(
            outer_results_df=outer_df,
            X=X,
            y=y,
            study_labels=study_labels,
            pipe=pipe,
            fold_type=args.fold_type,
            fs_method=args.fs_method,
            knn_n_genes=args.knn_n_genes,
            cache_dir=args.cache_dir,
            strict=True,
        )

    if args.overwrite and args.output_csv is not None:
        raise ValueError("Use either --overwrite or --output_csv, not both.")

    if args.overwrite:
        out_path = outer_csv
    elif args.output_csv:
        out_path = Path(args.output_csv).resolve()
    else:
        out_path = outer_csv.with_name(f"{outer_csv.stem}_with_knn{outer_csv.suffix}")

    out_df.to_csv(out_path, index=False)
    print(f"Saved backfilled CSV: {out_path}")


if __name__ == "__main__":
    main()
