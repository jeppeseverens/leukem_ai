import argparse
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_test
import transformers


def _is_final_leftout_csv(csv_path: Path, df: pd.DataFrame) -> bool:
    if "_leftout" in csv_path.name or "_final_" in csv_path.name:
        if "outer_fold" in df.columns:
            folds = pd.unique(df["outer_fold"])
            if len(folds) == 1 and int(folds[0]) == train_test.FINAL_LEFTOUT_OUTER_FOLD:
                return True
        if "_leftout" in csv_path.name:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Backfill KNN reject-feature columns into final-selection or final-leftout CSVs."
    )
    parser.add_argument("--input_csv", required=True, help="Path to final-selection or final-leftout CSV.")
    parser.add_argument(
        "--mode",
        choices=["auto", "selection", "leftout"],
        default="auto",
        help="CSV type: final-selection CV folds, final left-out, or auto-detect.",
    )
    parser.add_argument("--fold_type", default="loso", choices=["CV", "loso"], help="Fold type for selection CSVs.")
    parser.add_argument("--fs_method", default="eta2", choices=["eta2", "mad"], help="Feature selection method.")
    parser.add_argument("--knn_n_genes", type=int, default=500, help="n_genes for KNN feature space.")
    parser.add_argument("--cache_dir", default=None, help="Optional cache directory.")
    parser.add_argument("--output_csv", default=None, help="Optional output path. Defaults to *_with_knn.csv.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input CSV in-place.")
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")

    print(f"Loading CSV: {input_csv}")
    df = pd.read_csv(input_csv)

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

    if args.mode == "leftout":
        is_leftout = True
    elif args.mode == "selection":
        is_leftout = False
    else:
        is_leftout = _is_final_leftout_csv(input_csv, df)

    print(f"Backfilling KNN (mode={'leftout' if is_leftout else 'selection'})...")
    if is_leftout:
        out_df = train_test.backfill_knn_columns_in_final_leftout_results(
            leftout_results_df=df,
            X=X,
            y=y,
            study_labels=study_labels,
            X_leftout=X_leftout,
            leftout_global_idx=leftout_global_idx,
            pipe=pipe,
            fs_method=args.fs_method,
            knn_n_genes=args.knn_n_genes,
            cache_dir=args.cache_dir,
            strict=True,
        )
    else:
        out_df = train_test.backfill_knn_columns_in_outer_results(
            outer_results_df=df,
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
        out_path = input_csv
    elif args.output_csv:
        out_path = Path(args.output_csv).resolve()
    else:
        out_path = input_csv.with_name(f"{input_csv.stem}_with_knn{input_csv.suffix}")

    out_df.to_csv(out_path, index=False)
    print(f"Saved backfilled CSV: {out_path}")


if __name__ == "__main__":
    main()
