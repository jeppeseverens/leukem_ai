#!/usr/bin/env python3
"""
Generate predictions for left-out subtypes (AML NOS, Missing data, Multi).

This script:
- Loads the full dataset (meta, counts, RGAs) from ../data.
- Identifies samples whose ICC_Subtype is in the left-out set and whose study
  is in the same selected studies used for training.
- Builds a CSV with these samples (rows) and genes (columns), matching the
  format expected by predict_new_samples.py.
- Runs predict_new_samples.py on that CSV so you can inspect which major
  classes and probabilities are assigned to the left-out subtypes.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Subtypes that were excluded in the training filters
EXCLUDED_SUBTYPES = ["AML NOS", "Multi"]

# Studies used in the main analyses (mirrors R and python/train_test.py)
SELECTED_STUDIES = [
    "TCGA-LAML",
    "LEUCEGENE",
    "BEATAML1.0-COHORT",
    "AAML0531",
    "AAML1031",
    "AAML03P1",
    "100LUMC",
]


def build_leftout_matrix(data_dir: Path, output_csv: Path) -> pd.DataFrame:
    """
    Build a new-samples matrix (samples x genes) for:
    - All subtypes with n < 10, plus
    - Explicitly excluded subtypes (AML NOS, Missing data, Multi),
    restricted to the selected studies.

    Assumes meta, counts and RGAs are already perfectly aligned by row
    order (as in python/train_test.load_data).

    Returns the DataFrame that was written, or an empty frame if nothing matched.
    """
    meta_path = data_dir / "meta_20aug25.csv"
    counts_path = data_dir / "counts_20aug25.csv"
    rgas_path = data_dir / "rgas_10feb26.csv"

    if not meta_path.exists() or not counts_path.exists() or not rgas_path.exists():
        raise FileNotFoundError(
            f"Expected files not found in {data_dir}: "
            f"{meta_path.name}, {counts_path.name}, {rgas_path.name}"
        )

    # Load counts and put samples on rows, genes on columns (mirrors train_test.load_data)
    counts_df = pd.read_csv(counts_path)
    counts_df = counts_df.set_index(counts_df.columns[0])
    counts_df.index.name = None
    counts_df.columns.name = None
    X_df = counts_df.transpose()  # rows = samples, columns = genes

    # Load subtypes and studies in the same order
    rgas_df = pd.read_csv(rgas_path, index_col=0)
    meta_df = pd.read_csv(meta_path)

    if "ICC_Subtype" not in rgas_df.columns:
        raise ValueError("rgas_10feb26.csv must contain 'ICC_Subtype' column.")
    if "Studies" not in meta_df.columns:
        raise ValueError("meta_20aug25.csv must contain 'Studies' column.")

    y_series = rgas_df["ICC_Subtype"].reset_index(drop=True)
    study_series = meta_df["Studies"].reset_index(drop=True)

    if not (len(y_series) == len(study_series) == X_df.shape[0]):
        raise ValueError(
            "meta, counts and RGAs are not aligned in length; "
            "expected perfect alignment by row order."
        )

    # Identify all classes with n < 10, plus explicitly excluded subtypes
    class_counts = y_series.value_counts()
    rare_classes = class_counts[class_counts < 10].index.tolist()
    leftout_labels = sorted(set(rare_classes) | set(EXCLUDED_SUBTYPES))

    subtype_mask = y_series.isin(leftout_labels)
    study_mask = study_series.isin(SELECTED_STUDIES)
    keep_mask = subtype_mask & study_mask

    if not keep_mask.any():
        print("No samples found for rare/left-out subtypes in selected studies.")
        return pd.DataFrame()

    leftout_matrix = X_df.loc[keep_mask.values].copy()
    print(f"Selected {leftout_matrix.shape[0]} left-out/rare samples across {leftout_matrix.shape[1]} genes.")

    # Simple per-subtype summary
    summary = (
        pd.DataFrame(
            {
                "Subtype": y_series[keep_mask].values,
                "Study": study_series[keep_mask].values,
            }
        )
        .value_counts(["Subtype", "Study"])
        .reset_index(name="n")
        .sort_values(["Subtype", "Study"])
    )
    print("Left-out/rare subtype counts (Subtype x Study):")
    print(summary.head(5))

    # Write CSV in the format expected by predict_new_samples.py:
    # - rows = samples
    # - columns = genes
    # - first column = sample names (index)
    leftout_matrix.to_csv(output_csv)
    print(f"Wrote left-out samples CSV to: {output_csv}")

    return leftout_matrix


def run_predict_new_samples(
    input_csv: Path,
    output_dir: Path,
    max_accepted_risk_pct: float | None = None,
    merged_only: bool = False,
    unmerged_only: bool = False,
):
    """
    Call predict_new_samples.py on the generated left-out samples CSV.
    """
    script_dir = Path(__file__).resolve().parent
    predict_script = script_dir / "predict_new_samples.py"

    if not predict_script.exists():
        raise FileNotFoundError(f"predict_new_samples.py not found at {predict_script}")

    cmd = [
        sys.executable,
        str(predict_script),
        "--input_file",
        str(input_csv),
        "--output_dir",
        str(output_dir),
    ]

    if max_accepted_risk_pct is not None:
        cmd.extend(["--max_accepted_risk_pct", str(max_accepted_risk_pct)])
    if merged_only:
        cmd.append("--merged_only")
    if unmerged_only:
        cmd.append("--unmerged_only")

    print("Running predict_new_samples.py on left-out samples:")
    print("  " + " ".join(cmd))

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a new-samples file containing left-out subtypes "
            "(AML NOS, Missing data, Multi) and run predict_new_samples.py on it."
        )
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Path to directory with meta_*, counts_*, rgas_* CSVs (default: ../data).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where prediction outputs from predict_new_samples.py will be written.",
    )
    parser.add_argument(
        "--max_accepted_risk_pct",
        type=float,
        default=None,
        help=(
            "Optional maximum accepted error rate (percentage) on accepted predictions. "
            "If set, this value is forwarded to predict_new_samples.py "
            "via --max_accepted_risk_pct."
        ),
    )
    parser.add_argument(
        "--merged_only",
        action="store_true",
        help="Forwarded flag to predict_new_samples.py to run only merged predictions.",
    )
    parser.add_argument(
        "--unmerged_only",
        action="store_true",
        help="Forwarded flag to predict_new_samples.py to run only unmerged predictions.",
    )

    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir is not None else (base_path / "data")
    output_dir = Path(args.output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    leftout_csv = output_dir / "leftout_subtypes_for_prediction.csv"

    leftout_df = build_leftout_matrix(data_dir, leftout_csv)
    if leftout_df.empty:
        print("No left-out samples to predict on; exiting.")
        return

    run_predict_new_samples(
        input_csv=leftout_csv,
        output_dir=output_dir,
        max_accepted_risk_pct=args.max_accepted_risk_pct,
        merged_only=args.merged_only,
        unmerged_only=args.unmerged_only,
    )


if __name__ == "__main__":
    main()

