#!/usr/bin/env python3
"""Validate exported elastic-net params CSV against optional feature rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from enet_rejector_scoring import score_enet_head_from_feature_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params_csv", required=True, help="Exported multivariate_params_*.csv")
    parser.add_argument(
        "--features_csv",
        default=None,
        help="Optional CSV with feature columns used to smoke-test scoring.",
    )
    args = parser.parse_args()

    params = pd.read_csv(args.params_csv)
    heads = sorted(params["head"].unique())
    print(f"Loaded {len(params)} coefficient rows for heads: {', '.join(heads)}")

    if args.features_csv is None:
        print("No --features_csv provided; params file parsed successfully.")
        return

    feat_df = pd.read_csv(args.features_csv)
    for head in heads:
        head_params = params[params["head"] == head]
        scores = score_enet_head_from_feature_df(feat_df, head_params)
        print(
            f"{head}: n={len(scores)} min={scores.min():.6f} "
            f"median={pd.Series(scores).median():.6f} max={scores.max():.6f}"
        )


if __name__ == "__main__":
    main()
