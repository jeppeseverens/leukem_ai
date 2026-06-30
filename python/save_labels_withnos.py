"""Generate the NOS-inclusive label mapping used by the with-NOS inner-CV track.

Mirrors save_labels.ipynb but keeps "AML NOS" as a class (include_nos=True), so
the encoded label set matches run_inner_cv_array.py --include_nos. Writes
data/label_mapping_withnos.csv in the same format as label_mapping_all.csv.
"""
import os
from pathlib import Path

import pandas as pd

import train_test

data_dir = Path(__file__).resolve().parent.parent / "data"
X, y, study_labels = train_test.load_data(data_dir)
X, y, study_labels = train_test.filter_data(X, y, study_labels, min_n=10, include_nos=True)
_, label_mapping = train_test.encode_labels(y)

label_df = pd.DataFrame([
    {"Label": label, "Encoded": encoded}
    for label, encoded in sorted(label_mapping.items(), key=lambda x: x[1])
])
out_path = data_dir / "label_mapping_withnos.csv"
label_df.to_csv(out_path, index=True)
print(f"Wrote {out_path} with {len(label_df)} classes (incl. AML NOS).")
print(label_df.to_string())
