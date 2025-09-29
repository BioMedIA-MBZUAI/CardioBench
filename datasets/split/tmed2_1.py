import os
import pandas as pd
from config import TMED_2_gt, TMED_2_SPLIT_CSV, TMED2_ROOT, TMED_2_OUT_CSV

df = pd.read_csv(TMED_2_gt)
df = df[df["diagnosis_label"] != "Not_Provided"].copy()

df["diagnosis_label"] = df["diagnosis_label"].apply(lambda x: 0 if x == "no_AS" else 1)

split_df = pd.read_csv(TMED_2_SPLIT_CSV)

split_df["file_path_from_split"] = split_df.apply(
    lambda r: os.path.join(
        TMED2_ROOT, str(r["SourceFolder"]).strip("/"), r["query_key"]
    ),
    axis=1,
)

merged = df.merge(
    split_df[["query_key", "diagnosis_classifier_split", "file_path_from_split"]],
    how="left",
    on="query_key",
    validate="m:1",
)

merged["file_path"] = merged["file_path_from_split"]
merged = merged.rename(columns={"diagnosis_classifier_split": "split"})

merged["exists"] = merged["file_path"].apply(
    lambda p: isinstance(p, str) and os.path.exists(p)
)
missing_files = (~merged["exists"]).sum()
if missing_files:
    print(f"Warning: {missing_files} files listed are missing on disk; dropping them.")

merged = merged[merged["exists"]].copy()
merged.drop(columns=["file_path_from_split", "exists"], inplace=True)

total = len(merged)
print(f"Total usable rows: {total}")
print(f"Matched official split")
print("\nCounts by split:")

print("\nDiagnosis distribution per split:")
for s in merged["split"].unique():
    dist = (
        merged.loc[merged["split"] == s, "diagnosis_binary"]
        .value_counts(normalize=True)
        .sort_index()
        .round(3)
    )
    print(f"\n{s}:\n{dist}")

merged.to_csv(TMED_2_OUT_CSV, index=False)
