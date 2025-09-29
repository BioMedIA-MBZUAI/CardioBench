import pandas as pd
from config import TMED_2_OUT_CSV

df = pd.read_csv(TMED_2_OUT_CSV)

df["study_id"] = df["query_key"].str.replace(r"_\d+\.png$", "", regex=True)

study_df = df.groupby("study_id", as_index=False).agg(
    {"diagnosis_label": "first", "split": "first"}
)

study_df.to_csv(
    "/data/ahmedaly/public/TMED_2/stenosis_split_study_level.csv", index=False
)

print(study_df.head())
