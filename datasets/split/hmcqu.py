import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import A2C_gt, A4C_gt, A2C_data_folder, A4C_data_folder, HMCQU_OUT_DIR


def counts(df, name):
    by_class = df.groupby(["split", "STEMI"]).size().rename("n").reset_index()
    by_pat = df.drop_duplicates(["patient_id", "split"]).groupby(["split"]).size()
    print(f"\n=== {name} ===")
    print("Rows by split & STEMI:\n", by_class)
    print("Unique patients by split:\n", by_pat)


os.makedirs(HMCQU_OUT_DIR, exist_ok=True)

a2c = pd.read_csv(A2C_gt)
a4c = pd.read_csv(A4C_gt)

a2c_videos = A2C_data_folder
a4c_videos = A4C_data_folder

# A2C
a2c["view"] = "A2C"
a2c["patient_id"] = a2c["ECHO"].str.extract(r"(ES\d+)")
a2c["STEMI"] = (a2c.filter(like="SEG").eq("MI").any(axis=1)).astype(int)
a2c["path"] = a2c["ECHO"].apply(lambda x: os.path.join(a2c_videos, f"{x}.avi"))

# A4C
a4c["view"] = "A4C"
a4c["patient_id"] = a4c["ECHO"].str.extract(r"(ES\d+)")
a4c["STEMI"] = (a4c.filter(like="SEG").eq("MI").any(axis=1)).astype(int)
a4c["path"] = a4c["ECHO"].apply(lambda x: os.path.join(a4c_videos, f"{x}.avi"))
a2c = a2c[["path", "view", "patient_id", "STEMI"]]
a4c = a4c[["path", "view", "patient_id", "STEMI"]]

final = pd.concat([a2c, a4c], ignore_index=True)


patient_labels = (
    final.groupby("patient_id")["STEMI"].max().rename("STEMI_patient").reset_index()
)

trainval_patients, test_patients = train_test_split(
    patient_labels,
    test_size=0.15,
    stratify=patient_labels["STEMI_patient"],
    random_state=42,
)

val_ratio_of_trainval = 0.15 / 0.90
train_patients, val_patients = train_test_split(
    trainval_patients,
    test_size=val_ratio_of_trainval,
    stratify=trainval_patients["STEMI_patient"],
    random_state=42,
)

split_map = {}
for pid in train_patients["patient_id"]:
    split_map[pid] = "train"
for pid in val_patients["patient_id"]:
    split_map[pid] = "val"
for pid in test_patients["patient_id"]:
    split_map[pid] = "test"

final["split"] = final["patient_id"].map(split_map)

missing = final[~final["path"].apply(os.path.exists)]


final_out = os.path.join(HMCQU_OUT_DIR, "split.csv")
final.to_csv(final_out, index=False)

counts(final, "Overall")
