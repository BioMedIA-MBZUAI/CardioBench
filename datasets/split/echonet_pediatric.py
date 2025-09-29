import pandas as pd
from config import PEDIATRIC_A4C, PEDIATRIC_PSAX, OUT_PEDIATRIC_FILELIST_CSV

a_4c = pd.read_csv(PEDIATRIC_A4C)
psax = pd.read_csv(PEDIATRIC_PSAX)

a_4c["view"] = "A4C"
psax["view"] = "PSAX"
combined = pd.concat([a_4c, psax])

combined.to_csv(OUT_PEDIATRIC_FILELIST_CSV, index=False)
