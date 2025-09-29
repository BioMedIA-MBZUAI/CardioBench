from pathlib import Path

# CAMUS
CAMUS_ROOT = ""  # path to the CAMUS root directory, e.g. /CAMUS_public/database_nifti
CAMUS_SPLIT_DIR = ""  # path to the CAMUS split directory, e.g. /data/ahmedaly/public/CAMUS_public/database_split
CAMUS_OUT_CSV = ""  # path to save the CAMUS split CSV file

# CARDIACNET
ASD_ROOT = Path(
    ""
)  # path to the CardiacNet-ASD directory, e.g. ./CardiacNet/CardiacNet-ASD
ASD_OUT_DIR = Path("./CardiacNet")

PAH_ROOT = Path("")  # e.g. ./CardiacNet/CardiacNet-PAH
PAH_OUT_DIR = Path("")  # e.g. ./CardiacNet

# ECHONET PEDIATRIC
PEDIATRIC_A4C = ""  # e.g. ./A4C/FileList.csv
PEDIATRIC_PSAX = ""  # e.g. ./PSAX/FileList.csv
OUT_PEDIATRIC_FILELIST_CSV = ""  # e.g. ./FileList.csv

# ECHONET LVH
LVH_MEASUREMENT_LIST = ""  # e.g. ./EchoNet-LVH/MeasurementsList.csv
LVH_VIDEOS_DIR = Path()  # e.g. ./EchoNet-LVH/videos

# HMC-QU
A2C_gt = ""  # e.g. HMC_QU/A2C.csv
A4C_gt = ""  # e.g. ./HMC_QU/A4C.cs
A2C_data_folder = ""  # e.g. ./HMC_QU/videos/A2C
A4C_data_folder = ""  # e.g. HMC_QU/videos/A4C
HMCQU_OUT_DIR = ""  # e.g. HMC_QU

# TMED-2
TMED_2_gt = ""  # e.g. labels_per_image.csv
TMED2_ROOT = ""  # e.g. ./TMED_2
TMED_2_OUT_CSV = ""  # e.g. tmed2_split.csv
