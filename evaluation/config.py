# GENERAL
B = 1000
SEED = 42
SPLIT = "test"
VIEW_CLASS_NAMES = ["A2C", "A3C", "A4C", "PSAX", "PLAX", "Other"]

# CAMUS
CAMUS_SPLIT_CSV = "data/splits/camus_split.csv"
CAMUS_EF_PRED_DIR = "evaluation/example_predictions/CAMUS/EF"
CAMUS_VIEW_PRED_DIR = "evaluation/example_predictions/CAMUS/view"
CAMUS_OUTPUT_DIR = "evaluation/output/CAMUS/"
VIEW_MAP = {
    "2CH": "A2C",
    "4CH": "A4C",
}

# CARDIACNET
ASD_GT = "data/splits/cardiacnet/cardiacnet_asd_split.csv"
PAH_GT = "data/splits/cardiacnet/cardiacnet_pah_split.csv"
PRED_ROOT = "evaluation/example_predictions/CardiacNet"
CARDIACNET_OUT_DIR = "evaluation/output/CardiacNet/"

# ECHONET DYNAMIC
GT_ECHONET_DYNAMIC = "data/splits/EchoNet-Dynamic/FileList.csv"

EF_MODELS = {
    "modelname": "evaluation/example_predictions/EchoNet-Dynamic/EF/modelname.csv",  # path to modelname EF predictions CSV
}

VIEW_MODELS = {
    "modelname": "evaluation/example_predictions/EchoNet-Dynamic/view/modelname.csv",  # path to modelname view predictions CSV
}

ECHONETDYNAMIC_OUT_DIR = "evaluation/output/EchoNet-Dynamic/"

# HMC-QU
HMCQU_CSV = "data/splits/hmcqu_split.csv"
STEMI_PRED_DIR = "evaluation/example_predictions/HMCQU/stemi"
VIEW_PRED_DIR = "evaluation/example_predictions/HMCQU/view"
HMCQU_OUT_DIR = "evaluation/output/HMCQU/"

# ECHONET LVH
LVH_GT_FILES = {
    "IVSd": "data/splits/echonet_lvh/IVSd_split.csv",
    "LVIDd": "data/splits/echonet_lvh/LVIDd_split.csv",
    "LVPWd": "data/splits/echonet_lvh/LVPWd_split.csv",
}

IVSd_ROOT = "evaluation/example_predictions/EchoNet-LVH/IVSd"
LVIDd_ROOT = "evaluation/example_predictions/EchoNet-LVH/LVIDd"
LVPWd_ROOT = "evaluation/example_predictions/EchoNet-LVH/LVPWd"

LVH_VIEW_ROOT = "evaluation/example_predictions/EchoNet-LVH/view"

LVH_OUT_DIR = "evaluation/output/EchoNet-LVH/"

# PEDIATRIC
PEDIATRIC_FILELIST_CSV = "data/splits/echonet_pediatric_filelist.csv"

PEDIATRIC_EF_PRED_DIR = "evaluation/example_predictions/EchoNet-Pediatric/EF"

PEDIATRIC_VIEW_MODELS = {
    "modelname": "evaluation/example_predictions/EchoNet-Pediatric/view/modelname.csv"
}

PEDIATRIC_OUT_DIR = "evaluation/output/EchoNet-Pediatric/"

# RWMA
RWMA_GT = "data/splits/segrwma_split.csv"
RWMA_PRED_DIR = "evaluation/example_predictions/segRWMA/rwma"
RWMA_VIEW_PRED_DIR = "evaluation/example_predictions/segRWMA/view"
RWMA_OUT_DIR = "evaluation/output/segRWMA/"
EVAL_MODALITIES = ["2D"]

# TMED2
TMED2_SPLIT_PER_IMAGE = "data/splits/tmed2/per_image.csv"
TMED2_SPLIT_PER_STUDY = "data/splits/tmed2/per_study.csv"
AS_PRED_DIR = "evaluation/example_predictions/TMED2/AS"
TMED2_VIEW_PRED_DIR = "evaluation/example_predictions/TMED2/view"
TMED2_OUT_DIR = "evaluation/output/TMED2/"
