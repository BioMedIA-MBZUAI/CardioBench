#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: pipeline.sh --dataset NAME --root PATH --out-root PATH [options]

Required arguments:
  --dataset NAME            Dataset key registered with Pipeline
  --root PATH               Dataset root directory
  --out-root PATH           Directory to write embeddings/results

Common options:
  --split-csv PATH          Optional CSV controlling dataset splits
  --split NAME              Single split to embed (default: use --splits)
  --splits LIST             Comma-separated list of splits to embed (default: train, val, test)
  --train-split NAME        Training split name (default: train)
  --test-split NAME         Test split name (default: test)
  --view NAME               Optional view filter
  --modality NAME           Optional modality filter
  --fold INT                Optional fold selector for specific datasets
  --model NAME              Visual model alias/id for embeddings (default: echo_clip)
  --model-id ID             Explicit visual model id overriding --model
  --text-model NAME         Text model alias/id for zero-shot stages (default: value of --model)
  --device NAME             Torch device (default: cuda)
  --precision NAME          Precision (default: bf16)
  --res INT                 Input resolution (default: 224)
  --max-frames INT          Max frames per clip (default: 10)
  --stride INT              Frame stride (default: 1)
  --batch-size INT          Embedding batch size (default: 16)
  --key-frame INT           Default key frame (default: 0)
  --num-workers INT         DataLoader workers for probe stages (default: 8)
  --normalize/--no-normalize    Toggle embedding normalization (default: on)
  --overwrite               Overwrite existing embeddings/index.csv
  --no-channels-last        Disable channels_last when encoding images
  --no-pin-memory           Disable pin_memory during encoding
  --skip-embeddings         Skip embedding generation
  --skip-view               Skip zero-shot view classification
  --skip-binary             Skip zero-shot binary classification
  --skip-regression         Skip zero-shot regression (EF/LVH)
  --run-regression          Force zero-shot regression stage
  --run-probe               Train + evaluate linear probe on CSV labels

Binary classification options:
  --binary-mode MODE        threshold|argmax (default: threshold)
  --binary-pos-key KEY      Prompt key for positive class
  --binary-neg-key KEY      Prompt key for negative class (argmax mode)
  --binary-pos-prompts LIST Comma-separated custom positive prompts
  --binary-neg-prompts LIST Comma-separated custom negative prompts
  --binary-threshold FLOAT  Explicit threshold for threshold mode
  --binary-calib-csv PATH   CSV with labels for calibration (threshold mode)
  --binary-csv NAME         Output CSV name (default: binary_predictions.csv)
  --binary-pos-name NAME    Positive class label (default: positive)
  --binary-neg-name NAME    Negative class label (default: negative)

Regression options:
  --regression-tasks LIST   Comma-separated subset of ef,lvh (default: ef,lvh)
  --ef-csv NAME             Output CSV for EF regression (default: ef_predictions.csv)
  --lvh-csv NAME            Output CSV for LVH regression (default: lvh_predictions.csv)

Linear probe options:
  --probe-task TASK         regression|classification (default: classification)
  --probe-num-classes INT   Number of classes for classification probes (default: 1)
  --probe-max-frames INT    Frames per sample for probes (default: 1)
  --probe-lr FLOAT          Probe learning rate (default: 1e-4)
  --probe-devices SPEC      Lightning devices argument (default: 1)
  --probe-out-dir PATH      Directory for probe runs (default: runs)
  --probe-exp-name NAME     Experiment name for probe (default: probe_run)

Misc:
  --help                    Show this message and exit
USAGE
}

PYTHON_BIN=${PYTHON:-python}

# defaults
DATASET="${DEFAULT_DATASET:-}"
ROOT="${DEFAULT_ROOT:-}"
OUT_ROOT="${DEFAULT_OUT_ROOT:-}"
SPLIT="${DEFAULT_SPLIT:-}"
SPLIT_LIST="${DEFAULT_SPLITS:-train,val,test}"
SPLIT_CSV=""
TRAIN_SPLIT="train"
TEST_SPLIT="test"
VIEW="${DEFAULT_VIEW:-}"
MODALITY="${DEFAULT_MODALITY:-}"
FOLD=""
MODEL="echo_clip"
MODEL_ID=""
TEXT_MODEL=""
DEVICE="${DEFAULT_DEVICE:-cuda}"
PRECISION="${DEFAULT_PRECISION:-bf16}"
RES=224
MAX_FRAMES=16
STRIDE=1
BATCH_SIZE=64
NUM_WORKERS=8
KEY_FRAME=0
NORMALIZE=1
OVERWRITE=0
USE_CHANNELS_LAST=1
PIN_MEMORY=1
RUN_EMBED=1
RUN_VIEW=1
RUN_BINARY=1
RUN_REGRESSION=0
RUN_PROBE=0
REGRESSION_TASKS="ef,lvh"
REGRESSION_TASKS_FROM_CLI=0
VIEW_CSV="view_predictions.csv"
BINARY_MODE="argmax"
BINARY_POS_KEY=""
BINARY_NEG_KEY=""
BINARY_POS_PROMPTS=""
BINARY_NEG_PROMPTS=""
BINARY_THRESHOLD=""
BINARY_CALIB_CSV=""
BINARY_CSV="binary_predictions.csv"
BINARY_POS_NAME="positive"
BINARY_NEG_NAME="negative"
EF_CSV="ef_predictions.csv"
LVH_CSV="lvh_predictions.csv"
PROBE_TASK="classification"
PROBE_NUM_CLASSES=1
PROBE_MAX_FRAMES=1
PROBE_LR="1e-4"
PROBE_DEVICES="1"
PROBE_OUT_DIR="runs"
PROBE_EXP_NAME="probe_run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET=$2; shift; shift;;
    --root) ROOT=$2; shift; shift;;
    --out-root) OUT_ROOT=$2; shift; shift;;
    --split-csv) SPLIT_CSV=$2; shift; shift;;
    --split) SPLIT=$2; shift; shift;;
    --splits) SPLIT_LIST=$2; shift; shift;;
    --train-split) TRAIN_SPLIT=$2; shift; shift;;
    --test-split) TEST_SPLIT=$2; shift; shift;;
    --view) VIEW=$2; shift; shift;;
    --modality) MODALITY=$2; shift; shift;;
    --fold) FOLD=$2; shift; shift;;
    --model) MODEL=$2; shift; shift;;
    --model-id) MODEL_ID=$2; shift; shift;;
    --text-model) TEXT_MODEL=$2; shift; shift;;
    --device) DEVICE=$2; shift; shift;;
    --precision) PRECISION=$2; shift; shift;;
    --res) RES=$2; shift; shift;;
    --max-frames) MAX_FRAMES=$2; shift; shift;;
    --stride) STRIDE=$2; shift; shift;;
    --batch-size) BATCH_SIZE=$2; shift; shift;;
    --num-workers) NUM_WORKERS=$2; shift; shift;;
    --key-frame) KEY_FRAME=$2; shift; shift;;
    --normalize) NORMALIZE=1; shift;;
    --no-normalize) NORMALIZE=0; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --no-channels-last) USE_CHANNELS_LAST=0; shift;;
    --no-pin-memory) PIN_MEMORY=0; shift;;
    --skip-embeddings) RUN_EMBED=0; shift;;
    --skip-view) RUN_VIEW=0; shift;;
    --skip-binary) RUN_BINARY=0; shift;;
    --skip-regression) RUN_REGRESSION=0; shift;;
    --run-regression) RUN_REGRESSION=1; shift;;
    --run-probe) RUN_PROBE=1; shift;;
    --view-csv) VIEW_CSV=$2; shift; shift;;
    --binary-mode) BINARY_MODE=$2; shift; shift;;
    --binary-pos-key) BINARY_POS_KEY=$2; shift; shift;;
    --binary-neg-key) BINARY_NEG_KEY=$2; shift; shift;;
    --binary-pos-prompts) BINARY_POS_PROMPTS=$2; shift; shift;;
    --binary-neg-prompts) BINARY_NEG_PROMPTS=$2; shift; shift;;
    --binary-threshold) BINARY_THRESHOLD=$2; shift; shift;;
    --binary-calib-csv) BINARY_CALIB_CSV=$2; shift; shift;;
    --binary-csv) BINARY_CSV=$2; shift; shift;;
    --binary-pos-name) BINARY_POS_NAME=$2; shift; shift;;
    --binary-neg-name) BINARY_NEG_NAME=$2; shift; shift;;
    --regression-tasks) REGRESSION_TASKS=$2; REGRESSION_TASKS_FROM_CLI=1; RUN_REGRESSION=1; shift; shift;;
    --ef-csv) EF_CSV=$2; shift; shift;;
    --lvh-csv) LVH_CSV=$2; shift; shift;;
    --probe-task) PROBE_TASK=$2; shift; shift;;
    --probe-num-classes) PROBE_NUM_CLASSES=$2; shift; shift;;
    --probe-max-frames) PROBE_MAX_FRAMES=$2; shift; shift;;
    --probe-lr) PROBE_LR=$2; shift; shift;;
    --probe-devices) PROBE_DEVICES=$2; shift; shift;;
    --probe-out-dir) PROBE_OUT_DIR=$2; shift; shift;;
    --probe-exp-name) PROBE_EXP_NAME=$2; shift; shift;;
    --help|-h) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 1;;
  esac
done

if [[ -z "$DATASET" ]] || [[ -z "$ROOT" ]] || [[ -z "$OUT_ROOT" ]]; then
  echo "Error: --dataset, --root, and --out-root are required." >&2
  usage
  exit 1
fi

TEXT_MODEL=${TEXT_MODEL:-$MODEL}

IFS=',' read -r -a SPLIT_ARRAY <<< "${SPLIT_LIST}" || true
if [[ -n "$SPLIT" ]]; then
  SPLIT_ARRAY=($SPLIT)
fi
if [[ ${#SPLIT_ARRAY[@]} -eq 0 ]]; then
  SPLIT_ARRAY=($TRAIN_SPLIT $TEST_SPLIT)
fi

POS_PROMPT_LIST=()
if [[ -n "$BINARY_POS_PROMPTS" ]]; then
  IFS=',' read -r -a POS_PROMPT_LIST <<< "$BINARY_POS_PROMPTS"
fi
NEG_PROMPT_LIST=()
if [[ -n "$BINARY_NEG_PROMPTS" ]]; then
  IFS=',' read -r -a NEG_PROMPT_LIST <<< "$BINARY_NEG_PROMPTS"
fi

IFS=',' read -r -a REG_TASK_LIST <<< "$REGRESSION_TASKS"

if (( RUN_BINARY )) && (( REGRESSION_TASKS_FROM_CLI )) && [[ -z "$BINARY_POS_KEY" ]] && (( ${#POS_PROMPT_LIST[@]} == 0 )); then
  echo "Skipping zero-shot binary classification (regression-only run)"
  RUN_BINARY=0
fi

log() { printf '[pipeline] %s\n' "$*"; }

get_prompt_keys() {
  if [[ -z "${PROMPT_KEYS_CACHE:-}" ]]; then
    PROMPT_KEYS_CACHE=$("$PYTHON_BIN" - <<'PY'
from src import prompts

def collect(keys_dict):
    if isinstance(keys_dict, dict):
        return list(keys_dict.keys())
    return []

keys = set()
for mapping_name in ("zero_shot_prompts", "stemi_prompts", "additional_prompts"):
    mapping = getattr(prompts, mapping_name, {})
    keys.update(collect(mapping))
if not keys:
    print("(no predefined prompt keys)")
else:
    print(", ".join(sorted(str(k) for k in keys)))
PY
    )
  fi
  printf '%s' "$PROMPT_KEYS_CACHE"
}

mkdir -p "$OUT_ROOT"

if (( RUN_EMBED )); then
  for split_name in "${SPLIT_ARRAY[@]}"; do
    target_dir="$OUT_ROOT/$split_name"
    mkdir -p "$target_dir"
    cmd=("$PYTHON_BIN" -m src.embeddings
         --dataset "$DATASET"
         --root "$ROOT"
         --out_dir "$target_dir"
         --device "$DEVICE"
         --precision "$PRECISION"
         --res "$RES"
         --max_frames "$MAX_FRAMES"
         --stride "$STRIDE"
         --batch_size "$BATCH_SIZE"
         --key_frame "$KEY_FRAME"
         --model "$MODEL")
    if [[ -n "$MODEL_ID" ]]; then
      cmd+=(--model_id "$MODEL_ID")
    fi
    if (( !USE_CHANNELS_LAST )); then
      cmd+=(--no_channels_last)
    fi
    if (( !PIN_MEMORY )); then
      cmd+=(--no_pin_memory)
    fi
    if (( OVERWRITE )); then
      cmd+=(--overwrite)
    fi
    if (( !NORMALIZE )); then
      cmd+=(--no_normalize)
    fi
    if [[ -n "$SPLIT_CSV" ]]; then
      cmd+=(--split_csv "$SPLIT_CSV")
    fi
    if [[ -n "$split_name" ]]; then
      cmd+=(--split "$split_name")
    fi
    if [[ -n "$VIEW" ]]; then
      cmd+=(--view "$VIEW")
    fi
    if [[ -n "$MODALITY" ]]; then
      cmd+=(--modality "$MODALITY")
    fi
    if [[ -n "$FOLD" ]]; then
      cmd+=(--fold "$FOLD")
    fi
    log "Generating embeddings for split '$split_name'"
    log "Command: ${cmd[*]}"
    "${cmd[@]}"
  done
fi

TEST_DIR="$OUT_ROOT/$TEST_SPLIT"
TRAIN_DIR="$OUT_ROOT/$TRAIN_SPLIT"

if (( RUN_VIEW )); then
  log "Running zero-shot view classification on $TEST_DIR"
  view_cmd=("$PYTHON_BIN" -m src.classification.view
            --emb_dir "$TEST_DIR"
            --csv_name "$VIEW_CSV"
            --model_id "$TEXT_MODEL"
            --device "$DEVICE"
            --precision "$PRECISION")
  log "Command: ${view_cmd[*]}"
  "${view_cmd[@]}"
fi

if (( RUN_BINARY )); then
  log "Running zero-shot binary classification ($BINARY_MODE)"
  binary_cmd=("$PYTHON_BIN" -m src.classification.binary
              --emb_dir "$TEST_DIR"
              --csv_name "$BINARY_CSV"
              --mode "$BINARY_MODE"
              --model_id "$TEXT_MODEL"
              --device "$DEVICE"
              --precision "$PRECISION"
              --pos_name "$BINARY_POS_NAME"
              --neg_name "$BINARY_NEG_NAME")
  if [[ -n "$BINARY_POS_KEY" ]]; then
    binary_cmd+=(--pos_key "$BINARY_POS_KEY")
  fi
  for prompt in "${POS_PROMPT_LIST[@]}"; do
    binary_cmd+=(--pos_prompt "$prompt")
  done
  if [[ "$BINARY_MODE" == "argmax" ]]; then
    if [[ -z "$BINARY_NEG_KEY" && ${#NEG_PROMPT_LIST[@]} -eq 0 ]]; then
      echo "Error: argmax mode requires --binary-neg-key or --binary-neg-prompts" >&2
      echo "Available prompt keys: $(get_prompt_keys)" >&2
      exit 1
    fi
    if [[ -n "$BINARY_NEG_KEY" ]]; then
      binary_cmd+=(--neg_key "$BINARY_NEG_KEY")
    fi
    for prompt in "${NEG_PROMPT_LIST[@]}"; do
      binary_cmd+=(--neg_prompt "$prompt")
    done
  else
    if [[ -n "$BINARY_THRESHOLD" ]]; then
      binary_cmd+=(--threshold "$BINARY_THRESHOLD")
    else
      if [[ -z "$BINARY_CALIB_CSV" ]]; then
        echo "Error: threshold mode without --binary-threshold requires --binary-calib-csv" >&2
        exit 1
      fi
      binary_cmd+=(
        --calib_dir "$TRAIN_DIR" --calib_csv "$BINARY_CALIB_CSV"
        --labels_id_col "unique_id"
        --labels_split_col "split" --labels_split "train"
        --map_scores_from_col "path" --map_scores_to_col "unique_id"
        --map_output_from_col "path" --map_output_to_col "unique_id"
      )
    fi
  fi
  log "Command: ${binary_cmd[*]}"
  "${binary_cmd[@]}"
fi

if (( RUN_REGRESSION )); then
  for task in "${REG_TASK_LIST[@]}"; do
    case "$task" in
      ef)
        log "Running EF regression"
        ef_cmd=("$PYTHON_BIN" -m src.regression ef
                --emb_dir "$TEST_DIR"
                --device "$DEVICE"
                --precision "$PRECISION"
                --model "$TEXT_MODEL"
                --csv_name "$EF_CSV")
        log "Command: ${ef_cmd[*]}"
        "${ef_cmd[@]}"
        ;;
      lvh)
        log "Running LVH regression"
        lvh_cmd=("$PYTHON_BIN" -m src.regression lvh
                 --emb_dir "$TEST_DIR"
                 --device "$DEVICE"
                 --precision "$PRECISION"
                 --model "$TEXT_MODEL"
                 --csv_name "$LVH_CSV")
        log "Command: ${lvh_cmd[*]}"
        "${lvh_cmd[@]}"
        ;;
      "") ;;
      *)
        echo "Warning: unknown regression task '$task' ignored" >&2
        ;;
    esac
  done
fi

if (( RUN_PROBE )); then
  if [[ -z "$SPLIT_CSV" ]]; then
    echo "Error: linear probe stage requires --split-csv" >&2
    exit 1
  fi
  log "Training linear probe"
train_cmd=("$PYTHON_BIN" -m src.linear_probe.train
             --csv "$SPLIT_CSV"
             --root "$ROOT"
             --task "$PROBE_TASK"
             --num_classes "$PROBE_NUM_CLASSES"
             --model "$MODEL"
             --precision "$PRECISION"
             --res "$RES"
             --max_frames "$PROBE_MAX_FRAMES"
             --batch_size "$BATCH_SIZE"
             --num_workers "$NUM_WORKERS"
             --lr "$PROBE_LR"
             --devices "$PROBE_DEVICES"
             --out_dir "$PROBE_OUT_DIR"
             --exp_name "$PROBE_EXP_NAME")
  if [[ -n "$MODEL_ID" ]]; then
    train_cmd+=(--model_id "$MODEL_ID")
  fi
  if [[ -n "$VIEW" ]]; then
    train_cmd+=(--view "$VIEW")
  fi
  if [[ -n "$MODALITY" ]]; then
    train_cmd+=(--modality "$MODALITY")
  fi
  log "Command: ${train_cmd[*]}"
  "${train_cmd[@]}"

  ckpt_dir="$PROBE_OUT_DIR/$PROBE_EXP_NAME"
  ckpt_file=$(ls -t "$ckpt_dir"/best-*.ckpt 2>/dev/null | head -n1 || true)
  if [[ -z "$ckpt_file" ]]; then
    echo "Error: no checkpoint found in $ckpt_dir" >&2
    exit 1
  fi

  log "Predicting with linear probe ($ckpt_file)"
  predict_cmd=("$PYTHON_BIN" -m src.linear_probe.predict
               --csv "$SPLIT_CSV"
               --root "$ROOT"
               --ckpt "$ckpt_file"
               --model "$MODEL"
               --precision "$PRECISION"
               --res "$RES"
               --max_frames "$PROBE_MAX_FRAMES"
               --stride "$STRIDE"
               --batch_size "$BATCH_SIZE"
               --num_workers "$NUM_WORKERS")
  if [[ -n "$MODEL_ID" ]]; then
    predict_cmd+=(--model_id "$MODEL_ID")
  fi
  if [[ -n "$VIEW" ]]; then
    predict_cmd+=(--view "$VIEW")
  fi
  if [[ -n "$MODALITY" ]]; then
    predict_cmd+=(--modality "$MODALITY")
  fi
  log "Command: ${predict_cmd[*]}"
  "${predict_cmd[@]}"

  predictions_csv="$(dirname "$ckpt_file")/predictions_test.csv"
  if [[ -f "$predictions_csv" ]]; then
    dest="$TEST_DIR/${DATASET}_probe_predictions.csv"
    cp "$predictions_csv" "$dest"
    log "Probe predictions copied to $dest"
  fi
fi

log "Pipeline complete"
