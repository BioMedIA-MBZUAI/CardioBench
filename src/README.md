# Usage Guide

This folder contains code for CardioBench zero-shot pipeline linear probes. Example commands are shown below to run the experiments.

## Prerequisites
- **Environment**: activate the `cardiobench` conda env. To create it:
  ```bash
  conda create -n cardiobench python=3.11
  conda activate cardiobench
  pip install -r src/requirements.txt
  ```
- **Datasets**: download the datasets and note their roots. Valid dataset names live in `src/datasets.py` (`DEFAULT_DATASET_REGISTRY`).
- **Models**: model aliases are defined in `src/models.py` (`MODEL_ALIASES`). Defaults to `echo_clip`.
- **Prompts**: prompt key names come from `src/prompts.py`. Binary tasks require one positive key and one negative key.

## What the pipeline does
Running `bash src/pipeline.sh ...` executes the following:
1. **Embeddings** – `src.embeddings`: reads videos and writes per-video `.pt` embedding files to `--out-root/<split>`.
2. **View classification** – `src.classification.view`: predicts views for each embedding (skippable via `--skip-view`).
3. **Zero-shot task** – `src.classification.binary` for classification and `src.regression` for EF/LVH. Binary stage runs unless you pass `--skip-binary`; regression only runs when `--run-regression` or `--regression-tasks ...` is supplied.
4. **Linear probe** – optional fine-tuning stage enabled via `--run-probe` (see `src/linear_probe/*`).

A "Quick configuration" block near the top of `src/pipeline.sh` lets you prefill common defaults (`DEFAULT_DATASET`, `DEFAULT_ROOT`, etc.). CLI flags always override those values.

## Example commands
### Binary classification
Generic template (replace placeholders with the values below):
```bash
bash src/pipeline.sh \
  --dataset <dataset> \
  --root <dataset_root> \
  --model <model_alias> \
  --out-root embeddings/<model_alias>/<dataset>/ \
  --binary-pos-key <positive_prompt_key> \
  --binary-neg-key <negative_prompt_key>
```

Note:
- `--view` to restrict datasets with multiple views (e.g., `--view A4C`).
- Enable the linear-probe stage with `--run-probe`
- You can use `--split-csv` to specify a csv file that directly contains paths 


### Regression
Regression tasks are run by adding the flag `--run-regression` and choosing the task `--regression-tasks`. For example:

```bash
bash src/pipeline.sh \
  --dataset <dataset> \
  --root <dataset_root> \
  --model <model_alias> \
  --out-root embeddings/<model_alias>/<dataset>/ \
  --binary-pos-key <positive_prompt_key> \
  --binary-neg-key <negative_prompt_key> \
  --run-regression \
  --regression-tasks ef
  ```


