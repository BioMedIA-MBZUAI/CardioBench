# Datasets Download Guide

This repository does not redistribute data. The table below summarises where each dataset can be obtained and what credentials you need. Once you have the raw data, point the workflow configuration (`configs/datasets.json`) at the downloaded location.

| Dataset | Source | Split |
| --- | --- | --- |
| EchoNet-Dynamic | [https://echonet.github.io/dynamic/](https://echonet.github.io/dynamic/) | Official patient split |
| EchoNet-Pediatric | [https://echonet.github.io/pediatric/](https://echonet.github.io/pediatric/) | Official train/val/test split |
| EchoNet-LVH | [https://echonet.github.io/lvh/](https://echonet.github.io/lvh/) | Official split: 0 fold - test, 1...9 - training |
| CAMUS | [https://www.creatis.insa-lyon.fr/Challenge/camus/](https://www.creatis.insa-lyon.fr/Challenge/camus/) | Official train/val/test split |
| CardiacNet ASD/PAH | [https://www.kaggle.com/datasets/xiaoweixumedicalai/abnormcardiacechovideos](https://www.kaggle.com/datasets/xiaoweixumedicalai/abnormcardiacechovideos) | Custom, provided in data/splits/cardiacnet/ |
| HMC-QU | [https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset/data](https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset/data) | Custom, provided in data/splits/hmcqu_split |
| TMED-2 | [https://tmed.cs.tufts.edu/tmed_v2.html](https://tmed.cs.tufts.edu/tmed_v2.html) | Official DEV479, 0 fold split, provided in data/splits/tmed2/ |
| SegRWMA | [https://www.kaggle.com/datasets/xiaoweixumedicalai/regional-wall-motion-abnormality-echo](https://www.kaggle.com/datasets/xiaoweixumedicalai/regional-wall-motion-abnormality-echo) | Custom, provided in data/splits/ |

Once downloaded, organise the raw files under `data/raw/`.

Update `configs/datasets.json` to reflect the actual paths. The workflow CLI can then generate the benchmark splits:

```bash
python -m cardiobench.workflow.cli split --list
python -m cardiobench.workflow.cli split
```

Generated splits are written to `data/splits/` by default.