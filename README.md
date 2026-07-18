# Anonymous Code Submission

This repository contains the implementation and evaluation scripts accompanying
an anonymous paper submission. It intentionally excludes author information,
private paths, generated results, datasets, and trained model checkpoints.

## Contents

- `src/PartialDiffGED/`: training and evaluation implementation for the dense
  and disjoint model variants.
- `src/PartialDiffGED_main_table_snapshot/`: the implementation used for the
  reported dense inference and refinement experiments.
- `scripts/Overall_Performance/`: per-dataset commands for the main-table
  evaluation protocol.
- `third_party/App-BMao-tlimited_main_table/`: source for the t-limited
  App-BMao baseline used by the evaluation scripts.
- `environment.yml`: reproducible Conda environment specification.

## Environment

```bash
conda env create -f environment.yml
conda activate partialged
bash scripts/build_ittnotify_stub.sh   # only if the optional runtime stub is needed
bash scripts/test_env.sh
```

The experiments require a CUDA-capable GPU. The supplied environment targets
Python 3.10, PyTorch 2.3.1, and CUDA 11.8.

## External artifacts

Datasets, fixed evaluation pairs, and checkpoints are intentionally not
distributed in this anonymous repository. Place them under `artifacts/` using
the layout checked by:

```bash
python scripts/Overall_Performance/check_required_artifacts.py
```

The required layout is:

```text
artifacts/
  Overall_Performance/
    direct_data/json_data/{ogbg-molhiv,ogbg-molpcba,ogbg-code2-ast97-ref,IMDB}/
    fixed_pairs/
    app_bmao_per_dataset_wall/k100/
    graph_sources/
  model_save/partialdiff/
```

For training, set `DATA_ROOT` to a directory with this separate layout:

```text
<DATA_ROOT>/json_data/<DATASET>/{train,test}/
```

## Training

The unified training script supports the dense and disjoint variants, as well
as checkpoint evaluation:

```bash
# Train each variant for ten epochs on ogbg-molhiv.
bash scripts/train_molhiv_dense_disjoint.sh train_all

# Train and evaluate both variants.
CUDA_VISIBLE_DEVICES=0 EPOCH_END=10 \
  bash scripts/train_molhiv_dense_disjoint.sh all
```

Set `DATA_ROOT`, `DATASET`, `EPOCH_START`, `EPOCH_END`, `BATCH_SIZE`, and
`NUM_WORKERS` to override the defaults. Checkpoints and metrics are written to
`results/training/`, which is excluded from version control.

## Reproducing evaluation

First build the baseline executable:

```bash
make -C third_party/App-BMao-tlimited_main_table
```

Then run a dataset-specific script from the repository root, for example:

```bash
bash scripts/Overall_Performance/ogbg-molhiv/partialdiff.sh
bash scripts/Overall_Performance/ogbg-molhiv/partialdiff_refine.sh
bash scripts/Overall_Performance/ogbg-molhiv/app_bmao_k100.sh
```

