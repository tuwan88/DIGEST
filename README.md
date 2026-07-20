# DIGEST

Code for training and evaluating the dense and disjoint model variants.

## Setup

```bash
conda env create -f environment.yml
conda activate partialged
```



## Data

Datasets and checkpoints under `artifacts/`. Check the required evaluation files with:

```bash
python scripts/Overall_Performance/check_required_artifacts.py
```


## Training and testing

```bash
DATA_ROOT=/path/to/data bash scripts/train_molhiv_dense_disjoint.sh train_all
DATA_ROOT=/path/to/data bash scripts/train_molhiv_dense_disjoint.sh test_all
```

## Evaluation

```bash

bash scripts/Overall_Performance/ogbg-molhiv/partialdiff.sh
```

