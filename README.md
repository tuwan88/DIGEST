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

## Training

For example, train on `ogbg-molhiv` with:

```bash
bash scripts/Overall_Performance/ogbg-molhiv/train.sh
```


## Testing

The following examples evaluate `ogbg-molhiv`. The same script layout is available for the other datasets.

Run the model without refinement:

```bash
bash scripts/Overall_Performance/ogbg-molhiv/partialdiff.sh
```

Run the model with GPU-based refinement:

```bash
bash scripts/Overall_Performance/ogbg-molhiv/partialdiff_refine.sh
```

Run the model with external App-BMao refinement:

```bash
bash scripts/Overall_Performance/ogbg-molhiv/partialdiff_app_bmao_refine.sh
```

Test results are written to `results/Overall_Performance/`.
