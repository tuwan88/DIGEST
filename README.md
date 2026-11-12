# DIGEST

Code for training and evaluating the dense and disjoint model variants.

## Setup

The environment targets Linux x86_64 (glibc 2.28 or newer) with Python 3.10,
PyTorch 2.10.0 (CUDA 12.8), and PyTorch Geometric 2.7.0. Use an NVIDIA driver compatible
with CUDA 12.8. The matching PyTorch and PyG wheels are configured in
`environment.yml`.

```bash
conda env create -f environment.yml
conda activate partialged
python -m pip check
```

Check the installed packages and GPU access:

```bash
python -c "import torch, torch_geometric, torch_scatter, torch_sparse; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); print('PyG:', torch_geometric.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

## Data

Datasets

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
