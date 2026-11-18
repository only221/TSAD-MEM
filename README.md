# TSAD-MEM

## Requirements

- Linux or WSL is recommended.
- Python 3.10 or newer.
- PyTorch 2.1 or newer.
- A CUDA GPU is recommended for the larger datasets. CPU execution is supported
  by passing --device cpu, but can be considerably slower.

Create an environment and install the dependencies:

    conda create -n tsad-mem-public python=3.10 -y
    conda activate tsad-mem-public
    python -m pip install -r requirements.txt

## Data

The five datasets can be downloaded from the following Figshare archive:

<https://figshare.com/s/906e1ce13813ee1668c3>

After downloading, place `SWaT.csv`, `WADI.csv`, `PSM.csv`, `MSL.csv`, and
`GECCO.csv` under `data/`. Verify the corresponding `train_lens` values in
`data/DETECT_META.csv`. Model checkpoints are intentionally excluded from this
release.

## Train and evaluate

From this directory, the following command trains the default configuration:

    bash scripts/run_dataset.sh PSM data/PSM.csv train


Evaluate a saved checkpoint:

    bash scripts/run_dataset.sh PSM data/PSM.csv test

Defaults are loaded from configs/<DATASET>.json. Outputs are written to
results/<DATASET>/ and checkpoints to checkpoints/<DATASET>.pth. Override
these locations with MODEL_PATH and SAVE_PATH:

    MODEL_PATH=/path/to/model.pth SAVE_PATH=/path/to/results \
    bash scripts/run_dataset.sh MSL data/MSL.csv test --device cuda


## Outputs

Training saves a checkpoint containing the model state and fitted
standardization statistics. Evaluation writes
results/<DATASET>/label_results_<DATASET>.json, including the threshold,
configuration, and point-wise, range-based, and affiliation metrics.

