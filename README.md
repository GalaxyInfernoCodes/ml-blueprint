# ML Blueprint

A starter template for machine learning projects as part of my newsletter series: https://www.sarahglasmacher.com/ml-repo-structure-challenge/

## Features

- Project structure for ML workflows using a Python package for easier deployment later on
- Example scripts for data processing, training, and evaluation
- Pyproject.toml file for dependency and project management

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ml-blueprint.git
    cd ml-blueprint
    ```

2. Install dependencies:
    ```bash
    uv sync
    ```

## Example Data
Uses example data from the following Kaggle dataset for demonstration purposes only:
https://www.kaggle.com/competitions/playground-series-s4e12/data

To use the repo, add the data yourself in the following structure: 


## Project Structure

```
ml-blueprint/
.
├── artifacts/                     # generated: duckdb, plots, metrics, temp exports
├── config/
│   ├── project.yaml               # paths, switches, seeds
│   ├── features.yaml              # feature toggles/params
│   └── model.yaml                 # model + training params
├── data/                          # tiny sample inputs only (ok to commit)
├── notebooks/
│   └── explore.ipynb
├── scripts/
│   ├── train.py                   # calls into src.<pkg>.* with configs
│   └── predict.py
├── src/
│   └── <your_pkg>/
│       ├── __init__.py
│       ├── config.py              # Pydantic loader for YAML files
│       ├── data.py                # load/split IO; write to DuckDB if used
│       ├── features.py            # transforms / feature sets
│       ├── modeling.py            # fit/persist; MLflow logging hooks
│       └── evaluation.py          # metrics & plots
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
├── pyproject.toml
├── README.md
└── uv.lock
```