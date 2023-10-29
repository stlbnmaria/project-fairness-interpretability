# Traffic Violations
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/stlbnmaria/project-fairness-interpretability/blob/main/.pre-commit-config.yaml)

Authors: Mykyta Alekseiev, Elizaveta Barysheva, Joao Melo, Thomas Schneider, Harshit Shangari and Maria Stoelben

## Description
The goal of this project is to predict a binary variable using white and black box models. Subsequently, the performance and fairness of the models with respect to certain protected features will be analysed. The protected attributes that will be focused on here are gender and race. Moreover, the models' predictions will be analysed with methods for interpretability.

## Data
For this project a dataset of traffic violations in Maryland, USA was selected. You can download the data [here](https://www.openml.org/search?type=data&status=active&sort=runs&order=desc&id=42345). The `.arff` should be placed in a `data/` folder in the root of your repository.

The processed data contains 65'203 instances with 15 columns, where 5 columns are categorical and the rest binary or numeric. The target column is Citation, which is equal to 1 when a citation was given by an officer and 0 if only a warning was declared.

## Setup
Create a virtual environment and install the requirements:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pre-commit install
```

## Data Preproessing
Check out the jupyter notebooks to understand the data the preprocessing decisions.

To run the data preprocessing and get a `data.csv` output for the following parts, run:
```bash
python src/data_preprocessing/data_preprocessor.py
```

## Modeling
The paramters can be changed in the `config/config_modeling.py`. Run the training with mlflow tracking with the following command:
```bash
python src/modeling.py
```
