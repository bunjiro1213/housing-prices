# Housing Prices Project

This repository contains a workflow for cleaning and modeling the Kaggle Housing Prices dataset. The goal is to predict home sale prices using various features provided in the dataset.

## Structure

```
.
├── data/                # Raw and cleaned CSV files
├── notebook/            # Exploratory analysis notebook
├── reports/             # Report notebook
├── src/                 # Data cleaning and model training scripts
└── requirements.txt     # Python package requirements
```

## Setup

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Clean the data**

   Run the cleaning script to generate `train_cleaned.csv` and `test_cleaned.csv` in the `data/` directory:

   ```bash
   python src/clean.py
   ```

2. **Train a model and make predictions**

   Execute the training script to train a random forest model and produce `submission.csv`:

   ```bash
   python src/train.py
   ```

The scripts load data from the `data/` directory and output files to the same location.

## Notebooks

- `notebook/exploratory.ipynb` contains exploratory data analysis.
- `reports/report.ipynb` offers a short summary of results.

## Dataset

The raw data files (`train.csv`, `test.csv`) and a detailed description (`data_description.txt`) originate from Kaggle's [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition.
