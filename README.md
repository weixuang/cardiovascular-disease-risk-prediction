# Cardiovascular Disease Risk Prediction Model Based on Multi-dimensional Health Examination Data

This repository contains project code for early prediction of cardiovascular disease risk, including:
- Data reading and training/validation (`src/main.py`)
- Result recovery and analysis (`src/recover_and_plot_part1.py`)
- Visualization such as confusion matrices (`src/plot_confusion_matrices.py`)
All experiments can be fully reproduced with a single command.
---

## 1. Quick Start

```bash
git clone https://github.com/weixuang/cardiovascular-disease-risk-prediction.git
cd cardiovascular-disease-risk-prediction

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

./run.sh
```

## 2. Data Access

The dataset for this project is already divided into 5 folds, which are located in the data/ folder. Each fold has separate training, validation, and testing files, such as:
- data/cardio_exp0_train.csv, data/cardio_exp0_test.csv, data/cardio_exp0_val.csv
- data/cardio_exp1_train.csv, data/cardio_exp1_test.csv, data/cardio_exp1_val.csv
- data/cardio_exp2_train.csv, data/cardio_exp2_test.csv, data/cardio_exp2_val.csv
- And so on for 5 folds (fold 0 to fold 4).

If you want to use your own dataset:
-Ensure the dataset is structured similarly and the columns are properly formatted as expected by the code.
-Simply replace the data/ folder with your dataset.

For privacy or size constraints, if the dataset cannot be uploaded, provide the dataset download link or instructions on how to obtain it.

## 3. Single Command Reproducibility

To ensure reproducibility of the entire experiment, simply run the run.sh script. This script automates the entire workflow, including training, evaluation, and generating visualizations.

```bash
./run.sh
```

This will:
- Run the 5-fold cross-validation.
- Generate various performance metrics.
- Save visualizations (e.g., confusion matrices, performance plots) in the output/ folder.