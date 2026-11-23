# Assessment 2: Concepts, Splits and Leakage

## Project Overview

This project demonstrates fundamental machine learning concepts using a Student Performance Dataset.

It covers dataset exploration, feature and target identification, train-validation-test splitting, data leakage prevention, and decision tree overfitting experiments.

## Topics Covered

1. Understanding features and target variables.
2. Splitting data into training, validation, and testing datasets.
3. Understanding data leakage.
4. Preventing data leakage using `StandardScaler`.
5. Experimenting with decision tree model complexity and overfitting.

## Dataset

**File:** `data/student_performance.csv`

- 20 student records.
- 3 independent features:
  - `Hours_Studied`
  - `Attendance`
  - `Previous_Score`
- 1 target variable: `Final_Score`.
- No missing values.

## Project Structure

```text
Task - 1 Concepts, Splits and Leakage/
│
├── data/
│   └── student_performance.csv
│
├── notebooks/
│   └── concepts_splits_leakage.ipynb
│
├── src/
│   ├── 01_concepts.py
│   ├── 02_train_test_split.py
│   ├── 03_data_leakage.py
│   └── 04_overfitting_experiment.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── overfitting_experiment.png
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run Python Scripts

Run these commands from the project directory:

```bash
python src/01_concepts.py
python src/02_train_test_split.py
python src/03_data_leakage.py
python src/04_overfitting_experiment.py
```

## Experiments and Expected Results

### 1. Concepts

- Loads the student performance dataset.
- Identifies independent features and the dependent target.
- Displays dataset information, statistical summaries, and missing-value counts.

### 2. Train, Validation and Test Split

- Splits data into training (60%), validation (20%), and testing (20%).
- Uses stratification to preserve target class proportions.
- Uses `random_state=42` for reproducibility.

### 3. Data Leakage

- Demonstrates incorrect scaling when the scaler is fitted using the entire dataset.
- Demonstrates the correct approach by fitting `StandardScaler` only on training data.
- Applies the fitted scaler to the test data.

### 4. Overfitting Experiment

- Trains DecisionTreeClassifier models with maximum depths of 1, 3, 5, 10, and None.
- Compares training and validation accuracy.
- Generates and saves the experiment graph as `overfitting_experiment.png`.

**Note:** The current dataset produces perfect training and validation accuracy for the tested tree depths. Therefore, this experiment does not necessarily demonstrate overfitting on this particular dataset.
