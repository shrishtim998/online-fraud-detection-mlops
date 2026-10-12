# Online Fraud Transaction Detection — MLOps

## 1. Project Overview

This project aims to detect fraudulent online financial transactions using machine learning. It includes data preprocessing, exploratory data analysis, model training and evaluation, a FastAPI-based prediction service, automated API testing, and continuous integration using GitHub Actions.

## 2. Objectives

- Analyze transaction data and identify patterns associated with fraud.
- Handle severe class imbalance in the dataset.
- Train and compare machine learning classification models.
- Evaluate models using precision, recall, F1-score, ROC-AUC, and Precision-Recall AUC.
- Save and reload the selected machine learning pipeline.
- Expose fraud predictions through a REST API.
- Automate API tests and quality checks using GitHub Actions.

## 3. Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Imbalanced-learn
- **Visualization:** Matplotlib, Seaborn
- **API:** FastAPI, Uvicorn
- **Testing:** Pytest
- **Version Control:** Git and GitHub
- **Continuous Integration:** GitHub Actions
- **Experiment Tracking and Cloud Deployment:** MLflow and Azure Machine Learning (planned/in progress)

## 4. Project Structure

```text
online-fraud-detection-mlops/
├── api/
│   ├── __init__.py
│   └── main.py
├── data/
│   └── onlinefraud.csv
├── models/
│   └── final_fraud_pipeline.pkl
├── reports/
│   └── evaluation metrics and figures
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── compare_models.py
│   ├── compare_imbalance.py
│   ├── evaluate_and_plot.py
│   └── build_final_pipeline.py
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── requirements.txt
└── README.md
```

## 5. Setup Instructions

### Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Run the API

From the project root, run:

```bash
uvicorn api.main:app --reload
```

Open the following URL in your browser to access Swagger UI:

`http://127.0.0.1:8000/docs`

The API provides endpoints for transaction prediction and health checking, subject to the implementation in `api/main.py`.

## 7. Run Automated Tests

Run the API tests from the project root:

```bash
python -m pytest tests/test_api.py -v
```

## 8. Model Evaluation

The project compares Logistic Regression, Decision Tree, Random Forest, and a tuned Decision Tree. Evaluation includes accuracy, precision, recall, F1-score, ROC-AUC, Precision-Recall AUC, confusion matrices, and model comparison reports.
The final Random Forest pipeline recorded the strongest reported results, achieving 99.991% accuracy, 93.50% precision, 99.82% recall, a 96.56% F1-score, 99.99% ROC-AUC, and 99.88% PR-AUC. These results suggest strong fraud-detection performance on the evaluated dataset. However, because fraud detection involves severe class imbalance, precision, recall, and PR-AUC are especially important. 

## 9. Continuous Integration

GitHub Actions is configured to execute the project's automated checks when changes are pushed to the `main` branch or a pull request targets `main`. The workflow installs the project dependencies and runs API tests. Ruff linting is included once configured in the workflow.

## 10. Version Control

Git and GitHub are used to track project changes. The workflow uses meaningful commits, feature branches, and pull requests to review changes before merging them into the main branch.

## 11. Disclaimer
 A model prediction is an estimate and should not be treated as definitive proof that a transaction is fraudulent or legitimate.