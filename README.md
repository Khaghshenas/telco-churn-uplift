# Telco Churn and Uplift Modeling

A machine learning pipeline for predicting customer churn and estimating the incremental impact of retention actions, using the Telco Customer Churn dataset.


This project consists of two main components:

- **Customer churn prediction** using a **Random Forest classifier**, trained on historical customer attributes and usage patterns to estimate churn risk.
- **Uplift modeling** to estimate the **incremental impact of retention interventions**, by comparing predicted outcomes between treated and control customer groups.

The uplift modeling step builds on the churn predictions to identify customers for whom a retention action is most likely to change the outcome (i.e., prevent churn), enabling more efficient and targeted customer engagement strategies.


## Project Structure

```text
telco-churn-uplift/
├── models/ # Saved model artifacts
├── notebooks/
│   └── churn_uplift_analysis.ipynb
├── src/ 
│   ├── etl/
│   ├── models/
│   └── serve/
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
*Note:* The dataset is not included in the repository and must be downloaded and placed in the `data/` directory.
```

## Usage (Local Setup)

1. Clone the repository:

```bash
git clone https://github.com/Khaghshenas/telco-churn-uplift.git
cd telco-churn-uplift
```

2. Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Prepare the dataset:

Place the dataset CSV in the data/ folder, and adjust file paths in code/notebooks if needed.

## Evaluation Results
## Churn Prediction

The churn prediction model was evaluated on a test set using probability-based and classification metrics.

- **ROC-AUC:** **0.82**, indicating strong discrimination between churners and non-churners.
- **Accuracy:** 79% on the test set.
- **Churn Precision:** 64%, meaning that nearly two-thirds of customers predicted as churners actually churned.
- **Churn Recall:** 51%, capturing over half of true churn cases.

### Confusion Matrix (Test Set)
```text
|               | Predicted No Churn | Predicted Churn |
|---------------|-------------------|-----------------|
| **Actual No Churn** | 927 | 106 |
| **Actual Churn**    | 185 | 189 |
```

These results show  a reasonable trade-off between false positives and false negatives. The predicted churn probabilities are later used as input for uplift modeling to prioritize targeted retention strategies.

## Uplift Modeling
### Uplift Model Evaluation

The uplift model predicts the incremental effect of retention actions for each customer. Sample uplift scores:
```text
[ 0.023 0.062 -0.002 -0.033 0.354 0.126 -0.017 -0.059 -0.134 0.015 ]
```

Evaluation on the test set:

- **Uplift @ top 20%:** 0.024 — targeting the top 20% predicted customers yields a small incremental benefit over random selection.  
- **Qini AUC:** -0.029 — the model’s ranking of customers by uplift is currently not better than random, likely due to synthetic treatment assignments.

> **Note:** These results are for demonstration purposes. Using real treatment data and model tuning is expected to significantly improve uplift performance.

## License

This project is licensed under the MIT License.

