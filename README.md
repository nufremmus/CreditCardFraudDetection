# Credit Card Fraud Detection

## Background

This project is based on the [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) provided by the Machine Learning Group at Université Libre de Bruxelles (ULB) in collaboration with Worldline.

The dataset contains credit card transactions made by European cardholders over two days in September 2013. The goal is to classify transactions as fraudulent or genuine.

## Dataset

| | |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.172%) |
| Time period | September 2013 (2-day window) |

**Features:**
- `V1`–`V28`: PCA-transformed features (original features are confidential)
- `Time`: Seconds elapsed since the first transaction
- `Amount`: Transaction amount
- `Class`: Target label — `1` = fraud, `0` = genuine

## Evaluation

The dataset is highly imbalanced, making standard accuracy a poor metric. **Area Under the Precision-Recall Curve (AUPRC)** is the recommended evaluation metric.
