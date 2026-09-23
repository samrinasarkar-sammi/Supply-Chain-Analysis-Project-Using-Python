# DataCo Supply Chain Performance Analysis

Exploratory data analysis and machine learning project on the [DataCo Smart Supply Chain dataset](https://data.mendeley.com/datasets/8gx2fvg2k6/5), investigating delivery delays, profitability, and building a predictive model for late-delivery risk.

## Overview

This project analyzes 172,765 orders to answer three questions:

1. **How reliable is delivery, and what drives late shipments?**
2. **Does lateness actually hurt profitability?**
3. **Can late-delivery risk be predicted before an order ships?**

Key finding: **54.7% of orders arrive late**, and the delay rate is driven overwhelmingly by *shipping mode* (First Class: 100% late, Second Class: 79.8%, Standard: 39.8%) rather than by region, customer segment, or time of year — pointing to a delivery-promise mismatch rather than a seasonal or capacity problem. Per-order profit stays roughly flat ($20–23) regardless of delay length, so the business risk from lateness is reputational, not margin-driven.

## Repository Contents

| File | Description |
|---|---|
| `Supply Chain Analysis Project.ipynb` | Main analysis notebook: data cleaning, EDA, profitability analysis, bottleneck/root-cause detection, time-based patterns, and ML modeling |
| `DataCoSupplyChainDataset.csv` | Source dataset (not included if too large for the repo — see [Data](#data) below) |
| `DataCo Supply Chain Performance Report.pdf` | Formatted supply chain performance report (executive summary, KPIs, findings, recommendations) |

## Data

The dataset contains order-level supply chain data: shipping mode, order status, delivery dates, product category, customer segment, region, and profit per order. Download it from [Mendeley Data](https://data.mendeley.com/datasets/8gx2fvg2k6/5) and place `DataCoSupplyChainDataset.csv` in the project root before running the notebook. The CSV uses `latin-1` encoding.

## Analysis Pipeline

1. **Data cleaning** — drop redundant/PII columns, remove canceled orders, parse order/shipping dates
2. **Feature engineering** — `Order Processing Time`, `Delay`, `Is_Delayed`, `Profitability Flag`, calendar features (month/day/hour)
3. **EDA** — KPI summary, profitability distribution, delay distribution vs. profit
4. **Bottleneck detection** — delay % by region, customer segment, shipping mode, order status, type, department
5. **Root cause analysis** — factor drill-down within the worst-performing region
6. **Time-based analysis** — delay % by month, day of week, hour of day
7. **Machine learning** — predicting `Late_delivery_risk` with frequency-encoded categoricals and class balancing, comparing:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - K-Nearest Neighbors

   Models are compared on Accuracy, Precision, Recall, F1, and ROC-AUC. **Random Forest** is the recommended model (ROC-AUC 0.827), since it ranks late-delivery risk most reliably for proactive flagging at booking time.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

> The notebook was originally built in Google Colab and mounts Google Drive to load the CSV. If running locally, replace the `drive.mount(...)` / `pd.read_csv('/content/drive/...')` cell with a local file path.

## Usage

```bash
jupyter notebook Untitled0.ipynb
```

Run all cells in order. The final cells fit and evaluate the classifiers; re-run the modeling section to reproduce the comparison table.

## Key Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.719 | 0.768 | 0.730 | 0.748 | 0.728 |
| **Random Forest** | 0.738 | 0.785 | 0.746 | 0.765 | **0.825** |
| KNN | 0.697 | 0.761 | 0.686 | 0.722 | 0.767 |
| Gradient Boosting | 0.725 | 0.893 | 0.590 | 0.710 | 0.778 |
| Logistic Regression | 0.697 | 0.833 | 0.590 | 0.690 | 0.709 |

Top predictive features (Random Forest): order hour, scheduled shipping days, shipping mode, order month, order region, product category.

## About Me

**Samrina Sarkar Sammi** — M2 Data Science & Network Intelligence student, Télécom SudParis

[LinkedIn](https://www.linkedin.com/in/samrina-sarkar-sammi-a8b716424/) · [GitHub](https://github.com/samrinasarkar-sammi) · samrinasarkar@gmail.com

