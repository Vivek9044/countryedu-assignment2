# Executive Analytics Report: Client Churn Risk & Revenue Forecasting
**Prepared for**: Managed IT Services Executive Leadership  
**Prepared by**: Senior Data Science & Software Engineering Team  
**Date**: July 6, 2026  

---

## 1. Executive Summary

This report delivers a data-driven risk management framework for our Managed IT Services portfolio of **200 enterprise clients**. By analyzing **10 million transactional records** covering 3 years of billing, usage logs, and customer support history, we have successfully developed:
1. **A Churn Prediction Engine** that identifies clients likely to churn in the next 3 months, isolating high-risk accounts before they terminate services.
2. **An Operational Audit** detailing the top 5 operational and service-related drivers of customer churn.
3. **A 12-Month Revenue Forecasting Model** that quantifies the direct financial impact of customer churn on our monthly billings.
4. **An Actionable Retention Strategy** demonstrating how targeted operational interventions can reduce overall active portfolio churn by **26.32% relative** (well exceeding our **5% business target**) and preserve **$16,373.81 per month** in monthly recurring billing.

---

## 2. Methodology & Pipeline Architecture

### Data scale & Streaming Aggregation
The dataset consists of **10,000,000 transaction records** (0.68 GB on disk). To process this data without exceeding standard server memory constraints, we built a streaming aggregation pipeline. The transactions are processed in MapReduce-style chunks of 1,000,000 rows, aggregating daily usage logs and billing events into a consolidated monthly panel dataset of **6,848 client-months** (200 clients over a 36-month timeline).

### Missing Financial Data Imputation (12% Nulls)
Exactly **12.12% of transaction amounts** and **13.50% of client MRR values** were missing. We evaluated two cleaning methods:
1. **Strategy A (Median Imputation)**: Filled missing values with the global median of the cohort.
2. **Strategy B (Linear Regression Imputation)**: Imputed missing client MRR using a linear regression model trained on client size, contract type, and services count.

*Justification*: Strategy B (Regression) was selected for MRR because it preserved the contract pricing tiers between Enterprise, Medium, and Small business classes (yielding an **R-squared of 0.73** in training). Strategy A (Median) was used for payment delay averages. Missing transaction amounts were resolved by looking up each client's clean MRR value, preserving individual client billing profiles perfectly.

### Imbalance Mitigation (18% base rate)
Only **18.00% of clients** in our portfolio (36 out of 200) represented historical churn. To prevent our machine learning models from ignoring this minority class:
- We split the data by `client_id` (GroupShuffleSplit: 75% train, 25% test) to prevent temporal data leakage.
- We applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training set, generating synthetic minority samples until the classes were exactly equal (5,012 positive vs. 5,012 negative client-months).

---

## 3. Exploratory Data Analysis (EDA) Highlights

Our exploratory analysis uncovered strong operational correlations with customer churn:
- **Contract Vulnerability**: Clients on **Monthly contracts** exhibit a massive churn rate, whereas those on **Annual or Multi-year contracts** show high retention rates.
- **Service Friction**: Churned clients averaged **2 to 5 SLA breaches** in the 90 days preceding their churn, and had support ticket satisfaction ratings dropping below **3.2 out of 5.0**.
- **Revenue Concentration ("Whale" Risk)**: The top **10% of clients** account for **32.8% of our total portfolio revenue**. The Gini coefficient of our MRR distribution is **0.428**, indicating high revenue concentration. Losing even one enterprise "whale" represents a substantial threat to cash flow.

---

## 4. Model Evaluation & Comparisons

### Churn Classification Model (Next 3 Months)
We compared three binary classification algorithms on a held-out test set of clients:

| Model | ROC-AUC | PR-AUC | Churn Recall (Class 1) | Churn Precision (Class 1) | Churn F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Best)** | **0.985** | **0.633** | **94.4%** | **25.6%** | **0.402** |
| Random Forest | 0.964 | 0.316 | 69.4% | 20.7% | 0.318 |
| XGBoost Classifier | 0.979 | 0.532 | 69.4% | 36.2% | 0.476 |

*Justification*: **Logistic Regression** (with SMOTE balancing) was selected as our core model because it maximized **Recall (94.4%)**, ensuring we catch almost every client about to churn. While XGBoost has slightly higher Precision, catching churn risk early is our primary business goal.

### 12-Month Revenue Forecasting Model
We compared two time-series approaches backtested against the last 6 months of historical monthly billings:

| Time Series Model | Backtest RMSE | Backtest MAPE (%) | Selection Status |
| :--- | :---: | :---: | :---: |
| **Holt-Winters Exponential Smoothing** | **$147,020.41** | **5.30%** | **SELECTED** |
| XGBoost Regressor (Lags 1-3) | $134,609.55 | 6.96% | Evaluated |

*Justification*: **Holt-Winters** achieved a highly accurate Mean Absolute Percentage Error (MAPE) of **5.30%** and was chosen due to its ability to handle monthly seasonality and trends on short data horizons (36 months).

---

## 5. Churn Driver Analysis (Top 5 Factors)

By extracting model feature coefficients, we ranked the operational factors driving customer churn:
1. **Total Product Usage Volume (20.27% Influence)**: Dropping usage volume is the strongest leading indicator of client dormancy and eventual churn.
2. **Months Since Last Upgrade (12.09% Influence)**: Clients that go >12 months without upgrading workload capacities or adopting new services are highly vulnerable to competitors.
3. **Revenue Trend (8.87% Influence)**: Gradual billings reduction (downsizing services) is a high-risk precursor to churn.
4. **Cloud Service Usage Volume (6.17% Influence)**: Low utilization of cloud workloads indicates poor integration, making contract cancellation easy.
5. **Product Usage Trend (5.58% Influence)**: A steep downward usage trajectory over 3 months is a strong trigger for client churn.

---

## 6. 12-Month Revenue Forecast Summary

Using the Holt-Winters model, we projected our next 12 months of monthly billings under two scenarios:
1. **Baseline Forecast**: Assumes active portfolio continues without interventions, projecting a stable line of around **$1.73M/month**.
2. **Churn-Adjusted Forecast**: Subtracts the probability-weighted expected revenue loss from our active clients predicted to churn in the next 3 months (expected loss of **$40,823.90/month**, phased in over months 1-3).

*Key Projections Table*:

| Forecast Month | Baseline ($) | Churn-Adjusted ($) | Monthly Churn Revenue Lost ($) |
| :---: | :---: | :---: | :---: |
| **Month 1 (Jul)** | $1,731,939.10 | $1,731,939.10 | $0.00 (Baseline) |
| **Month 2 (Aug)** | $1,731,939.10 | $1,718,467.21 | $13,471.89 (33% Phased Churn) |
| **Month 3 (Sep)** | $1,731,939.10 | $1,704,587.09 | $27,352.01 (67% Phased Churn) |
| **Month 4 (Oct)** | $1,731,939.10 | $1,691,115.20 | $40,823.90 (100% Churn Realized) |
| **Month 12 (Jun)** | $1,731,939.10 | $1,691,115.20 | $40,823.90 (Sustained Loss) |

---

## 7. Actionable Recommendations & Retention Impact

To meet and exceed our **5% churn reduction target**, we suggest focusing on the top prioritized alerts generated by our engine:

### Prioritized Client Interventions (Top 3 Alerts)
1. **Hooper PLC (CLT170)**: Churn Prob: **47.1%** | MRR: **$30,064.96** | Owner: **Diana Prince**
   - *Risk Driver*: High SLA Breach Rate (100% support ticket escalations).
   - *Action*: Schedule an immediate operations review to resolve technical service breaches.
2. **Robinson, Jones and Welch (CLT054)**: Churn Prob: **34.5%** | MRR: **$28,273.00** | Owner: **Alice Smith**
   - *Risk Driver*: High SLA Breach Rate.
   - *Action*: Technical audit to stabilize support response times.
3. **White-Estes (CLT121)**: Churn Prob: **32.5%** | MRR: **$10,400.12** | Owner: **Bob Jones**
   - *Risk Driver*: High SLA Breach Rate.
   - *Action*: Dedicated engineer assignment to resolve unresolved tickets.

### Portfolio-Wide Strategy & Business Impact
By executing these targeted outreach plays (assuming a 60% retention success rate), we project:
- **Total Clients Rescued**: 3 high/medium-risk accounts stabilized.
- **Monthly Revenue Saved**: **$16,373.81 per month** ($196,485.72 / year).
- **Churn Rate Reduction**: Overall active portfolio churn risk falls from **1.59%** to **1.17%**, which is a **26.32% relative reduction** in churn risk, exceeding the **5.00% relative reduction target** by more than 5x!

---

## 8. Assumptions, Limitations, & Real-World Translation

### Key Assumptions
1. **Causal Linearity**: We assume that correcting churn drivers (e.g. reducing SLA breaches via operational improvements) will result in a proportional reduction in churn probability as modelled.
2. **Progressive Rollout**: Churn predictions are modelled to realize fully in Month 4, assuming a 90-day contract notification term.

### Limitations
- **Data Seasonality**: 3 years of monthly data is the minimum necessary to extract seasonal patterns. In a real corporate setting, 5+ years of historical billing is preferred to improve Holt-Winters forecasting accuracy.
- **Unstructured Logs**: We simplified support ticket content to a binary flag (`notes_flag`). Real-world ticket systems contain unstructured text that should be processed using NLP (Natural Language Processing) sentiment analysis to capture client frustration.

### What Would Change with Real Corporate Data
1. **API Integrations**: Instead of reading static CSV dumps, the ETL pipeline would query SQL databases (billing systems) and CRM platforms (Salesforce) via APIs.
2. **Complex Churn Definitions**: Real corporate clients rarely churn abruptly; they often slowly downsize service lines over time ("silent churn"). The model would need to transition from binary classification to a multi-class model (Upgrade / Stable / Downsize / Churn).
3. **Advanced Time Series**: We would utilize ensemble forecasting (combining SARIMA with client-level pipeline forecasts) and model exogenous variables (e.g., industry-wide economic contractions).
