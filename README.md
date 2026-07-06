# Client Churn Risk & Revenue Forecasting for Managed IT Services

This repository contains a complete, end-to-end data analytics project designed for Managed IT Services providers. The project profiles 200 enterprise clients, aggregates 10 million transaction records, imputes missing financial data using machine learning, trains a churn prediction model, projects a 12-month revenue forecast (baseline vs. churn-adjusted), and displays the results in an interactive Streamlit dashboard.

---

## Results at a Glance

*   **Dataset Dimensions**: 200 clients, 10,000,000 transactions (0.68 GB CSV file).
*   **Code Quality & Linting**: 100% compliant with PEP8 static analysis rules. Running `ruff check .` returns **0 warnings/errors** across all modules.
*   **Imbalance Handling**: 18.00% historical churn rate (exactly 36 clients) balanced using SMOTE in training.
*   **ETL Imputation Quality**: Initial client MRR missing rate of 13.50% resolved using Linear Regression Imputation (R-squared: 0.73). Initial transaction amounts missing rate of 12.12% resolved using client MRR lookup.
*   **Churn Classification**: Logistic Regression chosen as the best model. Validation ROC-AUC: **0.985** | PR-AUC: **0.633** | Churn Recall: **94.4%** (maximizing risk detection).
*   **Inference Latency**: **0.097 milliseconds** per client, enabling real-time scoring.
*   **Top Churn Driver**: Total Product Usage Volume (accounting for 20.27% of model decision weight).
*   **12-Month Revenue Forecast**: Holt-Winters chosen as the best model (Backtest MAPE: **5.30%**).
*   **Retention Strategy Savings**: Proactive account outreach to the top medium/high-risk accounts projects a **26.32% relative reduction** in active portfolio churn risk, meeting our **5.00% target** and saving **$16,373.81 per month** in recurring billings.

---

## Step-by-Step Setup Guide

This guide is written for users with no previous Python or machine learning experience.

### 1. Open your Terminal/Command Prompt
Open your command-line interface and make sure you are in the project folder:
```powershell
cd c:\Users\kasau\Desktop\guptachurn
```

### 2. Create the Virtual Environment
Create a virtual python environment called `venv` to isolate the packages:
```powershell
python -m venv venv
```

### 3. Activate the Virtual Environment
Activate the environment so that commands use our isolated environment:
-   **Windows PowerShell**:
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
-   **Windows Command Prompt (cmd)**:
    ```cmd
    .\venv\Scripts\activate.bat
    ```

### 4. Install Project Dependencies
Install all required libraries (Pandas, Numpy, Scikit-Learn, XGBoost, Streamlit, Matplotlib, Seaborn, statsmodels):
```powershell
pip install -r requirements.txt
```

---

## Running the Pipeline

Run each script in order to execute the pipeline:

### Step 1: Generate Synthetic Data
Generates the roster of 200 clients and the 10M transactions.
```powershell
python src/data_generator.py
```
*Note on scale*: Generating 10 million transactions takes approximately **150-160 seconds** on standard hardware. It writes in optimized 1-million-row chunks to maintain a peak memory footprint of less than **170 MB**.

### Step 2: Run the ETL Pipeline
Loads, cleans, imputes the missing data, aggregates transactions into monthly metrics, engineers rolling features, and writes the model-ready dataset.
```powershell
python src/etl_pipeline.py
```
*Note on scale*: Processing and aggregating the 10 million records takes approximately **315 seconds (5 minutes)**. It runs in a chunked streaming loop to keep memory usage low.

### Step 3: Run the EDA Visualizations
Performs data profiling, Gini/Lorenz revenue concentration math, and exports the charts to `reports/figures/`.
```powershell
python src/run_eda.py
```

### Step 4: Train Machine Learning Models
Trains the churn model, generates the client risk rankings (`reports/churn_risk_ranking.csv`), and performs the 12-month revenue forecast.
```powershell
python src/models/churn_model.py
python src/models/churn_driver_analysis.py
python src/models/revenue_forecast_model.py
```

### Step 5: Execute Code Quality & Lint Check
Scans the code structure using Ruff to verify that the workspace is 100% compliant with PEP8 standards:
```powershell
ruff check .
```

### Step 6: Execute Automated Verification
Runs the 6 automated tests to verify dataset dimensions, imputation rates, predictions, latency, and forecast outputs.
```powershell
python tests/test_pipeline.py
```

### Step 7: Launch the Interactive Dashboard
Run the Streamlit app to explore the live KPIs, sortable risk rankings, forecasting comparisons, and the strategy simulator:
```powershell
streamlit run dashboard/app.py
```
*Note*: Streamlit will open a web browser tab displaying the interactive dashboard interface.

---

## Directory Structure

*   `data/`: Contains `clients.csv`, `transactions.csv`, and the clean aggregated `model_ready_dataset.csv`.
*   `src/`: Contains data generators, ETL scripts, and models.
    *   `src/models/churn_model.py`: Churn classifier.
    *   `src/models/churn_driver_analysis.py`: Extracts and plots feature importances.
    *   `src/models/revenue_forecast_model.py`: Time-series forecasting (baseline vs churn-adjusted).
    *   `src/models/strategy_simulator.py`: Interactive intervention math.
    *   `src/recommendations.py`: Generates prioritised alerts.
*   `notebooks/`: Contains `eda.ipynb` for Jupyter Notebook usage.
*   `dashboard/`: Contains `app.py` for the Streamlit dashboard.
*   `models_saved/`: Stores trained models (`churn_model.pkl`, `scaler.pkl`).
*   `reports/`: Contains `Final_Report.md`, risk rankings, and generated figures.
*   `tests/`: Contains the automated verification script `test_pipeline.py`.
