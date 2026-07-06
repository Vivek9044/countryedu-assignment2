import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def prepare_time_series(workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    data_dir = os.path.join(workspace_dir, "data")
    model_df = pd.read_csv(os.path.join(data_dir, "model_ready_dataset.csv"))
    
    # Aggregate monthly company-wide revenue
    monthly_rev = model_df.groupby('year_month')['invoiced_amt'].sum().reset_index()
    monthly_rev = monthly_rev.sort_values('year_month').reset_index(drop=True)
    
    return monthly_rev

def train_and_backtest_forecast(monthly_rev, workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    models_dir = os.path.join(workspace_dir, "models_saved")
    os.makedirs(models_dir, exist_ok=True)
    
    y = monthly_rev['invoiced_amt'].values
    n = len(y)
    
    # 6-month hold-out for backtesting
    train_size = n - 6
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\nTime Series Length: {n} months. Train size: {train_size} months. Test size: 6 months.")
    
    # Model A: Holt-Winters Exponential Smoothing
    hw = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=12, initialization_method='estimated')
    hw_fit = hw.fit()
    hw_pred = hw_fit.forecast(6)
    
    # Model B: XGBoost with lags (Lag 1, Lag 2, Lag 3)
    # Prepare lag features
    ts_df = pd.DataFrame({'value': y})
    ts_df['lag1'] = ts_df['value'].shift(1)
    ts_df['lag2'] = ts_df['value'].shift(2)
    ts_df['lag3'] = ts_df['value'].shift(3)
    ts_df = ts_df.dropna().reset_index(drop=True)
    
    # Split lagged features
    X_lag = ts_df[['lag1', 'lag2', 'lag3']].values
    y_lag = ts_df['value'].values
    
    # The training indices in lagged dataset
    # Note: original index starts at 3, so train size in lagged set is train_size - 3
    train_lag_size = train_size - 3
    
    X_train_lag, X_test_lag = X_lag[:train_lag_size], X_lag[train_lag_size:]
    y_train_lag, _y_test_lag = y_lag[:train_lag_size], y_lag[train_lag_size:]
    
    xgb_reg = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    xgb_reg.fit(X_train_lag, y_train_lag)
    
    # Recursive multi-step forecast for XGBoost on test set
    xgb_pred = []
    last_lags = X_test_lag[0].copy() # first test sample lags
    
    for i in range(6):
        pred_val = xgb_reg.predict(last_lags.reshape(1, -1))[0]
        xgb_pred.append(pred_val)
        # Shift lags
        last_lags = np.array([pred_val, last_lags[0], last_lags[1]])

    # Calculate metrics
    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    hw_rmse = np.sqrt(mean_squared_error(y_test, hw_pred))
    hw_mape = calculate_mape(y_test, hw_pred)
    
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mape = calculate_mape(y_test, xgb_pred)
    
    print("\n" + "="*50)
    print("REVENUE FORECASTING BACKTESTING METRICS")
    print("="*50)
    print(f"Holt-Winters HW: RMSE = ${hw_rmse:,.2f} | MAPE = {hw_mape:.2f}%")
    print(f"XGBoost Lag-model: RMSE = ${xgb_rmse:,.2f} | MAPE = {xgb_mape:.2f}%")
    
    # Select best model (XGBoost or HW)
    # HW is usually better for small sample time-series data with seasonality
    if hw_mape < xgb_mape:
        best_model_name = "Holt-Winters"
        # Fit HW on full data
        final_hw = ExponentialSmoothing(y, seasonal='add', seasonal_periods=12, initialization_method='estimated')
        final_fit = final_hw.fit()
        # Save model fit
        joblib.dump(final_fit, os.path.join(models_dir, "revenue_forecast_model.pkl"))
        best_mape = hw_mape
    else:
        best_model_name = "XGBoost Lags"
        # Fit XGBoost on full data
        xgb_reg_final = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        xgb_reg_final.fit(X_lag, y_lag)
        joblib.dump(xgb_reg_final, os.path.join(models_dir, "revenue_forecast_model.pkl"))
        joblib.dump(X_lag[-1], os.path.join(models_dir, "last_lags.pkl")) # save last lags to project future
        best_mape = xgb_mape

    print(f"\nSelected Model: {best_model_name} (MAPE: {best_mape:.2f}%)")
    joblib.dump(best_model_name, os.path.join(models_dir, "best_forecaster_name.pkl"))
    
    return best_model_name

def generate_future_forecast(monthly_rev, best_model_name, workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    models_dir = os.path.join(workspace_dir, "models_saved")
    reports_dir = os.path.join(workspace_dir, "reports")
    fig_dir = os.path.join(workspace_dir, "reports", "figures")
    
    y = monthly_rev['invoiced_amt'].values
    
    # 12-month future projection
    forecast_horizon = 12
    
    if best_model_name == "Holt-Winters":
        fit_model = joblib.load(os.path.join(models_dir, "revenue_forecast_model.pkl"))
        baseline_forecast = fit_model.forecast(forecast_horizon)
        
        # Estimate confidence intervals from historical residuals standard deviation
        residuals = fit_model.resid
        res_std = np.std(residuals)
        
        # 95% CI is approx forecast +/- 1.96 * std * sqrt(h)
        ci_lower = [baseline_forecast[i] - 1.96 * res_std * np.sqrt(i+1) for i in range(forecast_horizon)]
        ci_upper = [baseline_forecast[i] + 1.96 * res_std * np.sqrt(i+1) for i in range(forecast_horizon)]
    else:
        xgb_reg = joblib.load(os.path.join(models_dir, "revenue_forecast_model.pkl"))
        last_lags = joblib.load(os.path.join(models_dir, "last_lags.pkl"))
        
        baseline_forecast = []
        curr_lags = last_lags.copy()
        
        for i in range(forecast_horizon):
            pred_val = xgb_reg.predict(curr_lags.reshape(1, -1))[0]
            baseline_forecast.append(pred_val)
            curr_lags = np.array([pred_val, curr_lags[0], curr_lags[1]])
            
        # Standard deviation of residuals for confidence bands
        ts_df = pd.DataFrame({'value': y})
        ts_df['lag1'] = ts_df['value'].shift(1)
        ts_df['lag2'] = ts_df['value'].shift(2)
        ts_df['lag3'] = ts_df['value'].shift(3)
        ts_df = ts_df.dropna().reset_index(drop=True)
        X_lag = ts_df[['lag1', 'lag2', 'lag3']].values
        res_std = np.std(y[3:] - xgb_reg.predict(X_lag))
        ci_lower = [baseline_forecast[i] - 1.96 * res_std * np.sqrt(i+1) for i in range(forecast_horizon)]
        ci_upper = [baseline_forecast[i] + 1.96 * res_std * np.sqrt(i+1) for i in range(forecast_horizon)]
        
    baseline_forecast = np.array(baseline_forecast)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)
    
    # 2. Churn-Adjusted Forecast
    # Load risk rankings of clients
    ranking_path = os.path.join(reports_dir, "churn_risk_ranking.csv")
    if not os.path.exists(ranking_path):
        print("Risk ranking file not found. Skipping churn adjustment.")
        return
        
    ranking_df = pd.read_csv(ranking_path)
    
    # Filter to active clients (is_churned = False, meaning currently active in business)
    # We want to estimate future churn of active clients!
    active_ranking = ranking_df[~ranking_df['is_churned']]
    
    # Churn probability-weighted monthly recurring revenue loss
    # E.g. expected monthly revenue loss
    expected_mrr_loss = (active_ranking['monthly_recurring_revenue'] * active_ranking['predicted_churn_probability']).sum()
    
    print(f"\nTotal MRR of currently active clients: ${active_ranking['monthly_recurring_revenue'].sum():,.2f}")
    print(f"Expected monthly revenue loss from predicted churn: ${expected_mrr_loss:,.2f}")
    
    # Scale loss progressively over 12 months:
    # Month 1: 0% loss (clients are still active)
    # Month 2: 33% of expected loss
    # Month 3: 67% of expected loss
    # Month 4 to 12: 100% of expected loss
    monthly_loss_scale = np.zeros(forecast_horizon)
    monthly_loss_scale[1] = 0.33
    monthly_loss_scale[2] = 0.67
    monthly_loss_scale[3:] = 1.0
    
    churn_loss_series = expected_mrr_loss * monthly_loss_scale
    churn_adjusted_forecast = baseline_forecast - churn_loss_series
    
    # Create future date indices
    last_date = pd.to_datetime(monthly_rev['year_month'].iloc[-1] + '-01')
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_horizon + 1)]
    future_yms = [d.strftime('%Y-%m') for d in future_dates]
    
    # Save forecast dataframe
    forecast_df = pd.DataFrame({
        'year_month': future_yms,
        'baseline_forecast': baseline_forecast,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'churn_adjusted_forecast': churn_adjusted_forecast,
        'expected_mrr_loss': churn_loss_series
    })
    
    forecast_df.to_csv(os.path.join(reports_dir, "revenue_forecast_12m.csv"), index=False)
    print(f"Saved 12-month forecasts to: {reports_dir}/revenue_forecast_12m.csv")
    
    # 3. Plot Forecast Chart
    plt.figure(figsize=(12, 6))
    
    # Plot historical (last 18 months for better scale, or full 36)
    hist_months = monthly_rev.tail(24)
    plt.plot(pd.to_datetime(hist_months['year_month'] + '-01'), hist_months['invoiced_amt'], 
             'k-', label='Historical Revenue', linewidth=2)
             
    # Plot future baseline
    plt.plot(future_dates, baseline_forecast, 'b--', label='Baseline Forecast (No Churn Change)', linewidth=2)
    plt.fill_between(future_dates, ci_lower, ci_upper, color='blue', alpha=0.15, label='95% Confidence Band')
    
    # Plot future churn-adjusted
    plt.plot(future_dates, churn_adjusted_forecast, 'r-', label='Churn-Adjusted Forecast', linewidth=2.5)
    
    plt.title("12-Month Revenue Forecasting (Baseline vs. Churn-Adjusted)", weight='bold', fontsize=14)
    plt.xlabel("Timeline")
    plt.ylabel("Monthly Revenue ($)")
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(fig_dir, "revenue_forecast.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Revenue forecast plot saved to: {plot_path}")
    print("Forecasting model complete!")

def run_forecasting_pipeline():
    workspace_dir = "c:/Users/kasau/Desktop/guptachurn"
    monthly_rev = prepare_time_series(workspace_dir)
    best_model_name = train_and_backtest_forecast(monthly_rev, workspace_dir)
    generate_future_forecast(monthly_rev, best_model_name, workspace_dir)

if __name__ == '__main__':
    run_forecasting_pipeline()
