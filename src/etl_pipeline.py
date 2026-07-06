import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def impute_clients_data(clients_df):
    print("\n--- Imputing Clients Financial Data ---")
    
    # Check initial null counts
    mrr_nulls_before = clients_df['monthly_recurring_revenue'].isna().sum()
    delay_nulls_before = clients_df['payment_delay_days_avg'].isna().sum()
    print(f"Initial missing MRR rows: {mrr_nulls_before} ({mrr_nulls_before/len(clients_df):.2%})")
    print(f"Initial missing Payment Delay rows: {delay_nulls_before} ({delay_nulls_before/len(clients_df):.2%})")
    
    # Store original column to compare
    orig_mrr = clients_df['monthly_recurring_revenue'].copy()

    # Strategy A: Median Imputation
    median_mrr = clients_df['monthly_recurring_revenue'].median()
    mrr_median_imputed = clients_df['monthly_recurring_revenue'].fillna(median_mrr)
    
    # Strategy B: Regression-based Imputation
    # Prepare features for regression: client_size, industry, and number of services
    # Let's map client_size to numeric
    size_map = {'Small': 1, 'Medium': 2, 'Enterprise': 3}
    clients_df['size_num'] = clients_df['client_size'].map(size_map)
    
    # Encode industry
    industry_dummies = pd.get_dummies(clients_df['industry'], prefix='ind', drop_first=True)
    
    # Count of services
    clients_df['num_services'] = clients_df['services_subscribed'].apply(lambda x: len(x.split(';')))
    
    reg_features = pd.concat([clients_df[['size_num', 'num_services']], industry_dummies], axis=1)
    
    # Split into train (non-null MRR) and predict (null MRR)
    train_idx = clients_df['monthly_recurring_revenue'].notna()
    predict_idx = clients_df['monthly_recurring_revenue'].isna()
    
    X_train = reg_features[train_idx]
    y_train = clients_df.loc[train_idx, 'monthly_recurring_revenue']
    X_pred = reg_features[predict_idx]
    
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    predicted_mrr = reg_model.predict(X_pred)
    
    mrr_reg_imputed = clients_df['monthly_recurring_revenue'].copy()
    mrr_reg_imputed.loc[predict_idx] = predicted_mrr
    
    # Print comparisons
    print("\nComparison of MRR Imputation Strategies:")
    print(f"Original MRR Mean: ${orig_mrr.mean():,.2f} | Std: ${orig_mrr.std():,.2f}")
    print(f"Median Imputed MRR Mean: ${mrr_median_imputed.mean():,.2f} | Std: ${mrr_median_imputed.std():,.2f}")
    print(f"Regression Imputed MRR Mean: ${mrr_reg_imputed.mean():,.2f} | Std: ${mrr_reg_imputed.std():,.2f}")
    
    # We choose Regression Imputation as it preserves the correlation with client size and services subscribed
    clients_df['monthly_recurring_revenue'] = mrr_reg_imputed
    
    # Impute payment_delay_days_avg with median (simple and robust for this metric)
    median_delay = clients_df['payment_delay_days_avg'].median()
    clients_df['payment_delay_days_avg'] = clients_df['payment_delay_days_avg'].fillna(median_delay)
    
    # Clean up temporary helper columns
    clients_df = clients_df.drop(columns=['size_num', 'num_services'])
    
    # Verify 0 nulls
    mrr_nulls_after = clients_df['monthly_recurring_revenue'].isna().sum()
    delay_nulls_after = clients_df['payment_delay_days_avg'].isna().sum()
    print(f"Final missing MRR: {mrr_nulls_after}")
    print(f"Final missing Payment Delay: {delay_nulls_after}")
    print("Justification: Regression-based imputation chosen for MRR as it preserves the business-tier billing structure based on client size (R-squared: {:.2f}).".format(reg_model.score(X_train, y_train)))
    
    return clients_df

def process_and_aggregate_transactions(transactions_path, clients_df, chunksize=1000000):
    print("\n--- Processing and Aggregating Transactions in Chunks ---")
    start_time = time.time()
    
    # Map clients to their MRR for fast vectorised lookup
    client_mrr_map = clients_df.set_index('client_id')['monthly_recurring_revenue'].to_dict()
    
    chunk_list = []
    chunk_idx = 0
    total_rows = 0
    
    # We will read transactions in chunks
    for chunk in pd.read_csv(transactions_path, chunksize=chunksize):
        chunk_idx += 1
        total_rows += len(chunk)
        
        # 1. Imputation comparison for amount (Strategy A: Global Median vs Strategy B: Client MRR lookup)
        # Find missing count in chunk
        missing_amt_mask = chunk['amount'].isna() & chunk['transaction_type'].isin(['invoice', 'payment', 'service-upgrade', 'service-downgrade'])
        missing_amt_count = missing_amt_mask.sum()
        
        if chunk_idx == 1:
            print(f"First chunk initial missing financial amounts: {missing_amt_count} rows")
            
        # Strategy A: Impute with Global Median of the chunk
        global_median = chunk.loc[chunk['transaction_type'] == 'invoice', 'amount'].median()
        if pd.isna(global_median):
            global_median = 10000.0
            
        # Strategy B: Impute with Client MRR lookup (Chosen)
        # Fill missing amounts in invoices/payments with client MRR
        # For upgrades/downgrades, we can use a standard median or client-specific fraction
        lookup_values = chunk.loc[missing_amt_mask, 'client_id'].map(client_mrr_map)
        
        # Handle upgrades/downgrades separately (e.g. median for upgrades)
        up_down_mask = missing_amt_mask & chunk['transaction_type'].isin(['service-upgrade', 'service-downgrade'])
        chunk.loc[up_down_mask, 'amount'] = chunk.loc[up_down_mask, 'amount'].fillna(1500.0) # default value
        
        # Fill the rest (invoices/payments) with looked-up MRR
        chunk.loc[missing_amt_mask, 'amount'] = chunk.loc[missing_amt_mask, 'amount'].fillna(lookup_values)
        
        # 2. Extract year-month
        chunk['year_month'] = chunk['transaction_date'].str[:7]
        
        # 3. Intermediate aggregation to reduce memory
        # Group by client and month to sum amounts and usage units
        # We also need to count different transaction types
        
        # Invoices and payments
        invoice_rows = chunk[chunk['transaction_type'] == 'invoice']
        payment_rows = chunk[chunk['transaction_type'] == 'payment']
        ticket_rows = chunk[chunk['transaction_type'] == 'support-ticket']
        usage_rows = chunk[chunk['transaction_type'] == 'usage-log']
        upgrade_rows = chunk[chunk['transaction_type'] == 'service-upgrade']
        downgrade_rows = chunk[chunk['transaction_type'] == 'service-downgrade']
        
        # Group invoices
        inv_agg = invoice_rows.groupby(['client_id', 'year_month'])['amount'].sum().reset_index(name='invoiced_amt')
        
        # Group payments and statuses
        pay_agg = payment_rows.groupby(['client_id', 'year_month']).agg(
            paid_amt=('amount', 'sum'),
            late_pay_count=('payment_status', lambda x: (x == 'late').sum()),
            failed_pay_count=('payment_status', lambda x: (x == 'failed').sum()),
            total_pay_count=('payment_status', 'count')
        ).reset_index()
        
        # Group tickets
        tkt_agg = ticket_rows.groupby(['client_id', 'year_month']).agg(
            ticket_count=('transaction_id', 'count'),
            escalated_ticket_count=('notes_flag', 'sum')
        ).reset_index()
        
        # Group usage
        use_agg = usage_rows.groupby(['client_id', 'year_month']).agg(
            total_usage=('usage_units', 'sum'),
            usage_infra=('usage_units', lambda x: x[chunk.loc[x.index, 'service_line'] == 'Infrastructure'].sum()),
            usage_cloud=('usage_units', lambda x: x[chunk.loc[x.index, 'service_line'] == 'Cloud'].sum()),
            usage_security=('usage_units', lambda x: x[chunk.loc[x.index, 'service_line'] == 'Security'].sum())
        ).reset_index()
        
        # Group upgrades/downgrades
        up_agg = upgrade_rows.groupby(['client_id', 'year_month'])['amount'].sum().reset_index(name='upgrade_amt')
        down_agg = downgrade_rows.groupby(['client_id', 'year_month'])['amount'].sum().reset_index(name='downgrade_amt')
        
        # Merge intermediate tables for this chunk
        from functools import reduce
        dfs = [inv_agg, pay_agg, tkt_agg, use_agg, up_agg, down_agg]
        chunk_agg = reduce(lambda left, right: pd.merge(left, right, on=['client_id', 'year_month'], how='outer'), dfs)
        
        chunk_list.append(chunk_agg)
        print(f"Processed chunk {chunk_idx}: {len(chunk):,} rows aggregated to {len(chunk_agg)} client-months.")

    # 4. Final Aggregation of all chunks
    print("Combining chunked aggregates...")
    panel_df = pd.concat(chunk_list, ignore_index=True)
    
    # Fill NAs in panel prior to grouping
    fill_cols = ['invoiced_amt', 'paid_amt', 'late_pay_count', 'failed_pay_count', 'total_pay_count',
                 'ticket_count', 'escalated_ticket_count', 'total_usage', 'usage_infra', 'usage_cloud', 
                 'usage_security', 'upgrade_amt', 'downgrade_amt']
    panel_df[fill_cols] = panel_df[fill_cols].fillna(0)
    
    # Group by client and month again to sum up chunk overlaps
    panel_df = panel_df.groupby(['client_id', 'year_month']).agg({
        'invoiced_amt': 'sum',
        'paid_amt': 'sum',
        'late_pay_count': 'sum',
        'failed_pay_count': 'sum',
        'total_pay_count': 'sum',
        'ticket_count': 'sum',
        'escalated_ticket_count': 'sum',
        'total_usage': 'sum',
        'usage_infra': 'sum',
        'usage_cloud': 'sum',
        'usage_security': 'sum',
        'upgrade_amt': 'sum',
        'downgrade_amt': 'sum'
    }).reset_index()
    
    elapsed = time.time() - start_time
    print(f"Transactions processing completed. Total rows processed: {total_rows:,}. Panel size: {len(panel_df)} rows. Elapsed: {elapsed:.2f} seconds.")
    return panel_df

def engineer_features(panel_df, clients_df):
    print("\n--- Engineering Features and Creating Model Ready Dataset ---")
    
    # Sort for time-series operations
    panel_df = panel_df.sort_values(['client_id', 'year_month']).reset_index(drop=True)
    
    # Compute rolling metrics grouped by client
    # 3-month rolling averages
    print("Computing rolling trends and ratios...")
    
    # Helper to calculate average of prior 3 months (t-1, t-2, t-3) and t-4, t-5, t-6
    # Let's do this client-by-client or using rolling
    groupby_obj = panel_df.groupby('client_id')
    
    # Invoiced Revenue Trends
    panel_df['invoiced_avg_3mo'] = groupby_obj['invoiced_amt'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    panel_df['invoiced_avg_prior_3mo'] = groupby_obj['invoiced_amt'].transform(lambda x: x.shift(4).rolling(3, min_periods=1).mean())
    panel_df['revenue_trend_3mo'] = panel_df['invoiced_avg_3mo'] / panel_df['invoiced_avg_prior_3mo']
    # Replace infinite or NaN trends with 1.0 (stable)
    panel_df['revenue_trend_3mo'] = panel_df['revenue_trend_3mo'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Usage Trends
    panel_df['usage_avg_3mo'] = groupby_obj['total_usage'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    panel_df['usage_avg_prior_3mo'] = groupby_obj['total_usage'].transform(lambda x: x.shift(4).rolling(3, min_periods=1).mean())
    panel_df['usage_trend_3mo'] = panel_df['usage_avg_3mo'] / panel_df['usage_avg_prior_3mo']
    panel_df['usage_trend_3mo'] = panel_df['usage_trend_3mo'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Late Payment Rate
    panel_df['late_pay_sum_3mo'] = groupby_obj['late_pay_count'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    panel_df['pay_count_sum_3mo'] = groupby_obj['total_pay_count'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    panel_df['late_payment_rate'] = panel_df['late_pay_sum_3mo'] / panel_df['pay_count_sum_3mo']
    panel_df['late_payment_rate'] = panel_df['late_payment_rate'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # SLA Breach Rate (escalated tickets over total support tickets)
    panel_df['esc_ticket_sum_3mo'] = groupby_obj['escalated_ticket_count'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    panel_df['ticket_sum_3mo'] = groupby_obj['ticket_count'].transform(lambda x: x.rolling(3, min_periods=1).sum())
    panel_df['sla_breach_rate'] = panel_df['esc_ticket_sum_3mo'] / panel_df['ticket_sum_3mo']
    panel_df['sla_breach_rate'] = panel_df['sla_breach_rate'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Months Since Last Upgrade
    # Let's write a small helper to count consecutive months with upgrade_amt == 0
    # Or shift cumsum
    panel_df['has_upgrade'] = (panel_df['upgrade_amt'] > 0).astype(int)
    # running cumsum of upgrades to group by
    panel_df['upgrade_group'] = groupby_obj['has_upgrade'].transform(lambda x: x.cumsum())
    # count months within upgrade groups
    panel_df['months_since_last_upgrade'] = panel_df.groupby(['client_id', 'upgrade_group']).cumcount()
    # For months before the first upgrade, set it based on client tenure (approximate)
    # We can just leave it as cumulative count from start of simulation
    
    # Declining Engagement Flag
    # True if usage trend is declining and late payment rate or SLA breach rate is high
    panel_df['is_declining_engagement_flag'] = (
        (panel_df['usage_trend_3mo'] < 0.90) | 
        (panel_df['ticket_count'] > 5)
    ).astype(int)
    
    # Count of services
    client_services_map = clients_df.set_index('client_id')['services_subscribed'].apply(lambda x: len(x.split(';'))).to_dict()
    panel_df['services_count'] = panel_df['client_id'].map(client_services_map)
    
    # Static attributes from clients table
    static_cols = ['client_id', 'client_name', 'industry', 'client_size', 'contract_start_date', 
                   'contract_type', 'account_manager', 'region', 'nps_score', 'is_churned', 'churn_date', 'tenure_months']
    
    model_df = pd.merge(panel_df, clients_df[static_cols], on='client_id', how='left')
    
    # Drop rows that represent months after a client has churned
    # This prevents the model from training on data where the client has already left the company.
    initial_rows = len(model_df)
    model_df['year_month_date'] = pd.to_datetime(model_df['year_month'] + '-01')
    model_df['churn_date_dt'] = pd.to_datetime(model_df['churn_date'])
    
    # Keep row if not churned OR (churned and year_month_date <= churn_date_dt)
    keep_mask = model_df['churn_date_dt'].isna() | (model_df['year_month_date'] <= model_df['churn_date_dt'])
    model_df = model_df[keep_mask].reset_index(drop=True)
    
    # Drop temp date columns
    model_df = model_df.drop(columns=['year_month_date', 'churn_date_dt', 'has_upgrade', 'upgrade_group',
                                       'invoiced_avg_3mo', 'invoiced_avg_prior_3mo', 'usage_avg_3mo', 
                                       'usage_avg_prior_3mo', 'late_pay_sum_3mo', 'pay_count_sum_3mo', 
                                       'esc_ticket_sum_3mo', 'ticket_sum_3mo'])
    
    print(f"Dropped {initial_rows - len(model_df)} rows after clients' churn dates. Final shape: {model_df.shape}")
    
    return model_df

def run_etl():
    workspace_dir = "c:/Users/kasau/Desktop/guptachurn"
    data_dir = os.path.join(workspace_dir, "data")
    clients_path = os.path.join(data_dir, "clients.csv")
    transactions_path = os.path.join(data_dir, "transactions.csv")
    model_ready_path = os.path.join(data_dir, "model_ready_dataset.csv")
    
    print("Loading raw clients data...")
    clients_df = pd.read_csv(clients_path)
    
    # 1. Imputation
    clients_df_clean = impute_clients_data(clients_df)
    clients_df_clean.to_csv(clients_path, index=False) # update file with imputed values
    
    # 2. Aggregation
    panel_df = process_and_aggregate_transactions(transactions_path, clients_df_clean)
    
    # 3. Feature Engineering & Merge
    model_df = engineer_features(panel_df, clients_df_clean)
    
    # Save model ready dataset
    model_df.to_csv(model_ready_path, index=False)
    print(f"Model-ready dataset saved to: {model_ready_path}")
    print("ETL Pipeline completed successfully!")

if __name__ == '__main__':
    run_etl()
