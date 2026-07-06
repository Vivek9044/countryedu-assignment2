import os
import joblib
import numpy as np
import pandas as pd

def load_prediction_pipeline(workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    models_dir = os.path.join(workspace_dir, "models_saved")
    
    model = joblib.load(os.path.join(models_dir, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(models_dir, "feature_cols.pkl"))
    
    return model, scaler, feature_cols

def get_latest_active_data(workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    data_dir = os.path.join(workspace_dir, "data")
    model_df = pd.read_csv(os.path.join(data_dir, "model_ready_dataset.csv"))
    clients_df = pd.read_csv(os.path.join(data_dir, "clients.csv"))
    
    client_mrr_map = clients_df.set_index('client_id')['monthly_recurring_revenue'].to_dict()
    model_df['monthly_recurring_revenue'] = model_df['client_id'].map(client_mrr_map)
    
    # Map product_usage_score to panel
    client_usage_map = clients_df.set_index('client_id')['product_usage_score'].to_dict()
    model_df['product_usage_score'] = model_df['client_id'].map(client_usage_map)
    
    # Get only active clients (is_churned = False in static profile)
    active_client_ids = clients_df.loc[~clients_df['is_churned'], 'client_id'].values
    active_panel = model_df[model_df['client_id'].isin(active_client_ids)]
    
    # Get the latest row for each active client
    latest_active = active_panel.loc[active_panel.groupby('client_id')['year_month'].idxmax()].copy()
    
    # Add dummy columns for industry & region if not present
    # Map size and contract type
    size_map = {'Small': 1, 'Medium': 2, 'Enterprise': 3}
    latest_active['size_num'] = latest_active['client_size'].map(size_map)
    
    contract_map = {'Monthly': 1, 'Annual': 2, 'Multi-year': 3}
    latest_active['contract_type_num'] = latest_active['contract_type'].map(contract_map)
    
    # Dynamic dummy columns
    industries = ['Healthcare', 'Manufacturing', 'Retail', 'Technology']
    regions = ['North', 'South', 'West']
    
    for ind in industries:
        col = f'industry_{ind}'
        if col not in latest_active.columns:
            latest_active[col] = (latest_active['industry'] == ind).astype(int)
            
    for reg in regions:
        col = f'region_{reg}'
        if col not in latest_active.columns:
            latest_active[col] = (latest_active['region'] == reg).astype(int)
            
    return latest_active

def simulate_intervention(reduction_sla_pct, increase_usage_pct, increase_nps_points, workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    model, scaler, feature_cols = load_prediction_pipeline(workspace_dir)
    latest_active = get_latest_active_data(workspace_dir)
    
    if len(latest_active) == 0:
        return {}
        
    # Baseline predictions (before intervention)
    X_base = latest_active[feature_cols]
    X_base_scaled = scaler.transform(X_base)
    base_probs = model.predict_proba(X_base_scaled)[:, 1]
    
    # Modified features (apply interventions)
    latest_modified = latest_active.copy()
    
    # 1. Reduce SLA breach rate
    # reduction_sla_pct is 0 to 100
    sla_scale = 1.0 - (reduction_sla_pct / 100.0)
    latest_modified['sla_breach_rate'] = latest_modified['sla_breach_rate'] * sla_scale
    latest_modified['escalated_ticket_count'] = np.round(latest_modified['escalated_ticket_count'] * sla_scale)
    
    # 2. Increase product usage
    # increase_usage_pct is 0 to 100
    usage_scale = 1.0 + (increase_usage_pct / 100.0)
    latest_modified['product_usage_score'] = np.clip(latest_modified['product_usage_score'] * usage_scale, 0, 100)
    latest_modified['total_usage'] = latest_modified['total_usage'] * usage_scale
    latest_modified['usage_infra'] = latest_modified['usage_infra'] * usage_scale
    latest_modified['usage_cloud'] = latest_modified['usage_cloud'] * usage_scale
    latest_modified['usage_security'] = latest_modified['usage_security'] * usage_scale
    
    # 3. Increase NPS score
    # increase_nps_points is absolute points, NPS is -100 to 100
    latest_modified['nps_score'] = np.clip(latest_modified['nps_score'] + increase_nps_points, -100, 100)
    
    # If usage trend is improved and ticket counts decrease, we might unset declining engagement
    # For simplicity, if usage increases by > 10% and SLA breaches decrease by > 10%, we can reduce the declining flag
    if increase_usage_pct >= 10.0 or reduction_sla_pct >= 10.0:
        latest_modified.loc[latest_modified['usage_trend_3mo'] < 1.0, 'usage_trend_3mo'] = latest_modified['usage_trend_3mo'] * 1.10
        latest_modified.loc[latest_modified['usage_trend_3mo'] > 0.95, 'is_declining_engagement_flag'] = 0

    # Predictions after intervention
    X_mod = latest_modified[feature_cols]
    X_mod_scaled = scaler.transform(X_mod)
    mod_probs = model.predict_proba(X_mod_scaled)[:, 1]
    
    # Calculate aggregate business metrics
    mrr = latest_active['monthly_recurring_revenue'].values
    
    # Expected MRR loss: Sum of (MRR_i * P_i)
    base_expected_loss = np.sum(mrr * base_probs)
    mod_expected_loss = np.sum(mrr * mod_probs)
    mrr_retained = base_expected_loss - mod_expected_loss
    
    # Count of at-risk clients (P > 0.5)
    base_at_risk = np.sum(base_probs >= 0.5)
    mod_at_risk = np.sum(mod_probs >= 0.5)
    
    # Average probabilities
    base_avg_prob = np.mean(base_probs)
    mod_avg_prob = np.mean(mod_probs)
    
    # Count of active clients
    n_active = len(latest_active)
    
    # Project Churn reduction
    # Current active churn rate forecast (avg probability)
    # Target churn reduction: 5% reduction in overall churn rate
    # E.g., if baseline average prob is 18%, we want it to go down to 13%
    churn_rate_reduction = base_avg_prob - mod_avg_prob
    
    return {
        'num_active_clients': n_active,
        'base_avg_prob': base_avg_prob,
        'mod_avg_prob': mod_avg_prob,
        'churn_rate_reduction_pct': churn_rate_reduction * 100,
        'base_expected_mrr_loss': base_expected_loss,
        'mod_expected_mrr_loss': mod_expected_loss,
        'mrr_retained': mrr_retained,
        'base_at_risk_count': int(base_at_risk),
        'mod_at_risk_count': int(mod_at_risk)
    }

if __name__ == '__main__':
    # Test simulation
    res = simulate_intervention(30.0, 15.0, 20.0)
    print("\nSimulation Results for Intervention (30% SLA reduction, 15% Usage boost, +20 NPS points):")
    for k, v in res.items():
        if 'loss' in k or 'retained' in k:
            print(f"- {k}: ${v:,.2f}")
        elif 'prob' in k or 'reduction' in k:
            print(f"- {k}: {v:.2%}" if 'prob' in k else f"- {k}: {v:.2f}%")
        else:
            print(f"- {k}: {v}")
