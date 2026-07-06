import os
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, confusion_matrix

def prepare_data(workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    data_dir = os.path.join(workspace_dir, "data")
    model_df = pd.read_csv(os.path.join(data_dir, "model_ready_dataset.csv"))
    clients_df = pd.read_csv(os.path.join(data_dir, "clients.csv"))
    
    # Map client static monthly_recurring_revenue into the panel data
    client_mrr_map = clients_df.set_index('client_id')['monthly_recurring_revenue'].to_dict()
    model_df['monthly_recurring_revenue'] = model_df['client_id'].map(client_mrr_map)

    # 1. Define Target Variable: Churn in the next 3 months
    model_df['year_month_dt'] = pd.to_datetime(model_df['year_month'] + '-01')
    model_df['churn_date_dt'] = pd.to_datetime(model_df['churn_date'])
    
    # Calculate months to churn
    model_df['months_to_churn'] = (model_df['churn_date_dt'].dt.year - model_df['year_month_dt'].dt.year) * 12 + (model_df['churn_date_dt'].dt.month - model_df['year_month_dt'].dt.month)
    
    # target_churn_3m = 1 if is_churned and months_to_churn is between 0 and 3 months
    model_df['target_churn_3m'] = ((model_df['is_churned']) & 
                                   (model_df['months_to_churn'] >= 0) & 
                                   (model_df['months_to_churn'] <= 3)).astype(int)
    
    # Drop helper date columns
    model_df = model_df.drop(columns=['year_month_dt', 'churn_date_dt', 'months_to_churn'])
    
    # 2. Encode Categoricals
    size_map = {'Small': 1, 'Medium': 2, 'Enterprise': 3}
    model_df['size_num'] = model_df['client_size'].map(size_map)
    
    contract_map = {'Monthly': 1, 'Annual': 2, 'Multi-year': 3}
    model_df['contract_type_num'] = model_df['contract_type'].map(contract_map)
    
    # Encode industry & region using get_dummies
    categorical_cols = ['industry', 'region']
    model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Select Features for training
    feature_cols = [
        'monthly_recurring_revenue', 'nps_score', 'tenure_months', 'services_count',
        'revenue_trend_3mo', 'usage_trend_3mo', 'late_payment_rate', 'sla_breach_rate',
        'months_since_last_upgrade', 'is_declining_engagement_flag', 'ticket_count',
        'escalated_ticket_count', 'total_usage', 'usage_infra', 'usage_cloud', 'usage_security',
        'size_num', 'contract_type_num',
        'industry_Healthcare', 'industry_Manufacturing', 'industry_Retail', 'industry_Technology',
        'region_North', 'region_South', 'region_West'
    ]
    
    # Align features and targets
    X = model_df[feature_cols]
    y = model_df['target_churn_3m']
    groups = model_df['client_id']
    
    return X, y, groups, feature_cols, model_df, clients_df

def train_and_evaluate(X, y, groups, feature_cols, workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    models_dir = os.path.join(workspace_dir, "models_saved")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Group-based Train/Test Split (split by client_id to prevent leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"\nTrain set shape: {X_train.shape} | Positive class (churn in 3m): {y_train.sum()} ({y_train.mean():.2%})")
    print(f"Test set shape: {X_test.shape} | Positive class (churn in 3m): {y_test.sum()} ({y_test.mean():.2%})")
    
    # 2. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Handle Imbalance using SMOTE on training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"Balanced Train set shape: {X_train_res.shape} | Positive class: {y_train_res.sum()} ({y_train_res.mean():.2%})")
    
    # 4. Train Models
    # Model A: Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_res, y_train_res)
    
    # Model B: Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train_res, y_train_res)
    
    # Model C: XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss')
    xgb.fit(X_train_res, y_train_res)
    
    # 5. Evaluate on Test Set
    models = {'Logistic Regression': lr, 'Random Forest': rf, 'XGBoost': xgb}
    comparison_data = []
    
    print("\n" + "="*60)
    print("MODEL COMPARISON ON HELD-OUT CLIENTS TEST SET")
    print("="*60)
    
    best_auc = 0.0
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Classification report (dictionary)
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        f1_macro = rep['macro avg']['f1-score']
        precision_churn = rep['1']['precision']
        recall_churn = rep['1']['recall']
        f1_churn = rep['1']['f1-score']
        
        print(f"\nModel: {name}")
        print(f"ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f}")
        print(f"F1 Churn (Class 1): {f1_churn:.3f} (Precision: {precision_churn:.3f}, Recall: {recall_churn:.3f})")
        print(f"Confusion Matrix:\n{cm}")
        
        comparison_data.append({
            'Model': name,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'Precision Churn': precision_churn,
            'Recall Churn': recall_churn,
            'F1 Churn': f1_churn,
            'F1 Macro': f1_macro,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        })
        
        # Track best model based on PR-AUC (due to class imbalance) or ROC-AUC
        if pr_auc > best_auc:
            best_auc = pr_auc
            best_model_name = name
            best_model = model

    comparison_df = pd.DataFrame(comparison_data)
    
    print(f"\nBest Model selected: {best_model_name} (PR-AUC: {best_auc:.3f})")
    
    # Save the best model, scaler, and features list
    model_save_path = os.path.join(models_dir, "churn_model.pkl")
    scaler_save_path = os.path.join(models_dir, "scaler.pkl")
    
    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_cols.pkl"))
    
    print(f"Saved best model to: {model_save_path}")
    print(f"Saved scaler to: {scaler_save_path}")
    
    return best_model, scaler, best_model_name, comparison_df

def generate_risk_ranking(best_model, scaler, feature_cols, X, model_df, clients_df, workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    # Score all 200 clients based on their LATEST monthly feature values
    print("\nScoring all clients to generate Churn Risk Ranking...")
    
    # Find the latest year_month for each client in model_df
    # In model_df, clients might have different end dates due to churn, but active ones will end in 2026-06
    latest_rows = model_df.loc[model_df.groupby('client_id')['year_month'].idxmax()]
    
    X_latest = latest_rows[feature_cols]
    X_latest_scaled = scaler.transform(X_latest)
    
    # Predict probabilities
    probs = best_model.predict_proba(X_latest_scaled)[:, 1]
    
    latest_rows = latest_rows.copy()
    latest_rows['predicted_churn_probability'] = probs
    
    # Assign Risk Tiers
    def get_risk_tier(p):
        if p >= 0.70:
            return "High"
        elif p >= 0.30:
            return "Medium"
        else:
            return "Low"
            
    latest_rows['risk_tier'] = latest_rows['predicted_churn_probability'].apply(get_risk_tier)
    
    # Merge with original clients table to get account manager and other profile details
    risk_ranking = latest_rows[['client_id', 'predicted_churn_probability', 'risk_tier']].copy()
    
    final_ranking = pd.merge(
        clients_df[['client_id', 'client_name', 'account_manager', 'monthly_recurring_revenue', 'contract_type', 'is_churned']],
        risk_ranking,
        on='client_id',
        how='left'
    )
    
    # For clients who ALREADY churned, set probability to 1.0 and tier to High (or preserve model prediction, but mark is_churned)
    # The assignment says "ranked churn-risk list of all 200 clients with predicted probability".
    # For clients already churned historically, they are already churned.
    # To keep it consistent, we can set their probability to 1.0 or let the model score them (their latest row before churn represents their churn state).
    # Setting pre-churned clients' risk to 1.0 makes business sense, or we can just keep the model scored probability.
    # Let's let the model score them based on their final active month features, and print the ranking.
    # To be professional, we sort by predicted probability in descending order.
    
    final_ranking = final_ranking.sort_values('predicted_churn_probability', ascending=False).reset_index(drop=True)
    
    # Save the ranking csv
    reports_dir = os.path.join(workspace_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    ranking_path = os.path.join(reports_dir, "churn_risk_ranking.csv")
    
    final_ranking.to_csv(ranking_path, index=False)
    print(f"Risk rankings saved to: {ranking_path}")
    print(f"High risk clients: {sum(final_ranking['risk_tier']=='High')}")
    print(f"Medium risk clients: {sum(final_ranking['risk_tier']=='Medium')}")
    print(f"Low risk clients: {sum(final_ranking['risk_tier']=='Low')}")

def run_ml_pipeline():
    workspace_dir = "c:/Users/kasau/Desktop/guptachurn"
    X, y, groups, feature_cols, model_df, clients_df = prepare_data(workspace_dir)
    best_model, scaler, model_name, comparison_df = train_and_evaluate(X, y, groups, feature_cols, workspace_dir)
    generate_ranking_flag = True
    if generate_ranking_flag:
        generate_risk_ranking(best_model, scaler, feature_cols, X, model_df, clients_df, workspace_dir)
        
if __name__ == '__main__':
    run_ml_pipeline()
