import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_churn_drivers(workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    print("\n--- Running Churn Driver Analysis ---")
    
    models_dir = os.path.join(workspace_dir, "models_saved")
    fig_dir = os.path.join(workspace_dir, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "churn_model.pkl")
    feature_path = os.path.join(models_dir, "feature_cols.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        print("Model or feature columns file not found. Please train the churn model first.")
        return
        
    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)
    
    # Extract importances based on model type
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # If Logistic Regression, use absolute coefficients
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            importances = importances / importances.sum() # normalize
        else:
            print("Model does not support feature importance extraction.")
            return

    # Create Importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    # Rename features for business readability
    feature_rename_dict = {
        'monthly_recurring_revenue': 'Monthly Recurring Revenue ($)',
        'nps_score': 'NPS Score',
        'tenure_months': 'Client Tenure (Months)',
        'services_count': 'Number of Services Subscribed',
        'revenue_trend_3mo': 'Revenue Trend (3 Months)',
        'usage_trend_3mo': 'Product Usage Trend (3 Months)',
        'late_payment_rate': 'Late Payment Rate',
        'sla_breach_rate': 'SLA Breach Rate',
        'months_since_last_upgrade': 'Months Since Last Upgrade',
        'is_declining_engagement_flag': 'Declining Engagement Indicator',
        'ticket_count': 'Support Ticket Volume',
        'escalated_ticket_count': 'Escalated Ticket Volume',
        'total_usage': 'Total Product Usage Volume',
        'usage_infra': 'Infrastructure Usage Volume',
        'usage_cloud': 'Cloud Service Usage Volume',
        'usage_security': 'Security Service Usage Volume',
        'size_num': 'Client Size Tier',
        'contract_type_num': 'Contract Duration Tier',
        'industry_Healthcare': 'Industry: Healthcare',
        'industry_Manufacturing': 'Industry: Manufacturing',
        'industry_Retail': 'Industry: Retail',
        'industry_Technology': 'Industry: Technology',
        'region_East': 'Region: East',
        'region_South': 'Region: South',
        'region_West': 'Region: West'
    }
    
    importance_df['Business Feature'] = importance_df['Feature'].map(feature_rename_dict).fillna(importance_df['Feature'])

    print("\nTop 10 Churn Drivers:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"{idx+1}. {row['Business Feature']}: {row['Importance']:.2%}")

    # Plot
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # Plot top 12 features
    top_n = 12
    sns.barplot(
        data=importance_df.head(top_n),
        x='Importance',
        y='Business Feature',
        hue='Business Feature',
        palette='rocket',
        legend=False
    )
    
    plt.title(f"Top {top_n} Key Drivers of Customer Churn", weight='bold', fontsize=14)
    plt.xlabel("Relative Influence / Feature Importance")
    plt.ylabel("")
    plt.tight_layout()
    
    plot_path = os.path.join(fig_dir, "churn_drivers.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Churn drivers plot saved to: {plot_path}")
    print("Driver Analysis completed successfully!")

if __name__ == '__main__':
    analyze_churn_drivers()
