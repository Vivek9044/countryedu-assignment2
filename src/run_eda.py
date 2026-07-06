import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_figures(workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    print("--- Starting Exploratory Data Analysis (EDA) ---")
    data_dir = os.path.join(workspace_dir, "data")
    fig_dir = os.path.join(workspace_dir, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    clients_df = pd.read_csv(os.path.join(data_dir, "clients.csv"))
    model_df = pd.read_csv(os.path.join(data_dir, "model_ready_dataset.csv"))
    
    # We will use a clean seaborn style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16
    })
    
    # Define cohesive colors
    primary_color = "#1f77b4"
    secondary_color = "#ff7f0e"
    palette = [primary_color, secondary_color]
    
    # ----------------------------------------------------
    # Chart 1: Churn Rate by Business Segments
    # ----------------------------------------------------
    print("Generating Churn Rate by Segments...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Client Churn Rate by Business Segments", weight='bold', y=0.98)
    
    segments = ['industry', 'client_size', 'contract_type', 'region']
    titles = ['Industry', 'Client Size', 'Contract Type', 'Region']
    
    for i, (seg, title) in enumerate(zip(segments, titles)):
        ax = axes[i//2, i%2]
        # Calculate churn rate per segment
        seg_data = clients_df.groupby(seg)['is_churned'].mean().reset_index()
        seg_data['is_churned'] = seg_data['is_churned'] * 100 # percentage
        
        sns.barplot(data=seg_data, x=seg, y='is_churned', ax=ax, hue=seg, palette='Blues_r', legend=False)
        ax.set_title(f"Churn Rate by {title}")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("")
        ax.set_ylim(0, 100)
        
        # Add labels on top of bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(p.get_x() + p.get_width() / 2, height + 2),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "churn_rate_by_segments.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------
    # Chart 2: Revenue Distribution and Time-Series Trend
    # ----------------------------------------------------
    print("Generating Revenue Distribution & Trends...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Revenue Distribution and Historic Trends", weight='bold')
    
    # Left: MRR distribution for churned vs active
    sns.kdeplot(data=clients_df[~clients_df['is_churned']], x='monthly_recurring_revenue', 
                fill=True, label='Active Clients', ax=ax1, color=primary_color)
    sns.kdeplot(data=clients_df[clients_df['is_churned']], x='monthly_recurring_revenue', 
                fill=True, label='Churned Clients', ax=ax1, color=secondary_color)
    ax1.set_title("MRR Distribution: Active vs. Churned")
    ax1.set_xlabel("Monthly Recurring Revenue ($)")
    ax1.set_ylabel("Density")
    ax1.legend()
    
    # Right: Monthly total revenue over 36 months
    monthly_rev = model_df.groupby('year_month')['invoiced_amt'].sum().reset_index()
    monthly_rev['year_month_dt'] = pd.to_datetime(monthly_rev['year_month'] + '-01')
    
    # Also monthly revenue of churned vs active
    monthly_rev_by_status = model_df.groupby(['year_month', 'is_churned'])['invoiced_amt'].sum().reset_index()
    monthly_rev_by_status['year_month_dt'] = pd.to_datetime(monthly_rev_by_status['year_month'] + '-01')
    
    sns.lineplot(data=monthly_rev, x='year_month_dt', y='invoiced_amt', ax=ax2, color='green', marker='o', linewidth=2.5, label='Total Revenue')
    
    ax2.set_title("Historic Monthly Revenue Trend")
    ax2.set_xlabel("Time (Monthly)")
    ax2.set_ylabel("Total Monthly Revenue ($)")
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "revenue_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------
    # Chart 3: Correlation of Service Metrics with Churn
    # ----------------------------------------------------
    print("Generating Service Metrics vs Churn Boxplots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Service & Operational Metrics by Client Churn Status", weight='bold', y=0.98)
    
    metrics = ['sla_breaches_last_90_days', 'avg_ticket_satisfaction', 'product_usage_score', 'payment_delay_days_avg']
    titles = ['SLA Breaches (90 Days)', 'Avg Ticket Satisfaction', 'Product Usage Score (0-100)', 'Avg Payment Delay (Days)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i//2, i%2]
        # Clean labels
        sns.boxplot(data=clients_df, x='is_churned', y=metric, ax=ax, palette=palette, hue='is_churned', legend=False)
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Active", "Churned"])
        
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "service_metrics_vs_churn.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------
    # Chart 4: Cohort and Tenure Analysis
    # ----------------------------------------------------
    print("Generating Cohort/Tenure Analysis...")
    plt.figure(figsize=(10, 6))
    
    # Bin tenure months
    tenure_bins = [0, 6, 12, 18, 24, 30, 48]
    tenure_labels = ['0-6m', '6-12m', '12-18m', '18-24m', '24-30m', '30m+']
    clients_df['tenure_bin'] = pd.cut(clients_df['tenure_months'], bins=tenure_bins, labels=tenure_labels)
    
    tenure_churn = clients_df.groupby('tenure_bin', observed=False)['is_churned'].mean().reset_index()
    tenure_churn['is_churned'] = tenure_churn['is_churned'] * 100
    
    ax = sns.barplot(data=tenure_churn, x='tenure_bin', y='is_churned', palette='Oranges_r', hue='tenure_bin', legend=False)
    plt.title("Churn Rate by Client Contract Age / Tenure", weight='bold')
    plt.xlabel("Client Tenure Segment")
    plt.ylabel("Churn Rate (%)")
    plt.ylim(0, 100)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(p.get_x() + p.get_width() / 2, height + 2),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, weight='bold')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "cohort_tenure_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------
    # Chart 5: Revenue Concentration (Lorenz Curve / Whale Risk)
    # ----------------------------------------------------
    print("Generating Revenue Concentration (Whale Risk)...")
    plt.figure(figsize=(8, 8))
    
    # Sort MRR descending
    mrr_sorted = clients_df['monthly_recurring_revenue'].dropna().sort_values(ascending=True).values
    n_clients = len(mrr_sorted)
    
    # Cumulative percentages
    cum_clients = np.arange(1, n_clients + 1) / n_clients * 100
    cum_revenue = np.cumsum(mrr_sorted) / mrr_sorted.sum() * 100
    
    # Perfect equality line
    plt.plot([0, 100], [0, 100], 'k--', label='Perfect Equality (Baseline)')
    
    # Lorenz curve
    plt.plot(cum_clients, cum_revenue, 'b-', linewidth=2.5, label='Client Revenue Distribution')
    
    # Calculate Gini Coefficient
    # Gini = 1 - 2 * area under Lorenz Curve
    # Manual trapezoidal integration to support NumPy 2.0+
    x = cum_clients / 100
    y = cum_revenue / 100
    area = np.sum((y[:-1] + y[1:]) * np.diff(x) / 2)
    gini = 1 - 2 * area
    
    # Calculate concentration metric (e.g. revenue from top 10% clients)
    # Since sorted is ascending, top 10% is the last 10%
    top_10_index = int(n_clients * 0.9)
    rev_top_10_percent = (mrr_sorted[top_10_index:].sum() / mrr_sorted.sum()) * 100
    
    plt.fill_between(cum_clients, cum_clients, cum_revenue, color='lightblue', alpha=0.4)
    
    # Highlight points
    plt.scatter([90], [100 - rev_top_10_percent], color='red', s=80, zorder=5)
    plt.annotate(f"Top 10% of Clients\nAccount for {rev_top_10_percent:.1f}% of Revenue",
                 xy=(90, 100 - rev_top_10_percent), xytext=(40, 25),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=11, weight='bold', color='red')
                 
    plt.title(f"Client Revenue Concentration (Lorenz Curve)\nGini Coefficient: {gini:.3f} | 'Whale' Risk Profile", weight='bold')
    plt.xlabel("Cumulative Percentage of Clients (%)")
    plt.ylabel("Cumulative Percentage of Total MRR (%)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "revenue_concentration.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------
    # Chart 6: Heatmap of Features vs Churn
    # ----------------------------------------------------
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    
    # Select numeric columns for correlation from clients
    # Map is_churned to int
    clients_numeric = clients_df.copy()
    clients_numeric['is_churned_numeric'] = clients_numeric['is_churned'].astype(int)
    
    # Clean string objects
    # Categorical variables encoding for heatmap correlation
    clients_numeric['contract_type_num'] = clients_numeric['contract_type'].map({'Monthly': 1, 'Annual': 2, 'Multi-year': 3})
    clients_numeric['size_num'] = clients_numeric['client_size'].map({'Small': 1, 'Medium': 2, 'Enterprise': 3})
    
    cols_to_corr = [
        'is_churned_numeric',
        'monthly_recurring_revenue',
        'tenure_months',
        'support_tickets_last_90_days',
        'avg_ticket_satisfaction',
        'sla_breaches_last_90_days',
        'payment_delay_days_avg',
        'product_usage_score',
        'nps_score',
        'contract_type_num',
        'size_num'
    ]
    
    corr_matrix = clients_numeric[cols_to_corr].corr()
    
    # Rename for readability
    rename_dict = {
        'is_churned_numeric': 'Is Churned',
        'monthly_recurring_revenue': 'MRR ($)',
        'tenure_months': 'Tenure (Months)',
        'support_tickets_last_90_days': 'Support Tickets',
        'avg_ticket_satisfaction': 'Ticket Satisfaction',
        'sla_breaches_last_90_days': 'SLA Breaches',
        'payment_delay_days_avg': 'Payment Delay (Days)',
        'product_usage_score': 'Product Usage Score',
        'nps_score': 'NPS Score',
        'contract_type_num': 'Contract Duration Tier',
        'size_num': 'Client Size Tier'
    }
    
    corr_matrix = corr_matrix.rename(index=rename_dict, columns=rename_dict)
    
    # Generate heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm', fmt=".2f", vmin=-1.0, vmax=1.0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
                
    plt.title("Correlation Matrix of Client Characteristics & Churn Status", weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "correlation_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("EDA Visualizations exported successfully to: " + fig_dir)
    print("EDA Pipeline complete!")

if __name__ == '__main__':
    generate_eda_figures()
