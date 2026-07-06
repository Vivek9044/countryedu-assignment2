import os
import sys
import pandas as pd
import streamlit as st

# Add src folder to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.strategy_simulator import simulate_intervention
from src.recommendations import generate_recommendations, get_retention_strategy_impact

# App Configuration
st.set_page_config(
    page_title="Managed IT Services Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .reportview-container { background: #f8f9fa; }
    h1, h2, h3 { color: #1e3d59; font-family: 'Inter', sans-serif; }
    .stMetric { background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e9ecef; }
    .stAlert { border-radius: 8px; }
    .css-1r6g72h { font-weight: bold; }
    .risk-high { background-color: #f8d7da; color: #721c24; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
    .risk-medium { background-color: #fff3cd; color: #856404; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
    .risk-low { background-color: #d4edda; color: #155724; padding: 5px 10px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Data Paths
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(workspace_dir, "data")
reports_dir = os.path.join(workspace_dir, "reports")
figures_dir = os.path.join(reports_dir, "figures")

# Load Datasets
@st.cache_data
def load_data():
    clients = pd.read_csv(os.path.join(data_dir, "clients.csv"))
    ranking = pd.read_csv(os.path.join(reports_dir, "churn_risk_ranking.csv"))
    forecast = pd.read_csv(os.path.join(reports_dir, "revenue_forecast_12m.csv"))
    return clients, ranking, forecast

clients_df, ranking_df, forecast_df = load_data()
active_clients = ranking_df[~ranking_df['is_churned']]

# Title Header
st.title("💼 Client Churn Risk & Revenue Forecasting Dashboard")
st.subheader("Managed IT Services Division — Portofolio Health and Risk Intelligence")

# Sidebar navigation
st.sidebar.header("Navigation Menu")
page = st.sidebar.radio("Go to Page", [
    "Executive Overview", 
    "Client Risk Ledger", 
    "Churn Driver Analysis", 
    "Revenue Forecasting", 
    "Strategy Simulator", 
    "Actionable Recommendations"
])

# Summary Stats calculations
total_clients = len(clients_df)
active_count = len(active_clients)
historical_churned = total_clients - active_count
historical_churn_pct = (historical_churned / total_clients) * 100

active_avg_risk = active_clients['predicted_churn_probability'].mean() * 100
total_active_mrr = active_clients['monthly_recurring_revenue'].sum()
mrr_at_risk = (active_clients['monthly_recurring_revenue'] * active_clients['predicted_churn_probability']).sum()

next_12m_rev = forecast_df['churn_adjusted_forecast'].sum()

# ----------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# ----------------------------------------------------
if page == "Executive Overview":
    st.header("📈 Portfolio Performance & Churn Health")
    
    # KPI metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Active Clients", f"{active_count} / {total_clients}", f"-{historical_churned} historical churn")
    with col2:
        st.metric("Active Portfolio MRR", f"${total_active_mrr:,.2f}", "Current Monthly Billings")
    with col3:
        st.metric("Active Churn Risk Score", f"{active_avg_risk:.2f}%", "Avg Churn Prob")
    with col4:
        st.metric("Expected Revenue At Risk", f"${mrr_at_risk:,.2f}", f"{(mrr_at_risk/total_active_mrr)*100:.1f}% of MRR")
        
    st.write("---")
    
    # Layout sections
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### Historic Revenue vs. Forecast Horizon (12 Months)")
        forecast_img = os.path.join(figures_dir, "revenue_forecast.png")
        if os.path.exists(forecast_img):
            st.image(forecast_img, use_container_width=True)
        else:
            st.write("Forecast visualization not found. Please run the forecasting script.")
            
    with col_right:
        st.markdown("### Client Risk Segmentation")
        high_risk_count = sum(active_clients['risk_tier'] == 'High')
        med_risk_count = sum(active_clients['risk_tier'] == 'Medium')
        low_risk_count = sum(active_clients['risk_tier'] == 'Low')
        
        st.markdown(f"""
        - **🔴 High Risk Tier (Prob ≥ 70%)**: **{high_risk_count}** accounts
        - **🟡 Medium Risk Tier (30% - 70%)**: **{med_risk_count}** accounts
        - **🟢 Low Risk Tier (Prob < 30%)**: **{low_risk_count}** accounts
        """)
        
        # Simple color panel summary
        st.info("💡 **Executive Takeaway**: Operational efficiency metrics indicate that reducing service ticket escalation rates by even 20% and migrating at-risk clients from Monthly to Annual contract terms will preserve substantial billing revenue.")
        
        # Display small alert if high risk clients exist
        if high_risk_count > 0:
            st.error(f"⚠️ **Attention Required**: There are currently **{high_risk_count}** clients marked as **High Churn Risk**. Review the Recommendations tab to trigger account manager retention workflows immediately.")

# ----------------------------------------------------
# PAGE 2: CLIENT RISK LEDGER
# ----------------------------------------------------
elif page == "Client Risk Ledger":
    st.header("📋 Client Risk ledger & Risk Tiering")
    st.write("Use this table to search, filter, and sort the client roster by account risk probability and owner.")
    
    # Filter controls
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        search_query = st.text_input("Search Client Name or ID", "")
    with col_f2:
        tier_filter = st.multiselect("Filter by Risk Tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    with col_f3:
        am_filter = st.multiselect("Filter by Account Manager", sorted(clients_df['account_manager'].unique()))

    # Apply filters
    filtered_df = active_clients.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df['client_name'].str.contains(search_query, case=False) |
            filtered_df['client_id'].str.contains(search_query, case=False)
        ]
    if tier_filter:
        filtered_df = filtered_df[filtered_df['risk_tier'].isin(tier_filter)]
    if am_filter:
        filtered_df = filtered_df[filtered_df['account_manager'].isin(am_filter)]
        
    # Formatting columns
    display_df = filtered_df[[
        'client_id', 'client_name', 'account_manager', 'contract_type', 
        'monthly_recurring_revenue', 'predicted_churn_probability', 'risk_tier'
    ]].copy()
    
    display_df['predicted_churn_probability'] = display_df['predicted_churn_probability'].apply(lambda x: f"{x:.1%}")
    display_df['monthly_recurring_revenue'] = display_df['monthly_recurring_revenue'].apply(lambda x: f"${x:,.2f}")
    
    st.dataframe(display_df.rename(columns={
        'client_id': 'Client ID',
        'client_name': 'Client Name',
        'account_manager': 'Account Manager',
        'contract_type': 'Contract Type',
        'monthly_recurring_revenue': 'MRR ($)',
        'predicted_churn_probability': 'Churn Probability',
        'risk_tier': 'Risk Tier'
    }), use_container_width=True, hide_index=True)

# ----------------------------------------------------
# PAGE 3: CHURN DRIVER ANALYSIS
# ----------------------------------------------------
elif page == "Churn Driver Analysis":
    st.header("🔍 Core Churn Driver Analysis")
    st.write("Understand the key operational variables and customer service triggers that are statistically driving client churn.")
    
    col_plot, col_text = st.columns([3, 2])
    
    with col_plot:
        drivers_img = os.path.join(figures_dir, "churn_drivers.png")
        if os.path.exists(drivers_img):
            st.image(drivers_img, use_container_width=True)
        else:
            st.write("Drivers visualization not found. Please run the driver analysis script.")
            
    with col_text:
        st.subheader("Key Findings & Business Interpretation")
        st.markdown("""
        1. **Total Product Usage Volume (20.3% Influence)**: 
           This is the single strongest indicator. Clients whose bandwidth and usage drop below typical baselines show a massive statistical spike in churn probability (dormancy precedes churn).
        
        2. **Months Since Last Upgrade (12.1% Influence)**:
           Clients who remain stagnant without upgrading services or taking on new cloud workloads for >12 months are significantly more susceptible to competitor poaching.
           
        3. **Revenue Trend (8.9% Influence)**:
           A declining billing trend (e.g. cloud downsizing) is a strong leading indicator of contract terminations.
           
        4. **NPS detractor score (4.2% Influence)**:
           Clients scoring in the detractor zone (<0) show a clear correlation with churn. Detractors should receive proactive outreach from executive leadership.
           
        5. **SLA Breach Rate (3.8% Influence)**:
           More than 2 SLA breaches in 90 days results in client frustration, lowering ticket satisfaction scores and driving contract downgrades.
        """)

# ----------------------------------------------------
# PAGE 4: REVENUE FORECASTING
# ----------------------------------------------------
elif page == "Revenue Forecasting":
    st.header("🔮 12-Month Revenue Forecasting")
    st.write("This tab outlines future revenue trends comparing our **Baseline Forecast** (assuming past churn rates continue) against the **Churn-Adjusted Forecast** (adjusting for clients currently at risk of churning).")
    
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        forecast_img = os.path.join(figures_dir, "revenue_forecast.png")
        if os.path.exists(forecast_img):
            st.image(forecast_img, use_container_width=True)
        else:
            st.write("Forecast chart not found.")
            
    with col_table:
        st.markdown("### Monthly Projection Summary")
        disp_forecast = forecast_df.copy()
        
        # format currencies
        for col in ['baseline_forecast', 'ci_lower', 'ci_upper', 'churn_adjusted_forecast', 'expected_mrr_loss']:
            disp_forecast[col] = disp_forecast[col].apply(lambda x: f"${x:,.2f}")
            
        st.dataframe(disp_forecast.rename(columns={
            'year_month': 'Month',
            'baseline_forecast': 'Baseline ($)',
            'ci_lower': 'Lower CI ($)',
            'ci_upper': 'Upper CI ($)',
            'churn_adjusted_forecast': 'Churn-Adjusted ($)',
            'expected_mrr_loss': 'Expected Loss ($)'
        }), height=450, hide_index=True)
        
    st.warning("⚠️ **Forecast Risk Warning**: Expected MRR loss starts compounding in Month 2 and peaks at **${:,.2f}/month** in Month 4. Proactive mitigation is critical to protect this revenue stream.".format(forecast_df['expected_mrr_loss'].max()))

# ----------------------------------------------------
# PAGE 5: STRATEGY SIMULATOR
# ----------------------------------------------------
elif page == "Strategy Simulator":
    st.header("🎮 Churn Mitigation Strategy Simulator")
    st.write("Interact with the sliders below to simulate operational interventions (e.g. reducing technical issues, improving customer happiness) and calculate the projected reduction in customer churn and revenue saved.")
    
    # Simulator inputs in columns
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        sla_red = st.slider("Operational SLA Breach Reduction (%)", 0, 100, 0, step=5,
                            help="Reduces support ticket escalations and SLA breaches through proactive operations management.")
    with col_s2:
        usage_boost = st.slider("Product Engagement / Usage Boost (%)", 0, 100, 0, step=5,
                               help="Boosts client product engagement and cloud usage metrics through client enablement training.")
    with col_s3:
        nps_boost = st.slider("NPS Detractor Mitigation (Points)", 0, 50, 0, step=5,
                              help="Direct executive intervention to resolve client issues, increasing their Net Promoter Score.")
                              
    # Run simulation
    sim_results = simulate_intervention(sla_red, usage_boost, nps_boost, workspace_dir)
    
    st.write("---")
    
    if sim_results:
        # Results metrics row
        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            baseline_prob_pct = sim_results['base_avg_prob'] * 100
            new_prob_pct = sim_results['mod_avg_prob'] * 100
            st.metric("New Active Churn Rate", f"{new_prob_pct:.2f}%", 
                      f"-{baseline_prob_pct - new_prob_pct:.2f}% from baseline ({baseline_prob_pct:.2f}%)")
        with r_col2:
            st.metric("Expected Monthly Billing Saved", f"${sim_results['mrr_retained']:,.2f}", 
                      f"Expected Loss: ${sim_results['mod_expected_mrr_loss']:,.2f}")
        with r_col3:
            at_risk_saved = sim_results['base_at_risk_count'] - sim_results['mod_at_risk_count']
            st.metric("Clients Rescued from Churn Risk", f"{at_risk_saved} accounts", 
                      f"Remaining At-Risk: {sim_results['mod_at_risk_count']}")
            
        st.write("---")
        
        # Check target progress
        rel_reduction = sim_results['churn_rate_reduction_pct']
        # Progress towards relative 5% target (e.g. if we reduced active rate by 20% relative to starting)
        start_prob = sim_results['base_avg_prob']
        end_prob = sim_results['mod_avg_prob']
        relative_reduction = (start_prob - end_prob) / start_prob * 100 if start_prob > 0 else 0
        
        st.subheader("Progress Toward Churn Reduction Target (Relative Churn Reduction)")
        
        # Display progress bar
        progress_val = min(1.0, max(0.0, relative_reduction / 5.0)) # scale relative to 5% target
        st.progress(progress_val)
        
        if relative_reduction >= 5.0:
            st.success(f"🎉 **Target Met!** The simulated interventions achieve a **{relative_reduction:.2f}%** relative reduction in the active churn rate, exceeding the **5.00%** target. This saves **${sim_results['mrr_retained'] * 12:,.2f}/year** in recurring contract value!")
        else:
            st.warning(f"⚠️ **Target Not Met yet**: Currently achieving a **{relative_reduction:.2f}%** relative reduction in active churn risk. Adjust the sliders (e.g. increase product usage or resolve Detractor NPS) to reach the **5.00%** business target.")

# ----------------------------------------------------
# PAGE 6: ACTIONABLE RECOMMENDATIONS
# ----------------------------------------------------
elif page == "Actionable Recommendations":
    st.header("🎯 Prioritized Client Retention Recommendations")
    st.write("Prioritized, quantified list of retention actions for at-risk accounts based on dominant client service triggers.")
    
    # Load impact summary
    impact = get_retention_strategy_impact(workspace_dir)
    
    # Impact metrics row
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.metric("Target Accounts Identified", f"{impact['total_clients_targeted']} clients")
    with col_i2:
        st.metric("Total Monthly billing At-Risk", f"${impact['total_mrr_at_risk']:,.2f}")
    with col_i3:
        st.metric("Estimated Saved Revenue", f"${impact['projected_mrr_saved']:,.2f} / month", "60% mitigation success")
        
    st.write("---")
    
    # Table of alerts
    alerts = generate_recommendations(workspace_dir)
    
    if alerts:
        alerts_df = pd.DataFrame(alerts)
        
        # Display styled cards for the top 5 highest priority alerts
        st.markdown("### Top Priority Retention Alert Cards")
        for i, row in alerts_df.head(5).iterrows():
            with st.container():
                st.markdown(f"#### Priority {row['priority']}: {row['client_name']} ({row['client_id']})")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(f"**Risk Level**: `{row['risk_tier']}` (Probability: {row['predicted_probability']:.1%})")
                with c2:
                    st.write(f"**Contract MRR**: ${row['monthly_recurring_revenue']:,.2f}")
                with c3:
                    st.write(f"**Owner**: {row['account_manager']} | Contract: {row['contract_type']}")
                
                st.info(f"🚨 **Issue Identified**: {row['primary_issue']}  \n🎯 **Recommended Action**: {row['action_item']}")
                st.markdown("---")
                
        # Full table below
        st.markdown("### Full Actionable Account Alerts List")
        table_df = alerts_df[['priority', 'client_id', 'client_name', 'predicted_probability', 'risk_tier', 'account_manager', 'primary_issue', 'action_item']].copy()
        table_df['predicted_probability'] = table_df['predicted_probability'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(table_df.rename(columns={
            'priority': 'Priority',
            'client_id': 'Client ID',
            'client_name': 'Client Name',
            'predicted_probability': 'Churn Prob',
            'risk_tier': 'Risk Tier',
            'account_manager': 'Account Manager',
            'primary_issue': 'Identified Trigger',
            'action_item': 'Recommended Retention Action'
        }), use_container_width=True, hide_index=True)
    else:
        st.success("No active clients exceed the risk threshold (Prob ≥ 30%). Portfolio is in a healthy state!")
