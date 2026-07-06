import os
import pandas as pd

def generate_recommendations(workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    data_dir = os.path.join(workspace_dir, "data")
    reports_dir = os.path.join(workspace_dir, "reports")
    
    ranking_path = os.path.join(reports_dir, "churn_risk_ranking.csv")
    model_ready_path = os.path.join(data_dir, "model_ready_dataset.csv")
    
    if not os.path.exists(ranking_path) or not os.path.exists(model_ready_path):
        return []
        
    ranking_df = pd.read_csv(ranking_path)
    model_df = pd.read_csv(model_ready_path)
    
    # Filter to active clients
    active_ranking = ranking_df[~ranking_df['is_churned']].copy()
    
    # Get latest features for active clients to diagnose issues
    latest_rows = model_df.loc[model_df.groupby('client_id')['year_month'].idxmax()]
    latest_features = latest_rows.set_index('client_id')
    
    recommendations_list = []
    
    # Sort active clients by predicted churn probability descending
    active_ranking = active_ranking.sort_values('predicted_churn_probability', ascending=False)
    
    for idx, row in active_ranking.iterrows():
        cid = row['client_id']
        prob = row['predicted_churn_probability']
        mrr = row['monthly_recurring_revenue']
        am = row['account_manager']
        c_type = row['contract_type']
        
        # Determine dominant risk factors from latest features
        risk_reasons = []
        rec_action = ""
        est_retention_mrr = mrr * prob # expected value of revenue saved
        
        if cid in latest_features.index:
            feat = latest_features.loc[cid]
            
            # Check SLA
            if feat['sla_breach_rate'] > 0.15:
                risk_reasons.append("High SLA Breach Rate")
                rec_action = f"Schedule immediate technical service review with operations to address SLA breaches ({feat['sla_breach_rate']:.1%} escalated support tickets)."
            # Check Product Usage
            elif feat['total_usage'] < 300:
                risk_reasons.append("Low Product Usage Score")
                rec_action = "Initiate user enablement training session. Product usage has dropped below active benchmarks."
            # Check Late Payments
            elif feat['late_payment_rate'] > 0.25:
                risk_reasons.append("Chronic Late Payments")
                rec_action = f"Contact client accounts payable to transition from {c_type} invoicing to auto-pay or offer a 2% discount for annual contract conversion."
            # Check NPS
            elif feat['nps_score'] < 0:
                risk_reasons.append("Detractor NPS Score")
                rec_action = f"Account Manager {am} to conduct direct executive check-in. NPS is currently in detractor zone ({feat['nps_score']})."
            # Default
            else:
                risk_reasons.append("Declining Engagement")
                rec_action = "Proactive account management review. Schedule quarterly business review (QBR) to pitch a migration or upgrade path."
        else:
            risk_reasons.append("High Score Profile")
            rec_action = "Executive check-in to secure contract renewal."

        # We focus on Medium and High risk clients
        if prob >= 0.30:
            recommendations_list.append({
                'priority': len(recommendations_list) + 1,
                'client_id': cid,
                'client_name': row['client_name'],
                'predicted_probability': prob,
                'risk_tier': row['risk_tier'],
                'monthly_recurring_revenue': mrr,
                'account_manager': am,
                'contract_type': c_type,
                'primary_issue': ", ".join(risk_reasons),
                'action_item': rec_action,
                'expected_revenue_saved': est_retention_mrr
            })
            
    return recommendations_list

def get_retention_strategy_impact(workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    recs = generate_recommendations(workspace_dir)
    if not recs:
        return {}
        
    recs_df = pd.DataFrame(recs)
    
    # Let's say if we execute recommendations, we can mitigate churn by 60% for targeted clients
    mitigation_efficiency = 0.60
    
    total_mrr_at_risk = recs_df['monthly_recurring_revenue'].sum()
    total_expected_saved = recs_df['expected_revenue_saved'].sum() * mitigation_efficiency
    
    # Calculate impact on overall active churn rate
    # Active clients count
    ranking_path = os.path.join(workspace_dir, "reports", "churn_risk_ranking.csv")
    ranking_df = pd.read_csv(ranking_path)
    active_df = ranking_df[~ranking_df['is_churned']]
    len(active_df)
    
    baseline_churn_rate = active_df['predicted_churn_probability'].mean()
    
    # If we mitigate churn for high and medium risk clients:
    # New probability = old probability * (1 - efficiency)
    new_probs = active_df['predicted_churn_probability'].values.copy()
    # Map client IDs of recs to apply efficiency
    recs_clients = set(recs_df['client_id'])
    
    for i, cid in enumerate(active_df['client_id']):
        if cid in recs_clients:
            new_probs[i] = new_probs[i] * (1.0 - mitigation_efficiency)
            
    new_churn_rate = new_probs.mean()
    absolute_reduction = baseline_churn_rate - new_churn_rate
    relative_reduction = absolute_reduction / baseline_churn_rate if baseline_churn_rate > 0 else 0
    
    return {
        'total_clients_targeted': len(recs),
        'total_mrr_at_risk': total_mrr_at_risk,
        'projected_mrr_saved': total_expected_saved,
        'baseline_churn_rate': baseline_churn_rate * 100,
        'new_churn_rate': new_churn_rate * 100,
        'absolute_churn_reduction': absolute_reduction * 100,
        'relative_churn_reduction': relative_reduction * 100,
        'target_achieved': relative_reduction >= 0.05
    }

if __name__ == '__main__':
    recs = generate_recommendations()
    print(f"\nGenerated {len(recs)} prioritized client retention alerts.")
    if recs:
        print("\nTop 3 Prioritized Alerts:")
        for r in recs[:3]:
            print(f"Priority {r['priority']}: Client {r['client_name']} ({r['client_id']})")
            print(f" - Risk Tier: {r['risk_tier']} (Prob: {r['predicted_probability']:.1%}) | MRR: ${r['monthly_recurring_revenue']:,.2f}")
            print(f" - Owner: {r['account_manager']} | Contract: {r['contract_type']}")
            print(f" - Issue: {r['primary_issue']}")
            print(f" - Action: {r['action_item']}")
            print(f" - Expected Value Saved: ${r['expected_revenue_saved']:,.2f}\n")
            
        impact = get_retention_strategy_impact()
        print("\n" + "="*50)
        print("RETENTION STRATEGY BUSINESS IMPACT SUMMARY")
        print("="*50)
        print(f"Baseline Overall Churn Rate: {impact['baseline_churn_rate']:.2f}%")
        print(f"Projected Overall Churn Rate (after interventions): {impact['new_churn_rate']:.2f}%")
        print(f"Projected Churn Rate Reduction (Relative): {impact['relative_churn_reduction']:.2f}% (Target: 5.00%)")
        print(f"Target Met? {'YES' if impact['target_achieved'] else 'NO'}")
        print(f"Estimated Monthly Revenue Saved: ${impact['projected_mrr_saved']:,.2f} / month")
        print("="*50 + "\n")
