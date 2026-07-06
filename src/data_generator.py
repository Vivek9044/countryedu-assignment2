import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

def generate_datasets(workspace_dir="c:/Users/kasau/Desktop/guptachurn"):
    start_time = time.time()
    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)

    # 1. Create Directories if they don't exist
    data_dir = os.path.join(workspace_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    clients_path = os.path.join(data_dir, "clients.csv")
    transactions_path = os.path.join(data_dir, "transactions.csv")

    # Define baseline dates
    start_date = datetime(2023, 7, 1)
    end_date = datetime(2026, 6, 30)

    print("Generating client profiles...")
    num_clients = 200
    client_ids = [f"CLT{i:03d}" for i in range(1, num_clients + 1)]
    
    industries = ['Finance', 'Healthcare', 'Retail', 'Technology', 'Manufacturing']
    sizes = ['Small', 'Medium', 'Enterprise']
    contract_types = ['Monthly', 'Annual', 'Multi-year']
    regions = ['North', 'East', 'South', 'West']
    managers = ['Alice Smith', 'Bob Jones', 'Charlie Brown', 'Diana Prince', 'Evan Wright']
    services_list = ['Infrastructure', 'Cloud', 'Security', 'Support']

    clients_data = []

    for cid in client_ids:
        industry = np.random.choice(industries)
        size = np.random.choice(sizes, p=[0.5, 0.3, 0.2]) # More small/medium than enterprise
        
        # contract start between 2021-01-01 and 2024-06-30
        c_start_days = np.random.randint(0, 1276) # days between 2021-01-01 and 2024-06-30
        contract_start = datetime(2021, 1, 1) + timedelta(days=int(c_start_days))
        
        contract_type = np.random.choice(contract_types, p=[0.4, 0.4, 0.2])
        
        # Services subscribed (at least 1)
        num_services = np.random.randint(1, 5)
        subscribed = np.random.choice(services_list, size=num_services, replace=False)
        services_str = ";".join(subscribed)
        
        # Base monthly recurring revenue based on size
        if size == 'Small':
            mrr = np.random.uniform(2000, 5000)
        elif size == 'Medium':
            mrr = np.random.uniform(5000, 15000)
        else:
            mrr = np.random.uniform(15000, 50000)
            
        contract_value = mrr * (1 if contract_type == 'Monthly' else (12 if contract_type == 'Annual' else 36))
        manager = np.random.choice(managers)
        region = np.random.choice(regions)
        
        # Behavior metrics (initial baseline)
        support_tickets_90 = np.random.randint(0, 15)
        avg_satisfaction = np.random.uniform(3.5, 5.0)
        sla_breaches_90 = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
        payment_delay_avg = np.random.uniform(0, 5)
        product_usage = np.random.uniform(70, 100)
        nps = np.random.randint(20, 100)
        
        clients_data.append({
            'client_id': cid,
            'client_name': fake.company(),
            'industry': industry,
            'client_size': size,
            'contract_start_date': contract_start.strftime('%Y-%m-%d'),
            'contract_type': contract_type,
            'services_subscribed': services_str,
            'monthly_recurring_revenue': mrr,
            'contract_value': contract_value,
            'account_manager': manager,
            'region': region,
            'support_tickets_last_90_days': support_tickets_90,
            'avg_ticket_satisfaction': avg_satisfaction,
            'sla_breaches_last_90_days': sla_breaches_90,
            'payment_delay_days_avg': payment_delay_avg,
            'product_usage_score': product_usage,
            'nps_score': nps
        })

    clients_df = pd.DataFrame(clients_data)

    # Let's induce causal signal and designate EXACTLY 18% churned (36 clients)
    # Calculate a risk score for each client based on features
    risk_scores = np.zeros(num_clients)
    for idx, row in clients_df.iterrows():
        score = 0
        if row['contract_type'] == 'Monthly':
            score += 4.0
        elif row['contract_type'] == 'Annual':
            score += 1.0
            
        # Services count (fewer services -> higher churn risk)
        num_services = len(row['services_subscribed'].split(';'))
        if num_services == 1:
            score += 3.0
        elif num_services == 2:
            score += 1.5
            
        # Let's adjust client features to induce churn risk
        # We will modify the features of the highest risk clients to make the signal stronger
        risk_scores[idx] = score

    # Sort and pick top 36 clients to churn
    # To make it realistic, we will add some noise to risk scores
    risk_scores_with_noise = risk_scores + np.random.normal(0, 1.5, size=num_clients)
    churn_indices = np.argsort(risk_scores_with_noise)[-36:]
    
    clients_df['is_churned'] = False
    clients_df['churn_date'] = ""
    clients_df['tenure_months'] = 0

    # Adjust parameters of churned clients to introduce strong causal signals
    for idx in range(num_clients):
        cid = clients_df.loc[idx, 'client_id']
        c_start = datetime.strptime(clients_df.loc[idx, 'contract_start_date'], '%Y-%m-%d')
        
        if idx in churn_indices:
            clients_df.loc[idx, 'is_churned'] = True
            # Churn date distributed in the last 12 months (2025-07-01 to 2026-06-30)
            churn_days = np.random.randint(730, 1095) # days from start_date (2023-07-01)
            churn_dt = start_date + timedelta(days=int(churn_days))
            clients_df.loc[idx, 'churn_date'] = churn_dt.strftime('%Y-%m-%d')
            
            # Worsen the behavior metrics for churned clients
            clients_df.loc[idx, 'support_tickets_last_90_days'] = np.random.randint(15, 30)
            clients_df.loc[idx, 'avg_ticket_satisfaction'] = np.random.uniform(1.5, 3.2)
            clients_df.loc[idx, 'sla_breaches_last_90_days'] = np.random.randint(2, 6)
            clients_df.loc[idx, 'payment_delay_days_avg'] = np.random.uniform(8, 20)
            clients_df.loc[idx, 'product_usage_score'] = np.random.uniform(20, 65)
            clients_df.loc[idx, 'nps_score'] = np.random.randint(-100, 10)
            
            # Tenure is months from contract start to churn date
            tenure = (churn_dt.year - c_start.year) * 12 + churn_dt.month - c_start.month
            clients_df.loc[idx, 'tenure_months'] = max(1, tenure)
        else:
            # Tenure is months from contract start to end of simulation (2026-06-30)
            tenure = (end_date.year - c_start.year) * 12 + end_date.month - c_start.month
            clients_df.loc[idx, 'tenure_months'] = max(1, tenure)

    # 12% Missing financial data in clients.csv
    # Fields: monthly_recurring_revenue, payment_delay_days_avg
    missing_mask_mrr = np.random.rand(num_clients) < 0.12
    missing_mask_delay = np.random.rand(num_clients) < 0.12
    
    clients_df.loc[missing_mask_mrr, 'monthly_recurring_revenue'] = np.nan
    clients_df.loc[missing_mask_delay, 'payment_delay_days_avg'] = np.nan

    # Save clients.csv
    clients_df.to_csv(clients_path, index=False)
    print(f"Generated clients.csv with {len(clients_df)} rows. Churn rate: {clients_df['is_churned'].mean():.2%}")

    # 2. Generate Transactions (10 Million Rows)
    print("Generating transactions.csv (10 million rows)...")
    
    # Pre-calculate active period for each client in our simulation range
    client_periods = []
    total_active_client_months = 0
    
    for idx, row in clients_df.iterrows():
        cid = row['client_id']
        c_start = datetime.strptime(row['contract_start_date'], '%Y-%m-%d')
        
        # Active start in our simulation window is max(contract_start, start_date)
        act_start = max(c_start, start_date)
        
        # Active end is churn_date if churned, else end_date
        if row['is_churned'] and not pd.isna(row['churn_date']):
            act_end = datetime.strptime(row['churn_date'], '%Y-%m-%d')
        else:
            act_end = end_date
            
        if act_start > act_end:
            act_start = act_end - timedelta(days=30) # make sure active at least a bit
            
        months_active = (act_end.year - act_start.year) * 12 + act_end.month - act_start.month + 1
        client_periods.append({
            'client_id': cid,
            'start': act_start,
            'end': act_end,
            'months_active': months_active,
            'is_churned': row['is_churned'],
            'mrr': row['monthly_recurring_revenue'] if not pd.isna(row['monthly_recurring_revenue']) else 10000.0, # default if nan
            'services': row['services_subscribed'].split(';')
        })
        total_active_client_months += months_active

    # Determine total target rows (10,000,000)
    target_txn_count = 10000000
    
    # We will generate:
    # - Monthly Invoices and Payments (2 rows per active month per client)
    # - Support tickets (random per month per client)
    # - Upgrades/downgrades (random)
    # - Usage logs (the remaining bulk of the 10M rows)
    
    # First, let's create structural transactions: invoices, payments, tickets, upgrades
    struct_txns = []
    
    print("Generating structural transactions (invoices, payments, support)...")
    for cp in client_periods:
        cid = cp['client_id']
        mrr = cp['mrr']
        services = cp['services']
        
        # Invoices and payments
        curr_dt = cp['start']
        while curr_dt <= cp['end']:
            # Invoice
            inv_amt = mrr * np.random.uniform(0.95, 1.05)
            struct_txns.append({
                'client_id': cid,
                'transaction_date': curr_dt.strftime('%Y-%m-%d'),
                'transaction_type': 'invoice',
                'amount': np.nan if np.random.rand() < 0.12 else inv_amt,
                'service_line': np.random.choice(services),
                'payment_status': np.nan,
                'usage_units': np.nan,
                'notes_flag': 0
            })
            
            # Payment
            pay_days_delay = np.random.exponential(scale=15.0 if cp['is_churned'] else 2.0)
            pay_dt = curr_dt + timedelta(days=int(pay_days_delay))
            pay_status = 'late' if pay_days_delay > 5 else 'on-time'
            if np.random.rand() < (0.05 if cp['is_churned'] else 0.005):
                pay_status = 'failed'
                
            struct_txns.append({
                'client_id': cid,
                'transaction_date': pay_dt.strftime('%Y-%m-%d'),
                'transaction_type': 'payment',
                'amount': np.nan if np.random.rand() < 0.12 else inv_amt,
                'service_line': np.nan,
                'payment_status': pay_status,
                'usage_units': np.nan,
                'notes_flag': 0
            })
            
            # Support tickets
            num_tickets = np.random.poisson(lam=4.0 if cp['is_churned'] else 1.0)
            for _ in range(num_tickets):
                t_day = np.random.randint(0, 28)
                t_dt = curr_dt + timedelta(days=int(t_day))
                struct_txns.append({
                    'client_id': cid,
                    'transaction_date': t_dt.strftime('%Y-%m-%d'),
                    'transaction_type': 'support-ticket',
                    'amount': np.nan,
                    'service_line': 'Support',
                    'payment_status': np.nan,
                    'usage_units': np.nan,
                    'notes_flag': np.random.choice([0, 1], p=[0.9, 0.1])
                })
                
            # Service upgrades / downgrades
            if np.random.rand() < 0.05:
                up_dt = curr_dt + timedelta(days=int(np.random.randint(0, 28)))
                is_up = np.random.rand() > (0.70 if cp['is_churned'] else 0.20) # churned clients downgrade more
                struct_txns.append({
                    'client_id': cid,
                    'transaction_date': up_dt.strftime('%Y-%m-%d'),
                    'transaction_type': 'service-upgrade' if is_up else 'service-downgrade',
                    'amount': np.nan if np.random.rand() < 0.12 else (np.random.uniform(500, 3000) * (1 if is_up else -1)),
                    'service_line': np.random.choice(services),
                    'payment_status': np.nan,
                    'usage_units': np.nan,
                    'notes_flag': 0
                })
                
            # Move to next month
            # simple month increment
            if curr_dt.month == 12:
                curr_dt = datetime(curr_dt.year + 1, 1, 1)
            else:
                curr_dt = datetime(curr_dt.year, curr_dt.month + 1, 1)

    struct_df = pd.DataFrame(struct_txns)
    struct_count = len(struct_df)
    print(f"Generated {struct_count} structural transactions.")

    # Write structural transactions first
    struct_df['transaction_id'] = [f"TXN{i:08d}" for i in range(1, struct_count + 1)]
    cols = ['transaction_id', 'client_id', 'transaction_date', 'transaction_type', 
            'amount', 'service_line', 'payment_status', 'usage_units', 'notes_flag']
    struct_df = struct_df[cols]
    
    # Save the file (create/overwrite)
    struct_df.to_csv(transactions_path, index=False)

    # Now generate the usage logs (approx 10M - struct_count)
    usage_logs_needed = target_txn_count - struct_count
    print(f"Generating {usage_logs_needed} usage logs in chunks...")

    # We will write in chunks of 1 million rows
    chunk_size = 1000000
    chunks_written = 0
    total_written = struct_count
    
    # To sample client IDs and dates efficiently:
    # Prepare sampling weights based on months active for each client
    client_ids_pool = [cp['client_id'] for cp in client_periods]
    client_weights = np.array([cp['months_active'] for cp in client_periods])
    client_weights = client_weights / client_weights.sum()

    # Cache client active bounds to avoid datetime parsing inside the loop
    client_bounds = {}
    for cp in client_periods:
        client_bounds[cp['client_id']] = (
            int(cp['start'].timestamp()),
            int(cp['end'].timestamp()),
            cp['services'],
            cp['is_churned']
        )

    # Let's generate in loops
    while total_written < target_txn_count:
        rows_to_gen = min(chunk_size, target_txn_count - total_written)
        
        # Vectorized generation of client IDs
        sampled_cids = np.random.choice(client_ids_pool, size=rows_to_gen, p=client_weights)
        
        # We need to assign transaction dates, service lines, usage units
        # To do this fast, we can map using numpy or processes
        # Let's do it in a vectorized way by grouping by client
        unique_cids, cid_counts = np.unique(sampled_cids, return_counts=True)
        
        chunk_data = []
        for cid, count in zip(unique_cids, cid_counts):
            start_ts, end_ts, services, is_churned = client_bounds[cid]
            
            # Vectorized random timestamps
            random_ts = np.random.randint(start_ts, end_ts, size=count)
            # Format dates (fast string formatting via pandas)
            dates = pd.to_datetime(random_ts, unit='s').strftime('%Y-%m-%d')
            
            # Service line
            # Most usage logs are Infrastructure or Cloud
            usage_services = [s for s in services if s in ['Infrastructure', 'Cloud', 'Security']]
            if not usage_services:
                usage_services = services
            sampled_services = np.random.choice(usage_services, size=count)
            
            # Usage units (declining engagement for churned clients)
            if is_churned:
                # Usage declines as we get closer to churn
                # Generate usage units
                usage_units = np.random.uniform(5, 400, size=count)
            else:
                usage_units = np.random.uniform(100, 1000, size=count)
                
            sub_df = pd.DataFrame({
                'client_id': cid,
                'transaction_date': dates,
                'transaction_type': 'usage-log',
                'amount': np.nan,
                'service_line': sampled_services,
                'payment_status': np.nan,
                'usage_units': usage_units,
                'notes_flag': np.random.choice([0, 1], p=[0.98, 0.02], size=count)
            })
            chunk_data.append(sub_df)
            
        chunk_df = pd.concat(chunk_data, ignore_index=True)
        # shuffle
        chunk_df = chunk_df.sample(frac=1.0).reset_index(drop=True)
        
        # Assign IDs using vectorized operations
        start_idx = total_written + 1
        chunk_df['transaction_id'] = "TXN" + pd.Series(np.arange(start_idx, start_idx + len(chunk_df))).astype(str).str.zfill(8)
        chunk_df = chunk_df[cols]
        
        # Append to file
        chunk_df.to_csv(transactions_path, mode='a', header=False, index=False)
        
        chunks_written += 1
        total_written += len(chunk_df)
        print(f"Chunk {chunks_written} written: {len(chunk_df)} rows. Total transactions written: {total_written}")

    # Now let's rewrite transaction_ids to be unique strings TXN00000001 to TXN10000000
    # Wait, rewriting a 10M CSV file would require reading the whole thing.
    # To avoid this, we can write IDs as we go or just let the ETL pipeline assign transaction IDs.
    # But to satisfy the requirement "transactions.csv ... transaction_id, ...", we can just assign the transaction ID as a running integer in ETL, or write it directly.
    # Actually, can we write transaction_id as part of the CSV files?
    # Yes! In our generator, we put empty string as a placeholder. We can easily write transaction IDs during ETL, or let's check:
    # Does transactions.csv need to have the actual transaction_id strings?
    # Let's check: if transactions.csv has a blank transaction_id, that's fine as long as we fill it or generate it.
    # But wait, it's cleaner if transactions.csv actually contains the IDs.
    # To generate IDs without reading the file back: we can keep a global counter and assign the IDs before writing each chunk.
    # Let's modify the generator code to assign transaction IDs directly in memory before writing!
    # In the structural transactions part:
    # struct_df['transaction_id'] = [f"TXN{i:08d}" for i in range(1, struct_count + 1)]
    # In the usage logs part, we can do:
    # np.arange(total_written_before_chunk + 1, total_written_before_chunk + len(chunk_df) + 1)
    # formatted as TXN00000000 + idx.
    # Let's write this optimized version!

    print("Injecting 12% missing values to transaction amounts...")
    # Wait! We need to inject 12% missing data to the 'amount' field in transactions.
    # Let's do this directly during the generation of the structural transactions in the code above!
    # This is much faster than loading the file back.
    # Let's look at structural transactions generation in our python script. We can just add a random mask to amount.
    
    import psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Calculate stats
    clients_gen = len(clients_df)
    churn_rate = clients_df['is_churned'].mean()
    mrr_missing = clients_df['monthly_recurring_revenue'].isna().mean()
    delay_missing = clients_df['payment_delay_days_avg'].isna().mean()
    
    # Financial transaction stats
    fin_txns = struct_df[struct_df['transaction_type'].isin(['invoice', 'payment', 'service-upgrade', 'service-downgrade'])]
    fin_missing = fin_txns['amount'].isna().mean()
    
    file_size_gb = os.path.getsize(transactions_path) / (1024 * 1024 * 1024)

    print("\n" + "="*50)
    print("DATA GENERATION SUMMARY")
    print("="*50)
    print(f"Total Clients Generated: {clients_gen}")
    print(f"Client Churn Rate: {churn_rate:.2%} (Exactly {clients_df['is_churned'].sum()} clients)")
    print(f"Clients Missing MRR: {mrr_missing:.2%}")
    print(f"Clients Missing Payment Delay Avg: {delay_missing:.2%}")
    print(f"Total Transactions Generated: {total_written:,} rows")
    print(f"Financial Transactions Missing Amount: {fin_missing:.2%}")
    print(f"Transactions CSV File Size: {file_size_gb:.3f} GB")
    print(f"Memory Footprint: {mem_mb:.2f} MB")
    print(f"Total Elapsed Time: {elapsed:.2f} seconds")
    print("="*50 + "\n")

if __name__ == '__main__':
    generate_datasets()
