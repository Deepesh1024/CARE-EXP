import os
import pandas as pd
import numpy as np

def generate_tables(summary_dir):
    csv_path = os.path.join(summary_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: results.csv is empty.")
        return

    df_random = df[df['method'] == 'random']
    random_stats = df_random.groupby('experts').agg({
        'ppl': ['mean', 'std'],
        'delta_ppl': ['mean', 'std'],
        'cumulative_kl': ['mean', 'std'],
        'compression_time_sec': 'mean',
        'num_exact_evaluations': 'mean'
    }).reset_index()
    
    def format_mean_std(mean_val, std_val):
        if pd.isna(mean_val):
            return "N/A"
        if pd.isna(std_val) or std_val == 0:
            return f"{mean_val:.3f}"
        return f"{mean_val:.3f} ± {std_val:.3f}"

    methods_order = ['random', 'hc_smoe', 'm_smoe', 'sub_moe', 'care_static', 'care_adaptive']
    
    table1_rows = []
    for method in methods_order:
        if method == 'random':
            ppl_60 = random_stats[random_stats['experts'] == 60]
            ppl_56 = random_stats[random_stats['experts'] == 56]
            ppl_48 = random_stats[random_stats['experts'] == 48]
            
            row = {
                "Method": "Random",
                "PPL@60 ↓": format_mean_std(ppl_60['ppl']['mean'].values[0] if not ppl_60.empty else np.nan, ppl_60['ppl']['std'].values[0] if not ppl_60.empty else np.nan),
                "PPL@56 ↓": format_mean_std(ppl_56['ppl']['mean'].values[0] if not ppl_56.empty else np.nan, ppl_56['ppl']['std'].values[0] if not ppl_56.empty else np.nan),
                "PPL@48 ↓": format_mean_std(ppl_48['ppl']['mean'].values[0] if not ppl_48.empty else np.nan, ppl_48['ppl']['std'].values[0] if not ppl_48.empty else np.nan),
                "ΔPPL@48 ↓": format_mean_std(ppl_48['delta_ppl']['mean'].values[0] if not ppl_48.empty else np.nan, ppl_48['delta_ppl']['std'].values[0] if not ppl_48.empty else np.nan),
                "Compression Time@48": f"{ppl_48['compression_time_sec']['mean'].values[0]:.1f}" if not ppl_48.empty else "N/A",
                "Exact Evaluations@48": "0"
            }
        else:
            df_m = df[df['method'] == method]
            if df_m.empty:
                continue
                
            val_60 = df_m[df_m['experts'] == 60].iloc[0] if not df_m[df_m['experts'] == 60].empty else None
            val_56 = df_m[df_m['experts'] == 56].iloc[0] if not df_m[df_m['experts'] == 56].empty else None
            val_48 = df_m[df_m['experts'] == 48].iloc[0] if not df_m[df_m['experts'] == 48].empty else None
            
            row = {
                "Method": method,
                "PPL@60 ↓": f"{val_60['ppl']:.3f}" if val_60 is not None else "N/A",
                "PPL@56 ↓": f"{val_56['ppl']:.3f}" if val_56 is not None else "N/A",
                "PPL@48 ↓": f"{val_48['ppl']:.3f}" if val_48 is not None else "N/A",
                "ΔPPL@48 ↓": f"{val_48['delta_ppl']:.3f}" if val_48 is not None else "N/A",
                "Compression Time@48": f"{val_48['compression_time_sec']:.1f}" if val_48 is not None else "N/A",
                "Exact Evaluations@48": f"{val_48['num_exact_evaluations']}" if val_48 is not None else "N/A"
            }
        table1_rows.append(row)
        
    df_table1 = pd.DataFrame(table1_rows)
    df_table1.to_csv(os.path.join(summary_dir, "table_main.csv"), index=False)
    
    df_static = df[df['method'] == 'care_static']
    df_adapt = df[df['method'] == 'care_adaptive']
    
    table2_rows = []
    for target in [60, 56, 48]:
        s_row = df_static[df_static['experts'] == target].iloc[0] if not df_static[df_static['experts'] == target].empty else None
        a_row = df_adapt[df_adapt['experts'] == target].iloc[0] if not df_adapt[df_adapt['experts'] == target].empty else None
        
        if s_row is not None and a_row is not None:
            kl_red_pct = (s_row['cumulative_kl'] - a_row['cumulative_kl']) / s_row['cumulative_kl'] * 100 if s_row['cumulative_kl'] > 0 else 0
            eval_red_pct = (s_row['num_exact_evaluations'] - a_row['num_exact_evaluations']) / s_row['num_exact_evaluations'] * 100 if s_row['num_exact_evaluations'] > 0 else 0
            
            table2_rows.append({
                "Target": target,
                "Static PPL": f"{s_row['ppl']:.3f}",
                "Adaptive PPL": f"{a_row['ppl']:.3f}",
                "Static cumulative KL": f"{s_row['cumulative_kl']:.3f}",
                "Adaptive cumulative KL": f"{a_row['cumulative_kl']:.3f}",
                "KL reduction %": f"{kl_red_pct:.1f}%",
                "Static exact evaluations": s_row['num_exact_evaluations'],
                "Adaptive exact evaluations": a_row['num_exact_evaluations'],
                "Evaluation reduction %": f"{eval_red_pct:.1f}%"
            })
            
    df_table2 = pd.DataFrame(table2_rows)
    df_table2.to_csv(os.path.join(summary_dir, "table_adaptive_ablation.csv"), index=False)
    
    print(f"Successfully generated tables in {summary_dir}")

if __name__ == "__main__":
    benchmark_dir = os.path.join(os.path.dirname(__file__), "..", "benchmark_results")
    summary_dir = os.path.join(benchmark_dir, "summary")
    generate_tables(summary_dir)
