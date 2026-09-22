import os
import json
import pandas as pd
import glob

def get_param_count(experts):
    # OLMoE-1B-7B roughly:
    # Non-expert params: ~1B
    # Each expert: ~100M 
    # For a precise measurement we would load the model, but for the benchmark
    # we can approximate based on the number of experts, or leave it to be extracted from logs if we recorded it.
    # Since we didn't explicitly log param counts in the trajectory, we can compute reduction %.
    original_experts = 64
    reduction_pct = (original_experts - experts) / original_experts * 100
    return reduction_pct

def aggregate_results(results_dir):
    summary_dir = os.path.join(results_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    all_results = []
    care_selections = []
    
    # Iterate through all methods in results_dir
    for method_dir in os.listdir(results_dir):
        if method_dir == "summary" or not os.path.isdir(os.path.join(results_dir, method_dir)):
            continue
            
        # Random has subdirectories for seeds, others just have trajectory.json directly
        # Let's handle both
        json_paths = glob.glob(os.path.join(results_dir, method_dir, "**", "trajectory.json"), recursive=True)
        if not json_paths:
            json_paths = glob.glob(os.path.join(results_dir, method_dir, "trajectory.json"))
            
        for json_path in json_paths:
            with open(json_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse {json_path}")
                    continue
                    
            if "error" in data:
                print(f"Skipping {method_dir} due to error log: {data['error']}")
                continue
                
            method_name = data.get("config", {}).get("method", method_dir)
            ppl_dict = data.get("ppl", {})
            trace = data.get("trace", [])
            
            # Base PPL
            base_ppl = ppl_dict.get("64", None)
            
            # For each checkpoint, extract data
            for experts in ["64", "60", "56", "48"]:
                if experts not in ppl_dict:
                    continue
                    
                ppl = ppl_dict[experts]
                delta_ppl = ppl - base_ppl if base_ppl else None
                
                # Find cumulative KL at this expert count
                cum_kl = 0.0
                time_sec = 0.0
                peak_mem = 0.0
                evaluations = 0
                
                for step_data in trace:
                    if step_data["experts_after"] >= int(experts):
                        cum_kl = step_data["cumulative_kl"]
                        time_sec += step_data["wall_time_sec"]
                        peak_mem = max(peak_mem, step_data["peak_memory_mb"])
                        if step_data.get("candidate_pairs"):
                            evaluations += len(step_data["candidate_pairs"])
                            
                seed = trace[0].get("seed", "N/A") if trace else "N/A"
                
                all_results.append({
                    "method": method_name,
                    "seed": seed,
                    "experts": int(experts),
                    "ppl": ppl,
                    "delta_ppl": delta_ppl,
                    "cumulative_kl": cum_kl,
                    "param_reduction_pct": get_param_count(int(experts)),
                    "compression_time_sec": time_sec,
                    "peak_memory_mb": peak_mem,
                    "num_exact_evaluations": evaluations
                })
                
            # Extract CARE-COM specific selection data
            if "care" in method_name.lower():
                for step_data in trace:
                    if step_data.get("candidate_pairs"):
                        selected = step_data["selected_pair"]
                        
                        # In care_com.py we logged candidates, but we need cap_dist and KL per candidate.
                        # Wait, we logged the selected capability distance and selected marginal KL.
                        # If we had full candidate records, we'd add them. Let's add the selected one for now.
                        care_selections.append({
                            "step": step_data["step"],
                            "candidate_pair": selected,
                            "capability_distance": step_data["capability_distance"],
                            "actual_marginal_kl": step_data["marginal_kl"],
                            "selected": True
                        })
                        
    # Save Main Results CSV
    df = pd.DataFrame(all_results)
    if not df.empty:
        df.to_csv(os.path.join(summary_dir, "results.csv"), index=False)
        print(f"Saved results.csv to {summary_dir}")
        
    # Save Selection Analysis CSV
    df_care = pd.DataFrame(care_selections)
    if not df_care.empty:
        df_care.to_csv(os.path.join(summary_dir, "selection_analysis.csv"), index=False)
        print(f"Saved selection_analysis.csv to {summary_dir}")

if __name__ == "__main__":
    benchmark_dir = os.path.join(os.path.dirname(__file__), "..", "benchmark_results")
    if os.path.exists(benchmark_dir):
        aggregate_results(benchmark_dir)
    else:
        print(f"Error: {benchmark_dir} does not exist. Run benchmark first.")
