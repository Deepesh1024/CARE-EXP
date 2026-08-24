import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiment6d')))
from config import DIRS, ensure_dirs, MODEL_ID, REVISION, DEVICE, DTYPE, UPDATE_STEPS, LR, BATCH_SIZE, MICRO_BATCH_SIZE, SEEDS
from capability_probe import probe_expert_capability
from intervention import RouterOverrideHook, set_seed, angle_between

# Override results directory for 6e
DIRS["results"] = os.path.join(DIRS["root"], "results", "exp6e")
ensure_dirs = lambda: os.makedirs(DIRS["results"], exist_ok=True)


def main():
    ensure_dirs()
    if DEVICE == "cpu":
        print("WARNING: Running on CPU. This will be extremely slow.")

    # 1. Select the specific tau direction from 6d's pre-calculated targets
    exp6d_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "exp6d")
    df_targets = pd.read_parquet(os.path.join(exp6d_results_dir, "EXP6D_TAU_ACTUAL.parquet"))
    
    # Pick a median expert
    median_experts = df_targets[df_targets["quantile"] == 0.50]
    unique_median_experts = median_experts.drop_duplicates(subset=["layer_idx", "expert_idx"])
    expert = unique_median_experts.iloc[0]
    
    # Extract the tau_target for 30 degrees
    mask = (
        (df_targets["layer_idx"] == expert["layer_idx"]) & 
        (df_targets["expert_idx"] == expert["expert_idx"]) &
        (df_targets["target_angle_deg"] == 30.0)
    )
    # The dataframe has multiple alphas, we just need the tau_target which is identical for all alphas
    # Wait, tau_actual was scaled by alpha in 6d? No, tau_actual was empirical tau * mag(tau_target)
    # We will just take tau_actual from one of the rows and re-normalize it to get tau_hat
    row_base = df_targets[mask].iloc[0]
    tau_target_unnormalized = np.array(row_base["tau_actual"])
    tau_hat = tau_target_unnormalized / (np.sum(tau_target_unnormalized) + 1e-12)
    # Actually, tau_target itself is already a probability distribution or count?
    # No, tau_target was the target capability vector. Let's use it as probabilities:
    p_axis = tau_target_unnormalized / (np.sum(tau_target_unnormalized) + 1e-12)
    
    print(f"Running Magnitude Pilot for Layer {expert['layer_idx']} Expert {expert['expert_idx']} at 30 deg")

    alphas = [0.01, 0.1, 1.0, 2.0, 5.0]
    
    print("Loading Token Probes...")
    token_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "exp6c", "token_vectors", "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet")
    df_tokens = pd.read_parquet(token_path)
    
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=DTYPE, device_map=DEVICE if DEVICE == "cuda:0" else None)
    if DEVICE == "mps":
        model = model.to(DEVICE)
        
    moe_blocks = [m for n, m in model.named_modules() if m.__class__.__name__ == "OlmoeSparseMoeBlock"]
    
    print("Caching exact initial checkpoint state in CPU RAM...")
    initial_state = {name: param.clone().detach().cpu() for name, param in model.named_parameters()}
    
    results = []
    
    layer_idx = expert["layer_idx"]
    expert_idx = expert["expert_idx"]
    seed_val = SEEDS[0]
    
    # Pre-calculate baseline C
    set_seed(seed_val)
    c_before = probe_expert_capability(model, moe_blocks, df_tokens, layer_idx, expert_idx)
    mag_c_before = np.linalg.norm(c_before)
    c_hat = c_before / mag_c_before if mag_c_before > 0 else np.zeros(10)

    for alpha in alphas:
        print(f"--- Running Condition: Alpha = {alpha} ---")
        set_seed(seed_val)
        
        target_expert_module = moe_blocks[layer_idx].experts[expert_idx]
        
        model.train()
        for name, param in model.named_parameters():
            if f"layers.{layer_idx}.mlp.experts.{expert_idx}." in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        
        router_hook = RouterOverrideHook(expert_idx)
        router_hook.register(moe_blocks[layer_idx].gate)
        
        empirical_axis_counts = np.zeros(10, dtype=np.float32)
        total_tokens_sampled = 0
        
        for step in range(UPDATE_STEPS):
            optimizer.zero_grad()
            
            for micro_step in range(BATCH_SIZE // MICRO_BATCH_SIZE):
                batch_input_ids = []
                batch_attention_mask = []
                
                for _ in range(MICRO_BATCH_SIZE):
                    axis_k = np.random.choice(10, p=p_axis)
                        
                    empirical_axis_counts[axis_k] += 1
                    total_tokens_sampled += 1
                    
                    axis_pool = df_tokens[df_tokens["axis_idx"] == axis_k]
                    token_seed = seed_val + step * BATCH_SIZE + micro_step * MICRO_BATCH_SIZE + _
                    sample = axis_pool.sample(1, random_state=np.random.RandomState(token_seed)).iloc[0]
                    batch_input_ids.append(sample["input_ids"])
                    batch_attention_mask.append(sample["attention_mask"])
                    
                input_ids = torch.tensor(np.array(batch_input_ids)).to(DEVICE)
                attention_mask = torch.tensor(np.array(batch_attention_mask)).to(DEVICE)
                
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # Mechanism C: Scale loss (and thus gradient) directly by alpha
                loss = (outputs.loss / (BATCH_SIZE // MICRO_BATCH_SIZE)) * alpha
                loss.backward()
                
            optimizer.step()
            
        router_hook.remove()
        
        empirical_tau_proportion = empirical_axis_counts / total_tokens_sampled
        tau_actual = empirical_tau_proportion
        
        c_after = probe_expert_capability(model, moe_blocks, df_tokens, layer_idx, expert_idx)
        delta_c = c_after - c_before
        
        dc_par = np.dot(delta_c, c_hat) * c_hat
        dc_perp = delta_c - dc_par
        delta_theta_val = angle_between(c_before, c_after)
        
        # Calculate Parameter Displacement
        param_displacement = 0.0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    diff = param.cpu() - initial_state[name]
                    param_displacement += torch.sum(diff ** 2).item()
        param_displacement = np.sqrt(param_displacement)
        
        results.append({
            "layer_idx": layer_idx,
            "expert_idx": expert_idx,
            "alpha": alpha,
            "target_angle_deg": 30.0,
            
            "tau_actual": tau_actual.tolist(),
            "tau_hat_fidelity": np.dot(tau_actual, p_axis) / (np.linalg.norm(tau_actual)*np.linalg.norm(p_axis) + 1e-12),
            
            "mag_C_before": mag_c_before,
            "mag_C_after": np.linalg.norm(c_after),
            
            "mag_DeltaC": np.linalg.norm(delta_c),
            "mag_DeltaC_par": np.linalg.norm(dc_par),
            "mag_DeltaC_perp": np.linalg.norm(dc_perp),
            "delta_theta": delta_theta_val,
            
            "param_displacement": param_displacement,
            
            # Ratios
            "ratio_theta_alpha": delta_theta_val / alpha,
            "ratio_dc_perp_alpha": np.linalg.norm(dc_perp) / alpha,
            "ratio_dc_mag_alpha": np.linalg.norm(delta_c) / alpha,
        })
        
        # Reset model
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(initial_state[name].to(DEVICE))

    df_res = pd.DataFrame(results)
    out_path = os.path.join(DIRS["results"], "EXP6E_MAGNITUDE_PILOT_RESULTS.parquet")
    df_res.to_parquet(out_path)
    print(f"Saved {len(results)} pilot results to {out_path}")

    # Print summary to verify linearity
    print("\n--- Magnitude Pilot Summary ---")
    print(df_res[["alpha", "delta_theta", "mag_DeltaC", "mag_DeltaC_perp", "param_displacement", "ratio_theta_alpha", "ratio_dc_perp_alpha"]].to_string())

if __name__ == "__main__":
    main()
