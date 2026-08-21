"""
EXPERIMENT 6D - INTERVENTION
============================================================
The GPU training loop executing the controlled response study.
Every experimental condition starts from the identical baseline state.
Only the targeted expert's weights are updated.
This is a CONTROLLED EXPERT-RESPONSE INTERVENTION, not a natural
routing simulation. 100% routing is forced.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DIRS, ensure_dirs, MODEL_ID, REVISION, DEVICE, DTYPE, UPDATE_STEPS, LR, BATCH_SIZE, MICRO_BATCH_SIZE, SEEDS, CALIBRATION_CONFIGS
from capability_probe import probe_expert_capability

class RouterOverrideHook:
    def __init__(self, target_expert_idx):
        self.target_expert_idx = target_expert_idx
        self.handle = None

    def hook_fn(self, module, args, output):
        new_output = torch.full_like(output, float('-inf'))
        if new_output.dim() == 3:
            new_output[:, :, self.target_expert_idx] = 10000.0
        elif new_output.dim() == 2:
            new_output[:, self.target_expert_idx] = 10000.0
        else:
            raise ValueError(f"Unexpected router logits shape: {output.shape}")
        return new_output

    def register(self, gate_module):
        self.handle = gate_module.register_forward_hook(self.hook_fn)
        
    def remove(self):
        if self.handle:
            self.handle.remove()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))

def run_zero_update_noise_floor(model, moe_blocks, df_tokens, unique_experts):
    print("\n--- MEASURING ZERO-UPDATE NOISE FLOOR ---")
    noise_results = []
    
    for _, e in tqdm(unique_experts.iterrows(), total=len(unique_experts), desc="Noise Floor Probes"):
        layer_idx = e["layer_idx"]
        expert_idx = e["expert_idx"]
        
        # We run the probe twice sequentially without any model updates
        for rep in range(10): # 10 repetitions to get robust sigma
            set_seed(42 + rep)
            c1 = probe_expert_capability(model, moe_blocks, df_tokens, layer_idx, expert_idx)
            set_seed(100 + rep)
            c2 = probe_expert_capability(model, moe_blocks, df_tokens, layer_idx, expert_idx)
            
            delta_c = c2 - c1
            mag_c1 = np.linalg.norm(c1)
            c_hat = c1 / mag_c1 if mag_c1 > 0 else np.zeros(10)
            
            dc_par = np.dot(delta_c, c_hat) * c_hat
            dc_perp = delta_c - dc_par
            
            noise_results.append({
                "layer_idx": layer_idx,
                "expert_idx": expert_idx,
                "rep": rep,
                "mag_DeltaC": np.linalg.norm(delta_c),
                "mag_DeltaC_par": np.linalg.norm(dc_par),
                "mag_DeltaC_perp": np.linalg.norm(dc_perp),
                "delta_theta": angle_between(c1, c2)
            })
            
    df_noise = pd.DataFrame(noise_results)
    out_path = os.path.join(DIRS["results"], "EXP6D_NOISE_FLOOR.parquet")
    df_noise.to_parquet(out_path)
    print(f"Noise floor established across {len(noise_results)} measurements. Saved to {out_path}.")
    return df_noise

def main(mode="full"):
    ensure_dirs()
    if DEVICE == "cpu":
        print("WARNING: Running on CPU. This will be extremely slow.")
        
    print("Loading 6D Tau Environments...")
    df_targets = pd.read_parquet(os.path.join(DIRS["results"], "EXP6D_TAU_ACTUAL.parquet"))
    
    if mode == "calibrate" or mode == "pilot":
        print(f"\n=== {mode.upper()} MODE ACTIVE ===")
        # 3 experts: low, middle, high ||C||
        unique_experts = df_targets.drop_duplicates(subset=["layer_idx", "expert_idx"])
        experts_to_keep = []
        for q in [0.10, 0.50, 0.90]:
            q_experts = unique_experts[unique_experts["quantile"] == q]
            if len(q_experts) > 0:
                experts_to_keep.append(q_experts.iloc[0])
                
        df_subset = pd.DataFrame()
        for e in experts_to_keep:
            mask = (
                (df_targets["layer_idx"] == e["layer_idx"]) & 
                (df_targets["expert_idx"] == e["expert_idx"]) &
                (df_targets["alpha"].isin([0.10, 0.80, 2.00])) &
                ((df_targets["target_angle_deg"] == 0.0) | 
                 (df_targets["target_angle_deg"] == 30.0) | 
                 (df_targets["is_theta_max"] == True))
            )
            df_subset = pd.concat([df_subset, df_targets[mask]])
            
        df_targets = df_subset.copy()
        print(f"Reduced conditions to {len(df_targets)}.")
    
    print("Loading Token Probes...")
    token_path = os.path.join(DIRS["exp6c_root"], "token_vectors", "EXP6C_TOKEN_CAPABILITY_VECTORS.parquet")
    df_tokens = pd.read_parquet(token_path)
    
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=DTYPE, device_map=DEVICE if DEVICE == "cuda:0" else None)
    if DEVICE == "mps":
        model = model.to(DEVICE)
        
    moe_blocks = [m for n, m in model.named_modules() if m.__class__.__name__ == "OlmoeSparseMoeBlock"]
    
    if mode == "calibrate":
        unique_experts = df_targets.drop_duplicates(subset=["layer_idx", "expert_idx"])
        run_zero_update_noise_floor(model, moe_blocks, df_tokens, unique_experts)
    
    print("Caching exact initial checkpoint state in CPU RAM...")
    initial_state = {name: param.clone().detach().cpu() for name, param in model.named_parameters()}
    
    results = []
    
    print(f"Beginning Controlled Interventions ({len(df_targets)} conditions)...")
    
    def run_condition(row, seed_val, is_baseline=False, override_steps=None, override_lr=None, config_name="default"):
        set_seed(seed_val)
        
        layer_idx = row["layer_idx"]
        expert_idx = row["expert_idx"]
        tau_target = np.array(row["tau_actual"])
        
        c_before = probe_expert_capability(model, moe_blocks, df_tokens, layer_idx, expert_idx)
        mag_c_before = np.linalg.norm(c_before)
        c_hat = c_before / mag_c_before if mag_c_before > 0 else np.zeros(10)
        
        target_expert_module = moe_blocks[layer_idx].experts[expert_idx]
        
        model.train()
        frozen_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            if f"layers.{layer_idx}.mlp.experts.{expert_idx}." in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()
                
        active_lr = override_lr if override_lr is not None else LR
        active_steps = override_steps if override_steps is not None else UPDATE_STEPS
        
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=active_lr)
        
        router_hook = RouterOverrideHook(expert_idx)
        router_hook.register(moe_blocks[layer_idx].gate)
        
        empirical_axis_counts = np.zeros(10, dtype=np.float32)
        total_tokens_sampled = 0
        p_axis = tau_target / (np.sum(tau_target) + 1e-12)
        
        assert BATCH_SIZE % MICRO_BATCH_SIZE == 0
        
        for step in range(active_steps):
            optimizer.zero_grad()
            
            for micro_step in range(BATCH_SIZE // MICRO_BATCH_SIZE):
                batch_input_ids = []
                batch_attention_mask = []
                
                for _ in range(MICRO_BATCH_SIZE):
                    if is_baseline:
                        axis_k = np.random.choice(10)
                    else:
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
                loss = outputs.loss / (BATCH_SIZE // MICRO_BATCH_SIZE)
                loss.backward()
                
            optimizer.step()
            
        router_hook.remove()
        
        empirical_tau_proportion = empirical_axis_counts / total_tokens_sampled
        tau_actual = empirical_tau_proportion * np.linalg.norm(tau_target) 
        
        c_after = probe_expert_capability(model, moe_blocks, df_tokens, layer_idx, expert_idx)
        delta_c = c_after - c_before
        
        dc_par = np.dot(delta_c, c_hat) * c_hat
        dc_perp = delta_c - dc_par
        
        results.append({
            "layer_idx": layer_idx,
            "expert_idx": expert_idx,
            "quantile": row["quantile"],
            "alpha": row["alpha"],
            "target_angle_deg": row["target_angle_deg"],
            "is_theta_max": row["is_theta_max"],
            "seed": seed_val,
            "config_name": config_name,
            
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            
            "tau_target": tau_target.tolist(),
            "tau_actual": tau_actual.tolist(),
            "err_tau": np.linalg.norm(tau_target - tau_actual),
            "cos_tau": np.dot(tau_target, tau_actual) / (np.linalg.norm(tau_target)*np.linalg.norm(tau_actual) + 1e-12),
            "mag_tau_actual": np.linalg.norm(tau_actual),
            
            "C_before": c_before.tolist(),
            "mag_C_before": mag_c_before,
            "C_after": c_after.tolist(),
            "mag_C_after": np.linalg.norm(c_after),
            
            "DeltaC": delta_c.tolist(),
            "mag_DeltaC": np.linalg.norm(delta_c),
            "mag_DeltaC_par": np.linalg.norm(dc_par),
            "mag_DeltaC_perp": np.linalg.norm(dc_perp),
            "delta_theta": angle_between(c_before, c_after)
        })
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(initial_state[name].to(DEVICE))
                    
    if mode == "calibrate":
        for cal_config in CALIBRATION_CONFIGS:
            c_name = cal_config["name"]
            c_steps = cal_config["steps"]
            c_lr = cal_config["lr"]
            print(f"\n--- Running Calibration Config {c_name} (Steps: {c_steps}, LR: {c_lr}) ---")
            for idx, row in tqdm(df_targets.iterrows(), total=len(df_targets), desc=f"Config {c_name}"):
                run_condition(row, SEEDS[0], override_steps=c_steps, override_lr=c_lr, config_name=c_name)
    else:
        for idx, row in tqdm(df_targets.iterrows(), total=len(df_targets), desc="Standard Interventions"):
            run_condition(row, SEEDS[0])
            
        if mode == "pilot":
            print(f"Running Control E Replications...")
            for seed_val in SEEDS[1:]:
                for idx, row in tqdm(df_targets.iterrows(), total=len(df_targets), desc=f"Replication Seed {seed_val}"):
                    run_condition(row, seed_val)
        else:
            unique_experts = df_targets.drop_duplicates(subset=["layer_idx", "expert_idx"]).copy()
            for _, exp_row in tqdm(unique_experts.iterrows(), total=len(unique_experts), desc="Controls A & B"):
                row_A = exp_row.copy()
                row_A["target_angle_deg"] = -1.0
                row_A["alpha"] = 1.0
                run_condition(row_A, SEEDS[0], is_baseline=True)
                
                row_B = exp_row.copy()
                row_B["target_angle_deg"] = -2.0
                row_B["alpha"] = 1.0
                row_B["tau_actual"] = np.ones(10)
                run_condition(row_B, SEEDS[0])
                
            critical_conditions = df_targets[
                (df_targets["alpha"].isin([0.01, 2.00])) & 
                ((df_targets["target_angle_deg"] == 0) | (df_targets["is_theta_max"] == True))
            ]
            for seed_val in SEEDS[1:]:
                for idx, row in tqdm(critical_conditions.iterrows(), total=len(critical_conditions), desc=f"Replication Seed {seed_val}"):
                    run_condition(row, seed_val)
            
    df_res = pd.DataFrame(results)
    
    if mode == "calibrate":
        out_path = os.path.join(DIRS["results"], "EXP6D_CALIBRATION_RESULTS.parquet")
    elif mode == "pilot":
        out_path = os.path.join(DIRS["results"], "EXP6D_PILOT_RESULTS.parquet")
    else:
        out_path = os.path.join(DIRS["results"], "EXP6D_RAW_RESULTS.parquet")
        
    df_res.to_parquet(out_path)
    print(f"Saved {len(results)} intervention results to {out_path}")

if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        main(mode="calibrate")
    elif "--pilot" in sys.argv:
        main(mode="pilot")
    elif "--full" in sys.argv:
        main(mode="full")
    else:
        print("Run with --calibrate, --pilot, or --full")
        sys.exit(1)
