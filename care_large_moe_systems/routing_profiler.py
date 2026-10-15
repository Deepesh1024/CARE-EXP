"""
care_large_moe_systems/routing_profiler.py
Captures routing statistics, expert load distribution, routing entropy,
and load imbalance across layers without interfering with latency timing.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class RoutingProfiler:
    def __init__(self, model, num_experts: int, top_k: int = 8):
        self.model = model
        self.num_experts = num_experts
        self.top_k = top_k
        self.hooks = []
        self.layer_assignments: Dict[int, List[torch.Tensor]] = {}
        
    def _find_moe_layers(self) -> List[Tuple[int, nn.Module]]:
        moe_layers = []
        idx = 0
        for name, module in self.model.named_modules():
            # Check for gate/router in block
            if hasattr(module, "experts") and (hasattr(module, "gate") or hasattr(module, "router")):
                gate_module = getattr(module, "gate", getattr(module, "router", None))
                moe_layers.append((idx, gate_module))
                idx += 1
        return moe_layers

    def attach_hooks(self):
        self.clear()
        moe_layers = self._find_moe_layers()
        
        for layer_idx, gate_module in moe_layers:
            self.layer_assignments[layer_idx] = []
            
            def make_hook(l_idx):
                def hook_fn(module, input, output):
                    # Gate output can be a tuple (logits, routing_weights) or a raw tensor
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    # Shape of logits: [batch, seq_len, num_experts] or [batch*seq_len, num_experts]
                    with torch.no_grad():
                        flat_logits = logits.view(-1, logits.shape[-1])
                        topk = torch.topk(flat_logits, k=min(self.top_k, flat_logits.shape[-1]), dim=-1).indices
                        self.layer_assignments[l_idx].append(topk.detach().cpu())
                return hook_fn
                
            h = gate_module.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def clear(self):
        self.remove_hooks()
        self.layer_assignments.clear()

    def profile_workload(self, tokenizer, prompt: str, max_new_tokens: int = 64, batch_size: int = 1) -> Dict[str, Any]:
        """
        Runs generation with hooks attached and computes per-layer routing metrics.
        """
        self.attach_hooks()
        
        device = next(self.model.parameters()).device
        encoded = tokenizer([prompt] * batch_size, return_tensors="pt").to(device)
        
        with torch.no_grad():
            self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            
        self.remove_hooks()
        
        # Analyze captured routing statistics
        results = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "layers": {}
        }
        
        all_layer_entropies = []
        all_layer_imbalances = []
        
        for layer_idx, tensor_list in self.layer_assignments.items():
            if not tensor_list:
                continue
            all_topk = torch.cat(tensor_list, dim=0) # [total_tokens, top_k]
            flat_indices = all_topk.flatten().numpy()
            total_selections = len(flat_indices)
            
            # Count assignments per expert
            counts = np.bincount(flat_indices, minlength=self.num_experts)
            probs = counts / max(total_selections, 1)
            
            # Non-zero probabilities for entropy
            nz_probs = probs[probs > 0]
            entropy = -np.sum(nz_probs * np.log2(nz_probs)) if len(nz_probs) > 0 else 0.0
            max_entropy = math.log2(self.num_experts) if self.num_experts > 1 else 1.0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
            
            # Load imbalance (Coefficient of Variation: std / mean)
            mean_load = np.mean(counts)
            std_load = np.std(counts)
            load_imbalance_cv = float(std_load / mean_load) if mean_load > 0 else 0.0
            peak_to_mean = float(np.max(counts) / mean_load) if mean_load > 0 else 1.0
            active_expert_pct = float(np.count_nonzero(counts) / self.num_experts) * 100.0
            
            all_layer_entropies.append(norm_entropy)
            all_layer_imbalances.append(load_imbalance_cv)
            
            results["layers"][f"layer_{layer_idx}"] = {
                "total_token_selections": int(total_selections),
                "expert_counts": [int(c) for c in counts],
                "routing_entropy_bits": round(float(entropy), 4),
                "normalized_routing_entropy": round(float(norm_entropy), 4),
                "load_imbalance_cv": round(load_imbalance_cv, 4),
                "peak_to_mean_ratio": round(peak_to_mean, 4),
                "active_expert_coverage_pct": round(active_expert_pct, 2)
            }
            
        results["mean_normalized_routing_entropy"] = round(float(np.mean(all_layer_entropies)), 4) if all_layer_entropies else 1.0
        results["mean_load_imbalance_cv"] = round(float(np.mean(all_layer_imbalances)), 4) if all_layer_imbalances else 0.0
        
        return results
