import os
import sys
import argparse
import traceback
import json
import time
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

# Add REAP to path
REAP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "reap", "src"))
sys.path.insert(0, REAP_PATH)

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from benchmark.run_benchmark import load_config
from benchmark.core.evaluate import prepare_wikitext_eval_batches, compute_ppl
from benchmark.core.logging import BenchmarkLogger

from reap.data import DATASET_REGISTRY, LMDatasetProcessor

class WikitextLMDataset(LMDatasetProcessor):
    category_field: str = None
    @staticmethod
    def _map_fn(sample: dict) -> dict:
        return sample

DATASET_REGISTRY["wikitext"] = WikitextLMDataset


def detect_olmoe_moe_class(model):
    """Dynamically detect the MoE block class name in the loaded OLMoE model."""
    for name, module in model.named_modules():
        if hasattr(module, "experts") and (hasattr(module, "gate") or hasattr(module, "router")):
            cls_name = type(module).__name__
            print(f"[REAP] Detected MoE block class: {cls_name}")
            return cls_name
    return None


def register_olmoe_in_reap(model):
    """
    Dynamically register OLMoE in REAP's MODEL_ATTRS and OBSERVER_CONFIG_REGISTRY.
    This is NOT modifying the algorithm — it's just telling REAP where the model components are.
    OLMoE uses the exact same expert structure (gate_proj/up_proj/down_proj + gate router) as Qwen3.
    """
    from reap.model_util import MODEL_ATTRS
    from reap.observer import OBSERVER_CONFIG_REGISTRY, MoETransformerObserverConfig

    model_cls_name = type(model).__name__
    moe_cls_name = detect_olmoe_moe_class(model)

    # REAP dynamically accesses `num_experts` and `num_experts_per_tok` on the MoE block.
    for name, module in model.named_modules():
        if type(module).__name__ == moe_cls_name:
            if not hasattr(module.__class__, 'num_experts'):
                module.__class__.num_experts = property(lambda self: self.experts.num_experts)
            if not hasattr(module.__class__, 'num_experts_per_tok'):
                module.__class__.num_experts_per_tok = property(lambda self: getattr(self.gate, "top_k", 8) if hasattr(self, "gate") else 8)
            if not hasattr(module.__class__, 'router'):
                module.__class__.router = property(lambda self: getattr(self, "gate", None))
            break

    if model_cls_name not in MODEL_ATTRS:
        print(f"[REAP] Registering {model_cls_name} in MODEL_ATTRS...")
        MODEL_ATTRS[model_cls_name] = {
            "moe_block": "mlp",
            "gate_proj": "gate_up_proj",
            "up_proj": "gate_up_proj",
            "down_proj": "down_proj",
            "experts": "experts",
            "fused": True,
            "router": "gate",
            "num_experts": "num_experts",
            "num_experts_per_tok": "num_experts_per_tok",
        }

    if model_cls_name not in OBSERVER_CONFIG_REGISTRY:
        if moe_cls_name:
            print(f"[REAP] Registering observer config for {model_cls_name} (MoE block: {moe_cls_name})...")

            @dataclass
            class OLMoEObserverHookConfig(MoETransformerObserverConfig):
                module_class_name_to_hook_regex: str = moe_cls_name
                top_k_attr_name: str = "num_experts_per_tok"
                fused_experts: bool = True

            OBSERVER_CONFIG_REGISTRY[model_cls_name] = OLMoEObserverHookConfig
        else:
            raise RuntimeError(f"Could not detect MoE block class for {model_cls_name}")

    return model_cls_name, moe_cls_name


def run_reap_method(method, config):
    """
    Run an external method from the REAP family (REAP, HC-SMoE, M-SMoE, Sub-MoE) on OLMoE.
    
    Strategy:
    1. Load the model
    2. Dynamically register OLMoE in REAP's config registries
    3. Run the REAP observer to collect activation statistics
    4. Run the appropriate clustering method
    5. Run the merge
    6. Evaluate PPL on the merged model
    """
    print(f"\n[{method.upper()}] Attempting to apply method to {config['model']['name']}...")

    out_dir = os.path.join(PROJECT_ROOT, "benchmark_results", method)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    print(f"[{method.upper()}] Loading model...")
    dtype = torch.bfloat16 if config['model']['precision'] == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=config['model']['device'],
        offload_folder="offload"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dynamically register OLMoE
    try:
        model_cls, moe_cls = register_olmoe_in_reap(model)
    except Exception as e:
        error_msg = f"Failed to register OLMoE in REAP: {e}"
        print(f"[ERROR] {error_msg}")
        with open(os.path.join(out_dir, "error.log"), "w") as f:
            f.write(error_msg + "\n" + traceback.format_exc())
        with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
            json.dump({"error": "not reproduced on OLMoE", "message": error_msg}, f, indent=2)
        return False

    # Map method names to REAP clustering configs
    method_configs = {
        "reap": {
            "expert_sim": "characteristic_activation",
            "cluster_method": "agglomerative",
            "linkage_method": "complete",
            "description": "REAP (Characteristic Activation + Agglomerative)"
        },
        "hc_smoe": {
            "expert_sim": "router_logits",
            "cluster_method": "agglomerative",
            "linkage_method": "ward",
            "description": "HC-SMoE (Router Logits + Hierarchical Clustering)"
        },
        "m_smoe": {
            "expert_sim": "characteristic_activation",
            "cluster_method": "agglomerative",
            "linkage_method": "ward",
            "description": "M-SMoE (CA + Ward Clustering)"
        },
        "sub_moe": {
            "expert_sim": "characteristic_activation",
            "cluster_method": "kmeans",
            "linkage_method": None,
            "description": "Sub-MoE (CA + K-Means)"
        },
    }

    if method not in method_configs:
        print(f"[ERROR] Unknown REAP family method: {method}")
        return False

    mcfg = method_configs[method]
    print(f"[{method.upper()}] Running: {mcfg['description']}")

    try:
        from reap.observer import OBSERVER_CONFIG_REGISTRY, MoETransformerObserver
        from reap.cluster import hierarchical_clustering, kmeans_clustering
        from reap.merge import MergeMethod, MoEExpertMerger
        from reap.model_util import get_moe, MODEL_ATTRS
        from reap.data import load_category_batches
        from reap.metrics import get_distance_fn

        # Step 1: Record activations (observer)
        print(f"[{method.upper()}] Step 1: Recording activations...")
        observer_config = OBSERVER_CONFIG_REGISTRY[model_cls](
            distance_measure="cosine",
            renormalize_router_weights=False,
            record_pruning_metrics_only=False,
        )
        observer = MoETransformerObserver(model=model, hook_config=observer_config)

        # Fix OverflowError: explicitly set tokenizer.model_max_length so it doesn't 
        # evaluate to a massive fallback integer that crashes Rust bindings.
        tokenizer.model_max_length = 512
        # Load calibration data using REAP's data loader
        cal_batches = load_category_batches(
            dataset_name="wikitext",
            split="train",
            subset="wikitext-2-raw-v1",
            tokenizer=tokenizer,
            model_max_length=512,
            split_by_category=False,
            return_vllm_tokens_prompt=False,
            truncate=True,
            batches_per_category=32,
            batch_size=4,
        )

        with torch.no_grad():
            for batch in cal_batches.get("all", []):
                batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                attention_mask = batch.get("attention_mask", None)
                with observer.set_attention_mask(attention_mask):
                    model(**batch)

        observer_data = observer.report_state()
        observer.close_hooks()
        print(f"[{method.upper()}] Observer collected data for {len(observer_data)} layers.")

        # Step 2: Baseline PPL
        eval_chunks = prepare_wikitext_eval_batches(tokenizer, config)
        ppl_64 = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'])
        print(f"[{method.upper()}] Baseline PPL @ 64: {ppl_64:.4f}")

        # Step 3: Cluster and merge for each compression target
        logger = BenchmarkLogger(os.path.join(out_dir, "trajectory.json"), config)
        logger.log_ppl(64, ppl_64)

        for target in config['compression']['checkpoints']:
            if target >= 64:
                continue

            print(f"\n[{method.upper()}] Compressing to {target} experts per layer...")
            start_time = time.time()

            # Cluster
            cluster_labels = {}
            for layer in observer_data:
                ca = observer_data[layer]["characteristic_activation"]
                distance_fn = get_distance_fn("cosine")
                distance = distance_fn(ca.unsqueeze(0), ca.unsqueeze(1))

                if mcfg["cluster_method"] == "agglomerative":
                    labels = hierarchical_clustering(distance, mcfg["linkage_method"], target)
                elif mcfg["cluster_method"] == "kmeans":
                    labels = kmeans_clustering(ca, target)

                if isinstance(labels, torch.Tensor):
                    cluster_labels[layer] = labels
                else:
                    cluster_labels[layer] = torch.tensor(labels)

            # Merge
            model_attrs = MODEL_ATTRS[model_cls]
            for layer in cluster_labels:
                expert_proba = observer_data[layer]["expert_frequency"] / observer_data[layer]["total_tokens"]
                moe = get_moe(model, layer)
                merger = MoEExpertMerger(
                    moe=moe,
                    cluster_label=cluster_labels[layer],
                    expert_proba=expert_proba,
                    model_attrs=model_attrs,
                    merge_method=MergeMethod("average"),
                    dom_as_base=False,
                    select_top_k=False,
                    permute=False,
                    tie_tensors=False,
                )
                merger.merge_experts()

            step_time = time.time() - start_time
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

            # Evaluate
            ppl = compute_ppl(model, eval_chunks, batch_size=config['evaluation']['batch_size'])
            logger.log_ppl(target, ppl)

            logger.log_step({
                "method": method,
                "seed": None,
                "step": 64 - target,
                "experts_before": 64,
                "experts_after": target,
                "candidate_pairs": None,
                "selected_pair": None,
                "selection_score": None,
                "capability_distance": None,
                "marginal_kl": None,
                "cumulative_kl": None,
                "wall_time_sec": float(step_time),
                "peak_memory_mb": float(peak_mem)
            })

            print(f"[{method.upper()}] PPL @ {target}: {ppl:.4f} (Time: {step_time:.2f}s)")

            # IMPORTANT: Reload model for next compression target (since merging is destructive)
            if target != min(config['compression']['checkpoints']):
                del model
                torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(
                    config['model']['name'],
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    device_map=config['model']['device'],
                    offload_folder="offload"
                )
                model.eval()
                register_olmoe_in_reap(model)

        print(f"\n[{method.upper()}] Completed successfully!")
        return True

    except Exception as e:
        error_msg = f"Runtime error running {method}: {e}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        with open(os.path.join(out_dir, "error.log"), "w") as f:
            f.write(error_msg + "\n" + traceback.format_exc())
        with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
            json.dump({"error": str(e), "message": error_msg}, f, indent=2)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    config = load_config()
    run_reap_method(args.method, config)
