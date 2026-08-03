"""
CARE-MoE V3 -- Experiment 1
============================
Research question (the only thing this script answers):
    "Can existing expert-similarity metrics predict the true capability loss
     caused by merging experts in a pretrained Mixture-of-Experts model?"

Explicit non-goals (do not violate these while editing this file):
    - Do NOT design a new compression algorithm.
    - Do NOT implement CARE.
    - Do NOT invent a new proxy metric.

Everything here is either (a) a well-known, pre-existing similarity metric,
or (b) the oracle ground truth obtained by actually merging two experts and
measuring real output degradation. The only thing under test is whether
(a) correlates with (b).

Requires a CUDA GPU. The oracle sweep runs two full forward passes over the
calibration set (baseline + merged) PER expert pair -- see the runtime note
above run_oracle_pair(). This is intentionally expensive; do not "optimize"
it by caching the baseline distribution (see that same note for why).

Output layout
-------------
    output/
        output.json              <- ALL text/metric output (config, architecture
                                     summary, usage frequency, per-pair metrics,
                                     oracle matrix, correlations, ranked list)
        oracle_heatmap.png
        correlation_matrix.png
        ranked_merge_list.png
        scatterplots/
            <metric>_vs_oracle_kl.png   (one per existing metric)

Resumability
------------
This script checkpoints to output/output.json every CHECKPOINT_EVERY_N_PAIRS
pairs (default 50), plus once more at the end. If it dies or gets pre-empted
(shared server, walltime limit, OOM, SSH drop -- doesn't matter which), just
rerun it: already-completed pairs are loaded from output.json and skipped,
not recomputed. Worst case you lose the pairs since the last checkpoint
(at most CHECKPOINT_EVERY_N_PAIRS of them), not the whole run.
"""

import os
import json
import math
import copy
import random
import signal
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")  # headless -- no display on a server, just write files
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIG -- the only knobs you should need to touch
# ============================================================
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
SEED = 42
NUM_SEQUENCES = 256          # calibration sequences (spec requirement)
MAX_LENGTH = 512             # tokens per sequence (spec requirement)

LAYER_INDEX = 0              # <-- which MoE layer to evaluate (0-indexed)

# Oracle sweep is O(num_expert_pairs) full forward passes and is expensive.
# Qwen1.5-MoE-A2.7B has 60 routed experts per layer -> C(60,2) = 1770 pairs.
# Set MAX_PAIRS to a small int (e.g. 20) for a smoke test before committing
# to the full sweep. Set to None to run every pair (the actual experiment).
MAX_PAIRS = None

# Batch size used for calibration forward passes (memory/speed tradeoff only --
# does not change any metric definition).
CALIB_BATCH_SIZE = 4

# Number of calibration tokens used for the (cheap, one-forward-per-expert)
# output/activation similarity metrics. These are computed ONCE per expert
# (not per pair) and then compared pairwise, so this can be generous without
# blowing up runtime. Reduce this first if you hit GPU memory limits.
EVAL_TOKENS_FOR_EXPERT_METRICS = 4096

# GPU index WITHIN whatever CUDA_VISIBLE_DEVICES exposes to this process.
# On a Slurm/K8s job allocated exactly one GPU, that GPU is already index 0
# in-process (that's what CUDA_VISIBLE_DEVICES remapping does) -- so the
# default below is correct out of the box. Override via env var only if a
# launcher assigns you a specific slot within a multi-GPU allocation
# (e.g. LOCAL_RANK in a multi-process launch).
GPU_ID = int(os.environ.get("CARE_MOE_GPU_ID", 0))
DEVICE = f"cuda:{GPU_ID}"
DTYPE = torch.float16

# Checkpoint frequency: write output.json every N completed pairs, plus once
# more at the very end. Trades a bounded amount of lost work on crash/pre-emption
# (at most N pairs) for far less I/O than checkpointing every single pair.
CHECKPOINT_EVERY_N_PAIRS = 50

OUTPUT_DIR = "./output"
SCATTER_DIR = os.path.join(OUTPUT_DIR, "scatterplots")
RESULTS_JSON_PATH = os.path.join(OUTPUT_DIR, "output.json")

METRIC_COLS = [
    "Weight_Distance", "Weight_Cosine", "Activation_Similarity",
    "Output_Similarity", "Routing_Similarity", "Usage_Frequency",
]

# Set by _handle_shutdown_signal below. Checked inside the main sweep loop so
# a SIGTERM (AWS Spot reclaim, systemd stop) or SIGINT (Ctrl+C) triggers an
# immediate checkpoint instead of silently losing work since the last
# periodic save. This is what actually makes CHECKPOINT_EVERY_N_PAIRS safe
# to use on Spot -- without it, the 2-minute Spot reclaim warning is wasted.
_shutdown_requested = False


def _handle_shutdown_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\nReceived signal {signum} -- finishing current pair, checkpointing, and exiting. "
          f"Rerun the script afterward to resume from output.json.")


signal.signal(signal.SIGTERM, _handle_shutdown_signal)
signal.signal(signal.SIGINT, _handle_shutdown_signal)


# ============================================================
# output.json helpers
# ============================================================
def load_checkpoint():
    """Returns the existing output.json dict if present, else a fresh skeleton."""
    if os.path.exists(RESULTS_JSON_PATH):
        with open(RESULTS_JSON_PATH, "r") as f:
            data = json.load(f)
        print(f"Found existing checkpoint at {RESULTS_JSON_PATH} "
              f"with {len(data.get('results', []))} completed pairs -- resuming.")
        return data
    return {
        "config": {},
        "architecture_summary": {},
        "usage_frequency": {},
        "results": [],
        "oracle_matrix": None,
        "correlations": None,
        "ranked_merge_candidates": None,
    }


def save_checkpoint(data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_path = RESULTS_JSON_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, RESULTS_JSON_PATH)  # atomic-ish write, avoids truncated file on crash mid-write


# ============================================================
# Model / architecture
# ============================================================
def find_moe_layers(model):
    """
    Generic detector (not hard-coded to Qwen2Moe internals): returns every
    sub-module in the model that exposes a ModuleList called `experts`.
    This is what lets LAYER_INDEX refer to "the Nth MoE layer" without
    hand-coding transformer block indices.
    """
    found = []
    for name, module in model.named_modules():
        if hasattr(module, "experts") and isinstance(module.experts, torch.nn.ModuleList):
            found.append((name, module))
    return found


# ============================================================
# Merge operator (modular -- swap the concrete class to test a different
# strategy later; nothing downstream depends on the implementation)
# ============================================================
class MergeOperator:
    def merge(self, expert_a: torch.nn.Module, expert_b: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError


class ParameterAveragingMerge(MergeOperator):
    """
    BASELINE operator (spec requirement): elementwise average of every
    parameter tensor between two experts. Nothing more sophisticated --
    this experiment exists to test EXISTING similarity metrics against this
    baseline, not to build a better merge function.
    """
    def merge(self, expert_a, expert_b):
        merged = copy.deepcopy(expert_a)
        with torch.no_grad():
            for (name_m, p_m), (name_a, p_a), (name_b, p_b) in zip(
                merged.named_parameters(),
                expert_a.named_parameters(),
                expert_b.named_parameters(),
            ):
                assert name_m == name_a == name_b, "Parameter name mismatch between experts"
                p_m.copy_(0.5 * p_a.float() + 0.5 * p_b.float())
        return merged


# ============================================================
# Weight-space metrics
# ============================================================
def flatten_expert_weights(expert):
    return torch.cat([p.detach().flatten().float() for p in expert.parameters()])


# ============================================================
# Calibration activation capture (hidden states + router logits)
# ============================================================
@torch.no_grad()
def collect_calibration_activations(model, moe_block, input_ids, attention_mask, batch_size):
    hidden_list, router_list, spans = [], [], []

    def pre_hook(module, inputs):
        hidden_list.append(inputs[0].detach())

    def gate_hook(module, inputs, output):
        router_list.append(output.detach())

    h1 = moe_block.register_forward_pre_hook(pre_hook)
    h2 = moe_block.gate.register_forward_hook(gate_hook)

    n = input_ids.shape[0]
    for start in tqdm(range(0, n, batch_size), desc="Capturing layer activations"):
        end = min(start + batch_size, n)
        ids = input_ids[start:end].to(DEVICE)
        mask = attention_mask[start:end].to(DEVICE)
        model(input_ids=ids, attention_mask=mask)
        spans.append((start, end))

    h1.remove()
    h2.remove()

    hidden_all, router_all = [], []
    for hs, rl, (start, end) in zip(hidden_list, router_list, spans):
        mask = attention_mask[start:end]
        b, seq, hid = hs.shape
        hs_flat = hs.reshape(b * seq, hid)
        mask_flat = mask.reshape(b * seq).bool()
        hidden_all.append(hs_flat[mask_flat].cpu())
        router_all.append(rl.reshape(b * seq, -1)[mask_flat].cpu())

    return torch.cat(hidden_all, dim=0), torch.cat(router_all, dim=0)


# ============================================================
# Oracle -- ground-truth capability loss from actually merging a pair
#
# WHY BASELINE IS RECOMPUTED PER PAIR INSTEAD OF CACHED: caching the full
# (sequences x tokens x vocab) baseline distribution for this calibration
# set would need on the order of 40GB -- not a reasonable footprint for a
# single-GPU run. Recomputing it per pair is exact (not an approximation);
# it just trades compute time for memory safety. This is why the oracle
# sweep dominates total runtime -- do not "fix" this by caching it.
# ============================================================
def install_experts(moe_block, idx_list, modules):
    for idx, mod in zip(idx_list, modules):
        moe_block.experts[idx] = mod


@torch.no_grad()
def run_oracle_pair(model, moe_block, i, j, merge_operator, input_ids, attention_mask, batch_size):
    orig_i = moe_block.experts[i]
    orig_j = moe_block.experts[j]

    merged_expert = merge_operator.merge(orig_i, orig_j).to(DEVICE, dtype=DTYPE)

    n = input_ids.shape[0]
    total_tokens = 0
    kl_sum = 0.0
    ce_orig_sum = 0.0
    ce_merged_sum = 0.0
    top1_agree_sum = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ids = input_ids[start:end].to(DEVICE)
        mask = attention_mask[start:end].to(DEVICE)

        install_experts(moe_block, [i, j], [orig_i, orig_j])
        logits_orig = model(input_ids=ids, attention_mask=mask).logits

        install_experts(moe_block, [i, j], [merged_expert, merged_expert])
        logits_merged = model(input_ids=ids, attention_mask=mask).logits

        shift_logits_orig = logits_orig[:, :-1, :]
        shift_logits_merged = logits_merged[:, :-1, :]
        shift_labels = ids[:, 1:]
        shift_mask = mask[:, 1:].bool()

        logp_orig = F.log_softmax(shift_logits_orig.float(), dim=-1)
        logp_merged = F.log_softmax(shift_logits_merged.float(), dim=-1)
        p_orig = logp_orig.exp()

        kl_tok = (p_orig * (logp_orig - logp_merged)).sum(dim=-1)
        kl_sum += kl_tok[shift_mask].sum().item()

        ce_orig_tok = F.nll_loss(
            logp_orig.reshape(-1, logp_orig.size(-1)), shift_labels.reshape(-1), reduction="none"
        ).reshape(shift_labels.shape)
        ce_merged_tok = F.nll_loss(
            logp_merged.reshape(-1, logp_merged.size(-1)), shift_labels.reshape(-1), reduction="none"
        ).reshape(shift_labels.shape)
        ce_orig_sum += ce_orig_tok[shift_mask].sum().item()
        ce_merged_sum += ce_merged_tok[shift_mask].sum().item()

        top1_orig = shift_logits_orig.argmax(dim=-1)
        top1_merged = shift_logits_merged.argmax(dim=-1)
        top1_agree_sum += (top1_orig == top1_merged)[shift_mask].sum().item()

        total_tokens += shift_mask.sum().item()

        del logits_orig, logits_merged, logp_orig, logp_merged, p_orig
        # NOTE: deliberately NOT calling torch.cuda.empty_cache() here.
        # That forces a device sync and flushes the caching allocator on every
        # batch -- it destroys throughput. Let PyTorch's allocator reuse freed
        # blocks itself. If this loop OOMs, lower CALIB_BATCH_SIZE instead.

    # Restore the original, unmodified experts before returning.
    install_experts(moe_block, [i, j], [orig_i, orig_j])

    mean_kl = kl_sum / total_tokens
    mean_ce_orig = ce_orig_sum / total_tokens
    mean_ce_merged = ce_merged_sum / total_tokens

    return {
        "Oracle_KL": mean_kl,
        "CrossEntropy_Delta": mean_ce_merged - mean_ce_orig,
        "Perplexity_Delta": math.exp(mean_ce_merged) - math.exp(mean_ce_orig),
        "Top1_Agreement": top1_agree_sum / total_tokens,
    }


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCATTER_DIR, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    assert torch.cuda.is_available(), "This script requires a CUDA GPU."
    print(f"CUDA device (index {GPU_ID}):", torch.cuda.get_device_name(GPU_ID))

    checkpoint = load_checkpoint()
    checkpoint["config"] = {
        "MODEL_ID": MODEL_ID, "SEED": SEED, "NUM_SEQUENCES": NUM_SEQUENCES,
        "MAX_LENGTH": MAX_LENGTH, "LAYER_INDEX": LAYER_INDEX, "MAX_PAIRS": MAX_PAIRS,
        "CALIB_BATCH_SIZE": CALIB_BATCH_SIZE,
        "EVAL_TOKENS_FOR_EXPERT_METRICS": EVAL_TOKENS_FOR_EXPERT_METRICS,
    }

    # ---------------- Model ----------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model in FP16 on CUDA (device index {GPU_ID}, cached after first run)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map={"": GPU_ID},   # whole model on one GPU -- required for in-place expert swapping below
    )
    model.eval()
    print("Model loaded.")

    # ---------------- Architecture detection ----------------
    moe_layers = find_moe_layers(model)
    cfg = model.config

    arch_summary = {
        "num_moe_layers": len(moe_layers),
        "experts_per_layer": len(moe_layers[0][1].experts),
        "hidden_dimension": cfg.hidden_size,
        "expert_ffn_dimension": getattr(cfg, "moe_intermediate_size", getattr(cfg, "intermediate_size", None)),
        "routing_type": "top-k softmax gating (learned linear router per layer)",
        "top_k": getattr(cfg, "num_experts_per_tok", None),
        "norm_topk_prob": getattr(cfg, "norm_topk_prob", None),
        "shared_expert_present": hasattr(moe_layers[0][1], "shared_expert"),
    }
    checkpoint["architecture_summary"] = arch_summary
    print("=" * 70)
    print("MoE ARCHITECTURE SUMMARY")
    print("=" * 70)
    for k, v in arch_summary.items():
        print(f"{k:28s}: {v}")
    print("=" * 70)
    save_checkpoint(checkpoint)

    # ---------------- Dataset ----------------
    print("Loading WikiText-2...")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in raw["text"] if len(t.strip()) > 0]

    rng = random.Random(SEED)
    rng.shuffle(texts)

    tokenized_sequences = []
    for t in texts:
        if len(tokenized_sequences) >= NUM_SEQUENCES:
            break
        enc = tokenizer(
            t, truncation=True, max_length=MAX_LENGTH,
            padding="max_length", return_tensors="pt",
        )
        if enc["attention_mask"].sum().item() < 8:
            continue
        tokenized_sequences.append(enc)

    assert len(tokenized_sequences) == NUM_SEQUENCES, (
        f"Only found {len(tokenized_sequences)} usable sequences out of the requested "
        f"{NUM_SEQUENCES}."
    )

    calib_input_ids = torch.cat([s["input_ids"] for s in tokenized_sequences], dim=0)
    calib_attn_mask = torch.cat([s["attention_mask"] for s in tokenized_sequences], dim=0)
    print(f"Calibration set: {calib_input_ids.shape[0]} sequences x {calib_input_ids.shape[1]} tokens "
          f"= {int(calib_attn_mask.sum().item())} real (non-pad) tokens")

    # ---------------- Layer selection ----------------
    layer_name, moe_block = moe_layers[LAYER_INDEX]
    experts = moe_block.experts
    num_experts = len(experts)
    print(f"Selected layer index {LAYER_INDEX} -> module '{layer_name}' ({num_experts} routed experts)")

    merge_operator = ParameterAveragingMerge()  # <-- swap this line to test a different operator later

    # ---------------- Weight-space metrics ----------------
    flat_weights = [flatten_expert_weights(e) for e in experts]

    def weight_distance(i, j):
        return torch.norm(flat_weights[i] - flat_weights[j]).item()

    def weight_cosine(i, j):
        return F.cosine_similarity(flat_weights[i].unsqueeze(0), flat_weights[j].unsqueeze(0)).item()

    # ---------------- Capture router logits + layer input activations ----------------
    calib_hidden, calib_router_logits = collect_calibration_activations(
        model, moe_block, calib_input_ids, calib_attn_mask, batch_size=CALIB_BATCH_SIZE
    )
    print(f"Captured {calib_hidden.shape[0]} valid token activations at layer {LAYER_INDEX} "
          f"(hidden dim {calib_hidden.shape[1]}, router logits over {calib_router_logits.shape[1]} experts)")

    # ---------------- Routing similarity + usage frequency ----------------
    router_probs = F.softmax(calib_router_logits.float(), dim=-1)
    topk = getattr(cfg, "num_experts_per_tok", 4)
    topk_idx = torch.topk(router_probs, k=topk, dim=-1).indices

    # Per-token boolean: was expert e in the top-k for this token? Computed once
    # per expert and reused for both the marginal usage rate AND the pairwise
    # union frequency below -- no need to recompute this inside the pair loop.
    routed_to = {e: (topk_idx == e).any(dim=-1) for e in range(num_experts)}  # each: (T,) bool tensor

    usage_counts = torch.zeros(num_experts)
    for e in range(num_experts):
        usage_counts[e] = routed_to[e].sum().item()
    usage_frequency = (usage_counts / calib_router_logits.shape[0]).numpy()  # per-expert marginal rate

    def routing_similarity(i, j):
        r, _ = pearsonr(router_probs[:, i].numpy(), router_probs[:, j].numpy())
        return r

    def union_usage_frequency(i, j):
        """
        Fraction of calibration tokens routed to expert i OR expert j.
        This -- not the mean of the two marginal rates -- is what a merged
        slot would actually see: averaging overstates traffic when the two
        experts are frequently co-activated on the same tokens, and
        understates it when their token sets are disjoint. The union is the
        exact quantity, computed directly from the same top-k membership data
        already captured, not an invented proxy.
        """
        return (routed_to[i] | routed_to[j]).float().mean().item()

    checkpoint["usage_frequency"] = {str(k): float(v) for k, v in enumerate(usage_frequency)}
    print(f"Usage frequency range: min={usage_frequency.min():.4f}, "
          f"max={usage_frequency.max():.4f}, mean={usage_frequency.mean():.4f}")
    save_checkpoint(checkpoint)

    # ---------------- Output / activation similarity (one forward per expert) ----------------
    torch.manual_seed(SEED)
    n_avail = calib_hidden.shape[0]
    n_eval = min(EVAL_TOKENS_FOR_EXPERT_METRICS, n_avail)
    subsample_idx = torch.randperm(n_avail)[:n_eval]
    eval_hidden = calib_hidden[subsample_idx].to(DEVICE, dtype=DTYPE)

    expert_outputs = {}
    expert_activations = {}

    with torch.no_grad():
        for idx, expert in enumerate(tqdm(experts, desc="Per-expert forward (output/activation)")):
            captured = {}

            def act_pre_hook(module, inputs, _captured=captured):
                _captured["act"] = inputs[0].detach()

            h = expert.down_proj.register_forward_pre_hook(act_pre_hook)
            out = expert(eval_hidden)
            h.remove()

            expert_outputs[idx] = out.detach().half().cpu()
            expert_activations[idx] = captured["act"].detach().half().cpu()

    def output_similarity(i, j):
        a = expert_outputs[i].float()
        b = expert_outputs[j].float()
        return F.cosine_similarity(a, b, dim=-1).mean().item()

    def activation_similarity(i, j):
        a = expert_activations[i].float()
        b = expert_activations[j].float()
        return F.cosine_similarity(a, b, dim=-1).mean().item()

    # ---------------- Build pair list, skipping already-completed pairs ----------------
    all_pairs = list(itertools.combinations(range(num_experts), 2))
    print(f"Total expert pairs at layer {LAYER_INDEX}: {len(all_pairs)}")

    if MAX_PAIRS is not None:
        rng2 = random.Random(SEED)
        all_pairs = rng2.sample(all_pairs, min(MAX_PAIRS, len(all_pairs)))
        print(f"MAX_PAIRS set -> subsampled to {len(all_pairs)} pairs for this run")

    completed = {(r["Expert_A"], r["Expert_B"]) for r in checkpoint["results"]}
    pairs_to_run = [(i, j) for (i, j) in all_pairs if (i, j) not in completed]
    print(f"{len(completed)} pairs already completed (resumed from checkpoint), "
          f"{len(pairs_to_run)} remaining")

    # ---------------- Metrics + oracle sweep ----------------
    # try/finally guarantees a checkpoint save on ANY exit path -- normal
    # completion, an unhandled exception, or a shutdown signal (see
    # _handle_shutdown_signal above) -- not just the periodic 50-pair boundary.
    try:
        for pair_idx, (i, j) in enumerate(tqdm(pairs_to_run, desc=f"Layer {LAYER_INDEX}: metrics + oracle per pair")):
            row = {
                "Layer": LAYER_INDEX,
                "Expert_A": i,
                "Expert_B": j,
                "Weight_Distance": weight_distance(i, j),
                "Weight_Cosine": weight_cosine(i, j),
                "Activation_Similarity": activation_similarity(i, j),
                "Output_Similarity": output_similarity(i, j),
                "Routing_Similarity": routing_similarity(i, j),
                # Usage_Frequency: fraction of calibration tokens routed to EITHER
                # expert (union of top-k membership), not the mean of their marginal
                # rates -- see union_usage_frequency() docstring for why the mean is
                # wrong. Per-expert marginal rates are still in
                # checkpoint["usage_frequency"] if you need them separately.
                "Usage_Frequency": union_usage_frequency(i, j),
            }
            row.update(run_oracle_pair(
                model, moe_block, i, j, merge_operator,
                calib_input_ids, calib_attn_mask, batch_size=CALIB_BATCH_SIZE,
            ))
            checkpoint["results"].append(row)

            # Checkpoint every N pairs, OR immediately if a shutdown was requested --
            # bounds lost work to at most N pairs in the normal case, and to ~0 pairs
            # when responding to a Spot reclaim / Ctrl+C.
            if (pair_idx + 1) % CHECKPOINT_EVERY_N_PAIRS == 0 or _shutdown_requested:
                save_checkpoint(checkpoint)

            if _shutdown_requested:
                break
    finally:
        save_checkpoint(checkpoint)  # covers normal completion AND any unhandled exception

    if _shutdown_requested:
        print(f"Exiting early after {len(checkpoint['results'])} total completed pairs "
              f"due to shutdown signal. Rerun the script to resume and finish the sweep "
              f"(final analysis/plots are only generated once the full sweep completes).")
        return

    results_df = pd.DataFrame(checkpoint["results"])
    print(f"Total completed pairs: {len(results_df)}")

    # ---------------- Oracle matrix ----------------
    oracle_matrix = np.full((num_experts, num_experts), np.nan)
    for _, r in results_df.iterrows():
        a, b = int(r["Expert_A"]), int(r["Expert_B"])
        oracle_matrix[a, b] = r["Oracle_KL"]
        oracle_matrix[b, a] = r["Oracle_KL"]

    # NaN -> None for strict JSON compliance (diagonal + any un-run pairs)
    oracle_matrix_json = [[(None if np.isnan(v) else float(v)) for v in row] for row in oracle_matrix]
    checkpoint["oracle_matrix"] = {
        "expert_labels": [f"E{k}" for k in range(num_experts)],
        "matrix": oracle_matrix_json,
    }

    # ---------------- Statistical analysis ----------------
    corr_rows = []
    for col in METRIC_COLS:
        pear_r, pear_p = pearsonr(results_df[col], results_df["Oracle_KL"])
        spear_r, spear_p = spearmanr(results_df[col], results_df["Oracle_KL"])
        corr_rows.append({
            "Metric": col,
            "Pearson_r": pear_r, "Pearson_p": pear_p,
            "Spearman_r": spear_r, "Spearman_p": spear_p,
        })
    corr_df = pd.DataFrame(corr_rows).reindex(
        pd.DataFrame(corr_rows)["Pearson_r"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    checkpoint["correlations"] = corr_df.to_dict(orient="records")

    print("Correlation with Oracle_KL (ranked by |Pearson r|):")
    print(corr_df.to_string(index=False))

    # ---------------- Ranked merge candidates ----------------
    ranked = results_df.sort_values("Oracle_KL", ascending=True).reset_index(drop=True)
    checkpoint["ranked_merge_candidates"] = ranked.to_dict(orient="records")

    save_checkpoint(checkpoint)
    print(f"All text/metric output saved to {RESULTS_JSON_PATH}")

    # ---------------- Visualization (all files under OUTPUT_DIR) ----------------
    oracle_matrix_df = pd.DataFrame(
        oracle_matrix,
        index=[f"E{k}" for k in range(num_experts)],
        columns=[f"E{k}" for k in range(num_experts)],
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(oracle_matrix_df, cmap="viridis", square=True)
    plt.title(f"Oracle KL divergence - Layer {LAYER_INDEX} (expert x expert)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "oracle_heatmap.png"), dpi=150)
    plt.close()

    full_corr = results_df[METRIC_COLS + ["Oracle_KL"]].corr(method="pearson")
    plt.figure(figsize=(8, 6))
    sns.heatmap(full_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Pearson correlation matrix - existing metrics vs Oracle_KL")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"), dpi=150)
    plt.close()

    for col in METRIC_COLS:
        r_val = corr_df.loc[corr_df["Metric"] == col, "Pearson_r"].values[0]
        plt.figure(figsize=(5, 4))
        plt.scatter(results_df[col], results_df["Oracle_KL"], alpha=0.5, s=12)
        plt.xlabel(col)
        plt.ylabel("Oracle_KL")
        plt.title(f"{col} vs Oracle_KL (Pearson r = {r_val:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(SCATTER_DIR, f"{col}_vs_oracle_kl.png"), dpi=150)
        plt.close()

    top_n = min(20, len(ranked))
    labels = [f"E{a}-E{b}" for a, b in zip(ranked["Expert_A"][:top_n], ranked["Expert_B"][:top_n])]
    plt.figure(figsize=(8, max(4, top_n * 0.3)))
    plt.barh(labels[::-1], ranked["Oracle_KL"][:top_n][::-1])
    plt.xlabel("Oracle_KL (lower = safer merge)")
    plt.title(f"Top {top_n} lowest-capability-loss merge candidates - Layer {LAYER_INDEX}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ranked_merge_list.png"), dpi=150)
    plt.close()

    print(f"All visual output saved under: {os.path.abspath(OUTPUT_DIR)}")
    print(f"All text/metric output saved to: {os.path.abspath(RESULTS_JSON_PATH)}")


if __name__ == "__main__":
    main()