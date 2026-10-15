# Master Evaluation Protocol & Experimental Standards

**Model Under Study:** `Qwen/Qwen3-30B-A3B` (Architecture: 48 MoE layers, hidden dimension 2048, intermediate size 768 per expert, 128 initial experts per layer, default top-8 routing).  
**Hardware Environment:** NVIDIA L40S 48 GB GDDR6 VRAM, 188 GB Host CPU RAM, Ubuntu 22.04 LTS, CUDA 13.0, Driver 580.95.05.  
**Software Environment:** Python 3.11, PyTorch 2.5+, Hugging Face Transformers.

---

## 1. Single-Dataset / Single-Model Perplexity Standard

To maintain strict scientific hygiene and avoid any conflation across datasets or architectures:
1. **Single Model Guarantee:** Every table and evaluation record in this benchmark evaluates strictly `Qwen/Qwen3-30B-A3B`. Cross-architecture results (JetMoE-8B, Phi-3.5-MoE) are never mixed into Qwen3 tables.
2. **Single Dataset Guarantee:** Language modeling perplexity tables contain results evaluated exclusively on **WikiText-2**. Perplexity values from other datasets or task-specific reasoning prompts are never mixed into WikiText-2 tables.
3. **Exact Token Budgeting:**
   - **Full Benchmark Evaluation:** Evaluated on exactly **32 chunks × 1024 tokens = 32,768 tokens** from `Salesforce/wikitext` (`wikitext-2-raw-v1`, test split).
   - **Pilot Sweep Evaluation:** Evaluated on exactly **4 chunks × 512 tokens = 2,048 tokens** from the same test split.
   - Every result is explicitly annotated with its evaluation token budget (`Full (32k tok)` vs. `Pilot (2k tok)`).
4. **Loss Calculation & Perplexity Formulation:**
   - For input sequence $X = (x_1, \dots, x_T)$, logits $\hat{y}_t$ are computed via forward pass.
   - Cross-entropy loss is computed on shifted labels:
     $$\mathcal{L} = \frac{1}{T-1} \sum_{t=1}^{T-1} -\log P(x_{t+1} \mid x_1, \dots, x_t)$$
   - Perplexity is computed via exponentiation of token-weighted average NLL:
     $$\text{PPL} = \exp(\bar{\mathcal{L}})$$

---

## 2. Decoupling Retained Population ($E$) from Active Sparsity ($k$)

A foundational control of this benchmark is the strict decoupling of two independent variables:
1. **Retained Expert Population ($E$):**
   - The total number of physical expert modules retained in memory and storage per layer.
   - Evaluated target values: $E \in \{128, 112, 96, 80, 64, 60\}$.
   - $E$ directly determines the model parameter count ($32.05\text{B} \to 16.64\text{B}$), disk storage ($59.70\text{ GB} \to 31.00\text{ GB}$), and GPU VRAM residency threshold.
2. **Active Routing Sparsity (Inference Top-$k$):**
   - The number of experts dynamically activated per token during inference.
   - Evaluated values: $k \in \{8, 6, 4, 2\}$.
   - In `transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeTopKRouter`, routing weights are normalized:
     $$p_i = \frac{\exp(s_i)}{\sum_{j \in \text{Top-}k} \exp(s_j)}, \quad \sum_{i \in \text{Top-}k} p_i = 1.0$$
   - Modulating $k$ alters active arithmetic compute ($6 k H I$ FLOPs) but **does not reduce model storage or memory footprint**.

---

## 3. Systems Profiling & Regime Boundaries

Execution is strictly segregated into two hardware deployment regimes:
- **Regime A (Pure GPU-Resident, $E \le 80$):**
  - Model weights occupy $\le 39.44\text{ GB}$, fully residing in 48 GB L40S VRAM ($36.62\text{ GB}$ allocated).
  - Eliminates dynamic PCIe bus layer transfers.
  - Measures sustained GPU prefill throughput and decode throughput via FlashAttention / PyTorch SDPA with KV caching.
- **Regime B (CPU-Offloaded Streaming, $E > 80$):**
  - Model weights occupy $46.19\text{ to } 59.70\text{ GB}$, exceeding 48 GB VRAM.
  - Weights are offloaded to host CPU RAM and streamed sequentially into a reusable GPU layer buffer.
  - GPU VRAM allocation is constrained to $5.89\text{ GB}$; decode speed is bottlenecked by host-to-device PCIe bandwidth ($\sim 0.7\text{ tok/s}$).

### Measurement Rigor
- **Warm-Up:** 1 forward pass batch executed before measurement.
- **CUDA Synchronization:** `torch.cuda.synchronize()` invoked immediately before and after timing blocks.
- **Timing Repetitions:** Autoregressive decode throughput measured across 3 independent repetitions (32 new tokens generated with KV cache), recording mean and standard deviation.

---

## 4. Reasoning Evaluation Protocol

Reasoning is evaluated on dedicated benchmarks with isolated metrics:
1. **Datasets:**
   - **GSM8K:** Grade School Math 8K (100 representative questions, 8-shot chain-of-thought prompting).
   - **MATH-500:** Challenging competition mathematics (50 representative questions).
2. **Generation Parameters:** Greedy decoding (`do_sample=False`), temperature 0.0, `max_new_tokens=512`.
3. **Scoring & Answer Extraction:** Regex extraction of numeric targets (`#### <num>` or boxed answers), comparing exact numeric match.
4. **Audit Transparency:** Every raw prompt, model output (including `<think>` traces), and extraction score is preserved line-by-line in `reasoning_predictions.jsonl`.
