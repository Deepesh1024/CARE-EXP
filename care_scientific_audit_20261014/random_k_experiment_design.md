# Task 4: Controlled Experiment Design Document — Matched Random-K Baseline

**Document Type:** Formal Scientific Experiment Specification  
**Status:** Ready to Implement (Do Not Execute — Static Design Only)  
**Target Architecture:** `allenai/OLMoE-1B-7B-0924` (64 experts $\to$ 48 experts)  
**Reference Codebase:** [`benchmark/OLMoE-1B-7B/methods/care_com.py`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark/OLMoE-1B-7B/methods/care_com.py)

---

## 1. Scientific Motivation: The Uncontrolled Search Confound

In the submitted manuscript, **CARE-Adaptive** is compared against a baseline denoted as **"Random"**. However, inspecting [`benchmark_results/random/random_seed_42/trajectory.json`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/benchmark_results/random/random_seed_42/trajectory.json) reveals an unisolated algorithmic confound:

- **Existing "Random" Baseline ($K=1$, Zero Search):**
  At each step $t$, a single random pair $(i, j)$ is sampled and merged directly without evaluating its functional consequence. It conducts **zero Oracle KL evaluations** ($0$ FLOPs of search).
- **CARE-Adaptive ($K=5$, Local Search):**
  At each step $t$, capability geometry nominates $K=5$ candidate pairs. The algorithm **physically merges each candidate on a copy/snapshot, computes exact Oracle KL divergence on 128 calibration tokens, and greedily selects the minimum-damage candidate**.

### The Confound:
When CARE-Adaptive outperforms Random (PPL 20.14 vs. 38.10), is the advantage driven by **the geometric quality of the candidates**, or simply by **the optimization benefit of picking the best of 5 options via measured ground-truth KL**?

To isolate the value of capability geometry from the value of local functional search, we must implement a **Matched Random-K Control**.

---

## 2. Experimental Design Specification

### A. The Core Controlled Comparison
At each step $t$ ($N \to N-1$ experts):

$$\text{Candidate Generation: } \begin{cases} 
\mathcal{C}_{\text{CARE}} = \text{generate\_candidate\_pool}(d_C, K=5) & \text{[Geometry-Guided]} \\
\mathcal{C}_{\text{Random-}K} = \text{sample\_valid\_pairs}(N_t, K=5, \text{seed}) & \text{[Random-Uniform]}
\end{cases}$$

$$\text{Selection (Identical for Both): } (i^*, j^*) = \arg\min_{(i, j) \in \mathcal{C}} D_{\text{KL}}\left( P_{\text{baseline}} \parallel P_{\text{merged}(i, j)} \right)$$

$$\text{Physical Intervention: } \text{engine.merge\_experts}(i^*, j^*)$$

### B. Controlled Independent Variables (Strict Invariance)

| Parameter | Invariant Setting | Justification |
|---|---|---|
| **Initial Checkpoint** | `allenai/OLMoE-1B-7B-0924` (fp32 / bf16) | Guarantees identical initial parameter manifold ($N=64$). |
| **Search Budget ($K$)** | Exactly **$K = 5$** candidate pairs per step | Strictly equalizes physical evaluation FLOPs per step. |
| **Calibration Set** | 128 tokens from `EXP6C_TOKEN_CAPABILITY_VECTORS.parquet` | Strictly equalizes oracle objective function landscape. |
| **Target Checkpoints** | $N \in \{60, 56, 48\}$ experts | Matches standard benchmark checkpoints. |
| **Physical Merge Operator** | `PhysicalMergeEngine.merge_experts(i, j)` | Eliminates any variation in weight synthesis or router averaging. |
| **Evaluation Set** | WikiText-2 (`seq_len=512, batch_size=8`) | Standard language modeling test set. |

---

## 3. Implementation Blueprint (Extending `care_com.py`)

The proposed baseline requires a new method runner, `compress_random_k`, to be integrated into `benchmark/OLMoE-1B-7B/methods/random_k.py`:

```python
def generate_random_k_candidates(
    current_num_experts: int, pool_size: int, rng: np.random.RandomState
) -> List[Tuple[int, int]]:
  """Generates K uniformly random valid candidate pairs (i, j) without replacement."""
  all_possible = []
  for i in range(current_num_experts):
    for j in range(i + 1, current_num_experts):
      all_possible.append((i, j))

  indices = rng.choice(len(all_possible), size=pool_size, replace=False)
  return [all_possible[idx] for idx in indices]
```

At each step, `engine.snapshot(i)` and `engine.restore()` are used exactly as in CARE to evaluate all $K=5$ random candidates on `evaluate_marginal_kl`.

---

## 4. Statistical Protocol, Replications, and Uncertainty

1. **Experimental Unit:**
   An entire sequential compression trajectory from 64 down to 48 experts under a specific random seed.
2. **Replication Budget:**
   - CARE-Adaptive is deterministic given the calibration probe ($N=1$ trajectory, requiring 80 evaluations).
   - Random-$K$ is stochastic. To establish robust statistical confidence, run **$R = 5$ independent seeds**:
     $$\text{Seeds} \in \{101, 202, 303, 404, 505\}$$
   - Total evaluations for Random-$K$: $5 \text{ seeds} \times 16 \text{ steps} \times 5 \text{ evals} = \mathbf{400\text{ exact evaluations}}$.
   - Feasibility on 1× RTX 4090: $400 \times 3.8\text{s/eval} \approx 1,520\text{ seconds} \approx \mathbf{25.3\text{ minutes}}$ total compute.
3. **Primary Measured Outcomes:**
   - **WikiText-2 Test Perplexity (PPL)** at 60, 56, and 48 experts.
   - **Cumulative Intervention KL Divergence:** $\sum_{t=1}^{16} \Delta \text{KL}_t$.
   - **End-to-End Model KL Divergence:** $D_{\text{KL}}(P_{M_{64}} \parallel P_{M_{48}})$.
   - **Total Evaluation Wall-Clock Time.**
4. **Hypothesis Testing:**
   - One-sample t-test / Wilcoxon signed-rank testing whether CARE-Adaptive PPL is strictly lower than the empirical 95% confidence interval of Random-$K$:
     $$\text{Gain}_{\text{Geometry}} = \overline{\text{PPL}}_{\text{Random-}K} - \text{PPL}_{\text{CARE-Adaptive}}$$

---

## 5. Disentanglement Logic: Isolating the Geometry Contribution

This experiment cleanly separates three nested hypotheses:

```
Total Performance Gap = [CARE-Adaptive] vs. [Random-1]
   ├── Search Effect:   [Random-K] vs. [Random-1]   (Benefit of picking best of K with KL)
   └── Geometry Effect: [CARE-Adaptive] vs. [Random-K] (True value of capability geometry)
```

| Possible Experimental Outcome | Scientific Interpretation |
|---|---|
| **$\text{PPL}_{\text{CARE}} \ll \text{PPL}_{\text{Random-}K} < \text{PPL}_{\text{Random-}1}$** | **Geometry Confirmed:** Capability geometry successfully prunes the search space toward low-damage candidates far beyond uniform sampling. |
| **$\text{PPL}_{\text{CARE}} \approx \text{PPL}_{\text{Random-}K} \ll \text{PPL}_{\text{Random-}1}$** | **Geometry Falsified / Search Dominant:** Geometry provides no better candidate pool than random guessing; all performance gains stem from the Oracle KL filtering. |
| **$\text{PPL}_{\text{CARE}} > \text{PPL}_{\text{Random-}K}$** | **Geometry Harmful:** Geometry suffers from hub congestion or false similarity traps, and random sampling actually explores a healthier candidate diversity. |

---

## 6. Artifact Status: Existing vs. Required Work

- **Can be computed from existing artifacts:**
  - $\text{PPL}_{\text{CARE-Adaptive}} = 20.136$ (from `benchmark_results/OLMoE-1B-7B/summary/results.csv`).
  - $\text{PPL}_{\text{Random-}1} = 38.103$ (mean over seeds 42, 43, 44 in `results.csv`).
- **Requires new execution (when GPU compute permitted):**
  - Running `compress_random_k` for seeds 101–505 ($R=5$).
  - Logging output to `benchmark_results/random_k/random_seed_{seed}/trajectory.json`.
