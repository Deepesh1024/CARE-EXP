## Invalidated Phi Benchmark — Historical

> **Historical Phi benchmark:** An earlier Phi-3.5-MoE benchmark was invalidated during implementation audit. The previous runner did not physically remove merged experts and therefore did not realize the claimed reduction in expert count. Additionally, the earlier CARE implementation did not execute the CARE-COM candidate-evaluation-and-commit protocol. Those results are excluded from all scientific comparisons and are retained only as an implementation audit.

---

## Experiment 12 — Cross-Architecture Validation on Phi-3.5-MoE

### Objective

To determine whether the CARE-Adaptive intervention protocol transfers beyond the OLMoE architecture used in the primary CARE-COM study, we implemented an architecture-specific adaptation for `microsoft/Phi-3.5-MoE-instruct`.

The experiment evaluates whether capability-guided sequential consolidation remains effective when the underlying MoE implementation differs substantially from OLMoE, including fused expert projections and a different physical expert representation.

The experiment specifically targets:

\[
16\rightarrow14,\quad
16\rightarrow12,\quad
16\rightarrow10,\quad
16\rightarrow8
\]

local experts.

The primary comparison is between **CARE-Adaptive**, **REAP**, and **Random** under physically capacity-reducing interventions.

---

## 12.1 Implementation Validation

Before executing the full-model benchmark, three independent architecture-specific validation gates were required to pass.

### Gate 1 — Expert Equivalence

The Phi expert adapter was validated against the native `PhimoeExperts` computation.

The adapted evaluator reproduces the fused:

\[
\text{gate\_up}
\rightarrow
(\text{gate},\text{up})
\rightarrow
\operatorname{SiLU}(\text{gate})\odot\text{up}
\rightarrow
\text{down}
\]

computation while omitting router computation.

Testing covered:
- random seeds: 0, 42, 1337;
- tensor shapes: \(1\times1\times64\), \(1\times7\times64\), \(2\times13\times64\);
- all simulated experts;
- structural tensor assertions.

Maximum absolute error was:
\[
<10^{-6}.
\]

**Result: PASS.**

---

### Gate 2 — Physical Expert Deletion

The `PhiPhysicalMergeEngine` was independently tested to verify genuine \(N\rightarrow N-1\) architectural compression.

For a 16-expert model, merging experts \(3\) and \(11\) produced:
\[
16\rightarrow15
\]
with corresponding structural changes:
\[
W_{\text{gate-up}}:
(16,256,64)\rightarrow(15,256,64)
\]
and router output:
\[
16\rightarrow15.
\]

A forward pass successfully executed on the modified graph, and transaction restoration returned the model to its original 16-expert structure.

**Result: PASS.**

This explicitly excludes the dead-slot failure mode in which merged parameters are modified while the removed expert remains addressable by the router.

---

### Gate 3 — One-Step CARE-COM

The complete sequential intervention loop was tested before full benchmarking:

\[
C_t
\rightarrow
\text{candidate generation}
\rightarrow
\text{physical trial merge}
\rightarrow
\Delta KL
\rightarrow
\arg\min
\rightarrow
\text{commit}
\rightarrow
C_{t+1}.
\]

A five-pair candidate pool was generated from a simulated 200-token probe. Each candidate underwent:
1. snapshot,
2. physical merge,
3. damage evaluation,
4. restoration.

The minimum-damage candidate was permanently committed, after which capability vectors were recomputed. The capability dimensionality correctly changed from \(N\) to \(N-1\).

**Result: PASS.**

---

## 12.2 Benchmark Protocol

The full Phi benchmark used the corrected physical compression implementation.

At each compression step:
1. Compute the current capability state \(C_t\).
2. Generate the CARE candidate pool.
3. Physically trial-merge each candidate.
4. Measure functional damage.
5. Restore the baseline state.
6. Commit the lowest-damage candidate.
7. Physically remove the redundant expert.
8. Update the router and expert tensors.
9. Recompute \(C_{t+1}\).
10. Save the resulting state as an intermediate checkpoint.

The following invariants were enforced after every committed intervention:
\[
N_{t+1}=N_t-1
\]
\[
N_{\text{router}}
=
N_{\text{gate-up}}
=
N_{\text{down}}
=
N_{\text{capability}}.
\]

A restoration test additionally required baseline logits to be recovered to within \(10^{-3}\) before candidate evaluation proceeded.

---

## 12.3 Results

### WikiText-2 Perplexity

| Method | 16 → 14 | 16 → 12 | 16 → 10 | 16 → 8 |
|---|---:|---:|---:|---:|
| Uncompressed | 4.57 | 4.57 | 4.57 | 4.57 |
| **CARE-Adaptive** | **6.88** | **9.24** | **13.57** | **32.75** |
| REAP | 6.44 | 17.65 | 40.33 | 125.72 |
| Random | 8.50 | 23.34 | 44.74 | 71.90 |

### Performance Retention Visualization

![Performance Retention Bar Chart](results/plots/retention_bar_chart.png)
*Figure 1: Predictive probability retention (Baseline PPL / Compressed PPL). Higher is better.*

CARE-Adaptive is slightly worse than the evaluated REAP baseline at 14 experts:
\[
6.88 > 6.44.
\]
However, CARE-Adaptive becomes substantially better at stronger compression:
\[
9.24 < 17.65
\]
\[
13.57 < 40.33
\]
\[
32.75 < 125.72.
\]

At 50% expert reduction, CARE-Adaptive achieves:
\[
\text{PPL}=32.75
\]
compared with:
\[
\text{PPL}=125.72
\]
for REAP.

This corresponds to approximately **74.0% lower perplexity than REAP** at the 16→8 target.

---

## 12.4 Compression-Regime Dependence

The result exhibits a notable crossover between the two methods.

At relatively mild compression, REAP performs slightly better:
\[
16\rightarrow14:
\quad
6.44\text{ vs. }6.88.
\]

As compression becomes more aggressive, the performance gap reverses and expands:
\[
\begin{aligned}
16\rightarrow12 &: \quad 9.24\text{ vs. }17.65,\\
16\rightarrow10 &: \quad 13.57\text{ vs. }40.33,\\
16\rightarrow8 &: \quad 32.75\text{ vs. }125.72.
\end{aligned}
\]

Thus, within this experiment, CARE-Adaptive exhibits substantially more graceful degradation than the evaluated REAP baseline in the moderate-to-aggressive compression regime.

**We do not interpret this as universal superiority of CARE over REAP.** The experiment evaluates one Phi-3.5-MoE architecture, one language-model benchmark, and one implementation/protocol.

---

## 12.5 Baseline Implementation Audit

Several additional baseline implementations initially produced unusually low perplexities. These results were excluded from the scientific comparison because they did not perform genuine physical \(N\rightarrow N-1\) compression.

In particular, the affected merge-based implementations modified one expert's parameters while leaving the second expert intact and addressable by the router.

Consequently, these models retained the original expert count despite reporting a nominal compression target.

Such results cannot be compared to CARE-Adaptive as compression results.

Therefore, the following baseline results are classified as **invalid for physical-compression comparison**:
- `rw_l2`
- `parameter`
- `submoe`
- `care_static`

They may be retained in the implementation audit as evidence of the failure mode, but should not appear as scientific baseline results.

---

## 12.6 Interpretation

The Phi experiment provides preliminary evidence that the CARE intervention framework can transfer to an MoE architecture with substantially different expert implementation details.

More importantly, the result suggests a potential **compression-regime dependence**:

> CARE's advantage over the evaluated pruning baseline becomes increasingly pronounced as the fraction of removed experts increases.

One possible interpretation is that functional consolidation becomes increasingly valuable when aggressive expert removal would otherwise destroy broader capability coverage. However, this mechanism is **not established by the current experiment**.

The experiment demonstrates an empirical cross-architecture performance result, not a causal explanation for the observed gap.

---

## 12.7 Limitations

The following limitations should be recorded explicitly:

1. **Single architecture:** Phi-3.5-MoE-instruct is the only additional architecture evaluated under this corrected protocol.
2. **Single language benchmark:** Results are currently based on WikiText-2 perplexity.
3. **Single benchmark run:** Multi-seed robustness has not yet been established for the Phi experiment.
4. **Different intervention mechanisms:** CARE performs physical expert consolidation, whereas REAP performs expert pruning; therefore the comparison should not be interpreted as a controlled test of identical interventions.
5. **REAP implementation scope:** The reported REAP baseline should be identified precisely as the implementation used in this experiment rather than treated as a universal characterization of the published REAP method.
6. **No claim of universal superiority:** The observed CARE advantage is scoped to the evaluated Phi configuration and compression regime.

---

## 12.8 Key Finding

> **Finding — Cross-architecture validation:**  
> Under a corrected physical-compression protocol on Phi-3.5-MoE-instruct, CARE-Adaptive achieved PPL 6.88, 9.24, 13.57, and 32.75 at 14, 12, 10, and 8 experts respectively. CARE was slightly worse than the evaluated REAP baseline at 14 experts, but substantially better at 12, 10, and 8 experts, with a 74.0% lower PPL than REAP at 50% expert reduction. These results provide preliminary evidence that capability-guided sequential consolidation can transfer beyond OLMoE, while remaining scoped to the evaluated architecture and protocol.
