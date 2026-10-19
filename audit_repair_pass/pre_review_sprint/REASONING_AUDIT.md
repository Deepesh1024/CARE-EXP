# Phase 8: Mathematical Reasoning Pipeline & Mechanism Audit Report

**Project:** CARE-COM — ICLR 2027  
**Workspace:** `CARE-MoE/Experiments-V3`  
**Date:** October 18, 2026  
**Auditor:** Antigravity Autonomous Audit Agent  
**Artifact Location:** [`audit_repair_pass/pre_review_sprint/REASONING_AUDIT.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/pre_review_sprint/REASONING_AUDIT.md)  

---

## 1. Executive Summary & Verdict

Earlier benchmark iterations reported mathematical reasoning accuracy on **GSM8K** and **MATH-500** for `Qwen/Qwen3-30B-A3B`, claiming:
> *"CARE uniquely preserves specialized reasoning circuits in deep MoE layers, achieving 74.5% accuracy retention on complex multi-step reasoning tasks while baselines catastrophically collapse."*

Our forensic audit of raw generation records in [`care_large_moe_systems/complete_comparative_benchmark_v2/reasoning_predictions.jsonl`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/complete_comparative_benchmark_v2/reasoning_predictions.jsonl) uncovers **three fatal methodological defects**:
1. **The `<think>` Generation Truncation Flaw:** Capping generation at `max_new_tokens=512` prematurely severed the model's native internal chain-of-thought monologue in $>85\%$ of samples before the final answer was computed.
2. **Spurious Heuristic Regex Fallback:** When no final answer delimiter (`####` or `\boxed{}`) was found, a fallback parser extracted the last arbitrary number mentioned in the truncated thinking text, scoring arbitrary constants rather than mathematical solutions.
3. **Severe Sample-Size Asymmetry:** The pristine Uncompressed Baseline was evaluated on only **29 questions** (GSM8K) and **10 questions** (MATH-500), compared to **100 questions** and **50 questions** for compressed variants.

### Formal Audit Verdict:
Claims asserting *"specialized reasoning circuit preservation"* or *"74.5% reasoning retention"* are **UNSUPPORTED AND METHODOLOGICALLY INVALID**. The observed accuracy variations reflected prompt truncation length and regex artifacts rather than mathematical competence. All reasoning claims are formally **WITHDRAWN** from the scientific submission.

---

## 2. Forensic Breakdown of Methodological Flaws

### 2.1. The `<think>` Truncation Mechanism
Under standard prompting, `Qwen/Qwen3-30B-A3B` initiates an extensive multi-step deliberation enclosed in `<think> ... </think>` delimiters:
- Complex multi-step reasoning on GSM8K and MATH-500 requires 800–1,500 tokens of internal monologue.
- Imposing a strict cap of `max_new_tokens = 512` caused the generation to be cut off mid-thought.
- In over $85\%$ of all evaluated samples across all models, the closing tag `</think>` and the final solution marker were never reached.

---

### 2.2. Spurious Regex Fallback Scoring
The answer parsing harness in `run_exp14_reasoning_eval.py:L56-63` implemented the following fallback logic:
```python
# Fallback: Last integer/float in text
nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
if nums:
  raw = nums[-1].replace(",", "").strip()
  return float(raw) if "." in raw else int(raw)
```
When generation terminated mid-sentence inside `<think>`, this parser extracted whatever arbitrary intermediate digit happened to appear at token 512.

#### Concrete Case Evidence from `reasoning_predictions.jsonl`:
- **GSM8K Index 0:**
  - Question: *"Janet's ducks lay 16 eggs per day... sells remainder for $2 per egg. How much does she make?"*
  - True Solution: `18` ($(16 - 3 - 4) \times 2$)
  - Truncated Text: `...she uses some of them for breakfast (3 eggs) and baking muffins (4 eggs)...`
  - Extracted Number: `16` (simply the initial number repeated in the thought)
  - Scored: False.
- **GSM8K Index 2:**
  - Question: *"Josh buys a house for $80,000, repairs for $50,000... increases by 150%. How much profit?"*
  - True Solution: `70000`
  - Truncated Text: `...profit is selling price minus total investment (80,000 + 50,000 = 130,000)...`
  - Extracted Number: `0` or `2.5`
  - Scored: False.

---

### 2.3. Sample-Size Inconsistency Audit
Inspection of all 712 records in `complete_comparative_benchmark_v2/reasoning_predictions.jsonl` reveals that the uncompressed baseline was prematurely halted due to an offloading timeout:

| Evaluated Variant | GSM8K Sample Count | MATH-500 Sample Count | Measured GSM8K "Accuracy" | Audit Status |
|---|:---:|:---:|:---:|:---:|
| **Baseline-128 (Dense)** | **29** | **10** | **24.14%** (7 / 29) | **INVALID (Halted premature)** |
| **CARE-Adaptive-90** | 100 | 50 | 18.00% (18 / 100) | Truncated Artifact |
| **CARE-Adaptive-80** | 100 | 50 | 22.00% (22 / 100) | Truncated Artifact |
| **Sub-MoE-80** | 100 | 50 | 23.00% (23 / 100) | Truncated Artifact |
| **Random-80** | 100 | 50 | 28.00% (28 / 100) | Truncated Artifact |

**Key Finding:** Random-80 scored "28.0%" while CARE scored "22.0%" and Baseline scored "24.14%". These differences were entirely stochastic noise driven by whether intermediate numbers matched ground truth constants.

---

## 3. Disavowed Claims & Manuscript Action Items

The following claims are **FORMALLY RETRACTED** and must not appear in the submitted manuscript:
1. *"CARE uniquely protects specialized mathematical reasoning circuits in deep MoE layers."*
2. *"CARE achieves 74.5% reasoning accuracy retention at 90 experts."*
3. Any comparative reasoning table based on `max_new_tokens=512`.

### Minimum Standardized Protocol for Future Work:
If reasoning benchmarks are re-introduced in subsequent revisions:
1. **Extended Token Budget:** Set `max_new_tokens >= 2048`.
2. **Explicit Format Exemplars:** Use 8-shot chain-of-thought prompting that forces termination with `The final answer is: #### <answer>`.
3. **Strict Delimiter Scoring:** Disable all fallback regexes. A generation that does not output the closed delimiter is scored as a parsing failure.
4. **Matched Sample Counts:** Minimum 500 questions evaluated identically across baseline and all compressed variants.
