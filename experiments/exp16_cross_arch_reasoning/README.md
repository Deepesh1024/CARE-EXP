# CARE-COM Exp 16 — Cross-Architecture Reasoning Evaluation on RTX 4090

## 1. Executive Summary & Objective

This experiment evaluates the downstream reasoning capabilities of **CARE-Adaptive** compared against **REAP**, **Sub-MoE**, **HC-SMoE**, and **MC-SMoE** across two diverse Mixture-of-Experts architectures:
1. **JetMoE-8B** (`jetmoe/jetmoe-8b`): An 8-expert base model with dual MoE blocks (attention + MLP). Evaluated at **8 (uncompressed baseline), 7, 6, and 4 experts**.
2. **Phi-3.5-MoE** (`microsoft/Phi-3.5-MoE-instruct`): A 16-expert instruction-tuned model (42B total params, 6.6B active). Evaluated in 4-bit NF4 at **16 (uncompressed baseline), 14, 12, 10, and 8 experts**.

**Scientific Questions Addressed:**
1. Does CARE preserve reasoning accuracy better than REAP at matched retained-expert counts?
2. Does CARE outperform competing expert-merging baselines (Sub-MoE, HC-SMoE, MC-SMoE)?
3. Does CARE's relative reasoning advantage widen under severe compression (e.g., 50% expert retention)?
4. Do perplexity (PPL) rankings correlate with downstream reasoning (GSM8K and MATH-500) performance?
5. Are any apparent collapses due to reasoning failure or formatting/extraction failure?

---

## 2. Environment & Resource Constraints

- **Platform:** NVIDIA GeForce RTX 4090 (24GB VRAM), Intel Core i7-14700K / Ubuntu 22.04 LTS.
- **Project Directory:** `~/LINGO/PROJECTS/CARE-EXP/experiments/exp16_cross_arch_reasoning/`
- **Execution Constraints:**
  - Strictly **evaluation-only**: No model training or fine-tuning.
  - **No recomputation of PPL**: Reuses validated PPL baselines from Exp 12 and Exp 15.
  - **Zero disk risk**: Utilizes streaming JSONL generation logs and lightweight metadata to avoid filling VM storage.
  - **VRAM bounded**: JetMoE evaluated in native `bfloat16` (~16 GB VRAM); Phi-3.5 evaluated in `4-bit NF4` with single-target dynamic loading and explicit GPU garbage collection between runs (~21 GB peak VRAM).

---

## 3. Benchmarks & Prompting Specifications

All runs use strictly fixed question IDs to guarantee exact paired comparisons:
- **GSM8K:** 100 fixed questions (`indices: 0..99`, `openai/gsm8k`, test split).
  - Metric: Exact Match (numerical extraction of integer/float answers).
- **MATH-500:** 50 fixed questions (`indices: 0..49`, `HuggingFaceH4/MATH-500`, test split).
  - Metric: Normalized LaTeX / numerical equivalence on `\boxed{}` content.

### Prompt Templates by Architecture:
- **JetMoE-8B (Base Model):**
  - GSM8K: `Question: {question}\nAnswer: Let's think step by step.`
  - MATH-500: `Problem: {problem}\nPlease reason step by step, and put your final answer within \boxed{}.\nSolution:`
- **Phi-3.5-MoE (Instruct Model):**
  - Chat template applied via `tokenizer.apply_chat_template`:
    `<|user|>\nPlease reason step by step, and put your final answer within \boxed{}.\n\n{question}<|end|>\n<|assistant|>\n`

### Decoding Configuration:
- Greedy decoding (`temperature=0.0`, `do_sample=False`).
- `max_new_tokens=512`.
- Raw model outputs are captured into `raw_generations/*.jsonl` prior to answer extraction to independently audit parsing failures vs model errors.

---

## 4. Execution Workflow

### Phase 1 — Inventory & Registries
- Checkpoint registry: [checkpoint_registry.csv](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/exp16_cross_arch_reasoning/checkpoint_registry.csv)
- Coverage matrix: [coverage_matrix.csv](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/exp16_cross_arch_reasoning/coverage_matrix.csv)
- Evaluation config: [evaluation_config.json](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/experiments/exp16_cross_arch_reasoning/evaluation_config.json)

### Phase 2 — Smoke Tests
Run smoke tests on 2 questions each for Baseline and 1 compressed target per model:
```bash
python3 exp16_cross_arch_reasoning/run_smoke_tests.py
```

### Phase 3 — JetMoE Reasoning Runner
Evaluates 8, 7, 6, and 4 experts across Baseline, CARE-Adaptive, REAP, Sub-MoE, HC-SMoE, and MC-SMoE:
```bash
python3 exp16_cross_arch_reasoning/run_jetmoe_reasoning.py
```

### Phase 4 — Phi-3.5-MoE Reasoning Runner
Evaluates 16, 14, 12, 10, and 8 experts across Baseline, CARE-Adaptive, REAP, Sub-MoE, HC-SMoE, and MC-SMoE:
```bash
python3 exp16_cross_arch_reasoning/run_phi_reasoning.py
```

### Phase 5 & 6 — Aggregation & Final Report
Consolidates raw logs, computes paired statistical comparisons, and generates report tables:
```bash
python3 exp16_cross_arch_reasoning/aggregate_and_report.py
```

---

## 5. Artifacts and Output Files

1. `checkpoint_registry.csv` — Provenance and validation status for all evaluated models.
2. `coverage_matrix.csv` — Progress and completion tracker across all planned runs.
3. `evaluation_config.json` — Exact prompt templates, token limits, sample IDs, and decoding parameters.
4. `per_question_results.csv` — Per-question evaluation correctness, parse flags, latency, and tokens.
5. `summary_results.csv` — Aggregate accuracy, parse error rate, throughput, and peak VRAM.
6. `raw_generations/` — Streaming JSONL outputs for every model and benchmark.
7. `run_logs/` — Execution logs with timestamps and VRAM tracking.
8. `reasoning_benchmark_report.md` — Final scientific report including Tables A & B and detailed findings.
