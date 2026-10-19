# Experiment Inventory & Safety Preservation Audit

**Project:** CARE-COM — ICLR 2027  
**Workspace:** `CARE-MoE/Experiments-V3`  
**Audit Date:** October 18, 2026 (Sprint Kickoff: October 17, 2026)  
**Status:** Pre-Review Audit Sprint Phase 0  

---

## 1. Safety, Preservation, and Environment Record

### A. Manuscript and Artifact Preservation
- **Submitted Manuscript Title:** CARE-COM: Geometry-Guided Sequential Functional Intervention for Mixture-of-Experts Compression
- **Target Venue:** ICLR 2027 (Double-Blind Peer Review)
- **Submitted Manuscript PDF Path:** `/Users/deepeshkumarjha/Desktop/mypaper/CARE_COM.pdf`
- **Preserved Archive Path:** [`audit_repair_pass/pre_review_sprint/SUBMITTED_MANUSCRIPT_ICLR2027.pdf`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/pre_review_sprint/SUBMITTED_MANUSCRIPT_ICLR2027.pdf)
- **Preserved PDF SHA-256 Hash:**  
  `72489204cfea2e51590b8631444eaa85d8f7a074a18359653a745f0ff4550a65`
- **Integrity Rule:** The original submitted manuscript, source files, raw historical logs, and existing audit reports remain strictly immutable and untouched. All new scripts, findings, and logs are housed strictly within `audit_repair_pass/pre_review_sprint/`.

### B. Version Control & Anonymity Status
- **Current Git Commit:** `dd5837384833110efa746bc8a334614b8db99e7b`
- **Commit Message:** `Add CARE-COM Qwen3-30B master MoE merging benchmark, audited results, and final report`
- **Git Working Tree Status:**
  - Branch: `main` (tracked with `origin/main`)
  - Untracked: `audit_repair_pass/`
  - Submodule modified: `benchmark/OLMoE-1B-7B/external/reap`
  - Deleted (unstaged): `final_report.md` (recovered and audited from Git HEAD)
- **Anonymity & Remote Sharing Recommendation:**  
  > [!CAUTION]  
  > **CRITICAL COMPLIANCE NOTICE:** The remote repository is hosted at `https://github.com/Deepesh1024/CARE-EXP.git`.  
  > As documented in `PHASE1_ANONYMITY_AUDIT.md`, this repository contains un-anonymized commit authorship metadata, author personal email (`d4deepeshpro@gmail.com`), real name (`Deepesh Kumar Jha`), institutional domain (`SANDLOGIC.AD`), and public IP addresses.  
  > **Recommendation:** If permissions allow, change the repository visibility from **public** to **private** immediately to avoid de-anonymization during double-blind review. **DO NOT** push any changes or rewrite public Git history.

### C. Local Execution Environment
- **Host OS:** macOS Darwin 25.6.0 (Kernel `xnu-12377.161.14~5/RELEASE_ARM64_T8112`, arm64)
- **Hardware Model:** Mac14,15 (Apple Silicon MacBook Air, M2 architecture)
- **Physical Memory:** 8,589,934,592 bytes (8.00 GB unified memory)
- **CPU:** 8 cores (4 performance, 4 efficiency)
- **Local Python Environment:** `/opt/anaconda3/bin/python3` (Python 3.13.9)
- **Local PyTorch:** PyTorch 2.11.0 (CUDA: `False`, Apple MPS: `True`)
- **Implication for Execution:** Large-scale models (OLMoE-1B-7B at 14 GB BF16; Qwen3-30B-A3B at 59 GB BF16) cannot be evaluated locally in system RAM (8 GB) without immediate Out-Of-Memory termination. Local executions are restricted to statistical analysis, geometry modeling, and offline data audits.

### D. Historical Server Execution Environment (Benchmarked Systems)
- **Accelerator Model:** NVIDIA L40S GPU (Ada Lovelace, 48 GB / 49,140 MiB GDDR6 with ECC, 864 GB/s bandwidth)
- **Server CPU:** AMD EPYC 7763 64-Core Processor (16–128 vCPUs)
- **Server RAM:** 188 GB DDR4 ECC RAM
- **Software Stack:** Ubuntu 22.04 LTS, CUDA 12.4/13.0, PyTorch 2.5.1 / 2.6.0, Transformers 5.19.0

---

## 2. Comprehensive Experiment Inventory

| Exp ID | Architecture | Script / Configuration | Raw Data Location | Dataset & Budget | Key Metrics Reported | Completion Status | Trustworthiness & Audit Verdict |
|---|---|---|---|---|---|---|---|
| **Exp 1** | OLMoE-1B-7B-0924 (64E) | `experiments/experiment1/` | `results/exp1/output.json` | 24,192 pairs across First, Middle, Last layers | Weight distance, cosine, output sim, routing sim, Oracle KL | Completed (24,192 pairs) | **Trustworthy (Raw Observational Baseline).** True empirical measurements. |
| **Exp 3B** | OLMoE-1B-7B-0924 (64E) | `experiments/experiment3b/` | `results/exp3b/oracle_distance_matrix_middle.csv` | 64×64 Oracle KL pairwise merge matrix (2,016 pairs) | Oracle KL divergence, 4D SMACOF MDS coordinates | Completed (2,016 pairs) | **Trustworthy (Oracle Ground Truth).** Ground-truth damage matrix for middle layer. |
| **Exp 4 (Table 1)** | OLMoE-1B-7B-0924 (64E) | `experiments/experiment4/run_all.py` | `results/exp4/fold_metrics.csv`, `audit/experiment4_pair_level.csv` | 5 partitions × 3 folds = 15 folds (3,255 test pairs) | Spearman $\rho$ (0.7504 geo vs 0.4797 local), Prec@10/25/50 | Completed | **Conditionally Trustworthy / Misattributed.** Arithmetic is verified, but "Geometry" was fitted using 43 Oracle merge KLs per test expert. Not an activation probe. |
| **Exp 4 (Section 4)** | OLMoE-1B-7B-0924 (64E) | `experiments/experiment4/` | `audit/experiment4_pair_level.csv` | 186 pairs with $|\Delta d| < 10^{-4}$ at $d_C \approx 0.0022$ | Range of KL (0.001522 to 0.007096, 4.66× span) | Completed | **Trustworthy.** Confirms geometry is insufficient alone to determine merge damage. |
| **CARE-Static (OLMoE)** | OLMoE-1B-7B-0924 (64E) | `benchmark/OLMoE-1B-7B/methods/care_com.py` | `benchmark_results/care_static/trajectory.json` | WikiText-2 test (128 chunks × 512 tokens), 16 steps (64→48) | Cumulative KL: 17.694; PPL: 21.032; Wall time: 301.63s; Evals: 80 | Completed (Run 2: 16 steps) | **Partially Trustworthy.** Quality metrics (KL=17.69, PPL=21.03) are valid. Efficiency claim (115 evals, 3387s) was corrupted by concatenated aborted run. |
| **CARE-Adaptive (OLMoE)** | OLMoE-1B-7B-0924 (64E) | `benchmark/OLMoE-1B-7B/methods/care_com.py` | `benchmark_results/care_adaptive/trajectory.json` | WikiText-2 test (128 chunks × 512 tokens), 16 steps (64→48) | Cumulative KL: 14.881; PPL: 20.136; Wall time: 666.35s; Evals: 80 | Completed (16 steps) | **Trustworthy for Quality.** Cumulative KL is 14.881, PPL is 20.136. Slower than Static (666s vs 302s), not faster. |
| **External Baselines (OLMoE)** | OLMoE-1B-7B-0924 (64E) | `benchmark/OLMoE-1B-7B/run_benchmark.py` | `benchmark_results/OLMoE-1B-7B/` (`sub_moe`, `hc_smoe`, `m_smoe`, `random`) | WikiText-2 test (128 chunks × 512 tokens), $E \in \{60, 56, 48\}$ | PPL: Sub-MoE (19.48), HC-SMoE (30.07), MC-SMoE (27.65), Random (38.10) | Completed | **Named Approximations.** Sub-MoE omitted joint SVD; HC-SMoE used weight-space proxy; MC-SMoE omitted low-rank C-SMoE. |
| **Exp 13 / Master (Qwen3-30B)** | Qwen3-30B-A3B (128E, top-k=8) | `care_large_moe_systems/.../build_master_benchmark_v2.py` | `master_results.csv`, `benchmark_results/Qwen3-30B-A3B/all_results.json` | WikiText-2 (32 chunks × 1024 = 32,768 tokens), $E \in \{112, 96, 80, 64, 60\}$ | PPL: Base (9.57), CARE-80 (19.93), HC-80 (38.34), Sub-80 (90.72), MC-80 (235.32) | Completed | **Trustworthy for PPL.** Verified on 32k tokens with strict $k=8$ assertions. |
| **Qwen3 Phase 2 Rerun** | Qwen3-30B-A3B (128E, top-k=8) | `audit_repair_pass/run_phase2_perplexity_rerun.py` | `/home/ubuntu/DEEPESH/audit_repair_pass/raw_records/` | WikiText-2 (32 chunks × 1024), Fresh process rerun | PPL with SHA-256 tensor fingerprints | Interrupted / Inactive | **Blocked / Inactive.** Encountered device mismatch runtime error on Oct 15; no running process active on local or remote machine. |
| **Systems Benchmark (L40S)** | Qwen3-30B-A3B (128E) | `care_large_moe_systems/.../workloads.py` | `master_results.csv`, `PHASE4_SYSTEMS_AUDIT.md` | Batch size 4, seq_len 1024, decode across $E \in \{128, \dots, 60\}$ | Allocated VRAM (36.62 GiB @ 80E), Peak (40.53–41.37 GiB), Decode (19.29 vs 0.68 tok/s) | Completed | **Trustworthy with Disclaimers.** Speedup (22.4×–28.4×) is a discontinuous regime change from CPU-streaming to pure GPU, not a linear compression gain. |
| **Reasoning (GSM8K/MATH)** | Qwen3-30B-A3B (128E) | `run_exp14_reasoning_eval.py` | `reasoning_predictions.jsonl` | GSM8K (29 baseline vs 100 compressed), MATH-500 (10 baseline vs 50 compressed) | GSM8K accuracy (20% Base, 22% CARE, 28% Random); MATH (0% Base, 2% CARE) | Completed but Methodologically Flawed | **FATAL FLAWS (Retracted).** Truncated `<think>` traces at 512 tokens; spurious fallback regex extracted arbitrary constants; sample size asymmetry. |
| **Exp 15 (Cross-Arch)** | JetMoE-8B & Phi-3.5-MoE | `experiments/exp15/` | `benchmark_results/exp15_merging/` | WikiText-2 (15,000 tokens) | JetMoE PPL (7E: 10.17 vs Sub-MoE 27.06); Phi PPL (14E: 6.88 vs REAP 6.44) | Completed | **Trustworthy.** Multi-architecture transfer established under single-run fixed seeds. |

---

## 3. Preservation & Security Checklist
- [x] Exact submitted manuscript PDF (`CARE_COM.pdf`) copied to `audit_repair_pass/pre_review_sprint/SUBMITTED_MANUSCRIPT_ICLR2027.pdf`.
- [x] Bit-level SHA-256 hash computed and recorded (`72489204cfea...`).
- [x] Original source codes and historical JSON trajectories preserved unmodified.
- [x] Dedicated isolated directory `audit_repair_pass/pre_review_sprint/` initialized.
- [x] Host environment parameters, package versions, and hardware limits inventoried.
