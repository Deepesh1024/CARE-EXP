# Reproducibility Manifest & Environment Specification

**Project:** CARE-COM — ICLR 2027  
**Workspace:** `CARE-MoE/Experiments-V3`  
**Date:** October 18, 2026  
**Auditor:** Antigravity Autonomous Experimental Auditor  
**Artifact Location:** [`audit_repair_pass/final_validation/REPRODUCIBILITY_MANIFEST.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/final_validation/REPRODUCIBILITY_MANIFEST.md)  

---

## 1. Version Control & Source State

- **Repository Base Commit:** `dd5837384833110efa746bc8a334614b8db99e7b`
- **Working Tree Status:** Branch `main`, clean tracked files; all validation scripts, logs, and reports contained strictly within `audit_repair_pass/`.
- **Submitted Manuscript PDF:**
  - Preserved Location: `audit_repair_pass/final_validation/SUBMITTED_MANUSCRIPT_ICLR2027.pdf`
  - Bit-Level SHA-256 Hash: `72489204cfea2e51590b8631444eaa85d8f7a074a18359653a745f0ff4550a65` (100% untouched)

---

## 2. File & Script Bit-Level SHA-256 Fingerprints

| File Path | Description / Role | SHA-256 Checksum |
|---|---|---|
| `benchmark/OLMoE-1B-7B/benchmark_config.yaml` | Primary benchmark configuration | `2e87daae0f89c45bd75d8b0385e5b021f0cbd4679d8710121c57e580c9226be0` |
| `benchmark/OLMoE-1B-7B/methods/care_com.py` | Reference CARE-COM runner (Static & Adaptive) | `29e1a9b7d9a0463b652daf2e4a39769fb846178420f2f54e16c7adbaee4de4c4` |
| `experiments/care_com_v21/core.py` | Hardened `PhysicalMergeEngine` supporting both fused 3D and modular MoE | `11514a42c3979a0babc20b8cf2b7a0178ec817915a187552bcdd287f81972376` |
| `benchmark/OLMoE-1B-7B/core/evaluate.py` | WikiText-2 PPL & Marginal KL evaluation harness | `8583d1628e68e3da019b1728484d115a9a5b2e5933df3a95edc1705522bc8247` |
| `audit_repair_pass/final_validation/run_matched_random_k.py` | Audited matched Random-$K=5$ production runner | `7e024a0efeb1b2fd5f0932648d39a6777fc3517bb52615df431ed7ba2262ba2d` |
| `.../geometry_validation/run_leakage_free_validation.py` | Leakage-free geometry validation script | `204b965d0296326cba5069186dd0a9da909bb1e9cd9b86c3864857a76ae06b04` |

---

## 3. Hardware & Software Environments

### Local Workstation (Audit Host):
- **Hardware:** Apple Silicon MacBook Air (Mac14,15, M2, 8 CPU cores).
- **Physical RAM:** 8.00 GB total (1.59 GB available system memory).
- **OS:** macOS Darwin 25.6.0 (`arm64`).
- **Python:** Python 3.13.9 (`/opt/anaconda3/bin/python3`).
- **PyTorch:** PyTorch 2.11.0 (CUDA: `False`, Apple MPS: `True`).

### Remote Server (Verified Compute Host):
- **Hardware:** 1× NVIDIA L40S GPU (48 GB / 49,140 MiB GDDR6 with ECC, PCIe 4.0 x16).
- **Host CPU:** AMD EPYC 7763 64-Core Processor (128 vCPUs).
- **System RAM:** 188 GB DDR4 ECC RAM.
- **OS:** Ubuntu 22.04 LTS (Linux l40s-instance 5.15.0-186-generic).
- **NVIDIA Driver / CUDA:** Driver 580.95.05 / CUDA 13.0.
- **Conda Environment:** `/home/ubuntu/DEEPESH/.conda/qwen-care` (Python 3.11, PyTorch 2.14.1+cu130, Transformers 5.19.0).

---

## 4. Exact Execution Commands

### A. Smoke Test Execution:
```bash
ssh -i ~/.ssh/L40s-Key.pem ubuntu@45.194.3.117
cd /home/ubuntu/DEEPESH
/home/ubuntu/DEEPESH/.conda/qwen-care/bin/python audit_repair_pass/final_validation/run_matched_random_k.py \
  --seeds 101 \
  --max_steps 1 \
  --output_dir audit_repair_pass/final_validation/smoke_test
```

### B. Full Matched Random-K=5 Experiment Execution:
```bash
ssh -i ~/.ssh/L40s-Key.pem ubuntu@45.194.3.117
cd /home/ubuntu/DEEPESH
/home/ubuntu/DEEPESH/.conda/qwen-care/bin/python audit_repair_pass/final_validation/run_matched_random_k.py \
  --seeds 101 202 303 404 505 \
  --output_dir audit_repair_pass/final_validation/random_k5
```

### C. Download Output Trajectories:
```bash
scp -i ~/.ssh/L40s-Key.pem -r ubuntu@45.194.3.117:/home/ubuntu/DEEPESH/audit_repair_pass/final_validation/random_k5 audit_repair_pass/final_validation/
cp -r audit_repair_pass/final_validation/random_k5/* audit_repair_pass/final_validation/RANDOM_K5_TRAJECTORIES/
```

---

## 5. Provenance of Result Trajectories

| File | Size | Steps | Checkpoint PPLs Recorded | Status |
|---|---|:---:|---|:---:|
| `random_k_seed_101/trajectory.json` | 19.9 KB | 16 | 64: 10.2844, 60: 13.1525, 56: 17.3314, 48: 25.9504 | Completed |
| `random_k_seed_202/trajectory.json` | 20.0 KB | 16 | 64: 10.2844, 60: 13.1837, 56: 16.3054, 48: 34.8971 | Completed |
| `random_k_seed_303/trajectory.json` | 20.0 KB | 16 | 64: 10.2844, 60: 12.8976, 56: 14.9628, 48: 28.4293 | Completed |
| `random_k_seed_404/trajectory.json` | 20.0 KB | 16 | 64: 10.2844, 60: 13.7101, 56: 17.6313, 48: 31.4010 | Completed |
| `random_k_seed_505/trajectory.json` | 20.0 KB | 16 | 64: 10.2844, 60: 13.1228, 56: 16.7417, 48: 28.3924 | Completed |
