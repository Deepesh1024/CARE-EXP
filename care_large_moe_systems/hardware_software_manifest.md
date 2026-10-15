# Systems Benchmark: Hardware & Software Manifest

**Date:** October 15, 2026  
**Location:** [`care_large_moe_systems/hardware_software_manifest.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/care_large_moe_systems/hardware_software_manifest.md)

---

## 1. Compute Hardware Manifest

| Component | Specification | Operational Status |
|---|---|---|
| **GPU Model** | **NVIDIA L40S** (Ada Lovelace Architecture) | Active (`0000:06:00.0`) |
| **GPU VRAM** | **48 GB GDDR6 with ECC** (49,140 MiB total capacity) | Verified via `nvidia-smi` |
| **GPU Peak Bandwidth** | **864 GB/s** (384-bit memory bus width) | Native hardware spec |
| **PCIe Interface** | PCIe 4.0 x16 (~31.5 GB/s bidirectional throughput) | Host interconnect |
| **Host CPU** | **AMD EPYC 7763 64-Core Processor** (16 vCPUs allocated) | x86_64, 2.45 GHz base |
| **System RAM** | **188 GB DDR4 ECC** (170+ GB free memory) | Verified via `free -h` |
| **Storage Subsystem** | 993 GB NVMe Virtual Disk (/dev/vda1, 331 GB available) | Verified via `df -h` |
| **Shared Memory (`/dev/shm`)**| 95 GB RAM-backed tmpfs (high-speed checkpoint staging) | Verified via `df -h /dev/shm` |

---

## 2. Software & Inference Stack Manifest

| Software Layer | Exact Version / Configuration | Role in Systems Benchmark |
|---|---|---|
| **Operating System** | Ubuntu 22.04.5 LTS (GNU/Linux 5.15.0-186-generic) | Host execution environment |
| **NVIDIA Driver** | 580.95.05 | GPU kernel driver |
| **CUDA Runtime** | CUDA 13.0 | Hardware compute runtime |
| **Python Environment** | Python 3.11.14 (`/home/ubuntu/DEEPESH/.conda/qwen-care`) | Benchmark execution runtime |
| **PyTorch Framework** | PyTorch 2.5+ (Compiled with CUDA 12.4/13.0) | Tensor operations & memory manager |
| **Transformers** | Hugging Face Transformers 5.19.0 | Model architecture & tokenization |
| **Accelerate** | Hugging Face Accelerate | Device mapping & model loading |
| **FlashAttention / SDPA** | PyTorch Native SDPA (`torch.nn.functional.scaled_dot_product_attention`) | Constant attention backend |

---

## 3. Strict Deployment Regimes Definition

To prevent scientifically invalid comparisons between memory architectures, we formally define two separate deployment regimes:

### Regime A: Pure GPU-Resident Inference
- **Criterion:** All model parameters, activation buffers, router gates, and KV caches reside entirely in GPU VRAM (total memory allocation $\le 45\text{ GB}$).
- **Data Flow:** Zero PCIe layer swapping during prompt prefill or autoregressive token generation.
- **Applicability:**
  - **OLMoE-1B-7B:** All expert counts ($E \in \{64, 60, 56, 48\}$) requiring $10.5–14.0\text{ GB}$ VRAM.
  - **Qwen3-30B-A3B at $E \le 80$ experts:** Model parameters occupy $\approx 32.1–39.3\text{ GB}$ VRAM, fitting completely inside the 48 GB L40S.
- **Performance Characteristics:** Reflects true GPU memory-bandwidth-bound inference (~20 tok/s).

### Regime B: CPU-Offloaded / Layer-Streamed Inference
- **Criterion:** Total model parameters exceed available GPU VRAM ($> 48\text{ GB}$). Non-active layers reside in the 188 GB CPU RAM and are transferred sequentially via PCIe during forward execution.
- **Data Flow:** Each transformer layer undergoes host-to-device PCIe transfer per token or per prefill batch.
- **Applicability:**
  - **Qwen3-30B-A3B Baseline (128 experts):** Footprint is $\approx 61.4\text{ GB}$ BF16, necessitating PCIe streaming.
  - **Qwen3-30B-A3B (112 experts):** Footprint is $\approx 54.1\text{ GB}$ BF16, exceeding 48 GB VRAM.
- **Performance Characteristics:** Dominated by PCIe transfer latency (~0.68 tok/s).

> [!CRITICAL]
> **Scientific Control Rule:** Results from Regime A and Regime B must **NEVER** be plotted on a single continuous latency or throughput axis without explicit disclaimers and separate regime demarcations. A 30× speedup observed between 128 experts (Regime B) and 80 experts (Regime A) is an architectural regime transition (crossing the VRAM threshold), not a linear speedup from reducing expert count.
