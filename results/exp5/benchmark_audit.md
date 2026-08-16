# Benchmark Suite Audit

## Findings
I have inspected the repository (including `experiments/`, `results/`, `full_report.md`, and the `README.md`) for existing benchmark implementations. 
- There is NO `lm_eval` integration.
- There are NO custom evaluation scripts for tasks like MMLU, GSM8K, HumanEval, ARC, Hellaswag, etc.
- Previous experiments evaluated capability primarily through Oracle KL divergence on a calibration dataset (Wikitext), not through standard NLP benchmarks.

The project lacks an established benchmark suite suitable for evaluating the final compressed model. 

## Candidate Benchmarks (Requires Decision)
Since we must evaluate the quality of the compressed OLMoE model against the original, we need to select standard benchmarks that the model is expected to perform reasonably well on. Since OLMoE-1B-7B is a general-purpose language model, standard reasoning and knowledge benchmarks are appropriate.

| Benchmark | Task | Metric | Dataset | Number of Examples | Evaluation Method | Deterministic | Suitability for Compressed OLMoE | Classification |
|---|---|---|---|---|---|---|---|---|
| **Wikitext Perplexity** | Language Modeling | Perplexity | `Salesforce/wikitext` | 98 seqs x 512 | CrossEntropy Loss | Yes | Direct measurement of generative degradation. | **REQUIRED** |
| **MMLU** | Knowledge | Accuracy | `cais/mmlu` | ~14k | Loglikelihood Choice | Yes | Standard baseline for world knowledge. | **RECOMMENDED** |
| **ARC-Challenge** | Reasoning | Accuracy | `ai2_arc` | 1172 | Loglikelihood Choice | Yes | Tests reasoning capabilities under compression. | **RECOMMENDED** |
| **GSM8K** | Math Reasoning | Exact Match | `openai/gsm8k` | 1319 | Generation | No | High variance, sensitive to generation params. | **OPTIONAL** |

### Research Decision Required:
Please specify the exact benchmarks (e.g. from `lm-evaluation-harness`) to be used for **Gate B-BENCH**, or if Perplexity on Wikitext is the sole metric for Gate B-PPL and sufficient for the initial implementation.
