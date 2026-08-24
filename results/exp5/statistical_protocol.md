# Statistical Protocol for Experiment 5

## 1. Primary Comparison
**CARE_COM vs Strongest Conventional Baseline** at the primary compression level (e.g. 32 experts / 50% compression).

## 2. Secondary Comparisons
- **CARE_GEO vs Strongest Conventional Baseline**
- **CARE_COM vs CARE_GEO**
- **Iterative vs One-Shot variants of CARE**

## 3. Unit of Analysis
The unit of analysis is the **compressed model checkpoint** at a specific compression level (e.g. 56, 48, 40 experts). Individual merge events are NOT treated as independent observations.

## 4. Confidence Level
95% Confidence Interval ($\alpha = 0.05$) for all stochastic measurements.

## 5. Treatment of Stochastic Seeds
For stochastic baselines (e.g., Random), multiple runs with different random seeds will be executed. We report the mean and standard deviation of the metric across seeds.

## 6. Treatment of Benchmark Repetitions
For benchmarks involving generation (e.g., GSM8K) or stochastic sampling, evaluations will be run $N=5$ times with varying random seeds for the generation process to estimate benchmark variance (noise floor). For highly structured loglikelihood-based benchmarks (MMLU, ARC, Perplexity), a single run is sufficient unless hardware/driver non-determinism is detected.

## 7. Treatment of Latency Repetitions
Latency and throughput will be measured by running $N=100$ forward passes after $N=10$ warmup passes. The mean, median, p95, and standard deviation will be reported.

## 8. Multiple-Comparison Handling
No post-hoc fishing for significant results across all possible combinations of methods and metrics. We pre-specify the primary comparison (CARE_COM vs best baseline on mean normalized benchmark score).

## 9. Practical Significance Criterion
A compression method is considered successful if its mean performance degradation (e.g. Perplexity increase) is statistically significantly lower than the strongest baseline AND the difference is practically meaningful (e.g. $\Delta \text{PPL} > \text{noise\_floor\_std} \times 2$).

## 10. Indistinguishable from Noise
If the absolute difference in benchmark performance between two compressed models is smaller than the 95% Confidence Interval established during the uncompressed model's noise floor evaluation, the methods are statistically tied.
