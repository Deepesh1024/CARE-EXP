# Statistical Forensic Audit: Random-K=5 vs. CARE-Adaptive

**Project:** CARE-COM — ICLR 2027  
**Workspace:** `CARE-MoE/Experiments-V3`  
**Date:** October 19, 2026  
**Status:** FORENSIC AUDIT COMPLETE  
**Artifact:** [`audit_repair_pass/random_k5_forensic_audit/STATISTICAL_AUDIT.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/random_k5_forensic_audit/STATISTICAL_AUDIT.md)

---

## 1. Executive Summary: Statistical Claims Under Audit

This forensic audit rigorously evaluates the statistical metrics reported in [`RANDOM_K5_RESULTS.md`](file:///Users/deepeshkumarjha/Desktop/CARE-MoE/Experiments-V3/audit_repair_pass/final_validation/RANDOM_K5_RESULTS.md) comparing 5 random seeds of Random-$K=5$ against the single CARE-Adaptive benchmark run on `allenai/OLMoE-1B-7B-0924`.

### Reported Claims:
1. "Mean Random-$K=5$ PPL = $29.814 \pm 3.436$"
2. "Paired difference $\Delta = +9.678$ PPL points (+32.46% relative advantage)"
3. "$Z = 2.817\sigma$"
4. "$t = 6.298, p = 0.0032$ on paired $t$-test, $df = 4$"

### Audit Verdict:
- **Numerical Calculations:** **VERIFIED**. The sample mean, sample standard deviation, and numerical $t$-statistic are arithmetic matches to the raw trajectory files.
- **Hypothesis Test Classification:** **FAILED METHODOLOGY / MISCLASSIFICATION**. Calling this test a "paired $t$-test" is mathematically false. It is a **one-sample $t$-test against a single scalar constant**, because CARE-Adaptive was evaluated on only one run ($\text{Var}(\text{CARE}) = 0$).
- **Sample Size Constraint:** With $N = 5$, non-parametric rank tests (e.g., two-tailed Wilcoxon signed-rank) cannot reach $p < 0.05$ due to combinatorial floor constraints ($p_{\min} = 0.0625$).
- **Scientific Significance:** Despite the statistical misclassification, CARE-Adaptive's score ($20.1360$) lies **$5.41$ PPL points below the lower bound of the 95% confidence interval** of the Random-$K=5$ mean ($[25.55, 34.08]$), demonstrating decisive empirical separation.

---

## 2. Recomputation of Sample vs. Population Statistics

### 2.1. Raw Trajectory Data Points ($N = 5$)
The final perplexities evaluated on WikiText-2 (65,536 tokens) at 48 experts are:
- Seed 101: $X_1 = 25.950441$
- Seed 202: $X_2 = 34.897057$
- Seed 303: $X_3 = 28.429283$
- Seed 404: $X_4 = 31.401031$
- Seed 505: $X_5 = 28.392448$
- CARE-Adaptive (Benchmark point): $c = 20.135998$

### 2.2. Rigorous Sample vs. Population Quantities

| Statistic | Exact Value | Formula | Interpretation |
|:---|:---:|:---|:---|
| **Sample Size ($N$)** | $5$ | Count of independent seeds | Small sample size |
| **Sample Mean ($\bar{X}$)** | **$29.814052$** | $\frac{1}{N} \sum X_i$ | Unbiased central tendency |
| **Sample Variance ($s^2$)** | **$11.805494$** | $\frac{1}{N-1} \sum (X_i - \bar{X})^2$ | Sample variance ($df = 4$) |
| **Sample Std Dev ($s$)** | **$3.435912$** | $\sqrt{s^2}$ | Unbiased sample dispersion |
| **Population Std Dev ($\sigma_N$)** | **$3.073167$** | $\sqrt{\frac{1}{N}\sum (X_i - \bar{X})^2}$ | Biased $N$-denominator dispersion |
| **Standard Error of Mean ($\text{SEM}$)** | **$1.536589$** | $\frac{s}{\sqrt{N}}$ | Precision of sample mean |
| **Median** | **$28.429283$** | Seed 303 | Robust central tendency |
| **Interquartile Range (IQR)** | **$3.008583$** | $Q_3 (31.4010) - Q_1 (28.3924)$ | Middle 50% spread |
| **Range ($\Delta X$)** | **$8.946616$** | $\max(X) - \min(X)$ | $[25.9504, 34.8971]$ |

---

## 3. Forensic Deconstruction of the "Paired $t$-Test" Claim

### 3.1. Why the "Paired $t$-Test" Claim is Mathematically Invalid

In classical inferential statistics, a **paired (dependent-samples) $t$-test** requires observations paired across independent sampling units:
$$(X_1, Y_1), (X_2, Y_2), \dots, (X_N, Y_N)$$
where both $X$ and $Y$ are random variables exhibiting stochastic variation across seeds/batches. The variance of the difference is:
$$\text{Var}(X - Y) = \text{Var}(X) + \text{Var}(Y) - 2\text{Cov}(X, Y)$$

In the submitted evidence package:
1. $X_i$ represents Random-$K=5$ evaluated with random seed $i \in \{101, 202, 303, 404, 505\}$.
2. $Y_i$ is **not a random variable**. It is a single, deterministic run of CARE-Adaptive ($c = 20.135998$).
3. The recorded difference is:
   $$\Delta_i = X_i - c$$
4. Consequently:
   $$\text{Var}(\Delta) = \text{Var}(X - c) = \text{Var}(X)$$
   $$\text{SEM}(\Delta) = \frac{\sqrt{\text{Var}(X)}}{\sqrt{N}} = \text{SEM}(X)$$

### 3.2. Proper Formal Reclassification: One-Sample $t$-Test Against a Fixed Constant
The test conducted was mathematically:
$$H_0: \mu_{\text{Rand-}K=5} = c \quad \text{vs.} \quad H_1: \mu_{\text{Rand-}K=5} \neq c$$
where $c = 20.135998$.

The test statistic is:
$$t = \frac{\bar{X} - c}{\frac{s}{\sqrt{N}}} = \frac{29.814052 - 20.135998}{\frac{3.435912}{\sqrt{5}}} = \frac{9.678054}{1.536589} = \mathbf{6.2984}$$

Under $H_0$, with $df = N - 1 = 4$:
- **Two-tailed $p$-value:** **$0.00324$** ($p = 0.324\%$)
- **One-tailed $p$-value ($H_1: \mu_{\text{Rand}} > c$):** **$0.00162$** ($p = 0.162\%$)

### 3.3. What the Lack of Paired Variance Conceals
By failing to measure CARE-Adaptive across multiple random calibration draws or seeds:
- We assume CARE-Adaptive has zero sampling variance ($\sigma^2_{\text{CARE}} = 0$).
- If CARE-Adaptive's true performance varies across calibration sample subsets (e.g., $\sigma_{\text{CARE}} \approx 1.0$), a true two-sample or paired test would yield a wider standard error and a slightly lower $t$-ratio.
- To make a scientifically unassailable claim at ICLR, CARE must be evaluated across the same 5 seeds/draws.

---

## 4. Alternative Statistical Tests & Confidence Intervals

### 4.1. Parametric Student's $t$ Confidence Intervals ($df = 4$)
For $95\%$ two-sided confidence level, the critical $t$-value with $df = 4$ is $t_{0.975, 4} = 2.7764$.
$$\text{CI}_{95\%} = \bar{X} \pm t_{0.975, 4} \cdot \text{SEM} = 29.8141 \pm 2.7764 \times 1.5366 = 29.8141 \pm 4.2662$$
$$\mathbf{\text{CI}_{95\%}(\mu_{\text{Rand-}K=5}) = [25.5479, 34.0803]}$$

For $99\%$ two-sided confidence level ($t_{0.995, 4} = 4.6041$):
$$\mathbf{\text{CI}_{99\%}(\mu_{\text{Rand-}K=5}) = [22.7377, 36.8904]}$$

> [!IMPORTANT]
> **Key Finding on Separation:**
> Even at the 99% confidence level, the lower bound of Random-$K=5$ ($22.74$) remains **$+2.60$ PPL points above CARE-Adaptive ($20.14$)**.
> The null hypothesis that the mean of Random-$K=5$ equals or outperforms CARE-Adaptive is rejected at $\alpha = 0.01$.

### 4.2. Non-Parametric Rank & Sign Tests

#### A. Exact Sign Test (Binomial)
Every single random seed performs worse than CARE-Adaptive:
$$X_i - c > 0 \quad \text{for all } i \in \{1, 2, 3, 4, 5\}$$
Successes $k = 5$ out of $n = 5$ trials under $H_0: p = 0.5$:
$$P(k = 5) = \left(\frac{1}{2}\right)^5 = \frac{1}{32} = \mathbf{0.03125} \quad (\text{one-tailed})$$
$$p_{\text{two-tailed}} = 2 \times 0.03125 = \mathbf{0.0625}$$

#### B. Wilcoxon Signed-Rank Test
Differences: $[+5.81, +14.76, +8.29, +11.27, +8.26]$.
Ranks:
- $+5.81$: Rank 1
- $+8.26$: Rank 2
- $+8.29$: Rank 3
- $+11.27$: Rank 4
- $+14.76$: Rank 5
Sum of positive ranks: $W^+ = 1 + 2 + 3 + 4 + 5 = 15$.
Exact two-sided $p$-value for $W = 15, N = 5$ is **$p = 0.0625$**.

> [!NOTE]
> **Combinatorial Sample Size Ceiling:**
> For $N = 5$, the minimum possible two-tailed $p$-value for both the Sign Test and Wilcoxon Signed-Rank Test is $2 \times (1/2)^5 = 0.0625$.
> It is mathematically impossible for any two-tailed non-parametric test with $N = 5$ to achieve $p < 0.05$. To establish non-parametric significance at $\alpha = 0.05$, a minimum sample size of $N = 6$ ($p = 0.03125$) is required.

---

## 5. Standard Deviation Metric ($Z = 2.817\sigma$)

The reported $Z$-score:
$$Z = \frac{\bar{X} - c}{s} = \frac{29.8141 - 20.1360}{3.4359} = \mathbf{2.817}$$
- **Interpretation:** CARE-Adaptive lies **2.817 sample standard deviations below the mean of the Random-$K=5$ distribution**.
- If Random-$K=5$ perplexities follow a normal distribution, the probability of drawing a value $\le 20.1360$ from the random distribution is:
  $$\Phi(-2.817) = \mathbf{0.0024} \quad (\approx 1 \text{ in } 412 \text{ random runs})$$

---

## 6. Recommendations for Rebuttal and Final Manuscript

1. **Cease using the phrase "paired $t$-test"**:
   - Reframing: *"A one-sample $t$-test against the CARE-Adaptive point benchmark ($t = 6.298, df = 4, p = 0.0032$), with CARE-Adaptive lying $2.82\sigma$ below the Random-$K=5$ distribution mean."*
2. **Report Confidence Intervals directly**:
   - Rather than relying solely on $p$-values, highlight the non-overlapping confidence intervals:
   - *"The 95% confidence interval for Random-$K=5$ perplexity is $[25.55, 34.08]$, which excludes CARE-Adaptive (20.14) by an uncontested margin of $>5.4$ PPL points."*
3. **Execute a 5-seed matched run for CARE-Adaptive**:
   - To report a genuine paired $t$-test, CARE-Adaptive must be evaluated on 5 distinct calibration draws matched to the 5 random seeds, measuring empirical paired differences $(\Delta_i = X_i - Y_i)$.
