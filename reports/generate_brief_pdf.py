import base64
import os
import subprocess

chart_path = "reports/hero_compression_multiseed_chart.png"
with open(chart_path, "rb") as f:
    chart_b64 = base64.b64encode(f.read()).decode("utf-8")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CARE-COM: Toward More Capability-Aware MoE Expert Consolidation</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 0;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  body {{
    width: 210mm;
    height: 297mm;
    max-height: 297mm;
    margin: 0;
    padding: 8.5mm 12mm 7mm 12mm;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    line-height: 1.32;
    overflow: hidden;
  }}
  
  /* Header */
  .header {{
    border-bottom: 1.5px solid #0284c7;
    padding-bottom: 5px;
    margin-bottom: 6px;
  }}
  .eyebrow {{
    font-size: 6.8pt;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #0284c7;
    margin-bottom: 2px;
  }}
  h1 {{
    font-size: 15.2pt;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.15;
    margin-bottom: 2px;
  }}
  .subtitle {{
    font-size: 8.8pt;
    font-style: italic;
    color: #475569;
    margin-bottom: 4px;
  }}
  .thesis {{
    font-size: 8.0pt;
    color: #334155;
    line-height: 1.28;
  }}
  .scope-line {{
    font-size: 6.8pt;
    color: #475569;
    margin-top: 3.5px;
    padding-top: 2.5px;
    border-top: 1px dashed #cbd5e1;
  }}
  .scope-line strong {{
    color: #1e293b;
    text-transform: uppercase;
    font-size: 6.3pt;
    letter-spacing: 0.05em;
  }}
  
  /* Results Strip */
  .strip-container {{
    display: flex;
    gap: 8px;
    margin-top: 5px;
    margin-bottom: 6px;
  }}
  .strip-card {{
    flex: 1;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 4.5px 7px;
    text-align: center;
  }}
  .strip-card.highlight {{
    background: #f0fdf4;
    border-color: #bbf7d0;
  }}
  .strip-card.primary {{
    background: #f0f9ff;
    border-color: #bae6fd;
  }}
  .strip-val {{
    font-size: 13.5pt;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
    margin-bottom: 2px;
  }}
  .strip-card.primary .strip-val {{
    color: #0284c7;
  }}
  .strip-card.highlight .strip-val {{
    color: #16a34a;
  }}
  .strip-label {{
    font-size: 6.6pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #334155;
  }}
  .strip-sub {{
    font-size: 6.0pt;
    color: #64748b;
  }}

  /* Section Titles */
  .section-title {{
    font-size: 7.6pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #0f172a;
    margin-bottom: 3.5px;
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .section-title::before {{
    content: "";
    display: inline-block;
    width: 3px;
    height: 9px;
    background: #0284c7;
    border-radius: 1px;
  }}

  /* Method Architecture Diagram */
  .diagram-container {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 4px 6px;
    margin-bottom: 6px;
  }}
  .diagram-svg {{
    width: 100%;
    height: 38px;
    display: block;
  }}

  /* Hero Graphic */
  .chart-box {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 3px;
    margin-bottom: 6px;
    text-align: center;
  }}
  .chart-box img {{
    width: 100%;
    height: auto;
    display: block;
  }}
  .chart-caption {{
    font-size: 5.9pt;
    color: #64748b;
    margin-top: 2.5px;
    line-height: 1.2;
  }}

  /* Split Grid: Narrative & Table */
  .middle-grid {{
    display: grid;
    grid-template-columns: 1.05fr 1fr;
    gap: 8px;
    margin-bottom: 6px;
    align-items: start;
  }}
  
  /* Pillar text */
  .pillar-box {{
    font-size: 7.1pt;
    color: #334155;
  }}
  .pillar-item {{
    margin-bottom: 4px;
    line-height: 1.25;
  }}
  .pillar-item strong {{
    color: #0f172a;
  }}
  
  /* Table styling */
  .table-box {{
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 6.5pt;
    text-align: left;
  }}
  th {{
    background: #f1f5f9;
    color: #334155;
    font-weight: 700;
    padding: 3.0px 4px;
    border-bottom: 1px solid #cbd5e1;
  }}
  td {{
    padding: 2.5px 4px;
    border-bottom: 1px solid #f1f5f9;
    color: #334155;
  }}
  tr.highlight-row {{
    background: #f0f9ff;
    font-weight: 600;
  }}
  tr.highlight-row td {{
    color: #0369a1;
  }}
  tr:last-child td {{
    border-bottom: none;
  }}
  .table-note {{
    font-size: 5.7pt;
    color: #64748b;
    font-style: italic;
    padding: 2px 4px;
    background: #fafafa;
    border-top: 1px solid #f1f5f9;
  }}

  /* Bottom Split: Scope & Next Question */
  .bottom-grid {{
    display: grid;
    grid-template-columns: 1.05fr 1fr;
    gap: 8px;
    margin-bottom: 5px;
  }}
  
  .scope-box {{
    background: #fffbeb;
    border: 1px solid #fef3c7;
    border-left: 3px solid #d97706;
    border-radius: 4px;
    padding: 4.5px 7px;
    font-size: 6.8pt;
    color: #92400e;
    line-height: 1.22;
  }}
  .scope-box strong {{
    color: #78350f;
  }}

  .discussion-box {{
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 3px solid #16a34a;
    border-radius: 4px;
    padding: 4.5px 7px;
  }}
  .discussion-title {{
    font-size: 7.1pt;
    font-weight: 700;
    color: #166534;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 2.5px;
  }}
  .discussion-q {{
    font-size: 7.2pt;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.22;
    margin-bottom: 2.5px;
  }}
  .discussion-sub {{
    font-size: 6.5pt;
    color: #374151;
    line-height: 1.2;
  }}

  /* Footer */
  .footer {{
    border-top: 1px solid #e2e8f0;
    padding-top: 3.5px;
    font-size: 5.7pt;
    color: #64748b;
    line-height: 1.2;
    display: flex;
    justify-content: space-between;
  }}
  .footer-text strong {{
    color: #475569;
  }}
</style>
</head>
<body>

  <!-- 1. Header & Thesis -->
  <div class="header">
    <h1>CARE-COM: Toward More Capability-Aware MoE Expert Consolidation</h1>
    <div class="subtitle">Investigating capability-geometry signals for more informed expert consolidation under aggressive compression.</div>
    <div class="thesis">
      Mixture-of-Experts (MoE) architectures achieve superior training efficiency, but their massive aggregate parameter footprints create severe memory bandwidth bottlenecks and deployment costs during high-concurrency inference. Existing MoE merging methods decide which experts to combine using static heuristics or unguided router clustering, risking destructive parameter interference. <strong>CARE-COM</strong> investigates whether <strong>functional signals</strong> from expert representations can guide expert consolidation and ultimately help identify risky merges without retraining.
    </div>
    <div class="scope-line">
      <strong>Scope:</strong> OLMoE-1B-7B; WikiText-2 perplexity evaluation; MMLU results exploratory (N=100).
    </div>
  </div>

  <!-- 2. Three-Number Metric Strip -->
  <div class="strip-container">
    <div class="strip-card primary">
      <div class="strip-val">19.88 PPL</div>
      <div class="strip-label">Best Observed Merging PPL @ 48E</div>
      <div class="strip-sub">CARE-Adaptive (-25% budget, directional, n=3)</div>
    </div>
    <div class="strip-card highlight">
      <div class="strip-val">-2.45 PPL</div>
      <div class="strip-label">Margin vs. Random-K5 Control</div>
      <div class="strip-sub">Directional multi-seed gain (19.88 vs 22.33 PPL)</div>
    </div>
    <div class="strip-card">
      <div class="strip-val">6.9–15.0 PPL</div>
      <div class="strip-label">Improvement vs. Clustering Proxies</div>
      <div class="strip-sub">Vs. MC-SMoE (26.74), HC-SMoE (31.00), Sub-MoE (34.84)</div>
    </div>
  </div>

  <!-- 3. Method Architecture Diagram -->
  <div class="section-title">The CARE-COM Pipeline: Capability-Aware Candidate Selection</div>
  <div class="diagram-container">
    <svg class="diagram-svg" viewBox="0 0 760 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <!-- Step 1: Expert Representations -->
      <rect x="2" y="4" width="138" height="40" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>
      <text x="71" y="19" fill="#0f172a" font-size="8" font-weight="700" text-anchor="middle">MoE Activations</text>
      <text x="71" y="33" fill="#64748b" font-size="6.8" text-anchor="middle">X ~ D_calib (Norm Vectors)</text>
      
      <!-- Arrow 1 -->
      <path d="M141 24 H165" stroke="#0284c7" stroke-width="1.5" marker-end="url(#arrow)"/>
      
      <!-- Step 2: Distance Metric -->
      <rect x="168" y="4" width="144" height="40" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>
      <text x="240" y="19" fill="#0f172a" font-size="8" font-weight="700" text-anchor="middle">Functional Distance</text>
      <text x="240" y="33" fill="#0284c7" font-size="6.8" font-weight="600" text-anchor="middle">D(i, j) = ||C_i - C_j||_2</text>
      
      <!-- Arrow 2 -->
      <path d="M313 24 H337" stroke="#0284c7" stroke-width="1.5" marker-end="url(#arrow)"/>

      <!-- Step 3: Candidate Selection Pool -->
      <rect x="340" y="4" width="140" height="40" rx="4" fill="#f0f9ff" stroke="#0284c7" stroke-width="1.2"/>
      <text x="410" y="19" fill="#0369a1" font-size="8" font-weight="700" text-anchor="middle">Candidate Pool</text>
      <text x="410" y="33" fill="#0369a1" font-size="6.8" font-weight="700" text-anchor="middle">Top-K Nearest Pairs (K=5)</text>
      
      <!-- Arrow 3 -->
      <path d="M481 24 H505" stroke="#0284c7" stroke-width="1.5" marker-end="url(#arrow)"/>

      <!-- Step 4: Transactional Validation Gate -->
      <rect x="508" y="4" width="125" height="40" rx="4" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>
      <text x="570" y="19" fill="#0f172a" font-size="8" font-weight="700" text-anchor="middle">Validation Gate</text>
      <text x="570" y="33" fill="#64748b" font-size="6.8" text-anchor="middle">Greedy argmin ΔKL</text>
      
      <!-- Arrow 4 -->
      <path d="M634 24 H654" stroke="#0284c7" stroke-width="1.5" marker-end="url(#arrow)"/>

      <!-- Step 5: Physical Merging -->
      <rect x="657" y="4" width="100" height="40" rx="4" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <text x="707" y="19" fill="#0f172a" font-size="8" font-weight="700" text-anchor="middle">Consolidation</text>
      <text x="707" y="33" fill="#16a34a" font-size="6.8" font-weight="600" text-anchor="middle">W_new = (W_i + W_j)/2</text>

      <!-- Arrow Marker Definition -->
      <defs>
        <marker id="arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
          <path d="M 0 0 L 6 3 L 0 6 z" fill="#0284c7"/>
        </marker>
      </defs>
    </svg>
  </div>

  <!-- 4. Hero Visual: Multi-Seed Benchmark Curve -->
  <div class="chart-box">
    <img src="data:image/png;base64,{chart_b64}" alt="OLMoE Multi-Seed Compression Perplexity">
    <div class="chart-caption">OLMoE-1B-7B language modeling perplexity across Seeds 42, 43, 44 on WikiText-2 (65,536 tokens, exact token-weighted cross-entropy). External proxies represent zero-shot offline implementations. Error bars show ±1σ across seeds. Lower is better.</div>
  </div>

  <!-- 5. Middle Grid: Core Narrative & Quantitative Benchmark Table -->
  <div class="middle-grid">
    <div>
      <div class="section-title">The Three Scientific Pillars</div>
      <div class="pillar-box">
        <div class="pillar-item">
          <strong>1. The Problem: Merging Without Damage Estimation</strong><br>
          Existing methods merge experts based on router co-activation affinity or static weight distance, without predicting functional interference. Under aggressive consolidation (48 experts, -25% reduction), heuristic clustering baselines collapse (Sub-MoE degrades to 34.84 PPL; HC-SMoE to 31.00 PPL).
        </div>
        <div class="pillar-item">
          <strong>2. Offline Damage Geometry & Motivation</strong><br>
          Offline metric analysis (Exp 4) reveals that the physical merge damage matrix has an intrinsic low-rank structure (q=4 metric MDS captures 75% of damage rank variance). This demonstrates that merge interference is geometrically structured, motivating future research on finding genuinely pre-merge observational approximations that do not require prior merge interventions.
        </div>
        <div class="pillar-item">
          <strong>3. Multi-Seed Compression Results</strong><br>
          At 48 retained experts on OLMoE, CARE-Adaptive achieved the best observed multi-seed perplexity among the evaluated merging approaches: <strong>19.88 ± 0.923 PPL</strong> versus <strong>22.33 ± 0.916</strong> for Random-K5 (directional, n=3). Without dynamic loss feedback, frozen CARE-Static achieves <strong>24.33 ± 0.871 PPL</strong>.
        </div>
      </div>
    </div>

    <div>
      <div class="section-title">Multi-Seed Compression Benchmark (OLMoE @ 48E)</div>
      <div class="table-box">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>Paradigm</th>
              <th>PPL (μ ± σ)</th>
              <th>Held-out KL</th>
              <th>Δ vs Base</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Uncompressed Base</strong></td>
              <td>Reference (64E)</td>
              <td><strong>10.2804</strong></td>
              <td>0.0000</td>
              <td>Baseline</td>
            </tr>
            <tr class="highlight-row">
              <td><strong>CARE-Adaptive</strong></td>
              <td>Dynamic Geometry</td>
              <td><strong>19.88 ± 0.923</strong></td>
              <td><strong>0.658 ± 0.038</strong></td>
              <td>+9.60</td>
            </tr>
            <tr>
              <td><strong>Random-K5 Control</strong></td>
              <td>Unbiased Merge</td>
              <td><strong>22.33 ± 0.916</strong></td>
              <td>0.823 ± 0.043</td>
              <td>+12.05</td>
            </tr>
            <tr>
              <td><strong>CARE-Static</strong></td>
              <td>Frozen Geometry</td>
              <td><strong>24.33 ± 0.871</strong></td>
              <td>0.832 ± 0.051</td>
              <td>+14.05</td>
            </tr>
            <tr>
              <td><strong>MC-SMoE (Stages 1–4 Proxy)</strong></td>
              <td>Correlation Matching</td>
              <td>26.74 ± 0.000</td>
              <td>0.955 ± 0.000</td>
              <td>+16.46</td>
            </tr>
            <tr>
              <td><strong>HC-SMoE (Router Proxy)</strong></td>
              <td>Hierarchical Clustering</td>
              <td>31.00 ± 0.000</td>
              <td>1.082 ± 0.000</td>
              <td>+20.72</td>
            </tr>
            <tr>
              <td><strong>Sub-MoE (Joint SVD Proxy)</strong></td>
              <td>Co-activation Affinity</td>
              <td>34.84 ± 0.000</td>
              <td>1.174 ± 0.000</td>
              <td>+24.56</td>
            </tr>
          </tbody>
        </table>
        <div class="table-note">*All runs evaluated across Seeds 42, 43, 44 on WikiText-2 (65,536 tokens). External baselines represent zero-shot offline proxies without post-merge fine-tuning.</div>
      </div>
    </div>
  </div>

  <!-- 6. Bottom Grid: Scope Bound & Research Question -->
  <div class="bottom-grid">
    <div class="scope-box">
      <strong>Scope Bound & Methodological Safeguards:</strong>
      Evaluation at 48 experts is based on n=3 seeds; paired t-test yields p=0.1330. The advantage over Random-K5 is consistent and directional, but not statistically significant. In exploratory downstream testing (N=100 MMLU), all evaluated merging methods score near the 25% chance floor; capability retention remains unestablished.
    </div>

    <div class="discussion-box">
      <div class="discussion-title">Open Research Question for NVIDIA Discussion</div>
      <div class="discussion-q">
        "Can genuinely pre-merge functional signals identify destructive expert merges without requiring oracle merge measurements?"
      </div>
      <div class="discussion-sub">
        I would value feedback on: (1) designing pre-merge capability representations that approximate the low-rank damage space, (2) architectural limits of functional distance across non-uniform routing, and (3) hybrid architectures combining pruning for dormant experts with geometry-guided merging for active clusters.
      </div>
    </div>
  </div>

  <!-- 7. Footer -->
  <div class="footer">
    <div class="footer-text">
      <strong>Traceability:</strong> Multi-seed benchmark runs traceable to <code>reports/BENCHMARK_L40S/</code>; forensic audit in <code>reports/FINAL_AUDIT_REPORT.md</code>.
    </div>
    <div class="footer-text">
      <strong>Target:</strong> OLMoE-1B-7B • NVIDIA Technical Exchange
    </div>
  </div>

</body>
</html>
"""

with open("reports/care_com_brief.html", "w") as f:
    f.write(html)

print("Generated corrected reports/care_com_brief.html successfully.")
