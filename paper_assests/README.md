# CARE-COM Asset Manifest (ICLR 2027)

**Purpose:** This repository serves as the complete, traceable source-of-truth for all figures, tables, equations, architecture diagrams, and scientific claims used in the CARE-COM ICLR 2027 submission.

## Directory Tree
- `ASSET_MANIFEST.csv`: Inventory of all generated assets.
- `CLAIM_TO_EVIDENCE.md`: Validation of all core scientific claims.
- `DATA_INTEGRITY_AUDIT.md`: Fixes ensuring no missing data was plotted as zero.
- `EXPERIMENT_MAP.md`: Traceability of conclusions to specific codebase experiments.
- `NEGATIVE_RESULTS.md`: Documentation of failed experimental mechanisms (Exp 7).
- `LITERATURE_AUDIT.md`: Framing against static expert merging literature.
- `references.bib`: BibTeX citations.
- `equations.md`: Formal mathematical definitions.
- `MAIN_PAPER_ASSET_PLAN.md`: Proposed asset placement within the 9-page limit.
- `Figure_Captions.md`: ICLR-style detailed captions for all figures.
- `REPRODUCIBILITY.md`: Mapping of scripts to output artifacts.
- `ICLR_2027_SUBMISSION_CHECKLIST.md`: Compliance checklist.
- `AI_USE_STATEMENT.md`: Official LLM disclosure statement.
- `architecture/`: Mermaid diagrams.
- `tables/`: Rendered markdown tables for the paper.
- `fig1_method_overview/` to `fig6_efficiency/`: Generated scientific visualizations (PDF/PNG) and READMEs.

## ICLR Checklist Status
All assets are rigorously anonymized. The submission adheres strictly to the 9-page main text limit (exclusive of references and appendices), and an AI Use Statement has been drafted.

## Known Limitations & Missing Assets
- **Hardware Agnostic Runtime:** Figure 6 documents measured wall-clock time but emphasizes that true algorithmic complexity across disparate codebases (e.g., HC-SMoE) requires deeper structural analysis.
- **KL Divergence on Baselines:** Sub-MoE and HC-SMoE cumulative KL measurements are unavailable without invasive intervention tracking and have been explicitly omitted from the plots rather than falsely cast to zero.
