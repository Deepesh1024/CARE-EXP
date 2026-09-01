# VENUE / SCOPE AUDIT

## 1. Search Scope
We executed a comprehensive search across all markdown (`.md`) and text (`.txt`) files within the `CARE-MoE/Experiments-V3` repository, specifically looking for the terms "NeurIPS", "ICLR", "ICML", and "Interspeech".

## 2. Findings
- **NeurIPS**: The only reference found in the repository is a citation to the SHAP paper (Lundberg & Lee, 2017) in `results/exp2/report2.md`.
- **ICLR**: Found only in the source code of dependencies (e.g., `transformers`).
- **ICML**: Found only in the source code of dependencies (e.g., `scikit-learn`, `networkx`).
- **Interspeech**: No references found anywhere in the repository.

## 3. Conflicting References
The conflicting references mentioned (NeurIPS workshop, ICLR, ICML 2027, Interspeech 2027) **do not exist in the current committed research-plan documents** (such as `README.md`, `full_report.md`, or `research_roadmap.md`). 

## 4. Conclusion
It appears that the venue targets are either in a different repository, an uncommitted draft, or external documentation (e.g., shared documents, Notion, Slack).
- **Stale/Missing**: The repository itself contains *no* active venue targeting information.
- **Action Required**: The user must manually confirm the target venue, as the codebase itself is agnostic and lacks any venue-specific formatting or stated objectives.
