# Component Reuse Map

| Component | Source Experiment | Reusable? | Why? | Modifications |
|---|---|---|---|---|
| **Feature Extraction (Original)** | Exp 1 (`CARE_MoE_V3_E1.py`) | YES | Successfully extracted local features (weights, activations, routing) on calibration data. | Must adapt to the `OlmoeExperts` batched architecture. |
| **Feature Extraction (CARE Descriptors)** | Exp 2 (`phase1_descriptors.py`) | YES | Formulations for NPMI, JSD, Usage_Asymmetry, Spec_Diff are validated. | Adapt input formats from Exp 1 extraction for OLMoE structure. |
| **Geometry / MDS** | Exp 4 (`mds_embedding.py`) | YES | Contains robust, validated SMACOF and Out-Of-Sample embedding. | None. Plug and play. |
| **Local Predictor (CARE_COM)** | Exp 4 (`phase2_regression.py`) | YES | Pre-trained XGBoost weights contain the learned mapping from features to Oracle KL. | Need to load the serialized `.pkl` or `.json` model from Exp 4 rather than retraining. |
| **Oracle KL Calculator** | Exp 1 / Exp 3B | YES | Evaluates KL divergence accurately for post-hoc validation. | Must ensure it properly injects the new merge operator for OLMoE. |
| **Model Loader** | General / Exp 3C | YES | `AutoModelForCausalLM.from_pretrained` works correctly. | None. |
| **Calibration Data** | Exp 3C (`config.py`) | YES | Wikitext-2-raw-v1 parameterization is well-defined. | Re-use caching mechanism. |
| **Benchmark Suite** | N/A | NO | No existing evaluation suite found. | Must write new evaluation scripts for Perplexity and any chosen benchmarks. |
