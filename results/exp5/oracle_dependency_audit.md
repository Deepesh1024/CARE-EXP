# Oracle Dependency Audit

## Trace of the Dependency Graph for CARE

To deploy CARE_COM or CARE_GEO on the real OLMoE model, we must ensure Oracle KL is not a hidden dependency at inference time.

### Inputs for Feature Construction
- **Real Model (OLMoE):** Weights, Biases. (INPUT)
- **Calibration Data (Wikitext):** Text sequences. (INPUT)

### Feature Extraction (Pre-Merge Descriptors)
- **Features Extracted:** Weight_Distance, Weight_Cosine, Activation_Similarity, Output_Similarity, Routing_Similarity, Usage_Frequency, Jaccard_Overlap, Usage_Asymmetry, Routing_JSD_Proxy, Routing_NPMI_Proxy, Specialization_Diff.
- **Dependency:** Derived purely from running the original model on calibration data. 
- **Oracle Dependency:** CLEAN. None of these features require running experimental merges or evaluating Oracle KL.

### CARE_GEO
- **Representation:** Constructs a distance matrix from the functional statistics, then uses MDS (SMACOF) to embed into $q=4$ space. (REPRESENTATION)
- **Decision Signal:** Ranks candidates based on pairwise Euclidean distance $||z_i - z_j||_2$ in the embedding space. (DECISION SIGNAL)
- **Oracle Dependency:** CLEAN. The embedding and pairwise distances are computed purely from the pre-merge features. 

### CARE_COM
- **Predictor:** An XGBoost model combining the 11 local features and the `Geometry_Distance`.
- **Training Signal:** The XGBoost model was trained in Experiment 4 using `Oracle_KL` as the target variable. (TRAINING SIGNAL)
- **Inference/Decision Signal:** During deployment (Experiment 5), the model uses only the 12 pre-merge features to predict merge damage. (DECISION SIGNAL)
- **Oracle Dependency:** CLEAN for deployment. The model *learned* from Oracle KL offline in Exp 4, but it does *not* require Oracle KL to execute its predictions during the real compression trajectory in Exp 5. 

### Audit Evaluation
- **Oracle KL:** POST-HOC EVALUATION ONLY (and historical offline training).
- **Oracle Dependency for Deployment:** **CLEAN**.

CARE can be constructed and executed entirely from `REAL MODEL + CALIBRATION DATA` without Oracle information during the compression process.
