import os
import json
import numpy as np

meta_path = "results/exp7b/activations/expert_signatures_meta.json"
memmap_path = "results/exp7b/activations/expert_signatures.npy"

if not os.path.exists(meta_path) or not os.path.exists(memmap_path):
    print("Files not found!")
    exit(1)

with open(meta_path, "r") as f:
    meta = json.load(f)

shape = tuple(meta["shape"])
dtype = np.float16 if meta["dtype"] == "float16" else np.float32

print(f"Meta Shape: {shape}")
print(f"Meta Dtype: {dtype}")

activations = np.memmap(memmap_path, dtype=dtype, mode='r', shape=shape)

print(f"Actual shape: {activations.shape}")
print(f"Actual dtype: {activations.dtype}")

# Basic Stats
isnan = np.isnan(activations).sum()
isinf = np.isinf(activations).sum()
print(f"NaNs: {isnan}, Infs: {isinf}")

print(f"Min: {np.min(activations):.4f}, Max: {np.max(activations):.4f}")
print(f"Mean: {np.mean(activations):.4f}, Std: {np.std(activations):.4f}")

# Check if neurons or experts are identical
expert_diff = np.sum(np.abs(activations[0] - activations[1])) if shape[0] > 1 else -1
print(f"Diff between expert 0 and 1: {expert_diff}")

neuron_diff = np.sum(np.abs(activations[0, 0] - activations[0, 1])) if shape[1] > 1 else -1
print(f"Diff between expert 0's neuron 0 and 1: {neuron_diff}")

# Find a pair to compute similarity
cand_df = __import__("pandas").read_csv("results/exp7b/candidates/candidate_pairs.csv")
# Look for E7 and E19
e1 = 7
e2 = 19

expert_mapping = meta["expert_mapping"]
if str(e1) in expert_mapping and str(e2) in expert_mapping:
    idx1 = expert_mapping[str(e1)]
    idx2 = expert_mapping[str(e2)]
else:
    # Just grab the first two
    e1 = int(list(expert_mapping.keys())[0])
    e2 = int(list(expert_mapping.keys())[1])
    idx1 = expert_mapping[str(e1)]
    idx2 = expert_mapping[str(e2)]

print(f"Computing sanity check cosine similarity for E{e1} and E{e2}...")
# Extract a few neurons from each, e.g. first 5 neurons
n1 = activations[idx1, :5, :].astype(np.float32)
n2 = activations[idx2, :5, :].astype(np.float32)

# Compute cosine similarity between all pairs of these 5 neurons
norm1 = np.linalg.norm(n1, axis=1, keepdims=True)
norm2 = np.linalg.norm(n2, axis=1, keepdims=True)
# Avoid division by zero
norm1[norm1 == 0] = 1e-10
norm2[norm2 == 0] = 1e-10

sim = np.dot(n1, n2.T) / (norm1 * norm2.T)
print("Cosine Similarity Matrix (5x5):")
print(sim)
