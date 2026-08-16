import pandas as pd
import numpy as np

def get_pairs(path):
    df = pd.read_csv(path, header=None)
    mat = df.values
    pairs = []
    for i in range(mat.shape[0]):
        for j in range(i+1, mat.shape[1]):
            if not np.isnan(mat[i, j]):
                pairs.append((i, j))
    return set(pairs)

p10 = get_pairs("results/exp3c/checkpoint_10/middle/oracle_distance.csv")
p40 = get_pairs("results/exp3c/checkpoint_40/middle/oracle_distance.csv")
p70 = get_pairs("results/exp3c/checkpoint_70/middle/oracle_distance.csv")
p100 = get_pairs("results/exp3c/checkpoint_100/middle/oracle_distance.csv")

print(f"Pairs 10%: {len(p10)}")
print(f"Pairs 40%: {len(p40)}")
print(f"Pairs 70%: {len(p70)}")
print(f"Pairs 100%: {len(p100)}")

print(f"Overlap 10 & 40: {len(p10.intersection(p40))}")
print(f"Overlap 10 & 70: {len(p10.intersection(p70))}")
print(f"Overlap 40 & 70: {len(p40.intersection(p70))}")
