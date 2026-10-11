import json
import numpy as np
import pandas as pd
from collections import Counter

def main():
    with open("experiments/experiment8/results.json", "r") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # 1. Exact number of unique actual_delta_ce values and their counts.
    unique_ce_counts = df['actual_damage'].value_counts()
    print("=== Unique actual_damage values and counts (Top 20) ===")
    print(unique_ce_counts.head(20))
    print(f"Total unique values: {len(unique_ce_counts)}")
    
    # 2. Min/Max/Mean/Std/Median actual ΔCE
    print("\n=== Global Stats for actual_damage ===")
    print(df['actual_damage'].describe())
    
    # 3. Group ΔCE by source expert j
    print("\n=== Stats grouped by source expert j (Head 10) ===")
    group_j = df.groupby('j')['actual_damage'].agg(['mean', 'std', 'min', 'max', 'nunique'])
    print(group_j.head(10))
    
    # 4. Group ΔCE by target expert i
    print("\n=== Stats grouped by target expert i (Head 10) ===")
    group_i = df.groupby('i')['actual_damage'].agg(['mean', 'std', 'min', 'max', 'nunique'])
    print(group_i.head(10))

if __name__ == "__main__":
    main()
