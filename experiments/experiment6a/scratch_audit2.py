from scratch_audit import get_pairs
p1 = get_pairs("results/exp3c/checkpoint_10/first/oracle_distance.csv")
p2 = get_pairs("results/exp3c/checkpoint_10/middle/oracle_distance.csv")
p3 = get_pairs("results/exp3c/checkpoint_10/last/oracle_distance.csv")
print(f"Overlap first & middle: {len(p1.intersection(p2))}")
print(f"Overlap middle & last: {len(p2.intersection(p3))}")
