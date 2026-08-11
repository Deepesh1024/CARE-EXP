import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import numpy as np
from config import Q_VALUES
import phase3_cross_validation as p3

def profile():
    D = np.random.rand(64, 64)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    
    train_idx = np.arange(51)
    test_idx = np.arange(51, 64)
    q = 4
    seed = 42
    
    # Profile SMACOF
    start = time.time()
    D_train = D[np.ix_(train_idx, train_idx)]
    Z_train = p3.run_smacof_train(D_train, q, seed)
    smacof_time = time.time() - start
    
    # Profile OOS embedding
    start = time.time()
    for t, test_exp in enumerate(test_idx):
        d_to_train = D[test_exp, train_idx]
        z_test = p3.embed_single_test_expert(
            Z_train, d_to_train, q,
            seed=(seed * 10000 + t) % (2**32 - 1),
        )
    oos_time = time.time() - start
    
    print(f"SMACOF Time (D_train 51x51, q={q}): {smacof_time:.4f}s")
    print(f"OOS Time (13 experts, restarts={p3.OOS_N_RESTARTS}): {oos_time:.4f}s")
    print(f"OOS Time per expert: {oos_time/len(test_idx):.4f}s")

if __name__ == "__main__":
    profile()
