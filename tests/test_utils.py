import numpy as np
import scipy.sparse as sp
from src.utils import normalize_adj

def test_normalize_adj():
    # Example 3x3 adjacency matrix
    adj = sp.coo_matrix([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    
    norm_adj = normalize_adj(adj)
    
    # D_tilde = [2, 3, 2]
    # D^-1/2 = [0.707, 0.577, 0.707]
    # A_tilde = [1, 1, 0], [1, 1, 1], [0, 1, 1]
    
    # Expected diagonal elements:
    # 0,0: 1 * 0.707 * 0.707 = 0.5
    # 1,1: 1 * 0.577 * 0.577 = 0.333
    # 2,2: 1 * 0.707 * 0.707 = 0.5
    
    np.testing.assert_allclose(norm_adj.diagonal(), [0.5, 0.33333333, 0.5], atol=1e-5)
    print("test_normalize_adj passed!")

if __name__ == "__main__":
    test_normalize_adj()
