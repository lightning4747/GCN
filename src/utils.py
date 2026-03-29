import numpy as np
import scipy.sparse as sp
import torch
import warnings

def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix:
    A_hat = D^-1/2 * (A + I) * D^-1/2
    """
    # Add self-loops (A + I)
    adj = sp.coo_matrix(adj)
    adj_tilde = adj + sp.eye(adj.shape[0])
    
    # Calculate degree matrix D_tilde
    rowsum = np.array(adj_tilde.sum(1))
    
    # D^-1/2
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Compute D^-1/2 * A_tilde * D^-1/2
    return d_mat_inv_sqrt.dot(adj_tilde).dot(d_mat_inv_sqrt).tocoo()

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
    # Validation
    if indices.max() >= shape[0] or indices.min() < 0:
        raise ValueError(f"Indices out of bounds for shape {shape}")
    
    # Silence invariant warning and return coalesced tensor
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        t = torch.sparse_coo_tensor(indices, values, shape)
        return t.coalesce()
