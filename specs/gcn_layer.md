# GCN Layer Specification

## Interface
```python
class GCNLayer:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize weights W and bias b.
        Weights W: (in_features, out_features)
        Bias b: (out_features)
        """
        pass

    def forward(self, x: Tensor, adj_hat: SparseTensor) -> Tensor:
        """
        Compute: Z = adj_hat @ (x @ W) + b
        
        Inputs:
        - x: Node features (N, in_features)
        - adj_hat: Normalized adjacency matrix (N, N)
        
        Returns:
        - Output features (N, out_features)
        """
        pass
```

## Constraints
- `adj_hat` must be symmetrically normalized: $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$.
- Support for sparse-dense multiplication is required for scalability.
- Weight initialization should follow Glorot (Xavier) uniform distribution as per paper.
