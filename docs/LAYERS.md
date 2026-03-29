# Understanding GCN Layers (`src/layers.py`)

The `GCNLayer` is the fundamental building block of this project. It performs the specific "Graph Convolution" math defined in the paper.

## 1. What does a Layer actually do?
Typically, a standard neural network layer does this:
$$Output = Input \cdot Weights$$

A **GCN Layer** adds a "Friendship Mixer" to the calculation:
$$Output = Neighbors \cdot (Input \cdot Weights)$$

## 2. Key Code Implementation

### The Parameters
```python
self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
```
*   **Why?**: These are the learnable parts of the graph. We use **Xavier Initialization** (`nn.init.xavier_uniform_`) because it helps the model start with stable numbers, preventing them from being too small or too large.

### The Forward Pass
```python
support = torch.mm(input, self.weight)
output = torch.sparse.mm(adj, support)
```
1.  **`torch.mm(input, self.weight)`**: First, we transform the features of each node. We think of this as "processing" the raw data into a hidden representation.
2.  **`torch.sparse.mm(adj, support)`**: Then, we "mix" those results based on the adjacency matrix (`adj`). Every node now "averages" the processed data of its neighbors.

## 3. Why `torch.sparse.mm`?
In most graphs, most nodes are NOT connected. If we used a normal matrix, 99.9% of it would be zeros.
*   **Plain English**: Imagine a map of all people on Earth. You only know 500. Storing the "No, I don't know this person" for the other 7 billion people is a waste of memory.
*   **Implementation Choice**: We use **Sparse Tensors** to only store the actual "Friendships" (the non-zeros). This makes the code fast enough to run on a laptop!
