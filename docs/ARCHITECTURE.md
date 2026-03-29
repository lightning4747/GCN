# Project Architecture and 2-Layer Design (`src/models.py`)

The overall "Skeleton" of our GCN model is a 2-layer Graph Convolutional Network. This structure is the most successful architecture for "Cora-like" datasets.

## 1. Why 2 Layers?
In the paper, Kipf & Welling explain that 2 layers are almost always the best choice. Here's why:

*   **Layer 1**: Aggregates information from your **direct friends**.
*   **Layer 2**: Aggregates information from your **friends' friends**.

If you added a 3rd or 4th layer, the information would "mix" so much that every node would start to look exactly the same! This is a known problem in GNNs called **"Over-smoothing."**

## 2. Key Code Explanation

### The Optimizer
```python
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
```
*   **Plain English**: We use the **Adam Optimizer** because it is like a smart search engine for the best weights. 
*   **`weight_decay`**: This is a penalty for weights that are too large. It "regularizes" the model, preventing it from "memorizing" the training nodes.

### Dropout
```python
x = F.dropout(x, self.dropout, training=self.training)
```
*   **Plain English**: In each round of training, we "turn off" 50% of the neurons randomly.
*   **Why?**: This forces the model to find *multiple* ways to explain the data. For example, if it can't rely on word A, it has to look at word B and word C.

### Masked Loss and Accuracy (`src/train.py`)
```python
loss_train = F.nll_loss(output[idx_train], labels[idx_train])
```
*   **Why?**: We only calculate our "mistakes" on the 140 nodes we already know (the `idx_train`).
*   **The Power**: Even though we only learn from 140 nodes, the **Graph Structure** flows the knowledge through the neighbor links, allowing the model to accurately classify over 1000 nodes it has *never* seen before!
