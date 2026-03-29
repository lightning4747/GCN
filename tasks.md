# Implementation Tasks

## Phase 1: Core Utilities & Math Specs
- [x] Implement `normalize_adj(adj)`:
  - Add self-loops ($A + I$).
  - Calculate symmetric normalization $\hat{A} = D^{-1/2} A D^{-1/2}$.
- [x] Implement sparse matrix multiplication utilities (using `torch.sparse`).

## Phase 2: Model Architecture
- [/] Implement `GCNLayer`:
  - Input: Node features ($H$), Normalized Adjacency ($\hat{A}$).
  - Operation: $\hat{A} H W$.
  - Parameters: Weight matrix $W$, Optional bias $b$.
- [ ] Implement `GCNModel`:
  - 2-layer structure: `GCNLayer` -> `ReLU` -> `Dropout` -> `GCNLayer` -> `Softmax`.

## Phase 3: Training & Evaluation
- [ ] Implement `masked_cross_entropy`: Loss calculation only on `train_mask`.
- [ ] Implement `accuracy`: Evaluation metric only on `test_mask`.
- [ ] Training Loop: Optimizer setup (Adam), learning rate scheduling, and early stopping (per paper specs).

## Phase 4: Verification
- [ ] Load Cora dataset (common benchmark for GCN).
- [ ] Validate performance: Target ~81% accuracy on Cora test set.
