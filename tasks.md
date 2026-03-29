# Implementation Tasks

## Phase 0: Environment Setup
- [x] Create Python 3.12 venv.
- [x] Create `requirements.txt`.
- [ ] Ensure local package availability.

## Phase 1: Core Utilities & Math Specs
- [x] Implement `normalize_adj(adj)`:
  - Add self-loops ($A + I$).
  - Calculate symmetric normalization $\hat{A} = D^{-1/2} A D^{-1/2}$.
- [x] Implement sparse matrix multiplication utilities (using `torch.sparse`).

## Phase 2: Model Architecture
- [x] Implement `GCNLayer`:
  - Input: Node features ($H$), Normalized Adjacency ($\hat{A}$).
  - Operation: $\hat{A} H W$.
  - Parameters: Weight matrix $W$, Optional bias $b$.
- [x] Implement `GCNModel`:
  - 2-layer structure: `GCNLayer` -> `ReLU` -> `Dropout` -> `GCNLayer` -> `Softmax`.

## Phase 3: Training & Evaluation
- [x] Implement `masked_cross_entropy`: Loss calculation only on `train_mask`.
- [x] Implement `accuracy`: Evaluation metric only on `test_mask`.
- [x] Training Loop: Model training, validation, and testing logic.
    - [x] Analyze GCN paper requirements (Kipf & Welling)
    - [x] Design Spec-Driven Development structure
    - [x] Create `plan.md` with mathematical specifications
    - [x] Create `tasks.md` with implementation milestones
- [/] Execution Phase
    - [/] Setup Environment (Python 3.12 venv)
    - [x] Implement core math utilities (Normalizing adjacency matrix)
get ~81% accuracy on Cora test set.

## Phase 4: Verification
- [ ] Load Cora dataset (Manual step: place data in `data/cora/`).
- [ ] Validate performance: Run `python main.py` once dependencies are installed.
