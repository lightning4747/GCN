# GCN Implementation Plan: Spec-Driven Development

Based on the paper: "Semi-Supervised Classification with Graph Convolutional Networks" (Kipf & Welling, 2017).

## 1. Mathematical Specification
The core operation of a graph convolutional layer is:
$$Z = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X W$$
Where:
- $A \in \mathbb{R}^{N \times N}$ is the adjacency matrix.
- $\tilde{A} = A + I_N$ (Self-loops added).
- $\tilde{D}$ is the diagonal degree matrix of $\tilde{A}$, $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$.
- $X \in \mathbb{R}^{N \times C}$ is the input feature matrix.
- $W \in \mathbb{R}^{C \times F}$ is the learnable weight matrix.

### Model Architecture
A two-layer GCN for semi-supervised node classification:
$$Z = \text{softmax}(\hat{A} \text{ReLU}(\hat{A}XW^{(0)})W^{(1)})$$
where $\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$.

## 2. Component Design
- **Graph Processor**: Handles adjacency matrix normalization and self-loop injection.
- **GCN Layer**: Implements the sparse-matrix multiplication and learned transformation.
- **Model Wrapper**: Combines layers, activations, and dropout.
- **Loss Function**: Masked cross-entropy (computed only on labeled nodes).

## 3. Data Interface
- **Input**:
  - `adj`: Sparse adjacency matrix (COO or CSR format).
  - `features`: Dense or sparse feature matrix.
  - `labels`: Class indices for nodes.
  - `masks`: Boolean arrays (train/val/test) for semi-supervised splits.
