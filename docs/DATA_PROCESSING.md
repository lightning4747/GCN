# Data Processing and Normalization (`src/utils.py` & `src/data_loader.py`)

Before the GCN can start learning, we must "clean" and "prepare" the graph. This is the most critical step in the entire paper.

## 1. Why Normalization?
The core of the paper is the "Renormalization Trick." 
$$A_{hat} = D^{-1/2} (A + I) D^{-1/2}$$

### The "A + I" part (Self-Loops)
*   **Plain English**: If you only look at your neighbors, you don't look at yourself. 
*   **Why?**: In code (`adj_tilde = adj + sp.eye(adj.shape[0])`), we add a "connection" from every node to itself. This ensures that a node doesn't lose its own features in the "mixing" step.

### The "$D^{-1/2}$" part (Scaling)
*   **Plain English**: Some nodes have 1000 neighbors (like a celebrity), and some only have 2. 
*   **The problem**: If a celebrity "mixes" their features with their 1000 neighbors, the values will become massive, causing the model to "explode."
*   **The solution**: In `normalize_adj()`, we divide by the "degree" (number of neighbors). This "averages" the information so that a celebrity node doesn't have 1000 times more influence than a regular node.

## 2. Reading the Cora Dataset (`src/data_loader.py`)
The `load_data()` function reads two files:
1.  **`.content`**: This contains the node IDs and their 1433 word-features.
2.  **`.cites`**: This contains the "edges" – which paper cites which.

### How it's stored
*   **Features**: Stored as a large Float32 matrix.
*   **Adjacency**: Stored as a **COO (Coordinate)** sparse matrix. A COO matrix only stores **`(row, column, value)`** triplets for actual connections. This is the most educational way to handle sparse data because you can clearly see "who links to whom."

## 3. Why `scipy.sparse`?
We use Scipy for the "heavy lifting" during graph preparation because it is standard and easy to understand. Once the graph is normalized, we convert it to a **PyTorch Sparse Tensor** (`sparse_mx_to_torch_sparse_tensor`) for training.
