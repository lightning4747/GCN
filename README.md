# GCN: Graph Convolutional Networks

A pedagogical implementation of the ["Semi-Supervised Classification with Graph Convolutional Networks"](https://arxiv.org/abs/1609.02907) paper (Kipf & Welling, 2017).

## Prerequisites

This project requires **Python 3.12**. It is recommended to use a virtual environment to manage dependencies.

## Installation and Setup

### 1. Clone the Repository
```powershell
git clone <repository-url>
cd GCN
```

### 2. Create a Virtual Environment
```powershell
python -m venv venv
```

### 3. Activate the Virtual Environment
**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```
**Windows (Command Prompt):**
```cmd
.\venv\Scripts\activate.bat
```
**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

## Running the Project

### Execute Training
Run the following command to start the training process on the Cora dataset:
```powershell
python main.py
```

The script includes logic to automatically check for the Cora dataset in `data/cora/`. If the dataset is not found, it will download and extract the necessary files (`cora.content` and `cora.cites`) from the official source.

## Project Structure

* **`main.py`**: Entry point for training. Handles argument parsing and the execution loop.
* **`src/models.py`**: Defines the GCN model architecture (2-layer stack).
* **`src/layers.py`**: Implementation of the Graph Convolutional Layer.
* **`src/utils.py`**: Mathematical utilities, including adjacency matrix normalization.
* **`src/data_loader.py`**: Logic for loading and parsing the citation network.

## Documentation

Detailed explanations of the implementation are available in the `docs/` directory:

* **ARCHITECTURE.md**: Details on the model structure.
* **THEORY.md**: Breakdown of the mathematical foundations.
* **LAYERS.md**: Technical documentation for `src/layers.py`.
* **DATA_PROCESSING.md**: Overview of graph normalization and preprocessing.