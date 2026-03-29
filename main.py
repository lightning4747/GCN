import time
import argparse
import numpy as np

import torch
import torch.optim as optim

from src.data_loader import load_data
from src.models import GCN
from src.train import train, test

import os
import urllib.request
import tarfile

def check_and_download_data(data_path="data/cora/"):
    # The two main files your load_data() will likely look for
    files = ["cora.content", "cora.cites"]
    dataset_url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    
    # Create directory if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Check if files exist
    missing_files = [f for f in files if not os.path.exists(os.path.join(data_path, f))]
    
    if missing_files:
        print(f"Dataset missing. Downloading Cora dataset from {dataset_url}...")
        try:
            # Download to a temporary location
            archive_path = os.path.join(data_path, "cora.tgz")
            urllib.request.urlretrieve(dataset_url, archive_path)
            
            # Extract the .tgz file
            with tarfile.open(archive_path, "r:gz") as tar:
                # The archive usually has a 'cora/' prefix inside, 
                # we extract everything and then move files if necessary
                tar.extractall(path=data_path)
            
            # Move files out of the extracted 'cora' subfolder if it exists
            extracted_dir = os.path.join(data_path, "cora")
            if os.path.isdir(extracted_dir):
                for f in files:
                    os.rename(os.path.join(extracted_dir, f), os.path.join(data_path, f))
            
            # Clean up the archive
            os.remove(archive_path)
            print("Download and extraction complete.")
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please manually download the dataset to data/cora/")
    else:
        print("Cora dataset already present.")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else 
                     ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)

def main():
    # Ensure data is present
    check_and_download_data()
    
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    # Move tensors to device
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    # Move model to device
    model.to(device)

    # Training loop
    t_total = time.time()
    print(f"DEBUG: args.epochs = {args.epochs}")
    try:
        for epoch in range(args.epochs):
            train(epoch, model, optimizer, features, adj, labels, idx_train, idx_val)
            print(f"DEBUG: Completed epoch {epoch+1}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}")
    
    print("Optimization Finished!")
    print(f"Total time elapsed: {time.time() - t_total:.4f}s")

    # Testing
    test(model, features, adj, labels, idx_test)

if __name__ == "__main__":
    main()
