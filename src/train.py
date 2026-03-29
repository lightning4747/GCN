import torch
import torch.optim as optim
import torch.nn.functional as F
from src.models import GCN
from src.utils import accuracy

def train(epoch, model, optimizer, features, adj, labels, idx_train, idx_val):
    model.train()
    optimizer.zero_grad()
    
    output = model(features, adj)
    # Masked loss using idx_train
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    
    loss_train.backward()
    optimizer.step()

    # Validation
    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    print(f'Epoch: {epoch+1:04d}',
          f'loss_train: {loss_train.item():.4f}',
          f'acc_train: {acc_train.item():.4f}',
          f'loss_val: {loss_val.item():.4f}',
          f'acc_val: {acc_val.item():.4f}', flush=True)

def test(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test.item():.4f}")
