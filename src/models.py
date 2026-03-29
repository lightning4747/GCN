import torch.nn as nn
import torch.nn.functional as F
from src.layers import GCNLayer

class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network.
    
    Structure:
    Input -> GCNLayer -> ReLU -> Dropout -> GCNLayer -> LogSoftmax
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # First layer with ReLU activation
        x = F.relu(self.gc1(x, adj))
        
        # Dropout for regularization
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Second layer
        x = self.gc2(x, adj)
        
        # Output log-probabilities
        return F.log_softmax(x, dim=1)
