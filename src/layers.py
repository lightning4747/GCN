import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    """
    Standard GCN Layer as described in Kipf & Welling (2017).
    Operation: Z = adj_hat * X * W + b
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Trainable parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot (Xavier) initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        # Linear transformation
        support = torch.mm(input, self.weight)
        
        # Neighborhood aggregation
        # Use torch.sparse.mm which is more modern
        output = torch.sparse.mm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features} -> {self.out_features})'
