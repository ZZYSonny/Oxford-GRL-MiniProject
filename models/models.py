import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn

class ModelX(nn.Module):
    def __init__(self, in_dim, dim_out) -> None:
        super().__init__()
        self.gnn = tgnn.GATv2Conv(
            in_channels=in_dim,
            out_channels=in_dim,
            heads=1,
            add_self_loops=False
        )
        self.mlp_out = nn.Linear(in_dim, dim_out)
    
    def forward(self, data):
        h = data.x
        h = self.gnn(h, data.edge_index)
        # Apply mask
        h = h[data.mask]
        h = self.mlp_out(h)
        return h

class ModelQK(nn.Module):
    def __init__(self, query_dim, key_dim, dim_out) -> None:
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.gnn = tgnn.TransformerConv(
            in_channels=query_dim+key_dim,
            out_channels=query_dim+key_dim,
            heads=1,
            add_self_loops=False
        )
        self.mlp_out = nn.Linear(key_dim, dim_out)
    
    def forward(self, data):
        h = torch.cat([
            data.query, 
            F.one_hot(data.key, num_classes=self.key_dim).float(),
        ],dim=1)
        h = self.gnn(h, data.edge_index)
        # Apply mask
        h = h[data.mask]
        h = F.relu(h)
        h = h[:, self.query_dim:]
        h = self.mlp_out(h)
        return h
