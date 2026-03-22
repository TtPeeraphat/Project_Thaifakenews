# model_def.py — นิยาม GCNNet ที่เดียว ทุกไฟล์ import จากที่นี่
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x          = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

import torch
sd = torch.load("best_model.pth", map_location="cpu", weights_only=True)
print(list(sd.keys()))