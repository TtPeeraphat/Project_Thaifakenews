# model_def.py — นิยาม GCNNet ที่เดียว ทุกไฟล์ import จากที่นี่
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(torch.nn.Module):
    """
    Graph Convolutional Network สำหรับจำแนกข่าวจริง/ปลอม

    ใช้ GCNConv ซึ่งรองรับ edge_weight โดยตรง
    ทำให้ความคล้ายกันของข่าว (cosine similarity) ส่งผลต่อ
    graph propagation ได้จริง

    Args:
        in_channels:     ขนาด input feature (768 สำหรับ WangchanBERTa)
        hidden_channels: ขนาด hidden layer (default: 256)
        out_channels:    จำนวน class (2: จริง/ปลอม)
        dropout_rate:    อัตรา dropout (default: 0.4)
    """
    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        out_channels: int = 2,
        dropout_rate: float = 0.4
    ):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_attr', None)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x