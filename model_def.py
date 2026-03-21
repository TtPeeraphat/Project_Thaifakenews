# model_def.py — นิยาม GCNNet ที่เดียว ทุกไฟล์ import จากที่นี่
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(torch.nn.Module):
    """
    Graph Convolutional Network สำหรับจำแนกข่าวจริง/ปลอม

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


# model_def.py — เพิ่ม GraphSAGE version
from torch_geometric.nn import SAGEConv


class GraphSAGENet(torch.nn.Module):
    """
    GraphSAGE สำหรับ inductive learning
    เทรนและ inference ใช้ subgraph topology เหมือนกัน
    ไม่มี transductive/inductive mismatch

    เหมาะสำหรับ production และวิทยานิพนธ์ระดับสูงกว่า GCN
    """
    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        out_channels: int = 2,
        dropout_rate: float = 0.4
    ):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x