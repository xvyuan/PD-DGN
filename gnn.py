import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool as gap
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


# ---------- 有向图卷积 ----------
class DirectedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(DirectedGCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin(x)
        _, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


# ---------- 蛋白质编码器 ----------
class ProteinEncoder(nn.Module):
    def __init__(self, num_features_pro=128, output_dim=128, dropout=0.2):
        super(ProteinEncoder, self).__init__()
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = DirectedGCNConv(num_features_pro * 2, num_features_pro * 4)
        self.pro_fc_g1 = nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data_pro):
        x, edge_index, batch = data_pro.x, data_pro.edge_index, data_pro.batch
        xt = self.pro_conv1(x, edge_index)
        xt = self.relu(xt)
        xt = self.pro_conv2(xt, edge_index)
        xt = self.relu(xt)
        xt = self.pro_conv3(xt, edge_index)
        xt = self.relu(xt)
        xt_g = gap(xt, batch)
        xt_g = self.relu(self.pro_fc_g1(xt_g))
        xt_g = self.dropout(xt_g)
        return self.pro_fc_g2(xt_g)


# ---------- 注意力池化 ----------
class NodeAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_nn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x, batch):
        num_graphs = int(batch.max().item()) + 1
        scores = self.attn_nn(x).squeeze(-1)
        pooled = []
        for graph_id in range(num_graphs):
            mask = batch == graph_id
            x_g = x[mask]
            s_g = scores[mask]
            alpha = torch.softmax(s_g, dim=0).unsqueeze(-1)
            pooled_g = torch.sum(x_g * alpha, dim=0)
            pooled.append(pooled_g)
        return torch.stack(pooled, dim=0)


# ---------- 双向门控融合 ----------
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_a = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.gate_b = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, x_a, x_b):
        cat_feat = torch.cat([x_a, x_b], dim=-1)
        gate_a = self.gate_a(cat_feat)
        gate_b = self.gate_b(cat_feat)
        out_a = x_a * gate_a + x_b * (1.0 - gate_a)
        out_b = x_b * gate_b + x_a * (1.0 - gate_b)
        return out_a, out_b


# ---------- 增强药物编码器 ----------
class DrugEncoder(nn.Module):
    def __init__(self, num_features=78, output_dim=128, dropout=0.2):
        super().__init__()
        hidden_dim = output_dim // 2   # 64
        self.gat1 = GATv2Conv(num_features, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.gcn1 = GCNConv(num_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gate1 = GatedFusion(hidden_dim)
        self.gate2 = GatedFusion(hidden_dim)
        self.pool_gat = NodeAttentionPool(hidden_dim)
        self.pool_gcn = NodeAttentionPool(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_gat = F.relu(self.gat1(x, edge_index))
        x_gcn = F.relu(self.gcn1(x, edge_index))
        x_gat = self.dropout(x_gat)
        x_gcn = self.dropout(x_gcn)
        x_gat, x_gcn = self.gate1(x_gat, x_gcn)

        x_gat = F.relu(self.gat2(x_gat, edge_index))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        x_gat = self.dropout(x_gat)
        x_gcn = self.dropout(x_gcn)
        x_gat, x_gcn = self.gate2(x_gat, x_gcn)

        feat_gat = self.pool_gat(x_gat, batch)
        feat_gcn = self.pool_gcn(x_gcn, batch)
        return torch.cat([feat_gat, feat_gcn], dim=-1)


# ---------- 最终增强模型 ----------
class HybridGNNWithDesc(nn.Module):
    def __init__(self, n_output=1, num_features_mol=78, num_features_pro=128,
                 output_dim=128, dropout=0.2, desc_dim=12):
        super().__init__()
        print("=== HybridGNNWithDesc 增强模型 ===")
        self.drug_branch = DrugEncoder(num_features_mol, output_dim, dropout)
        self.protein_branch = ProteinEncoder(num_features_pro, output_dim, dropout)
        self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.desc_proj = nn.Sequential(
            nn.Linear(desc_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1024),
        )
        self.desc_gate = nn.Sequential(
            nn.Linear(1024 * 2, 1024),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        print("全连接层: 256 -> 1024 -> 256 -> 1")
        print("===============")

    def forward(self, data_mol, data_pro, morgan_fps=None, data_dgl=None):
        drug_feat = self.drug_branch(data_mol)
        prot_feat = self.protein_branch(data_pro)
        base_feat = torch.cat([drug_feat, prot_feat], dim=1)
        h_base = self.fc1(base_feat)
        if not hasattr(data_mol, "mol_desc"):
            raise ValueError("data_mol 缺少 mol_desc 字段，请使用 DTADatasetEnhanced")
        h_desc = self.desc_proj(data_mol.mol_desc)
        gate = self.desc_gate(torch.cat([h_base, h_desc], dim=1))
        h = h_base + gate * h_desc
        h = self.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout(h)
        return self.out(h)