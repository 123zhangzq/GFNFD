import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, HeteroConv
import torch_geometric.nn

# GNN for Order-Courier Dispatching
class OrderCourierHeteroGNN(nn.Module):
    def __init__(self, order_input_dim, rider_input_dim, hidden_dim, output_dim):
        super(OrderCourierHeteroGNN, self).__init__()

        # 第一层：从 order -> rider 传递信息
        self.conv1 = HeteroConv({
            ('order', 'assigns_to', 'rider'): GATConv((order_input_dim, rider_input_dim), hidden_dim)
        }, aggr='sum')

        # 第二层：可以设计 rider 再传给 order（可选），这里直接 rider self-update
        self.conv2 = HeteroConv({
            ('order', 'assigns_to', 'rider'): GATConv((hidden_dim, hidden_dim), output_dim)
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        x_dict: {'order': order_feats, 'rider': rider_feats}
        edge_index_dict: {('order', 'assigns_to', 'rider'): edge_index}
        edge_attr_dict: optional, 目前 GAT 不用 edge_attr
        """
        # 第一层传播
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # 第二层传播
        x_dict = self.conv2(x_dict, edge_index_dict)

        # 取 rider 节点的最终表示
        rider_embeddings = x_dict['rider']  # shape: (num_rider, output_dim)
        order_embeddings = x_dict['order']  # shape: (num_order, output_dim)

        # 根据 edge_index 计算每条边的匹配分数
        edge_index = edge_index_dict[('order', 'assigns_to', 'rider')]
        order_idx, rider_idx = edge_index

        # 点积得到匹配分数
        edge_scores = (order_embeddings[order_idx] * rider_embeddings[rider_idx]).sum(dim=1)

        return torch.sigmoid(edge_scores)  # 可选 sigmoid 归一化为概率

################################### END GNN for Order-Courier Dispatching #################################


# GNN for Nodes of Orders Embedding
class NodeEmbedGNN(nn.Module):
    def __init__(self, depth=12, feats=2, units=32, act_fn='silu', agg_fn='mean'):
        super().__init__()
        self.depth = depth
        self.feats = feats  # Features
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(torch_geometric.nn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([torch_geometric.nn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(1, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([torch_geometric.nn.BatchNorm(self.units) for i in range(self.depth)])

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        x = x
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x

################################### END GNN for Nodes of Orders Embedding #################################