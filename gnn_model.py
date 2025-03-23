import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, HeteroConv
import torch_geometric.nn

# GNN for Order-Courier Dispatching
class OrderCourierHeteroGNN(nn.Module):
    def __init__(self, order_input_dim, rider_input_dim, edge_attr_dim, hidden_dim):
        super(OrderCourierHeteroGNN, self).__init__()
        self.conv = HeteroConv({
            ('order', 'assigns_to', 'rider'): GATConv((order_input_dim, rider_input_dim), hidden_dim,
                                                      add_self_loops=False)
        }, aggr='sum')

        # order投影到hidden_dim
        self.order_proj = nn.Linear(order_input_dim, hidden_dim)

        # Edge MLP：灵活支持多维 edge_attr 输入
        self.edge_gate_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出在0-1之间，做 gating
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        order_embeddings = x_dict['order']  # (num_orders, order_input_dim)
        x_dict = self.conv(x_dict, edge_index_dict)
        rider_embeddings = x_dict['rider']  # (num_riders, hidden_dim)

        edge_index = edge_index_dict[('order', 'assigns_to', 'rider')]
        order_idx, rider_idx = edge_index

        # 把 order 先投影
        order_proj_embed = self.order_proj(order_embeddings)  # (num_orders, hidden_dim)

        # 点积
        raw_scores = (order_proj_embed[order_idx] * rider_embeddings[rider_idx]).sum(dim=1)

        # 处理 edge_attr
        edge_attr = edge_attr_dict[('order', 'assigns_to', 'rider')]  # [num_edges, edge_attr_dim]
        gate = self.edge_gate_mlp(edge_attr).squeeze(-1)  # [num_edges]

        # 结合 gate，生成最终匹配分数
        edge_scores = raw_scores * gate

        return {('order', 'assigns_to', 'rider'): edge_scores}





################################### END GNN for Order-Courier Dispatching #################################


# GNN for Nodes of Orders Embedding
class NodeEmbedGNN(nn.Module):
    def __init__(self, depth=12, feats=3, units=32, act_fn='silu', agg_fn='mean'):
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