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

    '''
    ################################## Example-to-use ###################################
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    emb_net = NodeEmbedGNN(feats=3).to(DEVICE)
    oc_net = OrderCourierHeteroGNN(order_input_dim = 65, rider_input_dim = 33, edge_attr_dim= 1, hidden_dim = 64).to(DEVICE)
    
    
    test1 = [generate_train_data(30, 5, device='cuda', seed=seed)
             for seed in range(1, 2)]
    
    epoch = 0
    for orders, riders in test1[epoch]:
        pickup_coor = np.column_stack((orders['pickup_lng'], orders['pickup_lat']))
        delivery_coor = np.column_stack((orders['delivery_lng'], orders['delivery_lat']))
        rider_coor = np.column_stack((riders['courier_lng'], riders['courier_lat']))
        all_coor = np.vstack((pickup_coor, delivery_coor, rider_coor))
    
        all_coor = aspect_ratio_normalize(all_coor)
    
        all_order_coor_tensor = torch.tensor(all_coor, dtype=torch.float32).to(DEVICE)
        pyg_node_emb = gen_pyg_data_nodes(all_order_coor_tensor, num_orders=30, k_sparse=30)
        x_order_node, edge_index_order_node, edge_attr_order_node = pyg_node_emb.x, pyg_node_emb.edge_index, pyg_node_emb.edge_attr
        node_emb = emb_net(x_order_node, edge_index_order_node, edge_attr_order_node)
    
        orders_emb = node_emb[:60, :]
        riders_emb = node_emb[60:, :]
    
        pyg_order_courier = gen_pyg_hetero_bigraph(num_orders=30, num_riders=5, order_emb= orders_emb, rider_emb=riders_emb)
    
        edge_attr = pyg_order_courier['order', 'assigns_to', 'rider'].edge_attr
        output_score = oc_net(pyg_order_courier.x_dict, pyg_order_courier.edge_index_dict, {('order', 'assigns_to', 'rider'): edge_attr})
    '''


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