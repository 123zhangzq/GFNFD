import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, HeteroConv, TransformerConv
import torch_geometric.nn
import math

# GNN for Order-Courier Dispatching
class OrderCourierHeteroGNN(nn.Module):
    def __init__(self, order_input_dim, rider_input_dim, edge_attr_dim, hidden_dim, omega_dim,
                 flg_gfn=False, Z_out_dim=1):
        super(OrderCourierHeteroGNN, self).__init__()

        # GAT
        self.conv = HeteroConv({
            ('order', 'assigns_to', 'rider'): TransformerConv((order_input_dim, rider_input_dim), hidden_dim,
                                                              edge_dim=edge_attr_dim)
        }, aggr='sum')

        # order投影到hidden_dim
        self.order_proj = nn.Linear(order_input_dim, hidden_dim)


        # 最终 edge score MLP（融合 order、rider 和 ω）
        self.residual_mlp  = nn.Sequential(
            nn.Linear(hidden_dim * 2 + omega_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # 输出一个分数
            #nn.Tanh()  # 限制在 (-1, 1) , 乘系数调节幅度 if need
        )

        # Z for GFlowNet
        self.Z_net = nn.Sequential(
            nn.Linear(order_input_dim - 1 + omega_dim, 64),  # input dim is order dim [1:], the first one is 0/1 for assignment or not
            nn.ReLU(),
            nn.Linear(64, Z_out_dim),
            nn.Softplus()
        ) if flg_gfn else None

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, omega_encoded):
        # GAT
        order_embeddings = x_dict['order']  # (num_orders, order_input_dim)
        x_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict)
        rider_embeddings = x_dict['rider']  # (num_riders, hidden_dim)

        edge_index = edge_index_dict[('order', 'assigns_to', 'rider')]
        order_idx, rider_idx = edge_index

        # 把 order 先投影
        order_proj_embed = self.order_proj(order_embeddings)  # (num_orders, hidden_dim)


        # 拼接 order_embed、rider_embed、omega 编码后的特征
        num_edges = order_idx.size(0)
        omega_expand = omega_encoded.unsqueeze(0).expand(num_edges, -1)
        order_edge_embed = order_proj_embed[order_idx]  # [num_edges, hidden_dim]
        rider_edge_embed = rider_embeddings[rider_idx]  # [num_edges, hidden_dim]

        dot_scores = (order_proj_embed[order_idx] * rider_embeddings[rider_idx]).sum(dim=1)
        dot_scores = dot_scores / math.sqrt(self.order_proj.out_features)

        residual_input = torch.cat([order_edge_embed, rider_edge_embed, omega_expand], dim=-1)  # [num_edges, hidden*2 + omega_enc]

        # 通过最终打分MLP计算edge分数
        residual_score = self.residual_mlp(residual_input).squeeze(-1)  # [num_edges]

        # 总分数：点积 + 残差
        edge_scores = dot_scores + residual_score

        # Clamp 输出范围
        edge_scores = torch.clamp(edge_scores, -10, 10)

        return {('order', 'assigns_to', 'rider'): edge_scores}

    def cal_logz(self, emb, omega_encoded):
        """
        emb: shape [num_orders, order_input_dim - 1]
        omega: shape [omega_dim]
        """
        assert self.Z_net is not None

        # pool emb
        pooled_emb = emb.mean(dim=0)  # shape: [order_input_dim - 1]

        # 拼接
        z_input = torch.cat([pooled_emb, omega_encoded], dim=-1)  # shape: [order_input_dim -1 + omega_encode_dim]

        logZ = self.Z_net(z_input)  # 直接输出 1 个 logZ
        return logZ


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

        self.mlp_head = nn.Sequential(
            nn.Linear(self.units, self.units * 2),
            nn.ReLU(),
            nn.Linear(self.units * 2, 2)
        )

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
        #return x
        return self.mlp_head(x)

################################### END GNN for Nodes of Orders Embedding #################################