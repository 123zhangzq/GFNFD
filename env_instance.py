import torch



class HeteroOrderDispatchEnv:
    def __init__(self, hetero_data, model):
        self.data = hetero_data
        self.model = model
        self.num_orders = self.data['order'].num_nodes
        self.num_riders = self.data['rider'].num_nodes

    def run_all(self):
        while True:
            unassigned_mask = self.data['order'].x[:, 0] == 0  # is_on_hand == 0
            if unassigned_mask.sum() == 0:
                break  # 全部分配完成
            self._assign_one_order()
        return self.data

    def _assign_one_order(self):
        # 1. 计算所有边的 score
        edge_attr = self.data['order', 'assigns_to', 'rider'].edge_attr
        out = self.model(self.data.x_dict, self.data.edge_index_dict, {('order', 'assigns_to', 'rider'): edge_attr})
        edge_logits = out['order', 'assigns_to', 'rider'].squeeze(-1)  # (num_edges,)

        # 2. 动态 mask 方案：直接把 is_on_hand==1 的订单相关边打低分
        src_orders = self.data['order', 'assigns_to', 'rider'].edge_index[0]  # 每条边的order_idx
        order_is_assigned = self.data['order'].x[:, 0] > 0  # (num_orders,)
        mask_assigned = order_is_assigned[src_orders]  # (num_edges,) True表示这条边来自已分配订单

        # 将已分配订单产生的边，score压低（不删除）
        edge_logits = edge_logits.masked_fill(mask_assigned, -1e6)

        # 3. sigmoid 转换为概率
        edge_prob = torch.sigmoid(edge_logits)  # (num_edges,)

        # 4. 采样一条边（按GFlowNet风格，从概率分布中sample）
        edge_distribution = torch.distributions.Categorical(probs=edge_prob)
        sampled_edge_idx = edge_distribution.sample().item()

        # 5. 拿到对应 order 和 rider
        order_idx = self.data['order', 'assigns_to', 'rider'].edge_index[0, sampled_edge_idx].item()
        rider_idx = self.data['order', 'assigns_to', 'rider'].edge_index[1, sampled_edge_idx].item()

        # 6. 更新 order 节点状态为已分配
        self.data['order'].x[order_idx, 0] = 1  # is_on_hand=1

        # 7. 更新 graph：移除旧边，只保留这条分配边
        old_edge_index = self.data['order', 'assigns_to', 'rider'].edge_index
        old_edge_attr = self.data['order', 'assigns_to', 'rider'].edge_attr

        # 移除当前order产生的边
        keep_mask = old_edge_index[0] != order_idx
        new_edge_index = old_edge_index[:, keep_mask]
        new_edge_attr = old_edge_attr[keep_mask]

        # 添加分配边
        new_edge_index = torch.cat([new_edge_index, torch.tensor([[order_idx], [rider_idx]], dtype=torch.long, device=new_edge_index.device)], dim=1)
        new_edge_attr = torch.cat([
            new_edge_attr,
            torch.tensor([[1.0]], device=new_edge_attr.device)  # 保证 device 一致
        ], dim=0)

        # 更新HeteroData
        self.data['order', 'assigns_to', 'rider'].edge_index = new_edge_index
        self.data['order', 'assigns_to', 'rider'].edge_attr = new_edge_attr




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
    
    
        env_dispatch = HeteroOrderDispatchEnv(pyg_order_courier, oc_net)
        env_dispatch.run_all()
        
        
        epoch += 1
    '''






###### OLD VERSION
# class OneOrderDispatchInstance:
#     def __init__(self, data, model):
#         """
#         订单-骑手分配环境
#         :param data: PyTorch Geometric Data，包含订单-骑手二分图
#         :param model: 训练好的 GFlowNet-GNN 模型
#         """
#         self.data = data.clone()  # 复制数据，避免修改原始数据
#         self.model = model
#         self.assigned_orders = {}  # 存储已分配的订单及其对应的骑手
#
#     def run_all(self):
#         """
#         运行订单分配，直到所有新订单被分配完，并返回最终的订单分配结果。
#         :return: PyTorch Geometric Data，最终的订单-骑手二分图（每个订单只连接一个骑手）
#         """
#         while len(self.assigned_orders) < self.data.num_nodes // 2:  # 订单数量的一半
#             order_idx, rider_idx = self._select_best_match()
#             if order_idx is None:
#                 break  # 如果没有可用的匹配对，提前结束
#
#             # 更新图，将当前订单标记为已分配
#             self._update_graph(order_idx, rider_idx)
#
#         return self._generate_final_assignment_graph()
#
#     def _select_best_match(self):
#         """
#         计算所有新订单-骑手的匹配分数，选择最佳匹配
#         :return: (订单索引, 骑手索引)
#         """
#         x, edge_index, edge_attr = self.data.x, self.data.edge_index, self.data.edge_attr
#         scores = self.model(x, edge_index, edge_attr)  # GFlowNet 计算所有匹配分数
#
#         # 仅考虑未分配订单的边
#         valid_edges = [(i, j) for i, j in edge_index.T.tolist() if i not in self.assigned_orders]
#         if not valid_edges:
#             return None, None  # 没有可分配订单
#
#         # 获取这些边的匹配分数
#         valid_scores = [scores[i, j].item() for i, j in valid_edges]
#
#         # 选择最高分数的订单-骑手匹配
#         best_idx = np.argmax(valid_scores)
#         return valid_edges[best_idx]
#
#     def _update_graph(self, order_idx, rider_idx):
#         """
#         更新图，标记该订单已分配，并移除与之相关的其他边
#         """
#         self.assigned_orders[order_idx] = rider_idx  # 记录订单 -> 骑手分配
#
#         # 仅保留未分配订单的边
#         mask = ~torch.tensor([i in self.assigned_orders for i in self.data.edge_index[0]], dtype=torch.bool)
#         self.data.edge_index = self.data.edge_index[:, mask]
#         self.data.edge_attr = self.data.edge_attr[mask]
#
#     def _generate_final_assignment_graph(self):
#         """
#         生成最终的订单-骑手二分图（每个订单只连接一个骑手）
#         :return: PyTorch Geometric Data
#         """
#         assigned_orders = list(self.assigned_orders.keys())
#         assigned_riders = list(self.assigned_orders.values())
#
#         # 生成新的 edge_index
#         edge_index = torch.tensor([assigned_orders, assigned_riders], dtype=torch.long)
#
#         # 生成新的 edge_attr（可设置为匹配分数）
#         edge_attr = torch.ones(len(assigned_orders), dtype=torch.float32).view(-1, 1)
#
#         # 构造新的二分图数据
#         final_graph = self.data.clone()
#         final_graph.edge_index = edge_index
#         final_graph.edge_attr = edge_attr
#
#         return final_graph
