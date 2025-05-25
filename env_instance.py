import torch



class HeteroOrderDispatchEnv:
    def __init__(self, hetero_data, model, preference):
        self.data = hetero_data
        self.model = model
        self.num_orders = self.data['order'].num_nodes
        self.num_riders = self.data['rider'].num_nodes
        self.preference_omega = preference

    def get_logz(self):
        emb_pd = self.data['order'].x[:, 1:]
        log_Z = self.model.cal_logz(emb=emb_pd, omega_encoded=self.preference_omega)
        return log_Z

    def run_all(self, epsilon=0, flg_train=False, flg_infer=False):
        log_probs_list = []
        while True:
            unassigned_mask = self.data['order'].x[:, 0] == 0  # is_on_hand == 0
            if unassigned_mask.sum() == 0:
                break  # 全部分配完成
            log_probs = self._assign_one_order(epsilon=epsilon, flg_train=flg_train, flg_infer=flg_infer)
            log_probs_list.append(log_probs)
        if flg_train:
            return self.data, torch.stack(log_probs_list)
        else:
            return self.data

    def _assign_one_order(self, epsilon=0, flg_train=False, flg_infer=False):
        # 1. 计算所有边的 score
        edge_attr = self.data['order', 'assigns_to', 'rider'].edge_attr
        out = self.model(self.data.x_dict, self.data.edge_index_dict, {('order', 'assigns_to', 'rider'): edge_attr}, self.preference_omega)
        edge_logits = out['order', 'assigns_to', 'rider'].squeeze(-1)  # (num_edges,)

        # 2. 动态 mask 方案：直接把 is_on_hand==1 的订单相关边打低分
        src_orders = self.data['order', 'assigns_to', 'rider'].edge_index[0]  # 每条边的order_idx
        order_is_assigned = self.data['order'].x[:, 0] > 0  # (num_orders,)
        mask_assigned = order_is_assigned[src_orders]  # (num_edges,) True表示这条边来自已分配订单

        # 将已分配订单产生的边，score压低（不删除）
        edge_logits = edge_logits.masked_fill(mask_assigned, -1e6)

        # 3. sigmoid 转换为概率
        edge_prob = torch.sigmoid(edge_logits)  + 1e-8 # (num_edges,)
        edge_prob = edge_prob.masked_fill(mask_assigned, 0)

        # 4. 采样一条边（按GFlowNet风格，从概率分布中sample）
        if flg_train:
            EPSILON = epsilon
        elif flg_infer:
            EPSILON = 0.7
        else:
            EPSILON = -1  # 推理时彻底禁用探索
        edge_distribution = torch.distributions.Categorical(probs=edge_prob)
        if torch.rand(1).item() < EPSILON:
            # 以 epsilon 概率进行探索（随机采样）
            action = edge_distribution.sample()
        else:
            # 以 (1 - epsilon) 概率进行利用（选最大概率的边）
            action = torch.argmax(edge_prob)
        sampled_edge_idx = action.item()
        log_probs = edge_distribution.log_prob(action)


        # 5. 拿到对应 order 和 rider
        order_idx = self.data['order', 'assigns_to', 'rider'].edge_index[0, sampled_edge_idx].item()
        rider_idx = self.data['order', 'assigns_to', 'rider'].edge_index[1, sampled_edge_idx].item()

        # 6. 更新 order 节点状态为已分配
        self.data['order'].x = self.data['order'].x.clone()
        self.data['order'].x[order_idx, 0] = 1
        # self.data['order'].x[order_idx, 0] = 1  # is_on_hand=1

        # 7. update rider node feature
        self.data['rider'].x = self.data['rider'].x.clone()
        self.data['rider'].x[rider_idx, 0] += 1

        # 8. 更新 graph：移除旧边，只保留这条分配边
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


        return log_probs


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
