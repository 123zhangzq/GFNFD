import torch
import numpy as np

class OneOrderDispatchInstance:
    def __init__(self, data, model):
        """
        订单-骑手分配环境
        :param data: PyTorch Geometric Data，包含订单-骑手二分图
        :param model: 训练好的 GFlowNet-GNN 模型
        """
        self.data = data.clone()  # 复制数据，避免修改原始数据
        self.model = model
        self.assigned_orders = {}  # 存储已分配的订单及其对应的骑手

    def run_all(self):
        """
        运行订单分配，直到所有新订单被分配完，并返回最终的订单分配结果。
        :return: PyTorch Geometric Data，最终的订单-骑手二分图（每个订单只连接一个骑手）
        """
        while len(self.assigned_orders) < self.data.num_nodes // 2:  # 订单数量的一半
            order_idx, rider_idx = self._select_best_match()
            if order_idx is None:
                break  # 如果没有可用的匹配对，提前结束

            # 更新图，将当前订单标记为已分配
            self._update_graph(order_idx, rider_idx)

        return self._generate_final_assignment_graph()

    def _select_best_match(self):
        """
        计算所有新订单-骑手的匹配分数，选择最佳匹配
        :return: (订单索引, 骑手索引)
        """
        x, edge_index, edge_attr = self.data.x, self.data.edge_index, self.data.edge_attr
        scores = self.model(x, edge_index, edge_attr)  # GFlowNet 计算所有匹配分数

        # 仅考虑未分配订单的边
        valid_edges = [(i, j) for i, j in edge_index.T.tolist() if i not in self.assigned_orders]
        if not valid_edges:
            return None, None  # 没有可分配订单

        # 获取这些边的匹配分数
        valid_scores = [scores[i, j].item() for i, j in valid_edges]

        # 选择最高分数的订单-骑手匹配
        best_idx = np.argmax(valid_scores)
        return valid_edges[best_idx]

    def _update_graph(self, order_idx, rider_idx):
        """
        更新图，标记该订单已分配，并移除与之相关的其他边
        """
        self.assigned_orders[order_idx] = rider_idx  # 记录订单 -> 骑手分配

        # 仅保留未分配订单的边
        mask = ~torch.tensor([i in self.assigned_orders for i in self.data.edge_index[0]], dtype=torch.bool)
        self.data.edge_index = self.data.edge_index[:, mask]
        self.data.edge_attr = self.data.edge_attr[mask]

    def _generate_final_assignment_graph(self):
        """
        生成最终的订单-骑手二分图（每个订单只连接一个骑手）
        :return: PyTorch Geometric Data
        """
        assigned_orders = list(self.assigned_orders.keys())
        assigned_riders = list(self.assigned_orders.values())

        # 生成新的 edge_index
        edge_index = torch.tensor([assigned_orders, assigned_riders], dtype=torch.long)

        # 生成新的 edge_attr（可设置为匹配分数）
        edge_attr = torch.ones(len(assigned_orders), dtype=torch.float32).view(-1, 1)

        # 构造新的二分图数据
        final_graph = self.data.clone()
        final_graph.edge_index = edge_index
        final_graph.edge_attr = edge_attr

        return final_graph
