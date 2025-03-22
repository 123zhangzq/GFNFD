import pickle
import random
import torch
import numpy as np
from torch_geometric.data import Data, HeteroData

from utils import aspect_ratio_normalize


# load the training dataset of neu_exp
def generate_train_data(num_orders=30, num_riders=5, device='cuda', seed=42,
                        raw_data_path='./dataset/train/neu_exp_dataset.pkl'):
    """读取原始数据，生成订单和骑手随机数据，固定随机种子"""

    # 读取 pickle 文件
    with open(raw_data_path, 'rb') as f:
        neu_exp_dataset = pickle.load(f)

    # Set a random seed
    random.seed(seed)

    # Randomly select num_orders orders from neu_exp_dataset
    random_orders = random.sample(list(neu_exp_dataset.keys()), min(num_orders, len(neu_exp_dataset)))

    # Create a subset dictionary for the selected orders
    selected_orders = {key: neu_exp_dataset[key] for key in random_orders}

    # Randomly select num_couriers couriers initial locations from neu_exp_dataset
    random_couriers = random.sample(list(neu_exp_dataset.keys()), min(num_riders, len(neu_exp_dataset)))

    # Create a subset dictionary for the selected orders
    selected_couriers = {key: neu_exp_dataset[key] for key in random_couriers}

    # 生成订单数据
    orders = {
        "pickup_lng": [order['sender_lng'] / 1e8 for order in selected_orders.values()],
        "pickup_lat": [order['sender_lat'] / 1e8 for order in selected_orders.values()],
        "delivery_lng": [order['recipient_lng'] / 1e8 for order in selected_orders.values()],
        "delivery_lat": [order['recipient_lat'] / 1e8 for order in selected_orders.values()]
    }

    # 生成骑手数据
    riders = {
        "courier_lng": [order['grab_lng'] / 1e8 for order in selected_couriers.values()],
        "courier_lat": [order['grab_lat'] / 1e8 for order in selected_couriers.values()]
    }

    yield orders, riders

    '''
    # Example
    test1 = [generate_train_data(30, 5, device='cuda', seed=seed)
             for seed in range(1, 2)]

    epoch = 0
    for orders, riders in test1[epoch]:
        pickup_coor = np.column_stack((orders['pickup_lng'], orders['pickup_lat']))
        delivery_coor = np.column_stack((orders['delivery_lng'], orders['delivery_lat']))
        rider_coor = np.column_stack((riders['courier_lng'], riders['courier_lat']))
        all_coor = np.vstack((pickup_coor, delivery_coor, rider_coor))

        all_coor = aspect_ratio_normalize(all_coor)

        all_order_coor = all_coor[:60]

        epoch += 1 
    '''



# function to calculate distance matrix
def gen_distance_matrix(coordinates):
    n_nodes = len(coordinates)
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e9 # note here
    return distances

# function to generate pyg_data for all orders
def gen_pyg_data_orders(orders_coordinates, k_sparse):
    '''
    生成所有orders的nodes的稀疏图的PyTorch Geometric数据对象。图的nodes前一半为所有orders的pickup node，后一半为对应的delivery node。
    图的nodes features为（x,y,PD_flag），其中前两个为横纵坐标，第三个feature 为一个0/1变量，1代表pickup node。
    图的edge强制连接pickup和对应的delivery node，以及其他k-1个临近node，且图中edge的feature为欧氏距离。

    :param orders_coordinates:
    :param k_sparse:
    :return:
    '''
    n_nodes = len(orders_coordinates)
    assert n_nodes % 2 == 0, "The number of nodes must be an even number!"
    half_n = n_nodes // 2

    distances = gen_distance_matrix(orders_coordinates)

    # 先正常生成topk
    topk_values, topk_indices = torch.topk(distances, k=k_sparse, dim=1, largest=False)

    # 转成list方便操作
    topk_values = topk_values.tolist()
    topk_indices = topk_indices.tolist()

    # 强制连接 (i, i+half_n)
    for i in range(half_n):
        a, b = i, i + half_n
        dist_a_b = distances[a, b].item()
        dist_b_a = distances[b, a].item()

        # 处理 a -> b 方向
        if b not in topk_indices[a]:
            # 找到a那一行里最远的那个
            max_idx = topk_values[a].index(max(topk_values[a]))
            topk_values[a][max_idx] = dist_a_b
            topk_indices[a][max_idx] = b

        # 处理 b -> a 方向
        if a not in topk_indices[b]:
            max_idx = topk_values[b].index(max(topk_values[b]))
            topk_values[b][max_idx] = dist_b_a
            topk_indices[b][max_idx] = a

    # 转回torch
    topk_values = torch.tensor(topk_values, device=orders_coordinates.device, dtype=orders_coordinates.dtype)
    topk_indices = torch.tensor(topk_indices, device=orders_coordinates.device, dtype=torch.long)

    # 构建edge_index
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device), repeats=k_sparse),
        torch.flatten(topk_indices)
    ])
    edge_attr = topk_values.reshape(-1, 1)

    # node features: (x,y,PD_flag)
    node_feature = orders_coordinates
    PD_feature = torch.zeros((n_nodes, 1), device=orders_coordinates.device)
    PD_feature[:half_n, 0] = 1.0
    node_feature = torch.cat([node_feature, PD_feature], dim=1)

    pyg_data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data



# function to generate pyg_data for bipartite graph of order-courier
def gen_pyg_hetero_bigraph(num_orders, num_riders, order_node_emb, rider_coor, on_hand_orders=None):
    """
    生成 HeteroData 结构的订单-骑手二分图
    :param num_orders: 总订单数（包括新订单和on-hand订单）
    :param num_riders: 骑手数
    :param on_hand_orders: dict {order_idx: rider_idx} 已分配订单映射
    :return: torch_geometric.data.HeteroData 对象
    """
    if on_hand_orders is None:
        on_hand_orders = {}

    order_features = []
    rider_features = []

    # 构建 order 特征，(1, h_emb, h_emb), which is (is_on_hand（1/0, emb of p, emb of d)
    for order_idx in range(num_orders):
        is_on_hand = 1 if order_idx in on_hand_orders else 0
        # 拼接：is_on_hand (1维) + order_node_emb[order_idx] + order_node_emb[order_idx + num_orders]
        emb1 = order_node_emb[order_idx]               # shape (h,)
        emb2 = order_node_emb[order_idx + num_orders]  # shape (h,)
        combined_feature = torch.cat([torch.tensor([is_on_hand], dtype=torch.float32), emb1, emb2], dim=0)
        order_features.append(combined_feature)

    # 统计每个 rider 当前手上有几个 order
    rider_load = np.zeros(num_riders, dtype=int)
    for order_idx, rider_idx in on_hand_orders.items():
        rider_load[rider_idx] += 1

    # rider 特征: (x,y, num_on_hand_order)
    for rider_idx in range(num_riders):
        x, y = rider_coor[rider_idx]
        load = rider_load[rider_idx]
        rider_features.append([x, y, load])

    # 构建 edge_index（只存 order -> rider）
    edge_src = []
    edge_dst = []
    edge_attr = []

    # 先加 on-hand 订单（已固定）
    for order_idx, rider_idx in on_hand_orders.items():
        edge_src.append(order_idx)
        edge_dst.append(rider_idx)
        edge_attr.append(np.random.rand())  # 可代表匹配分数或距离

    # 剩下新订单连所有 rider
    for order_idx in range(num_orders):
        if order_idx in on_hand_orders:
            continue
        for rider_idx in range(num_riders):
            edge_src.append(order_idx)
            edge_dst.append(rider_idx)
            edge_attr.append(np.random.rand())

    # 构建 HeteroData
    data = HeteroData()
    data['order'].x = torch.tensor(order_features, dtype=torch.float32)  # shape [num_orders, order_feat_dim]
    data['rider'].x = torch.tensor(rider_features, dtype=torch.float32)  # shape [num_riders, rider_feat_dim]

    # 转成 tensor
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)  # shape [2, num_edges]
    data['order', 'assigns_to', 'rider'].edge_index = edge_index
    data['order', 'assigns_to', 'rider'].edge_attr = torch.tensor(edge_attr, dtype=torch.float32).unsqueeze(-1)

    return data







###### test
from gnn_model import OrderCourierHeteroGNN, NodeEmbedGNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

emb_net = NodeEmbedGNN(feats=3).to(DEVICE)
# oc_net = OrderCourierHeteroGNN()


test1 = [generate_train_data(30, 5, device='cuda', seed=seed)
         for seed in range(1, 2)]

epoch = 0
for orders, riders in test1[epoch]:
    pickup_coor = np.column_stack((orders['pickup_lng'], orders['pickup_lat']))
    delivery_coor = np.column_stack((orders['delivery_lng'], orders['delivery_lat']))
    rider_coor = np.column_stack((riders['courier_lng'], riders['courier_lat']))
    all_coor = np.vstack((pickup_coor, delivery_coor, rider_coor))

    all_coor = aspect_ratio_normalize(all_coor)

    all_order_coor = all_coor[:60]


    all_order_coor_tensor = torch.tensor(all_order_coor, dtype=torch.float32).to(DEVICE)
    pyg_order_node = gen_pyg_data_orders(all_order_coor_tensor, k_sparse=30)
    x_order_node, edge_index_order_node, edge_attr_order_node = pyg_order_node.x, pyg_order_node.edge_index, pyg_order_node.edge_attr
    node_emb = emb_net(x_order_node, edge_index_order_node, edge_attr_order_node)


    epoch += 1
print('finish')
