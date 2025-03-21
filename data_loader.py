import pickle
import random
import torch
import numpy as np
from torch_geometric.data import Data

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
def gen_pyg_data_bigraph(num_orders, num_riders, on_hand_orders=None):
    """
    生成带已分配订单的二分图数据
    :param num_orders: 总订单数（包括新订单和on-hand订单）
    :param num_riders: 骑手数
    :param on_hand_orders: dict {order_idx: rider_idx} 已分配订单映射 (order和rider都是int索引)
    :return: PyTorch Geometric Data对象
    """
    if on_hand_orders is None:
        on_hand_orders = {}

    edges = []
    edge_features = []

    # 订单节点编号 [0, num_orders-1]
    # 骑手节点编号 [num_orders, num_orders + num_riders - 1]

    # 处理已分配订单，直接绑定到对应骑手
    for order_idx, rider_idx in on_hand_orders.items():
        edges.append((order_idx, num_orders + rider_idx))  # rider索引偏移
        edge_features.append(np.random.rand())  # 随机匹配分数或成本

    # 剩余新订单，对每个订单连接所有骑手
    for order_idx in range(num_orders):
        if order_idx in on_hand_orders:
            continue  # 已分配订单跳过
        for rider_idx in range(num_riders):
            edges.append((order_idx, num_orders + rider_idx))
            edge_features.append(np.random.rand())

    # 生成 node features
    # order features: is_on_hand (1/0)
    order_features = []
    for order_idx in range(num_orders):
        is_on_hand = 1 if order_idx in on_hand_orders else 0
        order_features.append([is_on_hand])

    # rider features: 先填0（后续可以加负载、位置等）
    rider_features = [[0] for _ in range(num_riders)]

    x = torch.tensor(order_features + rider_features, dtype=torch.float32)  # shape: (num_nodes, 1)
    edge_index = torch.tensor(edges, dtype=torch.long).T  # shape: (2, num_edges)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32).view(-1, 1)  # shape: (num_edges, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_orders + num_riders)







###### test

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
print('finish')
