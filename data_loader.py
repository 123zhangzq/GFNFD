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
def gen_pyg_data_nodes(all_coordinates, num_orders, k_sparse):
    '''
    生成所有nodes的稀疏图的PyTorch Geometric数据对象。图的nodes由两部分，第一部分前一半为所有orders的pickup node，后一半为对应的
    delivery node。图的后一部分为couriers的初始node。
    图的nodes features为（x,y,PD_flag），其中前两个为横纵坐标，第三个feature 为一个0/1变量，0代表 delivery node。
    图的edge强制连接pickup和对应的delivery node，以及其他k-1个临近node，且图中edge的feature为欧氏距离。

    :param all_coordinates:
    :param num_orders:
    :param k_sparse:
    :return:
    '''

    order_nodes = 2 * num_orders
    n_nodes = len(all_coordinates)

    half_n = order_nodes // 2

    distances = gen_distance_matrix(all_coordinates)

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
    topk_values = torch.tensor(topk_values, device=all_coordinates.device, dtype=all_coordinates.dtype)
    topk_indices = torch.tensor(topk_indices, device=all_coordinates.device, dtype=torch.long)

    # 构建edge_index
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device), repeats=k_sparse),
        torch.flatten(topk_indices)
    ])
    edge_attr = topk_values.reshape(-1, 1)

    # node features: (x,y,PD_flag)
    node_feature = all_coordinates
    PD_feature = torch.zeros((n_nodes, 1), device=all_coordinates.device)
    PD_feature[:half_n, 0] = 1.0
    PD_feature[order_nodes:, 0] = 1.0
    node_feature = torch.cat([node_feature, PD_feature], dim=1)

    pyg_data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data



# function to generate pyg_data for bipartite graph of order-courier
def gen_pyg_hetero_bigraph(num_orders, num_riders, order_emb, rider_emb, on_hand_orders=None):
    """
    生成 HeteroData 结构的订单-骑手二分图
    :param num_orders: 总订单数（包括新订单和on-hand订单）
    :param num_riders: 骑手数
    :param on_hand_orders: dict {order_idx: rider_idx} 已分配订单映射
    :return: torch_geometric.data.HeteroData 对象
    """

    DEVICE_ = order_emb.device
    if on_hand_orders is None:
        on_hand_orders = {}

    order_features = []
    rider_features = []

    # 构建 order 特征，维度为 (1, h_emb, h_emb), which indicates (is_on_hand 1/0, emb of p, emb of d)
    for order_idx in range(num_orders):
        is_on_hand = 1 if order_idx in on_hand_orders else 0
        # 拼接：is_on_hand (1维) + order_node_emb[order_idx] + order_node_emb[order_idx + num_orders]
        emb1 = order_emb[order_idx]               # shape (h,)
        emb2 = order_emb[order_idx + num_orders]  # shape (h,)
        combined_feature = torch.cat([torch.tensor([is_on_hand], dtype=torch.float32, device=DEVICE_), emb1, emb2], dim=0)
        order_features.append(combined_feature)

    # 统计每个 rider 当前手上有几个 order
    rider_load = np.zeros(num_riders, dtype=int)
    for order_idx, rider_idx in on_hand_orders.items():
        rider_load[rider_idx] += 1

    # rider 特征: (num_on_hand_order, h_emb), which indicate the embed of courier's location, and the num of on-hand orders
    for rider_idx in range(num_riders):
        emb = rider_emb[rider_idx, :]
        load = rider_load[rider_idx]
        combined_feature = torch.cat([torch.tensor([load], dtype=torch.float32, device=DEVICE_), emb], dim=0)
        rider_features.append(combined_feature)

    # 构建 edge_index（只存 order -> rider）
    edge_src = []
    edge_dst = []
    edge_attr = []

    # 先加 on-hand 订单（已固定）, featuer 是一个1
    for order_idx, rider_idx in on_hand_orders.items():
        edge_src.append(order_idx)
        edge_dst.append(rider_idx)
        edge_attr.append(1.0)  # TODO: think whether need features here

    # 剩下新订单连所有 rider, featuer 是一个0.2
    for order_idx in range(num_orders):
        if order_idx in on_hand_orders:
            continue
        for rider_idx in range(num_riders):
            edge_src.append(order_idx)
            edge_dst.append(rider_idx)
            edge_attr.append(0.2) # TODO: think whether need features here

    # 构建 HeteroData
    data = HeteroData()# shape [num_orders, order_feat_dim]
    data['order'].x = torch.stack(order_features, dim=0).to(dtype=torch.float32, device=DEVICE_)
    data['rider'].x = torch.stack(rider_features, dim=0).to(dtype=torch.float32, device=DEVICE_)# shape [num_riders, rider_feat_dim]

    # 正向边：order -> rider
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=DEVICE_)  # shape [2, num_edges]
    data['order', 'assigns_to', 'rider'].edge_index = edge_index
    data['order', 'assigns_to', 'rider'].edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=DEVICE_).unsqueeze(-1)


    return data







###### test
from gnn_model import OrderCourierHeteroGNN, NodeEmbedGNN
from env_instance import HeteroOrderDispatchEnv

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


    env_dispatch = HeteroOrderDispatchEnv(pyg_order_courier, oc_net)
    env_dispatch.run_all()




    epoch += 1
print('finish')
