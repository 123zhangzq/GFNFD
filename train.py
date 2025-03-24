import torch
import torch.optim as optim
import numpy as np
import wandb
from data_loader import generate_train_data
from config import CONFIG
from utils import init_wandb, log_metrics, sample_preference_vector
from gnn_model import OrderCourierHeteroGNN, NodeEmbedGNN
from env_instance import HeteroOrderDispatchEnv
import os
from lkh3_solver import solve_rider_with_LKH
import platform



def train():
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # 初始化 WandB
    #init_wandb()

    # generate training dataset (orders and couriers)
    train_dataset = [generate_train_data(CONFIG["num_orders"], CONFIG["num_riders"], device=DEVICE, seed=seed)
                     for seed in range(SEED, CONFIG["epochs"])]

    # 初始化 GNN 和 GFlowNet
    gnn_node_emb = NodeEmbedGNN(feats=3).to(DEVICE)
    gnn_order_deispatch = OrderCourierHeteroGNN(order_input_dim = 65, rider_input_dim = 33, edge_attr_dim= 1, hidden_dim = 64, flg_gfn=True).to(DEVICE)

    # 优化器
    optimizer = optim.Adam(list(gnn_order_deispatch.parameters()) + list(gnn_node_emb.parameters()), lr=CONFIG["lr"])

    # 训练循环
    # best_reward = float('-inf')
    # best_solution = None
    for epoch in range(len(train_dataset)):
        for orders, riders in train_dataset[epoch]:

            optimizer.zero_grad()

            # 采样偏好向量 ω
            preference = sample_preference_vector(alpha=CONFIG["preference_alpha"])





        # load one instance
        torch.manual_seed(SEED + epoch)  # 为每个 epoch 设定不同的种子
        index = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[index]

        # 计算订单-骑手匹配分数 (静态计算)
        node_emb = gnn_node_emb()  # dim(num_nodes_ hidden_dim)
        node_features = torch.rand((CONFIG["num_orders"] + CONFIG["num_riders"], 4))  # 订单 + 骑手特征
        # edge_index = torch.randint(0, CONFIG["num_riders"], (2, CONFIG["num_orders"]))  # 随机连接订单 & 骑手
        gnn_output = gnn_order_deispatch(edge_index, node_features, node_emb)

        # 订单分配轨迹采样
        state = torch.rand((CONFIG["num_orders"], 2 + len(preference)))
        actions = gflownet.select_action(state)

        # 计算奖励
        path_lengths = torch.norm(gnn_output[actions] - state[:, :2], dim=1)  # 路径长度
        order_counts = torch.bincount(actions, minlength=CONFIG["num_riders"])  # 每个骑手的订单数
        balance_metric = order_counts.max() - order_counts.min()  # 订单分配均衡性

        # 计算加权奖励
        reward = - (preference[0] * path_lengths.sum() + preference[1] * balance_metric)
        total_reward = reward.item()

        # 计算损失并优化
        loss = -reward
        loss.backward()
        optimizer.step()

        # 记录最优解
        if total_reward > best_reward:
            best_reward = total_reward
            best_solution = actions.detach().cpu().numpy()

        # 记录到 WandB
        log_metrics(epoch, loss.item(), total_reward, best_solution, preference)

        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {loss.item():.4f}, Reward: {total_reward:.4f}, Preference: {preference}")
