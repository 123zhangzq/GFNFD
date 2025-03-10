import torch
import torch.optim as optim
import wandb
from data_loader import generate_data
from gnn_model import OrderRiderGNN
from gflownet import GFlowNet
from config import CONFIG
from utils import init_wandb, log_metrics

# Initialize WandB
init_wandb()

# 生成订单 & 骑手数据
orders, riders = generate_data(CONFIG["num_orders"], CONFIG["num_riders"])

# 初始化 GNN 和 GFlowNet
gnn = OrderRiderGNN(input_dim=4, hidden_dim=CONFIG["hidden_dim"], output_dim=2)
gflownet = GFlowNet(state_dim=2, action_dim=CONFIG["num_riders"], hidden_dim=CONFIG["hidden_dim"])

# 优化器
optimizer = optim.Adam(list(gnn.parameters()) + list(gflownet.parameters()), lr=CONFIG["lr"])

# 训练循环
best_reward = float('-inf')
best_solution = None

for epoch in range(CONFIG["epochs"]):
    optimizer.zero_grad()

    # 计算订单-骑手匹配分数 (静态计算)
    edge_index = torch.randint(0, CONFIG["num_riders"], (2, CONFIG["num_orders"]))  # 随机连接订单 & 骑手
    node_features = torch.rand((CONFIG["num_orders"] + CONFIG["num_riders"], 4))  # 订单 + 骑手特征
    gnn_output = gnn(edge_index, node_features)

    # 订单分配轨迹采样
    state = torch.rand((CONFIG["num_orders"], 2))
    actions = gflownet.select_action(state)

    # 计算奖励
    reward = -torch.norm(gnn_output[actions] - state, dim=1)  # 偏好最短路径
    total_reward = reward.sum().item()

    # 计算损失并优化
    loss = -torch.sum(reward)
    loss.backward()
    optimizer.step()

    # 记录最优解
    if total_reward > best_reward:
        best_reward = total_reward
        best_solution = actions.detach().cpu().numpy()

    # 记录到 WandB
    log_metrics(epoch, loss.item(), total_reward, best_solution)

    print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {loss.item():.4f}, Reward: {total_reward:.4f}")
