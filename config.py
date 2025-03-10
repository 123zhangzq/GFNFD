# config.py - 训练参数
CONFIG = {
    "num_orders": 100,        # 订单数量
    "num_riders": 20,         # 骑手数量
    "max_steps": 50,          # GFlowNet 轨迹采样步数
    "lr": 0.001,              # 学习率
    "hidden_dim": 64,         # GNN 隐藏层维度
    "batch_size": 32,         # 训练批次大小
    "epochs": 50,             # 训练轮数
}
