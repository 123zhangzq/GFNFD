# config.py - 训练参数
CONFIG = {
    "run_name": "OrderAssignment",
    "num_orders": 30,        # 订单数量
    "num_riders": 5,         # 骑手数量
    "k_sparse": 40,
    "sample_k": 5,            # sample k times for one GFN in training
    "max_steps": 50,          # GFlowNet 轨迹采样步数
    "seed":42,
    "lr": 0.001,              # 学习率
    "hidden_dim": 64,         # GNN 隐藏层维度
    "batch_size": 32,         # 训练批次大小
    "epochs": 500,             # 训练轮数
    "wandb_project": "GFlowNet_Order_Assignment",  # WandB 项目名称
    "preference_bins": [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)],   # multi-obj weight
    "pretrained": None,       # Loading training checkpoint, e.g., "../pretrained/order_model/checkpoint_epoch10.pt"
    "output": "../pretrained"


}
