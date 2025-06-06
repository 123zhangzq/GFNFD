# config.py - 训练参数
CONFIG = {
    "run_name": "OrderAssignment",
    "num_orders": 10,        # 订单数量, e.g., 30
    "num_riders": 3,         # 骑手数量, e.g., 5
    "k_sparse": 21,          # 40
    "sample_k": 3,            # sample k times for one GFN in training, e.g.,10
    "seed": 42,
    "lr": 0.001,              # 学习率
    "hidden_dim": 64,         # GNN 隐藏层维度
    "batch_size": 20,         # 训练批次大小
    "epochs": 1000,             # 训练轮数
    # "wandb_project": "GFlowNet_Order_Assignment",  # WandB 项目名称
    "preference_bins": [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)],   # multi-obj weight
    #"preference_bins": [(0.01, 0.02)],   # only for f1
    "pretrained": None,       # Loading training checkpoint, e.g.,"./pretrained/10_3/checkpoint_epoch0.pt"
    "output": "./pretrained"


}
