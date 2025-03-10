import numpy as np
import torch


def generate_data(num_orders=100, num_riders=20):
    # 随机生成订单坐标 (x, y) 和优先级
    orders = {
        "positions": np.random.rand(num_orders, 2) * 10,
        "priorities": np.random.rand(num_orders),
    }

    # 随机生成骑手坐标 (x, y) 和当前状态
    riders = {
        "positions": np.random.rand(num_riders, 2) * 10,
        "current_load": np.zeros(num_riders),  # 初始接单数
        "path_length": np.zeros(num_riders),  # 初始路径长度
    }

    return orders, riders
