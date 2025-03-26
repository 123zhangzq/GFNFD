import wandb
from config import CONFIG
import torch
import numpy as np
import random
import math
import itertools

def init_wandb():
    wandb.init(
        project=CONFIG["wandb_project"],
        config=CONFIG,
        name="gflownet_training",
        resume="allow"
    )

def log_metrics(epoch, loss, reward, best_solution, preference):
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "reward": reward,
        "best_solution": best_solution,
        "preference": preference,
    })



def aspect_ratio_normalize(coords):
    min_vals = np.min(coords, axis=0)
    max_vals = np.max(coords, axis=0)
    scale = max(max_vals - min_vals)  # 保证xy尺度一致
    norm_coords = (coords - min_vals) / scale
    return norm_coords


def sample_uniform_per_bin(BINS, DEVICE):
    """
    均匀在每个bin内采样ω[0]，ω[1]自动补齐
    """
    bin_idx = random.randint(0, len(BINS) - 1)  # 均匀采bin
    bin_start, bin_end = BINS[bin_idx]
    omega_0 = np.random.uniform(bin_start, bin_end)
    omega_1 = 1.0 - omega_0
    omega = torch.tensor([omega_0, omega_1], dtype=torch.float32).to(DEVICE)
    return omega, bin_idx

def non_uniform_thermometer_encode(value, DEVICE):
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    encoding = torch.zeros(len(thresholds), dtype=torch.float32).to(DEVICE)
    for i, threshold in enumerate(thresholds):
        if value >= threshold:
            encoding[i] = 1.0
        else:
            break
    return encoding


def get_epsilon_exp(epoch, total_epochs, eps_start=0.2, eps_end=0.05):
    decay_rate = math.log(eps_end / eps_start) / total_epochs
    epsilon = eps_start * math.exp(decay_rate * epoch)
    return max(epsilon, eps_end)



def compute_1order_distance(rider_data):
    """
    计算从 depot 出发，依次访问 pickup 和 delivery 再返回 depot 的总距离（欧氏距离）
    输入：
        rider_dict: 包含 rider 的 start 和任务坐标信息的字典
        rider_idx: 当前要计算的骑手编号
    返回：
        总欧氏距离（float）
    """
    rider = rider_data
    depot = rider['start']  # depot 坐标
    task = rider['tasks'][0]
    pickup = (task[0], task[1])
    delivery = (task[2], task[3])

    # 计算欧氏距离
    def euclidean(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    total_distance = (
        euclidean(depot, pickup) +
        euclidean(pickup, delivery) +
        euclidean(delivery, depot)
    )

    return total_distance


def compute_2order_min_distance(rider_data):
    """
    计算从 depot 出发，访问两个订单（各含 pickup 和 delivery）的最短总距离。
    要求每个订单必须先访问 pickup，再访问 delivery，起点是 depot，路径必须回到 depot。

    参数：
        rider_data (dict): 含 depot 起点 和 两个任务，每个任务是 [px, py, dx, dy]

    返回：
        float: 所有合法路径中的最小欧氏距离
    """
    depot = rider_data['start']
    task1 = rider_data['tasks'][0]
    task2 = rider_data['tasks'][1]

    # 构造点位标签
    points = {
        'p1': (task1[0], task1[1]),
        'd1': (task1[2], task1[3]),
        'p2': (task2[0], task2[1]),
        'd2': (task2[2], task2[3]),
    }

    # 所有可能的访问顺序（只对 p1, d1, p2, d2 排列）
    all_perms = itertools.permutations(['p1', 'd1', 'p2', 'd2'])

    def is_valid(seq):
        # 每个订单必须满足 pickup 在 delivery 之前
        return seq.index('p1') < seq.index('d1') and seq.index('p2') < seq.index('d2')

    def euclidean(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    min_total = float('inf')

    for perm in all_perms:
        if not is_valid(perm):
            continue

        # 构造完整路径：depot -> ... -> depot
        path = [depot] + [points[label] for label in perm] + [depot]

        # 累积路径距离
        total_dist = sum(euclidean(path[i], path[i + 1]) for i in range(len(path) - 1))

        if total_dist < min_total:
            min_total = total_dist

    return min_total