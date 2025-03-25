import wandb
from config import CONFIG
import torch
import numpy as np
import random

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