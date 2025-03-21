import wandb
from config import CONFIG
import numpy as np

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


def sample_preference_vector(alpha=1.0, size=2):
    """
    从 Dirichlet 分布中采样偏好向量 ω。
    """
    return np.random.dirichlet([alpha] * size)


def aspect_ratio_normalize(coords):
    min_vals = np.min(coords, axis=0)
    max_vals = np.max(coords, axis=0)
    scale = max(max_vals - min_vals)  # 保证xy尺度一致
    norm_coords = (coords - min_vals) / scale
    return norm_coords