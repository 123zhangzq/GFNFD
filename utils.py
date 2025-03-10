import wandb
from config import CONFIG

def init_wandb():
    wandb.init(
        project=CONFIG["wandb_project"],
        config=CONFIG,
        name="gflownet_training",
        resume="allow"
    )

def log_metrics(epoch, loss, reward, best_solution):
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "reward": reward,
        "best_solution": best_solution
    })
