import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List
import torch


def plot_losses(
    epochs_seen: torch.Tensor,
    tokens_seen: List[float] | torch.Tensor,
    train_losses: List[float] | torch.Tensor,
    val_losses: List[float] | torch.Tensor,
):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()  # second x axis for for tokens seen
    ax2.plot(
        tokens_seen, train_losses, alpha=0
    )  # invisible plot for aligning ticks (wat?)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()
    return plt
