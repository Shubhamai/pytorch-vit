# Importing Libraries
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter


def save_model(model: nn.Module, target_path: str) -> None:
    """Save the model to a file.

    Args:
        model (nn.Module): The model to save.
        target_dir (str): The path to save the model to.

    Returns:
        None
    """

    if not os.path.exists(os.path.dirname(target_path)):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

    torch.save(model, target_path)


def load_model(model_path: str, device: str = "cuda") -> nn.Module:
    """Load the model from a file.

    Args:
        model_path (str): The path to load the model from.
        device (str): The device to use.

    Returns:
        nn.Module: The loaded model.
    """

    return torch.load(model_path, map_location=device)


"""
Using smooth function for loss and accuracy plot from https://stackoverflow.com/a/68510722
All the credits for the function goes to the author of the answer - https://stackoverflow.com/users/985012/tomselleck
"""


def smooth_line(scalars: list, weight: float) -> list:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def plot_results(
    data: dict,
    save_target_dir: str = "results",
    name: str = "results.jpg",
    smooth: bool = True,
) -> None:
    """Plot the results of the training or testing.

    Args:
        data (dict): The training data.
        save_path (str): The path to save the plot to.
        name (str): The name of the plot.
        smooth (bool): Whether to smooth the results.

    Returns:
        None
    """

    if not os.path.exists(save_target_dir):
        os.makedirs(save_target_dir, exist_ok=True)

    if smooth:
        data["step"]["loss"] = smooth_line(data["step"]["loss"], weight=0.9)
        data["step"]["accuracy"] = smooth_line(data["step"]["accuracy"], weight=0.9)

    plt.figure(figsize=(10, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Step Loss Chart

    ax1.plot(data["step"]["loss"], color="red")
    ax1.set_title("Step Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")

    # Step Accuracy Chart
    ax2.plot(data["step"]["accuracy"], color="blue")
    ax2.set_title("Step Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(save_target_dir, name))


def reproducibility(seed: int = 42):
    """Set the random seed.

    Args:
        seed (int): The seed to use.

    Returns:
        None
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)
