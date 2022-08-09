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


def load_model(model_path: str, device:str="cuda") -> nn.Module:
    """Load the model from a file.

    Args:
        model_path (str): The path to load the model from.
        device (str): The device to use.

    Returns:
        nn.Module: The loaded model.
    """

    return torch.load(model_path, map_location=device)


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
        data["step"]["loss"] = savgol_filter(data["step"]["loss"], 21, 2)
        data["step"]["accuracy"] = savgol_filter(data["step"]["accuracy"], 21, 2)

    plt.figure(figsize=(80, 20))

    # Step Loss Chart
    plt.subplot(1, 2, 1)
    plt.plot(data["step"]["loss"])
    plt.title("Step Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # Step Accuracy Chart
    plt.subplot(1, 2, 2)
    plt.plot(data["step"]["accuracy"])
    plt.title("Step Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")

    # # Epoch Accuracy Chart
    # plt.subplot(2, 2, 3)
    # plt.plot(data["epoch"]["loss"])
    # plt.title("Epoch Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")

    # # Epoch Accuracy Chart
    # plt.subplot(2, 2, 4)
    # plt.plot(data["epoch"]["accuracy"])
    # plt.title("Epoch Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")

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
