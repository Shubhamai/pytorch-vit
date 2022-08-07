# Importing Libraries
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def save_model(model: nn.Module, target_dir: str, name: str = "model.pt") -> None:
    """Save the model to a file.

    Args:
        model (nn.Module): The model to save.
        target_dir (str): The directory to save the model to.
        name (str): The name of the model.

    Returns:
        None
    """

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(target_dir, name))


def load_model(model: nn.Module, model_path: str) -> nn.Module:
    """Load the model from a file.

    Args:
        model (nn.Module): The model to load.
        model_path (str): The path to load the model from.

    Returns:
        nn.Module: The loaded model.
    """

    return model.load_state_dict(torch.load(model_path))


def plot_results(
    data: dict,
    save_target_dir: str = "results",
    name: str = "results.jpg",
) -> None:
    """Plot the results of the training or testing.

    Args:
        data (dict): The training data.
        save_path (str): The path to save the plot to.

    Returns:
        None
    """

    if not os.path.exists(save_target_dir):
        os.makedirs(save_target_dir, exist_ok=True)

    plt.figure(figsize=(30, 15))

    # Loss Chart
    plt.subplot(1, 2, 1)
    plt.plot(data["step"]["loss"])
    plt.title("Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # Accuracy Chart
    plt.subplot(1, 2, 2)
    plt.plot(data["step"]["accuracy"])
    plt.title("Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")

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
