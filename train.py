# Importing Libraries
import argparse

import torch

from dataloader import load_mnist
from model.vit import ViT
from trainer import train
from utils import plot_results, reproducibility, save_model


def main(config):
    """Main function."""

    device = config['device'] 

    # Reproducibility
    reproducibility(seed=config["seed"])

    # Load the data
    train_loader, _ = load_mnist(batch_size=config["batch_size"])

    # Create the model
    model = ViT(
        patch_size=config["patch_size"],
        image_size=config["image_size"],
        in_channels=config["in_channels"],
        embed_dim=config["embed_dim"],
        n_encoder=config["n_encoder"],
        n_heads=config["n_heads"],
        n_classes=config["n_classes"],
    ).to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    train_data = train(
        model, train_loader, optimizer, criterion, device=device, epoch=config["epochs"]
    )

    # Save the model
    save_model(model, target_path=config["model_path"])

    # Plot the results
    plot_results(train_data, save_target_dir=config["results_dir"], name="train_results.jpg")


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(description="Training script for ViT")

    # Data parameters
    my_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    # Model training  parameters
    my_parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    my_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    my_parser.add_argument("--device", type=str, default="cuda",choices=["cuda", "cpu"], help="Device to use")

    # Model parameters
    my_parser.add_argument("--patch_size", type=int, default=7, help="Patch size")
    my_parser.add_argument("--image_size",type=int,default=28,help="Image size (height and width must be equal)")
    my_parser.add_argument("--in_channels", type=int, default=1, help="Input channels (1 for grayscale)")
    my_parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension size")
    my_parser.add_argument( "--n_encoder", type=int, default=2, help="Number of encoder layers per head")
    my_parser.add_argument("--n_heads", type=int, default=8, help="Number of heads in the multi-head attention")
    my_parser.add_argument("--n_classes", type=int, default=10, help="Number of classes in the dataset")

    # Other parameters
    my_parser.add_argument("--seed", type=int, default=42, help="Seed")
    my_parser.add_argument("--model_path", type=str, default="./experiments/models/model.pt", help="Model path to save")
    my_parser.add_argument("--results_dir", type=str, default="./experiments/results", help="Results directory to save results")

    # Making a configuration
    args = my_parser.parse_args()
    config = {}
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    main(config)
