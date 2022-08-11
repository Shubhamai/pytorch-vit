# Importing Libraries
import argparse
import yaml

import torch

from dataloader import load_mnist, load_foodvision, load_cifar10
from vit import ViT
from trainer import train
from utils import plot_results, reproducibility, save_model


def main(config:dict):
    """Main function."""

    device = config['device'] 

    # Reproducibility
    reproducibility(seed=config["seed"])

    # Load the data
    if config["dataset_name"] == "mnist":
        train_loader, _ = load_mnist(batch_size=config["batch_size"], image_size=config["image_size"])
    elif config["dataset_name"] == "foodvision":
        train_loader, _ = load_foodvision(batch_size=config["batch_size"], image_size=config["image_size"])
    elif config["dataset_name"] == "cifar10":
        train_loader, _ = load_cifar10(batch_size=config["batch_size"], image_size=config["image_size"])
    else:
        assert False, "Unknown dataset name"

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

    # Main configuration
    my_parser.add_argument("--config_path", type=str, help="Path to the config file")

    # Data parameters
    my_parser.add_argument("--dataset_name", choices=["foodvision", "mnist"], default="mnist", help="Name of the dataset")
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
    my_parser.add_argument("--model_path", type=str, default="./experiments/models/mnist_model.pt", help="Model path to save")
    my_parser.add_argument("--results_dir", type=str, default="./experiments/results/mnist", help="Results directory to save results")

    # Making a configuration
    args = my_parser.parse_args()
    config = {}
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    # If config path is mentioned, load the configuration from the file
    if config['config_path']:
        print("Loading configuration from {}".format(config['config_path']))
        with open(config['config_path'], 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

    main(config)
