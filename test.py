# Importing Libraries
import argparse

import torch

from dataloader import load_mnist
from trainer import test
from utils import load_model, plot_results


def main(config):
    """Main function."""

    device = config['device']

    # Load the data
    _, test_loader = load_mnist(batch_size=config["batch_size"])

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Load the model
    model = load_model(model_path=config["model_path"], device=device)

    # Test the model
    test_data = test(model, test_loader, device=device, criterion=criterion)

    # Plot the results
    plot_results(
        test_data, save_target_dir=config['results_dir'], name="test_results.jpg"
    )


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(description="Testing script for ViT")

    # Data parameters
    my_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    # Model testing  parameters
    my_parser.add_argument("--device", type=str, default="cuda",choices=["cuda", "cpu"], help="Device to use")

    # Other parameters
    my_parser.add_argument("--model_path", type=str, default="./experiments/models/model.pt", help="Model directory")
    my_parser.add_argument("--results_dir", type=str, default="./experiments/results", help="Results directory")

    # Making a configuration
    args = my_parser.parse_args()
    config = {}
    for arg in vars(args):
        config[arg] = getattr(args, arg)

    main(config)
