# Importing Libraries
import argparse
import yaml

import torch

from dataloader import load_mnist, load_foodvision, load_cifar10
from trainer import test
from utils import load_model, plot_results


def main(config):
    """Main function."""

    device = config['device']

    # Load the data
    if config["dataset_name"] == "mnist":
        _, test_loader = load_mnist(batch_size=config["batch_size"], image_size=config["image_size"])
    elif config["dataset_name"] == "foodvision":
        _, test_loader = load_foodvision(batch_size=config["batch_size"], image_size=config["image_size"])
    elif config["dataset_name"] == "cifar10":
        _, test_loader = load_cifar10(batch_size=config["batch_size"], image_size=config["image_size"])
    else:
        assert False, "Unknown dataset name"

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

    # Main Configuration
    my_parser.add_argument("--config_path", type=str, help="Path to the config file")

    # Data parameters
    my_parser.add_argument("--dataset_name", choices=["foodvision", "mnist"], default="mnist", help="Name of the dataset")
    my_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    # Model testing parameters
    my_parser.add_argument("--image_size",type=int,default=28,help="Image size (height and width must be equal)")
    my_parser.add_argument("--device", type=str, default="cuda",choices=["cuda", "cpu"], help="Device to use")

    # Other parameters
    my_parser.add_argument("--model_path", type=str, default="./experiments/models/mnist_model.pt", help="Model directory")
    my_parser.add_argument("--results_dir", type=str, default="./experiments/results/mnist", help="Results directory")

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
