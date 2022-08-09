# Importing Libraries
import torch

from dataloader import load_mnist
from model.vit import ViT
from trainer import test
from utils import load_model, plot_results


def main():
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    _, test_loader = load_mnist()

    # Create the model
    model = ViT(
        patch_size=7,
        image_size=28,
        in_channels=1,
        embed_dim=64,
        n_encoder=2,
        n_heads=8,
        n_classes=10,
    ).to(device)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Load the model
    load_model(model, model_path="./experiments/models/model.pt")

    # Test the model
    test_data = test(model, test_loader, device=device, criterion=criterion)

    # Plot the results
    plot_results(
        test_data, save_target_dir="./experiments/results", name="test_results.jpg"
    )


if __name__ == "__main__":

    main()
