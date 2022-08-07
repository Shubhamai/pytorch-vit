# Importing Libraries
import torch
from vit_pytorch import ViT

from dataloader.mnist import load_mnist
from trainer import test
from utils import load_model, plot_results


def main():
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    _, test_loader = load_mnist()

    # Create the model
    model = ViT(
        image_size=28,
        patch_size=7,
        channels=1,
        num_classes=10,
        dim=64,
        depth=2,
        heads=8,
        mlp_dim=128,
        dropout=0.1,
        emb_dropout=0.1,
    ).to(device)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Load the model
    load_model(model, model_path="./experiments/models/model.pt")

    # Test the model
    test_data = test(model, test_loader, device=device, criterion=criterion)

    # Plot the results
    plot_results(test_data,save_target_dir="./experiments/results", name="test_results.jpg")


if __name__ == "__main__":

    main()
