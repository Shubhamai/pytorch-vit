# Importing Libraries
import torch

from dataloader import load_mnist
from model.vit import ViT
from trainer import train
from utils import plot_results, reproducibility, save_model


def main():
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    reproducibility()

    # Load the data
    train_loader, _ = load_mnist(batch_size=16)

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

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    train_data = train(
        model, train_loader, optimizer, criterion, device=device, epoch=2
    )

    # Save the model
    save_model(model, target_dir="./experiments/models")

    # Plot the results
    plot_results(
        train_data, save_target_dir="./experiments/results", name="train_results.jpg"
    )


if __name__ == "__main__":
    main()
