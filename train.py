# Importing Libraries
import os

import torch
from vit_pytorch import ViT

import dataloader
import trainer
import utils


def main():
    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    utils.reproducibility()

    # Load the data
    train_loader, _ = dataloader.mnist.load_mnist()

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

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    train_data = trainer.train(
        model, train_loader, optimizer, criterion, device=device, epoch=2
    )

    # Save the model
    utils.save_model(model, target_dir="./experiments/models")

    # Plot the results
    utils.plot_results(train_data, save_target_dir="./experiments/results", name="train_results.jpg")


if __name__ == "__main__":

    main()
