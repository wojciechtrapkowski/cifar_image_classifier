import os
from PIL import Image

import torch
import torchvision.transforms as transforms

from convolutional_neural_network import ConvolutionalNeuralNetwork
from dataset_loader import DatasetLoader
from consts import *


def get_model_save_path():
    """Generates a unique save path for the model if the file already exists."""
    counter = 1
    model_save_path = MODEL_SAVE_PATH

    while os.path.exists(model_save_path):
        model_save_path = f"{MODEL_SAVE_PATH[:-4]}_{counter}.pth"
        counter += 1

    print(f"Model will be saved with name {model_save_path}")
    return model_save_path


def test_on_own_image(path, classes, model, device):
    """Tests the model on a single image from the given path."""
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # Resize to CIFAR-10 image size
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),  # CIFAR-10 normalization
        ]
    )

    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    print(f"File: {path} - Predicted Label: {classes[predicted.item()]}")


def test_images_in_folder(folder_path, classes, model, device):
    """Tests the model on all images within the given folder."""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            test_on_own_image(
                os.path.join(folder_path, filename), classes, model, device
            )


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and data loader
    loader = DatasetLoader()
    model = ConvolutionalNeuralNetwork(loader, device).to(device)

    if not TRAIN_MODEL:
        try:
            model.load_state_dict(torch.load(MODEL_LOAD_PATH))
            model.eval()
            print("Model loaded.")
        except FileNotFoundError:
            print("Model not found, creating new model.")
    else:
        print("Training new model.")
        model.train_model(NUM_EPOCHS)

    if TEST_MODEL:
        model.test_model()

        if SAVE_MODEL:
            save_path = get_model_save_path()
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # Test model on custom images
    test_images_in_folder("tests/", loader.classes, model, device)


if __name__ == "__main__":
    main()
