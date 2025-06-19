import os
import logging
import torch
from models.discriminator import Discriminator
from models.generator import Generator
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(model_name):
    """
    Load a model from the specified weights file.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        torch.nn.Module: The loaded model.
    """

    weights_path = f"src/models/weights/{model_name}.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(weights_path):
        logger.error(f"Model weights file {weights_path} does not exist.")
        raise FileNotFoundError(f"Model weights file {weights_path} does not exist.")

    logger.info(f"Loading model {model_name} from {weights_path}")

    if model_name == "generator":
        model = Generator().to(device)

    elif model_name == "discriminator":
        model = Discriminator().to(device)

    else:
        logger.error(f"Unknown model name: {model_name}")
        raise ValueError(f"Unknown model name: {model_name}")

    try:
        model.load_state_dict(torch.load(weights_path))
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        raise

    logger.info(f"Model {model_name} loaded successfully.")
    return model


def save_model(model, model_name):
    """
    Save the model weights to the specified file.

    Args:
        model (torch.nn.Module): The model to save.
        model_name (str): Name of the model to save.
    """
    weights_path = f"src/models/weights/{model_name}.pth"

    try:
        torch.save(model.state_dict(), weights_path)
        logger.info(f"Model {model_name} saved successfully at {weights_path}.")
    except Exception as e:
        logger.error(f"Error saving model {model_name}: {e}")
        raise


def plot_images(images, img_shape=(28, 28)):
    """
    Plot a row of images.
    Args:
        images (np.ndarray): Array of images to plot.
        img_shape (tuple): Shape to reshape each image for display.
    """
    n = len(images)
    _, axes = plt.subplots(1, n, figsize=(n, 1))
    for img, ax in zip(images, axes):
        img_2d = img.squeeze().reshape(*img_shape)
        ax.imshow(img_2d, cmap="gray")
        ax.axis("off")
    plt.show()


def load_dataset():
    try:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

        return loader
    except Exception as e:
        logger.error(f"Error loading the dataset : {e}")
