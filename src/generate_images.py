import logging
import utils
import torch
from GAN import GAN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_images(generator, n=16, latent_dim=100, device="cpu"):
    """
    Generate fake images from the generator model.
    Returns a numpy array of generated images.
    """
    generator.eval()
    z = torch.randn(n, latent_dim).to(device)

    with torch.no_grad():
        fake_imgs = generator(z).cpu().numpy()

    generator.train()
    fake_imgs = (fake_imgs + 1) / 2  # Tanh → [0, 1]

    return fake_imgs


if __name__ == "__main__":
    logger.info("Starting image generation...")

    try:
        gan = GAN()
        utils.plot_images(gan.generate(n=16))

    except Exception as e:
        logger.error(f"An error occurred during image generation: {e}")
