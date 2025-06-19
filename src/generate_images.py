import logging
import utils
from models.GAN import GAN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Starting image generation...")

    try:
        gan = GAN()
        utils.plot_images(gan.generate(n=16))

    except Exception as e:
        logger.error(f"An error occurred during image generation: {e}")
