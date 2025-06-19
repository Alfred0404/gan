import logging
from models.GAN import GAN


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    gan = GAN()
    gan.load_dataset()
    gan.train(epochs=10)
    gan.save()
    gan.plot_loss()
    images = gan.generate(n=8)
