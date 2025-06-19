import matplotlib.pyplot as plt
import utils
import torch
from tqdm import tqdm
import logging

logging


class GAN:
    def __init__(
        self,
        latent_dim=100,
        img_dim=784,
        device=None,
        gen_path="generator.pth",
        disc_path="discriminator.pth",
    ):
        self.latent_dim = latent_dim
        self.img_dim = img_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gen_path = gen_path
        self.disc_path = disc_path
        self._load_models()

        self.disc_losses_list = []
        self.gen_losses_list = []

    def _load_models(self):
        self.generator = utils.load_model("generator")
        self.discriminator = utils.load_model("discriminator")

    def train(
        self, epochs=1, batch_size=64, lr=0.0002, betas=(0.5, 0.999), loader=None
    ):
        # Placeholder: à remplacer par ton vrai code d'entraînement
        # train_loader doit être passé lors de l'appel à train()
        if loader is None:
            logger.warning("Aucun train_loader fourni, entraînement annulé.")
            return

        criterion = torch.nn.BCELoss()
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=betas
        )

        for epoch in range(epochs):
            for real_imgs, _ in tqdm(loader, desc=f"Epoch {epoch + 1}"):
                real_imgs = real_imgs.view(real_imgs.size(0), -1).to(self.device)
                batch_size = real_imgs.size(0)

                # Train Discriminator
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                outputs = self.discriminator(real_imgs)
                d_loss_real = criterion(outputs, real_labels)

                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_imgs = self.generator(z)
                outputs = self.discriminator(fake_imgs.detach())
                d_loss_fake = criterion(outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                self.disc_losses_list.append(d_loss.item())
                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()

                # Train Generator
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_imgs = self.generator(z)
                outputs = self.discriminator(fake_imgs)
                g_loss = criterion(outputs, real_labels)
                self.gen_losses_list.append(g_loss.item())

                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()

            logger.info(
                f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}"
            )

    def generate(self, n=16):
        self.generator.eval()
        z = torch.randn(n, self.latent_dim).to(self.device)
        with torch.no_grad():
            fake_imgs = self.generator(z).cpu().numpy()
        self.generator.train()
        fake_imgs = (fake_imgs + 1) / 2  # Tanh → [0, 1]
        return fake_imgs

    def plot_loss(self, d_losses, g_losses):
        """
        Affiche l'évolution des pertes du discriminateur et du générateur.
        Args:
            d_losses (list): Liste des pertes du discriminateur.
            g_losses (list): Liste des pertes du générateur.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label="Discriminator Loss")
        plt.plot(g_losses, label="Generator Loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("GAN Loss During Training")
        plt.legend()
        plt.savefig("src/train_data/loss.png")
        plt.show()

    def save(self):
        utils.save_model(self.generator, "generator")
        utils.save_model(self.discriminator, "discriminator")


if __name__ == "__main__":
    loader = utils.load_dataset()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    gan = GAN()
    gan.train(epochs=1, loader=loader)
    gan.plot_loss(gan.disc_losses_list, gan.gen_losses_list)
    gan.save()
    images = gan.generate(n=8)
    utils.plot_images(images)
