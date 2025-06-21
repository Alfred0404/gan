import matplotlib.pyplot as plt
import torch
import os
from models.generator import Generator
from models.discriminator import Discriminator
from tqdm import tqdm
import logging
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
        logger.info(
            f"device: {torch.cuda.get_device_name(0)}"
            if torch.cuda.is_available()
            else "No GPU/cuda available, using CPU"
        )
        self.gen_path = gen_path
        self.disc_path = disc_path

        self.generator = self.load_model("generator")
        self.discriminator = self.load_model("discriminator")

        self.disc_losses_list = []
        self.gen_losses_list = []

    def train(self, epochs=1, batch_size=64, lr=0.0002, betas=(0.5, 0.999)):
        # Placeholder: à remplacer par ton vrai code d'entraînement
        # train_loader doit être passé lors de l'appel à train()
        if self.loader is None:
            logger.warning("Aucun train_loader fourni, entraînement annulé.")
            return

        criterion = torch.nn.BCELoss()
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=betas
        )

        for epoch in range(epochs):
            for real_imgs, _ in tqdm(self.loader, desc=f"Epoch {epoch + 1}"):
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

    def load_model(self, model_name):
        weights_path = f"src/models/weights/{model_name}.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(weights_path):
            logger.error(f"Model weights file {weights_path} does not exist.")
            raise FileNotFoundError(
                f"Model weights file {weights_path} does not exist."
            )

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

    def save_model(self, model, model_name):
        weights_path = f"src/models/weights/{model_name}.pth"

        try:
            torch.save(model.state_dict(), weights_path)
            logger.info(f"Model {model_name} saved successfully at {weights_path}.")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            raise

    def generate(self, n=16):
        self.generator.eval()
        z = torch.randn(n, self.latent_dim).to(self.device)
        with torch.no_grad():
            fake_imgs = self.generator(z).cpu().numpy()
        self.generator.train()
        fake_imgs = (fake_imgs + 1) / 2  # Tanh → [0, 1]
        return fake_imgs

    def plot_loss(self):
        """
        Affiche l'évolution des pertes du discriminateur et du générateur.
        Args:
            d_losses (list): Liste des pertes du discriminateur.
            g_losses (list): Liste des pertes du générateur.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.disc_losses_list, label="Discriminator Loss")
        plt.plot(self.gen_losses_list, label="Generator Loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.ylim(-1, 3)  # Set y-axis limits
        plt.title("GAN Loss During Training")
        plt.legend()
        plt.savefig("src/train_logs/loss.png")
        plt.show()

    def save(self):
        self.save_model(self.generator, "generator")
        self.save_model(self.discriminator, "discriminator")

    def load_dataset(self):
        try:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )

            dataset = datasets.MNIST(
                root="./data", train=True, download=True, transform=transform
            )

            self.loader = DataLoader(
                dataset, batch_size=64, shuffle=True, num_workers=2
            )

        except Exception as e:
            logger.error(f"Error loading the dataset : {e}")
