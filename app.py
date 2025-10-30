import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Generator Network
# ---------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)


# ---------------------------
# Discriminator Network
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # ✅ Flatten automatically if input is image (batch, 1, 28, 28)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.main(x)


# ---------------------------
# GAN Training Loop
# ---------------------------
def train_gan(data_loader, latent_dim, output_dim, data_type, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim, output_dim).to(device)
    discriminator = Discriminator(output_dim).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    g_losses, d_losses = [], []

    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            # ✅ Handle datasets with or without labels
            if isinstance(batch, (list, tuple)):
                real_data = batch[0]
            else:
                real_data = batch

            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # -----------------
            # Train Discriminator
            # -----------------
            optimizer_D.zero_grad()

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = generator(z)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = generator(z)
            output = discriminator(fake_data)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            optimizer_G.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    return generator, discriminator, g_losses, d_losses


# ---------------------------
# Synthetic Dataset Creation
# ---------------------------
def get_data_loader(data_type, batch_size=64):
    if data_type == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        output_dim = 28 * 28

    elif data_type == "Gaussian":
        data = np.random.randn(10000, 2).astype(np.float32)
        dataset = TensorDataset(torch.tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        output_dim = 2

    elif data_type == "Sine Wave":
        x = np.linspace(0, 2 * np.pi, 10000)
        y = np.sin(x) + 0.1 * np.random.randn(10000)
        data = np.stack((x, y), axis=1).astype(np.float32)
        dataset = TensorDataset(torch.tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        output_dim = 2

    else:
        raise ValueError("Unknown data type")

    return loader, output_dim


# ---------------------------
# Visualization Functions
# ---------------------------
def visualize_results(generator, latent_dim, data_type):
    generator.eval()
    z = torch.randn(1000, latent_dim)
    fake_data = generator(z).detach().cpu().numpy()

    if data_type == "MNIST":
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            axes[i].imshow(fake_data[i].reshape(28, 28), cmap="gray")
            axes[i].axis("off")
        st.pyplot(fig)

    else:
        plt.figure(figsize=(4, 4))
        plt.scatter(fake_data[:, 0], fake_data[:, 1], alpha=0.5)
        plt.title("Generated Synthetic Data")
        st.pyplot(plt)


# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.title("🎭 Synthetic Data Generation Playground")
    st.markdown(
        "This interactive playground demonstrates Generative Adversarial Networks (GANs) for synthetic data generation. "
        "Experiment with different architectures and parameters to understand how synthetic data is created!"
    )

    st.sidebar.header("⚙️ Configuration")

    data_type = st.sidebar.selectbox("Choose dataset type:", ["MNIST", "Gaussian", "Sine Wave"])
    latent_dim = st.sidebar.slider("Latent dimension (z)", 2, 100, 20)
    epochs = st.sidebar.slider("Training epochs", 1, 50, 5)
    lr = st.sidebar.number_input("Learning rate", 0.0001, 0.01, 0.001, step=0.0001)
    batch_size = st.sidebar.slider("Batch size", 16, 256, 64)

    if st.button("🚀 Train GAN"):
        with st.spinner("Training GAN... Please wait ⏳"):
            data_loader, output_dim = get_data_loader(data_type, batch_size)
            generator, discriminator, g_losses, d_losses = train_gan(
                data_loader, latent_dim, output_dim, data_type, epochs, lr
            )

        st.success("✅ Training complete!")
        st.subheader("📉 Loss Curves")
        plt.figure()
        plt.plot(g_losses, label="Generator Loss")
        plt.plot(d_losses, label="Discriminator Loss")
        plt.legend()
        st.pyplot(plt)

        st.subheader("🎨 Generated Samples")
        visualize_results(generator, latent_dim, data_type)


if __name__ == "__main__":
    main()
