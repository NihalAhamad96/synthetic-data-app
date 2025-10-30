import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms, datasets
from sklearn.datasets import make_classification, make_regression
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os


# ------------------------------
# Environment Setup
# ------------------------------
def setup_environment():
    torch.manual_seed(42)
    np.random.seed(42)
    plt.style.use('default')
    print("✅ Environment setup complete")
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ Streamlit version: {st.__version__}")


# ------------------------------
# Generator Network
# ------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, data_type):
        super(Generator, self).__init__()
        self.data_type = data_type
        if data_type == "MNIST Images":
            self.main = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, 784),
                nn.Tanh()
            )
        else:
            self.main = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, 256),
                nn.ReLU(True),
                nn.Linear(256, 128),
                nn.ReLU(True),
                nn.Linear(128, output_dim)
            )

    def forward(self, x):
        return self.main(x)


# ------------------------------
# Discriminator Network
# ------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) > 2:  # flatten images
            x = x.view(x.size(0), -1)
        return self.main(x)


# ------------------------------
# Data Preparation
# ------------------------------
def prepare_data(data_type, batch_size):
    print(f"📊 Preparing {data_type} data...")

    if data_type == "MNIST Images":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader, 784, (1, 28, 28)

    elif data_type == "Simple Tabular":
        data = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 1000)
        dataset = TensorDataset(torch.FloatTensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader, 2, None

    elif data_type == "Classification Dataset":
        X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=1, random_state=42)
        data = np.column_stack([X, y])
        dataset = TensorDataset(torch.FloatTensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader, 6, None

    else:  # Regression Dataset
        X, y = make_regression(n_samples=1000, n_features=3, noise=0.1, random_state=42)
        data = np.column_stack([X, y.reshape(-1, 1)])
        dataset = TensorDataset(torch.FloatTensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader, 4, None


# ------------------------------
# GAN Training
# ------------------------------
def train_gan(data_loader, latent_dim, output_dim, data_type, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    generator = Generator(latent_dim, output_dim, data_type).to(device)
    discriminator = Discriminator(output_dim).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()

    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_plot = st.empty()

    g_losses, d_losses = [], []

    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            if isinstance(batch, (list, tuple)):
                real_data = batch[0]
            else:
                real_data = batch

            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # Train Discriminator
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

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = generator(z)
            output = discriminator(fake_data)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            optimizer_G.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f'Epoch [{epoch + 1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}')

        if epoch % 10 == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(g_losses, label='Generator Loss')
            ax.plot(d_losses, label='Discriminator Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.set_title('Training Losses')
            loss_plot.pyplot(fig)
            plt.close()

    return generator, discriminator, g_losses, d_losses


# ------------------------------
# Sample Generation
# ------------------------------
def generate_samples(generator, num_samples, latent_dim, data_type):
    device = next(generator.parameters()).device
    z = torch.randn(num_samples, latent_dim).to(device)
    with torch.no_grad():
        samples = generator(z).cpu().numpy()

    if data_type == "MNIST Images":
        samples = samples.reshape(-1, 28, 28)
        samples = (samples + 1) / 2
    return samples


# ------------------------------
# Visualization
# ------------------------------
def plot_comparison(real_data, synthetic_data, data_type):
    if data_type == "MNIST Images":
        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
        fig.suptitle('Real vs Synthetic MNIST Digits')
        for i in range(10):
            axes[0, i].imshow(real_data[i], cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(synthetic_data[i], cmap='gray')
            axes[1, i].axis('off')
        st.pyplot(fig)
    else:
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Real Data', 'Synthetic Data'])
        fig.add_trace(go.Scatter(x=real_data[:, 0], y=real_data[:, 1], mode='markers', name='Real'), row=1, col=1)
        fig.add_trace(go.Scatter(x=synthetic_data[:, 0], y=synthetic_data[:, 1], mode='markers', name='Synthetic'),
                      row=1, col=2)
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------
# Statistics
# ------------------------------
def show_statistics(real_data, synthetic_data, data_type):
    st.subheader("📈 Statistical Comparison")
    if data_type == "MNIST Images":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Real Mean", f"{np.mean(real_data):.4f}")
            st.metric("Synthetic Mean", f"{np.mean(synthetic_data):.4f}")
        with col2:
            st.metric("Real Std", f"{np.std(real_data):.4f}")
            st.metric("Synthetic Std", f"{np.std(synthetic_data):.4f}")
        with col3:
            st.metric("Mean Abs Diff", f"{np.mean(np.abs(real_data - synthetic_data[:len(real_data)])):.4f}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Real Data**")
            st.dataframe(pd.DataFrame(real_data).describe())
        with col2:
            st.write("**Synthetic Data**")
            st.dataframe(pd.DataFrame(synthetic_data).describe())


# ------------------------------
# Streamlit App
# ------------------------------
def main():
    setup_environment()
    st.set_page_config(page_title="Synthetic Data Playground", page_icon="🎭", layout="wide")

    st.title("🎭 Synthetic Data Generation Playground")
    st.markdown("""
    Explore how Generative Adversarial Networks (GANs) generate synthetic data interactively!
    """)

    st.sidebar.header("⚙️ Configuration")
    data_type = st.sidebar.selectbox("Select Data Type",
                                     ["MNIST Images", "Simple Tabular", "Classification Dataset", "Regression Dataset"])
    latent_dim = st.sidebar.slider("Latent Dimension", 10, 200, 100, 10)
    epochs = st.sidebar.slider("Epochs", 10, 300, 100, 10)
    batch_size = st.sidebar.slider("Batch Size", 16, 256, 64, 16)
    lr = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.002, 0.0001)

    if st.sidebar.button("🚀 Start Training", type="primary"):
        with st.spinner("Preparing data and training GAN..."):
            data_loader, output_dim, img_shape = prepare_data(data_type, batch_size)
            generator, discriminator, g_losses, d_losses = train_gan(
                data_loader, latent_dim, output_dim, data_type, epochs, lr
            )
        st.success("🎉 Training completed!")

        real_samples = next(iter(data_loader))[0].numpy()
        if data_type == "MNIST Images":
            real_samples = real_samples.reshape(-1, 28, 28)
            real_samples = (real_samples + 1) / 2
        synthetic_samples = generate_samples(generator, len(real_samples), latent_dim, data_type)

        st.header("📊 Results")
        plot_comparison(real_samples, synthetic_samples, data_type)
        show_statistics(real_samples, synthetic_samples, data_type)

        # Download Section
        st.header("🎨 Generate & Download")
        num_additional = st.slider("Number of samples to generate", 10, 1000, 100)
        if st.button("✨ Generate More Samples"):
            new_samples = generate_samples(generator, num_additional, latent_dim, data_type)
            df = pd.DataFrame(new_samples.reshape(num_additional, -1))
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", data=csv, file_name="synthetic_data.csv", mime="text/csv")
            st.dataframe(df.head(10))
            st.success(f"✅ Generated {num_additional} samples!")

    else:
        st.info("👈 Configure settings and click Start Training.")


if __name__ == "__main__":
    main()
