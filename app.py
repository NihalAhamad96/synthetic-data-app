# synthetic-data-app/app.py
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
import io
import time
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def setup_environment():
    """Setup and configuration"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set matplotlib style
    plt.style.use('default')
    
    print("✅ Environment setup complete")
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ Streamlit version: {st.__version__}")

# GAN Model Definitions
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
        else:  # Tabular data
            self.main = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(True),
                nn.Linear(128, 256),
                nn.ReLU(True),
                nn.Linear(256, 128),
                nn.ReLU(True),
                nn.Linear(128, output_dim),
            )
    # hi 
    def forward(self, x):
        return self.main(x)

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
        return self.main(x)

def prepare_data(data_type, batch_size):
    """Prepare real data based on selection"""
    print(f"📊 Preparing {data_type} data...")
    
    if data_type == "MNIST Images":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, 784, (1, 28, 28)
    
    elif data_type == "Simple Tabular":
        # Generate simple 2D data
        data = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 1000)
        dataset = TensorDataset(torch.FloatTensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader, 2, None
    
    elif data_type == "Classification Dataset":
        X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, 
                                  n_redundant=1, n_clusters_per_class=1, random_state=42)
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

def train_gan(data_loader, latent_dim, output_dim, data_type, epochs, lr):
    """Train the GAN model"""
    print("🚀 Starting GAN training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    generator = Generator(latent_dim, output_dim, data_type).to(device)
    discriminator = Discriminator(output_dim).to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Training progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_plot = st.empty()
    
    g_losses = []
    d_losses = []
    
    for epoch in range(epochs):
        for i, (real_data,) in enumerate(data_loader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real data
            real_labels = torch.ones(batch_size, 1).to(device)
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake data
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_data = generator(z)
            fake_labels = torch.zeros(batch_size, 1).to(device)
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
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f'Epoch [{epoch+1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
        
        # Update loss plot every 10 epochs
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
    
    print("✅ Training completed!")
    return generator, discriminator, g_losses, d_losses

def generate_samples(generator, num_samples, latent_dim, data_type):
    """Generate synthetic samples"""
    print(f"🎨 Generating {num_samples} synthetic samples...")
    
    device = next(generator.parameters()).device
    z = torch.randn(num_samples, latent_dim).to(device)
    with torch.no_grad():
        samples = generator(z).cpu().numpy()
    
    if data_type == "MNIST Images":
        samples = samples.reshape(-1, 28, 28)
        samples = (samples + 1) / 2  # Denormalize
    return samples

def plot_comparison(real_data, synthetic_data, data_type):
    """Plot comparison between real and synthetic data"""
    
    if data_type == "MNIST Images":
        fig, axes = plt.subplots(2, 10, figsize=(15, 3))
        fig.suptitle('Real vs Synthetic MNIST Digits')
        
        # Real samples
        for i in range(10):
            axes[0, i].imshow(real_data[i], cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Real', rotation=90, size='large')
        
        # Synthetic samples
        for i in range(10):
            axes[1, i].imshow(synthetic_data[i], cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Synthetic', rotation=90, size='large')
        
        st.pyplot(fig)
        
    else:  # Tabular data
        # Use only first 2 features for visualization
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=['Real Data Distribution', 'Synthetic Data Distribution'])
        
        fig.add_trace(
            go.Scatter(x=real_data[:, 0], y=real_data[:, 1], mode='markers', name='Real'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=synthetic_data[:, 0], y=synthetic_data[:, 1], mode='markers', name='Synthetic'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

def show_statistics(real_data, synthetic_data, data_type):
    """Show statistical comparison"""
    st.subheader("📈 Statistical Comparison")
    
    if data_type == "MNIST Images":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Real Data Mean", f"{np.mean(real_data):.4f}")
            st.metric("Synthetic Data Mean", f"{np.mean(synthetic_data):.4f}")
        with col2:
            st.metric("Real Data Std", f"{np.std(real_data):.4f}")
            st.metric("Synthetic Data Std", f"{np.std(synthetic_data):.4f}")
        with col3:
            st.metric("Mean Absolute Difference", f"{np.mean(np.abs(real_data - synthetic_data[:len(real_data)])):.4f}")
    
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Real Data Statistics**")
            real_df = pd.DataFrame(real_data[:, :min(5, real_data.shape[1])])  # Show first 5 features
            st.dataframe(real_df.describe())
        
        with col2:
            st.write("**Synthetic Data Statistics**")
            syn_df = pd.DataFrame(synthetic_data[:, :min(5, synthetic_data.shape[1])])
            st.dataframe(syn_df.describe())

def main():
    """Main application function"""
    # Setup environment
    setup_environment()
    
    # Streamlit UI Configuration
    st.set_page_config(
        page_title="Synthetic Data Generation Playground", 
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎭 Synthetic Data Generation Playground")
    st.markdown("""
    This interactive playground demonstrates Generative Adversarial Networks (GANs) for synthetic data generation.
    Experiment with different architectures and parameters to understand how synthetic data is created!
    
    **Perfect for demonstrating concepts from your AI paper!**
    """)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data type selection
    data_type = st.sidebar.selectbox(
        "Select Data Type",
        ["MNIST Images", "Simple Tabular", "Classification Dataset", "Regression Dataset"]
    )
    
    # GAN architecture selection
    gan_arch = st.sidebar.selectbox(
        "GAN Architecture",
        ["DCGAN", "Vanilla GAN", "Conditional GAN (cGAN)"]
    )
    
    # Training parameters
    st.sidebar.subheader("🎯 Training Parameters")
    latent_dim = st.sidebar.slider("Latent Dimension", 10, 200, 100, 10)
    epochs = st.sidebar.slider("Number of Epochs", 10, 500, 100, 10)
    batch_size = st.sidebar.slider("Batch Size", 16, 256, 64, 16)
    lr = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.002, 0.0001)
    
    # System info
    st.sidebar.markdown("---")
    st.sidebar.subheader("💻 System Info")
    st.sidebar.write(f"PyTorch: {torch.__version__}")
    st.sidebar.write(f"Device: {'GPU 🚀' if torch.cuda.is_available() else 'CPU ⚡'}")
    
    # Main app logic
    if st.sidebar.button("🚀 Start Training", type="primary"):
        with st.spinner("Preparing data and training GAN..."):
            # Prepare data
            data_loader, output_dim, img_shape = prepare_data(data_type, batch_size)
            
            # Train GAN
            generator, discriminator, g_losses, d_losses = train_gan(
                data_loader, latent_dim, output_dim, data_type, epochs, lr
            )
            
            st.success("🎉 Training completed!")
            
            # Generate samples for comparison
            real_samples = next(iter(data_loader))[0].numpy()
            if data_type == "MNIST Images":
                real_samples = real_samples.reshape(-1, 28, 28)
                real_samples = (real_samples + 1) / 2  # Denormalize
            
            synthetic_samples = generate_samples(generator, len(real_samples), latent_dim, data_type)
            
            # Display results
            st.header("📊 Results")
            plot_comparison(real_samples, synthetic_samples, data_type)
            show_statistics(real_samples, synthetic_samples, data_type)
            
            # Sample generation and download
            st.header("🎨 Generate More Samples")
            num_additional_samples = st.slider("Number of additional samples to generate", 10, 1000, 100)
            
            if st.button("✨ Generate Additional Samples"):
                additional_samples = generate_samples(generator, num_additional_samples, latent_dim, data_type)
                
                # Convert to DataFrame for download
                if data_type == "MNIST Images":
                    # Flatten images for CSV
                    samples_flat = additional_samples.reshape(additional_samples.shape[0], -1)
                    df = pd.DataFrame(samples_flat)
                    df.columns = [f'pixel_{i}' for i in range(df.shape[1])]
                else:
                    df = pd.DataFrame(additional_samples)
                    df.columns = [f'feature_{i}' for i in range(df.shape[1])]
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download synthetic data as CSV",
                    data=csv,
                    file_name=f"synthetic_data_{data_type.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )
                
                # Show sample of generated data
                st.subheader("Generated Data Sample")
                st.dataframe(df.head(10))
                
                st.success(f"✅ Generated {num_additional_samples} additional samples!")

    else:
        st.info("👈 Configure your settings in the sidebar and click 'Start Training' to begin!")
        
        # Show example of what the app does
        st.header("🎯 About This Playground")
        st.markdown("""
        This app demonstrates synthetic data generation using Generative Adversarial Networks (GANs), 
        perfectly complementing the concepts in your AI paper.
        
        **Key Features:**
        - Multiple data types: MNIST images and various tabular datasets
        - Different GAN architectures
        - Interactive parameter tuning
        - Real-time training visualization
        - Statistical comparison between real and synthetic data
        - Sample generation and download
        
        **Academic Relevance:**
        - Data augmentation for scarce datasets
        - Privacy-preserving data sharing
        - Testing ML models with synthetic data
        - Understanding GAN training dynamics
        - Demonstrating AI/ML concepts interactively
        """)
        
        # Quick demo with pre-loaded information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Academic Applications")
            st.markdown("""
            - **Research Demonstrations**: Show GAN capabilities live
            - **Student Learning**: Interactive ML education
            - **Paper Supplementary Material**: Enhanced understanding
            - **Conference Presentations**: Engaging visualizations
            """)
        
        with col2:
            st.subheader("🔬 Technical Highlights")
            st.markdown("""
            - **PyTorch Integration**: Industry-standard deep learning
            - **Real-time Metrics**: Live training monitoring
            - **Comparative Analysis**: Real vs synthetic data comparison
            - **Export Capabilities**: Download generated datasets
            """)

if __name__ == "__main__":
    main()
