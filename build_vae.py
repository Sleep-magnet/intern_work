import torch
import torch.nn as nn
import numpy as np

# --- 1. THE NEURAL NETWORK ARCHITECTURE ---
class BionicShapeVAE(nn.Module):
    def __init__(self, num_points=10000, latent_dim=20):
        super(BionicShapeVAE, self).__init__()
        
        # We flatten the [10000, 3] tensor into a 30,000 length array
        self.input_dim = num_points * 3
        
        # ENCODER: Compresses 30,000 points down to 20 variables
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        # The Latent Space (Mean and Variance for the 20 parameters)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # DECODER: Expands the 20 variables back into a 30,000 point drone shape
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_shape = self.decode(z)
        return reconstructed_shape, mu, logvar

# --- 2. LOADING YOUR DATA ---
print("Loading the Flying Fish Tensor...")
# Load the [10000, 3] array
point_cloud = np.load('flying_fish_tensor.npy') 

# Convert it to a PyTorch tensor and flatten it to [1, 30000]
drone_tensor = torch.tensor(point_cloud, dtype=torch.float32).view(1, -1)
print(f"Tensor loaded successfully! Shape: {drone_tensor.shape}")

# --- 3. INITIALIZING THE AI ---
print("\nInitializing the Generative Autoencoder...")
# Set the Latent Space to 20 parameters
model = BionicShapeVAE(num_points=10000, latent_dim=20) 

# Pass the drone through the network (A forward pass)
reconstructed_drone, mu, logvar = model(drone_tensor)

print("\n--- AI ARCHITECTURE TEST SUCCESS ---")
print(f"1. Original Input Shape: {drone_tensor.shape}")
print(f"2. Compressed Latent Space (The 20 parameters): {mu.shape}")
print(f"3. Reconstructed Output Shape: {reconstructed_drone.shape}")
print("\nThe mathematical bottleneck is working perfectly.")