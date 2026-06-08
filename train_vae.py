import os
import torch
import torch.nn as nn
import torch.optim as optim
import trimesh
import numpy as np

# --- 1. THE VAE ARCHITECTURE ---
class BionicShapeVAE(nn.Module):
    def __init__(self, num_points=10000, latent_dim=20):
        super(BionicShapeVAE, self).__init__()
        self.input_dim = num_points * 3
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, self.input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# --- 2. THE CUSTOM LOSS FUNCTION ---
def vae_loss_function(reconstructed_x, original_x, mu, logvar):
    # Mean Squared Error: How close is the reconstructed shape to the original?
    MSE = nn.functional.mse_loss(reconstructed_x, original_x, reduction='sum')
    
    # KL Divergence: Forces the latent space into a smooth bell curve
    # Clamping logvar prevents it from shooting to infinity
    logvar = torch.clamp(logvar, min=-10, max=10) 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + KLD

# --- 3. DATA LOADER & NORMALIZATION ---
print("--- INITIATING DATA LOADER ---")
dataset_dir = "drone_dataset"
tensor_list = []

print("Extracting AI Tensors from mutated geometries...")
for file in os.listdir(dataset_dir):
    if file.endswith(".stl"):
        mesh = trimesh.load(os.path.join(dataset_dir, file))
        points, _ = trimesh.sample.sample_surface(mesh, 10000)
        tensor_list.append(points.flatten())

training_data = torch.tensor(np.array(tensor_list), dtype=torch.float32)

print("\nNormalizing CAD geometry to prevent exploding gradients...")
# THE FIX: Find the absolute largest coordinate and scale everything down to [-1, 1]
max_val = torch.max(torch.abs(training_data))
training_data = training_data / max_val

print(f"Training Batch Ready! Shape: {training_data.shape} (5 drones, 30,000 coordinates)")

# --- 4. THE TRAINING LOOP ---
print("\n--- INITIATING VAE TRAINING LOOP ---")
model = BionicShapeVAE()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    reconstructed_batch, mu, logvar = model(training_data)
    
    loss = vae_loss_function(reconstructed_batch, training_data, mu, logvar)
    
    loss.backward()
    
    # Gradient clipping: another safety net to prevent nan
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Total VAE Loss: {loss.item():.4f}")

print("\nPhase 2 Complete: VAE mathematically trained on dataset!")