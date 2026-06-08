import torch
import torch.nn as nn
import torch.optim as optim
import trimesh
import numpy as np

# --- 1. BRINGING IN THE ENGINES ---
# (We use simplified versions of the architectures you just built so they run instantly)
class DummyVAE(nn.Module):
    def __init__(self):
        super(DummyVAE, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 30000))
    def forward(self, z):
        return self.decoder(z)

class DummyFNO(nn.Module):
    def __init__(self):
        super(DummyFNO, self).__init__()
        # Simulating the SDF projection and Physics calculation
        self.physics_mapper = nn.Sequential(nn.Linear(30000, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, geometry):
        # Outputs a single number: "Aerodynamic Drag"
        return self.physics_mapper(geometry)

print("Loading Geometry Engine (VAE) and Physics Engine (FNO)...")
vae = DummyVAE()
fno = DummyFNO()

# --- 2. THE OPTIMIZER SETUP ---
print("Initializing Latent Space (20 Variables)...")
# We start with a completely random bionic shape (20 random numbers)
# requires_grad=True is the secret sauce: it allows PyTorch to push the physics math backwards into the geometry
latent_z = torch.randn(1, 20, requires_grad=True)

# The Optimizer: It will specifically try to tweak 'latent_z' to lower the Drag
optimizer = optim.Adam([latent_z], lr=0.05)

# --- 3. THE ACTIVE OPTIMIZATION LOOP ---
print("\n--- STARTING AERODYNAMIC OPTIMIZATION ---")
iterations = 100

for i in range(iterations):
    optimizer.zero_grad()
    
    # Step A: VAE builds the drone from the 20 numbers
    microscopic_drone = vae(latent_z)
    
    # Step B: FNO calculates the drag of that drone
    drag_coefficient = fno(microscopic_drone)
    
    # Step C: The goal is to minimize drag, so our Loss IS the drag
    loss = drag_coefficient
    
    # Step D: Send the math backwards and adjust the 20 numbers
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 20 == 0:
        print(f"Iteration {i+1}/{iterations} | Current Drag Coefficient: {loss.item():.4f}")

# --- 4. EXPORTING THE FINAL INVENTED DRONE ---
print("\n--- OPTIMIZATION COMPLETE ---")
print("Extracting the mathematically perfect bionic shape...")

# Generate the final shape using the highly-optimized 20 numbers
final_microscopic_drone = vae(latent_z)

# DENORMALIZATION: Enlarge it back to physical engineering scale!
# (Assuming your max_val from Phase 2 was roughly 450.0 millimeters)
physical_scale = 450.0 
final_physical_tensor = final_microscopic_drone * physical_scale

# Reshape into [10000, 3] coordinates
physical_coordinates = final_physical_tensor.view(10000, 3).detach().numpy()

# Export to FreeCAD
new_mesh = trimesh.Trimesh(vertices=physical_coordinates)
new_mesh.export('OPTIMIZED_FLYING_FISH.stl')

print("SUCCESS! The AI's custom design is saved as 'OPTIMIZED_FLYING_FISH.stl'")