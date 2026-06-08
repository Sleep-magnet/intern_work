import trimesh
import numpy as np
import os

# Create a dedicated folder for your dataset
output_dir = "drone_dataset"
os.makedirs(output_dir, exist_ok=True)

print("Loading baseline flying_fish.stl...")
baseline_mesh = trimesh.load('flying_fish.stl')

num_variations = 5 # Starting small for the test!

print(f"\n--- INITIATING AUTOMATED DATASET GENERATION ---")
print(f"Generating {num_variations} mutated bionic variations...\n")

for i in range(num_variations):
    # 1. Copy the baseline mesh so we don't permanently ruin the original
    mutated_mesh = baseline_mesh.copy()
    
    # 2. Randomize our Aerodynamic Parameters
    # Stretch or shrink the wingspan (X-axis) by +/- 15%
    span_stretch = np.random.uniform(0.85, 1.15) 
    # Thicken or thin the fuselage/airfoil (Z-axis) by +/- 10%
    thickness_stretch = np.random.uniform(0.90, 1.10) 
    # Stretch or shrink the length (Y-axis) by +/- 5%
    length_stretch = np.random.uniform(0.95, 1.05)
    
    # 3. Apply the mathematical transformation to warp the mesh
    transform_matrix = np.array([
        [span_stretch, 0, 0, 0],
        [0, length_stretch, 0, 0],
        [0, 0, thickness_stretch, 0],
        [0, 0, 0, 1]
    ])
    mutated_mesh.apply_transform(transform_matrix)
    
    # 4. Save the new, unique bionic drone
    filename = os.path.join(output_dir, f"bionic_drone_variant_{i+1}.stl")
    mutated_mesh.export(filename)
    print(f"Generated: {filename} | Span: {span_stretch:.2f}x | Thickness: {thickness_stretch:.2f}x")

print("\nDataset generation complete! Check the 'drone_dataset' folder.")