# Aerominds Bionic Drone Framework

**Project Owner:** Dhanush Thiyaree (Team Aerominds)
**Domain:** Artificial Intelligence, Deep Learning, Computational Fluid Dynamics (CFD), Aerospace Engineering.

## Overview
This repository contains the Aerominds Bionic Drone Framework, an end-to-end AI-driven pipeline for generating, simulating, and optimizing 3D aerodynamic structures using advanced deep learning techniques such as Variational Autoencoders (VAEs), Fourier Neural Operators (FNOs), and Clustered Long Short-Term Memory networks (C-LSTMs).

## The 6-Phase Technical Architecture

### Phase 1: Data Factory
A Python script using `trimesh` and affine transformation matrices systematically stretches and warps the Wing Span, Fuselage Length, and Airfoil Thickness of a baseline bionic drone. This allows for the automated generation of thousands of unique `.stl` datasets.

### Phase 2: Geometry Engine (VAE)
A Variational Autoencoder compresses the 30,000 3D drone coordinates into a 20-variable Latent Space. It utilizes KL Divergence to force a smooth probability distribution, ensuring generated shapes are continuous and physically viable.

### Phase 3: Physics Engine (FNO)
A Fourier Neural Operator bypasses traditional Navier-Stokes equations. By using Fast Fourier Transform (`torch.fft`), it shifts 3D grid data into the spectral domain, truncating high-frequency noise to instantly predict aerodynamic drag.

### Phase 4: The Optimizer
An active loop utilizing PyTorch gradient descent (`requires_grad=True`). The AI tweaks the 20 latent variables, the VAE decodes the shape, the FNO calculates the drag, and the math flows backward to minimize the Drag Coefficient. This ultimately outputs the `OPTIMIZED_FLYING_FISH.stl`.

### Phase 5: Flight Controller (C-LSTM)
A Clustered Long Short-Term Memory network acts as Edge AI. It ingests 431 MB/s of High-Velocity IIoT sensor data grouped into 4 physical clusters (Left Wing, Right Wing, Tail, Nose). Capable of analyzing spatiotemporal trends, such as an 80th-millisecond micro-burst hitting the left wing, to fire asymmetric wing-twist commands (<10ms) and neutralize aeroelastic flutter.


## Trial Data Specifications & Compute Constraints
*   **Deep Learning Trial Data:** Initial PyTorch Neural Network architecture tests (VAE and FNO) were trained on a micro-batch of 5 generated `.stl` files (30,000 coordinates per file) to prevent local hardware overheating.
*   **Machine Learning Validation Data:** The final validation model was trained on a simulated dataset of 5,000 geometries to prove statistical viability.
*   **Data Normalization:** All CAD coordinates were normalized to a `[-1.0, 1.0]` scale to prevent exploding gradients (`nan` errors) during neural network training, and later denormalized back to physical millimeters.

## Academic References (The Mathematical Defense)
The generated dataset and Deep Learning surrogate framework rely on the following peer-reviewed methodologies:
1.  **A systematic dataset generation technique applied to data-driven surrogate models for aerodynamics** (AIP Advances, 2024) - Validates the use of parametric free-form deformation to create synthetic aerodynamic datasets.
2.  **Fourier Neural Operator with Learned Deformations for PDEs on Irregular Geometries** (JMLR, Caltech, 2023) - Validates that FNOs can map irregular 3D meshes to instantly solve CFD in the frequency domain.
3.  **A deep learning framework for aerodynamic pressure prediction on general three-dimensional configurations** (Aerospace Science and Technology, 2023) - Validates the use of Latent Space generative AI to optimize entirely unseen 3D geometries.
