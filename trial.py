import trimesh
import numpy as np

print("Loading flying_fish.stl...")
mesh = trimesh.load('flying_fish.stl')

print("\n--- INITIATING MESH REPAIR ---")
# 1. Weld overlapping/close vertices
mesh.merge_vertices()
# 2. Fix inverted surface normals (so CFD knows which way wind blows)
trimesh.repair.fix_inversion(mesh)
trimesh.repair.fix_winding(mesh)
# 3. Attempt to fill microscopic holes
trimesh.repair.fill_holes(mesh)

print("\n--- POST-REPAIR DIAGNOSTICS ---")
if mesh.is_watertight:
    print("Mesh Status: PERFECT. The mesh is now completely watertight!")
else:
    print("Mesh Status: STILL NON-MANIFOLD. The gaps are too large for simple auto-fill.")
    print("Note: We may need a robust 'Shrinkwrap' algorithm for the CFD pipeline.")

# Export the repaired mesh for OpenFOAM
mesh.export('flying_fish_REPAIRED.stl')
print("Saved clean geometry as 'flying_fish_REPAIRED.stl' for CFD.")

print("\n--- AI PIPELINE ---")
print("Sampling 10,000 points from the bionic surface...")
point_cloud, _ = trimesh.sample.sample_surface(mesh, 10000)
np.save('flying_fish_tensor.npy', point_cloud)
print("Saved as 'flying_fish_tensor.npy'. Phase 1 complete.")