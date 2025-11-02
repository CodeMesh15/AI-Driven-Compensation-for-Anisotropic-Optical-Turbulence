# src/data_generation.py
# This script runs the simulation loop to generate a large dataset.

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import src.simulation as sim
import src.zernike as zern

# --- Configuration ---
GRID_SHAPE = (128, 128)      # Image size
VORTEX_CHARGE = 1           # Topological charge of the beam
N_MODES = 15                # Number of Zernike modes to calculate
N_SAMPLES = 1000            # Number of images to generate (Set to 10-100 for a quick test)
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
LABELS_FILE = os.path.join(DATA_DIR, "labels.csv")

# --- Turbulence Parameters (will be randomized) ---
POWER_LAW_RANGE = (3.0, 4.0)      # Alpha (Kolmogorov is 3.67)
ANISOTROPY_RANGE = (1.0, 5.0)     # Mu (1.0 is isotropic)
STRENGTH_RANGE = (0.5, 5.0)       # Equivalent Cn2 strength

def generate_dataset():
    """Generates and saves the dataset."""
    
    # 1. Create output directories
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 2. Get prerequisites
    print("Creating initial beam and Zernike basis...")
    initial_beam = sim.create_vortex_beam(GRID_SHAPE, VORTEX_CHARGE)
    # We calculate coefficients for modes j=1 to N_MODES
    zernike_basis = zern.get_zernike_basis(N_MODES, GRID_SHAPE)
    
    # 3. Setup labels file
    # We will save coeffs for j=2 through N_MODES (e.g., 14 coefficients)
    label_columns = ["filename"] + [f"z{j+1}" for j in range(1, N_MODES)]
    labels_data = []

    print(f"Generating {N_SAMPLES} data samples...")
    # 4. Run simulation loop
    for i in tqdm(range(N_SAMPLES)):
        # Randomize parameters for each sample
        power_law = np.random.uniform(*POWER_LAW_RANGE)
        strength = np.random.uniform(*STRENGTH_RANGE)
        
        # Randomize anisotropy
        if np.random.rand() > 0.5:
            # Stretch in X
            mu_x = np.random.uniform(*ANISOTROPY_RANGE)
            mu_y = 1.0
        else:
            # Stretch in Y
            mu_x = 1.0
            mu_y = np.random.uniform(*ANISOTROPY_RANGE)
            
        # 5. Run simulation
        phase_screen = sim.create_aniso_phase_screen(
            GRID_SHAPE, power_law, mu_x, mu_y, strength
        )
        distorted_beam = sim.propagate_beam(initial_beam, phase_screen)
        
        # 6. Get outputs
        intensity_image = sim.get_intensity_image(distorted_beam)
        all_coeffs = sim.get_zernike_coeffs(phase_screen, zernike_basis)
        
        # We only care about coefficients 2 through N_MODES (index 1 to N_MODES-1)
        coeffs_to_save = all_coeffs[1:N_MODES]
        
        # 7. Save data
        filename = f"beam_{i:05d}.png"
        filepath = os.path.join(IMAGE_DIR, filename)
        
        # Save image
        img = Image.fromarray(intensity_image)
        img.save(filepath)
        
        # Add label row
        labels_data.append([filename] + list(coeffs_to_save))

    # 8. Save labels CSV
    print(f"Saving labels to {LABELS_FILE}...")
    labels_df = pd.DataFrame(labels_data, columns=label_columns)
    labels_df.to_csv(LABELS_FILE, index=False)
    print("Data generation complete.")

if __name__ == "__main__":
    generate_dataset()
