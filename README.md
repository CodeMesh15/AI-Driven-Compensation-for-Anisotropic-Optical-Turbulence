# AI-Driven Compensation for Anisotropic Optical Turbulence

This repository contains the complete software framework for a B.Tech. Project investigating the use of Artificial Intelligence to compensate for optical distortions caused by anisotropic, non-Kolmogorov turbulence.

The project is divided into two main components:

Numerical Simulation: A Python-based simulation using the split-step Fourier method to generate distorted vortex beams. The simulation models anisotropic, non-Kolmogorov turbulence by generating "phase screens" based on a generalized power spectrum, as described in theoretical work (e.g., Zhai, 2020).

AI Compensation: A Compensation Neural Network (CNN), based on a ResNet architecture, that learns to predict the Zernike coefficients of a distortion just by looking at the intensity-only image of the distorted beam.

This repository provides the tools to generate a large synthetic dataset from the simulation and then train the AI model on that data.

Repository Structure

.
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── .gitignore              # Main gitignore
│
├── data/                   # For storing generated datasets
│   ├── .gitkeep
│   └── .gitignore          # Ignores the large dataset files
│
├── models/                 # For storing trained model weights
│   ├── .gitkeep
│   └── .gitignore          # Ignores the model .pth files
│
├── src/                    # Source code modules
│   ├── simulation.py       # Core physics simulation (phase screens, beam prop)
│   ├── zernike.py          # Functions for generating Zernike polynomial maps
│   ├── data_generation.py  # Script to generate the dataset
│   ├── model.py            # The CompensationNet (CNN) architecture
│   ├── dataset.py          # PyTorch Dataset class
│   ├── train.py            # Main training script
│   └── predict.py          # Script to run inference on a single image
│
└── notebooks/              # Jupyter notebooks for demonstration
    ├── 01_Simulation_Demo.ipynb       # Visually demonstrates the simulation
    ├── 02_Data_Generation.ipynb     # Runs data generation with visualization
    ├── 03_Model_Training.ipynb        # Loads data, trains model, shows loss
    └── 04_Model_Inference.ipynb       # Loads a trained model and tests it


Setup

Clone the repository:

git clone <repository-url>
cd <repository-name>


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Usage Workflow

You can run the entire project pipeline using either the Python scripts in src/ or the interactive Jupyter notebooks in notebooks/. The notebook approach is highly recommended for understanding the process.

Start Jupyter:

jupyter notebook


Then, open and run the notebooks in order:

1. notebooks/01_Simulation_Demo.ipynb

This notebook is the "physical schematic simulation" you wanted. It will visually walk you through:

Creating a perfect vortex beam.

Generating an anisotropic non-Kolmogorov phase screen.

Projecting that screen onto the Zernike basis to get the "ground truth" coefficients.

Propagating the beam through the screen.

Showing the final, distorted intensity image.

2. notebooks/02_Data_Generation.ipynb

This notebook runs the full data generation pipeline. It uses the functions from src/simulation.py to create a large dataset of paired images and labels, saving them to the data/ directory.

3. notebooks/03_Model_Training.ipynb

This notebook loads the dataset you just generated and feeds it into the CompensationNet model. It will run the training loop and show you a live-updating plot of the training loss, so you can see the AI learning.

4. notebooks/04_Model_Inference.ipynb

This notebook loads your saved, trained model and runs a prediction on a new, unseen distorted image. It will show you the distorted image, the AI's predicted Zernike coefficients, and the "ground truth" coefficients, so you can see how well it performed.
