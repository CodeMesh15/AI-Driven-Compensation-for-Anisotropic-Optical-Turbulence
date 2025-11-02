# src/simulation.py
# Core physics simulation functions.

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from src.zernike import get_zernike_basis

def create_vortex_beam(grid_shape, charge=1):
    """
    Creates a perfect optical vortex beam (Laguerre-Gaussian LG0p).
    
    Args:
        grid_shape (tuple): (height, width) of the grid.
        charge (int): The topological charge (OAM state).
        
    Returns:
        np.ndarray: A 2D complex array representing the beam.
    """
    y, x = np.indices(grid_shape)
    center_y, center_x = (grid_shape[0] - 1) / 2, (grid_shape[1] - 1) / 2
    
    rho = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Define a radial profile (e.g., Gaussian * r^|l|)
    w0 = grid_shape[1] / 6  # Beam waist
    radial_profile = (rho / w0)**np.abs(charge) * np.exp(-(rho**2) / w0**2)
    
    # Add the helical phase
    phase = charge * theta
    
    beam = radial_profile * np.exp(1j * phase)
    return beam / np.max(np.abs(beam))

def create_aniso_phase_screen(grid_shape, power_law, mu_x, mu_y, Cn2_equiv):
    """
    Generates a single phase screen using the FFT method based on a
    generalized anisotropic, non-Kolmogorov power spectrum.
    
    Args:
        grid_shape (tuple): (height, width) of the grid.
        power_law (float): The power law 'alpha' (Kolmogorov is 11/3 = 3.67).
        mu_x (float): Anisotropy factor in x.
        mu_y (float): Anisotropy factor in y.
        Cn2_equiv (float): Equivalent C_n^2, scales the strength.
        
    Returns:
        np.ndarray: A 2D array representing the phase screen.
    """
    # 1. Create frequency grid
    ky = fftshift(np.fft.fftfreq(grid_shape[0]))
    kx = fftshift(np.fft.fftfreq(grid_shape[1]))
    Kx, Ky = np.meshgrid(kx, ky)
    
    # 2. Define the anisotropic power spectrum
    # Phi_n(K) ~ (mu_x^2 * kx^2 + mu_y^2 * ky^2)^(-alpha/2)
    K_aniso_sq = (mu_x**2 * Kx**2) + (mu_y**2 * Ky**2)
    K_aniso_sq[K_aniso_sq == 0] = 1e-12 # Avoid division by zero at origin
    
    power_spectrum = K_aniso_sq ** (-power_law / 2.0)
    
    # 3. Create a filter from the spectrum
    # We scale by Cn2_equiv here. This is a simplification.
    # The 0.033 is from Kolmogorov theory, just for scaling.
    filter = np.sqrt(Cn2_equiv * 0.033 * power_spectrum)
    
    # 4. Create random noise in the frequency domain
    noise = (np.random.randn(*grid_shape) + 1j * np.random.randn(*grid_shape))
    
    # 5. Apply filter and inverse FFT
    fourier_screen = noise * filter
    phase_screen = np.real(ifft2(ifftshift(fourier_screen)))
    
    return phase_screen

def propagate_beam(beam, phase_screen):
    """
    Applies a phase screen to a beam (thin screen model).
    
    Args:
        beam (np.ndarray): 2D complex array of the beam.
        phase_screen (np.ndarray): 2D real array of phase shifts.
        
    Returns:
        np.ndarray: 2D complex array of the distorted beam.
    """
    return beam * np.exp(1j * phase_screen)

def get_zernike_coeffs(phase_screen, zernike_basis):
    """
    Decomposes a phase screen into Zernike coefficients.
    This is a projection, not a full fit.
    
    Args:
        phase_screen (np.ndarray): 2D phase screen.
        zernike_basis (list of np.ndarray): The basis maps.
        
    Returns:
        np.ndarray: 1D array of Zernike coefficients.
    """
    # Flatten the screen and the basis maps
    screen_flat = phase_screen.flatten()
    basis_flat = np.array([z.flatten() for z in zernike_basis]).T
    
    # Find the least-squares solution to: basis * coeffs = screen
    # This finds the coefficients 'c' that best reconstruct the screen.
    coeffs, _, _, _ = np.linalg.lstsq(basis_flat, screen_flat, rcond=None)
    
    return coeffs

def get_intensity_image(beam, bits=8):
    """
    Converts a complex beam into a normalized intensity image.
    
    Args:
        beam (np.ndarray): 2D complex array.
        bits (int): Bit depth for the output image (e.g., 8-bit for 0-255).
        
    Returns:
        np.ndarray: 2D array of integers (e.g., uint8).
    """
    intensity = np.abs(beam)**2
    # Normalize
    intensity -= intensity.min()
    if intensity.max() > 0:
        intensity /= intensity.max()
        
    # Scale to bit depth
    max_val = (2**bits) - 1
    image = (intensity * max_val).astype(np.uint8)
    
    return image
