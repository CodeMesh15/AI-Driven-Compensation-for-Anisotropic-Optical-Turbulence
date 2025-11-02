# This module provides functions to create Zernike polynomial maps.
# This is a complex topic, so this file provides a simplified but
# functional implementation for the first 15 (Noll-indexed) modes.

import numpy as np

def zernike_cartesian(x, y, n, m):
    """
    Calculates the Zernike polynomial Z(n, m) at Cartesian coordinates (x, y).
    Note: n is radial order, m is azimuthal order.
    """
    if (n - m) % 2 != 0:
        return np.zeros_like(x)

    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    rho[rho > 1] = 0 # Only defined inside the unit circle

    # Radial polynomial R(n, |m|)
    R = 0
    if n < abs(m): # Ensure n is greater or equal to |m|
        return np.zeros_like(x)
        
    for k in range((n - abs(m)) // 2 + 1):
        R += ((-1)**k * np.math.factorial(n - k) * rho**(n - 2*k)) / \
             (np.math.factorial(k) * np.math.factorial((n + abs(m)) // 2 - k) * \
              np.math.factorial((n - abs(m)) // 2 - k))
    
    # Angular part
    if m >= 0:
        Z = R * np.cos(m * theta)
    else:
        Z = R * np.sin(abs(m) * theta)
        
    # Piston term (n=0, m=0)
    if n == 0:
        Z = np.ones_like(x)
    
    # Normalization factor (simplification, not full ANSI standard but good for basis)
    Z *= np.sqrt(n + 1)
    if m != 0:
        Z *= np.sqrt(2)

    Z[rho > 1] = 0
    return Z

def get_zernike_basis(n_modes, grid_shape):
    """
    Generates a basis of Zernike polynomials as a list of 2D maps.
    Uses Noll's single-index 'j' scheme.
    
    Args:
        n_modes (int): Number of modes to generate (e.g., 15).
        grid_shape (tuple): (height, width) of the 2D maps.
    
    Returns:
        list of np.ndarray: A list where each element is a 2D Zernike map.
    """
    print(f"Generating Zernike basis for {n_modes} modes...")
    x = np.linspace(-1, 1, grid_shape[1])
    y = np.linspace(-1, 1, grid_shape[0])
    xx, yy = np.meshgrid(x, y)
    
    # Noll's (n, m) indices for j=1 to 15
    # (j=1 is piston, which we often ignore, but we'll include it)
    noll_nm = [
        (0, 0),  # j=1 (Piston)
        (1, 1),  # j=2 (Tilt Y)
        (1, -1), # j=3 (Tilt X)
        (2, 0),  # j=4 (Defocus)
        (2, -2), # j=5 (Oblique Astig)
        (2, 2),  # j=6 (Vertical Astig)
        (3, -1), # j=7 (Vertical Coma)
        (3, 1),  # j=8 (Horizontal Coma)
        (3, -3), # j=9
        (3, 3),  # j=10
        (4, 0),  # j=11 (Spherical)
        (4, -2), # j=12
        (4, 2),  # j=13
        (4, -4), # j=14
        (4, 4)   # j=15
    ]
    
    if n_modes > len(noll_nm):
        raise ValueError(f"This function only supports up to {len(noll_nm)} modes.")

    basis = []
    for j in range(n_modes):
        n, m = noll_nm[j]
        Z_map = zernike_cartesian(xx, yy, n, m)
        
        # Orthonormalize within the unit circle
        mask = (xx**2 + yy**2) <= 1
        Z_map_masked = Z_map[mask]
        if Z_map_masked.size > 0:
            norm_factor = np.sqrt(np.sum(Z_map_masked**2))
            if norm_factor > 0:
                Z_map /= norm_factor
        
        basis.append(Z_map)
        
    print("Zernike basis generated.")
    return basis
