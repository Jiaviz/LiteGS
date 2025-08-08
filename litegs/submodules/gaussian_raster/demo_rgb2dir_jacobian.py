#!/usr/bin/env python3

import torch
import litegs_fused
import numpy as np

def demonstrate_rgb2dir_jacobian():
    """
    Demonstrate the usage of the RGB to direction Jacobian function.
    
    This function computes the Jacobian matrix that describes how RGB colors
    change with respect to viewing direction changes in spherical harmonics representation.
    """
    
    print("RGB to Direction Jacobian Demonstration")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parameters
    batch_size = 1
    num_points = 5  # Small example for demonstration
    max_sh_degree = 2  # SH degree up to 2
    
    print(f"\nSetup:")
    print(f"- Batch size: {batch_size}")
    print(f"- Number of points: {num_points}")
    print(f"- SH degree: {max_sh_degree}")
    
    # Create spherical harmonics coefficients
    # SH_base: DC component (degree 0) - shape: [1, 3, num_points]
    sh_base = torch.tensor([
        [[0.5, 0.3, 0.7, 0.2, 0.8],  # R channel
         [0.4, 0.6, 0.2, 0.9, 0.1],  # G channel  
         [0.8, 0.1, 0.5, 0.4, 0.6]]  # B channel
    ], device=device, dtype=torch.float32)
    
    # SH_rest: Higher order components - shape: [num_rest_coeffs, 3, num_points]
    # For degree 2: (2+1)^2 - 1 = 8 rest coefficients
    num_rest_coeffs = (max_sh_degree + 1) ** 2 - 1  # 8 for degree 2
    sh_rest = torch.randn(num_rest_coeffs, 3, num_points, device=device) * 0.1
    
    # Create viewing directions - shape: [batch, 3, num_points]
    # Example directions (normalized)
    dirs = torch.tensor([
        [[1.0, 0.0, 0.0, 0.707, -0.707],   # X components
         [0.0, 1.0, 0.0, 0.707, 0.0],      # Y components
         [0.0, 0.0, 1.0, 0.0, 0.707]]      # Z components
    ], device=device, dtype=torch.float32)
    
    print(f"\nTensor shapes:")
    print(f"- SH base: {sh_base.shape}")
    print(f"- SH rest: {sh_rest.shape}")  
    print(f"- Directions: {dirs.shape}")
    
    # Compute Jacobian
    print(f"\nComputing Jacobian...")
    jacobian = litegs_fused.rgb2dir_jacobian(max_sh_degree, dirs, sh_base, sh_rest)
    
    print(f"Jacobian shape: {jacobian.shape}")
    print(f"Expected shape: [batch_size, 3_rgb, 3_dir, num_points]")
    
    # Display results for first point
    point_idx = 0
    J = jacobian[0, :, :, point_idx]  # Shape: [3, 3]
    
    print(f"\nJacobian matrix for point {point_idx}:")
    print(f"Direction: [{dirs[0, 0, point_idx]:.3f}, {dirs[0, 1, point_idx]:.3f}, {dirs[0, 2, point_idx]:.3f}]")
    print("Jacobian J[rgb, dir]:")
    print("       dx      dy      dz")
    for i, color in enumerate(['dR', 'dG', 'dB']):
        row = [f"{J[i, j].item():7.3f}" for j in range(3)]
        print(f"{color}: {' '.join(row)}")
    
    print(f"\nInterpretation:")
    print("- Each row shows how one RGB channel changes with direction")  
    print("- Each column shows how RGB changes with one direction component")
    print("- J[i,j] = ∂RGB[i]/∂dir[j]")
    
    # Show statistics
    print(f"\nJacobian statistics:")
    print(f"- Min value: {jacobian.min().item():.6f}")
    print(f"- Max value: {jacobian.max().item():.6f}")
    print(f"- Mean absolute value: {jacobian.abs().mean().item():.6f}")
    print(f"- Standard deviation: {jacobian.std().item():.6f}")
    
    # Demonstrate usage for different SH degrees
    print(f"\nTesting different SH degrees:")
    for degree in [0, 1, 2, 3]:
        try:
            if degree == 0:
                sh_rest_deg = torch.zeros(0, 3, num_points, device=device)
            else:
                rest_coeffs = (degree + 1) ** 2 - 1
                sh_rest_deg = torch.randn(rest_coeffs, 3, num_points, device=device) * 0.1
            
            jac = litegs_fused.rgb2dir_jacobian(degree, dirs, sh_base, sh_rest_deg)
            print(f"  Degree {degree}: ✓ (shape: {jac.shape})")
        except Exception as e:
            print(f"  Degree {degree}: ✗ ({e})")
    
    return jacobian

if __name__ == "__main__":
    jacobian_result = demonstrate_rgb2dir_jacobian()
    print(f"\n🎉 Demonstration completed successfully!")
    print(f"The RGB to direction Jacobian function is ready for use!")
