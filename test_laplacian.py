# Description: Test the Laplacian convolutional layer by comparing it to a matrix-vector multiplication of the Laplacian with Neumann boundary conditions.
# The Laplacian matrix is created using a 5-point stencil with Neumann boundary conditions.
# The Laplacian convolutional layer is defined with a 3x3 kernel with padding to handle the boundary conditions.

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import diags

def create_laplacian(N):
    """Create Laplacian matrix with Neumann boundary conditions."""
    main_diag = -4 * np.ones(N * N)
    off_diag1 = np.ones(N * N - 1)
    off_diagN = np.ones(N * N - N)

    # Avoid wrap-around between rows
    for i in range(1, N):
        off_diag1[i * N - 1] = 0  # Prevents wrap-around effect

    # Adjust for Neumann boundary conditions
    for i in range(N):
        main_diag[i] += 1                  # Top row
        main_diag[N*(N-1) + i] += 1        # Bottom row
        main_diag[i*N] += 1                # Left column
        main_diag[i*N + (N-1)] += 1        # Right column

    diagonals = [main_diag, off_diag1, off_diag1, off_diagN, off_diagN]
    offsets = [0, -1, 1, -N, N]
    L = diags(diagonals, offsets, shape=(N**2, N**2), format="csr")

    return L.toarray()  # Convert to dense array for easier handling

class LaplacianConv(nn.Module):
    def __init__(self, h=1.0):
        super(LaplacianConv, self).__init__()
        self.h = h
        # Define the Laplacian kernel
        kernel = torch.tensor([[[[0., 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]]]]) / (self.h ** 2)
        # Initialize the convolutional layer
        self.lap = nn.Conv2d(1, 1, 3, padding=1, bias=False, padding_mode='replicate')
        self.lap.weight = nn.Parameter(kernel)
        self.lap.weight.requires_grad = False

    def forward(self, A):
        return self.lap(A)

def main():
    # Define grid size
    N = 3

    # Create a sample input grid (1 batch, 1 channel, 3x3 grid)
    A = torch.tensor([[[[1., 2., 3.],
                        [4., 5., 6.],
                        [7., 8., 9.]]]], dtype=torch.float32)

    # Create the Laplacian matrix
    L_matrix = create_laplacian(N)
    print("Laplacian Matrix:\n", L_matrix)

    # Flatten the input grid to a vector
    A_vector = A.view(N*N).numpy()
    print("\nInput Vector:\n", A_vector)

    # Compute the Laplacian using matrix-vector multiplication
    lap_A_matrix = L_matrix @ A_vector
    print("\nLaplacian via Matrix-Vector Multiplication:\n", lap_A_matrix.reshape(N, N))

    # Initialize the convolutional Laplacian operator
    laplacian_conv = LaplacianConv(h=1.0)

    # Apply the Laplacian using convolution
    lap_A_conv = laplacian_conv(A)
    lap_A_conv_np = lap_A_conv.detach().numpy().reshape(N, N)
    print("\nLaplacian via Convolution:\n", lap_A_conv_np)

    # Compare the two results
    difference = np.abs(lap_A_matrix.reshape(N, N) - lap_A_conv_np)
    print("\nDifference between Matrix and Convolution Results:\n", difference)

    # Verify if the results are identical within a tolerance
    if np.allclose(lap_A_matrix.reshape(N, N), lap_A_conv_np, atol=1e-6):
        print("\nSuccess: Both methods yield consistent results.")
    else:
        print("\nError: The results differ between the two methods.")

if __name__ == "__main__":
    main()