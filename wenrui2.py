
# this is for nonlinear pde with one solution
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def create_rhs(N, h, s):
    """Create the right-hand side of the linear system."""
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    rhs = -s * np.sin(np.pi * X) * np.sin(np.pi * Y)
    return rhs.flatten()

def create_laplacian_matrix_with_boundary_conditions(N, h):
    """
    Create a sparse matrix for the discrete Laplacian operator,
    incorporating the zero Dirichlet boundary conditions.
    """
    # Main diagonal (interior points)
    main_diag = 4.0 * np.ones(N**2)

    # Off diagonals for adjacent points in the grid
    off_diag1 = -np.ones(N**2 - 1)
    off_diag2 = -np.ones(N**2 - N)

    # Apply boundary conditions by removing connections to boundary points
    for i in range(1, N):
        off_diag1[i * N - 1] = 0

    diagonals = [main_diag, off_diag1, off_diag1, off_diag2, off_diag2]
    offsets = [0, -1, 1, -N, N]
    L = diags(diagonals, offsets, shape=(N**2, N**2), format="csr")

    return L / h**2

def newton_method_with_corrected_laplacian(N, s, u, tol=1e-8, max_iter=10):
    h = 1.0 / (N + 1)
    
    # Create the Laplacian matrix with boundary conditions
    A = create_laplacian_matrix_with_boundary_conditions(N, h)

    # Flatten u for easier computation
    u_flat = u.flatten()
    fixed_rhs = create_rhs(N, h, s)
    u_list, delta_u_list = [], []
    converged = False
    for i in range(max_iter):
        # Update the right-hand side with nonlinear term
        rhs = fixed_rhs - u_flat**2 - A.dot(u_flat)

        # Record the copy of u_flat for the iteration 
        u_list.append(u_flat.copy())

        # Update the Jacobian matrix for the current iteration
        diagonal = 2 * u_flat
        J = A + diags([diagonal], [0], shape=(N**2, N**2), format="csr")

        # Solve the linear system
        delta_u = spsolve(J, rhs)

        # Record the copy of update for the iteration
        delta_u_list.append(delta_u.copy())

        # Update the solution
        u_flat += delta_u

        # Reshape to the original grid
        u = u_flat.reshape((N, N))

        # Check for convergence
        print(np.linalg.norm(fixed_rhs - u_flat**2 - A.dot(u_flat), np.inf))
        if np.linalg.norm(delta_u, np.inf) < tol:
            print("Converged at iteration", i)
            converged = True
            break
    if not converged:
        print("Failed to converge after", max_iter, "iterations")
        return u, [], []
    else:
        return u, u_list[:3], delta_u_list[:3]

def generate_initial_guess_sine_series(s):
    # Create meshgrid for x and y values
    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    X, Y = np.meshgrid(x, y)

    # Initialize u as zeros
    u = np.zeros((s, s))
    K1, K2 = np.meshgrid(np.arange(s), np.arange(s), indexing='ij')
    tau = 0.5
    alpha = 4
    coef =  np.pi**2 * (np.pi**2 * (K1**2 + K2**2) + tau**2)**(-alpha/2)
    # Random coefficients xi_ij
    xi_ij = np.random.randn(s, s)
    xi_ij = xi_ij * coef
    # Summation over sine series
    for i in range(1, s):
        for j in range(1, s):
            u += 100 * xi_ij[i, j] * np.sin(np.pi * X * i) * np.sin(np.pi * Y * j)
    return u

# Solve the PDE with the revised Newton's method
N = 63  # grid size
s = 100   # parameter in the PDE
u = np.zeros((N, N))  # initial guess

# u_list, delta_u_list = torch.load('newton_method_iterations.pt')
# u_list.shape, delta_u_list.shape, u_list.dtype, delta_u_list.dtype
# solution, _, _ = newton_method_with_corrected_laplacian(N, s, u=u_list[140].numpy())
# u_int = np.zeros_like(solution)    # this is to generate the forth solution 
# u_int[31:, :] = solution[::2, :]
# solution, _, _ = newton_method_with_corrected_laplacian(N, s, u=u_int)


u_list, delta_u_list = [], []
for _ in range(2000):
    u_2_init = generate_initial_guess_sine_series(N)
    u, new_u_list, new_delta_u_list = newton_method_with_corrected_laplacian(N, s=s, u=u_2_init, max_iter=20, tol=1e-6)
    u_list += new_u_list
    delta_u_list += new_delta_u_list


u_list = torch.tensor(u_list, dtype=torch.float64)
delta_u_list = torch.tensor(delta_u_list, dtype=torch.float64)
# reshape the data
u_list = u_list.reshape(-1, 1, N, N)
delta_u_list = delta_u_list.reshape(-1, 1, N, N)
# save the data in a single file
torch.save((u_list, delta_u_list, u), 'newton_method_single_solution_2.pt')




