import numpy as np
from scipy.sparse import diags, bmat
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def create_rhs(N, h, DA, DS, mu, rho):
    """Create the right-hand side of the linear system."""
    size = N**2
    return np.zeros(2 * size)  # RHS is zero for the steady-state solution

def generate_initial_guess_sine_series(s):
    # generate the initial guess for A and S with Neumann boundary conditions
    # Create meshgrid for x and y values
    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    X, Y = np.meshgrid(x, y)

    # Initialize u as zeros
    u = np.zeros((s, s))
    K1, K2 = np.meshgrid(np.arange(s), np.arange(s), indexing='ij')
    tau = 0.5
    alpha = 3
    coef =  np.pi**2 * (np.pi**2 * (K1**2 + K2**2) + tau**2)**(-alpha/2)
    # Random coefficients xi_ij
    xi_ij = np.random.randn(s, s)
    xi_ij = xi_ij * coef
    # Summation over  series
    for i in range(1, s):
        for j in range(1, s):
            u += 100 * xi_ij[i, j] * np.cos(np.pi * X * i) * np.cos(np.pi * Y * j)
    return u



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

    return L

def create_jacobian(A_flat, S_flat, L_A, L_S, mu, rho, N):
    # Creating the sub-matrices of the Jacobian matrix
    J_AA = L_A + diags((-2 * S_flat * A_flat + (mu + rho)), offsets=0, shape=(N**2, N**2))
    J_AS = diags(-A_flat**2, offsets=0, shape=(N**2, N**2))
    J_SA = diags(2 * S_flat * A_flat, offsets=0, shape=(N**2, N**2))
    J_SS = L_S + diags(A_flat**2 + rho, offsets=0, shape=(N**2, N**2))

    # Combine the sub-matrices into a full Jacobian matrix
    J = bmat([[J_AA, J_AS], [J_SA, J_SS]], format='csr')
    return J

# Example usage within a Newton's method routine
def newton_method_for_gray_scott(A, S, DA, DS, mu, rho, N, h, tol=1e-4, max_iter=8):
    # Create Laplacian operators
    L = create_laplacian(N)
    L_A = -L * (DA / h**2)
    L_S = -L * (DS / h**2)

    AS_list, deltaAS_list = [], []
    A_flat = A.flatten()
    S_flat = S.flatten()

    for iteration in range(max_iter):
        AS_list.append(np.stack([A_flat.copy(), S_flat.copy()]))
        converged = False
        # Compute F vectors for A and S
        F_A = L_A.dot(A_flat) - S_flat * A_flat**2 + (mu + rho) * A_flat
        F_S = L_S.dot(S_flat) + S_flat * A_flat**2 - rho * (1 - S_flat)
        F = np.concatenate([F_A, F_S])
        
        # Compute the Jacobian matrix
        J = create_jacobian(A_flat, S_flat, L_A, L_S, mu, rho, N)

        # Solve the linear system
        delta = spsolve(J, -F)
        deltaAS_list.append(delta.copy().reshape(2, N**2))

        # Update A and S
        A_flat += delta[:N**2]
        S_flat += delta[N**2:]
        A = A_flat.reshape(N, N)
        S = S_flat.reshape(N, N)
        
        # Convergence check
        if np.linalg.norm(delta, np.inf) < tol:
            print("Converged after", iteration + 1, "iterations.")
            converged = True
            break
    if not converged:
        print("Failed to converge after", max_iter, "iterations")
        return A, S, [], []
    else: 
        return A, S, AS_list[-9::3], deltaAS_list[-9::3]

# # Example usage
# N = 50
# h = 1.0 / (N - 1)
# DA = 0.1
# DS = 0.05
# mu = 0.04
# rho = 0.06
# A = np.random.rand(N, N)  # Example initial matrix for A
# S = np.random.rand(N, N)  # Example initial matrix for S

# A, S, AS_list, deltaAS_list = newton_method_for_gray_scott(A, S, DA, DS, mu, rho, N, h)

# # Plot results
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

# ax1.contourf(X, Y, AS_list[-1][0].reshape(N, N), levels=50)
# ax1.set_title('Concentration of A')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# plt.colorbar(ax1.contourf(X, Y, A, levels=50))

# ax2.contourf(X, Y, deltaAS_list[-1][1].reshape(N, N), levels=50)
# ax2.set_title('Concentration of S')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# plt.colorbar(ax2.contourf(X, Y, S, levels=50))
# plt.savefig('GrayScott.png')



def solve_for_S(A, DS, rho, N, tol=1e-10, max_iter=20):
    """Solve for S using Newton's method given A."""
    h = 1.0 / (N - 1)
    L = create_laplacian(N, )
    L = -L * (DS / h**2)
    S = np.ones((N, N)) * 0.5  # Initial guess for S
    
    A_flat = A.flatten()
    for iteration in range(max_iter):
        S_flat = S.flatten()
        # Function F(S)
        F = L.dot(S_flat) + (A_flat**2) * S_flat - rho * (1 - S_flat)
        # Jacobian J(S)
        J = L + diags(A_flat**2 + rho, 0, shape=(N**2, N**2), format="csr")
        
        # Solve the linear system
        delta = spsolve(J, -F)
        
        # Update S
        S_flat += delta
        S = S_flat.reshape(N, N)
        
        # Check for convergence
        if np.linalg.norm(F, np.inf) < tol:
            print("Converged after", iteration, "iterations.", np.linalg.norm(F, np.inf), np.linalg.norm(S, np.inf))
            break
    else:
        print("Did not converge after maximum iterations.")
    
    return S

# # Example usage
# # Initial conditions
# N = 63  # Grid size
# h = 1.0 / (N - 1)
# DA = 2.5e-4
# DS = 5e-4
# mu = 0.065
# rho = 0.04
# X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

# S = np.ones((N, N)) * 0.01  # Initial guess for S
# Inital guess for A, this is from the data generated by Sun's code
# A_initials = np.load('initials.npy') 



# for i in range(100):
#     A = A_initials[i,:].reshape(N, N)

    # solve S for given A to give initial guess

    # S = solve_for_S(A, DS, rho, N)
    # # Plot results surface and contour for S in single plot
    # fig = plt.figure(figsize=(12, 6))
    # ax = fig.add_subplot(121, projection='3d')
    # X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    # ax.plot_surface(X, Y, A, cmap='viridis')
    # ax.set_title('Concentration of A')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('A')

    # ax = fig.add_subplot(122)
    # ax.contourf(X, Y, A, levels=50)
    # ax.set_title('Concentration of S')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # plt.colorbar(ax.contourf(X, Y, A, levels=50))
    # plt.savefig('GrayScottS.png')


    # A, S = newton_method_for_gray_scott(A, S, DA, DS, mu, rho, N, h)

    # # Plot results surface and contour for A and S in single plot
    # fig = plt.figure(figsize=(12, 6))
    # ax = fig.add_subplot(121)
    # ax.contourf(X, Y, S, levels=50)
    # ax.set_title('Concentration of S')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # plt.colorbar(ax.contourf(X, Y, S, levels=50))

    # ax = fig.add_subplot(122)
    # ax.contourf(X, Y, A, levels=50)
    # ax.set_title('Concentration of A')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # plt.colorbar(ax.contourf(X, Y, A, levels=50))
    # plt.savefig('GrayScottA.png')
    # plt.show()




    # # Initial conditions
    # N = 50  # Grid size
    # DA = 0.1
    # DS = 0.05
    # mu = 0.04
    # rho = 0.06

    # A_initial = generate_initial_guess_sine_series(N)
    # S_initial = generate_initial_guess_sine_series(N)

    # A, S = newton_method(N, DA, DS, mu, rho, A_initial, S_initial, max_iter=50)

    # # Plot results
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

    # ax1.contourf(X, Y, A, levels=50)
    # ax1.set_title('Concentration of A')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')

    # ax2.contourf(X, Y, S, levels=50)
    # ax2.set_title('Concentration of S')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # # set colorbar
    # plt.colorbar(ax1.contourf(X, Y, A, levels=50))
    # plt.colorbar(ax2.contourf(X, Y, S, levels=50))

    # plt.savefig('GrayScott.png')
    # plt.show()

    # A_list.append(A)
    # S_list.append(S)

# # check if the solution is duplicated (discrepancy between A solutions is less than 1e-2)
# # if duplicated, remove the duplicated solution for both A and S
# for i in range(len(A_list)):
#     for j in range(i+1, len(A_list)):
#         if np.linalg.norm(A_list[i] - A_list[j], np.inf) < 1e-2:
#             A_list.pop(j)
#             S_list.pop(j)
#             print('Duplicated solution found and removed.')
#             break

# # save the solutions
# np.save('A.npy', A_list)
# np.save('S.npy', S_list)


## Generate Data for training by sample the perturbation as the following
# u_list, delta_u_list = [], []
# for _ in range(2000):
#     u_2_init = solution +  4 * generate_initial_guess_sine_series(N)
#     u, new_u_list, new_delta_u_list = newton_method_with_corrected_laplacian(N, s=s, u=u_2_init, max_iter=6, tol=1e-6)
#     u_list += new_u_list
#     delta_u_list += new_delta_u_list


# u_list = torch.tensor(u_list, dtype=torch.float64)
# delta_u_list = torch.tensor(delta_u_list, dtype=torch.float64)
# # reshape the data
# u_list = u_list.reshape(-1, 1, N, N)
# delta_u_list = delta_u_list.reshape(-1, 1, N, N)
# # save the data in a single file
# torch.save((u_list, delta_u_list, solution), 'newton_method_iterations_7_solution.pt')

S = np.load('S.npy')
A = np.load('A.npy')
# Initial conditions
N = 63  # Grid size
h = 1.0 / (N - 1)
DA = 2.5e-4
DS = 5e-4
mu = 0.065
rho = 0.04
X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

AS_list, deltaAS_list = [], []
indices = [i for i in range(1, 50, 5) if i != 11]
for i in indices:
    for j in range(1):
        perturbation = 0.4 * generate_initial_guess_sine_series(N)
        A_initial = A[i] + ((A[i]+1e-2)*perturbation)
        S_initial = S[i] - ((S[i]+1e-2)*perturbation)
        _, _, AS, deltaAS = newton_method_for_gray_scott(A_initial, S_initial, DA, DS, mu, rho, N, h,tol=1e-2, max_iter=40)
        AS_list += AS
        deltaAS_list += deltaAS
        print(i, j)

AS_list = np.array(AS_list)
deltaAS_list = np.array(deltaAS_list)
AS_list = torch.tensor(AS_list, dtype=torch.float64)
deltaAS_list = torch.tensor(deltaAS_list, dtype=torch.float64)
# reshape the data
AS_list = AS_list.reshape(-1, 2, N, N)
deltaAS_list = deltaAS_list.reshape(-1, 2, N, N)
# save the data in a single file
torch.save((AS_list, deltaAS_list), 'newton_multiSolutions_samples.pt')

   
    






