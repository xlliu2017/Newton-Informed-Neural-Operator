import cupy as cp
from cupyx.scipy.sparse import csr_matrix, diags
from cupyx.scipy.sparse.linalg import spsolve
import numpy as np
import time
import cProfile
import pstats

def create_rhs(N, h, s):
    """Create the right-hand side of the linear system for GPU."""
    x = cp.linspace(0, 1, N, dtype=cp.float32)
    y = cp.linspace(0, 1, N, dtype=cp.float32)
    X, Y = cp.meshgrid(x, y)
    rhs = (-s * cp.sin(cp.pi * X) * cp.sin(cp.pi * Y)).astype(cp.float32)
    return rhs.flatten()

def create_laplacian_matrix_with_boundary_conditions(N, h):
    """Create a CuPy sparse matrix for the discrete Laplacian operator."""
    main_diag = cp.full((N**2,), 4.0, dtype=cp.float32)
    off_diag1 = cp.full((N**2 - 1,), -1.0, dtype=cp.float32)
    off_diag2 = cp.full((N**2 - N,), -1.0, dtype=cp.float32)
    for i in range(1, N):
        off_diag1[i * N - 1] = 0
    diagonals = [main_diag, off_diag1, off_diag1, off_diag2, off_diag2]
    offsets = [0, -1, 1, -N, N]
    L = diags(diagonals, offsets, shape=(N**2, N**2), format="csr", dtype=cp.float32)
    return L * (1 / h**2)

def solve_systems_gpu(A, u_flats, fixed_rhs, N):
    num_systems = u_flats.shape[1]
    fixed_rhs = fixed_rhs[:, cp.newaxis]
    rhs_list = (fixed_rhs + cp.power(u_flats, 2) - A.dot(u_flats)).astype(cp.float32)
    delta_u_list = cp.zeros_like(u_flats.T, dtype=cp.float32)
    for i in range(num_systems):
        rhs = rhs_list[i]
        J = A + diags([-2 * u_flats[:, i]], [0], shape=(N**2, N**2), format='csr')
        delta_u = spsolve(J, rhs)
        delta_u_list[i] = delta_u
    return delta_u_list

# def main():
#     N = 63
#     h = 1.0 / (N + 1)
#     h = cp.float32(h)
#     A = create_laplacian_matrix_with_boundary_conditions(N, h)
#     s = 1.0
#     fixed_rhs = create_rhs(N, h, s)

#     # Assume u_flats is some predefined data; for profiling we'll use random data
#     u_flats = cp.random.rand(N**2, 500).astype(cp.float32)

#     # Start profiling
#     start_time = time.time()
#     delta_u_list = solve_systems_gpu(A, u_flats, fixed_rhs, N)
#     cp.cuda.Stream.null.synchronize()  # Synchronize to ensure all GPU computations are finished
#     end_time = time.time()

#     print(f"Elapsed time: {end_time - start_time:.4f} seconds")

# # Profile the main function using cProfile
# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()
#     main()
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()

def solve_systems_gpu(A, u_flats, fixed_rhs, N, stream):
    with cp.cuda.Stream(stream):
        num_systems = u_flats.shape[1]
        fixed_rhs = fixed_rhs[:, cp.newaxis]
        rhs_list = (fixed_rhs + cp.power(u_flats, 2) - A.dot(u_flats)).astype(cp.float32)
        rhs_list = rhs_list.T
        delta_u_list = cp.zeros_like(u_flats.T, dtype=cp.float32)
        diagonal_list = (-2 * u_flats).astype(cp.float32)
        diagonal_list = diagonal_list.T
 
        for i in range(num_systems):
            rhs = rhs_list[i]
            J = A + diags([diagonal_list[i]], [0], shape=(N**2, N**2), format='csr')
            delta_u = spsolve(J, rhs)
            delta_u_list[i] = delta_u

    return delta_u_list


def main():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    u_list, delta_u_list, sol_true = torch.load('/home/liux0t/neural_MG/pytorch/newton_method_iterations_7_solution.pt')  
    x = u_list.to(device).float()
   
    dataOpt = {}
    dataOpt['dataSize'] = {'train': range(500), 'test': range(4000, 5000), 'val':range(600,650)}
    x_train = x[dataOpt['dataSize']['train'],...]

    N = 63
    h = 1.0 / (N + 1)
    h = cp.float32(h)
    A = create_laplacian_matrix_with_boundary_conditions(N, h)
    s = 1.0
    fixed_rhs = create_rhs(N, h, s)

    # Assume u_flats is some predefined data; for profiling we'll use random data
    total_systems = 500
    num_streams = 5
    systems_per_stream = total_systems // num_streams
    # u_flats = cp.random.rand(N**2, total_systems).astype(cp.float32)
    x_train = x_train.view(-1, 500)
    dlpack = torch.utils.dlpack.to_dlpack(x_train)  # Export tensor data to DLPack
    u_flats = cp.fromDlpack(dlpack).astype(cp.float32)
    
    streams = [cp.cuda.Stream() for _ in range(num_streams)]
    results = []

    start_time = time.time()

    # Dispatch jobs to streams
    for i in range(num_streams):
        start_idx = i * systems_per_stream
        end_idx = start_idx + systems_per_stream
        stream_u_flats = u_flats[:, start_idx:end_idx]
        result = solve_systems_gpu(A, stream_u_flats, fixed_rhs, N, streams[i])
        results.append(result)

    # Wait for all streams to complete
    for stream in streams:
        stream.synchronize()

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    return results

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()