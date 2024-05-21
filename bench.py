import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import matplotlib.pyplot as plt
import time
from utilities3 import *
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

u_list, delta_u_list, sol_true = torch.load('/home/liux0t/neural_MG/pytorch/newton_method_iterations_7_solution.pt')
y = delta_u_list.float().to(device).squeeze()
x = u_list.to(device).float()
# y = y/y.max()
# x = x/x.max()
dataOpt = {}
dataOpt['batch_size'] = 10
dataOpt['dataSize'] = {'train': range(500), 'test': range(4000, 5000), 'val':range(600,650)}

x_train = x[dataOpt['dataSize']['train'],...]
y_train = y[dataOpt['dataSize']['train'],...]
x_test = x[dataOpt['dataSize']['test'],...]
y_test = y[dataOpt['dataSize']['test'],...]
# x_val = x[dataOpt['dataSize']['val'],...]
# y_val = y[dataOpt['dataSize']['val'],...]

# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)
# if model_type == 'FNO':
# x_train = x_train.squeeze()
# x_test = x_test.squeeze()
# x_train = x_train[:, ..., np.newaxis]
# x_test = x_test[:, ..., np.newaxis]

# model = torch.load('/home/liux0t/FMM/MgNO/model/DeepONetns_merge2024-05-05 17:04:03.879338.pt')
# model.eval()
# l2loss = LpLoss(size_average=False) 
# test_l2 = 0.
# # with torch.no_grad():
# #     for x, y in test_loader:
# #         x, y = x.to(device), y.to(device)
# #         out = model(x).squeeze()
# #         test_l2 += l2loss(out, y).item()
        
# # test_l2/= len(dataOpt['dataSize']['test'])
# # test_l2

# # Warm-up
# for _ in range(10):
#     _ = model(x_train)

# # Measure inference time
# start_time = time.perf_counter()
# outputs = model(x_train)
# end_time = time.perf_counter()

# # Calculate and print the inference time
# inference_time = end_time - start_time
# print(f"Inference time: {inference_time} seconds")


## bench the newton solver

import cupy as cp
from cupyx.scipy.sparse import csr_matrix, diags
from cupyx.scipy.sparse.linalg import spsolve




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
    h_squared = (1/h**2).astype(cp.float32)
    print(L.dtype)
    return L * h_squared

def solve_systems_gpu(A, u_flats, fixed_rhs, N):
    num_systems = u_flats.shape[1]
    fixed_rhs = fixed_rhs[:, cp.newaxis]

    rhs_list = (fixed_rhs + cp.power(u_flats, 2) - A.dot(u_flats)).astype(cp.float32)
    diagonal_list = (-2 * u_flats).astype(cp.float32)

    rhs_list = rhs_list.T
    diagonal_list = diagonal_list.T
    delta_u_list = cp.zeros_like(u_flats.T, dtype=cp.float32)

    for i in range(num_systems):
        rhs = rhs_list[i]
        diagonal = diagonal_list[i]
        J = A + diags([diagonal], [0], shape=(N**2, N**2), format='csr')
        # print(A.dtype, rhs.dtype)
        delta_u = spsolve(J, rhs)
        # delta_u_list[i] = delta_u.astype(cp.float32)

    return  

# Usage
N = 63
h = 1.0 / (N + 1)
h = cp.float32(h)
A = create_laplacian_matrix_with_boundary_conditions(N, h)
print(A.dtype)
x_train = x_train.view(-1, 500)
dlpack = torch.utils.dlpack.to_dlpack(x_train)  # Export tensor data to DLPack
u_flats = cp.fromDlpack(dlpack).astype(cp.float32)
fixed_rhs = create_rhs(N, h, s=1.0)

start_time = time.perf_counter()
delta_u_list = solve_systems_gpu(A, u_flats, fixed_rhs, N)
end_time = time.perf_counter()

inference_time = end_time - start_time
print(f"Inference time: {inference_time} seconds")

