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

dataOpt = {}
dataOpt['batch_size'] = 10
dataOpt['dataSize'] = {'train': range(5000), 'test': range(4000, 5000), 'val':range(600,650)}

x_train = x[dataOpt['dataSize']['train'],...]
y_train = y[dataOpt['dataSize']['train'],...]
x_test = x[dataOpt['dataSize']['test'],...]
y_test = y[dataOpt['dataSize']['test'],...]

model = torch.load('/home/liux0t/FMM/MgNO/model/DeepONetns_merge2024-05-05 17:04:03.879338.pt')
model.eval()


# Warm-up
for _ in range(10):
    _ = model(x_train)

# Measure inference time
start_time = time.perf_counter()
outputs = model(x_train)
end_time = time.perf_counter()

# Calculate and print the inference time
inference_time = end_time - start_time
print(f"Inference time: {inference_time} seconds")



