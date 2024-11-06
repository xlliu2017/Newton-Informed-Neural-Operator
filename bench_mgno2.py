import torch
# torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import time
import torch.profiler

# Adjusted Conv_Dyn class to avoid dimension mismatch
class Conv_Dyn(nn.Module):
    def __init__(self, kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=1, bias=False, padding_mode='replicate', resolution=480):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=stride, padding=0, bias=True, padding_mode=padding_mode)

    def forward(self, out):
        u, f, a, diva, r = out
        if diva is None:
            diva = self.conv_1(a)
        if u is None:
            r = f if r is None else r 
            u = self.conv_0(torch.tanh(diva)) * r
        else:
            r = f - diva * u
            u = u + self.conv_0(torch.tanh(diva)) * r                             
        out = (u, f, a, diva, r)
        return out

class MgRestriction(nn.Module): 
    def __init__(self, kernel_size=3, in_channels=1, out_channels=1, stride=2, padding=0, bias=False, padding_mode='zeros'):
        super().__init__()

        self.R_1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=stride, padding=0, bias=bias, padding_mode=padding_mode)
        self.R_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.R_3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)

    def forward(self, out):
        u_old, f_old, a_old, diva, r_old = out
        if diva is None:
            a = self.R_1(a_old)  
        else:
            a = a_old                          
        r = self.R_3(r_old)                               
        out = (None, None, a, diva, r)
        return out
    

class MG_FEM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_iterations=[1, 1, 1, 1, 1]):
        super().__init__()
        self.num_iterations = num_iterations
        self.resolutions = [127, 63, 31, 15, 7, 4]
        
        self.transpose_layers = nn.ModuleList()
        for _ in range(len(num_iterations) - 1):
            self.transpose_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=3, 
                    stride=2, 
                    padding=0, 
                    bias=False
                )
            )

        self.conv_layers = nn.ModuleList()
        layers = []
        for layer_index, num_iterations_layer in enumerate(num_iterations):
            for _ in range(num_iterations_layer):
                layers.append(Conv_Dyn(
                    in_channels=in_channels, 
                    out_channels=out_channels,
                    resolution=self.resolutions[layer_index]
                ))
            self.conv_layers.append(nn.Sequential(*layers))

            if layer_index < len(num_iterations) - 1:
                layers = [MgRestriction(in_channels=in_channels, out_channels=out_channels)]

    def forward(self, u, f, a, diva_list=[None for _ in range(7)], r=None):
   
        out_list = [None] * len(self.num_iterations)
        
        for layer_index in range(len(self.num_iterations)):
            out = (u, f, a, diva_list[layer_index], r)
            u, f, a, diva, r = self.conv_layers[layer_index](out)
            out_list[layer_index] = (u, f, a, r)
            if diva_list[layer_index] is None:
                diva_list[layer_index] = diva

        for layer_index in range(len(self.num_iterations) - 2, -1, -1):
            u, f, a, r = out_list[layer_index]
            u_post = u + self.transpose_layers[layer_index](out_list[layer_index + 1][0])
            out_list[layer_index] = (u_post, f, a, r)
            
        return out_list[0][0], out_list[0][1], out_list[0][2], diva_list

# Initialize model
device = 'cuda'
in_channels = 32
out_channels = 32
input_size = 127
num_iterations = [1, 1, 1, 1, 1]
model = MG_FEM(in_channels=in_channels, out_channels=out_channels, num_iterations=num_iterations).to(device)

# Dummy input data
batch_size = 4
u = None
f = torch.randn((100, in_channels, input_size, input_size), device=device)
a = None
resolutions = [127, 63, 31, 15, 7, 3]

diva_list = [torch.randn((100, in_channels, resolutions[i], resolutions[i]), device=device) for i in range(len(num_iterations))]

# u = 
# f = torch.randn(batch_size, 1, 480, 480).to('cpu')
# a = torch.randn(batch_size, 1, 480, 480).to('cpu')
# diva_list = [None for _ in range(7)]
# r = torch.randn(batch_size, 1, 480, 480).to('cpu')

# Example forward pass through the model
def run_model():
    torch.cuda.synchronize()  # Ensure all GPU operations are complete before starting timing
    start_time = time.time()
    for _ in range(100):
        output = model(u, f, a, diva_list)
    torch.cuda.synchronize()  # Ensure all GPU operations are complete before stopping timing
    end_time = time.time()

    average_time_per_iteration = (end_time - start_time) / 100
    print(f"Average time per iteration: {average_time_per_iteration} seconds")

# Set up the profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,  # Optional: Records the shape of tensors
    profile_memory=True,  # Optional: Track memory usage
    with_stack=True  # Optional: Include stack traces
) as prof:
    with torch.profiler.record_function("model_inference"):
        run_model()  # Run the model inference you want to profile

# Print the profiler results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Optionally, export the results to a file (e.g., JSON or Chrome trace format)
prof.export_chrome_trace("trace.json")