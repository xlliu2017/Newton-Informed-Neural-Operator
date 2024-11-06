# profile the speed of the spectral convolutions
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.profiler
from functools import partial
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, init_scale=2, out_resolution=None):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels)) 
        self.fourier_weight1 = nn.Parameter(self.scale*
            torch.randn(in_channels, out_channels, modes1, modes2,  dtype=torch.cfloat)) 
        self.fourier_weight2 = nn.Parameter(self.scale*
            torch.randn(in_channels, out_channels, modes1, modes2,  dtype=torch.cfloat))                                      
        if init_scale:
            nn.init.uniform_(self.fourier_weight1, a=-self.scale*(1/init_scale), b=self.scale*(1/init_scale))
            nn.init.uniform_(self.fourier_weight2, a=-self.scale*(1/init_scale), b=self.scale*(1/init_scale))
        
        
    
    # Complex multiplication
    @staticmethod
    def compl_mul2d(input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    @staticmethod
    def complex_matmul_2d(a, b):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x, out_resolution=None):
        batch_size = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm='forward')
            
        # Multiply relevant Fourier modes the 2d Hermmit symmetric refers to two oppsite directions 
        out_ft = torch.zeros(batch_size, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.fourier_weight1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.fourier_weight2)

        # #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='forward')
        
        

        # Return to physical space
        # if out_resolution:
        #     p1d = (0, 0, 129, 129)
        #     out_ft = F.pad(out_ft, p1d, "constant", 0)
        #     x = torch.fft.irfft2(out_ft, s=out_resolution, norm='forward')
        # else:
        #     x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='forward')
        return x


device = 'cuda'
in_channels = 32
out_channels = 32
modes1 = 60
modes2 = 60
input_size = 127
model = SpectralConv2d(in_channels, out_channels, modes1, modes2, out_resolution=128).to(device)
input_tensor = torch.randn((100, in_channels, input_size, input_size), device=device)

def run_model():
    torch.cuda.synchronize()  # Ensure all GPU operations are complete before starting timing
    start_time = time.time()
    for _ in range(100):
        output = model(input_tensor)
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