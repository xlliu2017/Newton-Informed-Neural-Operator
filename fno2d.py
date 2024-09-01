

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.transforms import Lambda
from torchvision import transforms
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

import time
import tracemalloc
from timeit import default_timer
from utilities3 import *

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################
# class SpectralConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super(SpectralConv2d, self).__init__()

#         """
#         2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
#         """

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
#         self.modes2 = modes2

#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

#     # Complex multiplication
#     def compl_mul2d(self, input, weights):
#         # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
#         return torch.einsum("bixy,ioxy->boxy", input, weights)

#     def forward(self, x):
#         batchsize = x.shape[0]
#         #Compute Fourier coeffcients up to factor of e^(- something constant)
#         x_ft = torch.fft.rfft2(x)

#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes1, :self.modes2] = \
#             self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
#         out_ft[:, :, -self.modes1:, :self.modes2] = \
#             self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

#         #Return to physical space
#         x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
#         return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, normalizer=None):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.normalizer = normalizer
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.normalizer.decode(x.squeeze()) if self.normalizer is not None else x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0., activation='gelu', layer=2, LN=True):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'gelu':
            act = nn.GELU
        elif activation == 'tanh':
            act = nn.Tanh
        else: raise NameError('invalid activation')
         
        if LN:
            if layer == 2:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim)
                )
            elif layer == 3:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                    act(),
                    nn.LayerNorm(out_dim)
                )
            else: raise NameError('only accept 2 or 3 layers')
        else:
            if layer == 2:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                )
            elif layer == 3:
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                )
            else: raise NameError('only accept 2 or 3 layers')
            
    def forward(self, x):
        return self.net(x)
    def reset_parameters(self):
        for layer in self.children():
            for n, l in layer.named_modules():
                if hasattr(l, 'reset_parameters'):
                    print(f'Reset trainable parameters of layer = {l}')
                    l.reset_parameters()

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, init_scale=2):
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

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='forward')
        
        

        #Return to physical space
        # if out_resolution:
        #     p1d = (0, 0, 128, 128)
        #     out_ft = F.pad(out_ft, p1d, "constant", 0)
        #     x = torch.fft.irfft2(out_ft, s=out_resolution, norm='forward')
        # else:
        #     x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='forward')
        return x
    

                    
class SpectralDecoder(nn.Module):
    def __init__(self, modes=12, width=32, num_spectral_layers=4, mlp_hidden_dim=128, 
                lift=False, output_dim=1, mlp_LN=False, activation='gelu', 
                kernel_type='c', padding=0, resolution=None, init_scale=2, 
                add_pos=False, shortcut=True, normalizer=nn.Identity(), if_pre_permute=True):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes
        self.modes2 = modes
        self.add_pos = add_pos
        self.shortcut = shortcut    
        self.resolution = resolution
        self.padding = padding
        if add_pos:
            self.width = width + 2
        else:
            self.width = width        
        self.num_spectral_layers = num_spectral_layers
        if lift:
            self.lift = lift
            self.fc0 = nn.Linear(lift, self.width)
            self.conv0 = nn.Conv2d(lift, self.width, 1)
            
        self.pre_permute = partial(torch.permute, dims=(0, 3, 1, 2))
        self.post_permute = partial(torch.permute, dims=(0, 2, 3, 1))

        self.mlp = FeedForward(self.width, mlp_hidden_dim, output_dim, LN=mlp_LN)
        self.mlp2 = nn.Sequential(nn.Conv2d(self.width, mlp_hidden_dim, 1), nn.GELU(), nn.Conv2d(mlp_hidden_dim, output_dim, 1))
        if normalizer:
            self.normalizer = normalizer
     
        # pad the domain if input is non-periodic
        if if_pre_permute:
            if padding:
                if lift:  
                    self.pre_process = transforms.Compose([self.fc0, self.pre_permute, self.padding_trans, ])
                else: self.pre_process = transforms.Compose([self.pre_permute, self.padding_trans, ])
                if normalizer:
                    self.post_process = transforms.Compose([self.crop_trans, self.post_permute, self.mlp, torch.squeeze, self.normalizer.decode])
                else:
                    self.post_process = transforms.Compose([self.crop_trans, self.post_permute, self.mlp, torch.squeeze])
            else:
                if lift:  
                    self.pre_process = transforms.Compose([self.fc0, self.pre_permute])
                else: self.pre_process = self.pre_permute
                if normalizer:
                    self.post_process = transforms.Compose([self.post_permute, self.mlp, torch.squeeze, self.normalizer.decode])
                else:
                    self.post_process = transforms.Compose([self.post_permute, self.mlp, torch.squeeze])
        else:   
            self.post_process = transforms.Compose([self.crop_trans, self.mlp2, self.normalizer.decode]) # for wenrui's code
            self.pre_process = transforms.Compose([self.conv0, self.padding_trans, ])
            # if normalizer:
            #     self.post_process = transforms.Compose([self.post_permute, self.mlp, torch.squeeze, self.normalizer.decode])   
            # else:
            #     self.post_process = transforms.Compose([self.post_permute, self.mlp, torch.squeeze])

        self.Spectral_Conv_List = nn.ModuleList([])      
        for _ in range(num_spectral_layers):
            self.Spectral_Conv_List.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2, init_scale)) 


        self.Conv2d_list = nn.ModuleList([])         
        if kernel_type == 'p':
            for _ in range(num_spectral_layers):
                self.Conv2d_list.append(nn.Conv2d(self.width, self.width, 1))
        else:         
            for _ in range(num_spectral_layers):
                self.Conv2d_list.append(nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, dilation=1))  

        self.register_buffer('extrapolation', torch.ones(2, 2))     

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation')
  
        
        self.grid = None
        
        
    def forward(self, x, out_resolution=None):
        if self.add_pos:
            if self.grid is None:        
                grid = self.get_grid(x.shape, x.device)
                self.grid = grid
                x = torch.cat((x, grid), dim=-1)
            else: x = torch.cat((x, self.grid), dim=-1)

        # x = x.permute(0, 3, 1, 2)
        # if self.padding:
        #     x = F.pad(x, [0,self.padding, 0,self.padding])
        x = self.pre_process(x)
        x1 = self.Spectral_Conv_List[0](x)
        x2 = self.Conv2d_list[0](x)
        x = self.act(x1 + x2)
        if self.shortcut:
            x_shortcut = x

        for i in range(1, self.num_spectral_layers - 1):
            x = self.Spectral_Conv_List[i](x) + self.Conv2d_list[i](x)
            x = self.act(x)

        x1 = self.Spectral_Conv_List[-1](x, out_resolution=out_resolution) 
        x2 = self.Conv2d_list[-1](x)
        if self.shortcut:
            x = x1 + x2 + x_shortcut
        else:
            if out_resolution:
                x = x1 + torch.kron(x2, self.extrapolation)
            else:
                x = x1 + x2
        x = self.post_process(x)        
        # if self.padding:
        #     x = x[..., :self.resolution, :self.resolution]
        # x = x.permute(0, 2, 3, 1)
        # x = self.mlp(x)
        return x
    
    def get_grid(self, shape, device):
        batch_size, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batch_size, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batch_size, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def crop_trans(self, x):
        return x[..., :self.resolution, :self.resolution]
    def padding_trans(self, x):
        return  F.pad(x, [0, self.padding, 0, self.padding])

def benchmark_spectralconv2d(in_channels, out_channels, modes1, modes2, input_size, device):
    model = SpectralConv2d(in_channels, out_channels, modes1, modes2).to(device)
    input_tensor = torch.randn((10, in_channels, input_size, input_size), device=device)

    # CPU Memory usage
    tracemalloc.start()

   # GPU Memory & Time usage
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    
    start_time = time.time()


    # with torch.no_grad():
    # for _ in range(10):
    output = model(input_tensor)
    
    output.sum().backward()
    
    

    if device == 'cuda':
        torch.cuda.synchronize()
        peak_memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_memory_usage = 0

    cpu_memory_usage = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
    end_time = time.time()  

    tracemalloc.stop()

    elapsed_time = end_time - start_time

    print(f"SpectralConv2d Benchmark:")
    print(f"Time elapsed: {elapsed_time:.4f} seconds")
    print(f"Peak GPU memory usage: {peak_memory_usage:.2f} MB")
    print(f"Peak CPU memory usage: {cpu_memory_usage:.2f} MB\n")


class Conv_Dyn(nn.Module):
    def __init__(self, kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=1, bias=False, padding_mode='replicate', resolution=480):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=stride, padding=0, bias=True, padding_mode=padding_mode)


    def forward(self, out):
        u, f, a, diva, r = out
        if diva is None:
            # diva = self.conv_0(F.tanh((self.conv_1(a))))
            diva = self.conv_1(a)
        # f = self.conv_3(f) - diva * self.conv_2(u) 
        if u is None:
            r = f if r is None else r 
            u = self.conv_0(F.tanh(diva)) * r
        else:
            r = f - diva * u
            u = u + self.conv_0(F.tanh(diva)) * r                             
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
    


class MG_fem(nn.Module):
    # Multigrid Finite Element Method (MG_fem) module.
    # This module implements the Multigrid Finite Element Method (MG_fem) for solving partial differential equations.
    # It consists of a series of convolutional layers and restriction layers to iteratively solve the equation.
    # Args:
    #     in_channels (int): Number of input channels.
    #     out_channels (int): Number of output channels.
    #     num_iterations (list): List of integers specifying the number of iterations for each layer.
    # Attributes:
    #     resolutions (list): List of resolutions for each layer.
    #     transpose_layers (nn.ModuleList): List of transpose convolution layers.
    #     conv_layers (nn.ModuleList): List of convolution layers.
    # Methods:
    #     forward(u, f, a, diva_list=None, r=None): Performs forward pass of the MG_fem module.
    #     Initializes the MG_fem module.
    #     Args:
    #         in_channels (int, optional): Number of input channels. Defaults to 1.
    #         out_channels (int, optional): Number of output channels. Defaults to 1.
    #         num_iterations (list, optional): List of integers specifying the number of iterations for each layer. Defaults to [1, 1, 1, 1, 1].
    #     Performs forward pass of the MG_fem module.
    #     Args:
    #         u (torch.Tensor): Input tensor representing the solution.
    #         f (torch.Tensor): Input tensor representing the forcing term.
    #         a (torch.Tensor): Input tensor representing the coefficient function.
    #         diva_list (list, optional): List of tensors representing the divergence of a for each layer. Defaults to None.
    #         r (torch.Tensor, optional): Input tensor representing the residual. Defaults to None.
    #     Returns:
    #         torch.Tensor: Output tensor representing the solution.

    def __init__(self, in_channels=1, out_channels=1, num_iterations=[1, 1, 1, 1, 1]):
        super().__init__()
        self.num_iterations = num_iterations
        self.resolutions = [127, 63, 31, 15, 7, 4]
        
        # Initialize transpose convolution layers
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

        # Initialize convolution layers
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

            # Add restriction layer for all but the last layer
            if layer_index < len(num_iterations) - 1:
                layers = [MgRestriction(in_channels=in_channels, out_channels=out_channels)]

    def forward(self, u, f, a, diva_list=[None for _ in range(7)], r=None):
   
        out_list = [None] * len(self.num_iterations)
        
        # Down pass through each layer
        for layer_index in range(len(self.num_iterations)):
            out = (u, f, a, diva_list[layer_index], r)
            u, f, a, diva, r = self.conv_layers[layer_index](out)
            out_list[layer_index] = (u, f, a, r)
            if diva_list[layer_index] is None:
                diva_list[layer_index] = diva

        # Up pass through transpose layers
        for layer_index in range(len(self.num_iterations) - 2, -1, -1):
            u, f, a, r = out_list[layer_index]
            u_post = u + self.transpose_layers[layer_index](out_list[layer_index + 1][0])
            out_list[layer_index] = (u_post, f, a, r)
            
        return out_list[0][0], out_list[0][1], out_list[0][2], diva_list




def benchmark_mg_fem(in_channels, out_channels, input_size, num_iterations, device):
    model = MG_fem(in_channels, out_channels, num_iterations=num_iterations).to(device)
    input_tensor_u = None
    input_tensor_f = torch.randn((10, in_channels, input_size, input_size), device=device)
    input_tensor_a = None
    resolutions = [127, 63, 31, 15, 7, 3]

    input_tensor_diva = [torch.randn((10, in_channels, resolutions[i], resolutions[i]), device=device) for i in range(len(num_iterations))]

    # CPU Memory usage
    tracemalloc.start()

    # GPU Memory & Time usage
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    
    start_time = time.time()

    # with torch.no_grad():
    # for _ in range(10):
    output = model(input_tensor_u, input_tensor_f, input_tensor_a, input_tensor_diva)
    output[0].sum().backward()
    end_time = time.time()

    if device == 'cuda':
        torch.cuda.synchronize()
        peak_memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # In MB
    else:
        peak_memory_usage = 0  # No GPU memory usage on CPU

    cpu_memory_usage = tracemalloc.get_traced_memory()[1] / (1024 ** 2)  # In MB
    tracemalloc.stop()

    elapsed_time = end_time - start_time

    print(f"MG_fem Benchmark:")
    print(f"Time elapsed: {elapsed_time:.4f} seconds")
    print(f"Peak GPU memory usage: {peak_memory_usage:.2f} MB")
    print(f"Peak CPU memory usage: {cpu_memory_usage:.2f} MB\n")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
    # model = FNO2d(12, 12, 32).to(device)
    # model = SpectralDecoder(modes=12, width=32, num_spectral_layers=4, mlp_hidden_dim=128, 
    #             lift=2, output_dim=2, mlp_LN=False, activation='gelu', 
    #             kernel_type='c', padding=0, resolution=None, init_scale=2, 
    #             add_pos=False, shortcut=True, normalizer=nn.Identity(), if_pre_permute=False).to(device)
    # print(model)
    # x = torch.randn(2, 2, 64, 64).to(device)
    # y = model(x)
    # print(y.shape)
    benchmark_spectralconv2d(64, 64, 60, 60, 128, device)
    benchmark_mg_fem(64, 64, 127, [1, 1, 1, 1, 1, 1], device)

    