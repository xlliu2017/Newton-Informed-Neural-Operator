import torch
import torch.nn.functional as F

import numpy as np
import scipy.io
import h5py
from functools import reduce, partial
import operator
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from datetime import date, datetime

from Adam import Adam
#################################################
#
# Utilities
#
#################################################

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super().__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path, 'r')
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

def load_from_h5(h5_all_data_path):
    """Load all data from h5."""

    print(f'Load simulation data from h5 format.')

    # Init h5_keys
    h5_keys = ['realization', 'perm_3d', 'poil_3d', 'soil_3d', 'swat_3d', 'timeSteps']

    # If h5 file already save all the numpy array
  
    data = {}

    # Get all the numpy array first
    with h5py.File(h5_all_data_path, 'r') as hf:
        for key in h5_keys:
            if key in hf.keys():
                val = hf.get(name=key)[:]
                data.update({key: val})
                print(f'Complete loading {key}.')

    # - dimension parameters
    n_samples, n_time_steps, n_grid_x, n_grid_y, n_grid_z = data['poil_3d'].shape
    data.update({
        # - dimension parameters
        'n_samples': n_samples,
        'n_time_steps': n_time_steps,
        'n_grid_x': n_grid_x,
        'n_grid_y': n_grid_y,
        'n_grid_z': n_grid_z,
    })

    return data


def merge_params(args, optimizerScheduler_args, FMM_paras, dataOpt, Decoder_paras):
    """Merge parameters from args and default parameters."""
    for key in optimizerScheduler_args.keys():
        if key in args.keys():
            optimizerScheduler_args[key] = args[key]

        for key in FMM_paras.keys():
            if key in args.keys():
                FMM_paras[key] = args[key]

        for key in dataOpt.keys():
            if key in args.keys():
                dataOpt[key] = args[key]

        for key in Decoder_paras.keys():
            if key in args.keys():
                Decoder_paras[key] = args[key]
    FMM_paras['embed_dim'] = args['width']
    return optimizerScheduler_args, FMM_paras, dataOpt, Decoder_paras

# normalization, pointwise gaussian
class UnitGaussianNormalizer(nn.Module):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.register_buffer('mean', torch.mean(x, 0))
        self.register_buffer('std', torch.std(x, 0))
        self.eps = eps
    
    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

# normalization, Gaussian
class GaussianNormalizer(nn.Module):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

  

class GaussianImageNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super().__init__()

        # self.mean = torch.mean(x, [0,1,2])
        self.std = torch.std(x, [0,1,2])
        self.eps = eps

    def encode(self, x):
        x = (x) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) 
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, relative=True):
        super().__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.relative = relative

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.view(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                if self.relative:
                    return torch.sum(diff_norms/y_norms)
                else: 
                    return torch.sum(diff_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        
        return self.rel(x, y)



class HSloss_d(nn.MSELoss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)
    
    def forward(self, x, y):
        temp = x - y
        z0, z1 = torch.gradient(temp, dim=(-2, -1), spacing=1/x.size(-1)) 
        agg = 10*temp**2 + z0**2 + z1**2
        loss = torch.mean(torch.sqrt(torch.sum(agg, dim=(-2, -1))))  
        return loss, torch.zeros(1), torch.zeros(x.size(0)), torch.zeros(x.size(0))

class HSloss_d_2(nn.MSELoss):
    def __init__(self, reduction='sum'):
        super().__init__(reduction=reduction)
    
    def forward(self, x, y):
        temp = x - y
        z0, z1 = torch.gradient(temp, dim=(-2, -1), spacing=1/x.size(-1)) 
        agg = temp**2 + z0**2 + z1**2
        yg1, yg2 = torch.gradient(y, dim=(-2, -1), spacing=1/x.size(-1)) 
        aggy = y**2 + yg1**2 + yg2**2
        loss = torch.sum(torch.sqrt(torch.sum(agg, dim=(-2, -1)))/torch.sqrt(torch.sum(aggy, dim=(-2, -1))))  
        return loss, torch.zeros(1), torch.zeros(x.size(0)), torch.zeros(x.size(0))
        

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True, truncation=True, res=256, return_freq=True, return_l2=True, relative=True):
        super().__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average
        self.relative = relative
        self.res = res
        self.return_freq = return_freq
        self.return_l2 = return_l2
        if a == None:
            a = [1,] * k
        self.a = a
        k_x = torch.cat((torch.arange(start=0, end=res//2, step=1),torch.arange(start=-res//2, end=0, step=1)), 0).reshape(res,1).repeat(1,res)
        k_y = torch.cat((torch.arange(start=0, end=res//2, step=1),torch.arange(start=-res//2, end=0, step=1)), 0).reshape(1,res).repeat(res,1)
        # k_x[0,:] = 2
        # k_y[:,0] = 2
        if truncation:
            self.k_x = (torch.abs(k_x)*(torch.abs(k_x)<20)).reshape(1,res,res,1) 
            self.k_y = (torch.abs(k_y)*(torch.abs(k_y)<20)).reshape(1,res,res,1) 
        else:
            self.k_x = torch.abs(k_x).reshape(1,res,res,1) 
            self.k_y = torch.abs(k_y).reshape(1,res,res,1) 
            
    def cuda(self, device):
        self.k_x = self.k_x.to(device)
        self.k_y = self.k_y.to(device)

    def cpu(self):
        self.k_x = self.k_x.cpu()
        self.k_y = self.k_y.cpu()

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def abs(self, x, y):
        num_examples = x.size()[0]
        #Assume uniform mesh
        # h = 1.0 / (x.size()[1] - 1.0)
        all_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def __call__(self, x, y, a=None, return_l2=True):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], self.res, self.res, -1)
        y = y.view(y.shape[0], self.res, self.res, -1)

        

        x = torch.fft.fftn(x, dim=[1, 2], norm='ortho')
        y = torch.fft.fftn(y, dim=[1, 2], norm='ortho')

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (self.k_x**2 + self.k_y**2)
            if k >= 2:
                weight += a[1]**2 * (self.k_x**4 + 2*self.k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            if self.relative:
                loss = self.rel(x*weight, y*weight)
                l2loss = self.rel(x, y)
            else:
                loss = self.abs(x*weight, y*weight)
                l2loss = self.abs(x, y)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)
        
        if self.return_freq:
            return loss, l2loss, x[:, :, 0], y[:, :, 0]
        elif self.return_l2:
            return loss, l2loss
        else:
            return loss


class HsLoss_2(nn.Module):
    def __init__(self, d=2, p=2, k=1, a=None, size_average=True, reduction=True, truncation=True, res=256, relative=True):
        super().__init__()
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.relative = relative
        self.res = res
        self.truncation = truncation
        
        if a is None:
            a = [1,] * k

        # Ensure all tensors and operations are conducted on the correct device
        self.register_buffer('weight', self.calculate_weight(k, a, res, truncation))

    def calculate_weight(self, k, a, res, truncation):
        
        k_x = torch.cat((torch.arange(start=0, end=res//2, step=1),torch.arange(start=-res//2, end=0, step=1)), 0).reshape(res,1).repeat(1,res)
        k_y = torch.cat((torch.arange(start=0, end=res//2, step=1),torch.arange(start=-res//2, end=0, step=1)), 0).reshape(1,res).repeat(res,1)
                            
        if truncation:
            k_x, k_y = (torch.abs(k_x) * (torch.abs(k_x) < 20)), (torch.abs(k_y) * (torch.abs(k_y) < 20))
        else:
            k_x, k_y = torch.abs(k_x), torch.abs(k_y)

        weight = 1
        if k >= 1:
            weight += a[0]**2 * (k_x**2 + k_y**2)
        if k >= 2:
            weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
        weight = torch.sqrt(weight)

        return weight

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def abs(self, x, y):
        num_examples = x.size()[0]
        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def forward(self, x, y):
        # Adjust the shape of x and y for FFT and weight application
        x = x.view(x.shape[0], self.res, self.res, -1)
        y = y.view(y.shape[0], self.res, self.res, -1)

        x_fft = torch.fft.fftn(x, dim=[1, 2], norm='ortho')
        y_fft = torch.fft.fftn(y, dim=[1, 2], norm='ortho')
        
        # Apply weights
        x_weighted = x_fft * self.weight
        y_weighted = y_fft * self.weight
        
        # Compute the loss
        if self.relative:
            loss = self.rel(x_weighted, y_weighted)
        else:
            loss = self.abs(x_weighted, y_weighted)
        return loss
        


# class PDEloss(nn.MSELoss):
#     def __init__(self, reduction='sum', res=256):
#         super().__init__(reduction=reduction)
#         kernel00 = torch.tensor([[[[0.,0,0],[0,1,0],[0,0,0]]]])
#         kernel01 = torch.tensor([[[[0.,-1,0],[0,0,0],[0,0,0]]]])
#         kernel10 = torch.tensor([[[[0.,0,0],[0,0,-1],[0,0,0]]]])
#         kernel0_1 = torch.tensor([[[[0.,0,0],[0,0,0],[0,-1,0]]]])
#         kernel_10 = torch.tensor([[[[0.,0,0],[-1,0,0],[0,0,0]]]])
#         # concate the kernels to be the shape of [5,1,3,3]
#         kernel = torch.cat([kernel00, kernel01, kernel10, kernel0_1, kernel_10], dim=0)
#         # initialize the weight of the conv1 to be kernel

#         self.register_buffer('kernel', kernel)
#         # register the kernels as a buffer
  
#         f = torch.tensor.ones(res, res).reshape(1, res, res)/ (res**2)
#         self.register_buffer('f', f)
    
#     def diva_fem(self, u, a):
#         # u is in shape of batch*1*res*res, a is in shape of batch*5*res*res
#         # F.conv2d(u, self.kernel, padding=1) is in shape of batch*5*res*res
#         # return the F.conv2d(u, self.kernel, padding=1) * a and sum over the channel dimension
  
#         return torch.sum(F.conv2d(u, self.kernel, padding=1) * a, dim=1, keepdim=False)
    
#     def forward(self, u, f, a):
#         loss = torch.linalg.norm(self.diva_fem(u, a)-f)
#         # loss = super().forward(self.diva_fem(u, a), f)
#         return loss 

class PDEloss(nn.MSELoss):
    def __init__(self, reduction='sum'):
        super().__init__(reduction=reduction)
        k00 = .5 * torch.tensor([[[[0.,1],[1, 2,]], [[2.,1],[1,0,]]]],)
        k01 = .5 * torch.tensor([[[[0.,1],[0, 0,]], [[1.,0],[0,0,]]]],)
        k_10 = .5 * torch.tensor([[[[0.,0],[1, 0,]], [[1.,0],[0,0,]]]],)
        k10 = .5 * torch.tensor([[[[0.,0],[0, 1,]], [[0.,1],[0,0,]]]],)
        k0_1 = .5 * torch.tensor([[[[0.,0],[0, 1,]], [[0.,0],[1,0,]]]],)
        k = torch.cat([k00, k01, k10, k0_1, k_10], dim=0)

        kernel00 = torch.tensor([[[[0.,0,0],[0,1,0],[0,0,0]]]])
        kernel01 = torch.tensor([[[[0.,-1,0],[0,0,0],[0,0,0]]]])
        kernel10 = torch.tensor([[[[0.,0,0],[0,0,-1],[0,0,0]]]])
        kernel0_1 = torch.tensor([[[[0.,0,0],[0,0,0],[0,-1,0]]]])
        kernel_10 = torch.tensor([[[[0.,0,0],[-1,0,0],[0,0,0]]]])
        kernel = torch.cat([kernel00, kernel01, kernel10, kernel0_1, kernel_10], dim=0)

        avg_a_kernl = 1./3 * torch.tensor([[[[1.,1],[1,0,]]], [[[0.,1],[1,1,]]]],)

        # register the kernels as a buffer
        self.register_buffer('k', k)
    
        self.register_buffer('kernel', kernel)
       
        self.register_buffer('avg_a_kernl', avg_a_kernl)

    
    def diva_fem(self, u, a):
 
        result = torch.sum(F.conv2d(a, self.k, padding=0) * F.conv2d(u, self.kernel, padding=1), dim=1, keepdim=False)
        return result

    
    def forward(self, u, f, a):
        B = u.size(0)
        res = (self.diva_fem(u, a)-f).view(B, -1)
        # loss = torch.sum(torch.linalg.norm(res, dim=1))
        loss = torch.linalg.norm(self.diva_fem(u, a)-f)
        # loss = super().forward(self.diva_fem(u, a), f)
        return loss

class PDEloss_2(nn.MSELoss):

    #def create_rhs(N, h, s):
    """Create the right-hand side of the linear system."""
    # x = np.linspace(0, 1, N)
    # y = np.linspace(0, 1, N)
    # X, Y = np.meshgrid(x, y)
    # rhs = -s * np.sin(np.pi * X) * np.sin(np.pi * Y)
    # return rhs.flatten()
    def __init__(self, reduction='sum', N=63, s=1600):
        super().__init__(reduction=reduction)
        self.h = 1.0 / (N + 1)
        kernel = torch.tensor([[[[0.,-1,0],[-1,4,-1],[0,-1,0]]]])/self.h**2
        self.register_buffer('kernel', kernel)
        # register the rhs as a buffer
        
        x = torch.linspace(0, 1, N)
        y = torch.linspace(0, 1, N)
        X, Y = torch.meshgrid(x, y)
        rhs = -s * torch.sin(np.pi * X) * torch.sin(np.pi * Y)
        self.register_buffer('rhs', rhs.reshape(N, N))
    
    def laplace_minus_u(self, u_current, du):
        
        return torch.nn.functional.conv2d(du, self.kernel, padding=1) - 2*du*u_current

    def getGrad(self, u_current):

        return self.rhs + u_current**2 - torch.nn.functional.conv2d(u_current, self.kernel, padding=1)

    def forward(self, u_current, du):
        f = self.getGrad(u_current)
        # f = fixed_rhs + u_flat**2 - A.dot(u_flat)
        # loss = self.h * torch.linalg.norm((self.laplace_minus_u(u_current, du)-f).view(f.size(0), -1), dim=1)
        loss = self.h**2 * super().forward(self.laplace_minus_u(u_current, du).view(f.size(0), -1), f.view(f.size(0), -1))
        return torch.sqrt(loss)

class PDEloss_single(nn.MSELoss):

    #def create_rhs(N, h, s):
    """Create the right-hand side of the linear system."""
    # x = np.linspace(0, 1, N)
    # y = np.linspace(0, 1, N)
    # X, Y = np.meshgrid(x, y)
    # rhs = -s * np.sin(np.pi * X) * np.sin(np.pi * Y)
    # return rhs.flatten()
    def __init__(self, reduction='sum', N=63, s=100):
        super().__init__(reduction=reduction)
        self.h = 1.0 / (N + 1)
        kernel = torch.tensor([[[[0.,-1,0],[-1,4,-1],[0,-1,0]]]])/self.h**2
        self.register_buffer('kernel', kernel)
        # register the rhs as a buffer
        
        x = torch.linspace(0, 1, N)
        y = torch.linspace(0, 1, N)
        X, Y = torch.meshgrid(x, y)
        rhs = -s * torch.sin(np.pi * X) * torch.sin(np.pi * Y)
        self.register_buffer('rhs', rhs.reshape(N, N))
    
    def laplace_plus_u(self, u_current, du):
        
        return torch.nn.functional.conv2d(du, self.kernel, padding=1) + 2*du*u_current

    def getGrad(self, u_current):

        return self.rhs - u_current**2 - torch.nn.functional.conv2d(u_current, self.kernel, padding=1)

    def forward(self, u_current, du):
        f = self.getGrad(u_current)
        # f = fixed_rhs + u_flat**2 - A.dot(u_flat)
        # loss = self.h * torch.linalg.norm((self.laplace_minus_u(u_current, du)-f).view(f.size(0), -1), dim=1)
        loss = self.h**2 * super().forward(self.laplace_plus_u(u_current, du).view(f.size(0), -1), f.view(f.size(0), -1))
        return torch.sqrt(loss)

class PDEloss_GrayScott(nn.Module):
    def __init__(self, DA=2.5e-4, DS=5e-4, mu=0.065, rho=0.04, N=63):
        super().__init__()
        self.h = 1.0 / (N - 1)
        self.DA = DA
        self.DS = DS
        self.mu = mu
        self.rho = rho
        kernel = torch.tensor([[[[0., -1, 0], [-1, 4, -1], [0, -1, 0]]]]) / self.h**2
        # self.register_buffer('laplacian_kernel', kernel)
        self.lap = nn.Conv2d(1, 1, 3, padding=1, bias=False, padding_mode='replicate')
        self.lap.weight = nn.Parameter(kernel)
        self.lap.weight.requires_grad = False

    def forward(self, A, S, delta_A, delta_S):
        # Apply Laplacian using convolution
        lap_A = self.lap(A)
        lap_S = self.lap(S)

        # Compute residuals for A and S
        F_A = self.DA * lap_A - S * A**2 + (self.mu + self.rho) * A
        F_S = self.DS * lap_S + S * A**2 - self.rho * (1 - S)
        
        # Apply Laplacian to delta_A and delta_S
        lap_delta_A = self.lap(delta_A)
        lap_delta_S = self.lap(delta_S)
        
        # Compute J_A_delta_A, J_S_delta_S, J_A_delta_S, and J_S_delta_A
        J_AA_delta_A = self.DA * lap_delta_A - 2 * delta_A * S * A + (self.mu + self.rho) * delta_A
        J_SS_delta_S = self.DS * lap_delta_S + (A**2 + self.rho) * delta_S 
        J_AS_delta_S = - A**2 * delta_S
        J_SA_delta_A = 2 * S * A * delta_A 

        loss = torch.norm(J_AA_delta_A + J_AS_delta_S + F_A) + torch.norm(J_SA_delta_A + J_SS_delta_S + F_S)
        return loss

def count_params(model):
    """Returns the number of parameters of a PyTorch model"""
    return sum([p.numel()*2 if p.is_complex() else p.numel() for p in model.parameters()])

def getPath(data, flag):
    
    if data=='ns_merge':
        if flag=='xtrain':
            PATH = '/ibex/ai/home/liux0t/ns_merge/x_train_dataset.pt'
        elif flag=='ytrain':
            PATH = '/ibex/ai/home/liux0t/ns_merge/y_train_dataset.pt'
        elif flag=='xtest':
            PATH = '/ibex/ai/home/liux0t/ns_merge/x_test_dataset.pt'
        elif flag=='ytest':
            PATH = '/ibex/ai/home/liux0t/ns_merge/y_test_dataset.pt'

    elif data=='darcy':
        if flag=='train':
            PATH = os.path.join('/ibex/ai/home/liux0t/Xinliang/FMM/','data/piececonst_r421_N1024_smooth1.mat')
        else:
            PATH = os.path.join('/ibex/ai/home/liux0t/Xinliang/FMM/', 'data/piececonst_r421_N1024_smooth2.mat')
    elif data=='darcy20':
        # for ray tune
        TRAIN_PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/darcy_alpha2_tau5_512_train.mat'
        TEST_PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/darcy_alpha2_tau5_512_test.mat'
    elif data == 'darcy_contin':
        if flag == 'train':
            PATH = '/home/liux0t/FMM/experiments/darcy_contin_1_5_alpha2.2.mat'
        elif flag == 'test':
            PATH = '/home/liux0t/FMM/experiments/darcy_contin_1_5_alpha2.1.mat'
        elif flag == 'jin':
            PATH = '/home/liux0t/FMM/experiments/test.mat'
        else:
            PATH = '/home/liux0t/FMM/experiments/darcy_contin_1_5_sin.mat'
        
    elif data=='darcy20c6':
        # for ray tune
        if flag=='train':
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/darcy_alpha2_tau5_512_train.mat' 
        elif flag=='test':
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/darcy_alpha2_tau5_512_test.mat'
        elif flag=='val':
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/darcy_alpha2_tau5_512_train.mat' 
        elif flag=='gel':
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/darcy_alpha2_tau18_c3_512_test.mat'
        else: raise NameError('invalid flag name')
        
    elif data=='darcy20c6_c3':
        TRAIN_PATH = os.path.join(os.path.abspath(''), 'darcy_alpha2_tau5_512_train.mat')
        TEST_PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/darcy_alpha2_tau18_c3_512_test.mat'
    elif data=='darcy15c10':
        TRAIN_PATH = os.path.join(os.path.abspath(''), 'darcy_alpha2_tau15_c10_512_train.mat')
        TEST_PATH = os.path.join(os.path.abspath(''), 'darcy_alpha2_tau15_c10_512_test.mat')
    elif data=='a3f2':
        TRAIN_PATH = os.path.join(os.path.abspath(''), 'data/mul_res1023_a3f2_train.mat')
        TEST_PATH = os.path.join(os.path.abspath(''), 'data/mul_res1023_a3f2_test.mat')
    elif data=='a4f3':
        if flag=='train':
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/data/mul_res1023_a4f3_train.mat'
        else:
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/data/mul_res1023_a4f3_test.mat'
    elif data=='a4f1':
        if flag=='train':
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/data/mul_tri_train.mat'
        else:
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/data/mul_tri_test.mat'
    elif data=='checker':
        TRAIN_PATH = '/home/xubo/multiscale-attention/data/mul_res1023_a7f1m32_train.mat'
        TEST_PATH = '/home/xubo/multiscale-attention/data/mul_res1023_a7f1m32_test.mat'
    elif data=='checkerm4':
        TRAIN_PATH = '/home/xubo/multiscale-attention/data/mul_res1023_a7f1m4_train.mat'
        TEST_PATH = '/home/xubo/multiscale-attention/data/mul_res1023_a7f1m4_test.mat'
    elif data=='darcyF':
        if flag in ['train', 'val']:
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/data/darcy_alpha2_tau9_512_F_train.mat'
        else:
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/data/darcy_alpha2_tau9_512_F_test.mat'   
    elif data=='darcyF2':
        TRAIN_PATH = os.path.join(os.path.abspath(''), 'darcy_alpha2_tau9_512_F_train.mat')
        TEST_PATH = os.path.join(os.path.abspath(''), 'darcy_alpha2_tau9_512_F_test.mat')    
    elif data=='burgers':
        TRAIN_PATH = os.path.join(os.path.abspath(''), 'burgers_data_R10.mat')
    elif data=='navier':
        TRAIN_PATH = '/home/liux0t/FMM/data/ns_V1e-4_N10000_T30.mat'
        TEST_PATH = '/home/liux0t/FMM/data/ns_V1e-4_N10000_T30.mat'
    elif data=='helmholtz':
        if flag=='train':
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/data/Hel_train.mat'
        else:
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/data/Hel_test.mat'
    elif data=='helm':
        if flag=='x':
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/data/Helmholtz_inputs.npy'
        else:
            PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/data/Helmholtz_outputs.npy'
    elif data=='1e-5':
        PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/data/NavierStokes_V1e-5_N1200_T20.mat'
    elif data=='1e-4':
        PATH = '/ibex/ai/home/liux0t/Xinliang/FMM/data/ns_V1e-4_N10000_T30.mat'

    else: raise NameError('invalid data name')
    
    return PATH

def getDataSize(dataOpt):
    if dataOpt['data'] == 'darcy':
        dataOpt['dataSize'] = {'train': range(1000), 'test': range(100), 'val':range(100, 200)}
    elif dataOpt['data'] == 'darcy20c6':
        dataOpt['dataSize'] = {'train': range(1280), 'test': range(112), 'val':range(1280, 1280+112)}
    elif dataOpt['data'] == 'darcyF':
        dataOpt['dataSize'] = {'train': range(1000), 'test': range(100), 'val':range(100, 200)}
    elif dataOpt['data'] == "a4f1":
        dataOpt['dataSize'] = {'train': range(1000), 'val': range(100), 'test': range(100)} #, 'val':112
    elif dataOpt['data'] == 'darcy_contin':
        dataOpt['dataSize'] = {'train': range(2000), 'test': range(112), 'val':range(10)}
    elif dataOpt['data'] == 'helm':
        dataOpt['dataSize'] = {'train': range(4000), 'test': range(8000, 8000+200), 'val': range(8000+200, 8000+400)} #, 'val':112
    elif dataOpt['data'] == 'pipe':
        dataOpt['dataSize'] = {'train': range(1000), 'test': range(200), 'val':range(200, 400)}
    elif dataOpt['data'] == 'airfoil':
        dataOpt['dataSize'] = {'train': range(1000), 'test': range(200), 'val':range(200, 400)}
    elif dataOpt['data'] == 'ns_merge':
        dataOpt['dataSize'] = {'train': range(50000), 'test': range(50000, 50000+1000)}
    else:
        raise NameError('dataset not exist')
    return dataOpt

def getNS_merge_Dataset(dataOpt, return_normalizer=True, normalizer_type='GN'):
    PATH_X = getPath(dataOpt['data'], 'xtrain')
    PATH_Y = getPath(dataOpt['data'], 'ytrain')
    x = torch.load(PATH_X)
    y = torch.load(PATH_Y)
    GN = dataOpt['GN']
    if 'normalizer_type' in dataOpt:
        normalizer_type = dataOpt['normalizer_type']
    

    train_idx = dataOpt['dataSize']['train']
    test_idx = dataOpt['dataSize']['test']

    x_train = x[train_idx,...]
    x_test = x[test_idx,...]
    y_train = y[train_idx,...]
    y_test = y[test_idx,...]


    if return_normalizer:
        if normalizer_type=='PGN':
            x_normalizer = UnitGaussianNormalizer(x_train)
            y_normalizer = UnitGaussianNormalizer(y_train)
        else:
            x_normalizer = GaussianNormalizer(x_train)
            y_normalizer = GaussianNormalizer(y_train)
        if GN:        
            x_train = x_normalizer.encode(x_train)
            x_test = x_normalizer.encode(x_test)
    
        return x_train, y_train, x_test, y_test,  x_normalizer, y_normalizer

    return x_train, y_train, x_test, y_test

def getDarcyDataSet(dataOpt, flag, 
return_normalizer=False, normalizer_type='PGN', normalizer=None):
    PATH = getPath(dataOpt['data'], flag)
    r = dataOpt['sampling_rate']
    sample_idx = dataOpt['dataSize'][flag]
    GN = dataOpt['GN']
    if 'normalizer_type' in dataOpt:
        normalizer_type = dataOpt['normalizer_type']

    reader = MatReader(PATH)
    if dataOpt['sample_x']:
        x = reader.read_field('coeff')[sample_idx,::r,::r]
    else:
        x = reader.read_field('coeff')[sample_idx,...]
    y = reader.read_field('sol')[sample_idx,::r,::r]
    
    if return_normalizer:
        if normalizer_type=='PGN':
            x_normalizer = UnitGaussianNormalizer(x)
            y_normalizer = UnitGaussianNormalizer(y)
        else:
            x_normalizer = GaussianNormalizer(x)
            y_normalizer = GaussianNormalizer(y)
        if GN:        
            x = x_normalizer.encode(x)
            return x, y, x_normalizer, y_normalizer
        else:
            return x, y, x_normalizer, y_normalizer
    else:
        if GN:
            if normalizer is None:
                raise NameError('No normalizer')
            else:
                return normalizer.encode(x), y
        else:
            return x, y

def getHelmDataset(dataOpt, return_normalizer=True, normalizer_type='PGN'):

    PATH_X = getPath(dataOpt['data'], 'x')
    PATH_Y = getPath(dataOpt['data'], 'y')
    x = np.load(PATH_X)
    x = np.transpose(x, axes=[2, 0, 1])
    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    y = np.load(PATH_Y)
    y = np.transpose(y, axes=[2, 0, 1])
    y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    GN = dataOpt['GN']
    if 'normalizer_type' in dataOpt:
        normalizer_type = dataOpt['normalizer_type']
    
    r = dataOpt['sampling_rate']
    train_idx = dataOpt['dataSize']['train']
    test_idx = dataOpt['dataSize']['test']
    val_idx = dataOpt['dataSize']['val']
    x_train = x[train_idx,...]
    x_test = x[test_idx,...]
    y_train = y[train_idx,...]
    y_test = y[test_idx,...]
    x_val = x[val_idx,...]
    y_val = y[val_idx,...]

    if return_normalizer:
        if normalizer_type=='PGN':
            x_normalizer = UnitGaussianNormalizer(x_train)
            y_normalizer = UnitGaussianNormalizer(y_train)
        else:
            x_normalizer = GaussianNormalizer(x_train)
            y_normalizer = GaussianNormalizer(y_train)
        if GN:        
            x_train = x_normalizer.encode(x_train)
            x_test = x_normalizer.encode(x_test)
            x_val = x_normalizer.encode(x_val)
    
        return x_train, y_train, x_test, y_test, x_val, y_val, x_normalizer, y_normalizer

    return x_train, y_train, x_test, y_test

def getPipeDataset(dataOpt):
    INPUT_X = '/ibex/ai/home/liux0t/Xinliang/FMM/Pipe_X.npy'
    INPUT_Y = '/ibex/ai/home/liux0t/Xinliang/FMM/Pipe_Y.npy'
    OUTPUT_Sigma = '/ibex/ai/home/liux0t/Xinliang/FMM/Pipe_Q.npy'

    ntrain = 1000
    ntest = 200
    N = 1200
    r1 = 1
    r2 = 1
    s1 = int(((129 - 1) / r1) )
    s2 = int(((129 - 1) / r2) )

    ################################################################
    # load data and data normalization
    ################################################################
    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float)
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float)
    input = torch.stack([inputX, inputY], dim=-1)

    output = np.load(OUTPUT_Sigma)[:, 0]
    output = torch.tensor(output, dtype=torch.float)

    x_train = input[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = input[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
    x_val = input[:N][ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]
    y_val = output[:N][ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]
    x_train = x_train.reshape(ntrain, s1, s2, 2)
    x_test = x_test.reshape(ntest, s1, s2, 2)
    x_val = x_val.reshape(ntest, s1, s2, 2)
    x_train = x_train.permute(0, 3, 1, 2).contiguous()
    x_test = x_test.permute(0, 3, 1, 2).contiguous()  
    x_val = x_val.permute(0, 3, 1, 2).contiguous()
    return x_train, y_train, x_test, y_test, x_val, y_val

def getNavierDataSet(dataPath, r, ntrain, ntest, T_in, T, device, T_out=None, return_normalizer=False, GN=False, normalizer=None, full_train=False):

    if not T_out:
        T_out = T_in
    reader = MatReader(dataPath)
    temp = reader.read_field('u').to(device)
    if full_train:
        train_a = temp[:ntrain,::r,::r,:T_in+T]
    else:
        train_a = temp[:ntrain,::r,::r,:T_in]
    train_u = temp[:ntrain,::r,::r,T_out:T+T_out]


    test_a = temp[-ntest:,::r,::r,:T_in]
    test_u = temp[-ntest:,::r,::r,T_out:T+T_out]
    
    if return_normalizer:
        x_normalizer = UnitGaussianNormalizer(train_a)
        y_normalizer = UnitGaussianNormalizer(train_u)
        if GN:        
            train_a = x_normalizer.encode(train_a)
            test_a = x_normalizer.encode(test_a)
            # train_u = y_normalizer.encode(train_u)
            # test_u = y_normalizer.encode(test_u)
            return train_a, train_u, test_a, test_u, x_normalizer, y_normalizer
        else:
            return train_a, train_u, test_a, test_u, x_normalizer, y_normalizer
    else:
        if GN:
            if normalizer is None:
                raise NameError('No normalizer')
            else:
                return normalizer.decode(train_a), train_u, normalizer.decode(test_a), test_u
        else:
            return train_a, train_u, test_a, test_u

def getNavierDataSet2(opt, device, return_normalizer=False, GN=False, normalizer=None):
    dataPath, r, ntrain, ntest, T_in, T_out, T = opt['path'], opt['sampling'], opt['ntrain'], opt['ntest'], opt['T_in'], opt['T_out'], opt['T']

    reader = MatReader(dataPath)
    temp = reader.read_field('u').to(device)
    if opt['full_train']:
        train_a = temp[:ntrain,::r,::r,T_out-T_in:T_out+T]
    else:
        train_a = temp[:ntrain,::r,::r, T_out-T_in:T_out]
    train_u = temp[:ntrain,::r,::r,T_out:T+T_out]


    test_a = temp[-ntest:,::r,::r,T_out-T_in:T_out]
    test_u = temp[-ntest:,::r,::r,T_out:T+T_out]

    print(train_u.shape)
    print(test_u.shape)
    assert (opt['r'] == train_u.shape[-2])
 

    
    if return_normalizer:
        x_normalizer = GaussianImageNormalizer(train_a)
        y_normalizer = GaussianImageNormalizer(train_u)
        if GN:        
            train_a = x_normalizer.encode(train_a)
            test_a = x_normalizer.encode(test_a)
            train_u = y_normalizer.encode(train_u)
            test_u = y_normalizer.encode(test_u)
            return train_a, train_u, test_a, test_u, x_normalizer, y_normalizer
        else:
            return train_a, train_u, test_a, test_u, x_normalizer, y_normalizer
    else:
        if GN:
            if normalizer is None:
                raise NameError('No normalizer')
            else:
                return normalizer.decode(train_a), train_u, normalizer.decode(test_a), test_u
        else:
            return train_a, train_u, test_a, test_u

def getNavierDataSet3(opt, device, return_normalizer=False, GN=False, normalizer=None):
    
    dataPath, r, ntrain, ntest, T_in, T_out, T = opt['path'], opt['sampling'], opt['ntrain'], opt['ntest'], opt['T_in'], opt['T_out'], opt['T']

    reader = MatReader(dataPath)
    temp = reader.read_field('u').to(device)
    # train_a should be a slice of temp in terms of time window [0: T_in+1] and than [1: T_in+2] and so on till [T: T_in+T+1]
    # cacatenate all these slices in the first axis to get train_a
    # s is the start time index of the slice
   
    train_a = temp[:ntrain,::r,::r,T_out-T_in:T_out]
    train_u = temp[:ntrain,::r,::r,T_out:T_out+1]
    assert(opt['full_train_2'] is True)

    for i in range(1, T):
        train_a = torch.cat((train_a, temp[:ntrain,::r,::r,T_out-T_in+i:T_out+i]), dim=0)
        train_u = torch.cat((train_u, temp[:ntrain,::r,::r,T_out+i:T_out+i+1]), dim=0)


    test_a = temp[-ntest:,::r,::r,T_out-T_in:T_out]
    test_u = temp[-ntest:,::r,::r,T_out:T+T_out]
    print(train_a.shape)
    print(train_u.shape)
    print(test_a.shape)
    print(test_u.shape)
    assert (opt['r'] == train_u.shape[-2])
 
    if return_normalizer:
        x_normalizer = GaussianImageNormalizer(train_a)
        y_normalizer = GaussianImageNormalizer(train_u)
        if GN:        
            train_a = x_normalizer.encode(train_a)
            test_a = x_normalizer.encode(test_a)
            train_u = y_normalizer.encode(train_u)
            test_u = y_normalizer.encode(test_u)
            return train_a, train_u, test_a, test_u, x_normalizer, y_normalizer
        else:
            return train_a, train_u, test_a, test_u, x_normalizer, y_normalizer
    else:
        if GN:
            if normalizer is None:
                raise NameError('No normalizer')
            else:
                return normalizer.decode(train_a), train_u, normalizer.decode(test_a), test_u
        else:
            return train_a, train_u, test_a, test_u





def getOptimizerScheduler(parameters, epochs, optimizer_type='adam', lr=0.001,
 weight_decay=1e-4, final_div_factor=1e1, div_factor=1e1):
    if optimizer_type == 'sgd':
        optimizer =  torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        optimizer =  torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adagrad':
        optimizer =  torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer =  Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamax':
        optimizer =  torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer =  torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                               div_factor=div_factor, 
                               final_div_factor=final_div_factor,
                               pct_start=0.1,
                               steps_per_epoch=1, 
                               epochs=epochs)
    return optimizer, scheduler
        

def getNavierDataLoader(dataPath, r, ntrain, ntest, T_in, T, batch_size, device, model_name='vFMM', return_normalizer=False, GN=False, normalizer=None):
    train_a, train_u, test_a, test_u = getNavierDataSet(dataPath, r, ntrain, ntest, T_in, T, device, return_normalizer, GN, normalizer)
    if model_name=='vFMM':       
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a.permute(0, 3, 1, 2), train_u.permute(0, 3, 1, 2)), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a.permute(0, 3, 1, 2), test_u.permute(0, 3, 1, 2)), batch_size=batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def getSavePath(data, model_name, flag='log'):
    if flag=='log':
        MODEL_PATH = os.path.join(os.path.abspath(''), 'model/' + model_name + data + str(datetime.now()) + '.log')
    elif flag=='para':
        MODEL_PATH = os.path.join(os.path.abspath(''), 'model/' + model_name + data + str(datetime.now()) + '.pt')
    else:
        raise NameError('invalid path flag')
    return MODEL_PATH

def save_pickle(var, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(var, f)
        



def get_initializer(name):
    if name == 'xavier_normal':
        init_ = partial(nn.init.xavier_normal_)
    elif name == 'kaiming_uniform':
        init_ = partial(nn.init.kaiming_uniform_)
    elif name == 'kaiming_normal':
        init_ = partial(nn.init.kaiming_normal_)
    return init_

def visual2d(u, title):
    # Create meshgrid for plotting
    N = u.shape[0]
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    # Contour Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, u, 20, cmap='viridis')
    plt.colorbar()
    plt.title('Contour Plot of the Solution')
    plt.xlabel('x')
    plt.ylabel('y')

    # Surface Plot
    ax = plt.subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(X, Y, u, cmap='viridis', edgecolor='none')
    plt.colorbar(surf)
    ax.set_title('Surface Plot of the Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')

    plt.tight_layout()
    plt.show()
    plt.savefig(title + '.png')