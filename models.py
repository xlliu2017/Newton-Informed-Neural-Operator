import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchinfo import summary
from utilities3 import count_params


class GELU_2(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.norm = nn.GroupNorm(4, A.out_channels, affine=True)
    def forward(self, out):
        # return self.norm(F.gelu(out[0])), out[1]
        return F.gelu(self.norm(out[0])), out[1]
       
class Conv2dAttn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels*4, kernel_size, stride, padding, bias=bias, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(4, out_channels, affine=True)
        self.m = nn.Softmax(dim=1)
    def forward(self, x):
        qkv = self.conv(x)
        qkv = qkv.view(qkv.shape[0], 2, 2, self.out_channels, qkv.shape[2], qkv.shape[3])
        # q = qkv[:,:,0,...]
        k = qkv[:,0,...]
        v = qkv[:,1,...]
        # out1 = torch.einsum('ijlm, ijklm, ijklm -> ijlm', x, k, v)
        # xk = torch.einsum('iklm, ijklm -> ijlm', x, k)/400
        xk = self.m(torch.einsum('iklm, ijklm -> ijlm', x, k))
        out = torch.einsum('ijlm, ijklm -> iklm', xk, v)
        # out = self.norm(out)
        # assert torch.allclose(out1, out3)
        return out

class MgIte(nn.Module):
    def __init__(self, A, S):
        super().__init__()
 
        self.A = A
        self.S = S
        in_chans = self.A.in_channels
        out_channels = self.A.out_channels
        # self.norm1 = nn.GroupNorm(2, in_chans, affine=True)
        # self.norm2 = nn.GroupNorm(1, out_channels, affine=True)
    def forward(self, out):
        
        if isinstance(out, tuple):
            u, f = out
            # u = u + (self.S(F.gelu(self.norm2(f-self.A(F.gelu(self.norm1(u)))))))  
            # u = u + (self.S(F.gelu(f-self.A(u)))) 
            u = u + self.S(f-self.A(u)) 
        else:
            f = out
            # u = self.S(F.gelu(f))
            u = self.S(f)
        out = (u, f)
        return out

class MgIte_2(nn.Module):
    def __init__(self, A, S):
        super().__init__()
 
        self.A = A
        # self.S = S
        # in_chans = self.S.in_channels
        # out_channels = self.S.out_channels
        # self.norm = nn.GroupNorm(2, out_channels, affine=True)

    def forward(self, out):
        
        if isinstance(out, tuple):
            u, f = out
            # u = u + (f-self.A(u)) this is for MgNO_DC for attn
            u = self.A(u)
        else:
            f = out
            u = self.S(f)

        out = (u, f)
        return out

class MgIte_3(nn.Module):
    def __init__(self, A, S):
        super().__init__()
 
        self.A = A
        self.S = S
        in_chans = self.A.in_channels
        out_channels = self.A.out_channels
        self.norm1 = nn.GroupNorm(2, in_chans, affine=True)
        self.norm2 = nn.GroupNorm(1, out_channels, affine=True)
    def forward(self, out):
        
        if isinstance(out, tuple):
            u, f = out
            u = (self.S(F.gelu(self.norm2(f-self.A(F.gelu(self.norm1(u)))))))  
            # u = self.A(F.gelu(self.norm1(u)))
        else:
            f = out
            # u = self.S(F.gelu(f))
            u = self.S(f)
        out = (u, f)
        return out

class MgIte_init(nn.Module):
    def __init__(self, S):
        super().__init__()
        
        self.S = S

    def forward(self, f):
        u = self.S(f)
        return (u, f)

class Restrict(nn.Module):
    def __init__(self, Pi=None, R=None, A=None):
        super().__init__()
        self.Pi = Pi
        self.R = R
        self.A = A
    def forward(self, out):
        u, f = out
        if self.A is not None:
            f = self.R(f-self.A(u))
        else:
            f = self.R(f)
        u = self.Pi(u)                              
        out = (u,f)
        return out

class MgConv(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=False, use_res=False, elementwise_affine=True):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.norm_layer_list = nn.ModuleList([])
        for j in range(len(num_iteration)-1):
            self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 128//2**j, 128//2**j], elementwise_affine=elementwise_affine))

        self.RTlayers = nn.ModuleList()
        for j in range(len(num_iteration)-1):
            self.RTlayers.append(nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=4, stride=2, padding=1, bias=False))

        layers = []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
            post_smooth_layers = []
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                if l==0 and i==0:
                    layers.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    layers.append(MgIte(A, S))
            if not num_iteration_l[1] == 0:
                for i in range(num_iteration_l[1]):
                    S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer'+str(l), nn.Sequential(*post_smooth_layers))
            if l < len(num_iteration)-1:
                A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                Pi= nn.Conv2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False, padding_mode=padding_mode)
                R = nn.Conv2d(num_channel_f, num_channel_f, kernel_size=3, stride=2, padding=1, bias=False, padding_mode=padding_mode)
                if use_res:
                    layers= [Restrict(Pi, R, A)]
                else:
                    layers= [Restrict(Pi=Pi, R=R)]
        
    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f 

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out) 
            out_list[l] = out
        # upblock                                 
        for j in range(len(self.num_iteration)-2,-1,-1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = self.norm_layer_list[j](u + self.RTlayers[j](out_list[j+1][0]))
            out = (u_post, f)
            out_list[j] = getattr(self, 'post_smooth_layer'+str(j))(out) 
            
        return out_list[0][0]

class MgNO_NS(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=3,  normalizer=None, mlp_hidden_dim=0, output_dim=1, activation='gelu', padding_mode='circular', bias=False, use_res=False, elementwise_affine=True):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList([])  
        self.conv_list.append(MgConv(num_iteration, num_channel_u, num_channel_f, padding_mode=padding_mode, bias=bias, use_res=use_res, elementwise_affine=elementwise_affine))    
        for _ in range(num_layer-1):
            self.conv_list.append(MgConv(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode, bias=bias, use_res=use_res, elementwise_affine=elementwise_affine)) 
        if mlp_hidden_dim:
            self.last_layer = nn.Sequential(nn.Conv2d(num_channel_u, mlp_hidden_dim, kernel_size=1),
            nn.GELU(), nn.Conv2d(mlp_hidden_dim, output_dim, kernel_size=1, bias=False))
        else:
            self.last_layer = nn.Conv2d(num_channel_u, 1, kernel_size=1, stride=1, padding=0, bias=False)

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
        u_0 = u
        for i in range(self.num_layer):
            u = self.act((self.conv_list[i](u))) #+ self.linear_list[i](u))   u = self.act(self.norm_layer_list[i](self.conv_list[i](u))) #+ self.linear_list[i](u))
        u = self.last_layer(u)
        return u + u_0


class MgConv_DC(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=False, use_res=False, ):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.RTlayers = nn.ModuleList()
        for j in range(len(num_iteration)-1):
            self.RTlayers.append(nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=4, stride=2, padding=1, bias=bias))

        layers = []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
            post_smooth_layers = []
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                if l==0 and i==0:
                    layers.append(MgIte_init(S))
                else:
                    if num_channel_f==num_channel_u:
                        S = Conv2dAttn(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, )
                        A = Conv2dAttn(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, )
                        layers.append(MgIte_2(A, S))
                    else:
                        A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    layers.append(MgIte(A, S))
            if not num_iteration_l[1] == 0:
                for i in range(num_iteration_l[1]):
                    S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer'+str(l), nn.Sequential(*post_smooth_layers))
            if l < len(num_iteration)-1:
                A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                Pi= nn.Conv2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=bias, padding_mode=padding_mode)
                R = nn.Conv2d(num_channel_f, num_channel_f, kernel_size=3, stride=2, padding=1, bias=bias, padding_mode=padding_mode)
                if use_res:
                    layers= [Restrict(Pi, R, A)]
                else:
                    layers= [Restrict(Pi=Pi, R=R)]
        
    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f 

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out) 
            out_list[l] = out
        # upblock                                 
        for j in range(len(self.num_iteration)-2,-1,-1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = u + self.RTlayers[j](out_list[j+1][0])
            out = (u_post, f)
            out_list[j] = getattr(self, 'post_smooth_layer'+str(j))(out) 
            
        return out_list[0][0]

class MgNO_DC(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=1,  normalizer=None, output_dim=1, activation='gelu', padding_mode='zeros', last_layer='conv', linear_layer=False):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration
        self.linear_layer = linear_layer
        self.conv_list = nn.ModuleList([])
        self.linear_list = nn.ModuleList([])
        if linear_layer:
            self.linear_list.append(nn.Conv2d(num_channel_f, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))   
        self.conv_list.append(MgConv_DC(num_iteration, num_channel_u, num_channel_f, padding_mode=padding_mode))   
        for _ in range(num_layer-1):
            self.conv_list.append(MgConv_DC(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode)) 
            self.linear_list.append(nn.Conv2d(num_channel_u, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))
   
        if last_layer == 'conv':
            self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=3, padding=1, padding_mode=padding_mode)
        elif last_layer == 'linear':
            self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        else:
            raise NameError('invalid last_layer')
        # self.norm_list = nn.ModuleList([])
        # for _ in range(num_layer):
        #     self.norm_list.append(nn.GroupNorm(1, num_channel_u, affine=True))
        self.normalizer = normalizer

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
        if self.linear_layer:
            for i in range(self.num_layer):
                u = self.act((self.conv_list[i](u) + self.linear_list[i](u)))
        else:
            for i in range(self.num_layer):
                u = self.act(self.conv_list[i](u))
            
        u = self.normalizer.decode(self.linear(u)).squeeze() if self.normalizer else self.linear(u).squeeze()
        return u 

class MgConv_DC_5(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=False, use_res=False, ):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.RTlayers = nn.ModuleList()
        for j in range(len(num_iteration)-1):
            self.RTlayers.append(nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=0, bias=bias))

        layers = []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
            post_smooth_layers = []
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                if l==0 and i==0:
                    layers.append(MgIte_init(S))
                else:
                    if num_channel_f==num_channel_u:
                        S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, )
                        A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, )
                        layers.append(MgIte_2(A, S))
                    else:
                        A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    layers.append(MgIte(A, S))
            if not num_iteration_l[1] == 0:
                for i in range(num_iteration_l[1]):
                    S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer'+str(l), nn.Sequential(*post_smooth_layers))
            if l < len(num_iteration)-1:
                A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                Pi= nn.Conv2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=0, bias=bias, padding_mode=padding_mode)
                R = nn.Conv2d(num_channel_f, num_channel_f, kernel_size=3, stride=2, padding=0, bias=bias, padding_mode=padding_mode)
                if use_res:
                    layers= [Restrict(Pi, R, A)]
                else:
                    layers= [Restrict(Pi=Pi, R=R)]
        
    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f 

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out) 
            out_list[l] = out
        # upblock                                 
        for j in range(len(self.num_iteration)-2,-1,-1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = u + self.RTlayers[j](out_list[j+1][0])
            out = (u_post, f)
            out_list[j] = getattr(self, 'post_smooth_layer'+str(j))(out) 
            
        return out_list[0][0]

class MgNO_DC_5(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=1,  normalizer=None, output_dim=1, activation='gelu', padding_mode='zeros', last_layer='conv', linear_layer=False):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration
        self.linear_layer = linear_layer
        self.conv_list = nn.ModuleList([])
        self.linear_list = nn.ModuleList([])
        if linear_layer:
            self.linear_list.append(nn.Conv2d(num_channel_f, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))   
        self.conv_list.append(MgConv_DC_5(num_iteration, num_channel_u, num_channel_f, padding_mode=padding_mode))   
        for _ in range(num_layer-1):
            self.conv_list.append(MgConv_DC_5(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode)) 
            self.linear_list.append(nn.Conv2d(num_channel_u, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))
   
        if last_layer == 'conv':
            self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=3, padding=1, padding_mode=padding_mode)
        elif last_layer == 'linear':
            self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        else:
            raise NameError('invalid last_layer')
        self.norm_list = nn.ModuleList([])
        for _ in range(num_layer):
            self.norm_list.append(nn.GroupNorm(1, num_channel_u, affine=True))
        self.normalizer = normalizer

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
        if self.linear_layer:
            for i in range(self.num_layer):
                u = self.act((self.conv_list[i](u) + self.linear_list[i](u)))
        else:
            for i in range(self.num_layer):
                u = self.act(self.norm_list[i](self.conv_list[i](u)))
            
        u = self.normalizer.decode(self.linear(u)) if self.normalizer else self.linear(u)
        return u

class MgConv_DC_6(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=False, upnorm=True):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.up_layers = []
        for j in range(len(num_iteration)-1):
            upsample_layer = nn.Sequential(
                # nn.LayerNorm([(j+2)*num_channel_u, 128//2**(j+1), 128//2**(j+1)], elementwise_affine=True) if upnorm else nn.Identity(),
                nn.GroupNorm((j+2)*2, (j+2)*num_channel_u, affine=True) if upnorm else nn.Identity(),
                nn.GELU(),
                nn.ConvTranspose2d((j+2)*num_channel_u, (j+1)*num_channel_u, kernel_size=3, stride=2, padding=0, bias=bias),  
            )
            self.up_layers.append(upsample_layer)
        self.up_layers = nn.ModuleList(self.up_layers)

        down_layers, post_smooth_layers, layer = [], [], []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
            post_smooth_layer = []
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                if l==0 and i==0:
                    layer.append(MgIte_init(S))
                else:
                    if num_channel_f==num_channel_u:
                        S = nn.Conv2d((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                        A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                        layer.append(MgIte_2(A, S))
                    else:
                        A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                        layer.append(MgIte(A, S))
            if not num_iteration_l[1] == 0:
                for i in range(num_iteration_l[1]):
                    S = nn.Conv2d((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    post_smooth_layer.append(MgIte_2(A, S))
            else:
                post_smooth_layer.append(nn.Identity())

            # setattr(self, 'layer'+str(l), nn.Sequential(*layers))
            down_layers.append(nn.Sequential(*layer))
            post_smooth_layers.append(nn.Sequential(*post_smooth_layer))
            # setattr(self, 'post_smooth_layer'+str(l), nn.Sequential(*post_smooth_layer))
            if l < len(num_iteration)-1:
                Pi = nn.Conv2d((l+1)*num_channel_u, (l+2)*num_channel_u, kernel_size=3, stride=2, padding=0, bias=bias, padding_mode=padding_mode)
                R = nn.Conv2d((l+1)*num_channel_u, (l+2)*num_channel_u, kernel_size=3, stride=2, padding=0, bias=bias, padding_mode=padding_mode)
                layer = [Restrict(Pi=Pi, R=R)]
     
        
        self.down_layers = nn.ModuleList(down_layers)
        self.post_smooth_layers = nn.ModuleList(post_smooth_layers)

    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f 

        for l in range(len(self.num_iteration)):
            out = self.down_layers[l](out) 
            out_list[l] = out
        # upblock                                 
        for j in range(len(self.num_iteration)-2,-1,-1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = u + self.up_layers[j](out_list[j+1][0])
            out = (u_post, f)
            out_list[j] = self.post_smooth_layers[j](out)     
        return out_list[0][0]

class MgNO_DC_6(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=1,  normalizer=None, output_dim=1, activation='gelu', padding_mode='zeros', last_layer='linear', bias=False, mlp_hidden_dim=0, use_norm=False,  ):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration
 
        self.conv_list = nn.ModuleList([])
 
        self.conv_list.append(nn.Conv2d(num_channel_f, num_channel_u, kernel_size=1, stride=1, bias=True, padding=0))  
    
        for j in range(num_layer-1):
            self.conv_list.append(MgConv_DC_6(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode)) 
   
        self.norm_layer_list = nn.ModuleList([])
        for _ in range(num_layer):
            if use_norm:
                # self.norm_layer_list.append(nn.GroupNorm(2, num_channel_u, affine=True))
                self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 128, 128], elementwise_affine=True))
            else:
                self.norm_layer_list.append(nn.Identity())
        if last_layer == 'conv':
            self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=3, padding=1, padding_mode=padding_mode)
        elif last_layer == 'linear':
            self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        else:
            raise NameError('invalid last_layer')
        self.normalizer = normalizer

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
      
        for i in range(self.num_layer):
            u = self.act(self.norm_layer_list[i](self.conv_list[i](u) ))

        u = self.normalizer.decode(self.linear(u)) if self.normalizer else self.linear(u)
        return u 

class MgNO_DC_2(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=1,  normalizer=None, output_dim=1, activation='gelu', padding_mode='zeros', last_layer='conv'):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList([])
        # self.linear_list = nn.ModuleList([])
        # self.linear_list.append(nn.Conv2d(num_channel_f, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))   
        self.conv_list.append(MgConv_DC(num_iteration, num_channel_u, num_channel_f, padding_mode=padding_mode, bias=True))   
        for _ in range(num_layer-1):
            self.conv_list.append(MgConv_DC(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode)) 
            # self.linear_list.append(nn.Conv2d(num_channel_u, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))
        if last_layer == 'conv':
            self.last_layer = nn.Conv2d(num_channel_u, 1, kernel_size=3, padding=1, padding_mode=padding_mode)
        elif last_layer == 'linear':
            self.last_layer = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        else:
            raise NameError('invalid last_layer')
        self.normalizer = normalizer

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
     
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u))
        u = self.normalizer.decode(self.last_layer(u)).squeeze() if self.normalizer else self.last_layer(u).squeeze()
        return u 


class MgConv_DC_3(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=True, use_res=False, ):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.RTlayers = nn.ModuleList()
        for j in range(len(num_iteration)-1):
            self.RTlayers.append(nn.ConvTranspose2d((j+2)*num_channel_u, (j+1)*num_channel_u, kernel_size=4, stride=2, padding=1, bias=bias))

        layers = []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
            post_smooth_layers = []
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                if l==0 and i==0:
                    layers.append(MgIte_init(S))
                else:
                    if num_channel_f==num_channel_u:
                        S = Conv2dAttn((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                        A = Conv2dAttn((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                        layers.append(MgIte_2(A, S))
                    else:
                        A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                        layers.append(MgIte(A, S))
            if not num_iteration_l[1] == 0:
                for i in range(num_iteration_l[1]):
                    S = nn.Conv2d((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer'+str(l), nn.Sequential(*post_smooth_layers))
            if l < len(num_iteration)-1:
                A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                Pi= nn.Conv2d((l+1)*num_channel_u, (l+2)*num_channel_u, kernel_size=3, stride=2, padding=1, bias=bias, padding_mode=padding_mode)
                R = nn.Conv2d((l+1)*num_channel_f, (l+2)*num_channel_f, kernel_size=3, stride=2, padding=1, bias=bias, padding_mode=padding_mode)
                if use_res:
                    layers= [Restrict(Pi, R, A)]
                else:
                    layers= [Restrict(Pi=Pi, R=R)]
        
    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f 

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out) 
            out_list[l] = out
        # upblock                                 
        for j in range(len(self.num_iteration)-2,-1,-1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = u + self.RTlayers[j](F.gelu(out_list[j+1][0]))
            out = (u_post, f)
            out_list[j] = getattr(self, 'post_smooth_layer'+str(j))(out) 
            
        return out_list[0][0]

class MgNO_DC_3(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=1,  normalizer=None, output_dim=1, activation='gelu', padding_mode='circular', last_layer='conv', bias=False, mlp_hidden_dim=0):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList([])
        self.linear_list = nn.ModuleList([])
        self.linear_list.append(nn.Conv2d(num_channel_f, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))   
        self.conv_list.append(MgConv_DC_3(num_iteration, num_channel_u, num_channel_f, padding_mode=padding_mode))   
        for j in range(num_layer-1):
            self.conv_list.append(MgConv_DC_3(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode)) 
            # self.linear_list.append(nn.Conv2d(num_channel_u, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))
        
        # self.norm_layer_list = nn.ModuleList([])
        # for _ in range(num_layer):
        #     self.norm_layer_list.append(nn.GroupNorm(4, num_channel_u, affine=True))

        if last_layer == 'conv':
            self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=3, padding=1, padding_mode=padding_mode)
        elif last_layer == 'linear':
            self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        else:
            raise NameError('invalid last_layer')
        self.normalizer = normalizer

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
        u_0 = u
        for i in range(self.num_layer):
            # u = self.act(self.norm_layer_list[i](self.conv_list[i](u) ))
            u = self.act((self.conv_list[i](u) ))

        u = self.normalizer.decode(self.linear(u)) if self.normalizer else self.linear(u)
        return u + u_0

class MgConv_DC_31(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=True, use_res=False, act_before_smooth=True):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.RTlayers = nn.ModuleList()
        for j in range(len(num_iteration)-1):
            self.RTlayers.append(nn.ConvTranspose2d((j+2)*num_channel_u, (j+1)*num_channel_u, kernel_size=4, stride=2, padding=1, bias=bias))

        layers = []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
            post_smooth_layers = []
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                if l==0 and i==0:
                    layers.append(MgIte_init(S))
                else:
                    if num_channel_f==num_channel_u:
                        S = Conv2dAttn((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                        A = Conv2dAttn((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                        layers.append(MgIte_3(A, S))
                    else:
                        A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                        layers.append(MgIte_3(A, S))
            if not num_iteration_l[1] == 0:
                for i in range(num_iteration_l[1]):
                    S = nn.Conv2d((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte_3(A, S))
              
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer'+str(l), nn.Sequential(*post_smooth_layers))
            if l < len(num_iteration)-1:
                A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                Pi= nn.Conv2d((l+1)*num_channel_u, (l+2)*num_channel_u, kernel_size=3, stride=2, padding=1, bias=bias, padding_mode=padding_mode)
                R = nn.Conv2d((l+1)*num_channel_f, (l+2)*num_channel_f, kernel_size=3, stride=2, padding=1, bias=bias, padding_mode=padding_mode)
                if use_res:
                    layers= [Restrict(Pi, R, A)]
                else:
                    layers= [Restrict(Pi=Pi, R=R)]
        
    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f 

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out) 
            out_list[l] = out
        # upblock                                 
        for j in range(len(self.num_iteration)-2,-1,-1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = u + self.RTlayers[j](F.gelu(out_list[j+1][0]))
            out = (u_post, f)
            out_list[j] = getattr(self, 'post_smooth_layer'+str(j))(out) 
            
        return out_list[0][0]

class MgNO_DC_31(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=1,  normalizer=None, output_dim=1, activation='gelu', padding_mode='circular', last_layer='conv', bias=False, mlp_hidden_dim=0, lift=True):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList([])
        self.linear_list = nn.ModuleList([])
        self.linear_list.append(nn.Conv2d(num_channel_f, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))  
        # if lift:
        #     self.conv_list.append(nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)) 
        # else:
        #     self.conv_list.append(MgConv_DC_31(num_iteration, num_channel_u, num_channel_f, padding_mode=padding_mode))
        self.patch_embed = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)   
        for j in range(num_layer):
            self.conv_list.append(MgConv_DC_31(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode)) 
            # self.linear_list.append(nn.Conv2d(num_channel_u, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))
        
        self.norm_layer_list = nn.ModuleList([])
        for _ in range(num_layer):
            self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 128, 128]))

        if last_layer == 'conv':
            self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=3, padding=1, padding_mode=padding_mode)
            # self.linear = nn.Sequential(nn.Conv2d(num_channel_u, 128, kernel_size=3, padding=1, padding_mode=padding_mode),
            # nn.GELU(), nn.Conv2d(128, 1, kernel_size=1, bias=False))
        elif last_layer == 'linear':
            self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        else:
            raise NameError('invalid last_layer')
        self.normalizer = normalizer

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
        u_0 = u
        u = self.patch_embed(u)
        for i in range(self.num_layer):
            u = self.act(self.norm_layer_list[i](self.conv_list[i](u) ))
            # u = self.act((self.conv_list[i](u) ))

        u = self.normalizer.decode(self.linear(u)) if self.normalizer else self.linear(u)
        return u + u_0
        

class MgConv_DC_4(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=False, use_res=False, groups=1):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.RTlayers = nn.ModuleList()
        for j in range(len(num_iteration)-1):
            self.RTlayers.append(nn.ConvTranspose2d((j+2)*num_channel_u, (j+1)*num_channel_u, kernel_size=4, stride=2, padding=1, bias=bias, groups=groups))
        self.norm_layer_list = nn.ModuleList([])

        self.norm_layer_list = nn.ModuleList([])
        for j in range(len(num_iteration)-1):
            self.norm_layer_list.append(nn.GroupNorm((j+1)*2, (j+1)*num_channel_u, affine=True))
            # self.norm_layer_list.append(nn.LayerNorm([(j+1)*num_channel_u, 128//2**(j), 128//2**(j)]))

        layers = []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
            post_smooth_layers = []
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode, groups=groups)
                if l==0 and i==0:
                    layers.append(MgIte_init(S))
                else:
                    A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode, groups=groups)
                    layers.append(MgIte(A, S))
            if not num_iteration_l[1] == 0:
                for i in range(num_iteration_l[1]):
                    S = nn.Conv2d((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode, groups=groups)
                    A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode, groups=groups)
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer'+str(l), nn.Sequential(*post_smooth_layers))
            if l < len(num_iteration)-1:
                A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode, groups=groups)
                Pi= nn.Conv2d((l+1)*num_channel_u, (l+2)*num_channel_u, kernel_size=3, stride=2, padding=1, bias=bias, padding_mode=padding_mode, groups=groups)
                R = nn.Conv2d((l+1)*num_channel_f, (l+2)*num_channel_f, kernel_size=3, stride=2, padding=1, bias=bias, padding_mode=padding_mode, groups=groups)
                if use_res:
                    layers= [Restrict(Pi, R, A)]
                else:
                    layers= [Restrict(Pi=Pi, R=R)]
        
    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f 

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out) 
            out_list[l] = out
        # upblock                                 
        for j in range(len(self.num_iteration)-2,-1,-1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = u + self.RTlayers[j](out_list[j+1][0])
            u_post = F.gelu(self.norm_layer_list[j](u_post))
            out = (u_post, f)
            out_list[j] = getattr(self, 'post_smooth_layer'+str(j))(out) 
            
        return out_list[0][0]

class MgNO_DC_4(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=1,  normalizer=None, output_dim=1, activation='gelu', padding_mode='zeros', last_layer='conv'):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration
        self.proj = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv_list = nn.ModuleList([])
        self.linear_list = nn.ModuleList([])
        self.linear_list.append(nn.Conv2d(num_channel_u, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))   
        self.conv_list.append(MgConv_DC_4(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode))   
        for j in range(num_layer-1):
            self.conv_list.append(MgConv_DC_4(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode, groups=1)) 
            self.linear_list.append(nn.Conv2d(num_channel_u, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))
    

        if last_layer == 'conv':
            self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=3, padding=1, padding_mode=padding_mode)
        elif last_layer == 'linear':
            self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        else:
            raise NameError('invalid last_layer')
        self.normalizer = normalizer

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
        u = self.proj(u)
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u) + self.linear_list[i](u))
        u = self.normalizer.decode(self.linear(u)).squeeze() if self.normalizer else self.linear(u).squeeze()
        return u

class MgNO_DC_smooth(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=1,  normalizer=None, output_dim=1, activation='gelu', padding_mode='zeros', ):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList([])
        self.linear_list = nn.ModuleList([])
        self.linear_list.append(nn.Conv2d(num_channel_f, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))   
        self.conv_list.append(MgConv_DC_smooth(num_iteration, num_channel_u, num_channel_f, padding_mode=padding_mode))   
        for _ in range(num_layer-1):
            self.conv_list.append(MgConv_DC_smooth(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode)) 
            self.linear_list.append(nn.Conv2d(num_channel_u, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))
   
        self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        self.normalizer = normalizer

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
     
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u) + self.linear_list[i](u))
        u = self.normalizer.decode(self.linear(u)) if self.normalizer else self.linear(u)
        return u 

# the best helm for paper MgNO 7e-3 
# class MgConv_helm(nn.Module):
#     def __init__(self, num_iteration, num_channel_u, num_channel_f, init=False):
#         super().__init__()
#         self.num_iteration = num_iteration
#         self.num_channel_u = num_channel_u
#         self.init = init
#         self.RTlayers = nn.ModuleList()
#         self.RTlayers.append(nn.ConvTranspose2d(num_channel_u * 2, num_channel_u, kernel_size=3, stride=2, padding=0, bias=False, )) 
        
#         for j in range(len(num_iteration)-3):
#             self.RTlayers.append(nn.ConvTranspose2d(num_channel_u * (j+3), num_channel_u * (j+2), kernel_size=4, stride=2, padding=0, bias=False, )) 
#         self.RTlayers.append(nn.ConvTranspose2d(num_channel_u * 5, num_channel_u * 4, kernel_size=3, stride=2, padding=0, bias=False, )) 
           
#         layers = []
#         for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
           
#             for i in range(num_iteration_l):
#                 A = nn.Conv2d(num_channel_u * (l+1), num_channel_f * (l+1), kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect')
#                 S = nn.Conv2d(num_channel_f * (l+1), num_channel_u * (l+1), kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect')
#                 if self.init:
#                     torch.nn.init.xavier_uniform_(Pi.weight, gain=1/(num_channel_u**2))
#                     torch.nn.init.xavier_uniform_(R.weight, gain=1/(num_channel_u**2))   
#                 layers.append(MgIte(A, S))

#             setattr(self, 'layer'+str(l), nn.Sequential(*layers))
        
#             if l < len(num_iteration)-1:
#                 Pi = nn.Conv2d(num_channel_u * (l+1), num_channel_u * (l+2), kernel_size=3, stride=2, padding=0, bias=False, padding_mode='zeros')
#                 R  = nn.Conv2d(num_channel_f * (l+1), num_channel_f * (l+2), kernel_size=3, stride=2, padding=0, bias=False, padding_mode='zeros')
#                 if self.init:
#                     torch.nn.init.xavier_uniform_(Pi.weight, gain=1/(num_channel_u**2))
#                     torch.nn.init.xavier_uniform_(R.weight, gain=1/(num_channel_u**2))             
#                 layers= [Restrict(Pi, R)]
         

#     def forward(self, f):

#         u_list = []
#         out = f  
                                            
#         for l in range(len(self.num_iteration)):
#             out = getattr(self, 'layer'+str(l))(out) 
#             u, f = out                                       
#             u_list.append(u)                                

#         # upblock                                 

#         for j in range(len(self.num_iteration)-2,-1,-1):
#             u_list[j] = u_list[j] + self.RTlayers[j](u_list[j+1])
#         u = u_list[0]
        
#         return u        

class MgConv_helm(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, init=False, padding_mode='reflect'):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.init = init
        self.RTlayers = nn.ModuleList()
        self.RTlayers.append(nn.ConvTranspose2d(num_channel_u * 2, num_channel_u, kernel_size=3, stride=2, padding=0, bias=False, )) 
        
        for j in range(len(num_iteration)-3):
            self.RTlayers.append(nn.ConvTranspose2d(num_channel_u * (j+3), num_channel_u * (j+2), kernel_size=4, stride=2, padding=0, bias=False, )) 
        self.RTlayers.append(nn.ConvTranspose2d(num_channel_u * 5, num_channel_u * 4, kernel_size=3, stride=2, padding=0, bias=False, )) 
           
        layers = []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
           
            for i in range(num_iteration_l):
                A = nn.Conv2d(num_channel_u * (l+1), num_channel_f * (l+1), kernel_size=3, stride=1, padding=1, bias=True, padding_mode=padding_mode)
                S = nn.Conv2d(num_channel_f * (l+1), num_channel_u * (l+1), kernel_size=3, stride=1, padding=1, bias=True, padding_mode=padding_mode)
                if l==0 and i==0:
                    layers.append(MgIte_init(S))
                else:
                    if num_channel_f==num_channel_u:
                        S = Conv2dAttn((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                        A = Conv2dAttn((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                        layers.append(MgIte_2(A, S))
                    else:
                        layers.append(MgIte(A, S))

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))
        
            if l < len(num_iteration)-1:
                Pi = nn.Conv2d(num_channel_u * (l+1), num_channel_u * (l+2), kernel_size=3, stride=2, padding=0, bias=False, padding_mode='zeros')
                R  = nn.Conv2d(num_channel_f * (l+1), num_channel_f * (l+2), kernel_size=3, stride=2, padding=0, bias=False, padding_mode='zeros')
                if self.init:
                    torch.nn.init.xavier_uniform_(Pi.weight, gain=1/(num_channel_u**2))
                    torch.nn.init.xavier_uniform_(R.weight, gain=1/(num_channel_u**2))             
                layers= [Restrict(Pi, R)]
         

    def forward(self, f):

        u_list = []
        out = f  
                                            
        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out) 
            u, f = out                                       
            u_list.append(u)                                

        # upblock                                 

        for j in range(len(self.num_iteration)-2,-1,-1):
            u_list[j] = u_list[j] + self.RTlayers[j](u_list[j+1])
        u = u_list[0]
        
        return u  

class MgConv_helm2(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=False,  elementwise_affine=True, init=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.norm_layer_list = nn.ModuleList([])
        self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 101, 101], ))
        self.norm_layer_list.append(nn.LayerNorm([num_channel_u*2, 50, 50], elementwise_affine=False ))
        self.norm_layer_list.append(nn.LayerNorm([num_channel_u*3, 24, 24], elementwise_affine=False))
        self.norm_layer_list.append(nn.LayerNorm([num_channel_u*4, 11, 11], elementwise_affine=False))

        self.RTlayers = nn.ModuleList()
        self.RTlayers.append(nn.ConvTranspose2d(num_channel_u * 2, num_channel_u, kernel_size=3, stride=2, padding=0, bias=False, )) 
        for j in range(len(num_iteration)-3):
            self.RTlayers.append(nn.ConvTranspose2d(num_channel_u * (j+3), num_channel_u * (j+2), kernel_size=4, stride=2, padding=0, bias=False, )) 
        self.RTlayers.append(nn.ConvTranspose2d(num_channel_u * 5, num_channel_u * 4, kernel_size=3, stride=2, padding=0, bias=False, )) 
          
        layers = []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
            post_smooth_layers = []
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f * (l+1), num_channel_u * (l+1), kernel_size=3, stride=1, padding=1, bias=bias, padding_mode='reflect')
                if l==0 and i==0:
                    layers.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u * (l+1), num_channel_f * (l+1), kernel_size=3, stride=1, padding=1, bias=bias, padding_mode='reflect')
                    layers.append(MgIte(A, S))
            if not num_iteration_l[1] == 0:
                for i in range(num_iteration_l[1]):
                    S = nn.Conv2d(num_channel_f * (l+1), num_channel_u * (l+1), kernel_size=3, stride=1, padding=1, bias=bias, padding_mode='reflect')
                    A = nn.Conv2d(num_channel_u * (l+1), num_channel_f * (l+1), kernel_size=3, stride=1, padding=1, bias=bias, padding_mode='reflect')
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer'+str(l), nn.Sequential(*post_smooth_layers))
            if l < len(num_iteration)-1:
                Pi = nn.Conv2d(num_channel_u * (l+1), num_channel_u * (l+2), kernel_size=3, stride=2, padding=0, bias=False, padding_mode='zeros')
                R  = nn.Conv2d(num_channel_f * (l+1), num_channel_f * (l+2), kernel_size=3, stride=2, padding=0, bias=False, padding_mode='zeros')
                layers= [Restrict(Pi=Pi, R=R)]
        
    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f 

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out) 
            out_list[l] = out
        # upblock                                 
        for j in range(len(self.num_iteration)-2,-1,-1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = self.norm_layer_list[j](u + self.RTlayers[j](out_list[j+1][0]))
            out = (u_post, f)
            out_list[j] = getattr(self, 'post_smooth_layer'+str(j))(out) 
            
        return out_list[0][0]



class MgNO_helm(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=3,  normalizer=None,  output_dim=1, activation='gelu', init=False, padding_mode='reflect', last_layer='linear'):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.norm_layer_list = nn.ModuleList([])
        for _ in range(num_layer):
            self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 101, 101]))
            # self.norm_layer_list.append(nn.Identity())
        self.conv_list = nn.ModuleList([])   
        self.conv_list.append(MgConv_helm(num_iteration, num_channel_u, num_channel_f, init=init, padding_mode=padding_mode))   
        for _ in range(num_layer-1):
            self.conv_list.append(MgConv_helm(num_iteration, num_channel_u, num_channel_u, init=init, padding_mode=padding_mode))
        if last_layer == 'mgno':
            self.last_layer = MgConv_helm(num_iteration, 1, num_channel_u, init=init, padding_mode=padding_mode)
        elif last_layer == 'conv':
            self.last_layer = nn.Conv2d(num_channel_u, 1, kernel_size=3, padding=1, padding_mode=padding_mode)
        elif last_layer == 'linear':
            self.last_layer = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        else:
            raise NameError('invalid last_layer')

        self.normalizer = normalizer

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
        for i in range(self.num_layer):
            u = self.act(self.norm_layer_list[i](self.conv_list[i](u)))
        return self.normalizer.decode(torch.squeeze(self.last_layer(u))) if self.normalizer else self.last_layer(u) 

class MgNO_helm2(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=3,  normalizer=None, mlp_hidden_dim=128, output_dim=1, activation='gelu', init=False, if_mlp=False, padding_mode='reflect'):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration


        self.conv_list = nn.ModuleList([])   
        # self.norm_layer_list = nn.ModuleList([])
        # for _ in range(num_layer):
        #     self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 101, 101]))
        self.conv_list.append(MgConv_helm3(num_iteration, num_channel_u, num_channel_f, init=init, padding_mode=padding_mode))   
        for _ in range(num_layer-1):
            self.conv_list.append(MgConv_helm3(num_iteration, num_channel_u, num_channel_u, init=init, padding_mode=padding_mode))

        if if_mlp:
            self.mlp = MgConv_helm3(num_iteration, 1, num_channel_u, init=init)
        else:
            self.mlp = nn.Conv2d(num_channel_u, 1, kernel_size=1)  
         
        self.normalizer = normalizer

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
        for i in range(self.num_layer):
            u = self.act((self.conv_list[i](u))) #self.norm_layer_list[i]
        return self.normalizer.decode(torch.squeeze(self.mlp(u))) if self.normalizer else self.mlp(u)

class MgConv_helm3(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='reflect', bias=False,  elementwise_affine=True, init=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        # self.norm_layer_list = nn.ModuleList([])
        # self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 101, 101], ))
        # self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 50, 50], elementwise_affine=False ))
        # self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 24, 24], elementwise_affine=False))
        # self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 11, 11], elementwise_affine=False))

        self.RTlayers = nn.ModuleList()
        self.RTlayers.append(nn.ConvTranspose2d(2*num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=0, bias=False, )) 
        for j in range(len(num_iteration)-3):
            self.RTlayers.append(nn.ConvTranspose2d((j+3)*num_channel_u, (j+2)*num_channel_u, kernel_size=4, stride=2, padding=0, bias=False, )) 
        self.RTlayers.append(nn.ConvTranspose2d(5*num_channel_u, 4*num_channel_u, kernel_size=3, stride=2, padding=0, bias=False, )) 
          
        layers = []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
            post_smooth_layers = []
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode='reflect')
                if l==0 and i==0:
                    layers.append(MgIte_init(S))
                else:
                    if num_channel_f==num_channel_u:
                        S = Conv2dAttn((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                        A = Conv2dAttn((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                        layers.append(MgIte_2(A, S))
                    else:
                        A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode='reflect')
                        layers.append(MgIte(A, S))
            if not num_iteration_l[1] == 0:
                for i in range(num_iteration_l[1]):
                    S = nn.Conv2d((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode='reflect')
                    A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode='reflect')
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer'+str(l), nn.Sequential(*post_smooth_layers))
            if l < len(num_iteration)-1:
                Pi = nn.Conv2d((l+1)*num_channel_u, (l+2)*num_channel_u, kernel_size=3, stride=2, padding=0, bias=False, padding_mode='zeros')
                R  = nn.Conv2d((l+1)*num_channel_f, (l+2)*num_channel_f, kernel_size=3, stride=2, padding=0, bias=False, padding_mode='zeros')
                layers= [Restrict(Pi=Pi, R=R)]
        
    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f 

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out) 
            out_list[l] = out
        # upblock                                 
        for j in range(len(self.num_iteration)-2,-1,-1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = (u + self.RTlayers[j](F.gelu(out_list[j+1][0]))) #self.norm_layer_list[j]
            out = (u_post, f)
            out_list[j] = getattr(self, 'post_smooth_layer'+str(j))(out) 
            
        return out_list[0][0]


class MgConv_DC_smooth(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=False, use_res=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode
        self.RTlayers = nn.ModuleList()       
        self.RTlayers.append(nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False))
        self.RTlayers.append(nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=4, stride=2, padding=1, bias=False)) 
        self.RTlayers.append(nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False))
        self.RTlayers.append(nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False))
        self.RTlayers.append(nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=4, stride=2, padding=1, bias=False)) 
        layers = []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
            post_smooth_layers = []
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                if l==0 and i==0:
                    layers.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    layers.append(MgIte(A, S))
            if not num_iteration_l[1] == 0:
                for i in range(num_iteration_l[1]):
                    S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer'+str(l), nn.Sequential(*post_smooth_layers))
            if l < len(num_iteration)-1:
                A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                Pi= nn.Conv2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False, padding_mode=padding_mode)
                R = nn.Conv2d(num_channel_f, num_channel_f, kernel_size=3, stride=2, padding=1, bias=False, padding_mode=padding_mode)
                if use_res:
                    layers= [Restrict(Pi, R, A)]
                else:
                    layers= [Restrict(Pi=Pi, R=R)]
     
    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f 

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out) 
            out_list[l] = out
        # upblock                                 
        for j in range(len(self.num_iteration)-2,-1,-1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = u + self.RTlayers[j](out_list[j+1][0])
            out = (u_post, f)
            out_list[j] = getattr(self, 'post_smooth_layer'+str(j))(out) 
            
        return out_list[0][0]


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
                kernel_type='p', padding=9, resolution=None, init_scale=1, 
                add_pos=False, shortcut=True, normalizer=None, if_pre_permute=True):
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
            
        self.pre_permute = partial(torch.permute, dims=(0, 3, 1, 2))
        self.post_permute = partial(torch.permute, dims=(0, 2, 3, 1))

        self.mlp = FeedForward(self.width, mlp_hidden_dim, output_dim, LN=mlp_LN)
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
            self.pre_process = torch.nn.Identity()
            if normalizer:
                self.post_process = transforms.Compose([self.post_permute, self.mlp, torch.squeeze, self.normalizer.decode])
            else:
                self.post_process = transforms.Compose([self.post_permute, self.mlp, torch.squeeze])

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







if __name__ == "__main__":
    
    # torch.autograd.set_detect_anomaly(True)
    # model = MgNO_helm(num_layer=4, num_channel_u=24, num_channel_f=1, num_classes=1, num_iteration=[1 , 1, 1, 1, 2],).cuda()
    # model = MgNO_helm(num_layer=4, num_channel_u=20, num_channel_f=1, num_classes=1, num_iteration=[[1,0], [1,0], [1,0], [1,0], [2,0]]).cuda()
    model = MgNO_DC_6(num_layer=4, num_channel_u=24, num_channel_f=1, num_classes=1, num_iteration=[[1,0], [1,0], [1,0], [1,0], [1,0], ]).cuda()

    # print(model)
    # print(count_params(model))
    inp = torch.randn(4, 1, 63, 63).cuda()
    out = model(inp)
    print(out.shape)
    # summary(model, input_size=(4, 1, 63, 63))
    # backward check
    # out.sum().backward()
    print('success!')
    

