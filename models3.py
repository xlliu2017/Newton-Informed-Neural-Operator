import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from utilities3 import count_params



class Conv2dAttn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels*4, kernel_size, stride, padding, bias=bias, padding_mode=padding_mode)
        self.m = nn.Softmax(dim=1)
    def forward(self, x):
        qkv = self.conv(x)
        qkv = qkv.view(qkv.shape[0], 2, 2, self.out_channels, qkv.shape[2], qkv.shape[3])

        k = qkv[:,0,...]
        v = qkv[:,1,...]
        # out1 = torch.einsum('ijlm, ijklm, ijklm -> ijlm', x, k, v)
        # xk = torch.einsum('iklm, ijklm -> ijlm', x, k)/400
        xk = self.m(torch.einsum('iklm, ijklm -> ijlm', x, k)) #*self.out_channels**-0.5
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
        out_channels = self.A.out_channels
        self.norm = nn.GroupNorm(2, out_channels, affine=True)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1,  bias=True),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1,  bias=True),
        )

    def forward(self, out):
        
        u, f = out
        # u = u + self.A(u)
        u = self.mlp(self.norm(self.A(u))) 
        # u = self.A(u)
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
            u = u + (self.S(F.gelu(self.norm2(f-self.A(F.gelu(self.norm1(u)))))))  

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




class HAConv(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=True, use_res=False, ):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.RTlayers = nn.ModuleList()
        # self.normlayers = nn.ModuleList()
        for j in range(len(num_iteration)-1):
            self.RTlayers.append(nn.ConvTranspose2d((j+2)*num_channel_u, (j+1)*num_channel_u, kernel_size=4, stride=2, padding=1, bias=bias))
            # self.normlayers.append(nn.GroupNorm((j+1)*2, (j+1)*num_channel_u, affine=True))
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
            u_post = (u + self.RTlayers[j](F.gelu(out_list[j+1][0]))) #self.normlayers[j]
            out = (u_post, f)
            out_list[j] = getattr(self, 'post_smooth_layer'+str(j))(out) 
            
        return out_list[0][0]

class HANO(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=1,  normalizer=None, output_dim=1, activation='gelu', padding_mode='circular', last_layer='conv', bias=False, mlp_hidden_dim=0, use_norm=True, patch_embed=True):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.conv_list = nn.ModuleList([])
        self.linear_list = nn.ModuleList([])
        self.linear_list.append(nn.Conv2d(num_channel_f, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True)) 
        if patch_embed:
            self.conv_list.append(nn.Conv2d(num_channel_f, num_channel_u, kernel_size=1, stride=1, bias=True))  
        else:
            self.conv_list.append(HAConv(num_iteration, num_channel_u, num_channel_f, padding_mode=padding_mode))   
        for j in range(num_layer-1):
            self.conv_list.append(HAConv(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode)) 
            # self.linear_list.append(nn.Conv2d(num_channel_u, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))
        
        self.norm_layer_list = nn.ModuleList([])
        for _ in range(num_layer):
            if use_norm:
                self.norm_layer_list.append(nn.GroupNorm(2, num_channel_u, affine=True))
                # self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 64, 64], elementwise_affine=True))
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
        u_0 = u
        for i in range(self.num_layer):
            u = self.act(self.norm_layer_list[i](self.conv_list[i](u) ))
            # u = self.act((self.conv_list[i](u) ))

        u = self.normalizer.decode(self.linear(u)) if self.normalizer else self.linear(u)
        return u + u_0




class MgConv_helm2(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=False,  elementwise_affine=True, init=False):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.padding_mode = padding_mode

        self.norm_layer_list = nn.ModuleList([])
        self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 101, 101], elementwise_affine=False))
        self.norm_layer_list.append(nn.LayerNorm([num_channel_u*2, 50, 50], elementwise_affine=False))
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
                    A = Conv2dAttn((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
                    layers.append(MgIte_2(A, S))
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
            u_post = self.norm_layer_list[j](u + self.RTlayers[j](F.relu(out_list[j+1][0])))
            out = (u_post, f)
            out_list[j] = getattr(self, 'post_smooth_layer'+str(j))(out) 
            
        return out_list[0][0]



class HANO_helm(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=3,  normalizer=None,  output_dim=1, activation='gelu', init=False, padding_mode='reflect', last_layer='linear'):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        # self.norm_layer_list = nn.ModuleList([])
        # for _ in range(num_layer):
        #     self.norm_layer_list.append(nn.LayerNorm([num_channel_u, 101, 101]))
            # self.norm_layer_list.append(nn.Identity())
        self.patch_embed = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.conv_list = nn.ModuleList([])   
        for _ in range(num_layer):
            self.conv_list.append(MgConv_helm2(num_iteration, num_channel_u, num_channel_u, init=init, padding_mode=padding_mode))
 
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
        u = self.act(self.patch_embed(u))
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u))
        return self.normalizer.decode(torch.squeeze(self.last_layer(u))) if self.normalizer else self.last_layer(u)








if __name__ == "__main__":
    
    torch.autograd.set_detect_anomaly(True)
    model = HANO(num_layer=4, num_channel_u=24, num_channel_f=1, num_classes=1, num_iteration=[[1,0], [1,0], [1,0], [1,0], [1,0], [1,0]]).cuda()


    inp = torch.randn(10, 1, 64, 64).cuda()
    out = model(inp)
    print(out.shape)

    out.sum().backward()
    print('success!')
    ## NS 1e5 setting
    #{'num_layer': 4, 'in_chans': 1, 'num_channel_u': 24, 'num_channel_f': 1, 'num_classes': 1, 'num_iteration': [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]], 'mlp_hidden_dim': 0, 'output_dim': 1, 'padding_mode': 'circular', 'bias': True}
    #{'data': '1e-5', 'path': '/ibex/ai/home/liux0t/Xinliang/FMM/data/NavierStokes_V1e-5_N1200_T20.mat', 'ntrain': 1000, 'ntest': 100, 'batch_size': 50, 'epochs': 500, 'T_in': 1, 'T_out': 1, 'T': 10, 'step': 1, 'r': 64, 'sampling': 1, 'full_train': True, 'full_train_2': True, 'loss_type': 'l2', 'GN': False, 'learning_rate': 0.0006, 'final_div_factor': 50.0, 'div_factor': 2, 'weight_decay': 0.0001}

