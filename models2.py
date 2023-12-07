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
        self.norm = nn.GroupNorm(4, out_channels, affine=True)
        self.m = nn.Softmax(dim=1)
    def forward(self, x):
        
        if isinstance(x, tuple):
            x, k, v = x
            k = k.view(k.shape[0], 2, k.shape[1]//2, k.shape[2], k.shape[3])
            v = v.view(v.shape[0], 2, v.shape[1]//2, v.shape[2], v.shape[3])
        else:
            kv = self.conv(x)
            kv = kv.view(kv.shape[0], 2, 2, self.out_channels, kv.shape[2], kv.shape[3])
            # q = kv[:,:,0,...]
            k = kv[:,0,...]
            v = kv[:,1,...]
        # out1 = torch.einsum('ijlm, ijklm, ijklm -> ijlm', x, k, v)
        # xk = torch.einsum('iklm, ijklm -> ijlm', x, k)/400
        xk = self.m(torch.einsum('iklm, ijklm -> ijlm', x, k))
        x = torch.einsum('ijlm, ijklm -> iklm', xk, v)
        k = k.view(k.shape[0], self.out_channels*2, k.shape[3], k.shape[4])
        v = v.view(v.shape[0], self.out_channels*2, v.shape[3], v.shape[4])
        out = (x, k, v)
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
        
        # x, k, v = out
        # u = u + self.A(u)
        out = self.A(out)

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
    def __init__(self, Pi=None, R=None, R2=None, A=None):
        super().__init__()
        self.Pi = Pi
        self.R = R
        self.R2 = R2
        self.A = A
    def forward(self, out):
        x, k, v = out
        if self.A is not None:
            f = self.R(f-self.A(u))
        else:
            v = self.R(v)
        x = self.Pi(x) 
        k = self.R2(k)                           
        out = (x, k, v)
        return out



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
                S = Conv2dAttn((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                layers.append(S)

            if not num_iteration_l[1] == 0:
                for i in range(num_iteration_l[1]):
                    S = Conv2dAttn((l+1)*num_channel_f, (l+1)*num_channel_u, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                    post_smooth_layers.append(S)
            else:
                post_smooth_layers.append(nn.Identity())

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer'+str(l), nn.Sequential(*post_smooth_layers))
            if l < len(num_iteration)-1:
                A = nn.Conv2d((l+1)*num_channel_u, (l+1)*num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                Pi= nn.Conv2d((l+1)*num_channel_u, (l+2)*num_channel_u, kernel_size=3, stride=2, padding=1, bias=bias, padding_mode=padding_mode)
                R = nn.Conv2d((l+1)*num_channel_u*2, (l+2)*num_channel_u*2, kernel_size=3, stride=2, padding=1, bias=bias, padding_mode=padding_mode)
                R2 = nn.Conv2d((l+1)*num_channel_u*2, (l+2)*num_channel_u*2, kernel_size=3, stride=2, padding=1, bias=bias, padding_mode=padding_mode)
                if use_res:
                    layers= [Restrict(Pi, R, R2, A)]
                else:
                    layers= [Restrict(Pi=Pi, R=R, R2=R2)]
        
    def forward(self, x):
        out_list = [0] * len(self.num_iteration)
        out = x 

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out) 
            out_list[l] = out
        # upblock                                 
        for j in range(len(self.num_iteration)-2,-1,-1):
            x, k, v = out_list[j][0], out_list[j][1], out_list[j][2]
            x_post = x + self.RTlayers[j](F.gelu(out_list[j+1][0]))
            out = (x_post, k, v)
            out_list[j] = getattr(self, 'post_smooth_layer'+str(j))(out) 
            
        return out_list[0][0]

class HANO(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_classes, num_iteration, 
    in_chans=1,  normalizer=None, output_dim=1, activation='gelu', padding_mode='circular', last_layer='conv', bias=False, mlp_hidden_dim=0):
        super().__init__()
        self.num_layer = num_layer
        self.num_channel_u = num_channel_u
        self.num_channel_f = num_channel_f
        self.num_classes = num_classes
        self.num_iteration = num_iteration

        self.patch_embed = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv_list = nn.ModuleList([])
        for j in range(num_layer):
            self.conv_list.append(MgConv_DC_3(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode)) 
 
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
        u = self.patch_embed(u)
        for i in range(self.num_layer):
            # u = self.act(self.norm_layer_list[i](self.conv_list[i](u) ))
            u = self.act((self.conv_list[i](u) ))

        u = self.normalizer.decode(self.linear(u)) if self.normalizer else self.linear(u)
        return u + u_0










if __name__ == "__main__":
    
    torch.autograd.set_detect_anomaly(True)
    model = MgNO_DC_3(num_layer=3, num_channel_u=24, num_channel_f=1, num_classes=1, num_iteration=[[1,0], [1,0], [1,0], [1,0], [1,0]],).cuda()
    # model = MgNO_helm(num_layer=4, num_channel_u=20, num_channel_f=1, num_classes=1, num_iteration=[[1,0], [1,0], [1,0], [1,0], [2,0]]).cuda()

    print(model)
    print(count_params(model))
    inp = torch.randn(10, 1, 128, 128).cuda()
    out = model(inp)
    print(out.shape)
    # summary(model, input_size=(10, 1, 101, 512))
    # backward check
    out.sum().backward()
    print('success!')
    

