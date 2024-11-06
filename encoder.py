import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Convolutional block with Conv -> BN -> GELU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, if_linear=False):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=False
            ),
            nn.Identity() if if_linear else nn.GroupNorm(32, out_channels),
            nn.Identity() if if_linear else nn.GELU()
        )
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""
    def __init__(self, channels, if_linear=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, padding=1, if_linear=if_linear)
        # self.conv2 = ConvBlock(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        # out = self.conv2(out)
        out += identity  # Residual connection
        return out

class Encoder(nn.Module):
    def __init__(self, branch_features, if_linear=False):
        super(Encoder, self).__init__()
        self.initial = nn.Sequential(
            nn.GroupNorm(4, 4),
            nn.GELU(),
            ConvBlock(4, 64, kernel_size=7, stride=2, padding=3, if_linear=if_linear)
        )
        self.layer1 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=5, stride=2, padding=2, if_linear=if_linear),
            ResidualBlock(128, if_linear=if_linear)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1, if_linear=if_linear),
            ResidualBlock(256, if_linear=if_linear)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, stride=2, padding=1, if_linear=if_linear),
            ResidualBlock(512, if_linear=if_linear)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, branch_features)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
class GrayScott_Grad_2(nn.Module):
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

    def forward(self, A_S):
        # Apply Laplacian using convolution
        A = A_S[:, 0:1, ...]
        S = A_S[:, 1:2, ...]
        lap_A = self.lap(A)
        lap_S = self.lap(S)

        # Compute residuals for A and S
        F_A = self.DA * lap_A - S * A**2 + (self.mu + self.rho) * A
        F_S = self.DS * lap_S + S * A**2 - self.rho * (1 - S)
        
        # return the concatenation of the residuals and A_S
        
        return torch.cat((F_A, F_S), dim=1), A_S

class DeepONet_POD_2(nn.Module):
    def __init__(self, branch_features, trunk_features, grid_size=63, V=None):
        super(DeepONet_POD_2, self).__init__()
        self.grid_size = grid_size
        self.grad = GrayScott_Grad_2()
        self.branch_1 = Encoder(branch_features)
        self.branch_2 = Encoder(branch_features, if_linear=True)
        

    
    
    def forward(self, x_branch):
        batch_size = x_branch.shape[0]
        grad, A_S = self.grad(x_branch)
        branch_out_1 = self.branch_1(A_S)
        branch_out_2 = self.branch_2(grad)
        branch_out = branch_out_1 * branch_out_2
        # out = torch.mm(branch_out, self.trunk)
        # out = out.view(batch_size, 2, self.grid_size, self.grid_size) #+ self.branch_2(x_branch)
        return branch_out

if __name__ == "__main__":
    model = DeepONet_POD_2(branch_features=10, trunk_features=10)
    input_tensor = torch.randn(1, 2, 63, 63)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # torch.Size([1, 10])
# Example usage:
# Assuming input images are of size (4, 64, 64) and you want 128 features
# model = LinearEncoder(input_channels=4, input_height=64, input_width=64, branch_features=128)


# Example usage:
# model = Encoder(branch_features=10)
# input_tensor = torch.randn(1, 4, 63, 63)
# output_tensor = model(input_tensor)
# print(output_tensor.shape)  # torch.Size([1, 10])

# model = DeepONet_POD_2(branch_features=10, trunk_features=10)
# input_tensor = torch.randn(1, 2, 63, 63)
# output_tensor = model(input_tensor)
# print(output_tensor.shape)  # torch.Size([1, 10])



