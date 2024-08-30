import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from einops import rearrange
import einops
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import DropPath, trunc_normal_

# Res + DSwinT(windows=12,down=2) + AMU

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """
    def __init__(self, input_dim, output_dim, head_dim, window_size, type, down_r, qkv_bias=True):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.down_r = down_r
        self.type=type

        self.q = nn.Linear(self.input_dim, self.input_dim, bias=qkv_bias)
        self.k = nn.Linear(self.input_dim, self.input_dim, bias=qkv_bias)
        self.v = nn.Linear(self.input_dim, self.input_dim, bias=qkv_bias)

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        self.relative_position_table = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))
        trunc_normal_(self.relative_position_table, std=.02)
        self.relative_position_table = torch.nn.Parameter(self.relative_position_table.view(2*window_size-1, 2*window_size-1, self.n_heads).permute(2, 0, 1).contiguous())

    def relative_embedding(self):
        relative_coords = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relative_position_index = relative_coords[:, None, :] - relative_coords[None, :, :] + self.window_size -1 #The relative position shift starts from 0. shape:[window_size*window_size, window_size*window_size,2]

        relative_position_index = relative_position_index[:, 0:int(self.window_size*self.window_size//self.down_r//self.down_r), :] #Before cropping or after cropping? 

        relative_position_bias = self.relative_position_table[:, relative_position_index[:,:,0].long(), relative_position_index[:,:,1].long()] #nh,window_size*window_size,window_size*window_size

        # relative_position_bias = relative_position_bias[:, :, 0:int(self.window_size*self.window_size//self.down_r//self.down_r)] #Before cropping or after cropping? 

        return relative_position_bias
    
    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p//self.down_r, p//self.down_r, dtype=torch.bool, device=self.relative_position_table.device)

        s = p - shift
        attn_mask[-1, :, :s, :, s//self.down_r:, :] = True
        attn_mask[-1, :, s:, :, :s//self.down_r, :] = True
        attn_mask[:, -1, :, :s, :, s//self.down_r:] = True
        attn_mask[:, -1, :, s:, :, :s//self.down_r] = True
        attn_mask = einops.rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        # Only downsample the tensor after dividing the window, without affecting the window interaction.
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))

        x = einops.rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        nh = x.size(1) #w1
        nw = x.size(2) #w2

        x_down = einops.rearrange(x, 'b w1 w2 p1 p2 c -> (b w1 w2) c p1 p2')
        x_down = F.max_pool2d(x_down, self.down_r)
        x_down = einops.rearrange(x_down, '(b w1 w2) c dp1 dp2 -> b (w1 w2) (dp1 dp2) c', w1=nh, w2=nw)

        x = einops.rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        q = self.q(x)
        q = einops.rearrange(q, 'b nw np (nh c) -> nh b nw np c', c=self.head_dim)

        k = self.k(x_down)
        k = einops.rearrange(k, 'b nw ndp (nh c) -> nh b nw ndp c', c=self.head_dim)
        v = self.v(x_down)
        v = einops.rearrange(v, 'b nw ndp (nh c) -> nh b nw ndp c', c=self.head_dim)

        attn = (q * self.scale) @ k.transpose(-2, -1) # nh, b, nW*nH, p*p, dp*dp 
        # Adding learnable relative embedding
        attn = attn + einops.rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')

        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(nh, nw, self.window_size, shift=self.window_size//2)
            attn = attn.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(attn, dim=-1)
        output = probs @ v
        output = einops.rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = einops.rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=nh, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))

        return output


class SwinTBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', down_r=2):
        """ SwinTransformer Block
        """
        super(SwinTBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type, down_r)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        x = einops.rearrange(x, 'b h w c -> b c h w')
        return x

class SE_Layer(nn.Module):
    def __init__(self, channel=512, se_ratio=0.75, bias=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel*se_ratio), bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel*se_ratio), channel, bias=bias),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True, se_ratio=0.75, bias=False):
        super(ResBlock, self).__init__()
        self.use_se = use_se
        
        self.identity = nn.Conv2d(in_channels, out_channels, 1, 1, 0) if in_channels != out_channels else nn.Identity()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if bias==False else nn.Identity()
        if self.use_se:
            self.se = SE_Layer(out_channels, se_ratio=se_ratio, bias=bias)

    def forward(self, x):
        out = self.act(self.norm(self.conv_1(x)))
        out = self.act(self.conv_2(out))
        if self.use_se:
            out = self.se(out)
        out += self.identity(x)
        return out


class RSHBlock(nn.Module):
    def __init__(self, local_dim, global_dim, head_dim, window_size, drop_path_rate, use_se=True, se_ratio=0.25, bias=False, down_r=2):
        """ SwinTransformer and Conv Block
        """
        super(RSHBlock, self).__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.dim = local_dim + global_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.down_r = down_r

        self.config = [2,2]
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.config))]
        
        # first
        self.local_branch1 = ResBlock(self.local_dim, self.local_dim, use_se, se_ratio, bias=bias)
        self.global_branch1 = nn.Sequential(*[SwinTBlock(self.global_dim, self.global_dim, self.head_dim, self.window_size, self.dpr[0:2][i], 
                            'W' if not i%2 else 'SW', self.down_r) for i in range(2)]) if self.global_dim != 0 else nn.Identity()
        self.conv1_1 = nn.Conv2d(self.local_dim, self.global_dim, 3, 1, 1, groups=self.global_dim, bias=bias)
        self.conv1_2 = nn.Conv2d(self.local_dim, self.global_dim, 3, 1, 1, groups=self.global_dim, bias=bias)

        # second
        self.local_branch2 = ResBlock(self.global_dim, self.global_dim, use_se, se_ratio, bias=bias)
        self.global_branch2 = nn.Sequential(*[SwinTBlock(self.local_dim, self.local_dim, self.head_dim, self.window_size, self.dpr[2:][i], 
                            'W' if not i%2 else 'SW', self.down_r) for i in range(2)]) if self.local_dim != 0 else nn.Identity()
        self.conv2_1 = nn.Conv2d(self.global_dim, self.local_dim, 3, 1, 1, bias=bias)
        self.conv2_2 = nn.Conv2d(self.global_dim, self.local_dim, 3, 1, 1, bias=bias)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        res = x
        local_x0, global_x0 = torch.split(x, (self.local_dim, self.global_dim), dim=1)

        local_x1 = self.local_branch1(local_x0)
        gama1, beta1 = self.softmax(self.conv1_1(local_x1)), self.conv1_2(local_x1)
        global_x1 = self.global_branch1(global_x0)
        global_x1 = global_x1 * gama1 + beta1

        global_x2 = self.global_branch2(local_x1)
        gama2, beta2 = self.softmax(self.conv2_1(global_x2)), self.conv2_2(global_x2)
        local_x2 = self.local_branch2(global_x1)
        local_x2 = local_x2 * gama2 + beta2
        
        x = torch.cat((global_x2, local_x2), dim=1)
        x += res
        return x
    

class AIENet(nn.Module):
    def __init__(self, in_nc=3, channels=16, head_dim=4, window_size=16, drop_path_rate=0.1, use_se=True, se_ratio=0.25, bias=False, down_r=2):
        super(AIENet, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ResBlock(channels, channels, use_se, se_ratio, bias)
        )

        self.down = nn.Sequential(
            nn.Conv2d(channels, 2*channels, kernel_size=3, stride=2, padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        self.rsh_block = RSHBlock(channels, channels, head_dim, window_size, drop_path_rate, use_se=True, se_ratio=0.25, bias=False, down_r=down_r)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=in_nc, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, input):
        _, _, H0, W0 = input.shape
        x = self.in_conv(input)
        x = self.down(x)
        x = self.rsh_block(x)
        x = self.up(x)
        out = self.out_conv(x)
        illu = out + input
        return illu

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)