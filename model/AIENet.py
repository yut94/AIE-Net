import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import DropPath, trunc_normal_
from torch.nn import init


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
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
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = einops.rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)

        x = einops.rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = einops.rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + einops.rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = einops.rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = einops.rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]
    

class SwinTBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W'):
        """ SwinTransformer Block
        """
        super(SwinTBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
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

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

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


class ChannelAtten(nn.Module):
    def __init__(self, dim, head_dim, bias):
        super().__init__()
        
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = einops.rearrange(q, 'b (nh hd) h w -> b nh hd (h w)', hd=self.head_dim)
        k = einops.rearrange(k, 'b (nh hd) h w -> b nh hd (h w)', hd=self.head_dim)
        v = einops.rearrange(v, 'b (nh hd) h w -> b nh hd (h w)', hd=self.head_dim)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = einops.rearrange(out, 'b nh hd (h w) -> b (nh hd) h w', hd=self.head_dim, h=h)

        out = self.project_out(out)
        return out


class RSHBlock(nn.Module):
    def __init__(self, local_dim, global_dim, head_dim, window_size, drop_path_rate, use_se=True, se_ratio=0.25, bias=False):
        """ SwinTransformer and Conv Block
        """
        super(RSHBlock, self).__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.dim = local_dim + global_dim
        self.head_dim = head_dim
        self.window_size = window_size

        self.config = [2,2]
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.config))]

        self.local_branch = ResBlock(self.local_dim, self.local_dim, use_se, se_ratio, bias=bias)
 
        self.global_branch = nn.Sequential(*[SwinTBlock(self.global_dim, self.global_dim, self.head_dim, self.window_size, self.dpr[0:2][i], 'W' if not i%2 else 'SW') for i in range(2)]) if self.global_dim != 0 else nn.Identity()
       
        self.CA = ChannelAtten(self.dim, self.head_dim, bias)

    def forward(self, x):
        res = x
        local_x, global_x = torch.split(x, (self.local_dim, self.global_dim), dim=1)

        local_x = self.local_branch(local_x)
        global_x = self.global_branch(global_x)
        x = torch.cat((local_x, global_x), dim=1) #bchw
        x = self.CA(x) + x
        
        x += res
        return x
    

class CVTEnhancer(nn.Module):
    def __init__(self, in_nc=3, channels=16, head_dim=4, window_size=8, drop_path_rate=0.1, 
                 use_se=True, se_ratio=0.25, bias=False):
        super(CVTEnhancer, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ResBlock(channels, channels, use_se, se_ratio, bias)
        )

        self.down = nn.Sequential(
            nn.Conv2d(channels, 2*channels, kernel_size=3, stride=2, padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        self.rsh_block1 = RSHBlock(channels, channels, head_dim, window_size, drop_path_rate)
        self.rsh_block2 = RSHBlock(channels, channels, head_dim, window_size, drop_path_rate)

        self.up = nn.Sequential(
            # nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, padding=1, bias=bias),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(2*channels, channels*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def forward(self, input):
        _, _, H0, W0 = input.shape
        x = self.in_conv(input)
        x = self.down(x)
        x = self.rsh_block1(x)
        x = self.rsh_block2(x)
        # x_visualize = x
        # x_visualize = F.interpolate(x_visualize, size=(H0, W0), mode='bilinear', align_corners=True)
        # x = F.interpolate(x, size=(H0, W0), mode='bilinear', align_corners=True) # H, W 3c
        x = self.up(x)
        out = self.out_conv(x)
        
        illu = out + input
        ref = torch.div(input, illu + 1e-4)
        # ref = torch.clamp(ref, 0, 1)
        return illu, ref

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


if __name__ == "__main__":
    model = CVTEnhancer().cuda()
    x = torch.randn(1,3,640,640).cuda()
    n,c,h,w = x.shape
    padh = (h//2**6 + 1)*2**6-h if h%2**6 != 0 else 0
    padw = (w//2**6 + 1)*2**6-w if w%2**6 != 0 else 0
    pad_x = torch.nn.functional.pad(x, pad=(padw//2, padw-padw//2, padh//2, padh-padh//2))
    print(x.shape, "===>>", pad_x.shape)
    y,z = model(pad_x)
    print(y.shape)