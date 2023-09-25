# DMPfold2 Neural Network definition - D.T.Jones 2020

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from math import sqrt, log, asin, cos, pi, sin
from timm.models.layers import DropPath
from einops import rearrange

NUM_CHANNELS = 442


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x



class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class ConvNeXtV2_Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x





class Maxout2d(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, kernel_size=1, dilation=1, block=0, use_convnext_v2_block=False, convnext_drop_path_ratio=0):
        super(Maxout2d, self).__init__()
        self.in_channels, self.out_channels, self.pool_size = in_channels, out_channels, pool_size

        if use_convnext_v2_block:
            self.lin = nn.Sequential(*[ConvNeXtV2_Block(in_channels,convnext_drop_path_ratio),
                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels * pool_size,
                                        kernel_size=kernel_size, dilation=dilation, padding=dilation*(kernel_size-1)//2)])
            
        else:
            self.lin = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * pool_size,
                             kernel_size=kernel_size, dilation=dilation, padding=dilation*(kernel_size-1)//2)

        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        if block > 0:
            if not isinstance(self.lin, nn.Conv2d):
                nn.init.xavier_uniform_(self.lin[1].weight, gain=1.0 / sqrt(block))
            else:
                nn.init.xavier_uniform_(self.lin.weight, gain=1.0 / sqrt(block))
        else:
            if not isinstance(self.lin, nn.Conv2d):
                nn.init.xavier_uniform_(self.lin[1].weight, gain=1.0)
            else:
                nn.init.xavier_uniform_(self.lin.weight, gain=1.0)

    def forward(self, inputs):
        x = self.lin(inputs)

        N, C, H, W = x.size()

        x = x.view(N, C//self.pool_size, self.pool_size, H, W)
        x = x.max(dim=2)[0]
        x = self.norm(x)

        return x



class Self_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        _,_,h,w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h , w=w)
        

        return out






class CSE(nn.Module):
    def __init__(self, width, reduction=16):
        super(CSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(width, width // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(width // reduction, width, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, _, _ = x.size()
        y = self.avg_pool(x).view(N, C)
        y = self.fc(y).view(N, C, 1, 1)

        return x * y.expand_as(x)


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):

        y = self.conv(x)
        y = torch.sigmoid(y)

        return x * y


class SCSE(nn.Module):
    def __init__(self, in_ch, r):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)

        return cSE + sSE


# ResNet Module
class ResNet_Block(nn.Module):
    def __init__(self,width,fsize,dilv,nblock,use_convnext_v2_block=False, convnext_drop_path_ratio=0, use_self_attention=False):
        super(ResNet_Block, self).__init__()
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.layer1 = Maxout2d(in_channels=width, out_channels=width, pool_size=4,
                               kernel_size=fsize, dilation=dilv, block=nblock, use_convnext_v2_block=use_convnext_v2_block, convnext_drop_path_ratio= convnext_drop_path_ratio)
        if not use_self_attention:
            self.attention = SCSE(width, 16)
        else:
            #  Self_Attention 可以调 heads  dim_head 以及 dropout # 
            self.attention = Self_Attention(width, heads = 4, dim_head = 32, dropout = 0.)



    def forward(self, x):

        residual = x
        out = self.dropout1(x)
        out = self.dropout2(out)
        out = self.layer1(out)
        out = self.attention(out)
        out = out + residual

        return out


def refine_coords(coords, n_steps):
    vdw_dist, cov_dist = 3.0, 3.78
    k_vdw, k_cov = 100.0, 100.0
    min_speed = 0.001

    for i in range(n_steps):
        n_res = coords.size(0)
        accels = coords * 0

        # Steric clashing
        crep = coords.unsqueeze(0).expand(n_res, -1, -1)
        diffs = crep - crep.transpose(0, 1)
        dists = diffs.norm(dim=2).clamp(min=0.01, max=10.0)
        norm_diffs = diffs / dists.unsqueeze(2)
        violate = (dists < vdw_dist).to(torch.float) * (vdw_dist - dists)
        forces = k_vdw * violate
        pair_accels = forces.unsqueeze(2) * norm_diffs
        accels += pair_accels.sum(dim=0)

        # Adjacent C-alphas
        diffs = coords[1:] - coords[:-1]
        dists = diffs.norm(dim=1).clamp(min=0.1)
        norm_diffs = diffs / dists.unsqueeze(1)
        violate = (dists - cov_dist).clamp(max=3.0)
        forces = k_cov * violate
        accels_cov = forces.unsqueeze(1) * norm_diffs
        accels[:-1] += accels_cov
        accels[1: ] -= accels_cov

        coords = coords + accels.clamp(min=-100.0, max=100.0) * min_speed

    return coords


# Calculate main chain and Cbeta atom from Calphas, based on code by W. Taylor using the Levitt method
def calpha_to_main_chain(coords_ca):
    # Place dummy extra Cα atoms on end to get the required vectors
    vec_can2_can1 = coords_ca[:, :1   , :] - coords_ca[:, 1:2  , :]
    vec_can2_can3 = coords_ca[:, 2:3  , :] - coords_ca[:, 1:2  , :]
    vec_cac2_cac1 = coords_ca[:, -1:  , :] - coords_ca[:, -2:-1, :]
    vec_cac2_cac3 = coords_ca[:, -3:-2, :] - coords_ca[:, -2:-1, :]
    coord_ca_nterm = coords_ca[:, :1 , :] + 3.82 * F.normalize(torch.cross(vec_can2_can1, vec_can2_can3), dim=2)
    coord_ca_cterm = coords_ca[:, -1:, :] + 3.82 * F.normalize(torch.cross(vec_cac2_cac1, vec_cac2_cac3), dim=2)
    coords_ca_ext = torch.cat((coord_ca_nterm, coords_ca, coord_ca_cterm), dim=1)

    vec_ca_can = coords_ca_ext[:, :-2, :] - coords_ca_ext[:, 1:-1, :]
    vec_ca_cac = coords_ca_ext[:, 2: , :] - coords_ca_ext[:, 1:-1, :]
    mid_ca_can = (coords_ca_ext[:, 1:, :] + coords_ca_ext[:, :-1, :]) / 2
    cross_vcan_vcac = F.normalize(torch.cross(vec_ca_can, vec_ca_cac, dim=2), dim=2)
    coords_n = mid_ca_can[:, :-1, :] - vec_ca_can / 8 + cross_vcan_vcac / 4
    #coords_h = mid_ca_can[:, :-1, :] + cross_vcan_vcac * 1.0
    coords_c_shift = mid_ca_can[:, :-1, :] + vec_ca_can / 8 - cross_vcan_vcac / 2
    coords_o_shift = mid_ca_can[:, :-1, :] - cross_vcan_vcac * 1.8
    # Add the final C and O
    coords_c_cterm = mid_ca_can[:, -1:, :] - vec_ca_cac[:, -1:, :] / 8 + cross_vcan_vcac[:, -1:, :] / 2
    coords_o_cterm = mid_ca_can[:, -1:, :] + cross_vcan_vcac[:, -1:, :] * 2.0
    coords_c = torch.cat((coords_c_shift[:, 1:, :], coords_c_cterm), dim=1)
    coords_o = torch.cat((coords_o_shift[:, 1:, :], coords_o_cterm), dim=1)

    vec_n_ca = coords_ca - coords_n
    vec_c_ca = coords_ca - coords_c
    cross_vn_vc = torch.cross(vec_n_ca, vec_c_ca, dim=2)
    vec_ca_cb = vec_n_ca + vec_c_ca
    ang = pi / 2 - asin(1 / sqrt(3))
    sx = (1.5 * cos(ang) /   vec_ca_cb.norm(dim=2)).unsqueeze(2)
    sy = (1.5 * sin(ang) / cross_vn_vc.norm(dim=2)).unsqueeze(2)
    coords_cb = coords_ca + sx * vec_ca_cb + sy * cross_vn_vc

    out = torch.cat((coords_n.unsqueeze(2), #coords_h.unsqueeze(2),
                        coords_ca.unsqueeze(2), coords_c.unsqueeze(2),
                        coords_o.unsqueeze(2), coords_cb.unsqueeze(2)), dim=2)
    return out.view(coords_ca.size(0), 5 * coords_ca.size(1), 3)


# RNNResNet Module
class GRUResNet(nn.Module):
    def __init__(self,width,cwidth):
        super(GRUResNet, self).__init__()

        self.width = width
        self.cwidth = cwidth

        self.embed = nn.Embedding.from_pretrained(torch.eye(22), freeze=True)
        self.vgru = nn.GRU(22, width, batch_first=False, num_layers=2, bidirectional=False)
        self.hgru = nn.GRU(width, width//2, batch_first=False, num_layers=2, dropout=0.1, bidirectional=True)

        layers = []
        # 这里 use_convnext_v2_block 是否在Maxout2d 引入 convnext_v2的 block 默认为 Flase convnext_drop_path_ratio 意思是 convnext_v2的 block的drop_path_ratio 默认为 0
        layer = Maxout2d(in_channels=NUM_CHANNELS+width+1, out_channels=cwidth, pool_size=3, use_convnext_v2_block=False,convnext_drop_path_ratio=0)

        layers.append(layer)

        nblock = 1

        for rep in range(16):
            for fsize,dilv in [(5,1)]:
                if fsize > 0:
                    # 这里 use_convnext_v2_block 是否在Maxout2d 引入 convnext_v2的 block 默认为 Flase convnext_drop_path_ratio 意思是 convnext_v2的 block的drop_path_ratio 默认为 0
                    layer = ResNet_Block(cwidth, fsize, dilv, nblock, use_convnext_v2_block=False, convnext_drop_path_ratio=0, use_self_attention =False)
                    layers.append(layer)
                    nblock += 1

        layer = nn.Conv2d(in_channels=cwidth, out_channels=2, kernel_size=1)
        
        layers.append(layer)

        self.resnet = nn.Sequential(*layers)

        self.coord_gru = nn.GRU(width+8, width//2, batch_first=True, num_layers=3, dropout=0.1, bidirectional=True)

        self.coord_fc = nn.Linear(width, 3, bias=False)

        
    def forward(self, x, x2, nloops=5, refine_steps=0):

        nseqs = x.size()[0]
        nres = x.size()[1]

        x = self.embed(x)
        x = self.vgru(x)[0]
        x = self.hgru(x[-1,:,:].unsqueeze(1))[0]
        mat1d = x.permute(1,2,0)
        x = mat1d.unsqueeze(2) * mat1d.unsqueeze(3)

        resinp = torch.cat((x, x2), dim=1)

        if x.requires_grad:
            # now call the checkpoint API and get the output
            x = checkpoint_sequential(self.resnet, 4, resinp)
        else:
            x = self.resnet(resinp)

        dm = x[:,0,:,:]
        conf = x[:,1,:,:].mean(dim=2)
        best_conf = conf

        # Ensure symmetric matrix
        dm = (dm + dm.transpose(1, 2)) / 2
        # Force values to be non-negative
        dm = torch.abs(dm)
        # See https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
        M = 0.5 * (dm[:, 0:1, :].expand(-1, nres, -1) ** 2 + dm[:, :, 0:1].expand(-1, -1, nres) ** 2 - dm ** 2)
        # torch 1.8
        w, v = torch.symeig(M.float(), eigenvectors=True)
        # torch 2.0
        # w, v = torch.linalg.eigh(M.float())

        w = torch.clamp(F.relu(w, inplace=False), min = 1e-8)
        w = torch.diag_embed(w.sqrt())
        mds_coords = torch.matmul(v, w)[:, :, -8:]
        coordembed = torch.cat((mat1d.permute(0,2,1), mds_coords), dim=2)

        gru_out = self.coord_gru(coordembed)[0]

        ca_coords = self.coord_fc(gru_out)

        if refine_steps > 0:
            ca_coords = refine_coords(ca_coords.squeeze(0), refine_steps).unsqueeze(0)

        best_coords = ca_coords

        #print(best_conf.sum().item())

        for i in range(nloops):

            #print(ca_coords.size())

            # Add small amount of noise to seed coordinates
            #ca_coords = ca_coords + 0.5 * torch.randn_like(ca_coords)
            #if refine_steps > 0:
            #    ca_coords = refine_coords(ca_coords.squeeze(0), refine_steps).unsqueeze(0)
            dmap = torch.clamp((ca_coords - ca_coords.transpose(0,1)).pow(2).sum(dim=2), min = 1e-8).sqrt()
            resinp = torch.cat((resinp[:, :-1], dmap.unsqueeze(0).unsqueeze(0)), dim=1)

            if resinp.requires_grad:
                # now call the checkpoint API and get the output
                x = checkpoint_sequential(self.resnet, 4, resinp)
            else:
                x = self.resnet(resinp)

            dm = x[:,0,:,:]
            conf = x[:,1,:,:].mean(dim=2)

            #print(conf.sum().item())

            # Ensure symmetric matrix
            dm = (dm + dm.transpose(1, 2)) / 2
            # Force values to be non-negative
            dm = torch.abs(dm)
            # See https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
            M = 0.5 * (dm[:, 0:1, :].expand(-1, nres, -1) ** 2 + dm[:, :, 0:1].expand(-1, -1, nres) ** 2 - dm ** 2)
            
            # torch 1.8
            w, v = torch.symeig(M.float(), eigenvectors=True)
            # torch 2.0
            # w, v = torch.linalg.eigh(M.float())


            w = torch.clamp(F.relu(w, inplace=False), min = 1e-8)
            w = torch.diag_embed(w.sqrt())
            mds_coords = torch.matmul(v, w)[:, :, -8:]
            coordembed = torch.cat((mat1d.permute(0,2,1), mds_coords), dim=2)

            gru_out = self.coord_gru(coordembed)[0]

            ca_coords = self.coord_fc(gru_out)
            if conf.mean() > best_conf.mean():
                #if refine_steps > 0:
                #    ca_coords = refine_coords(ca_coords.squeeze(0), refine_steps).unsqueeze(0)
                best_conf = conf
                best_coords = ca_coords

        if refine_steps > 0:
            best_coords = refine_coords(best_coords.squeeze(0), refine_steps).unsqueeze(0)

        coords_out = calpha_to_main_chain(best_coords)
        conf_out = torch.sigmoid(best_conf)

        return coords_out, conf_out
