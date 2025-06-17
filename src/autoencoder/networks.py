import torch.nn as nn
from math import sqrt
from omegaconf import DictConfig, OmegaConf, ListConfig
from .encoder_decoder import ConvEncoder, ConvDecoder, ResnetEncoder, ResnetDecoder, LSTMEncoder, LSTMDecoder, FCEncoder, FCDecoder, ConvLSTMEncoder, ConvLSTMDecoder
from common.act_fn import get_actvn
from typing import Union

class AutoEncoder(nn.Module):
    def encode(self, x):
        if self.dim == 1:
            if len(x.shape) == 2:
                x = x.unsqueeze(0) # batch = 1
                S = x.shape[1]
                return self.encoder(x.reshape(-1,1,x.shape[-1])).reshape(S,-1)
            else:
                B = x.shape[0]
                S = x.shape[1]
                return self.encoder(x.reshape(-1,1,x.shape[-1])).reshape(B, S,-1)
        else:
            if len(x.shape) == 3:
                x = x.unsqueeze(0) # batch = 1
                B, S, h, w = x.shape
                out = self.encoder(x.reshape(B*S, 1, h, w))
                return out.reshape(S, -1)
            else:
                B, S, h, w = x.shape
                out = self.encoder(x.reshape(B*S, 1, h, w))
                return out.reshape(B, S, -1)
    
    def decode(self, x, t=None):
        if self.dim == 1:
            if len(x.shape) == 2:
                x = x.unsqueeze(0) # batch = 1
                S = x.shape[1]
                return self.decoder(x.reshape(-1,x.shape[-1]), t).reshape(S,-1)
            else:
                B = x.shape[0]
                S = x.shape[1]
                return self.decoder(x.reshape(-1,x.shape[-1]), t).reshape(B, S,-1)
        else:
            if len(x.shape) == 2:
                S = x.shape[0]
                out = self.decoder(x.reshape(-1,x.shape[-1]), t)
                h_out, w_out = out.shape[-2:]
                return out.reshape(S, h_out, w_out)
            else:
                B = x.shape[0]
                S = x.shape[1]
                out = self.decoder(x.reshape(-1,x.shape[-1]), t)
                h_out, w_out = out.shape[-2:]
                return out.reshape(B, S, h_out, w_out)
        
    def forward(self, x, t=None):
        return self.decode(self.encode(x), t)
    




class FCAutoEncoder(AutoEncoder):
    def __init__(self, config:DictConfig):
        super().__init__()
        in_dims = config.sample.spatial_resolution
        latents_dims = config.sample.latents_dims
        out_widths = config.downblocks.widths
        actvn = get_actvn(config.downblocks.actvn)

        self.encoder = FCEncoder(
            in_dims=in_dims,
            latents_dims=latents_dims,
            out_widths=out_widths,
            actvn=actvn
        )
        self.decoder = FCDecoder(
            out_dims=in_dims,
            latents_dims=latents_dims,
            in_widths=out_widths[::-1],
            actvn=actvn
        )


class ConvAutoEncoder(AutoEncoder):
    def __init__(self, config:DictConfig):
        super().__init__()
        # downblocks encoder configuration
        in_dims = config.sample.spatial_resolution
        latents_dims = config.sample.latents_dims
        out_channels = config.downblocks.channels
        kernel_stride_paddings = config.downblocks.kernel_stride_paddings
        actvn = get_actvn(config.downblocks.actvn)
        padding_mode = config.downblocks.get("padding_mode", 'zeros')
        self.dim = 1 if not (isinstance(in_dims, list) or isinstance(in_dims, ListConfig)) else len(in_dims)
        
        mlp = config.get("mlp", True)

        self.encoder = ConvEncoder(
            in_dims=in_dims, 
            latents_dims=latents_dims, 
            out_channels=out_channels, 
            kernel_stride_paddings=kernel_stride_paddings, 
            actvn=actvn,
            padding_mode=padding_mode,
            mlp=mlp
        )
        self.decoder = ConvDecoder(
            out_dims=in_dims, 
            latents_dims=latents_dims, 
            in_channels=out_channels[::-1],
            kernel_stride_paddings=kernel_stride_paddings[::-1], 
            actvn=actvn,
            padding_mode=padding_mode,
            mlp=mlp
        )

class ConvAutoEncoder2D(ConvAutoEncoder):
    def __init__(self, config:DictConfig):
        in_dims = config.sample.spatial_resolution
        config.sample.spatial_resolution = [int(sqrt(in_dims)), int(sqrt(in_dims))] if not (isinstance(in_dims, list) or isinstance(in_dims, ListConfig)) else in_dims
        self.dim = 1 if not (isinstance(in_dims, list) or isinstance(in_dims, ListConfig)) else len(in_dims)
        super().__init__(config)
        
    def hacky_unflatten(self, x):
        print(x.shape)
        return x.reshape(*x.shape[:-1], int(sqrt(x.shape[-1])), int(sqrt(x.shape[-1])))
    
    def hacky_flatten(self, x):
        print(x.shape)
        return x.reshape(*x.shape[:-2], x.shape[-2]*x.shape[-1])
    
    def encode(self, x):
        return super().encode(x)
    
    def decode(self, x, t=None):
        return super().decode(x, t)



class ResnetAutoEncoder(AutoEncoder):
    def __init__(self, config:DictConfig):
        super().__init__()
        # downblocks encoder configuration
        in_dims = config.sample.spatial_resolution
        latents_dims = config.sample.latents_dims
        out_channels = config.downblocks.channels
        kernel_stride_paddings = config.downblocks.kernel_stride_paddings
        actvn = get_actvn(config.downblocks.actvn)
        gn_groups = config.downblocks.gn_groups
        
        self.encoder = ResnetEncoder(
            in_dims=in_dims, 
            latents_dims=latents_dims, 
            out_channels=out_channels, 
            kernel_stride_paddings=kernel_stride_paddings, 
            actvn=actvn,
            gn_groups=gn_groups
        )
        self.decoder = ResnetDecoder(
            out_dims=in_dims, 
            latents_dims=latents_dims, 
            in_channels=out_channels[::-1],
            kernel_stride_paddings=kernel_stride_paddings[::-1], 
            actvn=actvn,
            gn_groups=gn_groups
        )
class LSTMAutoEncoder(AutoEncoder):
    def __init__(self, config:DictConfig):
        super().__init__()
        in_dims = config.sample.spatial_resolution
        latents_dims = config.sample.latents_dims
        self.encoder = LSTMEncoder(
            in_dims=in_dims,
            latents_dims=latents_dims,
        )
        self.decoder = LSTMDecoder(
            out_dims=in_dims,
            latents_dims=latents_dims,
        )
    def encode(self, x):
        if len(x.shape) == 2:
            x = x.view(-1,*x.shape)
        return self.encoder(x)
    
    def decode(self, x, t=None):
        out = self.decoder(x)
        if out.shape[0] == 1:
            out = out.view(*out.shape[1:])
        return out
        
    def forward(self, x, t=None):
        return self.decode(self.encode(x))

class ConvLSTMAutoEncoder(AutoEncoder):
    def __init__(self, config:DictConfig):
        super().__init__()
        
        in_dims = config.sample.spatial_resolution
        latents_dims = config.sample.latents_dims
        out_channels = config.downblocks.channels
        kernel_stride_paddings = config.downblocks.kernel_stride_paddings
        actvn = get_actvn(config.downblocks.actvn)
        
        self.encoder = ConvLSTMEncoder(
            in_dims=in_dims,
            latents_dims=latents_dims,
            out_channels=out_channels,
            kernel_stride_paddings=kernel_stride_paddings,
            actvn=actvn
        )
        self.decoder = ConvLSTMDecoder(
            out_dims=in_dims,
            latents_dims=latents_dims,
            out_channels=out_channels,
            kernel_stride_paddings=kernel_stride_paddings,
            actvn=actvn

        )