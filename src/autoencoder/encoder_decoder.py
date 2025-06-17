# MODEL 
import torch.nn as nn
import torch
import numpy as np 
import math
from typing import Union
from omegaconf import ListConfig

def prod(l:list,to_dtype=int):
    if len(l) > 0:
        p = 1.0
        for ele in l: 
            p *= ele
        return to_dtype(p)
    else:
        return 0

def tconv_out_dim(input_dim, kernel_size, stride, padding, output_padding=0):
    if isinstance(input_dim, list) or isinstance(input_dim, ListConfig):
        if not (isinstance(output_padding, list) or isinstance(output_padding, ListConfig)):
            output_padding = [output_padding] * len(input_dim)
        assert len(output_padding) == len(input_dim)
        out_dim = [tconv_out_dim(in_dim, kernel_size, stride, padding, out_padding) for in_dim, out_padding in zip(input_dim, output_padding)]
    else:
        out_dim = (input_dim-1)*stride-2*padding+kernel_size+output_padding
    return out_dim


def conv_out_dim(input_dim, kernel_size, stride, padding):
    if (isinstance(input_dim, list) or isinstance(input_dim, ListConfig)):
        out_dim = [conv_out_dim(in_dim, kernel_size, stride, padding) for in_dim in input_dim]
    else:
        out_dim = math.floor((input_dim-kernel_size+2*padding)/stride)+1
    return out_dim


class coder(nn.Module):
    def configure_dims(self, dims:Union[int, list]):
        if not (isinstance(dims, list) or isinstance(dims, ListConfig)):
            dims = [dims]
        self.dim = len(dims)
        return dims

    def forward(self, x, t=None):
        if t is None:
            return self.blocks(x) # uncondition
        else:
            x = torch.cat([x,t])
            return self.blocks(x)

# different format of a normal feedforward network
class FCEncoder(coder):
    def __init__(self, 
        in_dims, 
        latents_dims, 
        out_widths, 
        actvn=nn.ReLU(),
        ):
        super().__init__()

        in_dims = self.configure_dims(in_dims)
        out_widths.insert(0, prod(in_dims))
        out_widths.append(latents_dims)

        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(out_widths[i-1], out_widths[i]),
                actvn,
            ) for i in range(1,len(out_widths)-1)],
            nn.Linear(out_widths[-2], out_widths[-1])
        )

class FCDecoder(coder):
    def __init__(self, out_dims, latents_dims, in_widths, actvn=nn.ReLU(), in_channel=1, dim=2, uncond=True, **kwargs):
        super().__init__()
        if not uncond:
            latents_dims += 1
        out_dims = self.configure_dims(out_dims)
        in_widths.append(prod(out_dims))
        in_widths.insert(0, latents_dims)
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(in_widths[i], in_widths[i+1]),
                actvn
            ) for i in range(len(in_widths)-2)],
            nn.Linear(in_widths[-2], in_widths[-1])
            )



class ConvEncoder(coder):
    def __init__(self, in_dims, latents_dims, out_channels, kernel_stride_paddings, actvn=nn.ReLU(), in_channel=1, padding_mode="zeros", mlp=True, **kwargs):
        super().__init__()
        in_dims = self.configure_dims(in_dims)
        self.configure_conv_dims(in_dims, kernel_stride_paddings)

        conv = nn.Conv2d if self.dim == 2 else nn.Conv1d
        out_channels.insert(0, in_channel)

        if mlp:
            self.blocks = nn.Sequential(
                *[nn.Sequential(
                    conv(out_channels[i], out_channels[i+1], ksp[0], ksp[1], ksp[2], padding_mode=padding_mode),
                    actvn
                ) for i, ksp in enumerate(kernel_stride_paddings)],
                nn.Flatten(),
                nn.Linear(out_channels[-1]*prod(self.conv_layers_config[-1]["out_dims"]), 512),
                actvn,
                nn.Linear(512, latents_dims)
            )
        else:
            self.blocks = nn.Sequential(
                *[nn.Sequential(
                    conv(out_channels[i], out_channels[i+1], ksp[0], ksp[1], ksp[2], padding_mode=padding_mode),
                    actvn
                ) for i, ksp in enumerate(kernel_stride_paddings)],
                nn.Flatten(),
                nn.Linear(out_channels[-1]*prod(self.conv_layers_config[-1]["out_dims"]), latents_dims),
            )
    
    def configure_conv_dims(self, in_dims, kernel_stride_paddings):
        self.conv_layers_config = []
        for i, ksp in enumerate(kernel_stride_paddings):
            in_dims = in_dims if i == 0 else self.conv_layers_config[-1]["out_dims"]
            self.conv_layers_config.append(dict(
                in_dims=in_dims, 
                out_dims=conv_out_dim(in_dims, ksp[0], ksp[1], ksp[2]),
                kernel_size=ksp[0],
                stride=ksp[1],
                padding=ksp[2],
            ))


class ConvDecoder(coder):
    def __init__(self, out_dims, latents_dims, in_channels, kernel_stride_paddings, actvn=nn.ReLU(), out_channel=1, uncond=True, padding_mode="zeros", mlp=True, **kwargs):
        super().__init__()
        if not uncond:
            latents_dims += 1
        out_dims = self.configure_dims(out_dims)
        outpaddings = self.configure_tconv_dims(out_dims, kernel_stride_paddings)

        tconv = nn.ConvTranspose2d if self.dim == 2 else nn.ConvTranspose1d
        in_channels.append(out_channel)

        if mlp:
            self.blocks = nn.Sequential(
                nn.Linear(latents_dims, 512),
                actvn,
                nn.Linear(512, in_channels[0]*prod(self.tconv_layers_config[0]["in_dims"])),
                nn.Unflatten(1, (in_channels[0], *self.tconv_layers_config[0]["in_dims"])),
                *[nn.Sequential(
                    actvn,
                    tconv(in_channels[i], in_channels[i+1], ksp[0], ksp[1], ksp[2], outpadding, padding_mode=padding_mode)
                ) for i, (ksp, outpadding) in enumerate(zip(kernel_stride_paddings, outpaddings))]
            )
        else:
            self.blocks = nn.Sequential(
                nn.Linear(latents_dims, in_channels[0]*prod(self.tconv_layers_config[0]["in_dims"])),
                nn.Unflatten(1, (in_channels[0], *self.tconv_layers_config[0]["in_dims"])),
                *[nn.Sequential(
                    actvn,
                    tconv(in_channels[i], in_channels[i+1], ksp[0], ksp[1], ksp[2], outpadding, padding_mode=padding_mode)
                ) for i, (ksp, outpadding) in enumerate(zip(kernel_stride_paddings, outpaddings))]
            )
        
    def configure_tconv_dims(self, out_dims, kernel_stride_paddings):
        self.tconv_layers_config = []
        for i, ksp in enumerate(kernel_stride_paddings[::-1]):
            out_dims = out_dims if i == 0 else self.tconv_layers_config[0]["in_dims"]
            self.tconv_layers_config.insert(0, dict(
                in_dims=conv_out_dim(out_dims, ksp[0], ksp[1], ksp[2]), 
                out_dims=out_dims,
                kernel_size=ksp[0],
                stride=ksp[1],
                padding=ksp[2],
            ))
        outpaddings = self.get_outpaddings()
        return outpaddings
    
    def get_outpaddings(self):
        outpaddings = []
        for config in self.tconv_layers_config:
            outpaddings.append([out_dim - out_dim_no_padding 
                       for (out_dim, out_dim_no_padding) in zip(config["out_dims"], tconv_out_dim(config["in_dims"], config["kernel_size"], config["stride"], config["padding"]))
                        ])
        return outpaddings


class ResnetEncoder(ConvEncoder):
    def __init__(self, in_dims, latents_dims, out_channels, kernel_stride_paddings, actvn=nn.ReLU(), in_channel=1, gn_groups=4, **kwargs):
        super(ConvEncoder, self).__init__()
        in_dims = self.configure_dims(in_dims)
        self.configure_conv_dims(in_dims, kernel_stride_paddings)

        conv = nn.Conv2d if self.dim == 2 else nn.Conv1d
        out_channels.insert(0, in_channel)

        self.blocks = nn.Sequential(
            *[nn.Sequential(
                conv(out_channels[i], out_channels[i+1], ksp[0], ksp[1], ksp[2]),
                actvn,
                ResBlock(out_channels[i+1], out_channels[i+1], self.dim, groups=gn_groups),
                ResBlock(out_channels[i+1], out_channels[i+1], self.dim, groups=gn_groups)
            ) for i, ksp in enumerate(kernel_stride_paddings)],
            nn.Flatten(),
            nn.Linear(out_channels[-1]*prod(self.conv_layers_config[-1]["out_dims"]), 512),
            actvn,
            nn.Linear(512, latents_dims)
        )

class ResnetDecoder(ConvDecoder):
    def __init__(self, out_dims, latents_dims, in_channels, kernel_stride_paddings, actvn=nn.ReLU(), out_channel=1, gn_groups=4, **kwargs):
        super(ConvDecoder, self).__init__()
        out_dims = self.configure_dims(out_dims)
        outpaddings = self.configure_tconv_dims(out_dims, kernel_stride_paddings)

        tconv = nn.ConvTranspose2d if self.dim == 2 else nn.ConvTranspose1d
        in_channels.append(out_channel)

        self.blocks = nn.Sequential(
            nn.Linear(latents_dims, 512),
            actvn,
            nn.Linear(512, in_channels[0]*prod(self.tconv_layers_config[0]["in_dims"])),
            nn.Unflatten(1, (in_channels[0], *self.tconv_layers_config[0]["in_dims"])),
            *[nn.Sequential(
                actvn,
                ResBlock(in_channels[i], in_channels[i], self.dim, groups=gn_groups),
                ResBlock(in_channels[i], in_channels[i], self.dim, groups=gn_groups),
                tconv(in_channels[i], in_channels[i+1], ksp[0], ksp[1], ksp[2], outpadding)
            ) for i, (ksp, outpadding) in enumerate(zip(kernel_stride_paddings, outpaddings))]
        )

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dim=1, act=nn.ReLU(), groups=4):
        super().__init__()
        conv = nn.Conv2d if dim == 2 else nn.Conv1d
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = conv(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(groups, in_channels)
        self.conv2 = conv(in_channels, out_channels, 3, 1, 1)
        self.act = act
        self.skip = conv(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.skip(x)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = x+y
        x = self.act(x)

        return x

class LSTMEncoder(coder):
    def __init__(self, in_dims, latents_dims, num_layers=1, **kwargs):
        super().__init__()
        in_dims = self.configure_dims(in_dims)
        self.latents_dims, self.num_layers = latents_dims, num_layers
        self.lstm = nn.LSTM(
            prod(in_dims), 
            latents_dims, 
            num_layers=num_layers, 
            bias=True, 
            batch_first=True, 
            dropout=0.0, 
            bidirectional=False, 
            proj_size=0, 
            device=None)
        
    def _forward_lstm(self, x):
        hidden_state = []
        prev_h = torch.rand(self.num_layers,x.shape[0],self.latents_dims)
        prev_c = torch.rand(self.num_layers,x.shape[0],self.latents_dims)
        for s in range(x.shape[1]):
            out, (prev_h, prev_c) = self.lstm(x[:,s:s+1,:], (prev_h, prev_c))
            hidden_state.append(prev_h[-1:,:,:])
        return torch.cat(hidden_state, dim=0).transpose(0,1) # B, S, latents_dims
        
    def forward(self, x):
        x = x.view(*x.shape[:2], -1) # B, S, HxW
        x = self._forward_lstm(x)
        return x # B, S, latent dimension



class LSTMDecoder(coder):
    def __init__(self, out_dims, latents_dims, num_layers=1, **kwargs):
        super().__init__()
        out_dims = self.configure_dims(out_dims)
        self.num_layers, self.out_dims = num_layers, out_dims
        self.lstm = nn.LSTM(
            latents_dims, 
            prod(out_dims), 
            num_layers=num_layers, 
            bias=True, 
            batch_first=True, 
            dropout=0.0, 
            bidirectional=False, 
            proj_size=0, 
            device=None)
        
    def _forward_lstm(self, x):
        hidden_state = []
        prev_h = torch.rand(self.num_layers,x.shape[0],prod(self.out_dims))
        prev_c = torch.rand(self.num_layers,x.shape[0],prod(self.out_dims))
        for s in range(x.shape[1]):
            out, (prev_h, prev_c) = self.lstm(x[:,s:s+1,:], (prev_h, prev_c))
            hidden_state.append(prev_h[-1:,:,:])
        return torch.cat(hidden_state, dim=0).transpose(0,1) # B, S, latents_dims
        
    def forward(self, x):
        x = self._forward_lstm(x) # B, S, latent dimension
        x = x.view(*x.shape[:2], *self.out_dims) # B, S, HxW
        return x

class ConvLSTMEncoder(ConvEncoder):
    def __init__(self, in_dims, latents_dims, out_channels, kernel_stride_paddings, actvn=nn.ReLU(), in_channel=1, **kwargs):
        super(ConvEncoder, self).__init__()
        in_dims = self.configure_dims(in_dims)
        self.configure_conv_dims(in_dims, kernel_stride_paddings)

        conv = nn.Conv2d if self.dim == 2 else nn.Conv1d
        out_channels.insert(0, in_channel)

        self.blocks = nn.Sequential(
            *[nn.Sequential(
                conv(out_channels[i], out_channels[i+1], ksp[0], ksp[1], ksp[2]),
                actvn
            ) for i, ksp in enumerate(kernel_stride_paddings)],
            nn.Flatten(),
        )

        self.lstm = LSTMEncoder(in_dims=out_channels[-1]*prod(self.conv_layers_config[-1]["out_dims"]), latents_dims=latents_dims)
    
    def forward(self, x):
        # B, S, H, W
        data_shape = x.shape
        x = x.view(-1, 1, *data_shape[2:])
        x = self.blocks(x)
        x = x.view(*data_shape[:2], -1)
        x = self.lstm(x)
        return x

class ConvLSTMDecoder(ConvDecoder):
    def __init__(self, out_dims, latents_dims, in_channels, kernel_stride_paddings, actvn=nn.ReLU(), out_channel=1, **kwargs):
        super(ConvDecoder, self).__init__()
        out_dims = self.configure_dims(out_dims)
        outpaddings = self.configure_tconv_dims(out_dims, kernel_stride_paddings)

        tconv = nn.ConvTranspose2d if self.dim == 2 else nn.ConvTranspose1d
        in_channels.append(out_channel)

        self.lstm = LSTMDecoder(out_dims=in_channels[0]*prod(self.tconv_layers_config[0]["in_dims"]), latents_dims=latents_dims)
        self.blocks = nn.Sequential(
            nn.Unflatten(1, (in_channels[0], *self.tconv_layers_config[0]["in_dims"])),
            *[nn.Sequential(
                actvn,
                tconv(in_channels[i], in_channels[i+1], ksp[0], ksp[1], ksp[2], outpadding)
            ) for i, (ksp, outpadding) in enumerate(zip(kernel_stride_paddings, outpaddings))]
        )
    def forward(self, x):
        # B, S, latents_dim
        data_shape = x.shape
        x = self.lstm(x)
        x = x.view(-1, 1, x.shape[-1])
        x = self.blocks(x)
        x = x.view(*data_shape[:2], -1)
        return x