# Setup code

import time
import glob
import itertools
import datetime
import copy
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import seaborn as sns
import deepxde as dde
import matplotlib.cm as cm
import ruptures as rpt
from concurrent.futures import ThreadPoolExecutor

from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from copy import deepcopy

from omegaconf import DictConfig, OmegaConf, ListConfig

import utils
from omegaconf import DictConfig, OmegaConf


plt.rcParams["figure.figsize"] = (7, 3)

BASEDIR = "savedmodels/ae"

from common.config import create_object, load_config
from jcmodels.networks import FCAutoEncoder, ConvAutoEncoder, ConvAutoEncoder2D, ResnetAutoEncoder, LSTMAutoEncoder, ConvEncoder, ConvLSTMAutoEncoder
JC_Modules = [FCAutoEncoder, ConvAutoEncoder, ConvAutoEncoder2D, ResnetAutoEncoder, LSTMAutoEncoder, ConvEncoder, ConvLSTMAutoEncoder]

def get_activation(activation):
  if activation.lower() == "relu":
    return nn.ReLU()
  else:
    return nn.ReLU()

def determine_param(dataset, encoding_param):
  if encoding_param == -1:
    encoding_param = []
    P = dataset.params.shape[1]

    for p in range(P):
      if np.abs(dataset.params[0, p] - dataset.params[1, p]) > 0:
        encoding_param.append(p)
        
  else:
    return encoding_param

class FFNet(nn.Module):
  def __init__(self, seq, activation):
    super().__init__()

    self.layers = nn.ModuleList([nn.Linear(seq[i], seq[i+1]) for i in range(len(seq) - 1)])
    
    if type(activation) == type(""):
      activation = get_activation(activation)
      
    self.s = activation

  def forward(self, x):
    for i, layer in enumerate(self.layers):
        x = layer(x)
        if not i == len(self.layers) - 1:
            x = self.s(x)
    return x

class DeepONet(nn.Module):
  def __init__(self, branchseq, trunkseq, activation):
    super().__init__()

    self.branchnet = nn.ModuleList([nn.Linear(branchseq[i], branchseq[i+1]) for i in range(len(branchseq) - 1)])
    self.trunknet = nn.ModuleList([nn.Linear(trunkseq[i], trunkseq[i+1]) for i in range(len(trunkseq) - 1)])
    self.s = activation

  def forward(self, u, x):
    for i, layer in enumerate(self.branchnet):
      u = layer(u)
      if not i == len(self.branchnet) - 1:
        u = self.s(u)

    for i, layer in enumerate(self.trunknet):
      x = layer(x)
      if not i == len(self.trunknet) - 1:
        x = self.s(x)

    out = torch.einsum("nmk,tk->nmt", x, u).transpose(0, 2)
    return out

# Feedforward NN
class PCAAutoencoder(nn.Module):
    def __init__(self, inputdim, reduced, datadim=1):
        super().__init__()

        self.inputdim = inputdim
        self.reduced = reduced
        self.datadim = datadim

        self.pcaTensor = torch.zeros((inputdim, reduced))
        self.pcaCenter = torch.zeros((inputdim))

    def train_pca(self, data):
        data = data.reshape([-1] + list(data.shape[-self.datadim:]))

        pca = PCA(n_components=self.reduced)
        pca = pca.fit(data)
        
        self.pcaTensor = torch.tensor(pca.components_, dtype=torch.float32)
        self.pcaCenter = torch.tensor(pca.mean_, dtype=torch.float32)

    def forward(self, x):
        return self.decode(self.encode(x))
    
    def encode(self, enc):
        preshape = list(enc.shape[:-self.datadim])
        enc = enc.reshape([-1] + list(enc.shape[-self.datadim:]))

        out = torch.matmul(enc - self.pcaCenter, self.pcaTensor.T)
        return out.reshape(preshape + [-1])
    
    def decode(self, dec):
        dec = torch.tensor(dec, device=self.pcaTensor.device, dtype=torch.float32)
        dec = torch.matmul(dec, self.pcaTensor) + self.pcaCenter
        dec = dec.reshape([-1] + list(dec.shape[-self.datadim:]))

        return dec

# Feedforward NN
class FFAutoencoder(nn.Module):
    def __init__(self, encodeSeq, decodeSeq, activation, datadim=1):
        super().__init__()

        self.encoder = FFNet(activation=activation, seq=encodeSeq)
        self.decoder = FFNet(activation=activation, seq=decodeSeq)
        self.s = activation

        self.reduced = encodeSeq[-1]
        self.datadim = datadim

        #assert(self.reduced == decodeSeq[0])
         
        self.pca = False

    def add_pca(self, data):
        assert(not self.pca)
        self.pca = True

        pca = PCA(n_components=self.reduced)
        pca = pca.fit(data)
        
        self.pcaTensor = torch.tensor(pca.components_).double()
        self.pcaCenter = torch.tensor(pca.mean_).double()

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, enc):
        if self.datadim == 2:
          enc = enc.reshape(list(enc.shape[:-2]) + [-1])
        if self.pca:
          return self.encoder(enc) + torch.matmul(enc - self.pcaCenter, self.pcaTensor.T)
        else:
          return self.encoder(enc)
    
    def decode(self, dec):
        if self.pca:
          decoded = self.decoder(dec) + torch.matmul(dec, self.pcaTensor) + self.pcaCenter
        else:
          decoded = self.decoder(dec)
        
        if self.datadim == 2:
          sqrt = int(np.sqrt(decoded.shape[-1]))
          decoded = decoded.reshape(list(dec.shape[:-1]) + [sqrt, -1])

        return decoded
    
    def save_model(self, filename):
        addr = f"{BASEDIR}/{filename}"

        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(addr, "wb") as handle:
            pickle.dump({"model": self}, handle, protocol=pickle.HIGHEST_PROTOCOL)

class FFVAE(nn.Module):
    def __init__(self, encodeSeq, decodeSeq, activation=nn.ReLU(), datadim=1, reg=0.1):
      super().__init__()
      self.activation = activation
      self.datadim = datadim
      self.reg = reg

      es = list(encodeSeq)
      self.latentdim = es[-1]
      es[-1] = 2*es[-1]

      self.reduced = self.latentdim

      self.encoder_net = FFNet(seq=es, activation=activation)

      self.decoder_net = FFNet(seq=decodeSeq, activation=activation)

    def encode(self, x, variance=False):
      if self.datadim == 2:
        batch = x.size(0)
        x = x.view(batch, -1)

      h = self.encoder_net(x)
      mu = h[..., :self.latentdim]
      logvar =  h[..., self.latentdim:]

      if variance:
          return mu, logvar
      else:
          return mu

    def reparameterize(self, mu, logvar):
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return mu + eps * std

    def decode(self, z):
      recon_flat = self.decoder_net(z)

      if self.datadim == 2:
        side = int(np.sqrt(self.input_dim))
        recon = recon_flat.view(-1, side, side)
        return recon

      return recon_flat

    def forward(self, x, variance=False):
      mu, logvar = self.encode(x, variance=True)
      z = self.reparameterize(mu, logvar)
      recon = self.decode(z)

      if variance:
          return recon, mu, logvar
      else:
          return recon

    def loss_function(self, recon, x, mu, logvar):
      if self.datadim == 2:
        batch = x.size(0)
        x_flat = x.view(batch, -1)
      else:
        x_flat = x

      recon_flat = recon.view_as(x_flat)
      recon_loss = nn.MSELoss()(recon_flat, x_flat)

      sigma2 = torch.exp(logvar)
      kld_element = mu.pow(2) + sigma2 - 1 - logvar
      kld = 0.5 * torch.sum(kld_element)    
      kld = (kld / x.size(0)) / recon_flat.shape[1]

      return (recon_loss, kld * self.reg)

    def save_model(self, filename):
      addr = os.path.join(BASEDIR, filename)
      os.makedirs(os.path.dirname(addr), exist_ok=True)

      with open(addr, "wb") as handle:
        pickle.dump({"model": self}, handle, protocol=pickle.HIGHEST_PROTOCOL)

class GIAutoencoder(nn.Module):
    def __init__(self, encodeSeq, decodeSeq, domaingrid, activation, decoderActivation=None, enforcetime=False, rangedim=1, ffeatures=1, datadim=1):
        super().__init__()

        if decoderActivation is None:
          decoderActivation = activation

        # self.enforcetime = enforcetime
        
        grid_np = np.asarray(domaingrid, dtype=np.float32)
        self.register_buffer('domaingrid', torch.from_numpy(grid_np))

        encodeSeq = list(encodeSeq)
        decodeSeq = list(decodeSeq)
        if ffeatures > 0:
          decodeSeq[0] = encodeSeq[-1] + (2 * ffeatures + 1) * self.domaingrid.shape[1]
        else:
          decodeSeq[0] = encodeSeq[-1] + 1

        decodeSeq[-1] = rangedim  

        self.rangedim = rangedim

        self.encoder = FFNet(activation=activation, seq=encodeSeq)
        self.decoder = FFNet(activation=decoderActivation, seq=decodeSeq)
        self.s = activation

        self.reduced = encodeSeq[-1]# + 1 if self.enforcetime else encodeSeq[-1]
        self.datashape = None
        self.ffeatures = ffeatures

    def forward(self, x, grid=None):
        return self.decode(self.encode(x), grid=grid)

    def encode(self, enc):
        # if self.datadim >= 2:
        #   self.datashape = list(enc.shape[-1*self.datadim:])
        #   enc = enc.reshape(list(enc.shape[:-1*self.datadim]) + [-1])

        # if self.enforcetime:
        #   times = enc[..., -1:]
        #   #enc = enc[..., :-1]

        output = self.encoder(enc)
        
        # if self.enforcetime:
        #   output = torch.cat([output, times], dim=-1)

        return output

    def decode(self, dec, grid=None):
        origshape = None
        if len(dec.shape) > 2:
          origshape = dec.shape
          dec = dec.reshape([-1] + [dec.shape[-1]])

        B, _ = dec.shape
        N, _ = self.domaingrid.shape

        decorig = dec

        if grid is None:
          grid = self.domaingrid

        grid = grid.to(dec.device)

        if self.ffeatures > 0:
          sins = [torch.sin(2*torch.pi*(n+1)*grid) for n in range(self.ffeatures)]
          coss = [torch.cos(2*torch.pi*(n+1)*grid) for n in range(self.ffeatures)]
          grid = torch.cat(tuple([grid] + sins + coss), dim=-1)

        # flatten code batch dims
        orig_shape = dec.shape
        if dec.dim() > 2:
            code_flat = dec.reshape(-1, orig_shape[-1])
        else:
            code_flat = dec

        B = code_flat.shape[0]
        N = grid.shape[0]

        dec_exp = dec.repeat_interleave(N, dim=0) 
        grid_exp = grid.repeat(B, 1)                    

        decoder_input = torch.cat([dec_exp, grid_exp], dim=-1)
        decoded = self.decoder(decoder_input)

        decoded = decoded.view(B, N, self.rangedim)

        if self.rangedim == 1:
          decoded = decoded.reshape(list(decorig.shape[:-1]) + [grid.shape[0]])
        
        if origshape is not None:
          decoded = decoded.reshape(list(origshape[:-1]) + [-1])

        return decoded

def get_actvn(act:str):
    if act == "relu":
        return nn.ReLU()

class TCAutoencoder(nn.Module):
    def __init__(self, encodeSeq, decodeSeq, activation, domaingrid, datadim=1, decoderActivation=None, numbasis=None):
        super().__init__()

        if decoderActivation is None:
          decoderActivation = activation
        
        grid_np = np.asarray(domaingrid, dtype=np.float32)
        self.register_buffer('domaingrid', torch.from_numpy(grid_np))

        encodeSeq = list(encodeSeq)
        decodeSeq = list(decodeSeq)

        assert(datadim==1)

        if numbasis is not None:
          decodeSeq[-1] = numbasis

        decodeSeq[-1] = 2 * (decodeSeq[-1] // 2) + 1

        self.encoder = FFNet(activation=activation, seq=encodeSeq)
        self.decoder = FFNet(activation=decoderActivation, seq=decodeSeq)

        self.reduced = encodeSeq[-1]# + 1 if self.enforcetime else encodeSeq[-1]
        self.datashape = None
        self.numbasis = decodeSeq[-1]

    def forward(self, x, grid=None):
        return self.decode(self.encode(x), grid=grid)

    def encode(self, enc):
        # if self.datadim >= 2:
        #   self.datashape = list(enc.shape[-1*self.datadim:])
        #   enc = enc.reshape(list(enc.shape[:-1*self.datadim]) + [-1])

        # if self.enforcetime:
        #   times = enc[..., -1:]
        #   #enc = enc[..., :-1]

        output = self.encoder(enc)
        
        # if self.enforcetime:
        #   output = torch.cat([output, times], dim=-1)

        return output

    def decode(self, dec, grid=None):
        origshape = None
        if len(dec.shape) > 2:
          origshape = dec.shape
          dec = dec.reshape([-1] + [dec.shape[-1]])

        B = dec.shape
        N = self.domaingrid.shape

        decorig = dec

        if grid is None:
          grid = self.domaingrid

        grid = grid.to(dec.device)
        grid = grid.reshape((-1))

        sins = torch.stack([torch.sin(2*torch.pi*(n+1)*grid) for n in range(self.numbasis // 2)], dim=0)
        coss = torch.stack([torch.cos(2*torch.pi*(n+1)*grid) for n in range(self.numbasis // 2)], dim=0)
        constant = torch.ones_like(grid)

        # flatten code batch dims
        orig_shape = dec.shape
        if dec.dim() > 2:
            code_flat = dec.reshape(-1, orig_shape[-1])
        else:
            code_flat = dec

        B = code_flat.shape[0]
        N = grid.shape[0]

        decodedcoeff = self.decoder(code_flat)

        decoded = decodedcoeff[:, :(self.numbasis // 2)] @ sins + decodedcoeff[:, (self.numbasis // 2):-1] @ coss + decodedcoeff[:, -1:] * constant
        
        if origshape is not None:
          decoded = decoded.reshape(list(origshape[:-1]) + [-1])

        return decoded


class TCAutoencoderConv(nn.Module):
    def __init__(self, config:DictConfig):
        super().__init__()
        # --- encoder (downblocks) config ---
        in_dims = config.sample.spatial_resolution
        latents_dims = config.sample.latents_dims
        out_channels = list(config.downblocks.channels)
        kernel_stride_paddings = [list(ksp) for ksp in config.downblocks.kernel_stride_paddings]
        actvn = get_actvn(config.downblocks.actvn)
        padding_mode = config.downblocks.get("padding_mode", "zeros")
        in_channel = config.downblocks.get("in_channel", 1)
        mlp = config.get("mlp", True)
        self.dim = 1 if not (isinstance(in_dims, list) or isinstance(in_dims, ListConfig)) else len(in_dims)

        # --- decoder (TC) config ---
        decodeSeq = list(config.decodeSeq)            # e.g., [latent, ..., numbasis]
        dec_actvn = get_actvn(config.get("decoderActivation", config.activation))
        numbasis = config.get("numbasis", None)
        # domain grid: optional; use config.domaingrid if provided, else derive simple [0,1) with N points
        if "domaingrid" in config and config.domaingrid is not None:
            grid_np = np.asarray(config.domaingrid, dtype=np.float32)
        else:
            # fallback: N = last decode dim (if provided) or 151 as a safe default like your example
            N = (numbasis if numbasis is not None else decodeSeq[-1]) if len(decodeSeq) > 0 else 151
            grid_np = np.linspace(0.0, 1.0, int(N), dtype=np.float32)
        self.register_buffer("domaingrid", torch.from_numpy(grid_np))

        # --- enforce odd numbasis: sin/cos pairs + constant ---
        if numbasis is not None:
            decodeSeq[-1] = int(numbasis)
        decodeSeq[-1] = 2 * (decodeSeq[-1] // 2) + 1

        # --- build encoder ---
        self.encoder = ConvEncoder(
            in_dims=in_dims,
            latents_dims=latents_dims,
            out_channels=out_channels,
            kernel_stride_paddings=kernel_stride_paddings,
            actvn=actvn,
            in_channel=in_channel,
            padding_mode=padding_mode,
            mlp=mlp,
        )

        # --- build decoder (latent -> Fourier coeffs) ---
        self.decoder = FFNet(activation=dec_actvn, seq=decodeSeq)

        # --- bookkeeping ---
        self.reduced = latents_dims
        self.datashape = None
        self.numbasis = decodeSeq[-1]

    def forward(self, x, grid=None):
        return self.decode(self.encode(x), grid=grid)

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

    def decode(self, dec, grid=None):
        origshape = None
        if dec.dim() > 2:
            origshape = dec.shape
            dec = dec.reshape([-1, dec.shape[-1]])

        if grid is None:
            grid = self.domaingrid
        grid = grid.to(dec.device).reshape((-1))

        half = self.numbasis // 2
        if half > 0:
            sins = torch.stack([torch.sin(2*torch.pi*(n+1)*grid) for n in range(half)], dim=0)
            coss = torch.stack([torch.cos(2*torch.pi*(n+1)*grid) for n in range(half)], dim=0)
        else:
            sins, coss = None, None

        constant = torch.ones_like(grid)

        decodedcoeff = self.decoder(dec)

        if half > 0:
            sin_coeff = decodedcoeff[:, :half]
            cos_coeff = decodedcoeff[:, half:-1]
            const_coeff = decodedcoeff[:, -1:]
            decoded = sin_coeff @ sins + cos_coeff @ coss + const_coeff * constant
        else:
            const_coeff = decodedcoeff[:, -1:]
            decoded = const_coeff * constant

        if origshape is not None:
            decoded = decoded.reshape(list(origshape[:-1]) + [-1])

        return decoded

Other_Modules = JC_Modules + [TCAutoencoderConv]

class WindowTrajectory():
  def find_window(self, t, left=True):
    foundw = []
    for (w, wvs) in enumerate(self.windowvals):
      if t in wvs:
        foundw.append(w)

    if len(foundw) > 0:
      return min(foundw) if left else max(foundw)
    else:
      return False

  def decode_window(self, w, tensor):
    return self.aes[w].decode(tensor)

  def project_window(self, w, tensor):
    return self.decode_window(w, self.encode_window(w, tensor))

  def encode_window(self, w, tensor):
    return self.encode_model(self.aes[w], tensor)

class WeldNet(WindowTrajectory):
  def determine_windows(self, alg="uniform"):
    if alg == "uniform":
      total = self.T + self.W - 1
      M = total // self.W
      remainder = total - self.W * M
      
      left = [list(range(k*(M-1), (k+1)*(M-1) + 1)) for k in range(self.W - remainder)]
      start = left[-1][-1]
      right = [list(range(start + k*(M), start + (k+1)*(M) + 1)) for k in range(remainder)]
      return left + right
    
    elif alg == "change-l2":
      wvals = np.asarray([rpt.Dynp(model="l2").fit(x).predict(n_bkps = self.W-1)[:-1] for x in self.alltrain])
      bkpts = [0] + np.median(wvals, axis=0).astype(int).tolist() + [self.T-1]

      windowvals = [list(range(bkpts[i], bkpts[i+1]+1)) for i in range(len(bkpts)-1)]
      return windowvals


  def __init__(self, dataset, windows, aeclass, aeparams, propclass, propparams, transclass, transparams, dynamicwindow=False, passparams=False, straightness=0, td=None, seed=0, device=0, kinetic=0, decodedprop=False, accumulateprop=False, tiprop=False, autonomous=True):    
    self.dataset = dataset
    self.device = device
    self.td = td
    self.straightness = straightness
  
    if self.td is None:
      self.prefix = f'{self.dataset.name}{str(aeclass.__name__)}-{propparams["seq"][-1]}-{"auton" if autonomous else "nonauton"}'
    else:
      self.prefix = self.td

    self.kinetic = kinetic
    self.autonomous = autonomous
    self.residualprop = True
    self.tiprop = tiprop

    assert(not self.tiprop)

    self.accumulateprop = accumulateprop
    assert(autonomous)

    self.decodedprop = decodedprop

    self.passparams = passparams

    assert((not self.passparams) or (not self.tiprop))

    if not self.autonomous:
      propparams["seq"] = list(propparams["seq"])
      propparams["seq"][0] = propparams["seq"][-1] + 1

    if self.passparams:
      f = self.dataset.params.shape[1]
      propparams["seq"] = list(propparams["seq"])
      propparams["seq"][0] = propparams["seq"][0] + f

      transparams["seq"] = list(transparams["seq"])
      transparams["seq"][0] = transparams["seq"][0] + f

      aeparams["decodeSeq"] = list(aeparams["decodeSeq"])
      aeparams["decodeSeq"][0] = aeparams["decodeSeq"][0] + f

    assert(self.straightness == 0 or self.kinetic == 0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    self.seed = seed

    datacopy = self.dataset.data.copy()
    self.numtrain = int(datacopy.shape[0] * 0.9)
    
    self.T = self.dataset.data.shape[1]
    self.W = windows
    
    self.aes = []
    self.props = []
    self.trains = []
    self.tests = []

    self.timetaken = 0

    self.alltrain = datacopy[:self.numtrain]
    self.alltest = datacopy[self.numtrain:]

    self.paramtrain = self.dataset.params[:self.numtrain]
    self.paramtest = self.dataset.params[self.numtrain:]

    if self.passparams:
      T = self.alltrain.shape[1] 
      
      paramtrain = np.repeat(self.paramtrain[:, None, :], T, axis=1)
      paramtest  = np.repeat(self.paramtest[:, None, :], T, axis=1) 

      self.alltrain = np.concatenate([self.alltrain, paramtrain], axis=-1)
      self.alltest = np.concatenate([self.alltest, paramtest], axis=-1)

    aeparams["datadim"] = len(self.dataset.data.shape) - 2
    self.aedata = [aeclass.__name__, aeparams, windows, self.straightness, self.kinetic]
    self.propdata = [propclass.__name__, propparams, windows, self.autonomous, self.residualprop, self.accumulateprop, self.decodedprop]

    self.aeepochs = []
    self.propepochs = []
    self.transepochs = []

    if transclass is not None and windows > 1:
      self.transcoderdata = [transclass.__name__, transparams, windows, self.residualprop]
    else:
      self.transcoderdata = None

    if dynamicwindow:
      self.windowvals = self.determine_windows("change-l2")
    else:
      self.windowvals = self.determine_windows("uniform")

    print("Windows:", [[x[0], x[-1]] for x in self.windowvals])
  
    self.transcoders = []

    self.aeclass = aeclass
    self.aeparams = aeparams
    self.propclass = propclass
    self.propparams = propparams
    self.transclass = transclass
    self.transparams = transparams

    self.metadata = {
      "aeinfo": self.aedata,
      "propinfo": self.propdata,
      "transinfo": self.transcoderdata,
      "trainedtogether": False,
      "dataset_name": dataset.name,
      "data_shape": list(dataset.data.shape),
      "data_checksum": float(np.sum(dataset.data)),
      "seed": seed,
      "passparams": False
    }

    self.epochs = []

    if self.passparams:
      self.metadata["passparams"] = True
  
  def transcode(self, t, codes):
    w = self.find_window(t)
    assert(self.windowvals[w][-1] == t)

    if len(self.transcoders) == 0:
      decoded = self.decode_window(w, codes)
      codes = self.encode_window(w+1, decoded)
      print(f"Default transcoding {w} to {w+1} at time {t}")
    else:
      if isinstance(self.transcoders[w], nn.Module):
        outs = self.transcoders[w](codes)

      else:
        if torch.is_tensor(codes):
          codes_np = codes.detach().cpu().numpy()
          outs = torch.tensor(self.transcoders[w].predict(codes_np), dtype=torch.float32, device=codes.device)

        else:
          outs = self.transcoders[w].predict(codes)

      if self.residualprop:
        codes = codes + outs
      else:
        codes = outs

      print(f"Explicit transcoding {w} to {w+1} at time {t}")
    
    return torch.tensor(codes, dtype=torch.float32)

  # encodes, propagates, does NOT decode
  def propagate(self, arr, t, steps, arrencoded=False, fixedw=False):
    assert(t + steps < self.T)

    if fixedw:
      w = fixedw
    else:
      w = self.find_window(t)

    inputt = torch.tensor(arr).to(self.device, dtype=torch.float32)

    if arrencoded:
      codes = inputt
    else:
      codes = self.encode_window(w, inputt)

    codeslist = []
    wprev = w

    if self.tiprop:
      times = []
      startt = t
      for step in range(steps):
        tcurr = t + 1 + step

        if fixedw:
          wcurr = w
        else:
          wcurr = self.find_window(tcurr)

        if wcurr != wprev and tcurr-1-startt > 0:
          times.append((tcurr-1-startt, wcurr))

        wprev = wcurr

      if len(times) == 0 or (len(times) > 0 and times[-1] != (tcurr-startt, wcurr)):
        times.append((tcurr-startt, wcurr))

      for i, (amount, ww) in enumerate(times):
        if ww > 1 and i < len(times) - 1:
          codes = self.transcode(self.windowvals[ww-1][-1], codes)

        out = self.prop_forward(self.props[ww], codes, ts=torch.arange(1, amount+1)/self.T)

        out = list(torch.unbind(out, dim=1))
        codeslist = codeslist + out
        codes = out[-1]

    else:
      for step in range(steps):
        tcurr = t + 1 + step

        if fixedw:
          wcurr = w
        else:
          wcurr = self.find_window(tcurr)

        if wcurr != wprev:
          codes = self.transcode(tcurr-1, codes)
          inputt = codes

        if self.autonomous:
          codeinput = codes
        else:
          ttensor = torch.tensor(np.repeat((tcurr*0 - 1), codes.shape[0])).unsqueeze(1).to(self.device).float()
          codeinput = torch.cat((codes, ttensor), dim=1)

        codes = self.prop_forward(self.props[wcurr], codeinput)

        wprev = wcurr
        codeslist.append(codes)

    return codeslist

  def get_proj_errors(self, model, testarr, ords=(2,)):
    if isinstance(testarr, np.ndarray):
      testarr = torch.tensor(testarr, dtype=torch.float32)

    testarr = testarr.to(next(model.parameters()).device)
      
    proj = model.decode(self.encode_model(model, testarr))
    
    if len(testarr.shape) > 3:
      assert(len(testarr.shape) == 4)
      testarr = testarr.reshape(list(testarr.shape[:-2]) + [-1])
      proj = proj.reshape(list(proj.shape[:-2]) + [-1])

    if self.passparams:
      testarr = testarr[..., :-self.dataset.params.shape[1]]
   
    n = testarr.shape[0]
    testarr = testarr.cpu().detach().numpy().reshape([n, -1])
    proj = proj.cpu().detach().numpy().reshape([n, -1])

    testerrs = []

    for o in ords:
      testerro = np.mean(np.linalg.norm(testarr - proj, axis=1, ord=o) / np.linalg.norm(testarr, axis=1, ord=o))
      testerrs.append(testerro)

    return tuple(testerrs)

  def encode_model(self, model, batchh):
    if self.passparams:
      batch = batchh[..., :-self.dataset.params.shape[1]]
      params = batchh[..., -self.dataset.params.shape[1]:]
    else:
      batch = batchh

    out = model.encode(batch)

    if self.passparams:
      out = torch.cat([out, params], dim=-1)

    return out

  def prop_forward(self, prop, batch, ts=None):
    batchbase = batch
    if self.passparams:
      params = batch[..., -self.dataset.params.shape[1]:]
      batchbase = batch[..., :-self.dataset.params.shape[1]]

    if ts is not None and self.tiprop:
      z_shape = batchbase.shape
      *leading_dims, N = z_shape
      T = ts.shape[0]

      z_expanded = batchbase.unsqueeze(-2).expand(*leading_dims, T, N)

      t_shape = [1] * len(leading_dims) + [T, 1]
      t_expanded = ts.view(*t_shape).expand(*leading_dims, T, 1)

      batch = torch.cat([z_expanded, t_expanded], dim=-1)
      batchbase = z_expanded
    
    out = prop.forward(batch)
    
    if self.residualprop:
      out = out + batchbase

    if self.passparams:
      out = torch.cat([out, params], dim=-1)

    return out

  def trans_forward(self, trans, batch):
    if self.passparams:
      params = batch[..., -self.dataset.params.shape[1]:]

    out = trans.forward(batch)

    if self.passparams:
      out = torch.cat([out, params], dim=-1)

    return out

  def train_aes(self, epochs_first, warmstart_epochs=0, save=True, onlydecoder=False, roll=False, optim=torch.optim.AdamW, lr=1e-4, plottb=False, gridbatch=None, printinterval=10, batch=32, ridge=0, loss=None, encoding_param=-1, best=True, verbose=False):    
      def ae_epoch(model, dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
        losses = []
        testerrors1 = []
        testerrors2 = []
        testerrorsinf = []

        device = self.device

        def closure(batch):
          optimizer.zero_grad()

          assert(self.straightness + self.kinetic == 0)

          # for N in range(batch.shape[0]):
          #   traj = batch[N, :, :]
          #   enc = model.encode(traj)
          #   proj = model.decode(enc)

          #   # compute regularization here
          #   # add penalization to enc here
          #   # (enc[:-1] - enc[1:]) ** 2 is proportional to velocity

          #   res = loss(traj, proj)

          #   if self.straightness > 0:
          #     T = traj.shape[0]
          #     i_values = torch.arange(1, T)
          #     weights = (T - i_values) / T
          #     term1 = torch.outer(weights, enc[0, :])
          #     term2 = torch.outer((i_values / T), enc[-1, :])
          #     term3 = enc[i_values, :]
          #     penalty = loss(term1 + term2, term3)
          #     penalties += self.straightness * penalty
          #   elif self.kinetic > 0:
          #     #acceleration
          #     #starts = enc[:-2]
          #     #mids = enc[1:-1]
          #     #ends = enc[2:]

          #     # maybe scale?

          #     # # order 2
          #     # starts = enc[:-2]
          #     # ends = enc[2:]
          #     # penalty = loss(starts - ends, torch.zeros_like(starts))
          #     # penalties += self.kinetic * penalty

          #     # order one
          #     starts = enc[:-1, :]
          #     ends = enc[1:, :]
          #     penalty = loss(starts - ends, torch.zeros_like(starts))
          #     penalties += self.kinetic * penalty

          #   total += res

          if roll:
            rot = np.random.randint(0, batch.shape[-1])
            batch = torch.roll(batch, shifts=rot, dims=-1)

          if isinstance(model, FFVAE):
            recon, mu, logvar = model(batch, variance=True)
            self.reconerr, self.kld = model.loss_function(recon, batch, mu, logvar)

            res = self.reconerr + self.kld
            res.backward()
            
          else:
            enc = self.encode_model(model, batch)

            if isinstance(model, GIAutoencoder) or isinstance(model, TCAutoencoder) or isinstance(model, TCAutoencoderConv):
              full_grid = model.domaingrid
              N = full_grid.shape[0]
              if gridbatch is None or gridbatch >= N:
                grid_subset = full_grid
                idx = None
              else:
                perm = torch.randperm(N, device=full_grid.device)
                idx = perm[:gridbatch]
                grid_subset = full_grid[idx]
                batch = batch[:, :, idx]
              
              proj = model.decode(enc, grid=grid_subset)

            else:
              proj = model.decode(enc)

            if self.passparams:
              batch = batch[..., :-self.dataset.params.shape[1]]

            res = loss(batch, proj)
            res.backward()
          
          if writer is not None and self.aestep % 5:
            writer.add_scalar("main/loss", float(res.cpu().detach()), global_step=self.aestep)
            #writer.add_scalar("main/penalty", penalties, global_step=self.aestep)

          return res

        for batch in dataloader:
          self.aestep += 1
          error = optimizer.step(lambda: closure(batch))
          losses.append(float(error.cpu().detach()))

        if scheduler is not None and ep > epochs_first // 2:
          scheduler.step(np.mean(losses))

        # print test
        if printinterval > 0 and (ep % printinterval == 0):
          testerr1, testerr2, testerrinf = self.get_proj_errors(model, testarr, ords=(1, 2, np.inf))

          if isinstance(model, FFVAE):
            prefix = f"{ep+1}: Train Loss {self.reconerr:.3e} + {self.kld:.3e}"
          else:    
            prefix = f"{ep+1}: Train Loss {error:.3e}"

          if scheduler is not None:
            print(f"{prefix}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
          else:
            print(f"{prefix}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

          
          if writer is not None:
              writer.add_scalar("misc/relativeL1proj", testerr1, global_step=ep)
              writer.add_scalar("main/relativeL2proj", testerr2, global_step=ep)
              writer.add_scalar("misc/relativeLInfproj", testerrinf, global_step=ep)

        return losses, testerrors1, testerrors2, testerrorsinf

      loss = nn.MSELoss() if loss is None else loss()
      encoding_param = determine_param(self.dataset, encoding_param)

      losses_all, testerrors1_all, testerrors2_all, testerrorsinf_all = [], [], [], []

      start = time.time()
      print(f"Training {self.W} WeldNet AEs")
      self.trains = []
      self.tests = []
      for w in range(self.W):
        if len(self.aes) <= w:
          self.aes.append(self.aeclass(**self.aeparams) if self.aeclass not in Other_Modules else self.aeclass(self.aeparams.copy()))

        ae = self.aes[w]

        losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
        bestdict = { "loss": float(np.inf), "ep": 0 }

        self.aestep = 0
        epochs = epochs_first
        train = torch.tensor(self.alltrain[:, self.windowvals[w]], dtype=torch.float32)
        test = self.alltest[:, self.windowvals[w]]

        if isinstance(ae, PCAAutoencoder):
          ae.train_pca(train.cpu().numpy())
          testerr1, testerr2, testerrinf = self.get_proj_errors(ae, test, ords=(1, 2, np.inf))
          print(f"Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

          continue

        self.trains.append(train)
        self.tests.append(test)

        if onlydecoder:
          trainparams = ae.decoder.parameters()
        else:
          trainparams = ae.parameters()

        opt = optim(trainparams, lr=lr, weight_decay=ridge)
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.3)
        dataloader = DataLoader(train, shuffle=False, batch_size=batch)

        writer = None
        if self.td is not None:
          name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}-weld{w}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
          writer = torch.utils.tensorboard.SummaryWriter(name)
          print("Tensorboard writer location is " + name)

        print("Number of NN trainable parameters", utils.num_params(ae))
        print(f"Starting training WeldNet AE {w+1}/{self.W} ({self.windowvals[w][0]}->{self.windowvals[w][-1]}) at {time.asctime()}...")

        #print(train, train.shape)
        print("train", train.shape, "test", test.shape)

        # warm start
        if warmstart_epochs > 0 and w > 0:
          epochs = warmstart_epochs
          state = self.aes[w-1].state_dict()
          ae.load_state_dict(state)
          print(f"Warm started model {w} from model {w-1}")

        self.aestep = 0
        for ep in range(epochs):
            lossesN, testerrors1N, testerrors2N, testerrorsinfN = ae_epoch(ae, dataloader, scheduler=scheduler, optimizer=opt, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
            losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

            if best and ep > epochs // 2:
              avgloss = np.mean(lossesN)
              if avgloss < bestdict["loss"]:
                bestdict["model"] = ae.state_dict()
                bestdict["opt"] = opt.state_dict()
                bestdict["loss"] = avgloss
                bestdict["ep"] = ep
              elif verbose:
                print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")

            if ep % 5 == 0 and plottb:
              WeldHelper.plot_encoding_window(self, w, encoding_param, step=self.aestep, writer=writer, tensorboard=True)
        
        print(f"Finish training AE {w} at {time.asctime()}.")
        losses_all.append(losses)
        testerrors1_all.append(testerrors1)
        testerrors2_all.append(testerrors2)
        testerrorsinf_all.append(testerrorsinf)

        if best:
          ae.load_state_dict(bestdict["model"])
          opt.load_state_dict(bestdict["opt"])
            
      self.aeepochs.append(epochs_first)
      if epochs != epochs_first:
        self.aeepochs.append(epochs)

      if save and False:
        dire = "savedmodels/weld"
        addr = f"{dire}/{self.prefix}{self.W}w-{datetime.datetime.now().strftime('%d-%B-%Y-%H.%M')}.pickle"

        if not os.path.exists(dire):
          os.makedirs(dire)

        with open(addr, "wb") as handle:
          pickle.dump({"aes": self.aes, "aedata": self.aedata, "datadata": self.datadata}, handle, protocol=pickle.HIGHEST_PROTOCOL)
          print("AEs saved at", addr)

      end = time.time()
      self.timetaken += end - start
      print("Finished training all timewindows")
      return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

  def train_aes_plus_props(self, epochs, lamb=0.1, save=True, roll=False, optim=torch.optim.AdamW, lr=1e-4, plottb=False, gridbatch=None, printinterval=10, batch=32, ridge=0, loss=None, encoding_param=-1, best=True, verbose=False):    
    def both_epoch(model, modelprop, dataloader, writer=None, w=0, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []

      def closure(batch):
        optimizer.zero_grad()

        assert(self.straightness + self.kinetic == 0)

        if roll:
          rot = np.random.randint(0, batch.shape[-1])
          batch = torch.roll(batch, shifts=rot, dims=-1)

        enc = self.encode_model(model, batch)

        if isinstance(model, GIAutoencoder) or isinstance(model, TCAutoencoder) or isinstance(model, TCAutoencoderConv):
          full_grid = model.domaingrid
          N = full_grid.shape[0]
          if gridbatch is None or gridbatch >= N:
            grid_subset = full_grid
            idx = None
          else:
            perm = torch.randperm(N, device=full_grid.device)
            idx = perm[:gridbatch]
            grid_subset = full_grid[idx]
            batch = batch[:, :, idx]
          
          proj = model.decode(enc, grid=grid_subset)
          
        else:
          proj = model.decode(enc)

        if self.passparams:
          batch = batch[..., :-self.dataset.params.shape[1]]

        res = loss(batch, proj)

        starts = enc[:, :-1]
        exacts = enc[:, 1:]

        # predicted = modelprop(starts)

        # if self.residualprop:
        #   if self.passparams:
        #     zeroarr = torch.zeros(list(predicted.shape[:-1]) + [self.dataset.params.shape[1]])
        #     predicted = torch.concat([predicted, zeroarr], axis=-1)
            
        #   predicted = starts + predicted

        
        if self.tiprop:
          predicted = self.propagate(starts[:, 0], self.windowvals[w][0], exacts.shape[1], arrencoded=True)
          predicted = torch.stack(predicted, dim=1) 

        elif self.accumulateprop and False:
          x0 = starts[:, :1]

          xlist = []
          for _ in range(starts.shape[1]):
            x0 = self.prop_forward(modelprop, x0)
            xlist.append(x0)

          predicted = torch.cat(xlist, dim=1)

        else:
          predicted = self.prop_forward(modelprop, starts)

        error = loss(predicted, exacts)

        totalloss = res + lamb * error

        totalloss.backward()
        
        num = totalloss.cpu().detach()
        if writer is not None and self.aestep % 5:
          
          writer.add_scalar("main/loss", float(num), global_step=self.aestep)
          #writer.add_scalar("main/penalty", penalties, global_step=self.aestep)

        return totalloss

      for batch in dataloader:
        batch = batch.to(next(model.parameters()).device)
        self.aestep += 1
        lossout = optimizer.step(lambda: closure(batch))
        losses.append(float(lossout.cpu().detach()))

      if scheduler is not None and ep > epochs // 2:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testerr1, testerr2, testerrinf = self.get_proj_errors(model, testarr, ords=(1, 2, np.inf))

        if scheduler is not None:
          print(f"{w+1}/{ep+1}: Train Loss {lossout:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{w+1}/{ep+1}: Train Loss {lossout:.3e}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1proj", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2proj", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInfproj", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf

    def train_window(w, parallel=False):
      ae = self.aes[w]
      prop = self.props[w]

      if parallel:
        ae.to(f"cuda:{w}")
        prop.to(f"cuda:{w}")

      losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
      bestdict = { "loss": float(np.inf), "ep": 0 }

      self.aestep = 0
      train = torch.tensor(self.alltrain[:, self.windowvals[w], :], dtype=torch.float32)
      test = self.alltest[:, self.windowvals[w], :] 

      if isinstance(ae, PCAAutoencoder):
        ae.train_pca(train.cpu().numpy())
        testerr1, testerr2, testerrinf = self.get_proj_errors(ae, test, ords=(1, 2, np.inf))
        print(f"Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        return

      self.trains.append(train)
      self.tests.append(test)

      opt = optim( list(ae.parameters()) + list(prop.parameters()), lr=lr, weight_decay=ridge)
      scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.3)
      dataloader = DataLoader(train, shuffle=False, batch_size=batch)

      writer = None
      if self.td is not None:
        name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}-weld{w}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
        writer = torch.utils.tensorboard.SummaryWriter(name)
        print("Tensorboard writer location is " + name)

      print("Number of NN trainable parameters", utils.num_params(ae))
      print(f"Starting training WeldNet AE + Prop {w+1}/{self.W} ({self.windowvals[w][0]}->{self.windowvals[w][-1]}) at {time.asctime()}...")

      #print(train, train.shape)
      print("train", train.shape, "test", test.shape)

      self.aestep = 0
      for ep in range(epochs):
          lossesN, testerrors1N, testerrors2N, testerrorsinfN = both_epoch(ae, prop, dataloader, w=w, scheduler=scheduler, optimizer=opt, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
          losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

          if best and ep > epochs // 2:
            avgloss = np.mean(lossesN)
            if avgloss < bestdict["loss"]:
              bestdict["model"] = ae.state_dict()
              bestdict["opt"] = opt.state_dict()
              bestdict["loss"] = avgloss
              bestdict["ep"] = ep
            elif verbose:
              print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")

          if ep % 5 == 0 and plottb:
            WeldHelper.plot_encoding_window(self, w, encoding_param, step=self.aestep, writer=writer, tensorboard=True)
      
      print(f"Finish training AE and Prop {w} at {time.asctime()}.")
      losses_all.append(losses)
      testerrors1_all.append(testerrors1)
      testerrors2_all.append(testerrors2)
      testerrorsinf_all.append(testerrorsinf)

      if best:
        ae.load_state_dict(bestdict["model"])
        opt.load_state_dict(bestdict["opt"])

    loss = nn.MSELoss() if loss is None else loss()
    encoding_param = determine_param(self.dataset, encoding_param)

    losses_all, testerrors1_all, testerrors2_all, testerrorsinf_all = [], [], [], []

    start = time.time()
    self.metadata["trainedtogether"] = True
    print(f"Training {self.W} WeldNet AEs and props together")
    self.trains = []
    self.tests = []

    # prepare models
    for w in range(self.W):
      if len(self.aes) <= w:
        self.aes.append(self.aeclass(**self.aeparams).to(self.device) if self.aeclass not in Other_Modules else self.aeclass(self.aeparams.copy()))
      if len(self.props) <= w:
        self.props.append(self.propclass(**self.propparams).to(self.device) if self.propclass not in Other_Modules else self.propclass(self.propclass.copy()))

    # train models
    if self.W > 1 and torch.cuda.device_count() >= self.W:
        print("Spawning threads for each window")
        with ThreadPoolExecutor(max_workers=self.W) as ex:
          futures = [ex.submit(train_window, rank, True) for rank in range(self.W)]
          for f in futures:
              f.result()
    else:
      for ww in range(self.W):
        train_window(ww)

    self.aeepochs.append(epochs)

    end = time.time()
    self.timetaken += end - start
    print("Finished training all timewindows")
    return {}#{ "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf } 

  def train_transcoders(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, verbose=False, propagated_trans=True, printinterval=10, batch=32, ridge=0, loss=None, encoding_param=-1, best=True):
    def transcoder_epoch(model, dataloader, writer=None, scheduler=None, optimizer=None, ep=0, printinterval=10, loss=None, testarr=None):
      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []

      def closure(batch):
        optimizer.zero_grad()
        total = 0
        penalties = 0

        x = batch[:, :, 0]
        y = batch[:, :, 1]

        predict = self.trans_forward(model, x)

        if self.residualprop:
          predict = x + predict

        res = loss(predict, y)
        total += res

        total.backward()
        
        if writer is not None and self.aestep % 5:
          writer.add_scalar("main/loss", total, global_step=self.aestep)
          writer.add_scalar("main/penalty", penalties, global_step=self.aestep)

        return total

      for batch in dataloader:
        self.transstep += 1
        error = optimizer.step(lambda: closure(batch))
        losses.append(float(error.cpu().detach()))

      if scheduler is not None and ep > epochs // 2:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testdom = torch.tensor(testarr[:, :, 0], dtype=torch.float32)
        testran = testarr[:, :, 1]
        predict = model(testdom)
        
        if self.residualprop:
          predict = testdom + predict
          
        predict = predict.cpu().detach().numpy()

        testerr1 = np.mean(np.linalg.norm(testran - predict, axis=1, ord=1) / np.linalg.norm(testran, axis=1, ord=1))
        testerr2 = np.mean(np.linalg.norm(testran - predict, axis=1, ord=2) / np.linalg.norm(testran, axis=1, ord=2))
        testerrinf = np.mean(np.linalg.norm(testran - predict, axis=1, ord=np.inf) / np.linalg.norm(testran, axis=1, ord=np.inf))

        testerrors1.append(testerr1)
        testerrors2.append(testerr2)
        testerrorsinf.append(testerrinf)
        
        if scheduler is not None:
          print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative Transcoding Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{ep+1}: Train Loss {error:.3e}, Relative Transcoding Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1proj", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2proj", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInfproj", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf

    assert(self.transclass is not None)

    loss = nn.MSELoss() if loss is None else loss()
    encoding_param = determine_param(self.dataset, encoding_param)
    
    losses_all, testerrors1_all, testerrors2_all, testerrorsinf_all = [], [], [], []

    start = time.time()
    print(f"Training {self.W-1} WeldNet Transcoders")
    for w in range(self.W - 1):
      if len(self.transcoders) <= w:
        self.transcoders.append(self.transclass(**self.transparams))

      losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
      bestdict = { "loss": float(np.inf), "ep": 0 }

      self.transstep = 0
      data = torch.tensor(self.alltrain[:, self.windowvals[w], :], dtype=torch.float, device=self.device)
      datatest = torch.tensor(self.alltest[:, self.windowvals[w], :], dtype=torch.float, device=self.device)
      
      #encodeddom = self.encode_window(w, data[:, 0]).detach()
      encodedran = self.encode_window(w+1, data[:, -1]).detach()

      #encodedtestdom = self.encode_window(w, datatest[:, 0]).detach()
      encodedtestran = self.encode_window(w+1, datatest[:, -1]).detach()

      #t0 = self.windowvals[w][0]
      #t1 = self.windowvals[w][-1]
      #encodedinputs = torch.tensor(self.propagate(encodeddom, t0, t1-t0, arrencoded=True, fixedw=w)[-1].detach())
      #encodedtestinputs = torch.tensor(self.propagate(encodedtestdom, t0, t1-t0, arrencoded=True, fixedw=w)[-1].detach())
      
      if propagated_trans:
        encodeddom = self.encode_window(w, data[:, 0]).detach()
        encodedtestdom = self.encode_window(w, datatest[:, 0]).detach()

        t0 = self.windowvals[w][0]
        t1 = self.windowvals[w][-1]
        encodedinputs = torch.tensor(self.propagate(encodeddom, t0, t1-t0, arrencoded=True, fixedw=w)[-1].detach())
        encodedtestinputs = torch.tensor(self.propagate(encodedtestdom, t0, t1-t0, arrencoded=True, fixedw=w)[-1].detach())
      else:
        encodedinputs = self.encode_window(w, data[:, -1]).detach()
        encodedtestinputs = self.encode_window(w, datatest[:, -1]).detach()


      train = torch.stack((encodedinputs, encodedran), dim=2)
      test = torch.stack((encodedtestinputs, encodedtestran), dim=2).detach().cpu().numpy()

      trans = self.transcoders[w]
      opt = optim(trans.parameters(), lr=lr, weight_decay=ridge)
      scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=30, factor=0.3)

      dataloader = DataLoader(train, shuffle=False, batch_size=batch)

      writer = None
      if self.td is not None:
        name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}-weld{w}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
        writer = torch.utils.tensorboard.SummaryWriter(name)
        print("Tensorboard writer location is " + name)

      print("Number of NN trainable parameters", utils.num_params(trans))
      print(f"Starting training Weldnet transcoder {w+1}/{self.W-1} ({self.windowvals[w][-1]}) at {time.asctime()}...")
      print("train", train.shape, "test", test.shape)

      self.transstep = 0
      for ep in range(epochs):
          lossesN, testerrors1N, testerrors2N, testerrorsinfN = transcoder_epoch(trans, dataloader, optimizer=opt, writer=writer, scheduler=scheduler, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
          losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

          if best and ep > epochs // 2:
            avgloss = np.mean(lossesN)
            if avgloss < bestdict["loss"]:
              bestdict["model"] = trans.state_dict()
              bestdict["opt"] = opt.state_dict()
              bestdict["loss"] = avgloss
              bestdict["ep"] = ep
            elif verbose:
              print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")
      
      print(f"Finish training transcoder {w} at {time.asctime()}.")
      losses_all.append(losses)
      testerrors1_all.append(testerrors1)
      testerrors2_all.append(testerrors2)
      testerrorsinf_all.append(testerrorsinf)

      if best:
        trans.load_state_dict(bestdict["model"])
        opt.load_state_dict(bestdict["opt"])

    if save and False:
      dire = "savedmodels/weld/trans"
      addr = f"{dire}/{self.prefix}{self.W}w-{datetime.datetime.now().strftime('%d-%B-%Y-%H.%M')}.pickle"

      if not os.path.exists(dire):
        os.makedirs(dire)

      with open(addr, "wb") as handle:
        pickle.dump({"trans": self.transcoders, "transdata": self.transcoderdata, "datadata": self.datadata}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Transcoders saved at", addr)

    self.transepochs.append(epochs)
    end = time.time()
    self.timetaken += end - start
    print("Finished training all timewindow transcoders")
    return { "losses": losses_all, "testerrors1": testerrors1_all, "testerrors2": testerrors2_all, "testerrorsinf": testerrorsinf_all }

  def train_transcoders_krr(self, save=True, ridge=1.0, kernel="rbf", gamma=None, encoding_param=-1, printinterval=1):
    encoding_param = determine_param(self.dataset, encoding_param)

    testerrors1_all = []
    testerrors2_all = []
    testerrorsinf_all = []

    start = time.time()
    print(f"Training {self.W} WeldNet Transcoders via Kernel Ridge Regression")
    for w in range(self.W - 1):   
        data_train_raw = torch.tensor(
            self.alltrain[:, self.windowvals[w], :],
            dtype=torch.float,
            device=self.device
        )   # shape: (N_train, raw_dim)

        data_test_raw = torch.tensor(
            self.alltest[:, self.windowvals[w], :],
            dtype=torch.float,
            device=self.device
        )   # shape: (N_test, raw_dim)

        # 4b. Encode them
        encodeddom = self.encode_window(w, data_train_raw[:, 0, :])
        encodedran = self.encode_window(w+1, data_train_raw[:, -1, :]).detach().cpu().numpy() # (N_train, latent_dim)

        encodedtestdom = self.encode_window(w, data_test_raw[:, 0, :])   # (N_test, latent_dim)
        encodedtestran = self.encode_window(w+1, data_test_raw[:, -1, :]).detach().cpu().numpy()  # (N_test, latent_dim)

        # 4c. Fit a KernelRidge model for this window

        t0 = self.windowvals[w][0]
        t1 = self.windowvals[w][-1]
        encodedinputs = self.propagate(encodeddom, t0, t1-t0, arrencoded=True, fixedw=w)[-1].cpu().detach().numpy()
        kr_model = KernelRidge(alpha=ridge, kernel=kernel, gamma=gamma)
        
        if self.residualprop:
          kr_model.fit(encodedinputs, encodedran - encodedinputs)
        else:
          kr_model.fit(encodedinputs, encodedran)  

        # 4d. Store model
        if len(self.transcoders) > w:
            self.transcoders[w] = kr_model
        else:
            self.transcoders.append(kr_model)

        # 4e. Compute test-set predictions and relative errors
        encodedinputstest = self.propagate(encodedtestdom, t0, t1-t0, arrencoded=True, fixedw=w)[-1].cpu().detach().numpy()
        pred_test = kr_model.predict(encodedinputstest)  # (N_test, latent_dim)

        if self.residualprop:
          pred_test = encodedinputstest + pred_test

        # Compute relative L1, L2, L∞ for each test sample
        rel1 = np.mean(
          np.linalg.norm(encodedtestran - pred_test, axis=1, ord=1)
          / np.linalg.norm(encodedtestran, axis=1, ord=1)
        )
        rel2 = np.mean(
          np.linalg.norm(encodedtestran - pred_test, axis=1, ord=2)
          / np.linalg.norm(encodedtestran, axis=1, ord=2)
        )
        relinf = np.mean(
          np.linalg.norm(encodedtestran - pred_test, axis=1, ord=np.inf)
          / np.linalg.norm(encodedtestran, axis=1, ord=np.inf)
        )

        testerrors1_all.append(rel1)
        testerrors2_all.append(rel2)
        testerrorsinf_all.append(relinf)

        print(
            f"Window {w+1}/{self.W-1}: "
            f"Relative Test Error (L1, L2, Linf) = "
            f"{rel1:.3e}, {rel2:.3e}, {relinf:.3e}"
        )

    if save and False:
      pass

    end = time.time()
    self.timetaken += end - start
    print("Finished training all timewindow transcoders using KRR")
    return {
      "testerrors1": testerrors1_all,
      "testerrors2": testerrors2_all,
      "testerrorsinf": testerrorsinf_all
    }
  
  def train_propagators(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, encoding_param=-1, best=True, verbose=False):
    def prop_epoch(w, dataloader, writer=None, scheduler=None, optimizer=None, ep=0, printinterval=10, loss=None, testtensor=None):
      model = self.props[w]

      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []
      
      def closure(x, y, t):
        optimizer.zero_grad()

        if t is not None:
          xt = torch.cat((x, t), dim=2)
        else:
          xt = x

        if self.tiprop:
          predict = self.propagate(x[:, 0], self.windowvals[w][0], xt.shape[1], arrencoded=True)
          predict = torch.stack(predict, dim=1)
          
        elif self.accumulateprop:
          x0 = xt[:, :1]

          xlist = []
          for _ in range(xt.shape[1]):
            x0 = self.prop_forward(model, x0)
            xlist.append(x0)

          predict = torch.cat(xlist, dim=1)

        else:
          predict = self.prop_forward(model, x)

        if self.decodedprop:
          decoder = self.aes[w]
          predict_decoded = decoder.decode(predict)
          target_decoded = decoder.decode(y).detach()
          res = loss(predict_decoded, target_decoded)
        else:
          res = loss(predict, y)

        lossval = res
        lossval.backward(retain_graph=False)

        if writer is not None and self.propstep % 10 == 0:
          writer.add_scalar("main/loss", res, global_step=self.propstep)
        
        return res

      half = int(testtensor.shape[1] / 2)
      for batch in dataloader:
        self.propstep += 1
        batch = batch.to(self.device)

        x = batch[:, :-1]
        y = batch[:, 1:]

        if not self.autonomous and not self.accumulateprop:
          t = -1 + 0*torch.tensor(np.repeat(np.expand_dims(self.windowvals[w][:-1], 0), x.shape[0], axis=0)).unsqueeze(2).to(self.device).float()
        else:
          t = None

        error = optimizer.step(lambda: closure(x, y, t))
        losses.append(float(error.cpu().detach()))

        if writer is not None:
          if self.propstep % 5 == 0:
            writer.add_scalar("propagator/loss", float(error), global_step=self.propstep)

      if scheduler is not None and ep > epochs // 2:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testinputs = testtensor[:, 0, :]
        testoutputs = testtensor[:, -1, :].cpu().detach().numpy()

        t0 = self.windowvals[w][0]
        t1 = self.windowvals[w][-1]
        predict = self.propagate(testinputs, t0, t1-t0, arrencoded=True, fixedw=w)[-1].cpu().detach().numpy()

        testerr1 = np.mean(np.linalg.norm(predict - testoutputs, axis=1, ord=1) / np.linalg.norm(testoutputs, axis=1, ord=1))
        testerr2 = np.mean(np.linalg.norm(predict - testoutputs, axis=1, ord=2) / np.linalg.norm(testoutputs, axis=1, ord=2))
        testerrinf = np.mean(np.linalg.norm(predict - testoutputs, axis=1, ord=np.inf) / np.linalg.norm(testoutputs, axis=1, ord=np.inf))

        testerrors1.append(testerr1)
        testerrors2.append(testerr2)
        testerrorsinf.append(testerrinf)
        
        if scheduler is not None:
          print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative Propagator Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{ep+1}: Train Loss {error:.3e}, Relative Propagator Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1prop", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2prop", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInfprop", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf
    
    loss = nn.MSELoss() if loss is None else loss()
    encoding_param = determine_param(self.dataset, encoding_param)

    losses_all, testerrors1_all, testerrors2_all, testerrorsinf_all = [], [], [], []
   
    start = time.time()
    print(f"Training {self.W} WeldNet Propagators")
    for w in range(self.W):
      if len(self.props) <= w:
        self.props.append(self.propclass(**self.propparams) if self.propclass not in Other_Modules else propclass(propclass.copy()))

      losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
      bestdict = { "loss": float(np.inf), "ep": 0 }

      self.propstep = 0

      data = torch.tensor(self.alltrain[:, self.windowvals[w], :], dtype=torch.float, device=self.device)
      datatest = torch.tensor(self.alltest[:, self.windowvals[w], :], dtype=torch.float, device=self.device)
      
      encoded = self.encode_window(w, data).detach()
      encoded = encoded[torch.randperm(encoded.shape[0]), :]
      encodedTest = self.encode_window(w, datatest).detach()

      prop = self.props[w]
      opt = optim(prop.parameters(), lr=lr, weight_decay=ridge)
      scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
                                      
      dataloader = DataLoader(encoded, batch_size=batch)
      writer = None
      
      if self.td is not None:
          name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}-weldprop{w}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
          writer = torch.utils.tensorboard.SummaryWriter(name)
          print("Tensorboard writer location is " + name)

      print("Number of NN trainable parameters", utils.num_params(prop))
      print(f"Starting training prop {w+1}/{self.W} at {time.asctime()}...")
      print("train", encoded.shape, "test", encodedTest.shape)

      self.propstep = 0
      for ep in range(epochs):
          lossesN, testerrors1N, testerrors2N, testerrorsinfN = prop_epoch(w, dataloader, optimizer=opt, writer=writer, ep=ep, scheduler=scheduler, printinterval=printinterval, loss=loss, testtensor=encodedTest)
          losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN
      
      print(f"Finish training propagator {w+1} at {time.asctime()}.")
      losses_all += losses; testerrors1_all += testerrors1; testerrors2_all += testerrors2; testerrorsinf_all += testerrorsinf

    if save and False:
      dire = "savedmodels/weld/props"
      addr = f"{dire}/{self.prefix}{self.W}w-props-{datetime.datetime.now().strftime('%d-%B-%Y-%H.%M')}.pickle"

      if not os.path.exists(dire):
        os.makedirs(dire)

      with open(addr, "wb") as handle:
        pickle.dump({"props": self.props, "propdata": self.propdata, "datadata": self.datadata, "aedata": self.aedata, "aes": self.aes}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Propagators saved at", addr)
    
    end = time.time()
    self.timetaken += end - start
    
    self.propepochs.append(epochs)
    print("Finished training all propagators")

    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf}

  def load_models(self, filename_prefix, verbose=False, min_epochs=0, together=False):
    search_path = f"savedmodels/{'weldtogether' if together else 'weld'}/{filename_prefix}*.pickle"

    matching_files = glob.glob(search_path)

    print("Searching for model files matching prefix:", filename_prefix)
    if not hasattr(self, "metadata"):
        raise ValueError("Missing self.metadata. Cannot match models without metadata. Ensure model has been initialized with same config.")

    for addr in matching_files:
      try:
        with open(addr, "rb") as handle:
          dic = pickle.load(handle)
      except Exception as e:
        if verbose:
          print(f"Skipping {addr} due to read error: {e}")
        continue

      meta = dic.get("metadata", {})
      is_match = all(
        str(meta.get(k)) == str(self.metadata.get(k))
        for k in meta.keys()
      )

      # Check if model meets the minimum epoch requirement
      model_epochs = dic.get("aeepochs")
      if model_epochs is None:
          if verbose:
              print(f"Skipping {addr} due to missing epoch metadata.")
          continue
      elif isinstance(model_epochs, list):  # handle legacy or list format
          if sum(model_epochs) < min_epochs:
              if verbose:
                  print(f"Skipping {addr} due to insufficient epochs ({sum(model_epochs)} < {min_epochs})")
              continue
      elif model_epochs < min_epochs:
          if verbose:
              print(f"Skipping {addr} due to insufficient epochs ({model_epochs} < {min_epochs})")
          continue

      if is_match:
          print("Model match found. Loading from:", addr)

          self.aes = dic["aes"]
          self.props = dic["props"]
          self.transcoders = dic["trans"]
          self.timetaken = dic["timetaken"]

          self.epochs = model_epochs

          return True
      elif verbose:
          print("Metadata mismatch in file:", addr)
          for k in self.metadata:
            if meta.get(k) != self.metadata.get(k):
              print(f"{k}: saved={meta.get(k)} vs current={self.metadata.get(k)}")

    print("Load failed. No matching models found.")
    print("Searched:", matching_files)
    return False

  def save_models(self):
    assert(len(self.aes) == self.W)
    assert(len(self.props) == self.W)

    if self.transcoders is None:
      assert(self.W == 1)
      num_paramstrans = 0
    else:
      assert(len(self.transcoders) == self.W - 1)
      num_paramstrans = sum([utils.num_params(x) for x in self.transcoders])

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    num_paramsAE = sum([utils.num_params(x) for x in self.aes])
    num_paramsprops = sum([utils.num_params(x) for x in self.props])

    # Compute total training epochs
    total_epochs = sum(self.aeepochs) 

    filename = (
        f"{self.dataset.name}_"
        f"{self.aeclass.__name__}_"
        f"{num_paramsAE}_"
        f"{num_paramsprops}_"
        f"{num_paramstrans}_"
        f"{self.seed}_"
        f"{total_epochs}ep_"
        f"{now}.pickle"
    )

    if self.metadata.get("trainedtogether", False):
      dire = "savedmodels/weldtogether"
    else:
      dire = "savedmodels/weld"

    addr = os.path.join(dire, filename)

    if not os.path.exists(dire):
        os.makedirs(dire)

    with open(addr, "wb") as handle:
        pickle.dump({
            "aes": self.aes,
            "props": self.props,
            "trans": self.transcoders,
            "aeepochs": self.aeepochs,
            "propepochs": self.propepochs, 
            "transepochs": self.transepochs,
            "metadata": self.metadata,
            "timetaken": self.timetaken
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Models saved at", addr)    

class NewTimeInputModel():
  def __init__(self, dataset, ticlass, tiinfo, activation, td=None, seed=0, device=0, residual=0):
    self.dataset = dataset
    self.device = device
    self.td = td
    self.residual = residual
  
    if self.td is None:
      self.prefix = f"{self.dataset.name}{str(ticlass.__name__)}"
    else:
      self.prefix = self.td

    torch.manual_seed(seed)
    np.random.seed(seed)
    self.seed = seed
    self.timetaken = 0

    datacopy = self.dataset.data.copy()
    self.numtrain = int(datacopy.shape[0] * 0.9)
    
    self.T = self.dataset.data.shape[1]
    self.trainarr = datacopy[:self.numtrain]
    self.testarr = datacopy[self.numtrain:]

    self.ticlass = ticlass
    self.optparams = None

    self.datadim = len(self.dataset.data.shape) - 2
    if len(tiinfo) == 1:
      self.model = self.ticlass(tiinfo[0], activation).to(self.device)
    elif len(tiinfo) == 2:
      self.model = self.ticlass(tiinfo[0], tiinfo[1], activation).to(self.device)
    else:
      assert(False)

    #self.tidata = [ticlass, tiinfo]
    #self.datadata = [np.floor(np.sum(self.dataset.data) * 100), self.dataset.data.shape]

    self.metadata = {
      "model_class": ticlass.__name__,
      "tiinfo": tiinfo,
      "activation": activation.__name__ if hasattr(activation, '__name__') else str(activation),
      "dataset_name": dataset.name,
      "data_shape": list(dataset.data.shape),
      "data_checksum": float(np.sum(dataset.data)),
      "seed": seed,
    }

    self.epochs = []

  def get_ti_errors(self, testarr, ords=(2,), times=None, aggregate=True, tstart=0):
    assert(aggregate or len(ords) == 1)
    
    if isinstance(testarr, np.ndarray):
      testarr = torch.tensor(testarr, dtype=torch.float32)

    if times is None:
      times = torch.linspace(0, 1, self.T, dtype=testarr.dtype)[tstart+1:]
  
    out = self.forward(testarr[:, tstart], times)
    
    n = testarr.shape[0]
    orig = testarr[:, tstart+1:].cpu().detach().numpy()
    out = out.cpu().detach().numpy()

    if aggregate:
      orig = orig.reshape([n, -1])
      out = out.reshape([n, -1])
      testerrs = []
      for o in ords:
        testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))

      return tuple(testerrs)
    
    else:
      o = ords[0]
      testerrs = []

      if len(times) == 1:
        t = times[0]
        origslice = orig[:, t-1].reshape([n, -1])
        outslice = out.reshape([n, -1])
        return np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)
      else:
        for t in range(orig.shape[1]):
          origslice = orig[:, t].reshape([n, -1])
          outslice = out[:, t].reshape([n, -1])
          testerrs.append(np.mean(np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)))

        return testerrs

  def forward(self, x, ts):
    if isinstance(self.model, FFNet):
      origshape = list(x.shape[-self.datadim:])

      x = x.reshape(list(x.shape[:-self.datadim]) + [-1])

      T = ts.shape[0]
      x_exp = x.unsqueeze(-2)
      t_exp = ts.reshape(*([1] * (x.dim() - 1)), -1, 1)

      x_brd = x_exp.expand(*x.shape[:-1], T, x.shape[-1])
      t_brd = t_exp.expand(*x.shape[:-1], T, 1)

      xts = torch.cat((x_brd, t_brd), dim=-1)         
  
      out = self.model(xts)


      output = out.reshape(list(out.shape)[:-1] + origshape)

      if self.residual == 2:
        return xts[..., :-1] + t_brd * output
      elif self.residual == 1:
        return xts[..., :-1] + output
      else:
        return output
        
    elif isinstance(self.model, DeepONet):
      B, S = x.shape
      device = x.device

      ts_tensor = torch.as_tensor(ts, device=device)
      spaces = torch.linspace(0, 1, S, device=device)

      s_grid, t_grid = torch.meshgrid(spaces, ts_tensor, indexing='ij')

      inputs = torch.stack((s_grid, t_grid), dim=-1)
      
      out = self.model(x, inputs)
      return out
    
    else:
      assert(False)

  def load_model(self, filename_prefix, verbose=False, min_epochs=0):
    search_path = f"savedmodels/timeinput/{filename_prefix}*.pickle"
    matching_files = glob.glob(search_path)

    print("Searching for model files matching prefix:", filename_prefix)
    if not hasattr(self, "metadata"):
        raise ValueError("Missing self.metadata. Cannot match models without metadata. Ensure model has been initialized with same config.")

    for addr in matching_files:
      try:
          with open(addr, "rb") as handle:
              dic = pickle.load(handle)
      except Exception as e:
          if verbose:
              print(f"Skipping {addr} due to read error: {e}")
          continue

      meta = dic.get("metadata", {})
      is_match = all(
          meta.get(k) == self.metadata.get(k)
          for k in meta.keys()
      )

      # Check if model meets the minimum epoch requirement
      model_epochs = dic["epochs"]
      if model_epochs is None:
          if verbose:
              print(f"Skipping {addr} due to missing epoch metadata.")
          continue
      elif isinstance(model_epochs, list):  # handle legacy or list format
          if sum(model_epochs) < min_epochs:
              if verbose:
                  print(f"Skipping {addr} due to insufficient epochs ({sum(model_epochs)} < {min_epochs})")
              continue
      elif model_epochs < min_epochs:
          if verbose:
              print(f"Skipping {addr} due to insufficient epochs ({model_epochs} < {min_epochs})")
          continue

      if is_match:
          print("Model match found. Loading from:", addr)
          self.model.load_state_dict(dic["model"])
          self.epochs = model_epochs
          self.timetaken = dic["timetaken"]
          if "opt" in dic:     
            self.optparams = dic["opt"]

          return True
      elif verbose:
          print("Metadata mismatch in file:", addr)
          for k in self.metadata:
              print(f"{k}: saved={meta.get(k)} vs current={self.metadata.get(k)}")

    print("Load failed. No matching models found.")
    print("Searched:", matching_files)
    return False

  def train_model(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, best=True, verbose=False, numts=1):
    def train_epoch(dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []

      def closure(batch):
        optimizer.zero_grad()
        
        #res = 0
        alltimes = torch.linspace(0, 1, self.T, dtype=batch.dtype)
        #for t in range(self.T):
        
        ts = random.sample(range(self.T), numts)
        res = 0
        for t in ts:
          out = self.forward(batch[:, t], alltimes[t+1:])
          res += loss(batch[:, t+1:], out)
          
        res /= len(ts)
        res.backward()
        
        if writer is not None and self.trainstep % 5 == 0:
          writer.add_scalar("main/loss", res, global_step=self.trainstep)

        return res

      for batch in dataloader:
        self.trainstep += 1
        error = optimizer.step(lambda: closure(batch))
        losses.append(float(error.cpu().detach()))

      if scheduler is not None and ep > epochs // 2:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testerr1, testerr2, testerrinf = self.get_ti_errors(testarr, ords=(1, 2, np.inf))
        if scheduler is not None:
          print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative TI Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{ep+1}: Train Loss {error:.3e}, Relative TI Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1error", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2error", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInferror", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf
  
    loss = nn.MSELoss() if loss is None else loss()

    losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
    self.trainstep = 0

    train = torch.tensor(self.trainarr, dtype=torch.float32).to(self.device)
    test = self.testarr  

    opt = optim(self.model.parameters(), lr=lr, weight_decay=ridge)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
    dataloader = DataLoader(train, shuffle=False, batch_size=batch)

    if self.optparams is not None:
      opt.load_state_dict(self.optparams)

    writer = None
    if self.td is not None:
      name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
      writer = torch.utils.tensorboard.SummaryWriter(name)
      print("Tensorboard writer location is " + name)

    print("Number of NN trainable parameters", utils.num_params(self.model))
    print(f"Starting training TI model {self.metadata['model_class']} at {time.asctime()}...")
    print("train", train.shape, "test", test.shape)
      
    start = time.time()
    bestdict = { "loss": float(np.inf), "ep": 0 }
    for ep in range(epochs):
      lossesN, testerrors1N, testerrors2N, testerrorsinfN = train_epoch(dataloader, optimizer=opt, scheduler=scheduler, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
      losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

      if best and ep > epochs // 2:
        avgloss = np.mean(lossesN)
        if avgloss < bestdict["loss"]:
          bestdict["model"] = self.model.state_dict()
          bestdict["opt"] = opt.state_dict()
          bestdict["loss"] = avgloss
          bestdict["ep"] = ep
        elif verbose:
          print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")
      
    end = time.time()
    self.timetaken += end - start
    print(f"Finished training TI model {self.metadata['model_class']} at {time.asctime()}...")

    if best:
      self.model.load_state_dict(bestdict["model"])
      opt.load_state_dict(bestdict["opt"])

    self.optparams = opt.state_dict()
    self.epochs.append(epochs)

    if save:
      now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      readable_shape = "x".join(map(str, self.metadata["tiinfo"]))

      # Compute total training epochs
      total_epochs = sum(self.epochs) if isinstance(self.epochs, list) else self.epochs

      filename = (
          f"{self.dataset.name}_"
          f"{self.ticlass.__name__}_"
          f"{self.metadata['activation']}_"
          f"{readable_shape}_"
          f"{self.seed}_"
          f"{total_epochs}ep_"
          f"{now}.pickle"
      )

      dire = "savedmodels/timeinput"
      addr = os.path.join(dire, filename)

      if not os.path.exists(dire):
          os.makedirs(dire)

      with open(addr, "wb") as handle:
          pickle.dump({
              "model": self.model.state_dict(),
              "metadata": self.metadata,
              "opt": self.optparams,
              "epochs": self.epochs,
              "timetaken": self.timetaken
          }, handle, protocol=pickle.HIGHEST_PROTOCOL)

      print("Model saved at", addr)

    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

class NewTimeInputHelper():
  def __init__(self, config):
    self.update_config(config)

  def update_config(self, config):
    self.config = deepcopy(config)

  def create_timeinput(self, dataset, config=None, **args):
    if config is None:
      config = self.config

    assert(len(dataset.data.shape) < 5)
    if len(dataset.data.shape) == 3:
      din = dataset.data.shape[2]
    else:
      din = dataset.data.shape[2] * dataset.data.shape[3]

    td = args.get("td", None)
    seed = args.get("seed", 0)
    device = args.get("device", 0)
    k = args.get("k", None)

    residual = args.get("residual", 0)

    ticlass = args.get("ticlass", config.ticlass)

    assert(ticlass == "FFNet")
    ticlass = FFNet
    ffseq = deepcopy(args.get("ffseq", config.ffseq))

    ffseq[0] = din + 1
    ffseq[-1] = din

    tiinfo = (ffseq,)

    activation = get_activation(args.get("activation", config.activation))

    return NewTimeInputModel(dataset, ticlass, tiinfo, activation, td=td, seed=seed, device=device, residual=residual)

  @staticmethod
  def get_operrs(ti, times=None, testonly=True):
    if isinstance(ti, NewTimeInputModel):
      if testonly:
        data = ti.testarr
      else:
        data = np.concatenate((ti.trainarr, ti.testarr), axis=0)

      errors = ti.get_ti_errors(data, times=times, aggregate=False)
    elif isinstance(ti, WeldNet):
      errors = WeldHelper.get_operrs(ti, testonly=testonly)
    elif isinstance(ti, HighDimProp):
      errors = HighDimPropHelper.get_operrs(ti, testonly=testonly)
    elif isinstance(ti, ELDNet):
      errors = ELDHelper.get_operrs(ti, testonly=testonly)
      
    return errors
  
  @staticmethod
  def plot_op_predicts(ti: NewTimeInputModel, testonly=True, xs=None, cmap="viridis"):
    if testonly:
      data = ti.trainarr
    else:
      data = np.concatenate((ti.trainarr, ti.testarr), axis=0)

    if xs is None:
        xs = np.linspace(0, 1, data.shape[2])

    data = torch.tensor(np.float32(data)).to(ti.device)

    times = torch.arange(1, ti.T)
    tt = times.to(ti.device) / ti.T
    predicts = ti.forward(data[:, 0], tt)
    
    predicts = predicts.cpu().detach()
    data = data.cpu().detach()

    errors = []
    n = predicts.shape[0]
    for s in times:
      currpredict = predicts[:, s-1].reshape((n, -1))
      currreference = data[:, s].reshape((n, -1))
      errors.append(np.mean(np.linalg.norm(currpredict - currreference, axis=1) / np.linalg.norm(currreference, axis=1)))
        
    print(f"Average Relative L2 Error over all times: {np.mean(errors):.4f}")

    if len(data.shape) == 3:
      fig, ax = plt.subplots(figsize=(4, 3))
    elif len(data.shape) == 4:
      fig, axes = plt.subplots(1, 4, figsize=(12, 3))
      fig.subplots_adjust(right=0.90)
      sub_ax = plt.axes([0.91, 0.15, 0.02, 0.65])

    @widgets.interact(i=(0, n-1), s=(1, ti.T-1))
    def plot_interact(i=0, s=1):
      print(f"Avg Relative L2 Error for t0 to t{s}: {errors[s-1]:.4f}")

      if len(data.shape) == 3:
        ax.clear()
        ax.set_title(f"RelL2 {np.linalg.norm(predicts[i, s-1] - data[i, s]) / np.linalg.norm(data[i, s])}")
        ax.plot(xs, data[i, 0], label="Input", linewidth=1)
        ax.plot(xs, predicts[i, s-1], label="Predicted", linewidth=1)
        ax.plot(xs, data[i, s], label="Exact", linewidth=1)
        ax.legend()
      elif len(data.shape) == 4:
        for axx in axes:
          axx.clear()

        axes[0].imshow(data[i, 0], cmap=cmap)
        axes[0].set_title("Initial")
        axes[1].imshow(data[i, s], cmap=cmap)
        axes[1].set_title("Exact")
        axes[2].imshow(predicts[i, s-1], cmap=cmap)
        axes[2].set_title("Predicted")

        cb = axes[3].imshow(np.abs(predicts[i, s-1] - data[i, s]), cmap=cmap)
        axes[3].set_title("|Difference|")
        fig.colorbar(cb, cax=sub_ax)

  @staticmethod
  def compare_operrs(models, labels=None):
    fig, ax = plt.subplots()

    if labels is None:
      labels = range(len(models))

    for lbl, x in zip(labels, models):
      operrs = TimeInputHelper.get_operrs(x, testonly=True)
      ax.plot(np.log10(operrs), label=lbl)

    ax.legend()
    #ax.set_title("Operator Error for Various #Windows")
    ax.set_ylabel("$log_{10}$(Operator Error)")
    ax.set_xlabel("Time")

    fig.tight_layout()
    return fig

  @staticmethod
  def plot_errorparams(ti, param=-1):
    if param == -1:
        # Auto-detect one varying parameter
        param = 0
        P = ti.dataset.params.shape[1]
        for p in range(P):
            if np.abs(ti.dataset.params[0, p] - ti.dataset.params[1, p]) > 0:
                param = p
                break

    l2error = np.asarray(TimeInputHelper.get_operrs(ti, times=[ti.T - 1]))
    params = ti.dataset.params

    print(params.shape, l2error.shape)

    if isinstance(param, (list, tuple, np.ndarray)) and len(param) == 2:
        # 3D scatter plot for 2 varying parameters
        x = params[:, param[0]]
        y = params[:, param[1]]
        z = l2error

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)

        ax.set_xlabel(f"Param {param[0]}")
        ax.set_ylabel(f"Param {param[1]}")
        ax.set_zlabel("Operator Error")
        fig.colorbar(sc, ax=ax, label="Operator Error")

    else:
        # Fallback to 2D scatter if param is 1D
        fig, ax = plt.subplots()
        ax.scatter(params[:, param], l2error, s=2)
        ax.set_xlabel(f"Parameter {param}")
        ax.set_ylabel("Operator Error")

    fig.tight_layout()

  @staticmethod
  def compare_errorparams(tis, labels=None, param=-1):
    if param == -1:
      param = 0
      P = tis[0].dataset.params.shape[1]

      for p in range(P):
        if np.abs(tis[0].dataset.params[0, p] - tis[0].dataset.params[1, p]) > 0:
          param = p
          break
      
    if labels is None:
      labels = [utils.num_params(x.model) for x in tis]

    fig, ax = plt.subplots()
    for lbl, ti in zip(labels, tis):
      l2error = np.asarray(TimeInputHelper.get_operrs(ti, times=[ti.T-1], testonly=False))
      print(l2error.shape, ti.dataset.params[:, param].shape)
      ax.scatter(ti.dataset.params[:, param], l2error, label=lbl, s=2)
  
    ax.set_title(f"Error vs. Parameter {param}")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Operator Error")
    ax.legend()

    fig.tight_layout()
    return fig


class TimeInputModel():
  def __init__(self, dataset, ticlass, tiinfo, activation, useparams=False, td=None, seed=0, device=0, residual=0):
    self.dataset = dataset
    self.device = device
    self.td = td
    self.useparams = useparams
    self.residual = residual
  
    if self.td is None:
      self.prefix = f"{self.dataset.name}{str(ticlass.__name__)}"
    else:
      self.prefix = self.td

    torch.manual_seed(seed)
    np.random.seed(seed)
    self.seed = seed
    self.timetaken = 0

    datacopy = self.dataset.data.copy()
    self.numtrain = int(datacopy.shape[0] * 0.9)
    
    self.T = self.dataset.data.shape[1]
    self.trainarr = datacopy[:self.numtrain]
    self.testarr = datacopy[self.numtrain:]

    if self.useparams:
      T = self.trainarr.shape[1] 
      
      paramtrain = np.repeat(self.dataset.params[:self.numtrain, None, :], T, axis=1)
      paramtest  = np.repeat(self.dataset.params[self.numtrain:, None, :], T, axis=1) 

      self.trainarr = np.concatenate([self.trainarr, paramtrain], axis=-1)
      self.testarr = np.concatenate([self.testarr, paramtest], axis=-1)

    self.ticlass = ticlass
    self.optparams = None

    self.datadim = len(self.dataset.data.shape) - 2
    if len(tiinfo) == 1:
      if self.useparams:
        tiinfo[0][0] += dataset.params.shape[1]

      self.model = self.ticlass(tiinfo[0], activation).to(self.device)
    elif len(tiinfo) == 2:
      assert(not self.useparams) # not yet implemented
      self.model = self.ticlass(tiinfo[0], tiinfo[1], activation).to(self.device)
    else:
      assert(False)

    #self.tidata = [ticlass, tiinfo]
    #self.datadata = [np.floor(np.sum(self.dataset.data) * 100), self.dataset.data.shape]

    self.metadata = {
      "model_class": ticlass.__name__,
      "tiinfo": tiinfo,
      "activation": activation.__name__ if hasattr(activation, '__name__') else str(activation),
      "dataset_name": dataset.name,
      "data_shape": list(dataset.data.shape),
      "data_checksum": float(np.sum(dataset.data)),
      "seed": seed,
      "useparams": self.useparams
    }

    self.epochs = []

  def get_ti_errors(self, testarr, ords=(2,), times=None, aggregate=True):
    assert(aggregate or len(ords) == 1)
    
    if isinstance(testarr, np.ndarray):
      testarr = torch.tensor(testarr, dtype=torch.float32)

    if times is None:
      times = torch.linspace(0, 1, self.T)[1:]
  
    out = self.forward(testarr[:, 0], times)
    
    n = testarr.shape[0]
    orig = testarr[:, 1:].cpu().detach().numpy()
    out = out.cpu().detach().numpy()

    if self.useparams:
      orig = orig[..., :-self.dataset.params.shape[1]]

    if aggregate:
      orig = orig.reshape([n, -1])
      out = out.reshape([n, -1])
      testerrs = []
      for o in ords:
        testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))

      return tuple(testerrs)
    
    else:
      o = ords[0]
      testerrs = []

      if len(times) == 1:
        t = times[0]
        origslice = orig[:, t-1].reshape([n, -1])
        outslice = out.reshape([n, -1])
        return np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)
      else:
        for t in range(orig.shape[1]):
          origslice = orig[:, t].reshape([n, -1])
          outslice = out[:, t].reshape([n, -1])
          testerrs.append(np.mean(np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)))

        return testerrs

  def forward(self, x, ts):
    if isinstance(self.model, FFNet):
      origshape = list(x.shape[-self.datadim:])

      x = x.reshape(list(x.shape[:-self.datadim]) + [-1])

      T = np.asarray(ts).shape[0]
      x_exp = x.unsqueeze(-2)
      t_exp = ts.reshape(*([1] * (x.dim() - 1)), -1, 1)

      x_brd = x_exp.expand(*x.shape[:-1], T, x.shape[-1])
      t_brd = t_exp.expand(*x.shape[:-1], T, 1)

      xts = torch.cat((x_brd, t_brd), dim=-1)         
  
      out = self.model(xts)

      if self.useparams:
        origshape[-1] -= self.dataset.params.shape[1]

      output = out.reshape(list(out.shape)[:-1] + origshape)

      if self.residual == 2:
        return xts[..., :-1] + t_brd * output
      elif self.residual == 1:
        return xts[..., :-1] + output
      else:
        return output
        
    elif isinstance(self.model, DeepONet):
      B, S = x.shape
      device = x.device

      ts_tensor = torch.as_tensor(ts, device=device)
      spaces = torch.linspace(0, 1, S, device=device)

      s_grid, t_grid = torch.meshgrid(spaces, ts_tensor, indexing='ij')

      inputs = torch.stack((s_grid, t_grid), dim=-1)
      
      out = self.model(x, inputs)
      return out
    
    else:
      assert(False)

  def load_model(self, filename_prefix, verbose=False, min_epochs=0):
    search_path = f"savedmodels/timeinput/{filename_prefix}*.pickle"
    matching_files = glob.glob(search_path)

    print("Searching for model files matching prefix:", filename_prefix)
    if not hasattr(self, "metadata"):
        raise ValueError("Missing self.metadata. Cannot match models without metadata. Ensure model has been initialized with same config.")

    for addr in matching_files:
      try:
          with open(addr, "rb") as handle:
              dic = pickle.load(handle)
      except Exception as e:
          if verbose:
              print(f"Skipping {addr} due to read error: {e}")
          continue

      meta = dic.get("metadata", {})
      is_match = all(
          meta.get(k) == self.metadata.get(k)
          for k in meta.keys()
      )

      # Check if model meets the minimum epoch requirement
      model_epochs = dic["epochs"]
      if model_epochs is None:
          if verbose:
              print(f"Skipping {addr} due to missing epoch metadata.")
          continue
      elif isinstance(model_epochs, list):  # handle legacy or list format
          if sum(model_epochs) < min_epochs:
              if verbose:
                  print(f"Skipping {addr} due to insufficient epochs ({sum(model_epochs)} < {min_epochs})")
              continue
      elif model_epochs < min_epochs:
          if verbose:
              print(f"Skipping {addr} due to insufficient epochs ({model_epochs} < {min_epochs})")
          continue

      if is_match:
          print("Model match found. Loading from:", addr)
          self.model.load_state_dict(dic["model"])
          self.epochs = model_epochs
          self.timetaken = dic["timetaken"]
          if "opt" in dic:     
            self.optparams = dic["opt"]

          return True
      elif verbose:
          print("Metadata mismatch in file:", addr)
          for k in self.metadata:
              print(f"{k}: saved={meta.get(k)} vs current={self.metadata.get(k)}")

    print("Load failed. No matching models found.")
    print("Searched:", matching_files)
    return False

  def train_model(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, best=True, verbose=False):
    def train_epoch(dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []

      def closure(batch):
        optimizer.zero_grad()
        
        out = self.forward(batch[:, 0], torch.linspace(0, 1, self.T, dtype=batch._dtype)[1:])

        if self.useparams:
          batch = batch[..., :-self.dataset.params.shape[1]]

        res = loss(batch[:, 1:], out)
        res.backward()
        
        if writer is not None and self.trainstep % 5 == 0:
          writer.add_scalar("main/loss", res, global_step=self.trainstep)

        return res

      for batch in dataloader:
        self.trainstep += 1
        error = optimizer.step(lambda: closure(batch))
        losses.append(float(error.cpu().detach()))

      if scheduler is not None and ep > epochs // 2:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testerr1, testerr2, testerrinf = self.get_ti_errors(testarr, ords=(1, 2, np.inf))
        if scheduler is not None:
          print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative TI Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{ep+1}: Train Loss {error:.3e}, Relative TI Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1error", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2error", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInferror", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf
  
    loss = nn.MSELoss() if loss is None else loss()

    losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
    self.trainstep = 0

    train = torch.tensor(self.trainarr, dtype=torch.float32).to(self.device)
    test = self.testarr  

    opt = optim(self.model.parameters(), lr=lr, weight_decay=ridge)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
    dataloader = DataLoader(train, shuffle=False, batch_size=batch)

    if self.optparams is not None:
      opt.load_state_dict(self.optparams)

    writer = None
    if self.td is not None:
      name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
      writer = torch.utils.tensorboard.SummaryWriter(name)
      print("Tensorboard writer location is " + name)

    print("Number of NN trainable parameters", utils.num_params(self.model))
    print(f"Starting training TI model {self.metadata['model_class']} at {time.asctime()}...")
    print("train", train.shape, "test", test.shape)
      
    start = time.time()
    bestdict = { "loss": float(np.inf), "ep": 0 }
    for ep in range(epochs):
      lossesN, testerrors1N, testerrors2N, testerrorsinfN = train_epoch(dataloader, optimizer=opt, scheduler=scheduler, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
      losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

      if best and ep > epochs // 2:
        avgloss = np.mean(lossesN)
        if avgloss < bestdict["loss"]:
          bestdict["model"] = self.model.state_dict()
          bestdict["opt"] = opt.state_dict()
          bestdict["loss"] = avgloss
          bestdict["ep"] = ep
        elif verbose:
          print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")
      
    end = time.time()
    self.timetaken += end - start
    print(f"Finished training TI model {self.metadata['model_class']} at {time.asctime()}...")

    if best:
      self.model.load_state_dict(bestdict["model"])
      opt.load_state_dict(bestdict["opt"])

    self.optparams = opt.state_dict()
    self.epochs.append(epochs)

    if save:
      now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      readable_shape = "x".join(map(str, self.metadata["tiinfo"]))

      # Compute total training epochs
      total_epochs = sum(self.epochs) if isinstance(self.epochs, list) else self.epochs

      filename = (
          f"{self.dataset.name}_"
          f"{self.ticlass.__name__}_"
          f"{self.metadata['activation']}_"
          f"{readable_shape}_"
          f"{self.seed}_"
          f"{total_epochs}ep_"
          f"{now}.pickle"
      )

      dire = "savedmodels/timeinput"
      addr = os.path.join(dire, filename)

      if not os.path.exists(dire):
          os.makedirs(dire)

      with open(addr, "wb") as handle:
          pickle.dump({
              "model": self.model.state_dict(),
              "metadata": self.metadata,
              "opt": self.optparams,
              "epochs": self.epochs,
              "timetaken": self.timetaken
          }, handle, protocol=pickle.HIGHEST_PROTOCOL)

      print("Model saved at", addr)

    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

class TimeInputHelper():
  def __init__(self, config):
    self.update_config(config)

  def update_config(self, config):
    self.config = deepcopy(config)

  def create_timeinput(self, dataset, config=None, **args):
    if config is None:
      config = self.config

    assert(len(dataset.data.shape) < 5)
    if len(dataset.data.shape) == 3:
      din = dataset.data.shape[2]
    else:
      din = dataset.data.shape[2] * dataset.data.shape[3]

    td = args.get("td", None)
    seed = args.get("seed", 0)
    device = args.get("device", 0)
    k = args.get("k", None)

    useparams = args.get("useparams", False)
    residual = args.get("residual", 0)

    ticlass = args.get("ticlass", config.ticlass)
    if ticlass == "DeepONet":
      ticlass = DeepONet
      tibranch = deepcopy(args.get("branchseq", config.branchseq))
      titrunk = deepcopy(args.get("trunkseq", config.trunkseq))

      tibranch[0] = din
      titrunk[0] = len(dataset.data.shape) - 2 + 1

      if k is not None:
          tibranch[-1] = k
          titrunk[-1] = k

      tiinfo = (tibranch, titrunk)

    else:
      assert(ticlass == "FFNet")
      ticlass = FFNet
      ffseq = deepcopy(args.get("ffseq", config.ffseq))

      ffseq[0] = din + 1
      ffseq[-1] = din

      tiinfo = (ffseq,)

    activation = get_activation(args.get("activation", config.activation))

    return TimeInputModel(dataset, ticlass, tiinfo, activation, td=td, seed=seed, device=device, useparams=useparams, residual=residual)

  @staticmethod
  def get_operrs(ti, times=None, testonly=True):
    if isinstance(ti, TimeInputModel):
      if testonly:
        data = ti.testarr
      else:
        data = np.concatenate((ti.trainarr, ti.testarr), axis=0)

      errors = ti.get_ti_errors(data, times=times, aggregate=False)
    elif isinstance(ti, WeldNet):
      errors = WeldHelper.get_operrs(ti, testonly=testonly)
    elif isinstance(ti, HighDimProp):
      errors = HighDimPropHelper.get_operrs(ti, testonly=testonly)
    elif isinstance(ti, ELDNet):
      errors = ELDHelper.get_operrs(ti, testonly=testonly)
      
    return errors
  
  @staticmethod
  def plot_op_predicts(ti: TimeInputModel, testonly=True, xs=None, cmap="viridis"):
    if testonly:
      data = ti.trainarr
    else:
      data = np.concatenate((ti.trainarr, ti.testarr), axis=0)

    if xs is None:
      if ti.useparams:
        xs = np.linspace(0, 1, data.shape[2] - ti.dataset.params.shape[1])
      else:
        xs = np.linspace(0, 1, data.shape[2])

    data = torch.tensor(np.float32(data)).to(ti.device)

    times = torch.arange(1, ti.T)
    tt = times.to(ti.device) / ti.T
    predicts = ti.forward(data[:, 0], tt)
    
    predicts = predicts.cpu().detach()
    data = data.cpu().detach()

    if ti.useparams:
      data = data[..., :-ti.dataset.params.shape[1]]

    errors = []
    n = predicts.shape[0]
    for s in times:
      currpredict = predicts[:, s-1].reshape((n, -1))
      currreference = data[:, s].reshape((n, -1))
      errors.append(np.mean(np.linalg.norm(currpredict - currreference, axis=1) / np.linalg.norm(currreference, axis=1)))
        
    print(f"Average Relative L2 Error over all times: {np.mean(errors):.4f}")

    if len(data.shape) == 3:
      fig, ax = plt.subplots(figsize=(4, 3))
    elif len(data.shape) == 4:
      fig, axes = plt.subplots(1, 4, figsize=(12, 3))
      fig.subplots_adjust(right=0.90)
      sub_ax = plt.axes([0.91, 0.15, 0.02, 0.65])

    @widgets.interact(i=(0, n-1), s=(1, ti.T-1))
    def plot_interact(i=0, s=1):
      print(f"Avg Relative L2 Error for t0 to t{s}: {errors[s-1]:.4f}")

      if len(data.shape) == 3:
        ax.clear()
        ax.set_title(f"RelL2 {np.linalg.norm(predicts[i, s-1] - data[i, s]) / np.linalg.norm(data[i, s])}")
        ax.plot(xs, data[i, 0], label="Input", linewidth=1)
        ax.plot(xs, predicts[i, s-1], label="Predicted", linewidth=1)
        ax.plot(xs, data[i, s], label="Exact", linewidth=1)
        ax.legend()
      elif len(data.shape) == 4:
        for axx in axes:
          axx.clear()

        axes[0].imshow(data[i, 0], cmap=cmap)
        axes[0].set_title("Initial")
        axes[1].imshow(data[i, s], cmap=cmap)
        axes[1].set_title("Exact")
        axes[2].imshow(predicts[i, s-1], cmap=cmap)
        axes[2].set_title("Predicted")

        cb = axes[3].imshow(np.abs(predicts[i, s-1] - data[i, s]), cmap=cmap)
        axes[3].set_title("|Difference|")
        fig.colorbar(cb, cax=sub_ax)

  @staticmethod
  def compare_operrs(models, labels=None):
    fig, ax = plt.subplots()

    if labels is None:
      labels = range(len(models))

    for lbl, x in zip(labels, models):
      operrs = TimeInputHelper.get_operrs(x, testonly=True)
      ax.plot(np.log10(operrs), label=lbl)

    ax.legend()
    #ax.set_title("Operator Error for Various #Windows")
    ax.set_ylabel("$log_{10}$(Operator Error)")
    ax.set_xlabel("Time")

    fig.tight_layout()
    return fig

  @staticmethod
  def plot_errorparams(ti, param=-1):
    if param == -1:
        # Auto-detect one varying parameter
        param = 0
        P = ti.dataset.params.shape[1]
        for p in range(P):
            if np.abs(ti.dataset.params[0, p] - ti.dataset.params[1, p]) > 0:
                param = p
                break

    l2error = np.asarray(TimeInputHelper.get_operrs(ti, times=[ti.T - 1]))
    params = ti.dataset.params

    print(params.shape, l2error.shape)

    if isinstance(param, (list, tuple, np.ndarray)) and len(param) == 2:
        # 3D scatter plot for 2 varying parameters
        x = params[:, param[0]]
        y = params[:, param[1]]
        z = l2error

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)

        ax.set_xlabel(f"Param {param[0]}")
        ax.set_ylabel(f"Param {param[1]}")
        ax.set_zlabel("Operator Error")
        fig.colorbar(sc, ax=ax, label="Operator Error")

    else:
        # Fallback to 2D scatter if param is 1D
        fig, ax = plt.subplots()
        ax.scatter(params[:, param], l2error, s=2)
        ax.set_xlabel(f"Parameter {param}")
        ax.set_ylabel("Operator Error")

    fig.tight_layout()

  @staticmethod
  def compare_errorparams(tis, labels=None, param=-1):
    if param == -1:
      param = 0
      P = tis[0].dataset.params.shape[1]

      for p in range(P):
        if np.abs(tis[0].dataset.params[0, p] - tis[0].dataset.params[1, p]) > 0:
          param = p
          break
      
    if labels is None:
      labels = [utils.num_params(x.model) for x in tis]

    fig, ax = plt.subplots()
    for lbl, ti in zip(labels, tis):
      l2error = np.asarray(TimeInputHelper.get_operrs(ti, times=[ti.T-1], testonly=False))
      print(l2error.shape, ti.dataset.params[:, param].shape)
      ax.scatter(ti.dataset.params[:, param], l2error, label=lbl, s=2)
  
    ax.set_title(f"Error vs. Parameter {param}")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Operator Error")
    ax.legend()

    fig.tight_layout()
    return fig

class WeldHelper():
  def __init__(self, config):
    self.update_config(config)

  def update_config(self, config):
    self.config = config

  def create_weld(self, dataset, config=None, **args):
    if config is None:
      config = self.config

    assert(len(dataset.data.shape) < 5)

    windows = args.get("windows", config.windows)
    aeclass = globals()[args.get("aeclass", config.aeclass)]
    aeparams = dict(args.get("aeparams", config.aeparams))
    seqclass = globals()[args.get("seqclass", config.seqclass)]
    seqparams = dict(args.get("seqparams", config.seqparams))

    if len(dataset.data.shape) == 3:
      din = dataset.data.shape[2]
    else:
      din = dataset.data.shape[2] * dataset.data.shape[3]

    if aeclass in Other_Modules:
      aeparams = OmegaConf.create(aeparams)
      aeparams.sample.spatial_resolution = din
      if "k" in args:
        aeparams.sample.latents_dims = args["k"]
        seqparams["seq"][0] = args["k"]
        seqparams["seq"][-1] = args["k"]
      seqparams["activation"] = get_activation(seqparams["activation"])

      if aeclass == TCAutoencoderConv:
        aeparams["decodeSeq"][0] = args["k"]
    else:
      if aeclass == PCAAutoencoder:
        aeparams["inputdim"] = din
        aeparams["inputdim"] = din
        seqparams["activation"] = get_activation(seqparams["activation"])
        if "k" in args:
          aeparams["reduced"] = args["k"]
          seqparams["seq"][0] = args["k"]
          seqparams["seq"][-1] = args["k"]

      else:
        aeparams["encodeSeq"][0] = din

        if aeclass != TCAutoencoder:
          aeparams["decodeSeq"][-1] = din

        aeparams["activation"] = get_activation(aeparams["activation"])
        seqparams["activation"] = get_activation(seqparams["activation"])
        if "k" in args:
          aeparams["encodeSeq"][-1] = args["k"]
          aeparams["decodeSeq"][0] = args["k"]
          seqparams["seq"][0] = args["k"]
          seqparams["seq"][-1] = args["k"]

    straightness = args.get("straightness", 0)
    kinetic = args.get("kinetic", 0)
    td = args.get("td", None)
    seed = args.get("seed", 0)
    device = args.get("device", 0)
    autonomous = args.get("autonomous", True)
    accumulateprop = args.get("accumulateprop", False)
    decodedprop = args.get("decodedprop", False)
    passparams = args.get("passparams", False)
    dynamicwindow = args.get("dynamicwindow", False)

    if aeclass == GIAutoencoder or aeclass == TCAutoencoder or aeclass == TCAutoencoderConv:
      # assume it is an integer, equally spaced grid on [0, 1]
      domaingrid = np.linspace(0, 1, dataset.data.shape[-1]).reshape(-1, 1)
      aeparams["domaingrid"] = domaingrid.tolist()
      #aeparams["decoderActivation"] = get_activation(aeparams["decoderActivation"])

    propparams = copy.deepcopy(seqparams)
    tiprop = args.get("tiprop", False)
    if tiprop:
      propparams["seq"][0] = propparams["seq"][-1] + 1

    return WeldNet(dataset, windows, aeclass, aeparams.copy(), seqclass, propparams, seqclass, copy.deepcopy(seqparams), dynamicwindow=dynamicwindow, passparams=passparams, straightness=straightness,accumulateprop=accumulateprop, decodedprop=decodedprop, kinetic=kinetic, td=td, tiprop=tiprop, seed=seed, device=device, autonomous=autonomous)
    #return WeldNet(dataset, windows, aeclass, aeparams, seqclass, seqparams, seqclass, seqparams, straightness=straightness, accumulateprop=accumulateprop, decodedprop=decodedprop, kinetic=kinetic, td=td, seed=seed, device=device, autonomous=autonomous)
  
  @staticmethod
  def plot_encoding_window(weld, ws=-1, p=-1, writer=None, step=None, tensorboard=False, maxscatter=10000, testonly=False, threedim=False):
    if p == -1:
      p = utils.determine_params(weld.dataset.params)

    if type(p) == type(0):
      p = [p]

    if ws == -1:
      ws = range(len(weld.aes))
    
    if type(ws) == type(0):
      ws = [ws]

    for w in ws:
      dim = weld.aes[w].reduced

      arr = weld.tests[w] if testonly else np.concatenate([weld.tests[w], weld.trains[w].cpu().numpy()]) 
      arr, params = utils.collect_times_dataparams(arr, weld.dataset.params)
      arr = torch.tensor(arr).to(weld.device, dtype=torch.float32)
      enc = weld.encode_window(w, arr)

      points = enc.cpu().detach().numpy()
      
      plt.rcParams.update({'font.size': 12})
      for pp in p:
        fig = plt.figure(figsize=(8, 3))

        if dim == 2 or tensorboard or not threedim:
          ax0 = fig.add_subplot(121)
          ax1 = fig.add_subplot(122)

          sc0 = ax0.scatter(points[:maxscatter, 0], points[:maxscatter, 1], c=params[:maxscatter, pp], s=2)
          plt.colorbar(sc0, ax=ax0, location="right", pad=0)
          ax0.set_xlabel("$z_1$")
          ax0.set_ylabel("$z_2$")
          ax0.set_title(f"Parameter {pp}")
          
          sc1 = ax1.scatter(points[:maxscatter, 0], points[:maxscatter, 1], c=params[:maxscatter, -1], s=2)
          plt.colorbar(sc1, ax=ax1, location="right", pad=0)
          ax1.set_xlabel("$z_1$")
          ax1.set_ylabel("$z_2$")
          ax1.set_title(f"Time")
        elif dim >= 3:
          ax0 = fig.add_subplot(121, projection="3d")
          ax1 = fig.add_subplot(122, projection="3d")

          sc0 = ax0.scatter(points[:maxscatter, 0], points[:maxscatter, 1], points[:maxscatter, 2], c=params[:maxscatter, pp], s=2)
          plt.colorbar(sc0, ax=ax0, location="right", pad=0)
          ax0.set_xlabel("$z_1$")
          ax0.set_ylabel("$z_2$")
          ax0.set_zlabel("$z_3$")
          ax0.set_title(f"Parameter {pp}")

          sc1 = ax1.scatter(points[:maxscatter, 0], points[:maxscatter, 1], points[:maxscatter, 2], c=params[:maxscatter, -1], s=2)
          plt.colorbar(sc1, ax=ax1, location="right", pad=0)
          ax1.set_xlabel("$z_1$")
          ax1.set_ylabel("$z_2$")
          ax1.set_zlabel("$z_3$")
          ax1.set_title(f"Time")
        else:
          assert(False)

        fig.tight_layout()

        if tensorboard:
          assert(step is not None)
          fig.suptitle(step)
          writer.add_figure(f'main/latent-p{pp}', fig, global_step=step)
          writer.flush()
          torch.cuda.empty_cache()
          plt.close(fig)

  @staticmethod
  def plot_encoding_time(weld, ts=-1, p=-1, writer=None, step=None, tensorboard=False, maxscatter=10000, testonly=False, threedim=False):
    if p == -1:
      p = utils.determine_params(weld.dataset.params)

    if type(p) == type(0):
      p = [p]

    if ts == -1:
      ts = range(weld.T)
    
    if type(ts) == type(0):
      ts = [ts]

    allpoints = []
    allparams = []

    if testonly:
      dataset = weld.alltest
    else:
      dataset = np.concatenate([weld.alltest, weld.alltrain]) 

    for t in ts:
      w = weld.find_window(t)
      dim = weld.aes[w].reduced

      arr = dataset[:, t:t+1, :]
      arr, params = utils.collect_times_dataparams(arr, weld.dataset.params)
      arr = torch.tensor(arr).to(weld.device, dtype=torch.float32)
      enc = weld.encode_window(w, arr)

      points = enc.cpu().detach().numpy()
      points = points[:(maxscatter // len(ts))+1]
      params = params[:(maxscatter // len(ts))+1]

      allpoints.append(points)
      allparams.append(params)

    points = np.concatenate(allpoints, axis=0)
    params = np.concatenate(allparams, axis=0)
      
    plt.rcParams.update({'font.size': 12})
    for pp in p:
      fig = plt.figure(figsize=(8, 3))

      if dim == 2 or tensorboard or not threedim:
        ax0 = fig.add_subplot()

        sc0 = ax0.scatter(points[:maxscatter, 0], points[:maxscatter, 1], c=params[:maxscatter, pp], s=2)
        plt.colorbar(sc0, ax=ax0, location="right", pad=0)
        ax0.set_xlabel("$z_1$")
        ax0.set_ylabel("$z_2$")
        ax0.set_title(f"Parameter {pp}")
      elif dim >= 3:
        ax0 = fig.add_subplot(projection="3d")

        sc0 = ax0.scatter(points[:maxscatter, 0], points[:maxscatter, 1], points[:maxscatter, 2], c=params[:maxscatter, pp], s=2)
        plt.colorbar(sc0, ax=ax0, location="right", pad=0)
        ax0.set_xlabel("$z_1$")
        ax0.set_ylabel("$z_2$")
        ax0.set_zlabel("$z_3$")
        ax0.set_title(f"Parameter {pp}")

      else:
        assert(False)

      fig.tight_layout()

      if tensorboard:
        assert(step is not None)
        fig.suptitle(step)
        writer.add_figure(f'main/latent-p{pp}', fig, global_step=step)
        writer.flush()
        torch.cuda.empty_cache()
        plt.close(fig)

    return fig

  @staticmethod
  def compare_pcaproj(weld, k=10, testonly=False, windowaverage=False):
    aeerrs = [] 
    pcaerrs = []

    for w in range(weld.W):
      data = weld.tests[w] if testonly else weld.dataset.data[:, weld.windowvals[w], :]
      data = torch.tensor(data).to(weld.device, dtype=torch.float32)
      proj = weld.project_window(w, data)

      data = data.cpu().detach().numpy()
      proj = proj.cpu().detach().numpy()

      pca = PCA(n_components=k)
      pca = pca.fit(data.reshape(-1, data.shape[-1]))

      if windowaverage:
        aeerrs.append(np.mean(np.linalg.norm(proj - data) / np.linalg.norm(data)))
        pcaerrs.append(utils.get_pca_error(data, k))
      else:
        for t in range(data.shape[1]):
          projslice = proj[:, t]
          dataslice = data[:, t]
          aeerrs.append(np.mean(np.linalg.norm(projslice - dataslice) / np.linalg.norm(dataslice)))

          components = pca.transform(dataslice)
          rdata = pca.inverse_transform(components)
          pcaerrs.append(np.mean(np.linalg.norm(rdata - dataslice) / np.linalg.norm(dataslice)))

    fig, ax = plt.subplots()

    if windowaverage:
      times = np.arange(weld.W)
      ax.set_xlabel("Window")
    else:
      times = np.arange(weld.T)
      ax.set_xlabel("Time")

    ax.plot(times, aeerrs, marker='o', label="AE")
    ax.plot(times, pcaerrs, marker='o', label=f"PCA{k}")
    
    ax.set_ylabel("RelL2 Reconstruction Error")
    ax.legend()

    fig.tight_layout()
  
  #todo, merge this with get_projerr
  @staticmethod
  def get_projerr_times(weld, times=None, testonly=False, relative=True):
    if times is None:
      times = range(weld.T)

    errors = []
    for t in times:
      data = weld.alltest[:, t:t+1, :] if testonly else np.concatenate([weld.alltest, weld.alltrain])[:, t:t+1, :]
      data = utils.collect_times_dataparams(data)
      data = torch.tensor(data).to(weld.device, dtype=torch.float32)

      w = weld.find_window(t)
      proj = weld.project_window(w, data).cpu().detach().numpy()
      data = data.cpu().detach().numpy()

      if weld.passparams:
        data = data[..., :-weld.dataset.params.shape[1]]

      if relative:
        errors.append(np.mean(np.linalg.norm(proj - data) / np.linalg.norm(data)))
      else:
        errors.append(np.mean(np.linalg.norm(proj - data)))

    return errors
  @staticmethod
  def get_projerr(weld, testonly=True, relative=True):
    aeerrs = [] 

    for w in range(weld.W):
      data = weld.tests[w] if testonly else weld.dataset.data[:, weld.windowvals[w], :]
      data = utils.collect_times_dataparams(data)
      data = torch.tensor(data).to(weld.device, dtype=torch.float32)
      proj = weld.project_window(w, data).cpu().detach().numpy()
      data = data.cpu().detach().numpy()

      if relative:
        aeerrs.append(np.mean(np.linalg.norm(proj - data) / np.linalg.norm(data)))
      else:
        aeerrs.append(np.mean(np.linalg.norm(proj - data)))

    return aeerrs

  @staticmethod
  def get_operrs(weld, steps=-1, t=0, testonly=False, fullerror=False):
    if steps == -1:
      steps = weld.T - t - 1

    assert(t + steps < weld.T)

    if testonly:
      data = weld.alltest
    else:
      data = np.concatenate([weld.alltest, weld.alltrain])

    inputt = torch.tensor(data[:, t, :]).to(weld.device, dtype=torch.float32)

    if weld.passparams:
      references = [data[:, t+s, :-weld.dataset.params.shape[1]] for s in range(1, steps+1)]
    else:
      references = [data[:, t+s, :] for s in range(1, steps+1)]

    predicteds = weld.propagate(inputt, t, steps)
    predictedvals = [weld.decode_window(weld.find_window(t+i+1), x).cpu().detach().numpy() for i, x in enumerate(predicteds)]
    
    predicteds = [x.cpu().detach().numpy() for x in predicteds]

    inputt = inputt.cpu().detach().numpy()
  
    N = predictedvals[0].shape[0]
    predictedvals = [x.reshape([N, -1]) for x in predictedvals]
    references = [x.reshape([N, -1]) for x in references]

    errors = []
    
    for s in range(1, steps+1):
      if fullerror:
        errors.append(np.linalg.norm(predictedvals[s-1] - references[s-1], axis=1) / np.linalg.norm(references[s-1], axis=1))
      else:
        errors.append(np.mean(np.linalg.norm(predictedvals[s-1] - references[s-1], axis=1) / np.linalg.norm(references[s-1], axis=1)))

    return errors
  
  @staticmethod
  def get_properrs(weld, steps=-1, t=0, testonly=False, relative=True):
    if steps == -1:
      steps = weld.T - t - 1

    assert(t + steps < weld.T)

    if testonly:
      data = weld.dataset.data[weld.numtrain:, :, :]
    else:
      data = weld.dataset.data

    datatensor = torch.tensor(data).to(weld.device, dtype=torch.float32)
    input = datatensor[:, t, :]
      
    references = [weld.encode_window(weld.find_window(t+s+1), datatensor[:, t+s+1, :]).cpu().detach().numpy() for s in range(steps)]

    predicteds = weld.propagate(input, t, steps)
    predicteds = [x.cpu().detach().numpy() for x in predicteds]

    errors = []
    for s in range(steps):
      if relative:
        errors.append(np.mean(np.linalg.norm(predicteds[s] - references[s], axis=1) / np.linalg.norm(references[s], axis=1)))
      else:
        errors.append(np.mean(np.linalg.norm(predicteds[s] - references[s], axis=1)))

    return errors

  @staticmethod
  def plot_op_predicts(weld, t=0, steps=-1, xs=None, testonly=False, cmap="viridis"):
    if steps == -1:
      steps = weld.T - t - 1
      print(t, t+steps)

    assert(t + steps < weld.T)

    data = weld.dataset.data

    if xs is None:
      xs = np.linspace(0, 1, data.shape[2])

    if weld.passparams:
      T = data.shape[1] 
    
      params = np.repeat(weld.dataset.params[:, None, :], T, axis=1)
      data = np.concatenate([data, params], axis=-1)

    if testonly:
      data = data[weld.numtrain:,]

    input = torch.tensor(data[:, t, :]).to(weld.device, dtype=torch.float32)
    references = [data[:, t+s+1, :] for s in range(steps)]

    predicteds = weld.propagate(input, t, steps)
    predictedvals = [weld.decode_window(weld.find_window(t+i+1), x).cpu().detach().numpy() for i, x in enumerate(predicteds)]
    
    predicteds = [x.cpu().detach().numpy() for x in predicteds]

    input = input.cpu().detach().numpy()

    errors = []

    if weld.passparams:
      references = [x[..., :-weld.dataset.params.shape[1]] for x in references]
      input = input[..., :-weld.dataset.params.shape[1]]
    
    n = predictedvals[0].shape[0]
    for s in range(1, steps+1):
      predict = predictedvals[s-1].reshape((n, -1))
      reference = references[s-1].reshape((n, -1))
      errors.append(np.mean(np.linalg.norm(predict - reference, axis=1) / np.linalg.norm(reference, axis=1)))
      
    print(f"Average Relative L2 Error over all times: {np.mean(errors):.4f}")

    if len(data.shape) == 3:
      fig, ax = plt.subplots(figsize=(4, 3))
    elif len(data.shape) == 4:
      fig, axes = plt.subplots(1, 4, figsize=(12, 3))
      fig.subplots_adjust(right=0.90)
      sub_ax = plt.axes([0.91, 0.15, 0.02, 0.65])

    n = references[0].shape[0]

    @widgets.interact(i=(0, n-1), s=(1, steps))
    def plot_interact(i=0, s=1):
      predict = predictedvals[s-1].reshape((n, -1))
      reference = references[s-1].reshape((n, -1))
      error = np.mean(np.linalg.norm(predict - reference, axis=1) / np.linalg.norm(reference, axis=1))
      print(f"Avg Relative L2 Error for t{t} to t{t+s}: {error:.4f}")

      if len(data.shape) == 3:
        ax.clear()
        ax.set_title(f"RelL2 {np.linalg.norm(predictedvals[s-1][i, :] - references[s-1][i, :]) / np.linalg.norm(references[s-1][i, :])}")
        ax.plot(xs, input[i], label="Input", linewidth=1)
        ax.plot(xs, predictedvals[s-1][i], label="Predicted", linewidth=1)
        ax.plot(xs, references[s-1][i], label="Exact", linewidth=1)
        ax.legend()
      elif len(data.shape) == 4:
        for axx in axes:
          axx.clear()

        axes[0].imshow(input[i], cmap=cmap)
        axes[0].set_title("Initial")
        axes[1].imshow(references[s-1][i], cmap=cmap)
        axes[1].set_title("Exact")
        axes[2].imshow(predictedvals[s-1][i], cmap=cmap)
        axes[2].set_title("Predicted")

        cb = axes[3].imshow(np.abs(predictedvals[s-1][i] - references[s-1][i]), cmap=cmap)
        axes[3].set_title("|Difference|")
        fig.colorbar(cb, cax=sub_ax)

    #return errors
  
  @staticmethod
  def plot_ae_projection(weld, xs=None, testonly=False, cmap="viridis"):
    if testonly:
      data = weld.dataset.data[weld.numtrain:,]
    else:
      data = weld.dataset.data

    if xs is None:
      xs = np.linspace(0, 1, data.shape[2])

    if weld.passparams:
      T = weld.dataset.data.shape[1] 
      params = np.repeat(weld.dataset.params[:, None, :], T, axis=1)
      data = np.concatenate([data, params], axis=-1)

    exacts = [torch.tensor(data[:, i, :]).to(weld.device, dtype=torch.float32) for i in range(data.shape[1])]
    projecteds = [weld.project_window(weld.find_window(i), exacts[i]).cpu().detach().numpy() for i in range(len(exacts))]
    exacts = [x.cpu().detach().numpy() for x in exacts]

    errors = []

    if weld.passparams:
      exacts = [x[..., :-weld.dataset.params.shape[1]] for x in exacts]
    
    for s in range(weld.T):
      exact = exacts[s]
      project = projecteds[s]
      exact = exact.reshape(list(exact.shape[:1]) + [-1])
      project = project.reshape(list(project.shape[:1]) + [-1])

      errors.append(np.mean(np.linalg.norm(exact - project, axis=1) / np.linalg.norm(exact, axis=1)))
      
    print(f"Average Relative L2 AE Error over all times: {np.mean(errors):.4f}")

    if len(data.shape) == 3:
      fig, ax = plt.subplots(figsize=(4, 3))
    elif len(data.shape) == 4:
      fig, axes = plt.subplots(1, 3, figsize=(10, 3))
      fig.subplots_adjust(right=0.90)
      sub_ax = plt.axes([0.91, 0.15, 0.02, 0.65])

    @widgets.interact(i=(0, exacts[0].shape[0]-1), s=(0, weld.T-1))
    def plot_interact(i=0, s=1):
      err = np.linalg.norm(exacts[s][i] - projecteds[s][i]) / np.linalg.norm(exacts[s][i])
      print(f"Relative L2 AE Error: {err:.4f}")
    
      if len(data.shape) == 3:
        ax.clear()
        ax.plot(xs, projecteds[s][i], label="Predicted", linewidth=1)
        ax.plot(xs, exacts[s][i], label="Exact", linewidth=1)
        ax.legend()
      elif len(data.shape) == 4:
        axes[0].clear()
        axes[0].imshow(exacts[s][i], cmap=cmap)
        axes[0].set_title("Exact")
        axes[1].clear()
        axes[1].imshow(projecteds[s][i], cmap=cmap)
        axes[1].set_title("Predicted")

        axes[2].clear()
        cb = axes[2].imshow(np.abs(projecteds[s][i] - exacts[s][i]), cmap=cmap)
        axes[2].set_title("Difference")
        fig.colorbar(cb, cax=sub_ax)


    #return errors

  @staticmethod
  def plot_prop_scatter(weld, t=0, steps=-1, p=0, testonly=False):
    if steps == -1:
      steps = weld.T - t - 1

    assert(t + steps < weld.T)

    if testonly:
      data = weld.dataset.data[weld.numtrain:, :, :]
      params = weld.dataset.params[weld.numtrain:, :, :]
    else:
      data = weld.dataset.data
      params = weld.dataset.params

    if weld.passparams:
      T = data.shape[1] 
    
      params = np.repeat(params[:, None, :], T, axis=1)
      data = np.concatenate([data, params], axis=-1)

    reference = torch.tensor(data[:, t+steps, :]).to(weld.device, dtype=torch.float32)

    window = weld.find_window(t)
    windowtarget = weld.find_window(t + steps)

    correct = weld.encode_window(windowtarget, reference)

    arr = torch.tensor(data[:, t, :]).to(weld.device, dtype=torch.float32)
    predicted = weld.propagate(arr, t, steps)[-1]
    predictedvals = weld.decode_window(windowtarget, predicted).cpu().detach().numpy()
    
    predicted = predicted.cpu().detach().numpy()
    correct = correct.cpu().detach().numpy()
    reference = reference.cpu().detach().numpy()

    if weld.passparams:
      reference = reference[..., :-weld.dataset.params.shape[1]]

    error = np.mean(np.linalg.norm(predictedvals - reference, axis=1) / np.linalg.norm(reference, axis=1))
    print(f"Relative L2 Error for t{t} to t{t+steps}", error)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    sc0 = axes[0].scatter(predicted[:, 0], predicted[:, 1], c=params[:, p], s=2)
    sc1 = axes[1].scatter(correct[:, 0], correct[:, 1], c=params[:, p], s=2)
    plt.colorbar(sc0, ax=axes[0])
    plt.colorbar(sc1, ax=axes[1])

    axes[0].set_xlabel("Encoded Param 1")
    axes[0].set_ylabel("Encoded Param 2")
    axes[0].set_title(f"Predicted t{t+steps} from t{t}, {windowtarget - window + 1} windows")
    
    axes[1].set_xlabel("Encoded Param 1")
    axes[1].set_ylabel("Encoded Param 2")
    axes[1].set_title(f"Exact t{t+steps}")
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(predicted[:, 0], predicted[:, 1], c=params[:, p], s=10, marker="+", cmap="flag", label="Predicted")
    sc = ax.scatter(correct[:, 0], correct[:, 1], c=params[:, p], s=10, marker=".", cmap="flag", label="Exact")
    plt.colorbar(sc, ax=ax)
    ax.set_xlabel("Encoded Param 1")
    ax.set_ylabel("Encoded Param 2")
    ax.set_title(f"Comparison t{t} to t{t+steps}, {windowtarget - window + 1} windows")
    ax.legend()

  @staticmethod
  def plot_projops(weld, title="", save=False, testonly=True):
    fig, ax = plt.subplots(figsize=(8, 5))

    projerrs = WeldHelper.get_projerr_times(weld, testonly=testonly)
    operrs = WeldHelper.get_operrs(weld, testonly=testonly)

    ax.plot(range(len(projerrs)), projerrs, label="Reconstruction Error")
    ax.plot(range(1, len(operrs)+1), operrs, label="Operator Error")

    for vals in weld.windowvals[:-1]:
      v = vals[-1]
      ax.axvline(v+1, linestyle=":", alpha=0.5)

    ax.set_xlabel("t")
    ax.set_ylabel("Relative L2 Error")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(title)

    fig.tight_layout()
    
    if save:
      plt.savefig(f"{title}-projoperror.pdf")

  #start
  @staticmethod
  def compare_projops(models, labels=None, relative=True, title=None, ylims=None, difference=False, testonly=True, windowlines=True, t=0):
    fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    for i, x in enumerate(models):    
      if windowlines:
        for xx in x.windowvals[:-1]:
          if xx[-1] > t:
            ax.axvline(xx[-1], linestyle=":", alpha=0.5, color=colors[i])

      projerrs = np.asarray(WeldHelper.get_projerr_times(x, testonly=testonly, relative=relative)[t+1:])
      operrs = np.asarray(WeldHelper.get_operrs(x, testonly=testonly, t=t))

      if difference:
        ax.plot(range(t, t+len(operrs)), np.log10(operrs - projerrs), label=f"{x.W if labels is None else labels[i]} Propagator Gap", c=colors[i])
      else:
        ax.plot(range(t, t+len(operrs)), np.log10(projerrs), "--", c=colors[i], alpha=0.5)
        ax.plot(range(t, t+len(operrs)), np.log10(operrs), label=f"{x.W if labels is None else labels[i]}", c=colors[i])

    ax.legend()

    ax.set_ylabel("$log_{10}$(Error)")
    ax.set_xlabel("Time")

    if ylims:
      ax.set_ylim(ylims)

    if title:
      fig.suptitle(title)

    fig.tight_layout()
    return fig

  @staticmethod
  def compare_projerrs(models, labels=None, windowlines=True):
    fig, ax = plt.subplots()

    if labels is None:
      labels = [x.W for x in models]

    for x, l in zip(models, labels):
      properrs = WeldHelper.get_projerr_times(x, testonly=True)
      ax.plot(np.log10(properrs), label=l)

      if windowlines:
        for xx in x.windowvals[:-1]:
          ax.axvline(xx[-1] + 1, linestyle="--", alpha=0.5, color="gray")

    ax.legend()
    ax.set_title("Projection Error for Various #Windows")
    ax.set_ylabel("$log_{10}$(Projection Error)")
    ax.set_xlabel("Time")

    fig.tight_layout()
    return fig

  @staticmethod
  def compare_operrs(models, windowlines=True):
    fig, ax = plt.subplots()

    for x in models:
      properrs = WeldHelper.get_operrs(x, testonly=True)
      ax.plot(np.log10(properrs), label=x.W)

      if windowlines:
        for xx in x.windowvals[:-1]:
          ax.axvline(xx[-1], linestyle="--", alpha=0.5, color="gray")

    ax.legend()
    ax.set_title("Operator Error for Various #Windows")
    ax.set_ylabel("$log_{10}$(Operator Error)")
    ax.set_xlabel("Time")

    fig.tight_layout()

  @staticmethod
  def compare_properrs(models, labels=None, windowlines=True):
    fig, ax = plt.subplots()

    if labels is None:
      labels = [x.W for x in models]

    for x, l in zip(models, labels):
      properrs = WeldHelper.get_properrs(x, testonly=True)
      ax.plot(np.log10(properrs), label=l)

      if windowlines:
        for xx in x.windowvals[:-1]:
          ax.axvline(xx[-1], linestyle="--", alpha=0.5, color="gray")

    ax.legend()
    ax.set_title("Propagator Error for Various #Windows")
    ax.set_ylabel("$log_{10}$(Propagator Error)")
    ax.set_xlabel("Time")

    fig.tight_layout()
    return fig

  @staticmethod
  def plot_latent_trajectory(weld, testnums, t=0, steps=-1, threed=False, figax=None):
    shapes = ["*"]
    shapesactual = ["o"]
    if steps == -1:
      steps = weld.T - t - 1

    data = weld.dataset.data
    if weld.passparams:
      T = weld.dataset.data.shape[1] 
    
      params = np.repeat(weld.dataset.params[:, None, :], T, axis=1)
      data = np.concatenate([data, params], axis=-1)

    arr = torch.tensor(data[testnums, :, :]).to(weld.device, dtype=torch.float32)
    outlist = weld.propagate(arr[:, t, :], t, steps)
    actual = [weld.encode_window(weld.find_window(tt), arr[:, tt, :]) for tt in range(t + 1, t + steps + 1)]
    
    actualpoints = torch.stack(actual, dim=2).cpu().detach()
    points = torch.stack(outlist, dim=2).cpu().detach()

    if figax is None:
      fig = plt.figure()
      if threed:
        ax = fig.add_subplot(projection="3d")
      else:
        ax = fig.add_subplot()
    else:
      fig, ax = figax
      ax.clear()

    for i in range(points.shape[0]):
      if ax.name == "3d":
        ax.scatter(actualpoints[i, 0, :], actualpoints[i, 1, :], actualpoints[i, 2, :], marker=shapesactual[i], c=range(actualpoints.shape[2]), cmap="cool", s=15)
        sc = ax.scatter(points[i, 0, :], points[i, 1, :], points[i, 2, :], marker=shapes[i], c=range(points.shape[2]), cmap="cool", s=10)
      else:
        ax.scatter(actualpoints[i, 0], actualpoints[i, 1], marker=shapesactual[i], edgecolor="black", linewidths=0.5, c=range(actualpoints.shape[2]), cmap="cool", s=15)
        sc = ax.scatter(points[i, 0], points[i, 1], marker=shapes[i], c=range(points.shape[2]), cmap="cool", s=10)

    #fig.colorbar(sc, ax=ax)
    ax.set_title("o predicted, + exact")

    return (fig, ax)

  @staticmethod
  def plot_coordinate_props(weld, title="", num=None, steps=-1, i=0, difference=True, allwindows=True):
    figs = []

    if allwindows:
      ws = range(len(weld.windowvals))
    else:
      ws = [0]
      
    for w in ws:
      if steps == -1:
        steps = weld.windowvals[w][-1]

      data = weld.dataset.data

      if weld.passparams:
        T = data.shape[1] 
      
        params = np.repeat(weld.dataset.params[:, None, :], T, axis=1)
        data = np.concatenate([data, params], axis=-1)

      if num is None:
        num = np.random.randint(data.shape[0])

      tstart = weld.windowvals[w][0]
      arr = torch.tensor(data[num, tstart:(tstart+1), :]).to(weld.device).float()

      predicted = torch.stack(weld.propagate(arr, tstart, steps)).cpu().detach()
      correct = weld.encode_window(w, torch.tensor(data[num, tstart+1:tstart+steps+1, :]).float().to(weld.device)).cpu().detach()

      #print(predicted.shape, correct.shape)

      colors = cm.get_cmap('tab20', predicted.shape[2])

      if difference:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        ax = axes[0]
        ax1 = axes[1]
      else:
        fig, ax = plt.subplots(figsize=(5, 3))

      for j in range(predicted.shape[2]):
        ax.plot(predicted[:, 0, j], color=colors(j), label=f"Dimension {j}")
        ax.plot(correct[:, j], linestyle=":", color=colors(j))
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        if difference:
          ax1.plot(np.abs(predicted[:, 0, j] - correct[:, j]), color=colors(j), label=f"Dimension {j}")
          ax1.set_ylabel("|Predict - Correct|")
          ax1.set_xlabel("Time")

      ax.legend()
      fig.suptitle(weld.dataset.name + " " + title + " window " + str(w+1))

      figs.append(fig)

    return figs

  
  @staticmethod
  def compare_errorparams(welds, labels=None, param=-1):
    if param == -1:
      param = 0
      P = welds[0].dataset.params.shape[1]

      for p in range(P):
        if np.abs(welds[0].dataset.params[0, p] - welds[0].dataset.params[1, p]) > 0:
          param = p
          break
      
    if labels is None:
      labels = [len(x.aes) for x in welds]

    fig, ax = plt.subplots()
    for lbl, weld in zip(labels, welds):
      op = WeldHelper.get_operrs(weld, fullerror=True)
      l2error = np.linalg.norm(np.asarray(op), axis=0)

      ax.scatter(weld.dataset.params[:, param], l2error, label=lbl, s=2)
  
    ax.set_title(f"Error vs. Parameter {param}")
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("Operator Error")
    ax.legend()

    fig.tight_layout()
    return fig

  @staticmethod
  def plot_errorparams(weld, param=-1):
    if param == -1:
      param = 0
      P = weld.dataset.params.shape[1]

      for p in range(P):
        if np.abs(weld.dataset.params[0, p] - weld.dataset.params[1, p]) > 0:
          param = p
          break
      
    op = WeldHelper.get_operrs(weld, fullerror=True)
    l2error = np.linalg.norm(np.asarray(op), axis=0)

    fig, ax = plt.subplots()
    ax.scatter(weld.dataset.params[:, param], l2error, s=2)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Operator Error")

    fig.tight_layout()
    return fig

class LDHelper():
  def __init__(self, config):
    self.update_config(config)

  def update_config(self, config):
    self.config = deepcopy(config)

  def create_ldnet(self, dataset, k, config=None, **args):
    if config is None:
      config = self.config

    assert(len(dataset.data.shape) < 4)
    if len(dataset.data.shape) == 3:
      din = dataset.params.shape[-1]
      dout = dataset.data.shape[-1]

    td = args.get("td", None)
    seed = args.get("seed", 0)
    device = args.get("device", 0)

    dynclass = globals()[args.get("dynclass", config.dynclass)]
    dynparams = copy.deepcopy(dict(args.get("dynparams", config.dynparams)))
    decclass = globals()[args.get("decclass", config.decclass)]
    recparams = copy.deepcopy(dict(args.get("recparams", config.recparams)))

    dynparams["seq"][0] = k + din
    dynparams["seq"][-1] = k
    recparams["seq"][0] = k + din
    recparams["seq"][-1] = dout

    return LDNet(dataset, k, dynclass, dynparams, decclass, recparams, td=td, seed=seed, device=device)

  @staticmethod
  def get_operrs(ldnet, times=None, testonly=False):
    if testonly:
      data = ldnet.testarr
      params = ldnet.testparams
    else:
      data = np.concatenate((ldnet.trainarr, ldnet.testarr), axis=0)
      params = np.concatenate((ldnet.trainparams, ldnet.testparams), axis=0)
    
    errors = ldnet.get_errors(data, params, times=times, aggregate=False)

    return errors
  
  @staticmethod
  def plot_op_predicts(ldnet, testonly=False, xs=None, cmap="viridis"):
    if testonly:
      data = ldnet.dataset.data[ldnet.numtrain:,]
      params = ldnet.dataset.params[ldnet.numtrain:,]
    else:
      data = ldnet.dataset.data
      params = ldnet.dataset.params

    if xs == None:
      xs = np.linspace(0, 1, len(data[0, 0]))

    params = torch.tensor(np.float32(params)).to(ldnet.device)

    predicts = ldnet.propagate(params).cpu().detach()

    errors = []
    n = predicts.shape[0]
    for s in range(data.shape[1]):
      currpredict = predicts[:, s-1].reshape((n, -1))
      currreference = data[:, s].reshape((n, -1))
      errors.append(np.mean(np.linalg.norm(currpredict - currreference, axis=1) / np.linalg.norm(currreference, axis=1)))
        
    print(f"Average Relative L2 Error over all times: {np.mean(errors):.4f}")

    if len(data.shape) == 3:
      fig, ax = plt.subplots(figsize=(4, 3))

    @widgets.interact(i=(0, n-1), s=(1, ldnet.T-1))
    def plot_interact(i=0, s=1):
      print(f"Avg Relative L2 Error for t0 to t{s}: {errors[s-1]:.4f}")

      if len(data.shape) == 3:
        ax.clear()
        ax.set_title(f"RelL2 {np.linalg.norm(predicts[i, s-1] - data[i, s]) / np.linalg.norm(data[i, s])}")
        ax.plot(xs, data[i, 0], label="Input", linewidth=1)
        ax.plot(xs, predicts[i, s-1], label="Predicted", linewidth=1)
        ax.plot(xs, data[i, s], label="Exact", linewidth=1)
        ax.legend()
        
  @staticmethod
  def plot_errorparams(ldnet, param=-1):
    if param == -1:
        # Auto-detect one varying parameter
        param = 0
        P = ldnet.dataset.params.shape[1]
        for p in range(P):
            if np.abs(ldnet.dataset.params[0, p] - ldnet.dataset.params[1, p]) > 0:
                param = p
                break

    l2error = np.asarray(LDHelper.get_operrs(ldnet, times=[ldnet.T - 1]))
    params = ldnet.dataset.params

    print(params.shape, l2error.shape)

    if isinstance(param, (list, tuple, np.ndarray)) and len(param) == 2:
        # 3D scatter plot for 2 varying parameters
        x = params[:, param[0]]
        y = params[:, param[1]]
        z = l2error

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)

        ax.set_xlabel(f"Param {param[0]}")
        ax.set_ylabel(f"Param {param[1]}")
        ax.set_zlabel("Operator Error")
        fig.colorbar(sc, ax=ax, label="Operator Error")

    else:
        # Fallback to 2D scatter if param is 1D
        fig, ax = plt.subplots()
        ax.scatter(params[:, param], l2error, s=2)
        ax.set_xlabel(f"Parameter {param}")
        ax.set_ylabel("Operator Error")

    fig.tight_layout()

class LDNet():
  def __init__(self, dataset, k, dynclass, dynparams, decclass, recparams, td, seed, device):
    self.dataset = dataset
    self.device = device
    self.td = td
    self.k = k
    self.f = self.dataset.params.shape[1]
  
    if self.td is None:
      self.prefix = f"{self.dataset.name}{str(dynclass.__name__)}LDNet-{dynparams['seq'][-1]}"
    else:
      self.prefix = self.td

    torch.manual_seed(seed)
    np.random.seed(seed)
    self.seed = seed

    self.timetaken = 0

    datacopy = self.dataset.data.copy()
    self.numtrain = int(datacopy.shape[0] * 0.9)
    
    self.T = self.dataset.data.shape[1]
    self.trainarr = datacopy[:self.numtrain]
    self.testarr = datacopy[self.numtrain:]
    self.trainparams = self.dataset.params[:self.numtrain]
    self.testparams = self.dataset.params[self.numtrain:]
    self.optparams = None

    self.datadim = len(self.dataset.data.shape) - 2

    self.dynclass = dynclass
    self.dynparams = copy.deepcopy(dynparams)
    self.decclass = decclass
    self.recparams = copy.deepcopy(recparams)

    dynparams["seq"][0] = self.k + self.f
    dynparams["seq"][-1] = self.k
    recparams["seq"][0] = self.k + self.f
    recparams["seq"][-1] = self.dataset.data.shape[-1]

    self.dynnet = dynclass(**dynparams).float().to(device)
    self.recnet = decclass(**recparams).float().to(device)

    self.metadata = {
      "dynclass": dynclass.__name__,
      "dynparams": dynparams,
      "decclass": decclass.__name__,
      "recparams": recparams,
      "dataset_name": dataset.name,
      "data_shape": list(dataset.data.shape),
      "data_checksum": float(np.sum(dataset.data)),
      "seed": seed,
    }

    self.epochs = []

  def propagate(self, code, start=0, end=-1, returncodes=False):
    if end == -1:
      end = self.T - 1

    z = code

    # get first decode
    if z.shape[-1] != self.f + self.k:
      if z.shape[-1] == self.f:
        z_fixed = z
        z_dynamic = torch.zeros(list(z_fixed.shape[:-1]) + [self.k])
        z = torch.cat([z_fixed, z_dynamic], dim=-1)
      else:
        print(z.shape)
        assert(False)

    zpreds = [z]
    for t in range(start, end):
      z = self.forward(z)
      zpreds.append(z)

    zpreds = torch.stack(zpreds, dim=1)
    upreds = self.recnet(zpreds)

    if returncodes:
      return upreds, zpreds
    else:
      return upreds

  def get_errors(self, testarr, testparams, ords=(2,), times=None, aggregate=True):
    assert(aggregate or len(ords) == 1)
    
    if isinstance(testarr, np.ndarray):
      testarr = torch.tensor(testarr, dtype=torch.float32)

    if isinstance(testparams, np.ndarray):
      testparams = torch.tensor(testparams, dtype=torch.float32)

    if times is None:
      times = range(self.T-1)
  
    out = self.propagate(testparams)

    n = testarr.shape[0]
    orig = testarr.cpu().detach().numpy()
    out = out.cpu().detach().numpy()

    if aggregate:
      orig = orig.reshape([n, -1])
      out = out.reshape([n, -1])
      testerrs = []
      for o in ords:
        testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))

      return tuple(testerrs)
    
    else:
      o = ords[0]
      testerrs = []

      if len(times) == 1:
        t = times[0]
        origslice = orig[:, t].reshape([n, -1])
        outslice = out[:, t].reshape([n, -1])
        return np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)
      else:
        for t in range(orig.shape[1]):
          origslice = orig[:, t].reshape([n, -1])
          outslice = out[:, t].reshape([n, -1])
          testerrs.append(np.mean(np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)))

        return testerrs

  def forward(self, z_full, decode=False):
    if z_full.shape[-1] == self.f + self.k:
      z_fixed = z_full[..., :self.f]
      z_dynamic = z_full[..., self.f:]

    else:
      if z_full.shape[-1] == self.f:
        z_fixed = z_full
        z_dynamic = torch.zeros(list(z_fixed.shape[:-1]) + [self.k])
        z_full = torch.cat([z_fixed, z_dynamic], dim=-1)
      else:
        print(z_full.shape)
        assert(False)

    deltaz = self.dynnet(z_full)
    z_next_dynamic = z_dynamic + deltaz
    z_next_full = torch.cat([z_fixed, z_next_dynamic], dim=-1)

    if decode:
      decoded = self.recnet(z_next_full)
      return decoded, z_next_full
    
    else:
      return z_next_full

  def load_model(self, filename_prefix, verbose=False, min_epochs=0):
    search_path = f"savedmodels/ldnet/{filename_prefix}*.pickle"
    matching_files = glob.glob(search_path)

    print("Searching for model files matching prefix:", filename_prefix)
    if not hasattr(self, "metadata"):
        raise ValueError("Missing self.metadata. Cannot match models without metadata. Ensure model has been initialized with same config.")

    for addr in matching_files:
      try:
          with open(addr, "rb") as handle:
              dic = pickle.load(handle)
      except Exception as e:
          if verbose:
              print(f"Skipping {addr} due to read error: {e}")
          continue

      meta = dic.get("metadata", {})
      is_match = all(
          meta.get(k) == self.metadata.get(k)
          for k in self.metadata.keys()
      )

      # Check if model meets the minimum epoch requirement
      model_epochs = dic["epochs"]
      if model_epochs is None:
          if verbose:
              print(f"Skipping {addr} due to missing epoch metadata.")
          continue
      elif isinstance(model_epochs, list):  # handle legacy or list format
          if sum(model_epochs) < min_epochs:
              if verbose:
                  print(f"Skipping {addr} due to insufficient epochs ({sum(model_epochs)} < {min_epochs})")
              continue
      elif model_epochs < min_epochs:
          if verbose:
              print(f"Skipping {addr} due to insufficient epochs ({model_epochs} < {min_epochs})")
          continue

      if is_match:
          print("Model match found. Loading from:", addr)
          self.dynnet.load_state_dict(dic["dynnet"])
          self.recnet.load_state_dict(dic["recnet"])
          self.epochs = model_epochs
          self.timetaken = dic["timetaken"]
          if "opt" in dic:     
            self.optparams = dic["opt"]

          return True
      elif verbose:
          print("Metadata mismatch in file:", addr)
          for k in self.metadata:
              print(f"{k}: saved={meta.get(k)} vs current={self.metadata.get(k)}")

    print("Load failed. No matching models found.")
    print("Searched:", matching_files)
    return False

  def train_model(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, best=True, verbose=False):
    def train_epoch(dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None, testparams=None):
      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []

      def closure(values, params):
        optimizer.zero_grad()

        out = self.propagate(params)
        target = values
        
        res = loss(out, target)
        res.backward()
        
        if writer is not None and self.trainstep % 5 == 0:
          writer.add_scalar("main/loss", res, global_step=self.trainstep)

        return res

      for values, params in dataloader:
        self.trainstep += 1
        error = optimizer.step(lambda: closure(values, params))
        losses.append(float(error.cpu().detach()))

      if scheduler is not None and ep > epochs // 2:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testerr1, testerr2, testerrinf = self.get_errors(testarr, testparams, ords=(1, 2, np.inf))
        if scheduler is not None:
          print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative LDNet Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{ep+1}: Train Loss {error:.3e}, Relative LDNet Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1error", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2error", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInferror", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf
  
    loss = nn.MSELoss() if loss is None else loss()

    losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
    self.trainstep = 0

    train = torch.tensor(self.trainarr, dtype=torch.float32).to(self.device)
    params = torch.tensor(self.trainparams, dtype=torch.float32).to(self.device)
    test = self.testarr

    opt = optim(itertools.chain(self.dynnet.parameters(), self.recnet.parameters()), lr=lr, weight_decay=ridge)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
    dataloader = DataLoader(torch.utils.data.TensorDataset(train, params), shuffle=False, batch_size=batch)

    if self.optparams is not None:
      opt.load_state_dict(self.optparams)

    writer = None
    if self.td is not None:
      name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
      writer = torch.utils.tensorboard.SummaryWriter(name)
      print("Tensorboard writer location is " + name)

    print("Number of NN trainable parameters", utils.num_params(self.dynnet), "+", utils.num_params(self.recnet))
    print(f"Starting training LDNet model at {time.asctime()}...")
    print("train", train.shape, "test", test.shape)
      
    start = time.time()
    bestdict = { "loss": float(np.inf), "ep": 0 }
    for ep in range(epochs):
      lossesN, testerrors1N, testerrors2N, testerrorsinfN = train_epoch(dataloader, optimizer=opt, scheduler=scheduler, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test, testparams=self.testparams)
      losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

      if best and ep > epochs // 2:
        avgloss = np.mean(lossesN)
        if avgloss < bestdict["loss"]:
          bestdict["dynnet"] = self.dynnet.state_dict()
          bestdict["recnet"] = self.recnet.state_dict()
          bestdict["opt"] = opt.state_dict()
          bestdict["loss"] = avgloss
          bestdict["ep"] = ep
        elif verbose:
          print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")
    
    end = time.time()
    self.timetaken += end - start
    print(f"Finished training LDNet model at {time.asctime()}...")
    

    if best:
      self.dynnet.load_state_dict(bestdict["dynnet"])
      self.recnet.load_state_dict(bestdict["recnet"])
      opt.load_state_dict(bestdict["opt"])

    self.optparams = opt.state_dict()
    self.epochs.append(epochs)

    if save:
      now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

      # Compute total training epochs
      total_epochs = sum(self.epochs) if isinstance(self.epochs, list) else self.epochs

      filename = (
          f"{self.dataset.name}_"
          f"{self.dynclass.__name__}_"
          f"{self.dynparams['seq']}_"
          f"{self.decclass.__name__}_"
          f"{self.recparams['seq']}_"
          f"{self.seed}_"
          f"{total_epochs}ep_"
          f"{now}.pickle"
      )

      dire = "savedmodels/ldnet"
      addr = os.path.join(dire, filename)

      if not os.path.exists(dire):
          os.makedirs(dire)

      with open(addr, "wb") as handle:
          pickle.dump({
              "dynnet": self.dynnet.state_dict(),
              "recnet": self.recnet.state_dict(),
              "metadata": self.metadata,
              "opt": self.optparams,
              "epochs": self.epochs,
              "timetaken": self.timetaken
          }, handle, protocol=pickle.HIGHEST_PROTOCOL)

      print("Model saved at", addr)

    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

class LTIHelper():
  def __init__(self, config):
    self.update_config(config)

  def update_config(self, config):
    self.config = deepcopy(config)

  def create_ltinet(self, dataset, config=None, **args):
    if config is None:
      config = self.config

    assert(len(dataset.data.shape) < 4)
    if len(dataset.data.shape) == 3:
      din = dataset.params.shape[-1]
      dout = dataset.data.shape[-1]

    td = args.get("td", None)
    seed = args.get("seed", 0)
    device = args.get("device", 0)

    recclass = globals()[args.get("recclass", config.recclass)]
    recparams = copy.deepcopy(dict(args.get("recparams", config.recparams)))

    recparams["seq"][0] = din + 1
    recparams["seq"][-1] = dout

    return LTINet(dataset, recclass, recparams, td=td, seed=seed, device=device)

  @staticmethod
  def get_operrs(ltinet, times=None, testonly=False):
    if testonly:
      data = ltinet.testarr
      params = ltinet.testparams
    else:
      data = np.concatenate((ltinet.trainarr, ltinet.testarr), axis=0)
      params = np.concatenate((ltinet.trainparams, ltinet.testparams), axis=0)

    errors = ltinet.get_errors(data, params, times=times, aggregate=False)

    return errors
  
  @staticmethod
  def plot_op_predicts(ltinet, testonly=False, xs=None, cmap="viridis"):
    if testonly:
      data = ltinet.dataset.data[ltinet.numtrain:,]
      params = ltinet.dataset.params[ltinet.numtrain:,]
    else:
      data = ltinet.dataset.data
      params = ltinet.dataset.params

    if xs == None:
      xs = np.linspace(0, 1, len(data[0, 0]))

    params = torch.tensor(np.float32(params)).to(ltinet.device)

    predicts = ltinet.propagate(params).cpu().detach()

    errors = []
    n = predicts.shape[0]
    for s in range(data.shape[1]):
      currpredict = predicts[:, s-1].reshape((n, -1))
      currreference = data[:, s].reshape((n, -1))
      errors.append(np.mean(np.linalg.norm(currpredict - currreference, axis=1) / np.linalg.norm(currreference, axis=1)))
        
    print(f"Average Relative L2 Error over all times: {np.mean(errors):.4f}")

    if len(data.shape) == 3:
      fig, ax = plt.subplots(figsize=(4, 3))

    @widgets.interact(i=(0, n-1), s=(1, ltinet.T-1))
    def plot_interact(i=0, s=1):
      print(f"Avg Relative L2 Error for t0 to t{s}: {errors[s-1]:.4f}")

      if len(data.shape) == 3:
        ax.clear()
        ax.set_title(f"RelL2 {np.linalg.norm(predicts[i, s-1] - data[i, s]) / np.linalg.norm(data[i, s])}")
        ax.plot(xs, data[i, 0], label="Input", linewidth=1)
        ax.plot(xs, predicts[i, s-1], label="Predicted", linewidth=1)
        ax.plot(xs, data[i, s], label="Exact", linewidth=1)
        ax.legend()
        
  @staticmethod
  def plot_errorparams(ltinet, param=-1):
    if param == -1:
        # Auto-detect one varying parameter
        param = 0
        P = ltinet.dataset.params.shape[1]
        for p in range(P):
            if np.abs(ltinet.dataset.params[0, p] - ltinet.dataset.params[1, p]) > 0:
                param = p
                break

    l2error = np.asarray(LTIHelper.get_operrs(ltinet, times=[ltinet.T - 1]))
    params = ltinet.dataset.params

    print(params.shape, l2error.shape)

    if isinstance(param, (list, tuple, np.ndarray)) and len(param) == 2:
        # 3D scatter plot for 2 varying parameters
        x = params[:, param[0]]
        y = params[:, param[1]]
        z = l2error

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)

        ax.set_xlabel(f"Param {param[0]}")
        ax.set_ylabel(f"Param {param[1]}")
        ax.set_zlabel("Operator Error")
        fig.colorbar(sc, ax=ax, label="Operator Error")

    else:
        # Fallback to 2D scatter if param is 1D
        fig, ax = plt.subplots()
        ax.scatter(params[:, param], l2error, s=2)
        ax.set_xlabel(f"Parameter {param}")
        ax.set_ylabel("Operator Error")

    fig.tight_layout()

class LTINet():
  def __init__(self, dataset, recclass, recparams, td, seed, device):
    self.dataset = dataset
    self.device = device
    self.td = td
    self.f = self.dataset.params.shape[1]
  
    self.timetaken = 0

    if self.td is None:
      self.prefix = f"{self.dataset.name}{str(recclass.__name__)}LTINet"
    else:
      self.prefix = self.td

    torch.manual_seed(seed)
    np.random.seed(seed)
    self.seed = seed

    datacopy = self.dataset.data.copy()
    self.numtrain = int(datacopy.shape[0] * 0.9)
    
    self.T = self.dataset.data.shape[1]
    self.trainarr = datacopy[:self.numtrain]
    self.testarr = datacopy[self.numtrain:]
    self.trainparams = self.dataset.params[:self.numtrain]
    self.testparams = self.dataset.params[self.numtrain:]
    self.optparams = None

    self.datadim = len(self.dataset.data.shape) - 2

    self.recclass = recclass
    self.recparams = copy.deepcopy(recparams)

    recparams["seq"][0] = self.f + 1
    recparams["seq"][-1] = self.dataset.data.shape[-1]

    self.recnet = recclass(**recparams).float().to(device)

    self.metadata = {
      "recclass": recclass.__name__,
      "recparams": recparams,
      "dataset_name": dataset.name,
      "data_shape": list(dataset.data.shape),
      "data_checksum": float(np.sum(dataset.data)),
      "seed": seed,
    }

    self.epochs = []

  def propagate(self, code, start=0, end=-1):
    fullts = torch.linspace(0, 1, self.T).float().to(self.device)
    
    if end > 0:
      ts = fullts[start:end+1]
    else:
      ts = fullts[start:]

    out = self.forward(code, ts)
    return out

  def get_errors(self, testarr, testparams, ords=(2,), times=None, aggregate=True):
    assert(aggregate or len(ords) == 1)
    
    if isinstance(testarr, np.ndarray):
      testarr = torch.tensor(testarr, dtype=torch.float32)

    if isinstance(testparams, np.ndarray):
      testparams = torch.tensor(testparams, dtype=torch.float32)

    if times is None:
      times = range(self.T-1)
  
    out = self.propagate(testparams)

    n = testarr.shape[0]
    orig = testarr.cpu().detach().numpy()
    out = out.cpu().detach().numpy()

    if aggregate:
      orig = orig.reshape([n, -1])
      out = out.reshape([n, -1])
      testerrs = []
      for o in ords:
        testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))

      return tuple(testerrs)
    
    else:
      o = ords[0]
      testerrs = []

      if len(times) == 1:
        t = times[0]
        origslice = orig[:, t].reshape([n, -1])
        outslice = out[:, t].reshape([n, -1])
        return np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)
      else:
        for t in range(orig.shape[1]):
          origslice = orig[:, t].reshape([n, -1])
          outslice = out[:, t].reshape([n, -1])
          testerrs.append(np.mean(np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)))

        return testerrs

  def forward(self, z, ts):
    z_shape = z.shape
    *leading_dims, N = z_shape
    T = ts.shape[0]

    z_expanded = z.unsqueeze(-2).expand(*leading_dims, T, N)

    t_shape = [1] * len(leading_dims) + [T, 1]
    t_expanded = ts.view(*t_shape).expand(*leading_dims, T, 1)

    result = torch.cat([z_expanded, t_expanded], dim=-1)

    decoded = self.recnet(result)
    return decoded

  def load_model(self, filename_prefix, verbose=False, min_epochs=0):
    search_path = f"savedmodels/ltinet/{filename_prefix}*.pickle"
    matching_files = glob.glob(search_path)

    print("Searching for model files matching prefix:", filename_prefix)
    if not hasattr(self, "metadata"):
        raise ValueError("Missing self.metadata. Cannot match models without metadata. Ensure model has been initialized with same config.")

    for addr in matching_files:
      try:
          with open(addr, "rb") as handle:
              dic = pickle.load(handle)
      except Exception as e:
          if verbose:
              print(f"Skipping {addr} due to read error: {e}")
          continue

      meta = dic.get("metadata", {})
      is_match = all(
          meta.get(k) == self.metadata.get(k)
          for k in self.metadata.keys()
      )

      # Check if model meets the minimum epoch requirement
      model_epochs = dic["epochs"]
      if model_epochs is None:
          if verbose:
              print(f"Skipping {addr} due to missing epoch metadata.")
          continue
      elif isinstance(model_epochs, list):  # handle legacy or list format
          if sum(model_epochs) < min_epochs:
              if verbose:
                  print(f"Skipping {addr} due to insufficient epochs ({sum(model_epochs)} < {min_epochs})")
              continue
      elif model_epochs < min_epochs:
          if verbose:
              print(f"Skipping {addr} due to insufficient epochs ({model_epochs} < {min_epochs})")
          continue

      if is_match:
          print("Model match found. Loading from:", addr)
          self.recnet.load_state_dict(dic["recnet"])
          self.epochs = model_epochs
          self.timetaken = dic["timetaken"]
          if "opt" in dic:     
            self.optparams = dic["opt"]

          return True
      elif verbose:
          print("Metadata mismatch in file:", addr)
          for k in self.metadata:
              print(f"{k}: saved={meta.get(k)} vs current={self.metadata.get(k)}")

    print("Load failed. No matching models found.")
    print("Searched:", matching_files)
    return False

  def train_model(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, best=True, verbose=False):
    def train_epoch(dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None, testparams=None):
      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []

      def closure(values, params):
        optimizer.zero_grad()

        out = self.propagate(params)
        target = values
        
        res = loss(out, target)
        res.backward()
        
        if writer is not None and self.trainstep % 5 == 0:
          writer.add_scalar("main/loss", res, global_step=self.trainstep)

        return res

      for values, params in dataloader:
        self.trainstep += 1
        error = optimizer.step(lambda: closure(values, params))
        losses.append(float(error.cpu().detach()))

      if scheduler is not None  and ep > epochs // 2:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testerr1, testerr2, testerrinf = self.get_errors(testarr, testparams, ords=(1, 2, np.inf))
        if scheduler is not None:
          print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative LTINet Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{ep+1}: Train Loss {error:.3e}, Relative LTINet Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1error", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2error", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInferror", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf
  
    loss = nn.MSELoss() if loss is None else loss()

    losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
    self.trainstep = 0

    train = torch.tensor(self.trainarr, dtype=torch.float32).to(self.device)
    params = torch.tensor(self.trainparams, dtype=torch.float32).to(self.device)
    test = self.testarr

    opt = optim(self.recnet.parameters(), lr=lr, weight_decay=ridge)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
    dataloader = DataLoader(torch.utils.data.TensorDataset(train, params), shuffle=False, batch_size=batch)

    if self.optparams is not None:
      opt.load_state_dict(self.optparams)

    writer = None
    if self.td is not None:
      name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
      writer = torch.utils.tensorboard.SummaryWriter(name)
      print("Tensorboard writer location is " + name)

    print("Number of NN trainable parameters", utils.num_params(self.recnet))
    print(f"Starting training LTINet model at {time.asctime()}...")
    print("train", train.shape, "test", test.shape)
    start = time.time()
      
    bestdict = { "loss": float(np.inf), "ep": 0 }
    for ep in range(epochs):
      lossesN, testerrors1N, testerrors2N, testerrorsinfN = train_epoch(dataloader, optimizer=opt, scheduler=scheduler, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test, testparams=self.testparams)
      losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

      if best and ep > epochs // 2:
        avgloss = np.mean(lossesN)
        if avgloss < bestdict["loss"]:
          bestdict["recnet"] = self.recnet.state_dict()
          bestdict["opt"] = opt.state_dict()
          bestdict["loss"] = avgloss
          bestdict["ep"] = ep
        elif verbose:
          print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")
      
    print(f"Finished training LTINet model at {time.asctime()}...")
    end = time.time()
    self.timetaken += end - start

    if best:
      self.recnet.load_state_dict(bestdict["recnet"])
      opt.load_state_dict(bestdict["opt"])

    self.optparams = opt.state_dict()
    self.epochs.append(epochs)

    if save:
      now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

      # Compute total training epochs
      total_epochs = sum(self.epochs) if isinstance(self.epochs, list) else self.epochs

      filename = (
          f"{self.dataset.name}_"
          f"{self.recclass.__name__}_"
          f"{self.recparams['seq']}_"
          f"{self.seed}_"
          f"{total_epochs}ep_"
          f"{now}.pickle"
      )

      dire = "savedmodels/ltinet"
      addr = os.path.join(dire, filename)

      if not os.path.exists(dire):
          os.makedirs(dire)

      with open(addr, "wb") as handle:
          pickle.dump({
              "recnet": self.recnet.state_dict(),
              "metadata": self.metadata,
              "opt": self.optparams,
              "epochs": self.epochs,
              "timetaken": self.timetaken
          }, handle, protocol=pickle.HIGHEST_PROTOCOL)

      print("Model saved at", addr)

    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

class ETIHelper():
  def __init__(self, config):
    self.update_config(config)

  def update_config(self, config):
    self.config = deepcopy(config)

  def create_etinet(self, dataset, k, config=None, **args):
    if config is None:
      config = self.config

    assert(len(dataset.data.shape) < 4)
    if len(dataset.data.shape) == 3:
      din = dataset.params.shape[-1]
      dout = dataset.data.shape[-1]

    td = args.get("td", None)
    seed = args.get("seed", 0)
    device = args.get("device", 0)

    recclass = globals()[args.get("recclass", config.recclass)]
    recparams = copy.deepcopy(dict(args.get("recparams", config.recparams)))

    aeclass = globals()[args.get("aeclass", config.aeclass)]
    aeparams = copy.deepcopy(dict(args.get("aeparams", config.aeparams)))

    recparams["seq"][0] = k + 1
    recparams["seq"][-1] = dout

    return ETINet(dataset, k, aeclass, aeparams, recclass, recparams, td=td, seed=seed, device=device)

  @staticmethod
  def get_operrs(etinet, times=None, testonly=False):
    if testonly:
      data = etinet.dataset.data[etinet.numtrain:,]
    else:
      data = etinet.dataset.data

    inputs = torch.tensor(data[:, 0]).to(etinet.device)

    encode = etinet.aenet.encode(inputs)
    errors = etinet.get_errors(encode, data[:, 1:], times=times, aggregate=False)

    return errors
  
  @staticmethod
  def plot_op_predicts(etinet, testonly=False, xs=None, cmap="viridis", topdown=True):
    if testonly:
      data = etinet.dataset.data[etinet.numtrain:,]
    else:
      data = etinet.dataset.data

    if xs == None:
      xs = np.linspace(0, 1, len(data[0, 0]))

    inputs = etinet.aenet.encode(torch.tensor(data[:, 0]).to(etinet.device))
    predicts = etinet.propagate(inputs).cpu().detach().numpy()

    errors = []
    n = predicts.shape[0]
    for s in range(data[:, 1:].shape[1]):
      currpredict = predicts[:, s].reshape((n, -1))
      currreference = data[:, s].reshape((n, -1))
      errors.append(np.mean(np.linalg.norm(currpredict - currreference, axis=1) / np.linalg.norm(currreference, axis=1)))
        
    print(f"Average Relative L2 Error over all times: {np.mean(errors):.4f}")

    if len(data.shape) == 3:
      if topdown:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
         
      else:
        fig, ax = plt.subplots(figsize=(4, 3))

    if topdown:
      @widgets.interact(i=(0, n-1))
      def plot_interact(i=0):
        if len(data.shape) == 3:
          axes[0].clear()
          axes[1].clear()
          axes[2].clear()

          fig.suptitle(f"RelL2 {np.linalg.norm(predicts[i, s-1] - data[i, s]) / np.linalg.norm(data[i, s])}")

          axes[0].set_title("Exact")
          axes[0].imshow(data[i, 1:], cmap=cmap, origin="lower", aspect='auto')

          axes[1].set_title("Predict")
          axes[1].imshow(predicts[i, 1:], cmap=cmap, origin="lower", aspect='auto')

          axes[2].set_title("|Exact - Predict|")
          axes[2].imshow(np.abs(data[i, 1:] - predicts[i]), cmap=cmap, origin="lower", aspect='auto')
       
    else:
      @widgets.interact(i=(0, n-1), s=(1, etinet.T-1))
      def plot_interact(i=0, s=1):
        print(f"Avg Relative L2 Error for t0 to t{s}: {errors[s-1]:.4f}")

        if len(data.shape) == 3:
          ax.clear()
          ax.set_title(f"RelL2 {np.linalg.norm(predicts[i, s-1] - data[i, s]) / np.linalg.norm(data[i, s])}")
          ax.plot(xs, data[i, 0], label="Input", linewidth=1)
          ax.plot(xs, predicts[i, s-1], label="Predicted", linewidth=1)
          ax.plot(xs, data[i, s], label="Exact", linewidth=1)
          ax.legend()

  @staticmethod
  def plot_errorparams(etinet, param=-1):
    if param == -1:
        # Auto-detect one varying parameter
        param = 0
        P = etinet.dataset.params.shape[1]
        for p in range(P):
            if np.abs(etinet.dataset.params[0, p] - etinet.dataset.params[1, p]) > 0:
                param = p
                break

    l2error = np.asarray(ETIHelper.get_operrs(etinet, times=[etinet.T - 1]))
    params = etinet.dataset.params

    print(params.shape, l2error.shape)

    if isinstance(param, (list, tuple, np.ndarray)) and len(param) == 2:
        # 3D scatter plot for 2 varying parameters
        x = params[:, param[0]]
        y = params[:, param[1]]
        z = l2error

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)

        ax.set_xlabel(f"Param {param[0]}")
        ax.set_ylabel(f"Param {param[1]}")
        ax.set_zlabel("Operator Error")
        fig.colorbar(sc, ax=ax, label="Operator Error")

    else:
      # Fallback to 2D scatter if param is 1D
      fig, ax = plt.subplots()
      ax.scatter(params[:, param], l2error, s=2)
      ax.set_xlabel(f"Parameter {param}")
      ax.set_ylabel("Operator Error")

    fig.tight_layout()

class ETINet():
  def __init__(self, dataset, k, aeclass, aeparams, recclass, recparams, td, seed, device):
    self.dataset = dataset
    self.device = device
    self.td = td
    self.k = k

    self.timetaken = 0
  
    if self.td is None:
      self.prefix = f"{self.dataset.name}{str(recclass.__name__)}ETINet"
    else:
      self.prefix = self.td

    torch.manual_seed(seed)
    np.random.seed(seed)
    self.seed = seed

    datacopy = self.dataset.data.copy()
    self.numtrain = int(datacopy.shape[0] * 0.9)
    
    self.T = self.dataset.data.shape[1]
    self.trainarr = datacopy[:self.numtrain]
    self.testarr = datacopy[self.numtrain:]
    self.optparams = None

    self.datadim = len(self.dataset.data.shape) - 2
    self.aestep = 0
    self.recstep = 0

    aeparams["encodeSeq"][0] = self.dataset.data.shape[-1]
    aeparams["encodeSeq"][-1] = self.k
    aeparams["decodeSeq"][0] = self.k
    aeparams["decodeSeq"][-1] = self.dataset.data.shape[-1]

    recparams["seq"][0] = self.k + 1
    recparams["seq"][-1] = self.dataset.data.shape[-1]

    self.aeclass = aeclass
    self.aeparams = copy.deepcopy(aeparams)
    self.recclass = recclass
    self.recparams = copy.deepcopy(recparams)

    self.aenet = aeclass(**aeparams).float().to(device)
    self.recnet = recclass(**recparams).float().to(device)

    self.metadata = {
      "aeclass": aeclass.__name__,
      "aeparams": aeparams,
      "recclass": recclass.__name__,
      "recparams": recparams,
      "dataset_name": dataset.name,
      "data_shape": list(dataset.data.shape),
      "data_checksum": float(np.sum(dataset.data)),
      "seed": seed,
    }

    self.epochs = []

  def reconstruct(self, z, ts):
    z_shape = z.shape
    *leading_dims, N = z_shape
    T = ts.shape[0]

    z_expanded = z.unsqueeze(-2).expand(*leading_dims, T, N)

    t_shape = [1] * len(leading_dims) + [T, 1]
    t_expanded = ts.view(*t_shape).expand(*leading_dims, T, 1)

    result = torch.cat([z_expanded, t_expanded], dim=-1)

    recon = self.recnet(result)
    return recon

  def propagate(self, code, start=1, end=-1):
    fullts = torch.linspace(0, 1, self.T).float().to(self.device)
    
    if end > 0:
      ts = fullts[start:end+1]
    else:
      ts = fullts[start:]

    out = self.reconstruct(code, ts)
    return out

  def get_errors(self, testarr, testrest, ords=(2,), times=None, aggregate=True):
    assert(aggregate or len(ords) == 1)
    
    if isinstance(testarr, np.ndarray):
      testarr = torch.tensor(testarr, dtype=torch.float32)

    if isinstance(testrest, np.ndarray):
      testrest = torch.tensor(testrest, dtype=torch.float32)

    if times is None:
      times = range(self.T-1)
  
    out = self.propagate(testarr)

    n = testarr.shape[0]
    orig = testrest.cpu().detach().numpy()
    out = out.cpu().detach().numpy()

    if aggregate:
      orig = orig.reshape([n, -1])
      out = out.reshape([n, -1])
      testerrs = []
      for o in ords:
        testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))

      return tuple(testerrs)
    
    else:
      o = ords[0]
      testerrs = []

      if len(times) == 1:
        t = times[0]
        origslice = orig[:, t-1].reshape([n, -1])
        outslice = out[:, t-1].reshape([n, -1])
        return np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)
      else:
        for t in range(orig.shape[1]):
          origslice = orig[:, t].reshape([n, -1])
          outslice = out[:, t].reshape([n, -1])
          testerrs.append(np.mean(np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)))

        return testerrs
      
  def get_ae_errors(self, testarr, ords=(2,)):
    if isinstance(testarr, np.ndarray):
      testarr = torch.tensor(testarr, dtype=torch.float32)
  
    out = self.aenet(testarr).cpu().detach().numpy()
    orig = testarr.cpu().detach().numpy()

    testerrs = []
    for o in ords:
      testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))

    return tuple(testerrs)

  def load_models(self, filename_prefix, verbose=False, min_epochs=0):
    search_path = f"savedmodels/etinet/{filename_prefix}*.pickle"
    matching_files = glob.glob(search_path)

    print("Searching for model files matching prefix:", filename_prefix)
    if not hasattr(self, "metadata"):
        raise ValueError("Missing self.metadata. Cannot match models without metadata. Ensure model has been initialized with same config.")

    for addr in matching_files:
      try:
          with open(addr, "rb") as handle:
              dic = pickle.load(handle)
      except Exception as e:
          if verbose:
              print(f"Skipping {addr} due to read error: {e}")
          continue

      meta = dic.get("metadata", {})
      is_match = all(
          meta.get(k) == self.metadata.get(k)
          for k in self.metadata.keys()
      )

      # Check if model meets the minimum epoch requirement
      model_epochs = dic["epochs"]
      if model_epochs is None:
          if verbose:
              print(f"Skipping {addr} due to missing epoch metadata.")
          continue
      elif isinstance(model_epochs, list):  # handle legacy or list format
          if sum(model_epochs) < min_epochs:
              if verbose:
                  print(f"Skipping {addr} due to insufficient epochs ({sum(model_epochs)} < {min_epochs})")
              continue
      elif model_epochs < min_epochs:
          if verbose:
              print(f"Skipping {addr} due to insufficient epochs ({model_epochs} < {min_epochs})")
          continue

      if is_match:
          print("Model match found. Loading from:", addr)
          self.recnet.load_state_dict(dic["recnet"])
          self.aenet.load_state_dict(dic["aenet"])
          self.epochs = model_epochs
          self.timetaken = dic["timetaken"]
          if "opt" in dic:     
            self.optparams = dic["opt"]

          return True
      elif verbose:
          print("Metadata mismatch in file:", addr)
          for k in self.metadata:
              print(f"{k}: saved={meta.get(k)} vs current={self.metadata.get(k)}")

    print("Load failed. No matching models found.")
    print("Searched:", matching_files)
    return False

  def train_recnet(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, best=True, verbose=False):
    def recnet_epoch(dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None, testrest=None):
      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []

      def closure(codes, rest):
        optimizer.zero_grad()

        out = self.propagate(codes)
        target = rest
        
        res = loss(out, target)
        res.backward()
        
        if writer is not None and self.recstep % 5 == 0:
          writer.add_scalar("main/loss", res, global_step=self.recstep)

        return res

      for codes, rest in dataloader:
        self.recstep += 1
        error = optimizer.step(lambda: closure(codes, rest))
        losses.append(float(error.cpu().detach()))

      if scheduler is not None  and ep > epochs // 2:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testerr1, testerr2, testerrinf = self.get_errors(testarr, testrest, ords=(1, 2, np.inf))
        if scheduler is not None:
          print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative ETINet Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{ep+1}: Train Loss {error:.3e}, Relative ETINet Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1error", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2error", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInferror", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf
  
    assert(self.aestep > 0)

    loss = nn.MSELoss() if loss is None else loss()

    losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []

    initial = torch.tensor(self.trainarr[:, 0], dtype=torch.float32).to(self.device)
    rest = torch.tensor(self.trainarr[:, 1:], dtype=torch.float32).to(self.device)
    train = self.aenet.encode(initial).detach()

    testinitial = torch.tensor(self.testarr[:, 0], dtype=torch.float32).to(self.device)
    testrest = torch.tensor(self.testarr[:, 1:], dtype=torch.float32).to(self.device)
    test = self.aenet.encode(testinitial).detach()

    opt = optim(self.recnet.parameters(), lr=lr, weight_decay=ridge)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
    dataloader = DataLoader(torch.utils.data.TensorDataset(train, rest), shuffle=False, batch_size=batch)

    writer = None
    if self.td is not None:
      name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
      writer = torch.utils.tensorboard.SummaryWriter(name)
      print("Tensorboard writer location is " + name)

    print("Number of NN trainable parameters", utils.num_params(self.recnet))
    print(f"Starting ETINet rec model at {time.asctime()}...")
    print("train", train.shape, "test", test.shape)
    start = time.time()

    bestdict = { "loss": float(np.inf), "ep": 0 }
    for ep in range(epochs):
      lossesN, testerrors1N, testerrors2N, testerrorsinfN = recnet_epoch(dataloader, optimizer=opt, scheduler=scheduler, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test, testrest=testrest)
      losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

      if best and ep > epochs // 2:
        avgloss = np.mean(lossesN)
        if avgloss < bestdict["loss"]:
          bestdict["recnet"] = self.recnet.state_dict()
          bestdict["opt"] = opt.state_dict()
          bestdict["loss"] = avgloss
          bestdict["ep"] = ep
        elif verbose:
          print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")
      
    print(f"Finished training ETINet rec model at {time.asctime()}...")
    end = time.time()
    self.timetaken += end - start

    if best:
      self.recnet.load_state_dict(bestdict["recnet"])
      opt.load_state_dict(bestdict["opt"])

    self.aeoptparams = opt.state_dict()
    self.epochs.append(epochs)

    if save:
      now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

      # Compute total training epochs
      total_epochs = sum(self.epochs) if isinstance(self.epochs, list) else self.epochs

      filename = (
          f"{self.dataset.name}_"
          f"{self.aeclass.__name__}_"
          f"{self.aeparams['encodeSeq']}_"
          f"{self.recclass.__name__}_"
          f"{self.recparams['seq']}_"
          f"{self.seed}_"
          f"{total_epochs}ep_"
          f"{now}.pickle"
      )

      dire = "savedmodels/etinet"
      addr = os.path.join(dire, filename)

      if not os.path.exists(dire):
          os.makedirs(dire)

      with open(addr, "wb") as handle:
          pickle.dump({
              "aenet": self.aenet.state_dict(),
              "recnet": self.recnet.state_dict(),
              "metadata": self.metadata,
              "opt": self.optparams,
              "epochs": self.epochs,
              "timetaken": self.timetaken
          }, handle, protocol=pickle.HIGHEST_PROTOCOL)

      print("Model saved at", addr)

    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

  def train_aenet(self, epochs, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, best=True, verbose=False):
    def aenet_epoch(dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []

      def closure(codes):
        optimizer.zero_grad()

        out = self.aenet(codes)
        target = codes
        
        res = loss(out, target)
        res.backward()
        
        if writer is not None and self.aestep % 5 == 0:
          writer.add_scalar("main/aeloss", res, global_step=self.aestep)

        return res

      for codes in dataloader:
        self.aestep += 1
        error = optimizer.step(lambda: closure(codes))
        losses.append(float(error.cpu().detach()))

      if scheduler is not None and ep > epochs // 2:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testerr1, testerr2, testerrinf = self.get_ae_errors(testarr, ords=(1, 2, np.inf))
        if scheduler is not None:
          print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative ETINet AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{ep+1}: Train Loss {error:.3e}, Relative ETINet AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1error", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2error", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInferror", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf
  
    loss = nn.MSELoss() if loss is None else loss()

    losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []

    initial = torch.tensor(self.trainarr[:, 0], dtype=torch.float32).to(self.device)
    test = torch.tensor(self.testarr[:, 0], dtype=torch.float32).to(self.device)

    opt = optim(self.aenet.parameters(), lr=lr, weight_decay=ridge)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=50, factor=0.9)
    dataloader = DataLoader(initial, shuffle=False, batch_size=batch)

    if self.optparams is not None:
      opt.load_state_dict(self.optparams)

    writer = None
    if self.td is not None:
      name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
      writer = torch.utils.tensorboard.SummaryWriter(name)
      print("Tensorboard writer location is " + name)

    print("Number of NN trainable parameters", utils.num_params(self.aenet))
    print(f"Starting training ETINet AE model at {time.asctime()}...")
    print("train", initial.shape, "test", test.shape)
    start = time.time()

    bestdict = { "loss": float(np.inf), "ep": 0 }
    for ep in range(epochs):
      lossesN, testerrors1N, testerrors2N, testerrorsinfN = aenet_epoch(dataloader, optimizer=opt, scheduler=scheduler, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
      losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

      if best and ep > epochs // 2:
        avgloss = np.mean(lossesN)
        if avgloss < bestdict["loss"]:
          bestdict["aenet"] = self.aenet.state_dict()
          bestdict["opt"] = opt.state_dict()
          bestdict["loss"] = avgloss
          bestdict["ep"] = ep
        elif verbose:
          print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")
      
    print(f"Finished training ETINet AE model at {time.asctime()}...")
    end = time.time()
    self.timetaken += end - start

    if best:
      self.aenet.load_state_dict(bestdict["aenet"])
      opt.load_state_dict(bestdict["opt"])

    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

class ELDHelper():
  def __init__(self, config):
    self.update_config(config)

  def update_config(self, config):
    self.config = deepcopy(config)

  def create_eldnet(self, dataset, f, k, config=None, **args):
    if config is None:
      config = self.config

    assert(len(dataset.data.shape) < 4)
    if len(dataset.data.shape) == 3:
      dout = dataset.data.shape[-1]

    td = args.get("td", None)
    seed = args.get("seed", 0)
    device = args.get("device", 0)

    aeclass = globals()[args.get("aeclass", config.aeclass)]
    aeparams = copy.deepcopy(dict(args.get("aeparams", config.aeparams)))
    dynclass = globals()[args.get("dynclass", config.dynclass)]
    dynparams = copy.deepcopy(dict(args.get("dynparams", config.dynparams)))
    decclass = globals()[args.get("decclass", config.decclass)]
    recparams = copy.deepcopy(dict(args.get("recparams", config.recparams)))

    dynparams["seq"][0] = k + f
    dynparams["seq"][-1] = k
    recparams["seq"][0] = k + f
    recparams["seq"][-1] = dout

    return ELDNet(dataset, f, k, aeclass, aeparams, dynclass, dynparams, decclass, recparams, td=td, seed=seed, device=device)

  @staticmethod
  def get_operrs(ldnet, times=None, testonly=False):
    if testonly:
      data = ldnet.dataset.data[ldnet.numtrain:,]
    else:
      data = ldnet.dataset.data

    inputs = torch.tensor(data).to(ldnet.device)
    dataparams = ldnet.aenet.encode(inputs[:, 0]).detach()
    errors = ldnet.get_errors(data, dataparams, times=times, aggregate=False)

    return errors
  
  @staticmethod
  def plot_op_predicts(ldnet, testonly=False, xs=None, cmap="viridis"):
    if testonly:
      data = ldnet.dataset.data[ldnet.numtrain:,]
    else:
      data = ldnet.dataset.data

    if xs == None:
      xs = np.linspace(0, 1, len(data[0, 0]))

    pinput = torch.tensor(np.float32(data[:, 0])).to(ldnet.device)
    params = ldnet.aenet.encode(pinput).detach()

    predicts = ldnet.propagate(params).cpu().detach()

    errors = []
    n = predicts.shape[0]
    for s in range(data.shape[1]):
      currpredict = predicts[:, s-1].reshape((n, -1))
      currreference = data[:, s].reshape((n, -1))
      errors.append(np.mean(np.linalg.norm(currpredict - currreference, axis=1) / np.linalg.norm(currreference, axis=1)))
        
    print(f"Average Relative L2 Error over all times: {np.mean(errors):.4f}")

    if len(data.shape) == 3:
      fig, ax = plt.subplots(figsize=(4, 3))

    @widgets.interact(i=(0, n-1), s=(1, ldnet.T-1))
    def plot_interact(i=0, s=1):
      print(f"Avg Relative L2 Error for t0 to t{s}: {errors[s-1]:.4f}")

      if len(data.shape) == 3:
        ax.clear()
        ax.set_title(f"RelL2 {np.linalg.norm(predicts[i, s-1] - data[i, s]) / np.linalg.norm(data[i, s])}")
        ax.plot(xs, data[i, 0], label="Input", linewidth=1)
        ax.plot(xs, predicts[i, s-1], label="Predicted", linewidth=1)
        ax.plot(xs, data[i, s], label="Exact", linewidth=1)
        ax.legend()
        
  @staticmethod
  def plot_errorparams(ldnet, param=-1):
    if param == -1:
        # Auto-detect one varying parameter
        param = 0
        P = ldnet.dataset.params.shape[1]
        for p in range(P):
            if np.abs(ldnet.dataset.params[0, p] - ldnet.dataset.params[1, p]) > 0:
                param = p
                break

    l2error = np.asarray(LDHelper.get_operrs(ldnet, times=[ldnet.T - 1]))
    params = ldnet.dataset.params

    print(params.shape, l2error.shape)

    if isinstance(param, (list, tuple, np.ndarray)) and len(param) == 2:
        # 3D scatter plot for 2 varying parameters
        x = params[:, param[0]]
        y = params[:, param[1]]
        z = l2error

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)

        ax.set_xlabel(f"Param {param[0]}")
        ax.set_ylabel(f"Param {param[1]}")
        ax.set_zlabel("Operator Error")
        fig.colorbar(sc, ax=ax, label="Operator Error")

    else:
        # Fallback to 2D scatter if param is 1D
        fig, ax = plt.subplots()
        ax.scatter(params[:, param], l2error, s=2)
        ax.set_xlabel(f"Parameter {param}")
        ax.set_ylabel("Operator Error")

    fig.tight_layout()

class ELDNet():
  def __init__(self, dataset, f, k, aeclass, aeparams, dynclass, dynparams, decclass, recparams, td, seed, device):
    self.dataset = dataset
    self.device = device
    self.td = td
    self.k = k
    self.f = f

    self.timetaken = 0
  
    if self.td is None:
      self.prefix = f"{self.dataset.name}{str(dynclass.__name__)}ELDNet-{self.k+self.f}"
    else:
      self.prefix = self.td

    torch.manual_seed(seed)
    np.random.seed(seed)
    self.seed = seed

    datacopy = self.dataset.data.copy()
    self.numtrain = int(datacopy.shape[0] * 0.9)
    
    self.T = self.dataset.data.shape[1]
    self.trainarr = datacopy[:self.numtrain]
    self.testarr = datacopy[self.numtrain:]
    self.optparams = None

    self.datadim = len(self.dataset.data.shape) - 2

    self.aeclass = aeclass
    self.aeparams = copy.deepcopy(aeparams)
    self.dynclass = dynclass
    self.dynparams = copy.deepcopy(dynparams)
    self.decclass = decclass
    self.recparams = copy.deepcopy(recparams)

    aeparams["encodeSeq"][0] = self.dataset.data.shape[-1]
    aeparams["encodeSeq"][-1] = self.f
    aeparams["decodeSeq"][0] = self.f
    aeparams["decodeSeq"][-1] = self.dataset.data.shape[-1]

    dynparams["seq"][0] = self.k + self.f
    dynparams["seq"][-1] = self.k
    recparams["seq"][0] = self.k + self.f
    recparams["seq"][-1] = self.dataset.data.shape[-1]

    self.aenet = aeclass(**aeparams).float().to(device)
    self.dynnet = dynclass(**dynparams).float().to(device)
    self.recnet = decclass(**recparams).float().to(device)

    self.metadata = {
      "aeclass": aeclass.__name__,
      "aeparams": aeparams,
      "dynclass": dynclass.__name__,
      "dynparams": dynparams,
      "decclass": decclass.__name__,
      "recparams": recparams,
      "dataset_name": dataset.name,
      "data_shape": list(dataset.data.shape),
      "data_checksum": float(np.sum(dataset.data)),
      "seed": seed,
    }

    self.epochs = []

  def propagate(self, code, start=0, end=-1, returncodes=False):
    if end == -1:
      end = self.T - 1

    z = code

    # get first decode
    if z.shape[-1] != self.f + self.k:
      if z.shape[-1] == self.f:
        z_fixed = z
        z_dynamic = torch.zeros(list(z_fixed.shape[:-1]) + [self.k])
        z = torch.cat([z_fixed, z_dynamic], dim=-1)
      else:
        print(z.shape)
        assert(False)

    zpreds = [z]
    for t in range(start, end):
      z = self.forward(z)
      zpreds.append(z)

    zpreds = torch.stack(zpreds, dim=1)
    upreds = self.recnet(zpreds)

    if returncodes:
      return upreds, zpreds
    else:
      return upreds
     
  def get_ae_errors(self, testarr, ords=(2,)):
    if isinstance(testarr, np.ndarray):
      testarr = torch.tensor(testarr, dtype=torch.float32)
  
    out = self.aenet(testarr).cpu().detach().numpy()
    orig = testarr.cpu().detach().numpy()

    testerrs = []
    for o in ords:
      testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))

    return tuple(testerrs)

  def get_errors(self, testarr, testparams, ords=(2,), times=None, aggregate=True):
    assert(aggregate or len(ords) == 1)
    
    if isinstance(testarr, np.ndarray):
      testarr = torch.tensor(testarr, dtype=torch.float32)

    if isinstance(testparams, np.ndarray):
      testparams = torch.tensor(testparams, dtype=torch.float32)

    if times is None:
      times = range(self.T-1)
  
    out = self.propagate(testparams)

    n = testarr.shape[0]
    orig = testarr.cpu().detach().numpy()
    out = out.cpu().detach().numpy()

    if aggregate:
      orig = orig.reshape([n, -1])
      out = out.reshape([n, -1])
      testerrs = []
      for o in ords:
        testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))

      return tuple(testerrs)
    
    else:
      o = ords[0]
      testerrs = []

      if len(times) == 1:
        t = times[0]
        origslice = orig[:, t].reshape([n, -1])
        outslice = out[:, t].reshape([n, -1])
        return np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)
      else:
        for t in range(orig.shape[1]):
          origslice = orig[:, t].reshape([n, -1])
          outslice = out[:, t].reshape([n, -1])
          testerrs.append(np.mean(np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)))

        return testerrs

  def forward(self, z_full, decode=False):
    if z_full.shape[-1] == self.f + self.k:
      z_fixed = z_full[..., :self.f]
      z_dynamic = z_full[..., self.f:]

    else:
      if z_full.shape[-1] == self.f:
        z_fixed = z_full
        z_dynamic = torch.zeros(list(z_fixed.shape[:-1]) + [self.k])
        z_full = torch.cat([z_fixed, z_dynamic], dim=-1)
      else:
        print(z_full.shape)
        assert(False)

    deltaz = self.dynnet(z_full)
    z_next_dynamic = z_dynamic + deltaz
    z_next_full = torch.cat([z_fixed, z_next_dynamic], dim=-1)

    if decode:
      decoded = self.recnet(z_next_full)
      return decoded, z_next_full
    
    else:
      return z_next_full

  def load_models(self, filename_prefix, verbose=False, min_epochs=0):
    search_path = f"savedmodels/eldnet/{filename_prefix}*.pickle"
    matching_files = glob.glob(search_path)

    print("Searching for model files matching prefix:", filename_prefix)
    if not hasattr(self, "metadata"):
        raise ValueError("Missing self.metadata. Cannot match models without metadata. Ensure model has been initialized with same config.")

    for addr in matching_files:
      try:
          with open(addr, "rb") as handle:
              dic = pickle.load(handle)
      except Exception as e:
          if verbose:
              print(f"Skipping {addr} due to read error: {e}")
          continue

      meta = dic.get("metadata", {})
      is_match = all(
          meta.get(k) == self.metadata.get(k)
          for k in self.metadata.keys()
      )

      # Check if model meets the minimum epoch requirement
      model_epochs = dic["epochs"]
      if model_epochs is None:
          if verbose:
              print(f"Skipping {addr} due to missing epoch metadata.")
          continue
      elif isinstance(model_epochs, list):  # handle legacy or list format
          if sum(model_epochs) < min_epochs:
              if verbose:
                  print(f"Skipping {addr} due to insufficient epochs ({sum(model_epochs)} < {min_epochs})")
              continue
      elif model_epochs < min_epochs:
          if verbose:
              print(f"Skipping {addr} due to insufficient epochs ({model_epochs} < {min_epochs})")
          continue

      if is_match:
          print("Model match found. Loading from:", addr)
          self.aenet.load_state_dict(dic["aenet"])
          self.dynnet.load_state_dict(dic["dynnet"])
          self.recnet.load_state_dict(dic["recnet"])
          self.epochs = model_epochs
          self.timetaken = dic["timetaken"]
          if "opt" in dic:     
            self.optparams = dic["opt"]

          return True
      elif verbose:
          print("Metadata mismatch in file:", addr)
          for k in self.metadata:
              print(f"{k}: saved={meta.get(k)} vs current={self.metadata.get(k)}")

    print("Load failed. No matching models found.")
    print("Searched:", matching_files)
    return False

  def train_aenet(self, epochs, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, best=True, verbose=False):
    def aenet_epoch(dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []

      def closure(codes):
        optimizer.zero_grad()

        out = self.aenet(codes)
        target = codes
        
        res = loss(out, target)
        res.backward()
        
        if writer is not None and self.aestep % 5 == 0:
          writer.add_scalar("main/aeloss", res, global_step=self.aestep)

        return res

      for codes in dataloader:
        self.aestep += 1
        error = optimizer.step(lambda: closure(codes))
        losses.append(float(error.cpu().detach()))

      if scheduler is not None and ep > epochs // 2:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testerr1, testerr2, testerrinf = self.get_ae_errors(testarr, ords=(1, 2, np.inf))
        if scheduler is not None:
          print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative ELDNet AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{ep+1}: Train Loss {error:.3e}, Relative ELDNet AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1error", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2error", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInferror", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf
  
    self.aestep = 0
    loss = nn.MSELoss() if loss is None else loss()

    losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []

    initial = torch.tensor(self.trainarr[:, 0], dtype=torch.float32).to(self.device)
    test = torch.tensor(self.testarr[:, 0], dtype=torch.float32).to(self.device)

    opt = optim(self.aenet.parameters(), lr=lr, weight_decay=ridge)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=50, factor=0.9)
    dataloader = DataLoader(initial, shuffle=False, batch_size=batch)

    if self.optparams is not None:
      opt.load_state_dict(self.optparams)

    writer = None
    if self.td is not None:
      name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
      writer = torch.utils.tensorboard.SummaryWriter(name)
      print("Tensorboard writer location is " + name)

    print("Number of NN trainable parameters", utils.num_params(self.aenet))
    print(f"Starting training ELDNet AE model at {time.asctime()}...")
    print("train", initial.shape, "test", test.shape)
    start = time.time()

    bestdict = { "loss": float(np.inf), "ep": 0 }
    for ep in range(epochs):
      lossesN, testerrors1N, testerrors2N, testerrorsinfN = aenet_epoch(dataloader, optimizer=opt, scheduler=scheduler, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
      losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

      if best and ep > epochs // 2:
        avgloss = np.mean(lossesN)
        if avgloss < bestdict["loss"]:
          bestdict["aenet"] = self.aenet.state_dict()
          bestdict["opt"] = opt.state_dict()
          bestdict["loss"] = avgloss
          bestdict["ep"] = ep
        elif verbose:
          print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")
      
    print(f"Finished training ELDNet AE model at {time.asctime()}...")
    end = time.time()
    self.timetaken += end - start

    if best:
      self.aenet.load_state_dict(bestdict["aenet"])
      opt.load_state_dict(bestdict["opt"])

    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

  def train_ldnet(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, best=True, verbose=False):
    def train_epoch(dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None, testparams=None):
      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []

      def closure(values, params):
        optimizer.zero_grad()

        out = self.propagate(params)
        target = values
        
        res = loss(out, target)
        res.backward()
        
        if writer is not None and self.trainstep % 5 == 0:
          writer.add_scalar("main/loss", res, global_step=self.trainstep)

        return res

      for values, params in dataloader:
        self.trainstep += 1
        error = optimizer.step(lambda: closure(values, params))
        losses.append(float(error.cpu().detach()))

      if scheduler is not None and ep > epochs // 2:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testerr1, testerr2, testerrinf = self.get_errors(testarr, testparams, ords=(1, 2, np.inf))
        if scheduler is not None:
          print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative ELDNet Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{ep+1}: Train Loss {error:.3e}, Relative ELDNet Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1error", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2error", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInferror", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf
  
    loss = nn.MSELoss() if loss is None else loss()

    losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
    self.trainstep = 0

    train = torch.tensor(self.trainarr, dtype=torch.float32).to(self.device)
    params = self.aenet.encode(train[:, 0]).detach()

    test = torch.tensor(self.testarr, dtype=torch.float32).to(self.device)
    testparams = self.aenet.encode(test[:, 0]).detach()

    opt = optim(itertools.chain(self.dynnet.parameters(), self.recnet.parameters()), lr=lr, weight_decay=ridge)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
    dataloader = DataLoader(torch.utils.data.TensorDataset(train, params), shuffle=False, batch_size=batch)

    if self.optparams is not None:
      opt.load_state_dict(self.optparams)

    writer = None
    if self.td is not None:
      name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
      writer = torch.utils.tensorboard.SummaryWriter(name)
      print("Tensorboard writer location is " + name)

    print("Number of NN trainable parameters", utils.num_params(self.dynnet), "+", utils.num_params(self.recnet))
    print(f"Starting training ELDNet model at {time.asctime()}...")
    print("train", train.shape, "test", test.shape)
    start = time.time()

    bestdict = { "loss": float(np.inf), "ep": 0 }
    for ep in range(epochs):
      lossesN, testerrors1N, testerrors2N, testerrorsinfN = train_epoch(dataloader, optimizer=opt, scheduler=scheduler, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test, testparams=testparams)
      losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

      if best and ep > epochs // 2:
        avgloss = np.mean(lossesN)
        if avgloss < bestdict["loss"]:
          bestdict["dynnet"] = self.dynnet.state_dict()
          bestdict["recnet"] = self.recnet.state_dict()
          bestdict["opt"] = opt.state_dict()
          bestdict["loss"] = avgloss
          bestdict["ep"] = ep
        elif verbose:
          print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")
      
    print(f"Finished training ELDNet model at {time.asctime()}...")
    end = time.time()
    self.timetaken += end - start

    if best:
      self.dynnet.load_state_dict(bestdict["dynnet"])
      self.recnet.load_state_dict(bestdict["recnet"])
      opt.load_state_dict(bestdict["opt"])

    self.optparams = opt.state_dict()
    self.epochs.append(epochs)

    if save:
      now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

      # Compute total training epochs
      total_epochs = sum(self.epochs) if isinstance(self.epochs, list) else self.epochs

      filename = (
          f"{self.dataset.name}_"
          f"{self.dynclass.__name__}_"
          f"{self.dynparams['seq']}_"
          f"{self.decclass.__name__}_"
          f"{self.recparams['seq']}_"
          f"{self.seed}_"
          f"{total_epochs}ep_"
          f"{now}.pickle"
      )

      dire = "savedmodels/eldnet"
      addr = os.path.join(dire, filename)

      if not os.path.exists(dire):
          os.makedirs(dire)

      with open(addr, "wb") as handle:
          pickle.dump({
              "aenet": self.aenet.state_dict(),
              "dynnet": self.dynnet.state_dict(),
              "recnet": self.recnet.state_dict(),
              "metadata": self.metadata,
              "opt": self.optparams,
              "epochs": self.epochs,
              "timetaken": self.timetaken
          }, handle, protocol=pickle.HIGHEST_PROTOCOL)

      print("Model saved at", addr)

    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

class HighDimProp():
  def __init__(self, dataset, propclass, propseq, activation, autonomous=True, useparams=False, td=None, seed=0, residual=True, device=0):
    self.dataset = dataset
    self.device = device
    self.td = td
    self.useparams = useparams
    self.residual = residual
    self.autonomous = autonomous
  
    if self.td is None:
      self.prefix = f"HighDimProp{self.dataset.name}{str(propclass.__name__)}"
    else:
      self.prefix = self.td

    torch.manual_seed(seed)
    np.random.seed(seed)
    self.seed = seed
    self.timetaken = 0

    datacopy = self.dataset.data.copy()
    self.numtrain = int(datacopy.shape[0] * 0.9)
    
    self.T = self.dataset.data.shape[1]
    self.trainarr = datacopy[:self.numtrain]
    self.testarr = datacopy[self.numtrain:]

    if self.useparams:
      T = self.trainarr.shape[1] 
      
      paramtrain = np.repeat(self.dataset.params[:self.numtrain, None, :], T, axis=1)
      paramtest  = np.repeat(self.dataset.params[self.numtrain:, None, :], T, axis=1) 

      self.trainarr = np.concatenate([self.trainarr, paramtrain], axis=-1)
      self.testarr = np.concatenate([self.testarr, paramtest], axis=-1)

    self.propclass = propclass
    self.propseq = propseq
    self.optparams = None

    self.datadim = len(self.dataset.data.shape) - 2
    if self.useparams:
      propseq[0] += dataset.params.shape[1]

    self.prop = self.propclass(propseq, activation).to(self.device)
    self.propinfo = [propseq, activation.__name__ if hasattr(activation, '__name__') else str(activation), self.residual, self.useparams, self.autonomous]

    #self.tidata = [ticlass, tiinfo]
    #self.datadata = [np.floor(np.sum(self.dataset.data) * 100), self.dataset.data.shape]

    self.metadata = {
      "model_class": propclass.__name__,
      "propinfo": self.propinfo,
      "dataset_name": dataset.name,
      "data_shape": list(dataset.data.shape),
      "data_checksum": float(np.sum(dataset.data)),
      "seed": seed,
      "useparams": self.useparams
    }

    self.epochs = []

  def get_errors(self, testarr, ords=(2,), t=None, aggregate=True):
    assert(aggregate or len(ords) == 1)
    
    if isinstance(testarr, np.ndarray):
      testarr = torch.tensor(testarr, dtype=torch.float32)

    if t is None:
      t = self.T - 1
  
    out = torch.stack(self.propagate(testarr[:, 0], t), axis=1)
    
    n = testarr.shape[0]
    orig = testarr[:, 1:].cpu().detach().numpy()
    out = out.cpu().detach().numpy()

    if self.useparams:
      orig = orig[..., :-self.dataset.params.shape[1]]

    if aggregate:
      orig = orig.reshape([n, -1])
      out = out.reshape([n, -1])
      testerrs = []
      for o in ords:
        testerrs.append(np.mean(np.linalg.norm(orig - out, axis=1, ord=o) / np.linalg.norm(orig, axis=1, ord=o)))

      return tuple(testerrs)
    
    else:
      o = ords[0]
      testerrs = []

      if t == 1:
        origslice = orig[:, t].reshape([n, -1])
        outslice = out.reshape([n, -1])
        return np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)
      else:
        for tt in range(out.shape[1]):
          origslice = orig[:, tt].reshape([n, -1])
          outslice = out[:, tt].reshape([n, -1])
          testerrs.append(np.mean(np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)))

        return testerrs

  def load_model(self, filename_prefix, verbose=False, min_epochs=0):
    search_path = f"savedmodels/highdimprop/{filename_prefix}*.pickle"
    matching_files = glob.glob(search_path)

    print("Searching for model files matching prefix:", filename_prefix)
    if not hasattr(self, "metadata"):
        raise ValueError("Missing self.metadata. Cannot match models without metadata. Ensure model has been initialized with same config.")

    for addr in matching_files:
      try:
          with open(addr, "rb") as handle:
              dic = pickle.load(handle)
      except Exception as e:
          if verbose:
              print(f"Skipping {addr} due to read error: {e}")
          continue

      meta = dic.get("metadata", {})
      is_match = all(
          meta.get(k) == self.metadata.get(k)
          for k in meta.keys()
      )

      # Check if model meets the minimum epoch requirement
      model_epochs = dic["epochs"]
      if model_epochs is None:
          if verbose:
              print(f"Skipping {addr} due to missing epoch metadata.")
          continue
      elif isinstance(model_epochs, list):  # handle legacy or list format
          if sum(model_epochs) < min_epochs:
              if verbose:
                  print(f"Skipping {addr} due to insufficient epochs ({sum(model_epochs)} < {min_epochs})")
              continue
      elif model_epochs < min_epochs:
          if verbose:
              print(f"Skipping {addr} due to insufficient epochs ({model_epochs} < {min_epochs})")
          continue

      if is_match:
          print("Model match found. Loading from:", addr)
          self.prop.load_state_dict(dic["model"])
          self.epochs = model_epochs
          self.timetaken = dic["timetaken"]
          if "opt" in dic:     
            self.optparams = dic["opt"]

          return True
      elif verbose:
          print("Metadata mismatch in file:", addr)
          for k in self.metadata:
              print(f"{k}: saved={meta.get(k)} vs current={self.metadata.get(k)}")

    print("Load failed. No matching models found.")
    print("Searched:", matching_files)
    return False

  def train_model(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, printinterval=10, batch=32, ridge=0, loss=None, accumulateprop=False, best=True, verbose=False):
    def train_epoch(dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []

      def closure(batch):
        optimizer.zero_grad()
        
        if accumulateprop:
          outprop = self.propagate(batch[:, 0], self.T-1)
          propped = torch.stack(outprop, axis=1)

        else:
          propped = self.prop(batch[:, :-1])

        if self.useparams:
          batch = batch[..., :-self.dataset.params.shape[1]]
          
        res = loss(propped, batch[:, 1:])
        res.backward()
        
        if writer is not None and self.trainstep % 5 == 0:
          writer.add_scalar("main/loss", res, global_step=self.trainstep)

        return res

      for batch in dataloader:
        self.trainstep += 1
        error = optimizer.step(lambda: closure(batch))
        losses.append(float(error.cpu().detach()))

      if scheduler is not None and ep > epochs // 2:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testerr1, testerr2, testerrinf = self.get_errors(testarr, ords=(1, 2, np.inf))
        if scheduler is not None:
          print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative HDP Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{ep+1}: Train Loss {error:.3e}, Relative HDP Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1error", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2error", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInferror", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf
  
    loss = nn.MSELoss() if loss is None else loss()

    losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
    self.trainstep = 0

    train = torch.tensor(self.trainarr, dtype=torch.float32).to(self.device)
    test = self.testarr  

    opt = optim(self.prop.parameters(), lr=lr, weight_decay=ridge)
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.3)
    dataloader = DataLoader(train, shuffle=False, batch_size=batch)

    if self.optparams is not None:
      opt.load_state_dict(self.optparams)

    writer = None
    if self.td is not None:
      name = f"./tensorboard/{datetime.datetime.now().strftime('%d-%B-%Y')}/{self.td}/{datetime.datetime.now().strftime('%H.%M.%S')}/"
      writer = torch.utils.tensorboard.SummaryWriter(name)
      print("Tensorboard writer location is " + name)

    print("Number of NN trainable parameters", utils.num_params(self.prop))
    print(f"Starting training HighDimProp model {self.metadata['model_class']} at {time.asctime()}...")
    print("train", train.shape, "test", test.shape)
      
    start = time.time()
    bestdict = { "loss": float(np.inf), "ep": 0 }
    for ep in range(epochs):
      lossesN, testerrors1N, testerrors2N, testerrorsinfN = train_epoch(dataloader, optimizer=opt, scheduler=scheduler, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
      losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

      if best and ep > epochs // 2:
        avgloss = np.mean(lossesN)
        if avgloss < bestdict["loss"]:
          bestdict["model"] = self.prop.state_dict()
          bestdict["opt"] = opt.state_dict()
          bestdict["loss"] = avgloss
          bestdict["ep"] = ep
        elif verbose:
          print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")
      
    end = time.time()
    self.timetaken += end - start
    print(f"Finished training HighDimProp model {self.metadata['model_class']} at {time.asctime()}...")

    if best:
      self.prop.load_state_dict(bestdict["model"])
      opt.load_state_dict(bestdict["opt"])

    self.optparams = opt.state_dict()
    self.epochs.append(epochs)

    if save:
      now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      readable_shape = "x".join(map(str, self.propseq))

      # Compute total training epochs
      total_epochs = sum(self.epochs) if isinstance(self.epochs, list) else self.epochs

      filename = (
          f"{self.dataset.name}_"
          f"{self.propclass.__name__}_"
          f"{self.metadata['propinfo'][1]}_"
          f"{readable_shape}_"
          f"{self.seed}_"
          f"{total_epochs}ep_"
          f"{now}.pickle"
      )

      dire = "savedmodels/highdimprop"
      addr = os.path.join(dire, filename)

      if not os.path.exists(dire):
          os.makedirs(dire)

      with open(addr, "wb") as handle:
          pickle.dump({
              "model": self.prop.state_dict(),
              "metadata": self.metadata,
              "opt": self.optparams,
              "epochs": self.epochs,
              "timetaken": self.timetaken
          }, handle, protocol=pickle.HIGHEST_PROTOCOL)

      print("Model saved at", addr)

    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

  def propagate(self, arr, steps, t=0):
    assert(t + steps < self.T)

    codes = torch.tensor(arr).to(self.device, dtype=torch.float32)

    codeslist = []
    for step in range(steps):
      tcurr = t + 1 + step

      if self.autonomous:
        codeinput = codes
      else:
        ttensor = torch.tensor(np.repeat((tcurr - 1), codes.shape[0])).unsqueeze(1).to(self.device).float()
        codeinput = torch.cat((codes, ttensor), dim=1)

      codes = self.prop_forward(self.prop, codeinput)

      codeslist.append(codes)

    return codeslist
  
  def prop_forward(self, prop, batch):
    batchbase = batch
    if self.useparams:
      params = batch[..., -self.dataset.params.shape[1]:]
      batchbase = batch[..., :-self.dataset.params.shape[1]]

    out = prop.forward(batch)

    if self.residual:
      out = out + batchbase

    if self.useparams:
      out = torch.cat([out, params], dim=-1)

    return out

class HighDimPropHelper():
  def __init__(self, config):
    self.update_config(config)

  def update_config(self, config):
    self.config = deepcopy(config)

  def create_propnet(self, dataset, config=None, **args):
    if config is None:
      config = self.config

    assert(len(dataset.data.shape) < 4)
    if len(dataset.data.shape) == 3:
      din = dataset.data.shape[-1]

    td = args.get("td", None)
    seed = args.get("seed", 0)
    device = args.get("device", 0)
    autonomous = args.get("autonomous", True)
    useparams = args.get("useparams", False)
    residual = args.get("residual", False)

    propclass = globals()[args.get("propclass", config.propclass)]
    propseq = copy.deepcopy(args.get("propseq", config.propseq))
    activation = args.get("activation", config.activation)

    propseq[0] = din
    propseq[-1] = din

    return HighDimProp(dataset, propclass, propseq, activation, td=td, seed=seed, device=device, autonomous=autonomous, useparams=useparams, residual=residual)

  @staticmethod
  def get_operrs(propnet, times=None, testonly=False):
    if testonly:
      data = propnet.testarr
    else:
      data = np.concatenate((propnet.trainarr, propnet.testarr), axis=0)
    
    errors = propnet.get_errors(data, aggregate=False)

    return errors
  
  @staticmethod
  def plot_op_predicts(propnet, testonly=False, xs=None, cmap="viridis"):
    if testonly:
      data = propnet.dataset.data[propnet.numtrain:,]
    else:
      data = propnet.dataset.data

    if xs == None:
      xs = np.linspace(0, 1, len(data[0, 0]))

    datas = torch.tensor(np.float32(data)).to(propnet.device)

    predicts = torch.stack(propnet.propagate(datas[:, 0], propnet.T-1), axis=1).cpu().detach()

    errors = []
    n = predicts.shape[0]
    for s in range(predicts.shape[1]):
      currpredict = predicts[:, s].reshape((n, -1))
      currreference = data[:, s+1].reshape((n, -1))
      errors.append(np.mean(np.linalg.norm(currpredict - currreference, axis=1) / np.linalg.norm(currreference, axis=1)))
        
    print(f"Average Relative L2 Error over all times: {np.mean(errors):.4f}")

    if len(data.shape) == 3:
      fig, ax = plt.subplots(figsize=(4, 3))

    @widgets.interact(i=(0, n-1), s=(1, propnet.T-1))
    def plot_interact(i=0, s=1):
      print(f"Avg Relative L2 Error for t0 to t{s}: {errors[s-1]:.4f}")

      if len(data.shape) == 3:
        ax.clear()
        ax.set_title(f"RelL2 {np.linalg.norm(predicts[i, s-1] - data[i, s]) / np.linalg.norm(data[i, s])}")
        ax.plot(xs, data[i, 0], label="Input", linewidth=1)
        ax.plot(xs, predicts[i, s-1], label="Predicted", linewidth=1)
        ax.plot(xs, data[i, s], label="Exact", linewidth=1)
        ax.legend()
        
  @staticmethod
  def plot_errorparams(ldnet, param=-1):
    if param == -1:
        # Auto-detect one varying parameter
        param = 0
        P = ldnet.dataset.params.shape[1]
        for p in range(P):
            if np.abs(ldnet.dataset.params[0, p] - ldnet.dataset.params[1, p]) > 0:
                param = p
                break

    l2error = np.asarray(HighDimPropHelper.get_operrs(ldnet, times=[ldnet.T - 1]))
    params = ldnet.dataset.params[:l2error.shape[0]]


    if isinstance(param, (list, tuple, np.ndarray)) and len(param) == 2:
        # 3D scatter plot for 2 varying parameters
        x = params[:, param[0]]
        y = params[:, param[1]]
        z = l2error

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)

        ax.set_xlabel(f"Param {param[0]}")
        ax.set_ylabel(f"Param {param[1]}")
        ax.set_zlabel("Operator Error")
        fig.colorbar(sc, ax=ax, label="Operator Error")

    else:
        # Fallback to 2D scatter if param is 1D
        fig, ax = plt.subplots()
        ax.scatter(params[:, param], l2error, s=2)
        ax.set_xlabel(f"Parameter {param}")
        ax.set_ylabel("Operator Error")

    fig.tight_layout()

# abstract class
class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x):
        raise NotImplementedError()

    def decode(self, v):
        raise NotImplementedError()
    
    def right(self, v):
        raise NotImplementedError()

    def forward(self, x):
        v = self.encode(x)
        out = self.right(v)
        return out
    
    def project(self, x):
        v = self.encode(x)
        out = self.decode(v)
        return out
    
    def setup_data(self, dataset, trainnum, batch_size, t0=0, t1=1, loss=nn.MSELoss()):
        raise NotImplementedError

    def setup_trainloader(self, trainnum, batch_size, t0=0, t1=-1):
        raise NotImplementedError

    def setup_misc(self, loss=nn.MSELoss()):
        self.loss = loss

    def setup_models(self, **args):
        raise NotImplementedError
    
    def train_encoder(self, epochs, **args):
        raise NotImplementedError
    
    def plot_encoding(self, name, ep):
        assert(self.writer)
        
        half = int(self.testarray.shape[1] / 2)
        input = torch.tensor(self.testarray[:, :half]).to(self.device)

        test = self.encode(input)
        points = test.cpu().detach().numpy()

        for i in range(self.dataset.params.shape[1]):
            fig = plt.figure(figsize=(16, 4))
            fig.suptitle(f"Param {i+1}")

            n = points.shape[0]

            if points.shape[1] == 1:
                ax = fig.add_subplot()
                sns.kdeplot(points, fill=True, ax=ax)
                ax.set_ylabel("Probability")
            elif points.shape[1] == 2:
                ax = fig.add_subplot()
                sc = ax.scatter(points[:, 0], points[:, 1], c=self.dataset.params[-n:, i])
                colorbar = plt.colorbar(sc, ax=ax)
            else:
                m = points.shape[1]
                combinations_2 = list(combinations(range(m), 2))
                num = len(combinations_2)

                for idx, (col1, col2) in enumerate(combinations_2, start=1):
                    if idx > 6:
                        continue

                    ax = fig.add_subplot(1, min(6, num), idx)
                    ax.scatter(points[:, col1], points[:, col2], c=self.dataset.params[-n:, i])
                    ax.set_xlabel(f'{col1}')
                    ax.set_ylabel(f'{col2}')
                    ax.set_title(f'Scatter Plot {col1} vs {col2}')

            self.writer.add_figure(f'test/{name}-{i}', fig, global_step=ep)
            self.writer.flush()

        plt.close('all')

    def train_right(self):
        raise NotImplementedError() 

    def get_projerr(self, ord=2):
        half = int(self.testarray.shape[1] // 2)

        domain = torch.tensor(self.testarray[:, :half]).to(self.device)
        projected = self.project(domain).cpu().detach().numpy()
        domain = domain.cpu().detach().numpy()
        return np.mean(np.linalg.norm(projected - domain, axis=1, ord=ord) / np.linalg.norm(domain, axis=1, ord=ord))

    def get_operr(self, arr=None, ord=2):
        if arr is None:
          arr = self.testarray

        half = int(arr.shape[1] // 2)
        
        domain = torch.tensor(arr[:, :half]).to(self.device)
        rangee = arr[:, half:]
        operator = self.forward(domain).cpu().detach().numpy()
        return np.mean(np.linalg.norm(operator - rangee, axis=1, ord=ord) / np.linalg.norm(rangee, axis=1, ord=ord))
    
    def get_generr(self, arr=None):
        if arr is None:
          arr = self.testarray

        half = int(arr.shape[1] // 2)
        
        domain = torch.tensor(arr[:, :half]).to(next(self.parameters()).get_device())
        rangee = arr[:, half:]
        operator = self.forward(domain).cpu().detach().numpy()
        return np.mean((np.linalg.norm(operator - rangee, axis=1)) ** 2) / (arr.shape[1] - 1) # divide by the coefficient  


# # below is part of weldnet class
# def finetune_ae_with_prop(self, epochs, save=True, lr=1e-4, batch=32, ridge=0, printinterval=10, loss=None):
#     def finetune_epoch(ae, prop, dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
#       def closure(batch):
#         optimizer.zero_grad()

#         assert(self.straightness + self.kinetic == 0)
        
#         x = batch[:, :-1, :].to(self.device)  # [B, T-1, D]
#         y = batch[:, 1:, :].to(self.device)   # [B, T-1, D]

#         z = ae.encode(x)                     # Encode
#         with torch.no_grad():
#             z_next = prop(z)                # Propagate (frozen)
#         y_hat = ae.decode(z_next)           # Decode

#         res = loss(y_hat, y)
#         lossval.backward()
#         opt.step()

#         losses.append(float(lossval.detach().cpu()))
#         scheduler.step(np.mean(losses))
 
#         if writer is not None and self.aestep % 5:
#           writer.add_scalar("main/loss", float(res.cpu().detach()), global_step=self.aestep)

#         return res
      
#       losses = []
#       testerrors1 = []
#       testerrors2 = []
#       testerrorsinf = []

#       device = self.device

#       for batch in dataloader:
#         self.aestep += 1
#         error = optimizer.step(lambda: closure(batch))
#         losses.append(float(error.cpu().detach()))

#       if scheduler is not None:
#         scheduler.step(np.mean(losses))

#       # print test
#       if printinterval > 0 and (ep % printinterval == 0):
#         testerr1, testerr2, testerrinf = self.get_proj_errors(model, testarr, ords=(1, 2, np.inf))
#         if scheduler is not None:
#           print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
#         else:
#           print(f"{ep+1}: Train Loss {error:.3e}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

#         if writer is not None:
#             writer.add_scalar("misc/relativeL1proj", testerr1, global_step=ep)
#             writer.add_scalar("main/relativeL2proj", testerr2, global_step=ep)
#             writer.add_scalar("misc/relativeLInfproj", testerrinf, global_step=ep)

#       return losses, testerrors1, testerrors2, testerrorsinf

#     if loss is None:
#       loss = nn.MSELoss()
#     else:
#       loss = loss()

#     print(f"Fine-tuning {self.W} AEs using fixed propagators")

#     for w in range(self.W):
#       self.aestep = 0
#       ae = self.aes[w]
#       prop = self.props[w]

#       train = torch.tensor(self.alltrain[:, self.windowvals[w], :], dtype=torch.float32).to(self.device)
#       test = torch.tensor(self.alltest[:, self.windowvals[w], :], dtype=torch.float32).to(self.device)

#       dataloader = DataLoader(train, batch_size=batch, shuffle=True)
#       opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=ridge)
#       scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)

#       print(f"Fine-tuning AE {w+1}/{self.W}...")

#       for ep in range(epochs):
#         lossesN, testerrors1N, testerrors2N, testerrorsinfN = finetune_epoch(ae, prop, dataloader, scheduler=scheduler, optimizer=opt, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
#         losses += lossesN; testerrors1 += testerrors1N; testerrors2 += testerrors2N; testerrorsinf += testerrorsinfN

#         if best and ep > epochs // 2:
#           avgloss = np.mean(lossesN)
#           if avgloss < bestdict["loss"]:
#             bestdict["model"] = ae.state_dict()
#             bestdict["opt"] = opt.state_dict()
#             bestdict["loss"] = avgloss
#             bestdict["ep"] = ep
#           elif verbose:
#             print(f"Loss not improved at epoch {ep} (Ratio: {avgloss/bestdict['loss']:.2f}) from {bestdict['ep']} (Loss: {bestdict['loss']:.2e})")

#         if ep % 5 == 0 and plottb:
#           WeldHelper.plot_encoding_window(self, w, encoding_param, step=self.aestep, writer=writer, tensorboard=True)
      
#       print(f"Finished fine-tuning AE {w+1}/{self.W}")
#       losses_all.append(losses)
#       testerrors1_all.append(testerrors1)
#       testerrors2_all.append(testerrors2)
#       testerrorsinf_all.append(testerrorsinf)

#       if best:
#         ae.load_state_dict(bestdict["model"])
#         opt.load_state_dict(bestdict["opt"])

#     self.aedata.append((epochs))

#     if save and False:
#       pass

#     print("Finished finetuning all timewindows")
#     return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf}
