# Setup code

import time
import glob
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

from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from copy import deepcopy

import utils
from omegaconf import DictConfig, OmegaConf

plt.rcParams["figure.figsize"] = (7, 3)

BASEDIR = "savedmodels/ae"

from common.config import create_object, load_config
from autoencoder.networks import FCAutoEncoder, ConvAutoEncoder, ConvAutoEncoder2D, ResnetAutoEncoder, LSTMAutoEncoder, ConvLSTMAutoEncoder
JC_MODULES = [FCAutoEncoder, ConvAutoEncoder, ConvAutoEncoder2D, ResnetAutoEncoder, LSTMAutoEncoder, ConvLSTMAutoEncoder]

def get_activation(activation):
  if activation.lower() == "relu":
    return nn.ReLU()
  else:
    return nn.ReLU()

def get_proj_errors(model, testarr, ords=(2,)):
  if isinstance(testarr, np.ndarray):
    testarr = torch.tensor(testarr, dtype=torch.float32)
    
  proj = model(testarr)
  
  if len(testarr.shape) > 3:
    assert(len(testarr.shape) == 4)
    testarr = testarr.reshape(list(testarr.shape[:-2]) + [-1])
    proj = proj.reshape(list(proj.shape[:-2]) + [-1])

  n = testarr.shape[0]
  testarr = testarr.cpu().detach().numpy().reshape([n, -1])
  proj = proj.cpu().detach().numpy().reshape([n, -1])

  testerrs = []
  for o in ords:
    testerro = np.mean(np.linalg.norm(testarr - proj, axis=1, ord=o) / np.linalg.norm(testarr, axis=1, ord=o))
    testerrs.append(testerro)

  return tuple(testerrs)

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
    kld = kld / x.size(0)

    return recon_loss + self.reg * kld

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
    return self.aes[w].encode(tensor)

class TimeInputModel():
  def __init__(self, dataset, ticlass, tiinfo, activation, td, seed, device):
    self.dataset = dataset
    self.device = device
    self.td = td
  
    if self.td is None:
      self.prefix = f"{self.dataset.name}{str(ticlass.__name__)}"
    else:
      self.prefix = self.td

    torch.manual_seed(seed)
    np.random.seed(seed)
    self.seed = seed

    datacopy = self.dataset.data.copy()
    self.numtrain = int(datacopy.shape[0] * 0.8)
    
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
      "epochs": []
    }

  def get_ti_errors(self, testarr, ords=(2,), times=None, aggregate=True):
    assert(aggregate or len(ords) == 1)
    
    if isinstance(testarr, np.ndarray):
      testarr = torch.tensor(testarr, dtype=torch.float32)

    if times is None:
      times = range(1, self.T)
  
    out = self.forward(testarr[:, 0], times)
    
    n = testarr.shape[0]
    orig = testarr[:, 1:].cpu().detach().numpy()
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
      
      xts = [torch.cat((x, t * torch.ones((x.shape[0], 1))), dim=1) for t in ts]
      xt = torch.stack(xts, dim=1)
      out = self.model(xt)

      return out.reshape(list(out.shape)[:-1] + origshape)
        
    elif isinstance(self.model, DeepONet):
      spaces = torch.linspace(0, 1, x.shape[1]).reshape([-1, 1])

      inputlist = [torch.cat((spaces, t * torch.ones((spaces.shape[0], 1))), dim=1) for t in ts]
      inputs = torch.stack(inputlist, dim=1)
      out = self.model(x, inputs)
      return out

    elif isinstance(self.model, FNO1d):
      return self.model(x, ts)
  
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
          for k in ["model_class", "tiinfo", "activation", "dataset_name", "data_shape", "seed"]
      )

      # Check if model meets the minimum epoch requirement
      model_epochs = meta.get("epochs")
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
          self.metadata["epochs"] = meta.get("epochs")
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
        
        out = self.forward(batch[:, 0], range(1, self.T))

        res = loss(batch[:, 1:] - out, torch.zeros_like(out))
        res.backward()
        
        if writer is not None and self.trainstep % 5:
          writer.add_scalar("main/loss", res, global_step=self.trainstep)

        return res

      for batch in dataloader:
        self.trainstep += 1
        error = optimizer.step(lambda: closure(batch))
        losses.append(float(error.cpu().detach()))

      if scheduler is not None:
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
    scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=30)
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
      
    print(f"Finished training TI model {self.metadata['model_class']} at {time.asctime()}...")

    if best:
      self.model.load_state_dict(bestdict["model"])
      opt.load_state_dict(bestdict["opt"])

    self.optparams = opt.state_dict()
    self.metadata["epochs"].append(epochs)

    if save:
      now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      readable_shape = "x".join(map(str, self.metadata["tiinfo"]))

      # Compute total training epochs
      total_epochs = sum(self.metadata["epochs"]) if isinstance(self.metadata["epochs"], list) else self.metadata["epochs"]

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
              "opt": self.optparams
          }, handle, protocol=pickle.HIGHEST_PROTOCOL)

      print("Model saved at", addr)

    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

class WeldNet(WindowTrajectory):
  def __init__(self, dataset, windows, aeclass, aeparams, propclass, propparams, transclass, transparams, straightness=0, td=None, seed=0, device=0, kinetic=0, decodedprop=False, accumulateprop=False, autonomous=True):    
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
    self.accumulateprop = accumulateprop
    self.decodedprop = decodedprop

    if not self.autonomous:
        propparams["seq"] = list(propparams["seq"])
        propparams["seq"][0] = propparams["seq"][-1] + 1

    assert(self.straightness == 0 or self.kinetic == 0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    self.seed = seed

    datacopy = self.dataset.data.copy()
    self.numtrain = int(datacopy.shape[0] * 0.8)
    
    self.T = self.dataset.data.shape[1]
    self.W = windows
    
    self.aes = []
    self.props = []
    self.trains = []
    self.tests = []

    self.alltrain = datacopy[:self.numtrain]
    self.alltest = datacopy[self.numtrain:]

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

    total = self.T + self.W - 1
    M = total // self.W
    remainder = total - self.W * M
    
    left = [list(range(k*(M-1), (k+1)*(M-1) + 1)) for k in range(self.W - remainder)]
    start = left[-1][-1]
    right = [list(range(start + k*(M), start + (k+1)*(M) + 1)) for k in range(remainder)]
    self.windowvals = left + right

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
      "epochs": [],
    }
  
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
    for step in range(steps):
      tcurr = t + 1 + step

      if fixedw:
        wcurr = w
      else:
        wcurr = self.find_window(tcurr)

      if wcurr != wprev:
        codes = self.transcode(tcurr-1, codes)

      if self.autonomous:
        codeinput = codes
      else:
        ttensor = torch.tensor(np.repeat((tcurr*0 - 1), codes.shape[0])).unsqueeze(1).to(self.device).float()
        codeinput = torch.cat((codes, ttensor), dim=1)

      if isinstance(self.props[wcurr], nn.Module):
        modelout = self.props[wcurr].forward(codeinput)
      else:
        if torch.is_tensor(codes):
          codes = codes.detach().cpu().numpy()

        ttensor = np.full((codes.shape[0], 1), tcurr - 1, dtype=np.float32) 
        codeinput = np.concatenate([codes, ttensor], axis=1)
        modelout = self.props[w].predict(codeinput)

      if self.residualprop:
        codes = codes + modelout
      else:
        codes = modelout

      wprev = wcurr
      codeslist.append(torch.tensor(codes, dtype=torch.float32, device=self.device))

    return codeslist

  def load_all(self, filename, verbose=False):
    assert(False)
    matching_files = glob.glob(f"savedmodels/weld/props/{filename}*{self.W}w-props-*")

    matchedfile = False
    for addr in matching_files:
        with open(addr, "rb") as handle:
            dic = pickle.load(handle)
            matchedfile = True

            for i in range(len(self.propdata)):
              if type(self.propdata[i]) == type({}):
                if str(self.propdata[i].values()) != str(dic["propdata"][i].values()):
                  matchedfile = False
                  if verbose:
                    print("NO MATCH", self.propdata[i].values(), dic["propdata"][i].values())

              elif str(self.propdata[i]) != str(dic["propdata"][i]):
                matchedfile = False
                if verbose:
                  print("NO MATCH", self.propdata[i], dic["propdata"][i])

            for i in range(len(self.aedata)):
              if type(self.aedata[i]) == type({}):
                if str(self.aedata[i].values()) != str(dic["aedata"][i].values()):
                  matchedfile = False
                  if verbose:
                    print("NO MATCH", self.aedata[i].values(), dic["aedata"][i].values())

              elif str(self.aedata[i]) != str(dic["aedata"][i]):
                matchedfile = False
                if verbose:
                  print("NO MATCH", self.aedata[i], dic["aedata"][i])

            if matchedfile:   
              print("Loading aes and props from", addr)
              self.props = dic["props"]
              self.aes = dic["aes"]
              break
            elif verbose:
              print("---")

    if matchedfile:
      if self.transcoderdata:
        return self.load_transcoders(filename, verbose=verbose)
      else:
        return True
    
    else:   
      print(f"Propagator failed. Could not match with any files")
      print(matching_files)
      return False

  def load_aes(self, filename, verbose=False):
    assert(False)
    matching_files = glob.glob(f"savedmodels/weld/{filename}*{self.W}w-*")
    print("Searching for", str(self.aedata), str(self.datadata))

    for addr in matching_files:
      with open(addr, "rb") as handle:
        dic = pickle.load(handle)

        if str(self.aedata) == str(dic["aedata"]) and str(self.datadata) == str(dic["datadata"]):
          print("Loading AEs from", addr)
          self.aes = dic["aes"]
          return True
        elif verbose:
          print("NO MATCH", str(dic["aedata"]), str(dic["datadata"]))
            
    print(f"Load failed. Could not match with any files")
    print(matching_files)
    return False
  
  def load_aes_and_props(self, filename, verbose=False):
    assert(False)
    matching_files = glob.glob(f"savedmodels/weld/props/{filename}*{self.W}w-props-*")

    matchedfile = False
    for addr in matching_files:
        with open(addr, "rb") as handle:
            dic = pickle.load(handle)
            matchedfile = True

            for i in range(len(self.propdata)):
              if type(self.propdata[i]) == type({}):
                if str(self.propdata[i].values()) != str(dic["propdata"][i].values()):
                  matchedfile = False
                  if verbose:
                    print("NO MATCH", self.propdata[i].values(), dic["propdata"][i].values())

              elif str(self.propdata[i]) != str(dic["propdata"][i]):
                matchedfile = False
                if verbose:
                  print("NO MATCH", self.propdata[i], dic["propdata"][i])

            for i in range(len(self.aedata)):
              if type(self.aedata[i]) == type({}):
                if str(self.aedata[i].values()) != str(dic["aedata"][i].values()):
                  matchedfile = False
                  if verbose:
                    print("NO MATCH", self.aedata[i].values(), dic["aedata"][i].values())

              elif str(self.aedata[i]) != str(dic["aedata"][i]):
                matchedfile = False
                if verbose:
                  print("NO MATCH", self.aedata[i], dic["aedata"][i])

            if matchedfile:   
              print("Loading aes and props from", addr)
              self.props = dic["props"]
              self.aes = dic["aes"]
              break
            elif verbose:
              print("---")

    if matchedfile:
      return True
    else:   
      print(f"Propagator failed. Could not match with any files")
      print(matching_files)
      return False
    
  def load_transcoders(self, filename, verbose=False):
    assert(False)
    assert self.transcoderdata is not None
    
    matching_files = glob.glob(f"savedmodels/weld/trans/{filename}*{self.W}w-*")
    print("Searching for", str(self.transcoderdata), str(self.datadata))

    for addr in matching_files:
      with open(addr, "rb") as handle:
        dic = pickle.load(handle)

        if str(self.transcoderdata) == str(dic["transdata"]) and str(self.datadata) == str(dic["datadata"]):
          print("Loading transcoders from", addr)
          self.transcoders = dic["trans"]
          return True
        elif verbose:
          print("NO MATCH", str(dic["transdata"]), str(dic["datadata"]))
            
    print(f"Load failed. Could not match with any files")
    print(matching_files)
    return False

  def train_aes(self, epochs_first, warmstart_epochs=0, save=True, onlydecoder=False, optim=torch.optim.AdamW, lr=1e-4, plottb=False, gridbatch=None, printinterval=10, batch=32, ridge=0, loss=None, encoding_param=-1, best=True, verbose=False):    
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

          if isinstance(model, FFVAE):
            recon, mu, logvar = model(batch, variance=True)
            res = model.loss_function(recon, batch, mu, logvar)
            res.backward()
              
          else:
            enc = model.encode(batch)

            if isinstance(model, GIAutoencoder) or isinstance(model, TCAutoencoder):
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
              enc = model.encode(batch)
              proj = model.decode(enc)

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

        if scheduler is not None:
          scheduler.step(np.mean(losses))

        # print test
        if printinterval > 0 and (ep % printinterval == 0):
          testerr1, testerr2, testerrinf = get_proj_errors(model, testarr, ords=(1, 2, np.inf))
          if scheduler is not None:
            print(f"{ep+1}: Train Loss {error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
          else:
            print(f"{ep+1}: Train Loss {error:.3e}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

          
          if writer is not None:
              writer.add_scalar("misc/relativeL1proj", testerr1, global_step=ep)
              writer.add_scalar("main/relativeL2proj", testerr2, global_step=ep)
              writer.add_scalar("misc/relativeLInfproj", testerrinf, global_step=ep)

        return losses, testerrors1, testerrors2, testerrorsinf

      loss = nn.MSELoss() if loss is None else loss()
      encoding_param = determine_param(self.dataset, encoding_param)

      losses_all, testerrors1_all, testerrors2_all, testerrorsinf_all = [], [], [], []

      print(f"Training {self.W} WeldNet AEs")
      self.trains = []
      self.tests = []
      for w in range(self.W):
        if len(self.aes) <= w:
          self.aes.append(self.aeclass(**self.aeparams) if self.aeclass not in JC_MODULES else aeclass(aeparams.copy()))

        ae = self.aes[w]

        losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
        bestdict = { "loss": float(np.inf), "ep": 0 }

        self.aestep = 0
        epochs = epochs_first
        train = torch.tensor(self.alltrain[:, self.windowvals[w], :], dtype=torch.float32)
        test = self.alltest[:, self.windowvals[w], :] 

        if isinstance(ae, PCAAutoencoder):
          ae.train_pca(train.cpu().numpy())
          testerr1, testerr2, testerrinf = get_proj_errors(ae, test, ords=(1, 2, np.inf))
          print(f"Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

          continue

        self.trains.append(train)
        self.tests.append(test)

        if onlydecoder:
          trainparams = ae.decoder.parameters()
        else:
          trainparams = ae.parameters()

        opt = optim(trainparams, lr=lr, weight_decay=ridge)
        scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20)
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

      print("Finished training all timewindows")
      return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

  def train_aes_plus_props(self, epochs, lamb=0.1, save=True, optim=torch.optim.AdamW, lr=1e-4, plottb=False, gridbatch=None, printinterval=10, batch=32, ridge=0, loss=None, encoding_param=-1, best=True, verbose=False):    
    def both_epoch(model, modelprop, dataloader, writer=None, optimizer=None, scheduler=None, ep=0, printinterval=10, loss=None, testarr=None):
      losses = []
      testerrors1 = []
      testerrors2 = []
      testerrorsinf = []

      device = self.device

      def closure(batch):
        optimizer.zero_grad()

        assert(self.straightness + self.kinetic == 0)

        enc = model.encode(batch)

        if isinstance(model, GIAutoencoder) or isinstance(model, TCAutoencoder):
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
          enc = model.encode(batch)
          proj = model.decode(enc)

        self.res = loss(batch, proj)

        starts = enc[:, :-1]
        exacts = enc[:, 1:]

        predicted = modelprop(starts) # to do
        if self.residualprop:
          predicted = starts + predicted

        self.error = loss(predicted, exacts)

        totalloss = self.res + lamb * self.error

        totalloss.backward()
        
        num = totalloss.cpu().detach()
        if writer is not None and self.aestep % 5:
          
          writer.add_scalar("main/loss", float(num), global_step=self.aestep)
          #writer.add_scalar("main/penalty", penalties, global_step=self.aestep)

        return totalloss

      for batch in dataloader:
        self.aestep += 1
        lossout = optimizer.step(lambda: closure(batch))
        losses.append(float(lossout.cpu().detach()))

      if scheduler is not None:
        scheduler.step(np.mean(losses))

      # print test
      if printinterval > 0 and (ep % printinterval == 0):
        testerr1, testerr2, testerrinf = get_proj_errors(model, testarr, ords=(1, 2, np.inf))

        if scheduler is not None:
          print(f"{ep+1}: Train Loss {self.res:.3e} + {self.error:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")
        else:
          print(f"{ep+1}: Train Loss {self.res:.3e} + {self.error:.3e},, Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        if writer is not None:
            writer.add_scalar("misc/relativeL1proj", testerr1, global_step=ep)
            writer.add_scalar("main/relativeL2proj", testerr2, global_step=ep)
            writer.add_scalar("misc/relativeLInfproj", testerrinf, global_step=ep)

      return losses, testerrors1, testerrors2, testerrorsinf

    loss = nn.MSELoss() if loss is None else loss()
    encoding_param = determine_param(self.dataset, encoding_param)

    losses_all, testerrors1_all, testerrors2_all, testerrorsinf_all = [], [], [], []

    self.metadata["trainedtogether"] = True
    print(f"Training {self.W} WeldNet AEs and props together")
    self.trains = []
    self.tests = []
    for w in range(self.W):
      if len(self.aes) <= w:
        self.aes.append(self.aeclass(**self.aeparams) if self.aeclass not in JC_MODULES else aeclass(aeparams.copy()))
      if len(self.props) <= w:
        self.props.append(self.propclass(**self.propparams) if self.propclass not in JC_MODULES else propclass(propparams.copy()))

      ae = self.aes[w]
      prop = self.props[w]

      losses, testerrors1, testerrors2, testerrorsinf = [], [], [], []
      bestdict = { "loss": float(np.inf), "ep": 0 }

      self.aestep = 0
      train = torch.tensor(self.alltrain[:, self.windowvals[w], :], dtype=torch.float32)
      test = self.alltest[:, self.windowvals[w], :] 

      if isinstance(ae, PCAAutoencoder):
        ae.train_pca(train.cpu().numpy())
        testerr1, testerr2, testerrinf = get_proj_errors(ae, test, ords=(1, 2, np.inf))
        print(f"Relative AE Error (1, 2, inf): {testerr1:3f}, {testerr2:3f}, {testerrinf:3f}")

        continue

      self.trains.append(train)
      self.tests.append(test)

      opt = optim( list(ae.parameters()) + list(prop.parameters()), lr=lr, weight_decay=ridge)
      scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=20)
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
          lossesN, testerrors1N, testerrors2N, testerrorsinfN = both_epoch(ae, prop, dataloader, scheduler=scheduler, optimizer=opt, writer=writer, ep=ep, printinterval=printinterval, loss=loss, testarr=test)
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

    self.aeepochs.append(epochs)

    if save and False:
      dire = "savedmodels/weld"
      addr = f"{dire}/{self.prefix}{self.W}w-{datetime.datetime.now().strftime('%d-%B-%Y-%H.%M')}.pickle"

      if not os.path.exists(dire):
        os.makedirs(dire)

      with open(addr, "wb") as handle:
        pickle.dump({"aes": self.aes, "aedata": self.aedata, "datadata": self.datadata}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("AEs saved at", addr)

    print("Finished training all timewindows")
    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf } 

  def train_aes_alternate_props(self, epochs, lamb=0.1, save=True, optim=torch.optim.AdamW, lr=1e-4, plottb=False, gridbatch=None, printinterval=10, batch=32, ridge=0, loss=None, encoding_param=-1, best=True, verbose=False):
    def train_epoch_ae(ep):
      ae.train()
      running = []
      for batch_idx, batch in enumerate(dataloader):
        opt_ae.zero_grad()
        enc = ae.encode(batch)

        if gridbatch:
          full = ae.domaingrid
          idx  = torch.randperm(full.shape[0], device=full.device)[:gridbatch]
          recon = ae.decode(enc, grid=full[idx])
        else:
          recon = ae.decode(enc)

        loss_ae = recon_loss_fn(batch, recon)
        loss_ae.backward()
        opt_ae.step()
        running.append(loss_ae.item())

        if writer and (batch_idx + ep*len(dataloader)) % 5 == 0:
          writer.add_scalar("AE/loss", loss_ae.item(), ep*len(dataloader)+batch_idx)

      scheduler_ae.step(np.mean(running))
      return np.mean(running)

    def train_epoch_prop(ep):
      ae.eval()
      prop.train()
      running = []
      for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad(): 
          enc = ae.encode(batch)

        starts = enc[:, :-1]
        targets = enc[:, 1:]
        opt_prop.zero_grad()
        pred = prop(starts)
        if self.residualprop: pred = starts + pred
        loss_p = prop_loss_fn(pred, targets)
        loss_p.backward()
        opt_prop.step()
        running.append(loss_p.item())
        if writer and (batch_idx + ep*len(dataloader)) % 5 == 0:
          writer.add_scalar("Prop/loss", loss_p.item(), ep*len(dataloader)+batch_idx)

        scheduler_prop.step(np.mean(running))
        return np.mean(running)
 
    recon_loss_fn = loss() if loss is not None else nn.MSELoss()
    prop_loss_fn  = nn.MSELoss()
    encoding_param = determine_param(self.dataset, encoding_param)
    all_losses   = {'ae': [], 'prop': []}
    all_testerrs = []

    for w in range(self.W):
      if len(self.aes) <= w:
        self.aes.append(self.aeclass(**self.aeparams))
      if len(self.props) <= w:
        self.props.append(self.propclass(**self.propparams))

      ae   = self.aes[w].to(self.device)
      prop = self.props[w].to(self.device)
      train = torch.tensor(self.alltrain[:, self.windowvals[w], :], dtype=torch.float32, device=self.device)
      test  = self.alltest[:, self.windowvals[w], :]

      if isinstance(ae, PCAAutoencoder):
        ae.train_pca(train.cpu().numpy())
        e1,e2,ei = get_proj_errors(ae, test, ords=(1,2,np.inf))

        print("Trained PCA for window ", w+1)
        print(f"Relative AE Error (1, 2, inf): {e1:.3f}, {e2:.3f}, {ei:.3f}")
        continue

      opt_ae  = optim(ae.parameters(), lr=lr, weight_decay=ridge)
      opt_prop = optim(prop.parameters(), lr=lr / 5, weight_decay=ridge)
      scheduler_ae = lr_scheduler.ReduceLROnPlateau(opt_ae, patience=20)
      scheduler_prop = lr_scheduler.ReduceLROnPlateau(opt_prop, patience=20)
      dataloader = DataLoader(train, batch_size=batch)

      writer = None
      if self.td:
        tb_dir = f"./tensorboard/{datetime.datetime.now():%Y%m%d}/{self.td}-w{w}"
        writer = SummaryWriter(tb_dir)

      best = {'loss': float('inf'), 'state': None}
      ae_losses, prop_losses = [], []

      for ep in range(epochs):
        la = train_epoch_ae(ep)
        lp = train_epoch_prop(ep)

        losses_w.append(la)
        props_w.append(lp)
      
        if ep % printinterval == 0:
          t1,t2,ti = get_proj_errors(ae, test, ords=(1,2,np.inf))
          if scheduler:
            print(f"{ep+1}: Train Loss {la:.3e} + {lp:.3e}, LR {scheduler.get_last_lr()[-1]:.3e}, Relative AE Error (1, 2, inf): {t1:.3f}, {t2:.3f}, {ti:.3f}")
          else:
            print(f"{ep+1}: Train Loss {la:.3e} + {lp:.3e},, Relative AE Error (1, 2, inf): {t1:.3f}, {t2:.3f}, {ti:.3f}")

        if best and ep>epochs//2:
          if np.mean([la,lp]) < best['loss']:
            best.update({'loss': np.mean([la,lp]), 'model': ae.state_dict(), 'opt': opt_ae.state_dict(), 'ep': ep})
          elif verbose:
            print(f"Loss not improved at epoch {ep} (Ratio: {np.mean([la,lp])/best['loss']:.2f}) from {best['ep']} (Loss: {best['loss']:.2e})")
        
        print(f"Finish training AE and Prop {w} at {time.asctime()}.")

        if best: 
          ae.load_state_dict(best['model'])
          opt_ae.load_state_dict(best['opt'])

        all_losses['ae'].append(ae_losses)
        all_losses['prop'].append(prop_losses)
        all_testerrs.append(testerrs)

    if save:
      torch.save({'aes': [ae.state_dict() for ae in self.aes], 'props': [p.state_dict() for p in self.props]}, f"{self.prefix}_alttrain.pth")

    return all_losses, all_testerrs

  def train_transcoders(self, epochs, save=True, optim=torch.optim.AdamW, lr=1e-4, verbose=False, propagated_trans=False, printinterval=10, batch=32, ridge=0, loss=None, encoding_param=-1, best=True):
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

        predict = model.forward(x)

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

      if scheduler is not None:
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

    print(f"Training {self.W} WeldNet Transcoders")
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
      scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=30)

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

    self.transepoch.append(epochs)
    print("Finished training all timewindow transcoders")
    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf }

  def train_transcoders_krr(self, save=True, ridge=1.0, kernel="rbf", gamma=None, encoding_param=-1, printinterval=1):
    encoding_param = determine_param(self.dataset, encoding_param)

    testerrors1_all = []
    testerrors2_all = []
    testerrorsinf_all = []

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

        if self.accumulateprop:
          x0 = x[:, :1]

          xlist = []
          for i in range(x.shape[1]):
            if self.residualprop:
              x0 = x0 + model(x0)
            else:
              x0 = model(x0)

            xlist.append(x0)

          predict = torch.cat(xlist, dim=1)

        else:
          predict = model(xt)

          if self.residualprop:
            predict = x + predict

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

      if scheduler is not None:
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
    for w in range(self.W):
      if len(self.props) <= w:
        self.props.append(self.propclass(**self.propparams) if self.propclass not in JC_MODULES else propclass(propclass.copy()))

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
      scheduler = lr_scheduler.ReduceLROnPlateau(opt, patience=30)
                                      
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

    self.propepochs.append(epochs)
    print("Finished training all propagators")

    return { "losses": losses, "testerrors1": testerrors1, "testerrors2": testerrors2, "testerrorsinf": testerrorsinf}

  def load_models(self, filename_prefix, verbose=False, min_epochs=0):
    search_path = f"savedmodels/weld/{filename_prefix}*.pickle"
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

          self.metadata["epochs"] = meta.get("epochs")

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
            "metadata": self.metadata
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Models saved at", addr)    

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

    return TimeInputModel(dataset, ticlass, tiinfo, activation, td=td, seed=seed, device=device)

  @staticmethod
  def get_operrs(ti, times=None, testonly=False):
    if testonly:
      data = ti.dataset.data[ti.numtrain:,]
    else:
      data = ti.dataset.data

    if isinstance(ti, TimeInputModel):
      errors = ti.get_ti_errors(data, times=times, aggregate=False)
    else:
      errors = WeldHelper.get_operrs(ti, testonly=testonly)
    return errors
  
  @staticmethod
  def plot_op_predicts(ti: TimeInputModel, testonly=False, xs=None, cmap="viridis"):
    if testonly:
      data = ti.dataset.data[ti.numtrain:,]
    else:
      data = ti.dataset.data

    if xs == None:
      xs = np.linspace(0, 1, len(data[0, 0]))

    data = torch.tensor(np.float32(data)).to(ti.device)

    times = range(1, ti.T)
    predicts = ti.forward(data[:, 0], times)
    
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
      l2error = np.asarray(TimeInputHelper.get_operrs(ti, times=[ti.T-1]))
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

    if aeclass in JC_MODULES:
      aeparams = OmegaConf.create(aeparams)
      aeparams.sample.spatial_resolution = din
      if "k" in args:
        aeparams.sample.latents_dims = args["k"]
        seqclass["seq"][0] = args["k"]
        seqclass["seq"][-1] = args["k"]
      seqparams["activation"] = get_activation(seqparams["activation"])
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

    if aeclass == GIAutoencoder or aeclass == TCAutoencoder:
      # assume it is an integer, equally spaced grid on [0, 1]
      domaingrid = np.linspace(0, 1, dataset.data.shape[-1]).reshape(-1, 1)
      aeparams["domaingrid"] = domaingrid
      aeparams["decoderActivation"] = get_activation(aeparams["decoderActivation"])

    propparams = copy.deepcopy(seqparams)
    tiprop = args.get("tiprop", False)
    if tiprop:
      propparams["seq"][0] = propparams["seq"][-1] + 1


    return WeldNet(dataset, windows, aeclass, aeparams.copy(), seqclass, propparams, seqclass, copy.deepcopy(seqparams), straightness=straightness,accumulateprop=accumulateprop, decodedprop=decodedprop, kinetic=kinetic, td=td, seed=seed, device=device, autonomous=autonomous)
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

      arr = weld.tests[w] if testonly else weld.dataset.data[:, weld.windowvals[w], :]
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
    for t in ts:
      w = weld.find_window(t)
      dim = weld.aes[w].reduced

      arr = weld.alltest[:, t:t+1, :] if testonly else weld.dataset.data[:, t:t+1, :]
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
      data = weld.dataset.data[weld.numtrain:, t:t+1, :] if testonly else weld.dataset.data[:, t:t+1, :]
      data = utils.collect_times_dataparams(data)
      data = torch.tensor(data).to(weld.device, dtype=torch.float32)

      w = weld.find_window(t)
      proj = weld.project_window(w, data).cpu().detach().numpy()
      data = data.cpu().detach().numpy()

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
      data = weld.dataset.data[weld.numtrain:, :, :]
    else:
      data = weld.dataset.data

    inputt = torch.tensor(data[:, t, :]).to(weld.device, dtype=torch.float32)
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

    if testonly:
      data = weld.dataset.data[weld.numtrain:,]
    else:
      data = weld.dataset.data

    if xs is None:
      xs = np.linspace(0, 1, data.shape[2])

    input = torch.tensor(data[:, t, :]).to(weld.device, dtype=torch.float32)
    references = [data[:, t+s+1, :] for s in range(steps)]

    predicteds = weld.propagate(input, t, steps)
    predictedvals = [weld.decode_window(weld.find_window(t+i+1), x).cpu().detach().numpy() for i, x in enumerate(predicteds)]
    
    predicteds = [x.cpu().detach().numpy() for x in predicteds]

    input = input.cpu().detach().numpy()

    errors = []
    
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

    exacts = [torch.tensor(data[:, i, :]).to(weld.device, dtype=torch.float32) for i in range(data.shape[1])]
    projecteds = [weld.project_window(weld.find_window(i), exacts[i]).cpu().detach().numpy() for i in range(len(exacts))]
    exacts = [x.cpu().detach().numpy() for x in exacts]

    errors = []
    
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
  def plot_projops(weld, title="", save=False):
    fig, ax = plt.subplots(figsize=(8, 5))

    projerrs = WeldHelper.get_projerr_times(weld)
    operrs = WeldHelper.get_operrs(weld)

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
  def compare_projops(models, labels=None, relative=True, title=None, ylims=None, difference=False, testonly=True, windowlines=True):
    fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    for i, x in enumerate(models):    
      if windowlines:
        for xx in x.windowvals[:-1]:
          ax.axvline(xx[-1], linestyle=":", alpha=0.5, color="gray")

      projerrs = np.asarray(WeldHelper.get_projerr_times(x, testonly=testonly, relative=relative)[1:])
      operrs = np.asarray(WeldHelper.get_operrs(x, testonly=testonly))

      if difference:
        ax.plot(np.log10(operrs - projerrs), label=f"{x.W if labels is None else labels[i]} Propagator Gap", c=colors[i])
      else:
        ax.plot(np.log10(projerrs), "--", c=colors[i], alpha=0.5)
        ax.plot(np.log10(operrs), label=f"{x.W if labels is None else labels[i]}", c=colors[i])

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
  def plot_latent_trajectory(self, testnums, t=0, steps=-1, threed=False, figax=None):
    shapes = ["*"]
    shapesactual = ["o"]
    if steps == -1:
      steps = self.T - t - 1

    arr = torch.tensor(self.dataset.data[testnums, :, :]).to(self.device, dtype=torch.float32)
    outlist = self.propagate(arr[:, t, :], t, steps)
    actual = [self.encode_window(self.find_window(tt), arr[:, tt, :]) for tt in range(t + 1, t + steps + 1)]
    
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
#         testerr1, testerr2, testerrinf = get_proj_errors(model, testarr, ords=(1, 2, np.inf))
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
