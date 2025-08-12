import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import warnings
import ipywidgets as widgets
import matplotlib as mpl

from scipy.interpolate import interpn
from matplotlib.animation import FuncAnimation
from scipy.io.matlab.miobase import MatReadWarning
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
from omegaconf import DictConfig

import torch
import torch.nn as nn

def get_pca_error(data, k, ord=2):
  pca = PCA(n_components=k)

  principalComponents = pca.fit_transform(data)
  rdata = pca.inverse_transform(principalComponents)
  rerror = np.mean(np.linalg.norm(rdata - data, axis=1, ord=ord) / np.linalg.norm(data, axis=1, ord=ord))

  return rerror

def num_params(model):
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  return params

def barycentric_contraction(points, scale):
  center = np.mean(points, axis=0)
  return scale * (points - center)

def linear_interpolate(arr, orig, new): # look to make this parallel
  return np.apply_along_axis(lambda row: interpn([orig], row, new), 1, arr)

def cubic_interpolate(arr, orig, new): # look to make this parallel
  return np.apply_along_axis(lambda row: interpn([orig], row, new, method="cubic"), 1, arr)

def pde_system_video(array, t, T=1, noise=0):
  fig, ax = plt.subplots()
  noisedarray = array.copy()
  noisedarray[:, 1:, :] += np.random.normal(scale=noise, size=noisedarray[:, 1:, :].shape)

  def update(frame):
    ax.clear()
    ax.plot(array[frame, 0, :], 'r', label="t=0")
    ax.plot(noisedarray[frame, t, :], 'b', label=f"t={t / (array.shape[1] - 1) * T}")
    ax.set_title(frame)
    ax.legend()
    
    if frame % 10 == 0:
      print(frame)

  num_frames = 50
  ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)
  ani.save("output_video.mp4", writer="ffmpeg", fps=5)

def plot_pde_system(array, T=1, noise=0):
  half = int(array.shape[1] / 2)
  fig, ax = plt.subplots()
  noisedarray = array.copy()
  noisedarray[:, 1:, :] += np.random.normal(scale=noise, size=noisedarray[:, 1:, :].shape)
  
  @widgets.interact(j=(0, array.shape[0] - 1), t=(0, array.shape[1] - 1))
  def plot_io(j, t=20):
    print(j)
    ax.clear()
    ax.plot(array[j, 0, :], 'r', label="t=0")
    ax.plot(noisedarray[j, t, :], 'b', label=f"t={t / (array.shape[1] - 1) * T}")
    ax.set_title(j)
    ax.legend()

# assuming data is N x 2 array
def twodim_colors(data):
  hue_min, hue_max = 0, 0.7
  sat_min, sat_max = 0.3, 1 

  hue = np.interp(data[:, 0], (data[:, 0].min(), data[:, 0].max()), (hue_min, hue_max))
  sat = np.flip(np.interp(data[:, 1], (data[:, 1].min(), data[:, 1].max()), (sat_min, sat_max)))

  color_array = np.column_stack([hue, np.ones_like(hue), sat])

  return hsv_to_rgb(color_array)

def determine_params(paramarr):
  encoding_param = []
  P = paramarr.shape[1]

  if P == 1:
    return 0

  for p in range(P):
    if np.abs(paramarr[0, p] - paramarr[1, p]) > 0:
      encoding_param.append(p)

  return encoding_param

def reset_model_weights(layer):
  if hasattr(layer, 'reset_parameters'):
    layer.reset_parameters()
  else:
    if hasattr(layer, 'children'):
      for child in layer.children():
        reset_model_weights(child)

def duplicate_rows(arr, T):
    N, P = arr.shape

    new_shape = (N * T, P)
    result_array = np.empty(new_shape, dtype=arr.dtype)
    
    for i in range(N):
        result_array[i * T : (i + 1) * T, :] = arr[i, :]

    return result_array

def reduce_params(params, tol=1e-4):
  arr = np.asarray(params)
  if arr.ndim != 2:
      raise ValueError("params must be 2-D (N × p)")

  # np.ptp gives max–min along each column
  varying = np.ptp(arr, axis=0) > tol
  out = arr[:, varying]
  return out

# NEED TO IMPLEMENT SHIFTING FOR THE SCALE
class DynamicData(torch.utils.data.Dataset):
  def __init__(self, inconfig, seed=0, spacedim=1):
    if isinstance(inconfig, tuple):
      if isinstance(inconfig[0], str):
        with warnings.catch_warnings():
          warnings.simplefilter('ignore', MatReadWarning)
          self.origdata = scipy.io.loadmat(inconfig[0])

        self.data = self.origdata[inconfig[1]]

        if "params" in self.origdata:
          self.params = self.origdata["params"]
        else:
          self.params = None
          
      elif isinstance(inconfig[0], np.ndarray):
        self.data = inconfig[0]
        self.params = inconfig[1]
        self.origdata = {"data": self.data, "params": self.params}

      self.scale = 1
      self.name = None

    elif isinstance(inconfig, DictConfig):
      with warnings.catch_warnings():
        warnings.simplefilter('ignore', MatReadWarning)
        self.origdata = scipy.io.loadmat(inconfig.file.filestr)

      self.name = inconfig.file.name
      self.data = self.origdata[inconfig.file.dataname]

      if "params" in self.origdata:
        self.params = self.origdata["params"]
      else:
        self.params = None

      if "datasize" in inconfig:
        if inconfig.datasize.subset:
          self.subset_data(inconfig.datasize.subset)
        if inconfig.datasize.space:
          self.downsample(int(self.data.shape[2] / inconfig.datasize.space))
        if inconfig.datasize.time:
          self.downsample_time(int(self.data.shape[1] / inconfig.datasize.time))
        if inconfig.datasize.scaledown:
          self.scaledown()

        spacedim = inconfig.datasize.spacedim
        newshape = list(self.data.shape[:1+spacedim]) + [-1]
        self.data = self.data.reshape(newshape)
      
      else:
        spacedim = len(self.data.shape) - 2

    self.data = np.float32(self.data)
    self.params = np.float32(self.params)

    if self.params is not None:
      self.params = reduce_params(self.params)
    
    np.random.seed(seed)
    self.shuffle_inplace()

    del self.origdata
    
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx]
  
  def subset_data(self, num):
    self.data = self.data[:num]
    if self.params is not None:
      self.params = self.params[:num]

  def downsample_time(self, factor):
    f = int(self.data.shape[1] / (self.data.shape[1] // int(factor)))
    self.data = self.data[:, ::f]

  def downsample(self, factor):
    # downsample each axis by that factor
    factors = [int(self.data.shape[i] / (self.data.shape[i] // int(factor))) for i in range(len(self.data.shape) - 2)]
    dslice = [slice(None), slice(None)] + [slice(None, None, f) for f in factors]
    self.data = self.data[tuple(dslice)]

  def shuffle_inplace(self):
    p = np.random.permutation(self.data.shape[0])
    self.data = self.data[p]

    if self.params is not None:
      assert(self.data.shape[0] == self.params.shape[0])
      self.params = self.params[p]
    
  def traintest_split_alltime(self, trainnum, t0=0, t1=-1, noise=0):
    if t1 == -1:
      t1 = self.data.shape[1] - 1
    
    noisedarray = self.data.copy()
    if noise > 0:
      noisedarray[:, 1:] += np.random.normal(scale=noise, size=noisedarray[:, 1:].shape)
      
    somedata = noisedarray[:, t0:t1]
      
    return np.split(somedata, [trainnum])

  def traintest_split_inout(self, trainnum, t0=0, t1=-1, noise=0):
    if t1 == -1:
      t1 = self.data.shape[1] - 1
    
    noisedarray = self.data.copy()
    if noise > 0:
      noisedarray[:, 1:] += np.random.normal(scale=noise, size=noisedarray[:, 1:].shape)
      
    somedata = np.column_stack([noisedarray[:, t0], noisedarray[:, t1]])
      
    return np.split(somedata, [trainnum])
  
  # todo, also compute a bias shift
  def scaledown(self):
    self.scale = max(np.max(self.data), -1 * np.min(self.data))
    self.data = self.data / self.scale

  def collect_times(self):
    N, T = self.data.shape[:2]
    Ds = list(self.data.shape[2:])

    data = self.data.reshape([N*T] + Ds, order="F")
    
    if self.params is not None:
      params = np.column_stack([np.tile(self.params, (T, 1)), np.repeat(np.arange(0, T), N)])
      return data, params
    else:
      return data

  def plot_svd(self, title="", dpi=80, t=0, maxnum=-1):
    if t >= 0:
      arr = self.data[:, t]
      arr = self.data.reshape(list(self.data.shape[:2]) + [-1])
    else:
      arr, _ = self.collect_times()

    arr = arr.reshape((arr.shape[0], -1))
    fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi)

    if maxnum > 0:
      arr = arr[:maxnum]

    u, s, vh = np.linalg.svd(arr, full_matrices=False)

    ax.plot(s, color="blue")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value Magnitude")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(title)

    fig.tight_layout()

  def plot_pca3_pt(self, param=0, s=0):
    plt.rcParams.update({'font.size': 14})
    pca = PCA(n_components=s+3)

    data, params = self.collect_times()
    data = data.reshape((data.shape[0], -1))
    principalComponents = pca.fit_transform(data)
    x = principalComponents[:, s+0]
    y = principalComponents[:, s+1]
    z = principalComponents[:, s+2]

    rdata = pca.inverse_transform(principalComponents)
    rerror = np.mean(np.linalg.norm(rdata - data, axis=1, ord=2) / np.linalg.norm(data, axis=1, ord=2))
    print("Reconstruction error", rerror)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(projection='3d')
    ptcolors = twodim_colors(params[:, (param, -1)])
    sc = ax.scatter(x, y, z, c=ptcolors, s=2)
    ax.set_xlabel(f'Component {s+1}')
    ax.set_ylabel(f'Component {s+2}') 
    ax.set_zlabel(f'Component {s+3}')
    ax.set_box_aspect((1.3, 1.3, 1))
    ax.view_init(elev=30, azim=45)
    ax.set_title(f"Parameter {param} and Time")

  def plot_pca3(self, param=-1, s=0, t=-1):
    if param == -1:
      param = determine_params(self.params)[0]

    plt.rcParams.update({'font.size': 14})
    pca = PCA(n_components=s+3)

    if t >= 0:
      data = self.data[:, t]
      params = self.params
    else:
      data, params = self.collect_times()

    data = data.reshape((data.shape[0], -1))
    principalComponents = pca.fit_transform(data)
    x = principalComponents[:, s+0]
    y = principalComponents[:, s+1]
    z = principalComponents[:, s+2]

    rdata = pca.inverse_transform(principalComponents)
    rerror = np.mean(np.linalg.norm(rdata - data, axis=1, ord=2) / np.linalg.norm(data, axis=1, ord=2))
    print("Reconstruction error", rerror)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')

    sc = ax.scatter(x, y, z, c=params[:, param], s=2)
    plt.colorbar(sc, ax=ax, location="right", pad=0)
    ax.set_xlabel(f'Component {s+1}')
    ax.set_ylabel(f'Component {s+2}')
    ax.set_zlabel(f'Component {s+3}')
    ax.set_box_aspect((1.3, 1.3, 1))
    ax.view_init(elev=30, azim=45)
    ax.set_title(f"Parameter {param}")

    if t == -1:
      sc1 = ax1.scatter(x, y, z, c=params[:, -1], s=2, cmap="copper")
      plt.colorbar(sc1, ax=ax1, location="right", pad=0)
      ax1.set_xlabel(f'Component {s+1}')
      ax1.set_ylabel(f'Component {s+2}')
      ax1.set_zlabel(f'Component {s+3}')
      ax1.set_box_aspect((1.3, 1.3, 1))
      ax1.view_init(elev=30, azim=45)
      ax1.set_title("Time")

    return (fig, ax)

  def plot_pca2_pt(self, param=0, s=0):
    plt.rcParams.update({'font.size': 14})
    pca = PCA(n_components=s+2)

    data, params = self.collect_times()
    data = data.reshape((data.shape[0], -1))
    principalComponents = pca.fit_transform(data)
    x = principalComponents[:, s+0]
    y = principalComponents[:, s+1]

    rdata = pca.inverse_transform(principalComponents)
    rerror = np.mean(np.linalg.norm(rdata - data, axis=1, ord=2) / np.linalg.norm(data, axis=1, ord=2))
    print("Reconstruction error", rerror)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ptcolors = twodim_colors(params[:, (param, -1)])
    sc = ax.scatter(x, y, c=ptcolors, s=2)
    ax.set_xlabel(f'Component {s+1}')
    ax.set_ylabel(f'Component {s+2}')
    ax.set_title(f"Parameter {param} and Time")

  def plot_pca2(self, param=0, s=0, t=-1):
    plt.rcParams.update({'font.size': 14})
    pca = PCA(n_components=s+2)

    if t >= 0:
      data = self.data[:, t]
      params = self.params
    else:
      data, params = self.collect_times()

    data = data.reshape((data.shape[0], -1))
    principalComponents = pca.fit_transform(data)
    x = principalComponents[:, s+0]
    y = principalComponents[:, s+1]

    rdata = pca.inverse_transform(principalComponents)
    rerror = np.mean(np.linalg.norm(rdata - data, axis=1, ord=2) / np.linalg.norm(data, axis=1, ord=2))
    print("Reconstruction error", rerror)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    sc = ax.scatter(x, y, c=params[:, param], s=1)
    plt.colorbar(sc, ax=ax, location="right", pad=0)
    ax.set_xlabel(f'Component {s+1}')
    ax.set_ylabel(f'Component {s+2}')
    ax.set_title(f"Parameter {param}")

    if t < 0:
      sc = ax1.scatter(x, y, c=params[:, -1], s=1, cmap="copper")
      plt.colorbar(sc, ax=ax1, location="right", pad=0)
      ax1.set_xlabel(f'Component {s+1}')
      ax1.set_ylabel(f'Component {s+2}')
      ax1.set_title("Time")

    return fig

  def get_from_params(self, params):
    # Min-max normalize each column
    minv = self.params.min(axis=0)
    maxv = self.params.max(axis=0)
    scale = maxv - minv
    scale[scale == 0] = 1.0  # avoid divide-by-zero

    norm_self = (self.params - minv) / scale
    norm_input = (params - minv) / scale

    # L1 distance
    a = np.linalg.norm(norm_self - norm_input, axis=1, ord=1)
    j = int(np.argmin(a))
    return j

  def plot_data(self, noise=0, mode="normal", topdown=False):
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import Colorbar
    # ensure tensor rank
    assert len(self.data.shape) < 5, "Data tensor must have <5 dimensions"

    # prepare noisy copy
    noisedarray = np.copy(self.data)
    noisedarray[:, 1:] += np.random.normal(scale=noise, size=noisedarray[:, 1:].shape)
    cmin, cmax = np.min(noisedarray), np.max(noisedarray)

    # COMMON setup
    fig = None
    ax = None
    axes = None
    # create base figure and axes
    if len(self.data.shape) == 3:
        fig, ax = plt.subplots(figsize=(4, 3))
    elif len(self.data.shape) == 4:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    else:
        raise AssertionError("Data must be 3D or 4D for plotting")

    # NORMAL interactive mode
    if mode == "normal":
        @widgets.interact(j=(0, noisedarray.shape[0] - 1),
                         t0=(0, noisedarray.shape[1] - 1),
                         t1=(0, noisedarray.shape[1] - 1))
        def plot_io(j, t0=0, t1=1):
            # clear axes
            if len(self.data.shape) == 3:
                ax.clear()
            else:
                axes[0].clear()
                axes[1].clear()

            # plot
            if topdown:
                assert len(self.data.shape) == 3, "topdown only supported for 3D time-series data"
                img = ax.imshow(noisedarray[j], aspect='auto', vmin=cmin, vmax=cmax, cmap='jet', origin="lower")
                ax.set_title(f"Data point {j} over all time vs. features")
            else:
                if len(self.data.shape) == 3:
                    ax.plot(noisedarray[j, t0, :], 'r', label=f"t={t0}")
                    ax.plot(noisedarray[j, t1, :], 'b', label=f"t={t1}")
                    ax.legend()
                else:
                    axes[0].imshow(noisedarray[j, t0], vmin=cmin, vmax=cmax, cmap="jet", origin="lower")
                    axes[0].set_title(f"t={t0}")
                    img2 = axes[1].imshow(noisedarray[j, t1], vmin=cmin, vmax=cmax, cmap="jet", origin="lower")
                    axes[1].set_title(f"t={t1}")

            title = f"Params: {self.params[j]}" if hasattr(self, 'params') and self.params is not None else f"Data point {j}"
            fig.suptitle(title)
            fig.tight_layout()

    # PARAMS interactive mode
    elif mode == "params":
        # build sliders
        sliders = {"t0": widgets.IntSlider(value=0, min=0, max=noisedarray.shape[1] - 1),
                   "t1": widgets.IntSlider(value=0, min=0, max=noisedarray.shape[1] - 1)}
        for i in range(self.params.shape[1]):
            minval, maxval = np.min(self.params[:, i]), np.max(self.params[:, i])
            sliders[f"p{i}"] = widgets.FloatSlider(value=minval, min=minval, max=maxval, step=0.01)

        @widgets.interact(t0=sliders['t0'],
                         t1=sliders['t1'],
                         **{k: sliders[k] for k in sliders if k.startswith('p')})
        def update(t0=0, t1=1, **args):
            # clear axes
            if len(self.data.shape) == 3:
                ax.clear()
            else:
                axes[0].clear()
                axes[1].clear()

            # find index
            j = self.get_from_params(np.array(list(args.values())))

            # plot
            if topdown:
                assert len(self.data.shape) == 3, "topdown only supported for 2D time-series data"
                img = ax.imshow(noisedarray[j], aspect='auto', vmin=cmin, vmax=cmax, cmap='jet', origin="lower")
                ax.set_title(f"Data point {j} over all time vs. features")
            else:
                if len(self.data.shape) == 3:
                    ax.plot(noisedarray[j, t0, :], 'r', label=f"t={t0}")
                    ax.plot(noisedarray[j, t1, :], 'b', label=f"t={t1}")
                    ax.legend()
                else:
                    axes[0].imshow(noisedarray[j, t0], vmin=cmin, vmax=cmax, cmap="jet", origin="lower")
                    axes[0].set_title(f"t={t0}")
                    img2 = axes[1].imshow(noisedarray[j, t1], vmin=cmin, vmax=cmax, cmap="jet", origin="lower")
                    axes[1].set_title(f"t={t1}")

            title = f"Params: {self.params[j]}" if self.params is not None else f"Data point {j}"
            fig.suptitle(title)
            fig.tight_layout()

# this function is terrible, deprecate it
def collect_times_dataparams(data, parameters=None):
    assert(isinstance(data, np.ndarray))
    #print("Function is deprecated")
    N = data.shape[0]
    T = data.shape[1]
    rest = data.shape[2:]

    collected = data.reshape([N*T] + list(rest), order="F")

    if parameters is None:
      return collected
    else:
      params = np.column_stack([np.tile(parameters, (T, 1)), np.repeat(np.arange(0, T), N)])

      return collected, params
  
