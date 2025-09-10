import numpy as np
import scipy.io
import warnings

def hat(x, a, b, alpha):
  def relu(x):
      return np.maximum(x, 0)
    
  return 2 * alpha / (b - a) * (relu(x - a) - 2 * relu(x - (a+b)/2) + relu(x-b))

def ndim_hat(x, a, h, epsilon=0.2):
  product = np.ones(x.shape[0])
  for d in range(x.shape[1]):
    product *= hat(x[:, d], h, h+epsilon, a)

  return product

def generate_twodim_hat_data(num, arange=[0, 3], hrange=[0, 3], trange=[0, 0.3], numt=301, params=True, res=32):
  hvals = hrange[0] + (np.random.rand(num)) * (hrange[1] - hrange[0])
  avals = arange[0] + (np.random.rand(num)) * (arange[1] - arange[0])

  x = np.linspace(0, 1, res)
  y = np.linspace(0, 1, res)

  X, Y = np.meshgrid(x, y)

  input = np.column_stack((X.flatten(), Y.flatten()))
  alldata = np.zeros((num, numt, res, res))
  tvals = np.linspace(trange[0], trange[1], numt)
  for i in range(num):
    for t in range(numt):
      alldata[i, t, :, :] = ndim_hat(input, avals[i] + 1, 0.1*hvals[i] + tvals[t]).reshape(X.shape)

    print(i)

  if params:
    return (alldata, np.column_stack((avals, hvals)))
  else:
    return alldata

alldata, params = generate_twodim_hat_data(500, hrange=[0, 0], res=32)

scipy.io.savemat("hats2d_scale.mat", {'alldata': alldata, 'params': params})

print(params.shape)