from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

def L96(x, t, D, f):
    d = np.zeros(D)
    for i in range(D):
        d[i] = (x[(i + 1) % D] - x[i - 2]) * x[i - 1] - x[i] + f

    return d

def generate_l96_simple(N, T, M, F=8, params=True, length=2):
  alldata = np.zeros((N, T, M))
  avals = np.random.random((N, 1))

  for i in range(N):
    x0 = np.ones(M)
    x0[0] += length * avals[i, 0]

    t = np.linspace(0, 10, T)

    x = odeint(L96, x0, t, args=(M, F))
    alldata[i, :, :] = x
    print(i)

  if params:
    return (alldata, avals)
  else:
    return alldata

alldata, params = generate_l96_simple(800, 201, 256, length=1)
scipy.io.savemat("l96_shift_try3.mat", {'alldata': alldata, 'params': params})