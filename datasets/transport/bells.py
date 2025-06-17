import numpy as np
import scipy.io
import warnings


def bell_base(x, w):
    return np.exp(-1 * (w * x) ** 2)

def bell_mfd(x, a, h, w=50):
    return a * bell_base(x - 0.1, w=w) + bell_base(x - 0.3 - h, w=w)


def generate_bell_data(num, trange=[0, 0.3], xrange=[0, 1], h_range=[0, 3], a_range=[0, 3], res=1024, params=True):
    hvals = h_range[0] + np.random.rand(num) * (h_range[1] - h_range[0]) 
    avals = a_range[0] + np.random.rand(num) * (a_range[1] - a_range[0]) * 0

    x = np.expand_dims(np.linspace(xrange[0], xrange[1], res), axis=1)
    
    numt = 51
    alldata = np.zeros((num, numt, res))
    tvals = np.linspace(trange[0], trange[1], numt)
    for t in range(numt):
        alldata[:, t, :] = np.transpose(bell_mfd(x - tvals[t], avals + 1, hvals * 0.1))

    if params:
       return (alldata, np.column_stack((hvals, avals)))
    else:
      return alldata

alldata, params = generate_bell_data(2500, res=1024)

scipy.io.savemat("bells2_2500_shift.mat", {'alldata': alldata, 'params': params})

print(params.shape)