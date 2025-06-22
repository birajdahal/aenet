import numpy as np
import scipy.io
import warnings

def hat(x, a, b, alpha):
    def relu(x):
        return np.maximum(x, 0)
     
    return 2 * alpha / (b - a) * (relu(x - a) - 2 * relu(x - (a+b)/2) + relu(x-b))

def combined_hat(x, h, alpha, beta, length=0.05, a1=0.1, a2=0.2):
    return hat(x, a1, a1+length, alpha) + hat(x, a2 + h/10, a2 + length + h/10, beta)

def generate_hat_manifold(num, xrange=[0, 1], h_range=[0, 3], alpha_range=[1, 10], beta_range=[1, 10]):
    hvals = h_range[0] + np.random.rand(num) * (h_range[1] - h_range[0])
    avals = alpha_range[0] + np.random.rand(num) * (alpha_range[1] - alpha_range[0])
    bvals = beta_range[0] + np.random.rand(num) * (beta_range[1] - beta_range[0])

    x = np.expand_dims(np.linspace(xrange[0], xrange[1], 1024), axis=1)
    return np.transpose(combined_hat(x, hvals, avals, bvals))

def generate_hat_data(num, randomt=False, trange=[0, 0.3], xrange=[0, 1], h_range=[0, 3], alpha_range=[1, 4], beta_range=[1, 4], res=1024, params=True):
    hvals = h_range[0] + (np.random.rand(num)) * (h_range[1] - h_range[0])
    avals = alpha_range[0] + (0*np.random.rand(num)) * (alpha_range[1] - alpha_range[0]) # dim 1
    bvals = beta_range[0] + (0 * (np.random.rand(num))) * (beta_range[1] - beta_range[0]) # dim 2

    x = np.expand_dims(np.linspace(xrange[0], xrange[1], res), axis=1)
    
    numt = 201
    alldata = np.zeros((num, numt, res))

    if not randomt:
        tvals = np.linspace(trange[0], trange[1], numt)
    else:
        deltat = (trange[1] - trange[0])/numt
        tvals = np.linspace(trange[0], trange[1], numt) + (2 * np.random.rand(numt) - 1) * (deltat / 4)
        print(tvals)

    for t in range(numt):
        alldata[:, t, :] = np.transpose(combined_hat(x, hvals, avals, bvals, a1=0.1+tvals[t], a2=0.2+tvals[t]))

    if params:
       return (alldata, np.column_stack((hvals, avals)))
    else:
      return alldata

alldata, params = generate_hat_data(2500, res=512, randomt=False)

scipy.io.savemat("hatsshift_2500.mat", {'alldata': alldata, 'params': params})

print(params.shape)