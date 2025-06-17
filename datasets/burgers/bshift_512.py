import scipy.io
import numpy as np

def create_dataset():
    data_dict = scipy.io.loadmat("./grfarc2visc0p001-shift.mat")
    return {
        'data': data_dict['grfarc2visc0p001'],
        'params': data_dict['params'],
        'time': np.linspace(0,1,51),
        'space': np.linspace(0,1,512+1)[:-1]
    }
