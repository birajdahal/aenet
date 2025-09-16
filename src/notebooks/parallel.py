import sys
import torch
import datetime

sys.path.insert(0, "..")
basedir = "../.."

import models

from common.config import create_object, load_config

torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
dconfig = load_config(f"../configs/data/transport/hatsscale.yaml")
dconfig.datasize.spacedim = 1
dset = create_object(dconfig)

dset.downsample_time(6)

# models
config = load_config("../configs/experiments/weldnormal.yaml")
experiment = models.WeldHelper(config)

epochs = 300
weld = experiment.create_weld(dset, windows=1, k=4, td=None, seed=0, device=device, accumulateprop=True)

weld.train_aes_plus_props(epochs, lr=1e-3)
weld.train_propagators(epochs // 3, lr=1e-4)

models.WeldHelper.plot_projops(weld, datetime.datetime.now().strftime('%d%B%H%M%S'), save=True)