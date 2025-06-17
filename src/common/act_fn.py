import torch.nn as nn
def get_actvn(act:str):
    if act == "relu":
        return nn.ReLU()