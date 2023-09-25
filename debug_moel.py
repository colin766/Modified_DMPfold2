# DMPfold2 end-to-end approach training script - D.T.Jones 2020

import sys
import os
import time
import random
import torch

from math import sqrt, log, exp


from dmpfold.network import GRUResNet
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda")
network = GRUResNet(512,128).to(device)
inputs = torch.randint(0,21, (23,119)).to(device)
inputs2 = torch.randn((1,443,119,119)).to(device)


coords, confs = network(inputs, inputs2, nloops=3, refine_steps=100)