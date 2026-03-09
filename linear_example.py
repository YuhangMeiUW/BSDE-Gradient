'''
This code is for comparing the result on linear sde with four method 
1) Time reversal BSDE with exact solution for the score function
2) Time reversal BSDE with score network trained on denoising score matching
3) Auto differentiation from the terminal cost
4) Adjoint matching method
The ground truth is obtained by solving the Riccati equation for the linear-quadratic control problem. 
'''

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from utils import train_score_network, rollout, time_reversal, time_reversal_bsde, train_phi_network, batched_jacobian
from network import ScoreNetwork