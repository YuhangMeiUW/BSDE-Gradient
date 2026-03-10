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

# Parameters
T = 2.0  # End time
n = 2    # Dimension of state space
m = 2    # Dimension of Brownian motion
N = 100 # Number of training samples
dt = 0.05  # Time step size
steps = int(T/dt)  # Number of time steps
noise_level = 1  # Noise level in the SDE
kf = 5 # iterations for phi

# exp_num = 10 # Experiment number for saving results

# Generate initial data
X_0 = torch.randn((N, n)) 

# Linear system dynamics
def f(x, t, u_t=None):
    """
    Drift function of X_t for a linear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Drift vector. Shape (N, n)
    """
    A = torch.tensor([[0.0, 1.0], [-1.0, -0.5]])  
    return x @ A.T

def g(x):
    """
    Diffusion function of X_t for a linear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Diffusion matrix. Shape (N, n, m)
    """
    B = torch.tensor([[1.0, 0.0], [0.0, 1.0]]) * noise_level
    B = B.unsqueeze(0).repeat(x.shape[0], 1, 1)
    return B

def lf(x):
    """
    Terminal cost function for a linear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Terminal cost. Shape (N,)
    """
    Q_f = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return 0.5 * ((x @ Q_f) * x).sum(dim=1)

def partial_lf(x):
    """
    Gradient of the terminal cost function for a linear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Gradient of the terminal cost. Shape (N, n)
    """
    Q_f = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return x @ Q_f

time_grid = torch.arange(0, steps+1) * dt

W_f = torch.randn((steps+1, N, m)) * np.sqrt(dt)  # forward noise
W_b = torch.randn((steps+1, N, m)) * np.sqrt(dt)  # backward noise


X_forward = rollout(f, g, T, dt, X_0, W_f)

def exact_score(x, t):
    """
    Compute the exact solution of the linear SDE at time t.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
        t (float): Time point.
        X_train (torch.Tensor): Training data for empirical mean and covariance. Shape (steps+1, N, n)
    Returns:
        torch.Tensor: Score at time t. Shape (N, n)
    """
    m_k_t = X_forward.mean(dim=1) # shape (T, n)
    x_minus_m = X_forward - m_k_t[:, torch.newaxis, :] # shape (T, N, n)
    Sigma_k_t = torch.einsum('tni,tnj->tij', x_minus_m, x_minus_m) / (N-1)
    Sigma_inv = torch.inverse(Sigma_k_t[int(t/dt), :, :])
    mu = m_k_t[int(t/dt), :]
    return -(x - mu) @ Sigma_inv

score_nn = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=32, num_blocks=2)
score_optimizer = torch.optim.AdamW(score_nn.parameters(), lr=1e-3, weight_decay=1e-4)
score_scheduler = torch.optim.lr_scheduler.StepLR(score_optimizer, step_size=1000, gamma=0.9)
score_loss_history = train_score_network(score_nn, X_forward, time_grid, g, noise_level, score_optimizer, score_scheduler, batch_size=16, iterations=10000)

def fy(x, y, z, t):
    """
    Drift function for Y_t in the BSDE.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Drift vector. Shape (N, n)
    """
    A = torch.tensor([[0.0, 1.0], [-1.0, -0.5]])  # stable
    cost_term = torch.zeros_like(x)  # shape (N, n)
    lag_term_mat = (A.T).repeat(x.shape[0], 1, 1)  # shape (N, n, n)
    trace_term = torch.zeros_like(x)  # shape (N, n)
    return cost_term, lag_term_mat, trace_term

X_T = X_forward[-1, :, :]  # Terminal state
X_b = time_reversal(f, g, T, dt, X_T, W_b, [score_nn], nn_num=1)
# X_b = time_reversal(f, g, T, dt, X_T, W_b, exact_score)
Y_T = partial_lf(X_T)  # Terminal condition for Y_t

# Solve the BSDE by using time reversal, keep training phi network until convergence
phi_net = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=32, num_blocks=2)
optimizer = torch.optim.AdamW(phi_net.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)


for k in range(kf):
    Y_b = time_reversal_bsde(fy, g, phi_net, T, dt, Y_T, W_b, [score_nn], X_b=X_b, nn_num=1)
    # Y_b = time_reversal_bsde(fy, g, phi_net, T, dt, Y_T, W_b, exact_score, X_b=X_b)
    loss_history = train_phi_network(phi_net, X_b, Y_b, time_grid, optimizer, scheduler, batch_size=16, iterations=5000)
    print(f"sample number {N}, noise level {noise_level}, Phi Network Training Iteration {k+1}/{kf} completed.")


torch.save(phi_net.state_dict(), f'network/phi_network_linear_nl{noise_level}_samplenum{N}.pth')