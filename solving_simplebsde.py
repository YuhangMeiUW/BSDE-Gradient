import torch
import numpy as np

from utils import train_score_network, generate_initial_data, rollout, time_reversal, noise, time_reversal_bsde, train_phi_network
from network import ScoreNetwork, PhiNetwork

# Parameters
T = 4.0  # End time
n = 2    # Dimension of state space
m = 2    # Dimension of Brownian motion
N = 4000 # Number of training samples
INITIAL_DIST = 'Gaussian'  # Initial distribution type ('Gaussian', 'Bimodal', or 'Multimodal')
m_0 = torch.tensor([0.0, 0.0])  # Mean of initial distribution
sigma_0 = torch.eye(n) * 1  # Covariance of initial distribution
dt = 0.005  # Time step size
steps = int(T/dt)  # Number of time steps
noise_level = 0.7  # Noise level in the SDE
kf = 10 # iterations for phi

# exp_num = 10 # Experiment number for saving results

# Generate initial data
X_0 = generate_initial_data(INITIAL_DIST, m_0, sigma_0, N, shift=5)

# Linear system dynamics
def f(x):
    """
    Drift function of X_t for a linear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Drift vector. Shape (N, n)
    """
    A = torch.tensor([[0.0, 1.0], [-1.0, 0.5]]) # unstable
    # A = torch.tensor([[0.0, 1.0], [-1.0, -0.5]])  # stable
    # A = torch.tensor([[-1.0]])  
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

W_f = torch.zeros((steps+1, N, m))# forward noise
W_b = torch.zeros((steps+1, N, m))# backward noise
for noise_step in range(steps+1):
    W_f[noise_step, :, :] = noise(dt, N, m)
    W_b[noise_step, :, :] = noise(dt, N, m)

    # Generate training data by rolling out the SDE
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
    Sigma_k_t = torch.einsum('tni,tnj->tij', x_minus_m, x_minus_m) / N
    Sigma_inv = torch.inverse(Sigma_k_t[int(t/dt), :, :])
    mu = m_k_t[int(t/dt), :]
    return -(x - mu) @ Sigma_inv

# score_nn = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=64, num_blocks=3)
# score_optimizer = torch.optim.AdamW(score_nn.parameters(), lr=1e-3, weight_decay=1e-4)
# score_scheduler = torch.optim.lr_scheduler.StepLR(score_optimizer, step_size=200, gamma=0.9)
# score_loss_history = train_score_network(score_nn, X_forward, time_grid, g, noise_level, score_optimizer, score_scheduler, batch_size=64, iterations=10000)

def fy(x):
    """
    Drift function for Y_t in the BSDE.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Drift vector. Shape (N, n)
    """
    A = torch.tensor([[0.0, 1.0], [-1.0, 0.5]]) # unstable
    # A = torch.tensor([[0.0, 1.0], [-1.0, -0.5]])  # stable
    return x @ A

# X_T = X_forward[-1, :, :]  # Terminal state
# X_b = time_reversal(f, g, T, dt, X_T, W_b, exact_score)
# Y_T = partial_lf(X_T)  # Terminal condition for Y_t

# # Solve the BSDE by using time reversal, keep training phi network until convergence
# phi_net = PhiNetwork(input_dim=n+1, out_dim=n, hidden_dim=32)
# optimizer = torch.optim.AdamW(phi_net.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)


# for k in range(kf):
#     Y_b = time_reversal_bsde(fy, g, phi_net, T, dt, Y_T, W_b, exact_score, X_b)
#     loss_history = train_phi_network(phi_net, X_b, Y_b, time_grid, optimizer, scheduler, batch_size=64, iterations=5000)
#     print(f"Phi Network Training Iteration {k+1}/{kf} completed.")


# torch.save(phi_net.state_dict(), f'data/phi_network_linear_bsde_unstableA_nl{noise_level}.pth')

# Solve the question by adam on theta
theta = torch.randn((N,n), requires_grad=True)
loss = lf(rollout(f, g, T, dt, theta, W_f)[-1, :, :]).sum(dim=0)
grad_theta = torch.autograd.grad(loss.sum(), theta)[0]
# optimizer = torch.optim.Adam([theta], lr=1e-3)
# for step in range(1):
#     # print(f"Step {step}")
#     optimizer.zero_grad()
#     loss = 0.0
#     # x = theta.expand(N, -1)
#     loss = lf(rollout(f, g, T, dt, theta, W_f)[-1]).mean(dim=0)
#     loss.backward()
#     # optimizer.step()
#     ## print gradients in last step
#     if step == 0:
#         last_grad = theta.grad.detach().clone()  # safe copy of gradient
#         # print("Final gradient:", last_grad)
    
#     if step % 100 == 0 or step == 499:
#         print(f"Step {step}, Loss: {loss.item()/N}")

# torch.save(grad_theta, f'data/initial_theta_grad_stableA_nl{noise_level}.pth')
# torch.save(theta.detach(), f'data/initial_theta_value_stableA_nl{noise_level}.pth')
