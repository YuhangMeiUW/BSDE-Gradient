import torch
import numpy as np

from utils import train_score_network, rollout, time_reversal, time_reversal_bsde, train_phi_network, non_adapted_adjoint
from network import ScoreNetwork, PhiNetwork

# Parameters
T = 4.0  # End time
n = 2    # Dimension of state space
m = 1    # Dimension of Brownian motion
N = 10000 # Number of training samples
m_0 = torch.tensor([0.0, 0.0])  # Mean of initial distribution
initial_var = 1.0
sigma_0 = torch.eye(n) * initial_var  # Covariance of initial distribution
dt = 0.05  # Time step size
steps = int(T/dt)  # Number of time steps
noise_level = 0.5  # Noise level in the SDE
kf = 20 # iterations for phi

# Generate initial data
X_0 = torch.randn((N, n)) * torch.sqrt(sigma_0.diag()) + m_0  # Shape (N, n)

# Nonlinear system dynamics
def f(x, t, ut=None):
    """
    Drift function of X_t for a nonlinear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
        t (torch.Tensor, optional): Time tensor. Shape (N,) or (1,). Default is None.
    Returns:
        torch.Tensor: Drift vector. Shape (N, n)
    """
    df1 = x[:, 1]
    df2 = torch.sin(x[:, 0]) - 0.01 * x[:, 1]
    df = torch.stack((df1, df2), dim=1)
    return df


def g(x):
    """
    Diffusion function of X_t for a nonlinear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Diffusion matrix. Shape (N, n, m)
    """
    B = torch.tensor([[0.0], [1.0]]) * noise_level
    B = B.unsqueeze(0).repeat(x.shape[0], 1, 1)
    return B

def lf(x):
    """
    Terminal cost function for a nonlinear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Terminal cost. Shape (N,)
    """
    # Q_f = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # return 0.5 * ((x @ Q_f) * x).sum(dim=1)
    return 0.5 * x[:,1] * x[:,1] + 1 - torch.cos(x[:,0])

def partial_lf(x):
    """
    Gradient of the terminal cost function for a nonlinear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Gradient of the terminal cost. Shape (N, n)
    """
    dlf1 = torch.sin(x[:,0])
    dlf2 = x[:,1]
    return torch.stack((dlf1, dlf2), dim=1)


def H_x(x, y, z, t=None):
    """
    Partial derivative of Hamiltonian respect to the x for Y_t in the BSDE.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
        y (torch.Tensor): Costate vector. Shape (N, n)
        z (torch.Tensor): Second order term in BSDE. Shape (N, n, m)
        t (torch.Tensor, optional): Current time. Shape (1,). Default is None.
    Returns:
        # torch.Tensor: Drift vector. Shape (N, n)
        cost_term (torch.Tensor): Running cost term. Shape (N, n)
        lag_term_mat (torch.Tensor): Lagrange term y^T f_x. Shape (N, n, n)
        trace_term (torch.Tensor): Trace term Tr(z g_x). Shape (N, n)
    """
    # running cost term
    # zero for this example
    cost_term = torch.zeros_like(x)  # shape (N, n)

    # lagrange term y^T f_x
    lag_term_mat = torch.zeros((x.shape[0], n, n))  # shape (N, n, n)
    lag_term_mat[:, 0, 1] = torch.cos(x[:, 0]) # shape (N,)
    lag_term_mat[:, 1, 0] = 1.0
    lag_term_mat[:, 1, 1] = -0.01

    # trace term Tr(z g_x)
    trace_term = torch.zeros_like(x)  # shape (N, n)
    return cost_term, lag_term_mat, trace_term

def adjoint_dyn(x, t):
    """
    non-adapted adjoint process
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Drift vector. Shape (N, n)
        torch.Tensor: Lagrange term matrix. Shape (N, n, n)
    """
    cost_term = torch.zeros_like(x)  # shape (N, n)
    lag_term_mat = torch.zeros((x.shape[0], n, n))  # shape (N, n, n)
    lag_term_mat[:, 0, 1] = torch.cos(x[:, 0]) # shape (N,)
    lag_term_mat[:, 1, 0] = 1.0
    lag_term_mat[:, 1, 1] = -0.01

    return cost_term, lag_term_mat


time_grid = torch.arange(0, steps+1) * dt

W_f = torch.randn((steps+1, N, m)) * np.sqrt(dt)  # forward noise
W_b = torch.randn((steps+1, N, m)) * np.sqrt(dt)  # backward noise

# Generate training data by rolling out the SDE
X_forward = rollout(f, g, T, dt, X_0, W_f)

score_nn = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=64, num_blocks=3)
score_optimizer = torch.optim.AdamW(score_nn.parameters(), lr=1e-3, weight_decay=1e-4)
score_scheduler = torch.optim.lr_scheduler.StepLR(score_optimizer, step_size=1000, gamma=0.9)

score_loss_history = train_score_network(score_nn, X_forward, time_grid, g, noise_level, score_optimizer, score_scheduler, batch_size=64, iterations=10000)
print("Score Network Training completed.")



X_T = X_forward[-1, :, :]  # Terminal state
X_b = time_reversal(f, g, T, dt, X_T, W_b, [score_nn], nn_num=1)
Y_T = partial_lf(X_T)  # Terminal condition for Y_t

# Solve the BSDE by using time reversal, keep training phi network until convergence
phi_tr = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=32)
optimizer_tr = torch.optim.AdamW(phi_tr.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_tr = torch.optim.lr_scheduler.StepLR(optimizer_tr, step_size=1000, gamma=0.9)
phi_ad = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=32)
optimizer_ad = torch.optim.AdamW(phi_ad.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_ad = torch.optim.lr_scheduler.StepLR(optimizer_ad, step_size=1000, gamma=0.9)


for k in range(kf):
    # reinitialize phi network each iteration
    
    Y_b = time_reversal_bsde(H_x, g, phi_tr, T, dt, Y_T, W_b, [score_nn], X_b, nn_num=1)
    loss_history = train_phi_network(phi_tr, X_b, Y_b, time_grid, optimizer_tr, scheduler_tr, batch_size=64, iterations=5000)
    print(f"Phi Network Training Iteration {k+1}/{kf} completed.")

    Y_bn = non_adapted_adjoint(adjoint_dyn, X_forward, T, dt, Y_T).detach()  # shape (steps+1, N, n)
    loss_history_adjoint = train_phi_network(phi_ad, X_forward, Y_bn, time_grid, optimizer_ad, scheduler_ad, batch_size=64, iterations=5000)
    print(f"Phi Network Training with Adjoint Iteration {k+1}/{kf} completed.")


torch.save(phi_tr.state_dict(), f'network/tr_phi_network_ip_bsde_nl{noise_level}.pth')
torch.save(phi_ad.state_dict(), f'network/ad_phi_network_ip_bsde_nl{noise_level}.pth')




