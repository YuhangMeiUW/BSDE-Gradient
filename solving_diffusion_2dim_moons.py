import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import train_score_network, generate_initial_data, rollout, time_reversal, noise, time_reversal_bsde, train_phi_network, batched_jacobian
from network import ScoreNetwork, PhiNetwork

# Parameters
T = 1.0  # End time
n = 2    # Dimension of state space
m = 2    # Dimension of Brownian motion
N = 10000 # Number of training samples
INITIAL_DIST = 'Gaussian'  # Initial distribution type ('Gaussian', 'Bimodal', or 'Multimodal')
m_0 = torch.tensor([0.0, 0.0])  # Mean of initial distribution
initial_var = 1.0
sigma_0 = torch.eye(n) * initial_var  # Covariance of initial distribution
dt = 0.02  # Time step size
steps = int(T/dt)  # Number of time steps
noise_level = 2  # Noise level in the SDE
kf = 10 # iterations for phi
exp_num = 10000

# Generate initial data
X_0 = generate_initial_data(INITIAL_DIST, m_0, sigma_0, N, shift=0.0) 
score_nn = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=64, num_blocks=4)
score_nn.load_state_dict(torch.load(f'network/2dim_4gaussian_score_network_timesteps{steps}.pth'))

# Nonlinear system dynamics
def f(x, t):
    """
    Drift function of X_t for a nonlinear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
        t (torch.Tensor): torch.tensor of shape (1,): Current time
    Returns:
        torch.Tensor: Drift vector. Shape (N, n)
    """
    a = 2
    df = a * x + noise_level**2 * score_nn(x, (T - t).repeat(x.shape[0], 1))
    return df


def g(x):
    """
    Diffusion function of X_t for a nonlinear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Diffusion matrix. Shape (N, n, m)
    """
    B = torch.eye(n) * noise_level
    B = B.unsqueeze(0).repeat(x.shape[0], 1, 1)
    return B

def lf(x):
    """
    Terminal cost function for moons dataset.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Terminal cost. Shape (N,)
    """
    upper_center = torch.tensor([4.0, 0.0]).repeat(x.shape[0], 1)
    return 0.5 *(x - upper_center)**2

def partial_lf(x):
    """
    Gradient of the terminal cost function for a linear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Gradient of the terminal cost. Shape (N, n)
    """
    # Q_f = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # return x @ Q_f
    upper_center = torch.tensor([4.0, 0.0]).repeat(x.shape[0], 1)
    return x - upper_center

# time_grid = torch.arange(0, steps+1) * dt

def H_x(x, y, z, t):
    ##TODO: maybe change t to T-t##
    """
    Partial derivative of Hamiltonian respect to the x for Y_t in the BSDE.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
        y (torch.Tensor): Costate vector. Shape (N, n)
        z (torch.Tensor): Second order term in BSDE. Shape (N, n, m)
        t (torch.Tensor): Current time. Shape (1,)
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
    a = 2
    lag_term_mat = torch.ones((x.shape[0], n, n)) * a  # shape (N, n, n)
    x.requires_grad_(True)
    score = score_nn(x, (T - t).repeat(x.shape[0], 1))  # shape (N, n)
    grad_score = batched_jacobian(score, x).detach()  # shape (N, n, n)
    lag_term_mat = lag_term_mat + noise_level**2 * grad_score  # shape (N, n, n)

    # trace term Tr(z g_x)
    trace_term = torch.zeros_like(x)  # shape (N, n)
    # return cost_term + lag_term + trace_term
    return cost_term, lag_term_mat, trace_term

def special_f(x, t=None):
    """
    Drift function of X_t for a nonlinear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Drift vector. Shape (N, n)
    """
    a = 2
    return -a * x

time_grid = torch.arange(0, steps+1) * dt

W_f = torch.randn(steps + 1, N, m) * torch.sqrt(torch.tensor(dt))
W_b = torch.randn(steps + 1, N, m) * torch.sqrt(torch.tensor(dt))

X_forward = rollout(f, g, T, dt, X_0, W_f)
# plt.figure()
# plt.plot(X_forward[:, :1000, 0].detach().numpy(), color='blue', alpha=0.1)
# plt.show()


X_T = X_forward[-1, :, :].detach()  # Terminal state
X_b_rev = rollout(special_f, g, T, dt, X_T, W_b).detach()
X_b = X_b_rev.flip(dims=[0])
Y_T = partial_lf(X_T)  # Terminal condition for Y_t
# plt.figure()
# plt.plot(X_b_rev[:, :1000, 0].detach().numpy(), color='blue', alpha=0.1)
# plt.show()
# plt.figure()
# plt.plot(X_b[:, :1000, 0].detach().numpy(), color='blue', alpha=0.1)
# plt.show()

# Solve the BSDE by using time reversal, keep training phi network until convergence
phi_net = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=64, num_blocks=4)
optimizer = torch.optim.AdamW(phi_net.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)


for k in range(kf):
    # reinitialize phi network each iteration
    
    Y_b = time_reversal_bsde(H_x, g, phi_net, T, dt, Y_T, W_b, [score_nn], X_b, nn_num=1)
    phi_net = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=64, num_blocks=4)
    optimizer = torch.optim.AdamW(phi_net.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)
    loss_history = train_phi_network(phi_net, X_b, Y_b, time_grid, optimizer, scheduler, batch_size=64, iterations=8000)
    print(f"Phi Network Training Iteration {k+1}/{kf} completed.")


torch.save(phi_net.state_dict(), f'network/phi_network_diff_bsde_timesteps{steps}_kf{kf}_4gaussian.pth')




# Solve the question by autograd on theta, you can comment this part if not needed
# N = 10000
# exp_num = 10000
# grad_thetas = []
# shift = torch.zeros((N, n))
# shift[:,0] = 0.0
# shift[:,1] = 0.0
# theta = torch.randn((N, n), requires_grad=True) + shift
# for exp_i in range(exp_num):

#     # generate Brownian motion increments
#     # W_f = torch.zeros((steps+1, N, m))# forward noise
#     # for noise_step in range(steps+1):
#     #     W_f[noise_step, :, :] = noise(dt, N, m)
#     W_f = torch.randn(steps + 1, N, m) * torch.sqrt(torch.tensor(dt))
    
#     torch.autograd.set_detect_anomaly(True)
#     loss = lf(rollout(f, g, T, dt, theta, W_f)[-1, :, :]).sum(dim=0)

#     grad_theta = torch.autograd.grad(loss, theta)[0]
#     grad_thetas.append(grad_theta.detach().clone())
#     if (exp_i+1) % 500 == 0:
#         print(f"Experiment {exp_i+1}/{exp_num} completed.")

# grad_thetas = torch.stack(grad_thetas, dim=0) # (exp_num, N, n)
# torch.save(grad_thetas, f'data/initial_theta_grad_diff_nl{noise_level}_exp{exp_num}.pth')
# torch.save(theta, f'data/initial_theta_value_diff_nl{noise_level}_exp{exp_num}.pth')