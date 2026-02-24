import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import train_score_network, generate_initial_data, rollout, time_reversal, noise, time_reversal_bsde, train_phi_network, batched_jacobian
from network import ScoreNetwork, PhiNetwork

# Parameters
T = 1.0  # End time
n = 1    # Dimension of state space
m = 1    # Dimension of Brownian motion
N = 10000 # Number of training samples
dt = 0.02  # Time step size
steps = int(T/dt)  # Number of time steps
noise_level = 2  # Noise level in the SDE
kf = 12 # iterations for whole procedure
opt_iter = 5000
phi_iter = 6
# u_iter = 10000

# load pretrained score network
score_nn = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=32, num_blocks=2)
score_nn.load_state_dict(torch.load(f'network/toy_score_network_timesteps{steps}.pth'))

# Nonlinear system dynamics
def g(x):
    """
    Diffusion function of X_t for a nonlinear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Diffusion matrix. Shape (N, n, m)
    """
    B = torch.tensor([[1.0]]) * noise_level
    B = B.unsqueeze(0).repeat(x.shape[0], 1, 1)
    return B

def f(x, t, u_t=None):
    """
    Drift function of X_t for a nonlinear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
        t (torch.Tensor): torch.tensor of shape (1,): Current time
        score_nn (ScoreNetwork): Neural network for score function
        u_t (torch.Tensor): Open-loop control inputs at each time step. Shape (steps, m)
    Returns:
        torch.Tensor: Drift vector. Shape (N, n)
    """
    a = 2
    u = u_t[int(t/dt)] if u_t is not None else torch.zeros(m)  # shape (m,) make sure use u at the right time step
    gu = (g(x) @ u.unsqueeze(1)).squeeze(-1)  # shape (N, n)
    df = a * x + noise_level**2 * score_nn(x, (T - t).repeat(x.shape[0], 1)) + gu
    return df


def lf(x):
    """
    Terminal cost function for a linear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Terminal cost. Shape (N,)
    """
    return 0.5 *(x - 3.0)**2

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
    return x - 3.0

# time_grid = torch.arange(0, steps+1) * dt

def H_x(x, y, z, t, u=None):
    """
    Partial derivative of Hamiltonian respect to the x for Y_t in the BSDE.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
        y (torch.Tensor): Costate vector. Shape (N, n)
        z (torch.Tensor): Second order term in BSDE. Shape (N, n, m)
        t (torch.Tensor): Current time. Shape (1,)
        u (torch.Tensor): Open-loop control inputs at each time step. Shape (steps, m)
    Returns:
        # torch.Tensor: Drift vector. Shape (N, n)
        cost_term (torch.Tensor): Running cost term. Shape (N, n)
        lag_term_mat (torch.Tensor): Lagrange term y^T f_x. Shape (N, n, n)
        trace_term (torch.Tensor): Trace term Tr(z g_x). Shape (N, n)
    """
    # running cost term
    # l(x,u) = 0.5 u^T R u, R = I
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

def special_f(x, t=None, u_t=None):
    """
    Drift function of X_t for a nonlinear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
        u_t (torch.Tensor): Open-loop control inputs at each time step. Shape (steps, m) make sure use u at the right time step
        t (torch.Tensor): Current time. Shape (1,)
    Returns:
        torch.Tensor: Drift vector. Shape (N, n)
    """
    a = 2
    u = u_t[int((T-t)/dt)] if u_t is not None else torch.zeros(m)  # shape (m,) make sure use u at the right time step
    gu = (g(x) @ u.unsqueeze(1)).squeeze(-1)  # shape (N, n)
    return -a * x - gu

time_grid = torch.arange(0, steps+1) * dt

# Initialize initial distribution parameters and open-loop control inputs
mu = torch.zeros(n, requires_grad=True)  # Mean of initial distribution
Q = torch.eye(n, requires_grad=True)  # Sigma = Q Q^T for initial distribution
ut = torch.zeros(steps+1, m, requires_grad=True)  # Open-loop control inputs at each time step



#### We don't need to retian score because phi PDE still holds for a wrong s(t,x). ####
for k in range(kf):
    print(f"Iteration {k+1}/{kf}")
    # Generate initial distribution samples
    theta = (torch.randn(N, n) @ Q + mu).detach()  # shape (N, n)
    # Generate training data by rolling out the SDE
    W_f = torch.randn(steps + 1, N, m) * torch.sqrt(torch.tensor(dt)) # forward noise
    W_b = torch.randn(steps + 1, N, m) * torch.sqrt(torch.tensor(dt)) # backward noise
    X_f = rollout(f, g, T, dt, theta, W_f, u_t=ut)  # shape (steps+1, N, n)
    # plt.figure()
    # plt.plot(X_f[:, :1000, 0].detach().numpy(), color='blue', alpha=0.1)
    # plt.show()
    X_T = X_f[-1,:,:].detach()  # shape (N, n)
    X_b = rollout(special_f, g, T, dt, X_T, W_b, u_t=ut).detach().flip(dims=[0])
    Y_T = partial_lf(X_T)  # shape (N, n)
    # plt.figure()
    # plt.plot(X_b[:, :1000, 0].detach().numpy(), color='blue', alpha=0.1)
    # plt.show()

    # Train phi network
    phi_net = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=64, num_blocks=4)
    optimizer_phi = torch.optim.AdamW(phi_net.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler_phi = torch.optim.lr_scheduler.StepLR(optimizer_phi, step_size=2000, gamma=0.9)

    for phi_i in range(phi_iter):
        print(f"Total iteration {k+1}/{kf} | Phi training iteration {phi_i+1}/{phi_iter}")
        Y_b = time_reversal_bsde(H_x, g, phi_net, T, dt, Y_T, W_b, [score_nn], X_b, nn_num=1)
        phi_net = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=64, num_blocks=4)
        optimizer_phi = torch.optim.AdamW(phi_net.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler_phi = torch.optim.lr_scheduler.StepLR(optimizer_phi, step_size=2000, gamma=0.9)
        phi_loss_history = train_phi_network(phi_net, X_b, Y_b, time_grid, optimizer_phi, scheduler_phi, batch_size=64, iterations=5000)

    
    # Optimize initial distribution parameters and open-loop control inputs
    mu_opt = torch.optim.AdamW([mu], lr=1e-3, weight_decay=1e-4)
    Q_opt = torch.optim.AdamW([Q], lr=1e-3, weight_decay=1e-4)
    ut_opt = torch.optim.AdamW([ut], lr=1e-3, weight_decay=1e-4)
    for opt_i in range(opt_iter):
        
        mu_opt.zero_grad(set_to_none=True)
        Q_opt.zero_grad(set_to_none=True)
        

        # Compute mu Q gradients with the trained phi network
        theta = (torch.randn(N, n) @ Q + mu).detach()  # shape (N, n)
        phi_0 = phi_net(theta, torch.tensor(0.0).repeat(theta.shape[0], 1)).detach()  # shape (N, n)
        mu_grad = phi_0.mean(dim=0)  # shape (n,)
        temp = (theta - mu.clone().detach()) @ torch.linalg.pinv(Q.clone().detach() + 1e-6 * torch.eye(n))
        Q_grad = torch.einsum('nij,njk->nik', phi_0.unsqueeze(2), temp.unsqueeze(1)).mean(dim=0)  # shape (n, n)
        mu_kl_grad = mu.clone().detach().reshape(-1)
        Q_kl_grad = Q.clone().detach() - torch.linalg.pinv(Q.clone().detach() + 1e-6 * torch.eye(n)).T
        mu.grad = (mu_grad + mu_kl_grad).detach()
        Q.grad = (Q_grad + Q_kl_grad).detach()
        mu_opt.step()
        Q_opt.step()
        if (opt_i+1) % 1000 == 0:
            print(f"Total iteration {k+1}/{kf} | Optimization iteration {opt_i+1}/{opt_iter} | mu: {mu.detach().numpy()} | Q: {Q.detach().numpy()}")
    
    theta = (torch.randn(N, n) @ Q + mu).detach()  # shape (N, n)
    W_f = torch.randn(steps + 1, N, m) * torch.sqrt(torch.tensor(dt)) # forward noise
    X_f = rollout(f, g, T, dt, theta, W_f, u_t=ut).detach()  # shape (steps+1, N, n)
    X_flat = X_f.reshape(-1, n)  # ((T_steps*N), n)
    t_flat = time_grid.repeat_interleave(N, dim=0).reshape(-1, 1)  # ((T_steps*N), 1)
    Y_flat = phi_net(X_flat, t_flat).detach() # ((T_steps*N), n)
    Y_f = Y_flat.reshape(steps+1, N, n) # (T_steps, N, n)
    gX_flat = g(X_flat).detach() # ((T_steps*N), n, m)
    gX = gX_flat.reshape(steps+1, N, n, m)
    gTy = torch.einsum('tnij,tnj->tni', gX.transpose(-2, -1), Y_f).detach() # (T_steps, N, m)
    
    for opt_i in range(opt_iter):
        ut_opt.zero_grad(set_to_none=True)
        # Compute ut gradient with the trained phi network and Hamiltonian
        # H = l(x,u) + y^T f(x,u) + Tr(z^T g(x))
        # H_u = u + g^T y
        
        ut_grad = ut.clone().detach() + gTy.mean(dim=1)  # shape (m,)
        ut.grad = ut_grad.detach()
        ut_opt.step()
        if (opt_i+1) % 1000 == 0:
            print(f"Total iteration {k+1}/{kf} | Control optimization iteration {opt_i+1}/{opt_iter}")
    

torch.save(phi_net.state_dict(), f'network/finetune_phi_network_timesteps{steps}_iteration{kf}.pth')
torch.save(mu, f'network/finetune_mu_timesteps{steps}_iteration{kf}.pth')
torch.save(Q, f'network/finetune_Q_timesteps{steps}_iteration{kf}.pth')
torch.save(ut, f'network/finetune_ut_timesteps{steps}_iteration{kf}.pth')
