import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from utils import rollout, batched_jacobian, non_adapted_adjoint, train_ut_network
from network import ScoreNetwork

# Parameters
T = 1.0  # End time
n = 1    # Dimension of state space
m = 1    # Dimension of Brownian motion
N = 10000 # Number of training samples
dt = 0.02  # Time step size
steps = int(T/dt)  # Number of time steps
noise_level = 2  # Noise level in the SDE
kf = 30 # iterations for whole procedure
u_iter = 10 # iterations for training u network at each iteration of the whole procedure
opt_iter = 1000 # optimization iterations for training u network at each iteration of the whole procedure
temperature_schedule = lambda k: 1.0  # constant temperature schedule 

# load pretrained score network
score_nn = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=32, num_blocks=2)
score_nn.load_state_dict(torch.load(f'network/toy_score_network_timesteps{steps}_v2.pth'))

# temp_score_nn = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=64, num_blocks=3)
# temp_score_nn.load_state_dict(torch.load(f'network/toy_score_network_timesteps{steps}_v2.pth'))

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
        # u_t (ScoreNetwork): Neural network for feedback control law --- IGNORE ---
        u_t (torch.Tensor): control inputs at each time step for each sample. Shape (steps+1, N, m)
    Returns:
        torch.Tensor: Drift vector. Shape (N, n)
    """
    a = 2
    u = u_t(x, t.repeat(x.shape[0], 1)) if u_t is not None else torch.zeros(m)  # shape (N, m)
    gu = torch.einsum('nij,nj->ni', g(x), u)  if u_t is not None else torch.zeros_like(x) # shape (N, n)
    # u = u_t[int(t/dt)] if u_t is not None else torch.zeros((N, m))  # shape (N, m) make sure use u at the right time step
    # gu = torch.einsum('nij,nj->ni', g(x), u) # shape (N, n)
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

def adjoint_dyn(x, t):
    """
    Dynamics for the non-adapted adjoint process.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
        y (torch.Tensor): Costate vector. Shape (N, n)
        z (torch.Tensor): Second order term in BSDE. Shape (N, n, m)
        t (torch.Tensor): Current time. Shape (1,)
        u (torch.Tensor): Open-loop control inputs at each time step. Shape (steps, m)
    Returns:
        cost_term (torch.Tensor): Running cost term. Shape (N, n)
        lag_term_mat (torch.Tensor): Lagrange term y^T f_x. Shape (N, n, n)
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
    return cost_term, lag_term_mat
    


time_grid = torch.arange(0, steps+1) * dt

# Initialize initial distribution parameters and open-loop control inputs
mu = torch.zeros(n, requires_grad=True)  # Mean of initial distribution
Q = torch.tensor([[2.0]], requires_grad=True)  # Sigma = Q Q^T for initial distribution
ut = ScoreNetwork(input_dim=n+1, out_dim=m, hidden_dim=64, num_blocks=4)
ut.load_state_dict(torch.load(f'network/finetune_admatching_ut_timesteps{steps}_utiter{10}_optiter{1000}_iteration{30}_temperature{3.0}_initialQ2_updateQmu6times_lamony.pth'))



for k in range(kf):
    temperature = temperature_schedule(k)
    print(f"Starting iteration {k+1}/{kf} with temperature {temperature:.2f}")
    # Generate forward trajectories
    X_0 = torch.randn(N, n) * Q + mu  # shape (N, n)
    W_f = torch.randn(steps + 1, N, m) * torch.sqrt(torch.tensor(dt)) # forward noise
    ut_opt = torch.optim.AdamW(ut.parameters(), lr=1e-5, weight_decay=1e-4)
    ut_scheduler = torch.optim.lr_scheduler.StepLR(ut_opt, step_size=500, gamma=0.9)
    for u_i in range(u_iter):
        print(f"Total iteration {k+1}/{kf} | UT training iteration {u_i+1}/{u_iter}")
        ut.eval()  # set UT network to evaluation mode for generating trajectories
        X_f = rollout(f, g, T, dt, X_0, W_f, u_t=ut).detach()  # shape (steps+1, N, n)
        X_T = X_f[-1, :, :] # shape (N, n)
        # Generate non-adapted adjoint trajectories
        Y_T = partial_lf(X_T)  # shape (N, n)
        Y_b = non_adapted_adjoint(adjoint_dyn, X_f, T, dt, Y_T).detach()  # shape (steps+1, N, n)
        ut.train()  # set UT network to training mode for training
        ut_train_loss = train_ut_network(ut, X_f, Y_b, time_grid, g, temperature, ut_opt, ut_scheduler, batch_size=64, iterations=1500)
    

    if k in [4, 9, 14, 19, 24, 29]: # update initial distribution every 5 iterations
        mu_opt = torch.optim.AdamW([mu], lr=3e-3, weight_decay=1e-4)
        Q_opt = torch.optim.AdamW([Q], lr=3e-3, weight_decay=1e-4)
        for opt_i in range(opt_iter):
            mu_opt.zero_grad(set_to_none=True)
            Q_opt.zero_grad(set_to_none=True)

            Xi = torch.randn(N, n)
            theta = (Xi * Q + mu).detach()  # shape (N, n)
            Y_0 = Y_b[0, :, :].detach()  # shape (N, n)
            mu_grad = Y_0.mean(dim=0)  # shape (n,)
            temp = (theta - mu.clone().detach()) @ torch.linalg.pinv(Q.clone().detach()) # shape (N, n)
            Q_grad = torch.einsum('nij,njk->nik', Y_0.unsqueeze(2), temp.unsqueeze(1)).mean(dim=0) # shape (n, n)
            mu_kl_grad = mu.clone().detach().reshape(-1)
            Q_kl_grad = Q.clone().detach() - torch.linalg.pinv(Q.clone().detach()).T
            mu.grad = (mu_grad + temperature * mu_kl_grad).detach()
            Q.grad = (Q_grad + temperature * Q_kl_grad).detach()
            mu_opt.step()
            Q_opt.step()
            if opt_i % 100 == 0:
                print(f"Total iteration {k+1}/{kf} | Optimization iteration {opt_i+1}/{opt_iter} | mu: {mu.clone().detach().numpy()} | Q: {Q.clone().detach().numpy()}")

torch.save(ut.state_dict(), f'network/finetune_admatching_ut_timesteps{steps}_utiter{u_iter}_optiter{opt_iter}_iteration{kf}_temperature{temperature}_initialQ2_updateQmu6times_lamony.pth')
torch.save(mu, f'network/finetune_admatching_mu_timesteps{steps}_utiter{u_iter}_optiter{opt_iter}_iteration{kf}_temperature{temperature}_initialQ2_updateQmu6times_lamony.pth')
torch.save(Q, f'network/finetune_admatching_Q_timesteps{steps}_utiter{u_iter}_optiter{opt_iter}_iteration{kf}_temperature{temperature}_initialQ2_updateQmu6times_lamony.pth')

# torch.save(ut, f'network/finetune_ut_timesteps{steps}_iteration{kf}_phiiter{phi_iter}_temperature{temperature}.pth')
