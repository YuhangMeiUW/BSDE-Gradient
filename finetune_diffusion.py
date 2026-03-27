import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from utils import train_score_network, rollout, time_reversal, time_reversal_bsde, train_phi_network, batched_jacobian
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
opt_iter = 1000
phi_iter = 10
temperature_schedule = lambda k: 3.0  # constant temperature schedule 

# load pretrained score network
score_nn = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=32, num_blocks=2)
score_nn.load_state_dict(torch.load(f'network/toy_score_network_timesteps{steps}_v2.pth'))

temp_score_nn = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=64, num_blocks=3)

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
    return cost_term, lag_term_mat, trace_term

def special_f(x, t=None, u_t=None):
    """
    Drift function of X_t for a nonlinear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
        # u_t (ScoreNetwork): Neural network for feedback control law --- IGNORE ---
        u_t (torch.Tensor): control inputs at each time step for each sample. Shape (steps+1, N, m)
        t (torch.Tensor): Current time. Shape (1,)
    Returns:
        torch.Tensor: Drift vector. Shape (N, n)
    """
    a = 2
    u = u_t(x, (T-t).repeat(x.shape[0], 1)) if u_t is not None else torch.zeros(m)  # shape (N, m)
    gu = torch.einsum('nij,nj->ni', g(x), u)  if u_t is not None else torch.zeros_like(x) # shape (N, n)
    return -a * x - gu

def original_pdf(x):
    m1 = 3.0
    m2 = -3.0
    sigma = 1.0
    p1 = torch.exp(-0.5 * ((x - m1) / sigma)**2) / (sigma * torch.sqrt(torch.tensor(2.0) * torch.pi))
    p2 = torch.exp(-0.5 * ((x - m2) / sigma)**2) / (sigma * torch.sqrt(torch.tensor(2.0) * torch.pi))
    return 0.5 * (p1 + p2)


def tilted_pdf(x, temperature):
    return original_pdf(x)*torch.exp(-(x-3.0)**2/(2*temperature))

class UNetFromPhi(nn.Module):
    def __init__(self, phi_net, g_fn, temperature):
        super().__init__()
        self.phi_net = phi_net      # nn.Module
        self.g_fn = g_fn            # function: (N,n) -> (N,n,m)
        self.temperature = temperature

    def forward(self, x, t):
        # x: (N,n), t: (N,1)
        y = self.phi_net(x, t)          # (N,n)
        gx = self.g_fn(x)               # (N,n,m)
        u = -torch.einsum('nim,ni->nm', gx, y) / self.temperature  # (N,m) = -g(x)^T y / temperature
        return u

time_grid = torch.arange(0, steps+1) * dt

# Initialize initial distribution parameters and open-loop control inputs
mu = torch.zeros(n, requires_grad=True)  # Mean of initial distribution
Q = torch.tensor([[2.0]], requires_grad=True)  # Sigma = Q Q^T for initial distribution
phi_net = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=64, num_blocks=4)
phi_net.load_state_dict(torch.load(f'network/finetune_phi_network_timesteps{steps}_iteration{30}_phiiter{10}_optiter{1000}_temperature{6.0}_initialQ2_updateQmu6times_2score.pth'))
ut = UNetFromPhi(phi_net, g, 6.0)  # Initialize ut as None, which means no control at the beginning
mu_opt = torch.optim.AdamW([mu], lr=3e-3, weight_decay=1e-4)
Q_opt = torch.optim.AdamW([Q], lr=3e-3, weight_decay=1e-4)

#### We don't need to retian score because phi PDE still holds for a wrong s(t,x). ####
for k in range(kf):
    print(f"Iteration {k+1}/{kf}")
    temperature = temperature_schedule(k)
    print(f"Current temperature: {temperature}")
    # Generate initial distribution samples
    theta = (torch.randn(N, n) @ Q + mu).detach()  # shape (N, n)
    # Generate training data by rolling out the SDE
    W_f = torch.randn(steps + 1, N, m) * torch.sqrt(torch.tensor(dt)) # forward noise
    W_b = torch.randn(steps + 1, N, m) * torch.sqrt(torch.tensor(dt)) # backward noise
    X_f = rollout(f, g, T, dt, theta, W_f, u_t=ut)  # shape (steps+1, N, n)
    X_T = X_f[-1,:,:].detach()  # shape (N, n)
    # Train score network with the generated data
    temp_score_nn.train()
    score_optimizer = torch.optim.AdamW(temp_score_nn.parameters(), lr=1e-3, weight_decay=1e-4)
    score_scheduler = torch.optim.lr_scheduler.StepLR(score_optimizer, step_size=500, gamma=0.9)
    score_loss_history = train_score_network(temp_score_nn, X_f.detach(), time_grid, g, 1, score_optimizer, score_scheduler, batch_size=64, iterations=10000)
    temp_score_nn.eval()
    X_b = time_reversal(f, g, T, dt, X_T, W_b, [temp_score_nn], nn_num=1).detach()
    Y_T = partial_lf(X_T)  # shape (N, n)
    

    # Train phi network
    if k == 0:
        phi_net = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=64, num_blocks=4)
        phi_net.load_state_dict(torch.load(f'network/finetune_phi_network_timesteps{steps}_iteration{30}_phiiter{5}_optiter{1000}_temperature{8.0}_initialQ2_updateQmu6times_2score.pth'))
    optimizer_phi = torch.optim.AdamW(phi_net.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler_phi = torch.optim.lr_scheduler.StepLR(optimizer_phi, step_size=500, gamma=0.9)

    for phi_i in range(phi_iter):
        print(f"Total iteration {k+1}/{kf} | Phi training iteration {phi_i+1}/{phi_iter}")
        phi_net.eval()
        Y_b = time_reversal_bsde(H_x, g, phi_net, T, dt, Y_T, W_b, [temp_score_nn], X_b, nn_num=1)
        phi_net.train()
        phi_loss_history = train_phi_network(phi_net, X_b, Y_b, time_grid, optimizer_phi, scheduler_phi, batch_size=64, iterations=1500)
    
    
    # Update initial distribution parameters and feedback control law
    
    ut = UNetFromPhi(phi_net, g, temperature).eval()
    if k in [4, 9, 14, 19, 24, 29]:  # Save the phi network at certain iterations for updating initial distribution( phi only valid at the support of the empirical distribution of X_b)
        mu_opt = torch.optim.AdamW([mu], lr=3e-3, weight_decay=1e-4)
        Q_opt = torch.optim.AdamW([Q], lr=3e-3, weight_decay=1e-4)
        for opt_i in range(opt_iter):
            # print(f"Total iteration {k+1}/{kf} | Optimization iteration {opt_i+1}/{opt_iter}")
            mu_opt.zero_grad(set_to_none=True)
            Q_opt.zero_grad(set_to_none=True)

            # Compute mu Q gradients with the trained phi network
            Xi = torch.randn(N, n)
            theta = (Xi @ Q + mu).detach()  # shape (N, n)
            phi_0 = phi_net(theta, torch.tensor(0.0).repeat(theta.shape[0], 1)).detach()  # shape (N, n)
            mu_grad = phi_0.mean(dim=0)  # shape (n,)
            temp = (theta - mu.clone().detach()) @ torch.linalg.pinv(Q.clone().detach()) # shape (N, n)
            Q_grad = torch.einsum('nij,njk->nik', phi_0.unsqueeze(2), temp.unsqueeze(1)).mean(dim=0)  # shape (n, n)
            mu_kl_grad = mu.clone().detach().reshape(-1)
            Q_kl_grad = Q.clone().detach() - torch.linalg.pinv(Q.clone().detach()).T
            mu.grad = (mu_grad + temperature * mu_kl_grad).detach()
            Q.grad = (Q_grad + temperature * Q_kl_grad).detach()

            mu_opt.step()
            Q_opt.step()
            if opt_i % 100 == 0:
                print(f"Total iteration {k+1}/{kf} | Optimization iteration {opt_i+1}/{opt_iter} | mu: {mu.clone().detach().numpy()} | Q: {Q.clone().detach().numpy()}")
             

torch.save(phi_net.state_dict(), f'network/finetune_phi_network_timesteps{steps}_iteration{kf}_phiiter{phi_iter}_optiter{opt_iter}_temperature{temperature}_initialQ2_updateQmu6times_2score.pth')
torch.save(mu, f'network/finetune_mu_timesteps{steps}_iteration{kf}_phiiter{phi_iter}_optiter{opt_iter}_temperature{temperature}_initialQ2_updateQmu6times_2score.pth')
torch.save(Q, f'network/finetune_Q_timesteps{steps}_iteration{kf}_phiiter{phi_iter}_optiter{opt_iter}_temperature{temperature}_initialQ2_updateQmu6times_2score.pth')


