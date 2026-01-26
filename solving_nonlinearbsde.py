import torch
import numpy as np

from utils import train_score_network, generate_initial_data, rollout, time_reversal, noise, time_reversal_bsde, train_phi_network
from network import ScoreNetwork, PhiNetwork

# Parameters
T = 4.0  # End time
n = 2    # Dimension of state space
m = 1    # Dimension of Brownian motion
N = 10000 # Number of training samples
INITIAL_DIST = 'Gaussian'  # Initial distribution type ('Gaussian', 'Bimodal', or 'Multimodal')
m_0 = torch.tensor([0.0, 0.0])  # Mean of initial distribution
sigma_0 = torch.eye(n) * 1  # Covariance of initial distribution
dt = 0.05  # Time step size
steps = int(T/dt)  # Number of time steps
noise_level = 0.5  # Noise level in the SDE
kf = 20 # iterations for phi
exp_num = 10000

# Generate initial data
X_0 = generate_initial_data(INITIAL_DIST, m_0, sigma_0, N, shift=5)

# Nonlinear system dynamics
def f(x):
    """
    Drift function of X_t for a nonlinear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
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
    Terminal cost function for a linear system.
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
    Gradient of the terminal cost function for a linear system.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
    Returns:
        torch.Tensor: Gradient of the terminal cost. Shape (N, n)
    """
    # Q_f = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # return x @ Q_f
    dlf1 = torch.sin(x[:,0])
    dlf2 = x[:,1]
    return torch.stack((dlf1, dlf2), dim=1)

# time_grid = torch.arange(0, steps+1) * dt

W_f = torch.zeros((steps+1, N, m))# forward noise
# W_b = torch.zeros((steps+1, N, m))# backward noise
for noise_step in range(steps+1):
    W_f[noise_step, :, :] = noise(dt, N, m)
    # W_b[noise_step, :, :] = noise(dt, N, m)

# # Generate training data by rolling out the SDE
# X_forward = rollout(f, g, T, dt, X_0, W_f)

# score_nn = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=64, num_blocks=3)
# score_optimizer = torch.optim.AdamW(score_nn.parameters(), lr=1e-3, weight_decay=1e-4)
# score_scheduler = torch.optim.lr_scheduler.StepLR(score_optimizer, step_size=500, gamma=0.9)

# score_loss_history = train_score_network(score_nn, X_forward, time_grid, g, noise_level, score_optimizer, score_scheduler, batch_size=64, iterations=10000)
# print("Score Network Training completed.")
# N = 1000 # Reduce N for BSDE solving to save memory
# X_forward = X_forward[:, :N, :]
# W_f = W_f[:, :N, :]
# W_b = W_b[:, :N, :]


def H_x(x, y, z):
    """
    Partial derivative of Hamiltonian respect to the x for Y_t in the BSDE.
    Args:
        x (torch.Tensor): State vector. Shape (N, n)
        y (torch.Tensor): Costate vector. Shape (N, n)
        z (torch.Tensor): Second order term in BSDE. Shape (N, n, m)
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
    # lag1 = y[:, 1] * torch.cos(x[:, 0])  # shape (N,)
    # lag2 = y[:, 0] - 0.01 * y[:, 1]  # shape (N,)
    # lag_term = torch.stack((lag1, lag2), dim=1)  # shape (N, n)
    lag_term_mat = torch.zeros((x.shape[0], n, n))  # shape (N, n, n)
    lag_term_mat[:, 0, 1] = torch.cos(x[:, 0]) # shape (N,)
    lag_term_mat[:, 1, 0] = 1.0
    lag_term_mat[:, 1, 1] = -0.01

    # trace term Tr(z g_x)
    trace_term = torch.zeros_like(x)  # shape (N, n)
    # return cost_term + lag_term + trace_term
    return cost_term, lag_term_mat, trace_term

# X_T = X_forward[-1, :, :]  # Terminal state
# X_b = time_reversal(f, g, T, dt, X_T, W_b, [score_nn], nn_num=1)
# Y_T = partial_lf(X_T)  # Terminal condition for Y_t

# Solve the BSDE by using time reversal, keep training phi network until convergence
# phi_net = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=32)
# optimizer = torch.optim.AdamW(phi_net.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)


# for k in range(kf):
#     # reinitialize phi network each iteration
    
#     Y_b = time_reversal_bsde(H_x, g, phi_net, T, dt, Y_T, W_b, [score_nn], X_b, nn_num=1)
#     phi_net = ScoreNetwork(input_dim=n+1, out_dim=n, hidden_dim=32)
#     optimizer = torch.optim.AdamW(phi_net.parameters(), lr=1e-3, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)
#     loss_history = train_phi_network(phi_net, X_b, Y_b, time_grid, optimizer, scheduler, batch_size=64, iterations=4000)
#     print(f"Phi Network Training Iteration {k+1}/{kf} completed.")


# torch.save(phi_net.state_dict(), f'data/phi_network_ip_bsde_nl{noise_level}.pth')




### TODO: do this part N times and average the gradient ###
# Solve the question by autograd on theta, you can comment this part if not needed
# grad_thetas = []
# # thetas = []
shift = torch.zeros((N, n))
shift[:,0] = 0.0
shift[:,1] = 0.0
theta = torch.randn((N, n), requires_grad=True) + shift
# for exp_i in range(exp_num):

#     # generate Brownian motion increments
#     W_f = torch.zeros((steps+1, N, m))# forward noise
#     for noise_step in range(steps+1):
#         W_f[noise_step, :, :] = noise(dt, N, m)
    
#     torch.autograd.set_detect_anomaly(True)
#     loss = lf(rollout(f, g, T, dt, theta, W_f)[-1, :, :]).sum(dim=0)
#     # ad_loss_history = []
#     # bsde_loss_history = []

#     grad_theta = torch.autograd.grad(loss, theta)[0]
#     grad_thetas.append(grad_theta.detach().clone())
    # thetas.append(theta.detach().clone())
optimizer = torch.optim.Adam([theta], lr=1e-3)
for step in range(1000):
    optimizer.zero_grad()
    loss = 0.0
    loss = lf(rollout(f, g, T, dt, theta, W_f)[-1]).mean(dim=0)
    loss.backward()
    optimizer.step()
    # print gradients in last step
    # if step == 0:
    #     last_grad = theta.grad.detach().clone()  # safe copy of gradient
    #     print("Final gradient:", last_grad)
    
    # if step % 100 == 0 or step == 999:
    #     print(f"Step {step}, Loss: {loss.item()/N}")


    # # print gradients in last step
    # if step == 0:
    #     last_grad = theta.grad.detach().clone()  # safe copy of gradient
    #     print("Final gradient:", last_grad)
    
    # if step % 100 == 0 or step == 999:
    #     print(f"Step {step}, Loss: {loss.item()/N}")

# torch.save(grad_thetas, f'data/initial_theta_grad_IP_nl{noise_level}_exp{exp_num}.pth')
# torch.save(theta, f'data/initial_theta_value_IP_nl{noise_level}_exp{exp_num}.pth')



### We can compare the loss for this soc problem to see the convergence speed ###
