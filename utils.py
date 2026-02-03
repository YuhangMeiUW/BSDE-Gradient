import random
import numpy as np
import torch
import torch.nn as nn
from torch.func import vmap, jacfwd, jacrev
from scipy.integrate import solve_ivp

def jacobian(y: torch.Tensor, x: torch.Tensor, device='cpu', need_higher_grad=True) -> torch.Tensor:
    """
    Compute the Jacobian of y with respect to x using autograd.
    Args:
        y: Output tensor (N x out_dim)
        x: Input tensor (N x input_dim)
        device: Device to perform the computation on (e.g., 'cpu' or 'cuda')
        need_higher_grad: If True, allows higher order gradients
    Returns:
        Jac: Jacobian tensor (out_dim x input_dim)
        Example: If y is of shape (3,) and x is of shape (2,), then Jac will be of shape (3, 2) [dy1/dx1, dy1/dx2; dy2/dx1, dy2/dx2; dy3/dx1, dy3/dx2]
    """
    (Jac,) = torch.autograd.grad(
        outputs=(y.flatten(),),
        inputs=(x,),
        grad_outputs=(torch.eye(torch.numel(y)).to(device),),
        create_graph=need_higher_grad,
        allow_unused=True,
        is_grads_batched=True
    )
    if Jac is None:
        Jac = torch.zeros(size=(y.shape + x.shape))
    else:
        Jac.reshape(shape=(y.shape + x.shape))
    return Jac

def batched_jacobian(batched_y:torch.Tensor,batched_x:torch.Tensor,device='cpu', need_higher_grad = True) -> torch.Tensor:
    """
    Compute the Jacobian of batched_y with respect to batched_x using autograd.
    Args:
        batched_y: Output tensor (N x y_shape)
        batched_x: Input tensor (N x x_shape)
        device: Device to perform the computation on (e.g., 'cpu' or 'cuda')
        need_higher_grad: If True, allows higher order gradients
    Returns:
        J: Jacobian tensor (y_shape x N x x_shape) 
    """
    sumed_y = batched_y.sum(dim = 0) # y_shape
    Batch_J = jacobian(sumed_y,batched_x,device, need_higher_grad) # y_shape x N x x_shape

    dims = list(range(Batch_J.dim()))
    dims[0],dims[sumed_y.dim()] = dims[sumed_y.dim()],dims[0]
    Batch_J = Batch_J.permute(dims = dims) # N x y_shape x x_shape
    return Batch_J



def train_score_network(score_net, X_train, time_grid, g, noise_level, optimizer, scheduler, batch_size=64, iterations=1000):
    """
    Train the Score Network.
    
    Args:
        score_net: Instance of ScoreNetwork
        X_train: Training state data (T x N x n)
        time_grid: Time grid (T x 1)
        g: Diffusion function g
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        batch_size: Batch size for training
        iterations: Number of training iterations
    Returns:
        loss_history: List of loss values during training
    """
    score_net.train()
    loss = 0
    loss_history = []
    steps, N, n = X_train.shape
    X_train = X_train.permute(1, 0, 2) # (N, T, n)
    t_batch_size = 16
    for i in range(iterations):

        batch_idx = random.sample(range(N), batch_size)
        X_batch = X_train[batch_idx, :, :]
        time_batch = time_grid.repeat(batch_size, 1).unsqueeze(-1)  # (batch_size, T, 1)
        time_idx = random.sample(range(steps), t_batch_size)
        # time_idx = np.random.geometric(0.2, size=t_batch_size)
        # time_idx = np.clip(time_idx, 0, steps-1)  # Ensure indices are within bounds
        X_batch = X_batch[:, time_idx, :]  # (batch_size, t_batch_size, n)
        time_batch = time_batch[:, time_idx, :]  # (batch_size, t_batch_size, 1)
        X_batch = X_batch.view(-1, n) # (batch_size * t_batch_size, n)
        X_batch.requires_grad = True  # Enable gradient computation
        time_batch = time_batch.view(-1, 1) # (batch_size * t_batch_size, 1)
        score_pred = score_net(X_batch, time_batch)  # (batch_size * T, out_dim)
        batch_norm = torch.einsum('tij,tjk->tik', score_pred.unsqueeze(1), score_pred.unsqueeze(2)).squeeze(-1) # (batch_size * T, 1)

        # batch_jac = batched_jacobian(score_pred, X_batch, device=X_batch.device)  # (batch_size * T, out_dim, n)
        # batch_trace_test = batch_jac.diagonal(offset=0, dim1=1, dim2=2).sum(dim=1, keepdim=True) # (batch_size * T, 1)
        batch_trace = torch.zeros(batch_size * t_batch_size)
        for j in range(n):
            batch_trace += torch.autograd.grad(
                score_pred[:, j].sum(),
                X_batch,
                create_graph=True,
                retain_graph=True
            )[0][:, j]
        batch_trace = batch_trace.unsqueeze(-1)  # (batch_size * T, 1)
        # print("Trace difference:", torch.abs(batch_trace - batch_trace_test).max().item())
        loss = (0.5 * batch_norm + batch_trace).mean()  # Average loss over the batch
        ### loss is E[0.5*||score||^2 + Tr(grad(score))]
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 500 == 0 or i == iterations - 1:
            print(f"Iteration {i}, Loss: {loss.item()}")
            loss_history.append(loss.item())

    return loss_history



def kernel(X, Y, sigma=1, key='RBF', d=2, c=1):
    """
    Compute the Gaussian RBF kernel between two sets of points.
    Args:
        X: Tensor of shape (N, n)
        Y: Tensor of shape (N, n)
        sigma: Bandwidth parameter for the kernel
        key: Type of kernel ('RBF', 'Polynomial', 'Linear')
        d: Degree for polynomial kernel
    Returns:
        Kernel matrix of shape (N, N)
    """
    if key == 'RBF':
        D = torch.cdist(X, Y, p=2)  # (N, M)
        return torch.exp(- (D ** 2) / (2 * sigma ** 2))
    elif key == 'Polynomial':
        return (X @ Y.T + c)**d
    elif key == 'Linear':
        return X @ Y.T
    else:
        raise ValueError("Unsupported kernel type")



def MMD(XY, XY_target, kernel, sigma=1/(2*1**2), key='RBF', d=2):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two distributions.
    Args:
        XY: Samples from the first distribution (N, n)
        XY_target: Samples from the second distribution (M, n)
        kernel: Kernel function to use
        sigma: Bandwidth parameter for the kernel
        key: Type of kernel ('RBF', 'Polynomial', 'Linear')
        d: Degree for polynomial kernel
    Returns:
        MMD value (scalar)
    """
    return torch.sqrt(kernel(XY, XY, sigma=sigma, key=key, d=d).mean() + kernel(XY_target, XY_target, sigma=sigma, key=key, d=d).mean() - 2*kernel(XY, XY_target, sigma=sigma, key=key, d=d).mean())

def generate_initial_data(dist_type, mean, cov, num_samples, shift=3, mode_num=4):
    if dist_type == 'Gaussian':
        dist = torch.distributions.MultivariateNormal(mean, cov)
        return dist.sample((num_samples,))
    elif dist_type == 'Bimodal':
        dist1 = torch.distributions.MultivariateNormal(mean - shift, cov)
        dist2 = torch.distributions.MultivariateNormal(mean + shift, cov)
        samples1 = dist1.sample((num_samples // 2,))
        samples2 = dist2.sample((num_samples - num_samples // 2,))
        return torch.cat([samples1, samples2], dim=0)
    elif dist_type == 'Multimodal':
        modes = [mean + i * shift for i in range(-mode_num//2, mode_num//2 + 1)]
        samples = []
        for mode in modes:
            dist = torch.distributions.MultivariateNormal(mode, cov)
            samples.append(dist.sample((num_samples // len(modes),)))
        return torch.cat(samples, dim=0)
    else:
        raise ValueError("Unsupported distribution type")
    

# def rollout(f, g, tf, dt, x0, W):
#     """
#     Simulate the SDE using Euler-Maruyama method.
#     Args:
#         f: Drift function
#         g: Diffusion function
#         tf: Final time
#         dt: Time step size
#         x0: Initial state (N, n)
#         W: Brownian motion increments (steps+1, N, m)
#     Returns:
#         X: State trajectory (steps+1, N, n)
#     """
#     steps = int(tf/dt)
#     N, n = x0.shape
#     X = torch.zeros((steps+1, N, n), dtype=x0.dtype, device=x0.device)
#     X[0,:,:] = x0
#     for i in range(steps):
#         X[i+1,:,:] = X[i,:,:] + f(X[i,:,:])*dt + torch.einsum('nij,njk->nik', g(X[i,:,:]), W[i,:,:].unsqueeze(-1)).squeeze(-1)
#     return X

def rollout(f, g, tf, dt, x0, W):
    """
    Simulate the SDE using Euler-Maruyama method.
    Args:
        f: Drift function
        g: Diffusion function
        tf: Final time
        dt: Time step size
        x0: Initial state (N, n)
        W: Brownian motion increments (steps+1, N, m)
    Returns:
        X: State trajectory (steps+1, N, n)
    """
    steps = int(tf / dt)

    x = x0  
    traj = [x0]  

    for i in range(steps):
        drift = f(x, torch.tensor(i * dt))  # (N, n)
        diff = torch.einsum('nij,njk->nik', g(x), W[i].unsqueeze(-1)).squeeze(-1)
        x = x + drift * dt + diff   # out-of-place update
        traj.append(x)

    X = torch.stack(traj, dim=0)
    return X

def zero_div_ggT(x):
    return torch.zeros(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)

def time_reversal(f, g, tf, dt, xT, W_b, score, nn_num=0, div_ggT=zero_div_ggT):
    """
    Perform time reversal of the SDE using the learned score function.
    Args:
        f: Drift function
        g: Diffusion function
        tf: Final time
        dt: Time step size
        xT: Terminal state (N, n)
        W_b: Backward Brownian motion increments (steps+1, N, m)
        score: List of Trained neural networks or a callable function for score
    Returns:
        X_b: Reversed state trajectory (steps+1, N, n)
    """
    if nn_num == 0:
        score_net = score
    else:
        # nn_num >=1 cut time into nn_num segments
        score_nets = score
        # segment_times = [i * tf / nn_num for i in range(nn_num + 1)]
    steps = int(tf/dt)
    N, n = xT.shape
    X_b = torch.zeros((steps+1, N, n), dtype=xT.dtype, device=xT.device)
    X_b[-1,:,:] = xT
    time_grid = torch.linspace(0, tf, steps+1).unsqueeze(-1).to(xT.device)  # (steps+1, 1)
    for i in range(steps-1, -1, -1):
        if nn_num > 0:
            # print(f"Time reversal step {i}, using NN {min(i // (steps // nn_num), nn_num - 1)}")
            score_net = score_nets[min(i // (steps // nn_num), nn_num - 1)]
            score_net.eval()
            t = time_grid[i+1,:].repeat(N, 1)  # (N, 1)
            score = score_net(X_b[i+1,:,:], t).detach()  # (N, n)
        elif nn_num == 0:
            score = score_net(X_b[i+1,:,:], time_grid[i+1,:])  # (N, n)
        gg_T = torch.einsum('nij,njk->nik', g(X_b[i+1,:,:]), g(X_b[i+1,:,:]).transpose(1,2))  # (N, n, n)
        X_b[i,:,:] = X_b[i+1,:,:] - (f(X_b[i+1,:,:], time_grid[i+1,:]) - torch.einsum('nij,njk->nik', gg_T, score.unsqueeze(-1)).squeeze(-1) - div_ggT(X_b[i+1,:,:])) * dt + torch.einsum('nij,njk->nik', g(X_b[i+1,:,:]), W_b[i+1,:,:].unsqueeze(-1)).squeeze(-1)
    return X_b

def noise(dt, N, m):
    """
    Generate noise for the system that noise enter system through control channel.
    Args:
        dt (float): Time step size.
        N (int): Number of samples.
        m (int): Dimension of the noise.
    Returns:
        torch.ndarray: Noise samples of shape (N, m).
    """
    return torch.randn((N,m)) * torch.sqrt(torch.tensor(dt))

# def _phi_single(phi_net, x, t_scalar):
#     """
#     Single-sample wrapper: (2,), scalar -> (2,)
#     phi: nn.Module, maps (N,2),(N,1) -> (N,2)
#     x: (2,) tensor
#     t_scalar: scalar tensor or Python float
#     """
#     # if torch.is_tensor(t_scalar) and t_scalar.ndim > 0:
#     #     t_scalar = t_scalar.squeeze()  # ensure 0-D
#     return phi_net(x, t_scalar.unsqueeze(-1)).squeeze(0)  # (2,)
def _phi_single(phi_net, x, t_scalar):
    """
    Single-sample wrapper: x: (state_dim,), t_scalar: 0-D or 1-D -> output: (out_dim,)
    """
    # Make x batched: (1, state_dim)
    x_batched = x.unsqueeze(0)  # (1, D)

    # Make t_batched: (1, 1)
    if t_scalar.ndim == 0:
        t_batched = t_scalar.view(1, 1)
    else:
        # If it's already 1-D, make it (1, 1)
        t_batched = t_scalar.view(1, -1)  # typically (1, 1)

    # Forward through the net: (1, out_dim)
    y = phi_net(x_batched, t_batched)

    # Return unbatched: (out_dim,)
    return y.squeeze(0)

def batched_hessian_phi_wrt_x(phi_net, X, T):
    """
    Compute Hessian of phi w.r.t. x for a batch.

    Parameters
    ----------
    phi : nn.Module
        Model with forward(x, t) returning (N,4) given X:(N,4), T:(N,1)
    X : torch.Tensor, shape (N, 4)
    T : torch.Tensor, shape (N, 1)

    Returns
    -------
    H : torch.Tensor, shape (N, 4, 4, 4)
        H[n, i, j, k] = ∂² phi_i / ∂x_j ∂x_k at (X[n], T[n])
    """
    # assert X.ndim == 2 and X.size(-1) == 4
    # assert T.ndim == 2 and T.size(-1) == 1

    # Single-sample Hessian function
    hess_single_wrt_x = jacfwd(
        jacrev(lambda x, t: _phi_single(phi_net, x, t), argnums=0),
        argnums=0
    )

    # Vectorize over batch
    t_vec = T.squeeze(-1)  # (N,)
    H = vmap(hess_single_wrt_x, in_dims=(0, 0))(X, t_vec)  # (N, 4, 4, 4)
    return H

def time_reversal_bsde(H_x, g, phi_net, T, dt, Y_T, W_b, score, X_b, nn_num=0):
    """
    Perform time reversal of the BSDE using the learned phi function.
    Args:
        H_x: partial derivative of Hamiltonian w.r.t. x for Y_t
        g: Diffusion function
        phi_net: Trained neural network for phi
        T: Final time
        dt: Time step size
        Y_T: Terminal state (N, n)
        W_b: Backward Brownian motion increments (steps+1, N, m)
        exact_score: Exact score function for comparison
        X_b: Corresponding X_b trajectory (steps+1, N, n)
    Returns:
        Y_b: Reversed state trajectory (steps+1, N, n)
    """
    steps = int(T/dt)
    N, n = Y_T.shape
    Y_b = torch.zeros((steps+1, N, n), dtype=Y_T.dtype, device=Y_T.device)
    Y_b[-1,:,:] = Y_T
    time_grid = torch.linspace(0, T, steps+1).unsqueeze(-1).to(Y_T.device)  # (steps+1, 1)
    phi_net.eval()
    y_b = Y_T
    for i in range(steps-1, -1, -1):
        x_b = X_b[i, :, :]
        back_noise = W_b[i, :, :]
        if nn_num > 0:
            score_nets = score
            score_net = score_nets[min(i // (steps // nn_num), nn_num - 1)]
            score_net.eval()
            t = time_grid[i+1,:].repeat(N, 1)  # (N, 1)
            follmer = score_net(X_b[i+1,:,:], t).detach()  # (N, n)
        elif nn_num == 0:
            follmer = score(X_b[i+1,:,:], time_grid[i+1,:])
        gg_T = torch.einsum('nij,njk->nik', g(X_b[i+1,:,:]), g(X_b[i+1,:,:]).transpose(1,2))  # (N, n, n)
        follmer = torch.einsum('nij,njk->nik', gg_T, follmer.unsqueeze(-1)).squeeze(-1)
        Noise_sigma = g(x_b) # shape (N, n, m)
        x_b_nn = x_b.clone().requires_grad_(True)  # Enable gradient computation for x_b
        phi_pred = phi_net(x_b_nn, torch.tensor(i * dt).repeat(x_b.shape[0], 1))  # shape (N, 2)
        partial_phi_x = batched_jacobian(phi_pred, x_b_nn, device=x_b_nn.device).detach()  # shape (N, 4, 4)
        # trace term computation
        # Tr(D \partial_x^2 \phi)^i = sum_{jk} phi^i_xj_xk D_kj
        hessian_phi_x = batched_hessian_phi_wrt_x(phi_net, x_b_nn, torch.tensor(i * dt).repeat(x_b.shape[0], 1)).detach()  # shape (N, 4, 4, 4)
        D = torch.einsum('nij,njk->nik', Noise_sigma, Noise_sigma.transpose(1, 2))  # shape (N, n, n)
        trace_term = torch.einsum('njk,nijk->ni', D, hessian_phi_x) # shape (N, n)
        c_term = trace_term + torch.einsum('nij,nj->ni', partial_phi_x, follmer)  # shape (N, n)

        z_b = torch.einsum('nij,njk->nik', partial_phi_x, g(x_b))  # shape (N, n, m)
        phi_pred.detach_()
        # partial_H_x = H_x(x_b, Y_b[i+1,:,:], z_b)  # shape (N, n)
        # Change to exponential integrator
        cost_term, lag_term_mat, trace_term = H_x(x_b, Y_b[i+1,:,:], z_b, torch.tensor(i * dt))  # shape (N, n), (N, n, n), (N, n)
        help_mat = torch.zeros((N, n+n, n+n), dtype=x_b.dtype, device=x_b.device)
        help_mat[:, :n, :n] = lag_term_mat
        help_mat[:, :n, n:]  = torch.eye(n, dtype=x_b.dtype, device=x_b.device)
        exp_mat = torch.matrix_exp(help_mat * dt)  # shape (N, 2n, 2n)
        exp_lag = exp_mat[:, :n, :n]  # shape (N , n, n)
        exp_drift = exp_mat[:, :n, n:]  # shape (N , n, n)
        noise_term = torch.einsum('nij,njk->nik', z_b, back_noise.unsqueeze(-1)).squeeze(-1)  # shape (N, n)
        corrected_noise = torch.einsum('nij,nj->ni', exp_lag, noise_term)  # shape (N, n)
        y_b = torch.einsum('nij,nj->ni', exp_lag, y_b) + torch.einsum('nij,nj->ni', exp_drift, cost_term + trace_term + c_term) + corrected_noise
        # minus_dy = partial_H_x * dt + c_term * dt + torch.einsum('nij,njk->nik', z_b, back_noise.unsqueeze(-1)).squeeze(-1)  # shape (N, n)
        # y_b = y_b + minus_dy

        Y_b[i,:,:] = y_b
    return Y_b
    

def train_phi_network(phi_net, X_train, Y_train, time_grid, optimizer, scheduler, batch_size=64, iterations=1000):
    """
    Train the Phi Network.
    
    Args:
        phi_net: Instance of PhiNetwork
        X_train: Training state data (T x N x n)
        Y_train: Training target data (T x N x n)
        time_grid: Time grid (T x 1)
        g: Diffusion function g
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        batch_size: Batch size for training
        iterations: Number of training iterations
    Returns:
        loss_history: List of loss values during training
    """
    phi_net.train()
    loss = 0
    loss_history = []
    steps, N, n = X_train.shape
    X_train = X_train.permute(1, 0, 2) # (N, T, n)
    Y_train = Y_train.permute(1, 0, 2) # (N, T, n)
    t_batch_size = 16
    for i in range(iterations):

        batch_idx = random.sample(range(N), batch_size)
        X_batch = X_train[batch_idx, :, :]
        Y_batch = Y_train[batch_idx, :, :]
        time_batch = time_grid.repeat(batch_size, 1).unsqueeze(-1)  # (batch_size, T, 1)
        # time_idx = random.sample(range(steps), t_batch_size)
        # X_batch = X_batch[:, time_idx, :]  # (batch_size, t_batch_size, n)
        # Y_batch = Y_batch[:, time_idx, :]  # (batch_size, t_batch_size, n)
        # time_batch = time_batch[:, time_idx, :]  # (batch_size, t_batch_size, 1)
        X0_batch = X_batch[:, 0, :]
        XT_batch = X_batch[:, -1, :]
        Y0_batch = Y_batch[:, 0, :]
        YT_batch = Y_batch[:, -1, :]
        X_batch = X_batch.view(-1, n) # (batch_size * t_batch_size, n)
        time_batch = time_batch.view(-1, 1) # (batch_size * t_batch_size, 1)
        Y_batch = Y_batch.view(-1, n) # (batch_size * t_batch_size, n)
        phi_pred = phi_net(X_batch, time_batch)  # (batch_size * t_batch_size, n)

        # loss = nn.MSELoss()(phi_pred, Y_batch)  # Mean Squared Error loss
        loss = nn.SmoothL1Loss()(phi_pred, Y_batch)  # Huber loss
        t0_loss = nn.SmoothL1Loss()(phi_net(X0_batch, torch.tensor(0.0).repeat(batch_size, 1)), Y0_batch)
        t1_loss = nn.SmoothL1Loss()(phi_net(XT_batch, torch.tensor(4.0).repeat(batch_size, 1)), YT_batch)
        loss = loss + t0_loss*0.0 +  t1_loss*0.0
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 500 == 0 or i == iterations - 1:
            print(f"Iteration {i}, Loss: {loss.item()}")
            loss_history.append(loss.item())
    return loss_history

def riccati_eq(t, P_flat, A, dim=2):
    P = P_flat.reshape((dim,dim))
    dPdt = -(A.T@P + P@A)
    return dPdt.flatten()



def solve_riccati(A, Q_f, T, dt, dim=2):
    steps = int(T/dt)
    P_T = Q_f
    t_span = [T, 0]
    P_T_flat = P_T.flatten()
    sol = solve_ivp(riccati_eq, t_span, P_T_flat, args=(A, dim), method='RK45', dense_output=True)
    t = np.linspace(T, 0, steps)
    G = sol.sol(t)
    G_reversed = G[:,::-1]
    G_reversed = G_reversed.reshape((dim,dim,steps))
    return G_reversed


def sample_gaussian_mixture(batch_size, centers, std=0.5):
    """
    centers: (K,2) tensor
    """
    device = centers.device
    K = centers.shape[0]
    idx = torch.randint(0, K, (batch_size,), device=device)
    noise = torch.randn(batch_size, 2, device=device) * std
    return centers[idx] + noise
