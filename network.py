import torch
import torch.nn as nn
import math

class TimeFourierEmbedding(nn.Module):
    def __init__(self, dim_fourier: int = 32, dim_out: int = 128,
                 max_period: float = 10_000.0, mlp_depth: int = 2,
                 hidden_mult: float = 1.0, act=nn.SiLU, dropout: float = 0.0):
        """
        Maps scalar time t -> R^{dim_out} via [sin,cos] Fourier features + MLP.
        dim_fourier must be even.
        """
        super().__init__()
        assert dim_fourier % 2 == 0, "dim_fourier must be even"
        self.dim_fourier = dim_fourier
        self.max_period = max_period

        layers = []
        in_dim = dim_fourier
        hidden = int(dim_out * hidden_mult) if mlp_depth > 1 else dim_out
        for i in range(mlp_depth):
            out_dim = dim_out if i == mlp_depth - 1 else hidden
            layers += [nn.Linear(in_dim, out_dim)]
            if i != mlp_depth - 1:
                layers += [act()]
                if dropout > 0:
                    layers += [nn.Dropout(dropout)]
            in_dim = out_dim
        self.proj = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (N, 1) time tensor (any scale). Returns (N, dim_out).
        """
        half = self.dim_fourier // 2
        # frequencies: geometric from 1/max_period up to 1
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(0, half, device=t.device, dtype=t.dtype) / half
        )  # (half,)
        args = t * freqs[None, :]  # (N, half)
        fourier = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (N, dim_fourier)
        return self.proj(fourier)  # (N, dim_out)


class ResBlock(nn.Module):
    def __init__(self, width: int, act=nn.SiLU, dropout: float = 0.0, use_layernorm: bool = False):
        super().__init__()
        self.lin1 = nn.Linear(width, width)
        self.lin2 = nn.Linear(width, width)
        self.act = act()
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(width) if use_layernorm else nn.Identity()

    def forward(self, h):
        res = h
        h = self.act(self.lin1(h))
        h = self.do(h)
        h = self.lin2(h)
        h = self.ln(h + res)
        return self.act(h)
    

class ScoreNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,   # was (state_dim + 1); we keep it for drop-in compatibility
                 out_dim: int,
                 hidden_dim: int = 128,
                 num_blocks: int = 4,
                 time_fourier: int = 32,
                 time_emb_dim: int = 128,
                 time_mlp_depth: int = 2,
                 dropout: float = 0.0,
                 use_layernorm: bool = False):
        """
        Drop-in replacement:
          - Keeps the same constructor signature.
          - Internally: infers state_dim = input_dim - 1 and replaces raw 't'
            by a Fourier time embedding of size 'time_emb_dim'.

        Args:
          input_dim: was (state_dim + 1). We infer state_dim = input_dim - 1.
          out_dim: dimension of score (state dimension).
          hidden_dim: width of hidden layers.
          num_blocks: number of residual blocks.
          time_fourier: size of [sin,cos] bank (must be even).
          time_emb_dim: projected time embedding size.
          time_mlp_depth: MLP depth used to project Fourier features.
        """
        super().__init__()
        self.state_dim = input_dim - 1  # infer from your existing usage
        assert self.state_dim > 0, "input_dim must be >= 2 (state + time)"

        # time embedding module
        self.time_emb = TimeFourierEmbedding(
            dim_fourier=time_fourier,
            dim_out=time_emb_dim,
            mlp_depth=time_mlp_depth,
            hidden_mult=1.0,
            act=nn.SiLU,
            dropout=dropout
        )

        # input proj: [x, time_emb] -> hidden
        self.inp = nn.Linear(self.state_dim + time_emb_dim, hidden_dim)
        self.act = nn.SiLU()

        # residual trunk
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, act=nn.SiLU, dropout=dropout, use_layernorm=use_layernorm)
            for _ in range(num_blocks)
        ])

        # output head
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: (N, state_dim)          # same as before (do NOT append t)
        t: (N, 1)                  # same as before
        returns: (N, out_dim)      # score estimate ∇_x log p_t(x)
        """
        # time embedding
        t_feat = self.time_emb(t)                  # (N, time_emb_dim)
        # concat state + time features
        h = self.act(self.inp(torch.cat([x, t_feat], dim=-1)))  # (N, hidden_dim)
        # residual trunk
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)
    

class PhiNetwork(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=16):
        """
        Initialize the Phi Network.
        Args:
            input_dim: Dimension of the input vector (state X and time t)
            out_dim: Dimension of the output vector (BSDE Y dimension)
            hidden_dim: Dimension of the hidden layers
        """
        super(PhiNetwork, self).__init__()
        self.activation = nn.SiLU()
        self.layer_input = nn.Linear(input_dim, hidden_dim)
        self.layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, t):
        """
        Forward pass of the Phi Network.
        Args:
            x: Input state vector (N x input_dim)
            t: Time vector (N x 1)
        Returns:
            Output phi vector (N x out_dim)
        """
        z_in = torch.cat((x, t), dim=-1)
        
        h = self.layer_input(z_in)
        h_temp = self.activation(self.layer_1(h))
        h_temp = self.layer_2(h_temp)
        h = self.activation(h + h_temp)

        h_temp = self.activation(self.layer_3(h))
        h_temp = self.layer_4(h_temp)
        h = self.activation(h + h_temp)

        z_out = self.layer_output(h)

        return z_out
