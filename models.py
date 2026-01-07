import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flashfftconv import FlashDepthWiseConv1d
from typing import Optional

class S4DKernel(nn.Module):
    """Generate convolution kernel K(H,L) from diagonal SSM params (trainable)."""
    def __init__(self, d_model, N=64, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        H = d_model
        # trainable params
        self.log_dt = nn.Parameter(
            torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))  # store as real-imag
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, N // 2)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(N // 2).repeat(H, 1), requires_grad=False)

    def forward(self, L: int):  # return (H, L)
        dt = torch.exp(self.log_dt)                # (H)
        C = torch.view_as_complex(self.C)          # (H, N/2)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N/2)
        dtA = A * dt.unsqueeze(-1)                 # (H, N/2)

        t = torch.arange(L, device=dtA.device)     # (L,)
        exp_dtA_t = torch.exp(dtA.unsqueeze(-1) * t)        # (H, N/2, L)
        C_eff = C * (torch.exp(dtA) - 1.) / A               # (H, N/2)
        K = 2 * torch.einsum('hn,hnl->hl', C_eff, exp_dtA_t).real  # (H, L)
        return K


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

class S4D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        dropout=0.0,
        transposed=True,
        no_glu=False,
        no_gelu=False,
        bidirectional=True,
        **kernel_args,
    ):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        self.bidirectional = bidirectional
        self.no_glu = no_glu
        self.no_gelu = no_gelu

        # Initialize D parameter
        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel - create 2x channels if bidirectional
        kernel_h = self.h * 2 if self.bidirectional else self.h
        self.kernel = S4DKernel(kernel_h, N=self.n, **kernel_args)

        self.in_conv = nn.Conv1d(
            self.h, 3 * self.h, kernel_size=11, groups=self.h, padding=5
        )

        # Pointwise
        self.activation = nn.GELU() if not self.no_gelu else nn.Identity()
        self.dropout= nn.Dropout(dropout)

        # Position-wise output transform to mix features
        if not self.no_glu:
            self.output_linear = nn.Sequential(
                nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
                nn.GLU(dim=-2),
            )
        else:
            self.output_linear = nn.Conv1d(self.h, self.h, kernel_size=1)


        self._init_weights()

    def _init_weights(self):
        """Initialize weights following best practices"""
        # Initialize D
        nn.init.uniform_(self.D, -0.01, 0.01)

        # Initialize output linear
        for module in self.output_linear.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, u, **kwargs):
        """Input and output shape (B, H, L) if transposed else (B, L, H)"""
        if not self.transposed:
            u = u.transpose(-1, -2)
        u = u.contiguous()  # Ensure contiguous memory layout for FlashFFTConv
        xvz = self.in_conv(u)
        x, v, z = torch.chunk(xvz, 3, dim=1)
        u = v * x
        u = u.contiguous()

        L = u.size(-1)
        
        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L) or (2H L) if bidirectional

        # Handle bidirectional
        if self.bidirectional:
            # Split into forward and backward kernels
            k0, k1 = rearrange(k, "(s h) l -> s h l", s=2)
            # Pad and combine: forward padded right, backward flipped and padded left
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))

        # Convolution in frequency domain
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term - skip connection
        y = y + u * self.D.unsqueeze(-1)

        # Activation and dropout
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class PGC(nn.Module):
    '''
    Parallel Gated Convolution module with FFT-based convolution.
    This module projects the input, applies a gated FFT convolution,
    and projects back to the original dimension.
    '''
    def __init__(self, d_model, kernel_size=3,expansion_factor=1.0, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.kernel_size = kernel_size
        
        # Calculate expanded dimension
        expanded_dim = int(d_model * expansion_factor)
        
        # Input projection and normalization
        self.in_proj = nn.Linear(d_model, expanded_dim * 2)
        self.in_norm = RMSNorm(expanded_dim * 2, eps=1e-8)


        # Regular convolution for initialization (not used in forward, just for init)
        self.conv = nn.Conv1d(
            expanded_dim,
            expanded_dim,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1)//2,
            groups=expanded_dim
        )
        # disable gradients on conv since it's only used for initialization
        for p in self.conv.parameters():
            p.requires_grad = False
        
        # Flash FFT Conv layer, passing weights and bias from self.conv
        self.flash_conv = FlashDepthWiseConv1d(
            expanded_dim,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size - 1)//2,
            weights=self.conv.weight,
            bias=self.conv.bias
        )
        
        # Output projection and normalization
        self.out_proj = nn.Linear(expanded_dim, d_model)
        self.out_norm = RMSNorm(d_model, eps=1e-8)

    def forward(self, u):
        # Input projection and normalization
        xv = self.in_norm(self.in_proj(u))
        
        # Split into x and v for gating
        x, v = xv.chunk(2, dim=-1)
        
        x_feature_mixed = self.flash_conv(rearrange(x, 'b t f -> b f t').contiguous())
        x_feature_mixed = rearrange(x_feature_mixed, 'b f t -> b t f')
        
        # Apply gating with v
        gated_output = v * x_feature_mixed
        
        # Output projection and normalization
        out = self.out_norm(self.out_proj(gated_output))
        
        return out

class LyraSeqTagger(nn.Module):
    def __init__(self, d_input=4, d_model=64, d_state=64, dropout=0.25,
                 model_variant="s4d", num_blocks=8,transposed=False, 
                 **kernel_args):
        super().__init__()
        
        self.encoder = nn.Linear(d_input, d_model)
        
        # Create Lyra blocks with different configurations
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            # Create backbone
            backbone = S4D(d_model, d_state=d_state, dropout=dropout, 
                        transposed=transposed, **kernel_args)

            pgc1_expansion = 2.0
            pgc1_kernel = 11
            
            block = nn.ModuleDict({
                'pgc1': PGC(d_model, 
                                expansion_factor=pgc1_expansion, 
                                dropout=dropout, 
                                kernel_size=pgc1_kernel),
                'norm': RMSNorm(d_model),
                'backbone': backbone,
                'dropout': nn.Dropout(dropout)
            })
            self.blocks.append(block)
        
        # Output head
        self.output_head = nn.Linear(d_model, 3)  # per-position logits: [other, acceptor, donor]
        
    def forward(self, X):
        h = self.encoder(X)
        
        # Store outputs for skip connections
        skip_connections = []
        
        for i, block in enumerate(self.blocks):
            # Exact Lyra architecture
            h = block['pgc1'](h)
            z = h
            z = block['norm'](z)
            h = block['dropout'](block['backbone'](z)) + h
            
            # Add skip connection from 2 blocks back
            if i >= 2:
                h = h + skip_connections[i-2]
            
            # Store for future skip connections
            skip_connections.append(h)
        
        return self.output_head(h)

