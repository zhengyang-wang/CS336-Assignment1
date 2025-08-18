import torch
import math
from einops import einsum
from cs336_basics.model import Linear


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, device, dtype)
        self.linear2 = Linear(d_ff, d_model, device, dtype)
        self.linear3 = Linear(d_model, d_ff, device, dtype)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_input = self.linear1(x)
        silu_output = SiLU(silu_input)
        gated_output = silu_output * self.linear3(x)
        return self.linear2(gated_output)