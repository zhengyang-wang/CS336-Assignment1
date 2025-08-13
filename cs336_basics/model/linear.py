import torch
import math
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):  
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.nn.parameter.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )
        self.weight_init()
    

    def weight_init(self):
        sigma = math.sqrt(2/(self.in_features+self.out_features))
        torch.nn.init.trunc_normal_(
            self.weights,
            mean=0.0,
            std=sigma,
            a=-3.0*sigma,
            b=3.0*sigma,
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x, self.weights,
            "... d_in, d_out d_in -> ... d_out"
        )