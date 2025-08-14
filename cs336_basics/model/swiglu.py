import torch
import math
from einops import einsum


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1_weight = torch.nn.Parameter(
            torch.empty(
                d_ff,
                d_model,
                device=device,
                dtype=dtype,
            )
        )
        self.w2_weight = torch.nn.Parameter(
            torch.empty(
                d_model,
                d_ff,
                device=device,
                dtype=dtype,
            )
        )
        self.w3_weight = torch.nn.Parameter(
            torch.empty(
                d_ff,
                d_model,
                device=device,
                dtype=dtype,
            )
        )
        self.weight_init(self.w1_weight)
        self.weight_init(self.w2_weight)
        self.weight_init(self.w3_weight)


    def weight_init(self, linear_weights):
        d_out, d_in = linear_weights.shape
        sigma = math.sqrt(2/(d_in+d_out))
        torch.nn.init.trunc_normal_(
            linear_weights,
            mean=0.0,
            std=sigma,
            a=-3.0*sigma,
            b=3.0*sigma,
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_input = einsum(
            x, self.w1_weight,
            "... d_model, d_ff d_model -> ... d_ff"
        )
        silu_output = silu_input * torch.sigmoid(silu_input)
        gated_output = silu_output * einsum(
            x, self.w3_weight,
            "... d_model, d_ff d_model -> ... d_ff"
        )
        return einsum(
            gated_output, self.w2_weight,
            "... d_ff, d_model d_ff -> ... d_model"
        )