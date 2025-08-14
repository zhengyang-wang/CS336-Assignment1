import torch
from einops import reduce


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = torch.nn.Parameter(
            torch.empty(
                d_model,
                device=device,
                dtype=dtype,
            )
        )
        self.weight_init()

    
    def weight_init(self):
        torch.nn.init.constant_(self.g, 1.0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x * x, '... d -> ... 1', 'sum') / self.d_model + self.eps)
        result = x / rms * self.g
        return result.to(in_dtype)