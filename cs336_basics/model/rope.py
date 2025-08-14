import torch
from einops import einsum, rearrange


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.rope_init()

    
    def rope_init(self):
        angle_base = 1.0 / (
            self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k)
        ).to(self.device) # (d_k / 2,)
        positions = torch.arange(
            self.max_seq_len, dtype=angle_base.dtype, device=self.device
        ) # (max_seq_len, )
        all_thetas = einsum(
            positions, angle_base,
            "max_seq_len, d_k -> max_seq_len d_k"
        ) # theta_i_k = all_thetas[i][k]
        rope_buffer_cos_sin = torch.stack(
            [torch.cos(all_thetas), torch.sin(all_thetas)],
            dim=-1
        ) # (max_seq_len, d_k/2, 2)
        self.register_buffer("rope_buffer_cos_sin", rope_buffer_cos_sin, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        """
        rope_cos_sin = self.rope_buffer_cos_sin[token_positions].to(dtype=x.dtype) #  (..., seq_len, d_k/2, 2)
        x_rearranged = rearrange(x, '... (d two) -> ... d two', two=2)
        cos, sin = rope_cos_sin[..., 0], rope_cos_sin[..., 1]
        results_apply_rope = torch.stack(
            [
                x_rearranged[..., 0] * cos - x_rearranged[..., 1] * sin,
                x_rearranged[..., 0] * sin + x_rearranged[..., 1] * cos,
            ],
            dim=-1
        )
        result = rearrange(results_apply_rope, '... d two -> ... (d two)', two=2)
        return result