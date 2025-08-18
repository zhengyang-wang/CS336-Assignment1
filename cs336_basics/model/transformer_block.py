import torch
from cs336_basics.model import SwiGLU, CausalMultiHeadSelfAttention, RMSNorm, RotaryPositionalEmbedding


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_applier: RotaryPositionalEmbedding | None = None,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.position_wise_ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.rmsnorm_ffn = RMSNorm(d_model, eps, device, dtype)
        self.causal_multi_head_self_attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope_applier, device, dtype)
        self.rmsnorm_attn = RMSNorm(d_model, eps, device, dtype)

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        x = x + self.causal_multi_head_self_attn(self.rmsnorm_attn(x), token_positions)
        x = x + self.position_wise_ffn(self.rmsnorm_ffn(x))
        return x