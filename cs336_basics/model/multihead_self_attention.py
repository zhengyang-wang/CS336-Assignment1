import torch
import math
from einops import einsum, rearrange
from cs336_basics.model import scaled_dot_product_attention, RotaryPositionalEmbedding, Linear


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_applier: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"Error: d_model/h = {d_model/num_heads}, not an integer."
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope_applier = rope_applier
        self.qkv_proj = Linear(d_model, d_model*3, device, dtype)
        self.o_proj = Linear(d_model, d_model, device, dtype)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        # q,k,v projections
        qkv_combined = self.qkv_proj(x)
        q, k, v = torch.split(qkv_combined, [self.d_model, self.d_model, self.d_model], dim=-1)

        # split heads
        q_multi_head = rearrange(q, "batch ... (h d_k) -> h batch ... d_k", h=self.num_heads)
        k_multi_head = rearrange(k, "batch ... (h d_k) -> h batch ... d_k", h=self.num_heads)
        v_multi_head = rearrange(v, "batch ... (h d_k) -> h batch ... d_k", h=self.num_heads)

        # causal mask
        seq_len = x.shape[-2]
        mask_size = (1,) * (q_multi_head.dim() - 2) + (seq_len, seq_len)
        mask = ~torch.ones(mask_size, requires_grad=False, dtype=torch.bool, device=x.device).triu(1)

        # apply rope
        if token_positions is not None:
            q_multi_head = self.rope_applier(q_multi_head, token_positions)
            k_multi_head = self.rope_applier(k_multi_head, token_positions)

        # attention
        output = scaled_dot_product_attention(q_multi_head, k_multi_head, v_multi_head, mask)

        # concat heads
        output = rearrange(output, "h ... d_k -> ... (h d_k)")

        # o projection
        return self.o_proj(output)