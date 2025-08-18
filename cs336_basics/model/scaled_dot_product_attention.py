import torch
import math
from torch import Tensor
from jaxtyping import Float
from einops import einsum
from cs336_basics.model import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, "batch_size ... queries d_k"],
    K: Float[Tensor, "batch_size ... keys d_k"],
    V: Float[Tensor, "batch_size ... keys d_v"],
    mask: Float[Tensor, "queries keys"] | None = None,
) -> Float[Tensor, "batch_size ... d_v"]:
    attn_map_pre_softmax = einsum(
        Q, K,
        "... queries d_k, ... keys d_k -> ... queries keys"
    )
    attn_map_pre_softmax /= math.sqrt(Q.shape[-1])

    # apply the mask
    if mask is not None:
        attn_map_pre_softmax[~mask] -= torch.inf

    # softmax
    attn_map_after_softmax = softmax(attn_map_pre_softmax, dim=-1)
    
    # output
    output = einsum(
        attn_map_after_softmax, V,
        "... queries keys, ... keys d_v -> ... queries d_v",
    )
    return output