import torch
import math
from collections.abc import Iterable


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    with torch.no_grad():
        l2_norm_per_grad = [torch.norm(p.grad) for p in parameters if p.grad is not None]
        l2_norm_all_grads = math.sqrt(sum([l2 ** 2 for l2 in l2_norm_per_grad]))
        
        if l2_norm_all_grads >= max_l2_norm:
            for p in parameters:
                if p.grad is None:
                    continue
                
                grad = p.grad
                grad.mul_(max_l2_norm / (l2_norm_all_grads + eps))
