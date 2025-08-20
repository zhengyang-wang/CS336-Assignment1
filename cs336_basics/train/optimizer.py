import torch
import math
from typing import Callable, List, Optional, Tuple


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: List[torch.nn.Parameter],
        lr: float,
        weight_decay: float,
        betas: Tuple[float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)
        

    def step(self, closure: Optional[Callable] = None):
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group['params']:
                with torch.no_grad():
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    t = state.get("t", 1)
                    first_moment = state.get("first_moment", torch.zeros_like(p))
                    second_moment = state.get("second_moment", torch.zeros_like(p))
                    # get gradients
                    grad = p.grad
                    # update the first moment estimate
                    first_moment.mul_(beta1).add_(grad, alpha=1-beta1)
                    # update the second moment estimate
                    second_moment.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                    # compute adjusted LR
                    lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                    # upadate the parameters
                    denom = second_moment.sqrt().add_(eps)
                    p.addcdiv_(first_moment, denom, value=-lr_t)
                    # add weight decay
                    p.mul_(1 - lr * weight_decay)

                    # update state
                    state["t"] = t + 1
                    state["first_moment"] = first_moment
                    state["second_moment"] = second_moment