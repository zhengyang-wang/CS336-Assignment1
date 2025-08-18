import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x -= x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)