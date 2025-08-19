import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x_shifted)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)