import torch
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy_loss(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    batch_size, _ = inputs.shape
    inputs_shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    log_Z = torch.log(torch.sum(torch.exp(inputs_shifted), dim=-1))
    loss = log_Z - inputs_shifted[torch.arange(batch_size), targets]
    return loss.mean()
