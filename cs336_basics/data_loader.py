import numpy.typing as npt
import torch
import numpy as np


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    allowed_starting_indices = np.arange(dataset.shape[0] - context_length)
    sampled_starting_indices = np.random.choice(allowed_starting_indices, batch_size)
    x = np.stack([
        dataset[starting_index : starting_index+context_length]
        for starting_index in sampled_starting_indices
    ])
    y = np.stack([
        dataset[starting_index+1 : starting_index+context_length+1]
        for starting_index in sampled_starting_indices
    ])
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)