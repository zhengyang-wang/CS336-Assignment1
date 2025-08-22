import os
import torch
from typing import IO, Any, BinaryIO


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> Any:
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    obj = torch.load(src)
    
    # Handle compiled model state dicts with _orig_mod prefix
    if 'model' in obj:
        model_state_dict = obj['model']
        if any(k.startswith('_orig_mod.') for k in model_state_dict.keys()):
            # Remove _orig_mod. prefix from keys
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith('_orig_mod.'):
                    new_key = k.replace('_orig_mod.', '')
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            model_state_dict = new_state_dict
            
        model.load_state_dict(model_state_dict)
    
    if optimizer and 'optimizer' in obj:
        optimizer.load_state_dict(obj['optimizer'])
        
    return obj.get('iteration', 0)  # Default to 0 if iteration not found