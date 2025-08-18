import torch
from cs336_basics.model import TransformerBlock, Embedding, RMSNorm, Linear, softmax, RotaryPositionalEmbedding


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        d_k = int(d_model / num_heads)
        rope_applier = RotaryPositionalEmbedding(d_k=d_k, theta=rope_theta, max_seq_len=context_length)
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_applier=rope_applier,
                eps=eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.final_rmsnorm = RMSNorm(d_model, eps, device, dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
    

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        x = self.token_embeddings(x)
        for transformer_block in self.layers:
            x = transformer_block(x, token_positions)
        x = self.lm_head(self.final_rmsnorm(x))
        return x