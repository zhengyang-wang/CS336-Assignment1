import argparse
import sys
import torch
from cs336_basics.checkpoint import load_checkpoint
from cs336_basics.model import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use a Transformer Language Model to generate text."
    )

    # Create argument groups for better organization
    tokenizer_group = parser.add_argument_group('Tokenizer', 'Tokenizer parameters')
    model_group = parser.add_argument_group('Model', 'Model architecture parameters')
    generate_params_group = parser.add_argument_group('Generate Params', 'Decoding parameters')

    # Tokenizer arguments
    tokenizer_group.add_argument(
        "--tokenizer-vocab-path",
        type=str,
        help='Path to the vocab (.json) file'
    )

    tokenizer_group.add_argument(
        "--tokenizer-merges-path",
        type=str,
        help='Path to the merges (.txt) file'
    )

    tokenizer_group.add_argument(
        "--tokenizer-eos-tokens",
        type=str,
        nargs='+',
        default=["<|endoftext|>"],
        help='End of sequence token(s) (default: ["<|endoftext|>"])'
    )

    # Model arguments
    model_group.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Context length for transformer (default: 256)"
    )
    
    model_group.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Vocabulary size of the model"
    )
    
    model_group.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers (default: 12)"
    )
    
    model_group.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Hidden dimension of the model (default: 768)"
    )
    
    model_group.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help="Number of attention heads (default: 12)"
    )
    
    model_group.add_argument(
        "--d-ff",
        type=int,
        default=1344,
        help="Feedforward dimension (default: 3072)"
    )
    
    model_group.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help="RoPE theta parameter (default: 10000.0)"
    )

    model_group.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=f"Device to run training on (default: {'cuda' if torch.cuda.is_available() else 'cpu'})"
    )
    
    model_group.add_argument(
        "--load-from",
        help="Path to checkpoint to load from (default: None)"
    )
    

    # Generate Arguments
    generate_params_group.add_argument(
        "--max-gen-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens (default: 128)"
    )

    generate_params_group.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature scaling (default: 0.5)"
    )

    generate_params_group.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Threshold in top-p sampling (default: 0.9)"
    )

    return parser.parse_args()


def generate(input_string, model, tokenizer, eos_ids, max_gen_tokens, temperature, top_p, device):
    if temperature > 1.0 or temperature < 0.0:
        raise ValueError(f"Temperature should be [0, 1], but got {temperature}.")

    # encode
    input_ids = tokenizer.encode(input_string)
    
    num_gen_tokens = 0
    while True:
        num_gen_tokens += 1
        next_token_logits = model(torch.Tensor(input_ids).long().unsqueeze(0).to(device))[0, -1, :]

        if temperature > 0.0:
            # temperature scaling
            next_token_logits /= temperature
            next_token_probs = softmax(next_token_logits, dim=-1)
            # top-p sampling
            sorted_prob_desc, indices_desc = torch.sort(next_token_probs, descending=True)
            cumsum_probs = 0
            largest_index_to_keep = 0
            while largest_index_to_keep < sorted_prob_desc.shape[0]:
                cumsum_probs += sorted_prob_desc[largest_index_to_keep]
                largest_index_to_keep += 1
                if cumsum_probs >= top_p:
                    break
            sorted_prob_desc_to_keep = sorted_prob_desc[:largest_index_to_keep]
            sampled_index = torch.distributions.categorical.Categorical(sorted_prob_desc_to_keep).sample()
            next_token_id = indices_desc[sampled_index].item()
        else:
            # greedy decoding
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

        if next_token_id in eos_ids or len(input_ids) >= max_gen_tokens:
            break
        else:
            print(tokenizer.decode([next_token_id]), end='')
            sys.stdout.flush()
        input_ids += [next_token_id]
        

def main():
    args = parse_args()
    print(args)

    # Set device for M-series Mac (MPS if available)
    if args.device == "cuda" and torch.backends.mps.is_available():
        device = torch.device("mps")
        # Do not set TF32 precision for MPS devices
        print("Using MPS device - TF32 precision not supported", file=sys.stderr)
    else:
        device = torch.device(args.device)
        # Only set high precision for CUDA devices
        if device.type == "cuda":
            torch.set_float32_matmul_precision('high')
            print("Using CUDA device with high matmul precision", file=sys.stderr)
    
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    model.to(device)
    load_checkpoint(
        src=args.load_from,
        model=model,
    )
    model.eval()

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.tokenizer_vocab_path,
        merges_filepath=args.tokenizer_merges_path,
        special_tokens=args.tokenizer_eos_tokens)
    eos_ids = [tokenizer.vocab_bytes_to_int[eos_token.encode("utf-8")]
               for eos_token in args.tokenizer_eos_tokens]
    
    with torch.no_grad():
        print(">> Start prompting! " \
            "(NOTE: No multi turn support. Every prompt is a new session.)" \
            "\n================")
        while True:
            user_input = input("Prompt: ")
            print("Full sequence:")
            print(user_input, end='')
            generate(
                input_string=user_input,
                model=model,
                tokenizer=tokenizer,
                eos_ids=eos_ids,
                max_gen_tokens=args.max_gen_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device
            )
            print("\n================")


if __name__ == "__main__":
    main()