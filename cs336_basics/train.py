import argparse
import os
import sys
import time
import wandb
import torch
import numpy as np
from cs336_basics.data_loader import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.train import (
    cross_entropy_loss,
    AdamW,
    lr_cosine_schedule,
    gradient_clipping,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Transformer Language Model on text data."
    )

    # Create argument groups for better organization
    data_group = parser.add_argument_group('Data', 'Dataset and batching options')
    model_group = parser.add_argument_group('Model', 'Model architecture parameters')
    training_group = parser.add_argument_group('Training', 'Training loop parameters')
    optimizer_group = parser.add_argument_group('Optimizer', 'Optimization parameters')
    output_group = parser.add_argument_group('Output', 'Logging and checkpointing options')

    # Data arguments
    data_group.add_argument(
        "--training-dataset",
        required=True,
        help="Path to the training dataset (.npy file)"
    )

    data_group.add_argument(
        "--validation-dataset",
        required=True,
        help="Path to the validation dataset (.npy file)"
    )
    
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    
    data_group.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Context length for transformer (default: 256)"
    )
    
    # Model arguments
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

    # Training arguments
    training_group.add_argument(
        "--num-iterations",
        type=int,
        default=10000,
        help="Number of training iterations (default: 10000)"
    )
        
    training_group.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log model every N iterations (default: 10)"
    )

    training_group.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="Evaluate model every N iterations (default: 500)"
    )
    
    training_group.add_argument(
        "--save-every", 
        type=int,
        default=1000,
        help="Save checkpoint every N iterations (default: 1000)"
    )

    output_group.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=f"Device to run training on (default: {'cuda' if torch.cuda.is_available() else 'cpu'})"
    )
    
    output_group.add_argument(
        "--resume-from",
        help="Path to checkpoint to resume training from (default: None)"
    )
    
    # Optimizer arguments
    optimizer_group.add_argument(
        "--lr",
        type=float,
        default=6e-4,
        help="Maximum learning rate (default: 6e-4)"
    )
    
    optimizer_group.add_argument(
        "--min-lr",
        type=float,
        default=6e-5,
        help="Minimum learning rate (default: 6e-5)"
    )
    
    optimizer_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    
    optimizer_group.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 in AdamW (default: 0.9)"
    )
    
    optimizer_group.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Beta2 in AdamW (default: 0.999)"
    )
    
    optimizer_group.add_argument(
        "--warmup-iters",
        type=int,
        default=1000,
        help="Learning rate warmup iterations (default: 1000)"
    )
    
    optimizer_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (default: 1.0)"
    )

    # Output arguments
    output_group.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory to save checkpoints (default: 'checkpoints')"
    )
    
    output_group.add_argument(
        "--wandb-project",
        default="transformer-lm",
        help="Weights & Biases project name (default: 'transformer-lm')"
    )
    
    output_group.add_argument(
        "--wandb-name",
        help="Weights & Biases run name (default: None)"
    )

    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device for M-series Mac (MPS if available)
    if args.device == "cuda" and torch.backends.mps.is_available():
        device = torch.device("mps")
        # Do not set TF32 precision for MPS devices
        if args.verbose:
            print("Using MPS device - TF32 precision not supported", file=sys.stderr)
    else:
        device = torch.device(args.device)
        # Only set high precision for CUDA devices
        if device.type == "cuda":
            torch.set_float32_matmul_precision('high')
            if args.verbose:
                print("Using CUDA device with high matmul precision", file=sys.stderr)
    
    if args.verbose:
        print(f"Running with arguments: {args}", file=sys.stderr)
        print(f"Using device: {device}", file=sys.stderr)
    
    # wandb log
    wandb.login()

    config = {
        # Data arguments
        'training_dataset': args.training_dataset,
        'validation_dataset': args.validation_dataset,
        'batch_size': args.batch_size,
        'context_length': args.context_length,
        
        # Model arguments
        'vocab_size': args.vocab_size,
        'num_layers': args.num_layers,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'rope_theta': args.rope_theta,
        
        # Training arguments
        'num_iterations': args.num_iterations,
        'log_every': args.log_every,
        'eval_every': args.eval_every,
        'save_every': args.save_every,
        'device': str(device),
        'resume_from': args.resume_from,
        
        # Optimizer arguments
        'lr': args.lr,
        'min_lr': args.min_lr,
        'weight_decay': args.weight_decay,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'warmup_iters': args.warmup_iters,
        'max_grad_norm': args.max_grad_norm,
        
        # Output arguments
        'checkpoint_dir': args.checkpoint_dir,
        'wandb_project': args.wandb_project,
        'wandb_name': args.wandb_name,
    }

    with wandb.init(project=args.wandb_project, name=args.wandb_name, config=config) as run:
        # Create model
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
        
        # Device-specific model compilation
        if device.type == "cpu":
            model = torch.compile(model)  # Standard compilation for CPU
        elif device.type == "mps":
            model = torch.compile(model, backend="aot_eager")  # Optimize backward pass on MPS
        else:  # CUDA and other devices
            model = torch.compile(model)  # Standard compilation

        # Create optimizer
        optimizer = AdamW(
            params=model.parameters(),
            lr=args.lr, # will change during training according to lr_cosine_schedule
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )

        # Load dataset
        training_dataset = np.load(args.training_dataset, mmap_mode='r')
        validation_dataset = np.load(args.validation_dataset, mmap_mode='r')

        # Handle args.resume_from
        start_iter = 0
        if args.resume_from:
            if os.path.exists(args.resume_from):
                start_iter = load_checkpoint(
                    src=args.resume_from,
                    model=model,
                    optimizer=optimizer,
                )
                print(f"Resuming from {args.resume_from} at iteration {start_iter}")
            else:
                print(f"Checkpoint {args.resume_from} not found, starting from scratch")
        
        # Creat checkpoint folder
        checkpoint_dir = os.path.join(args.checkpoint_dir, args.wandb_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training loop
        for it in range(start_iter+1, args.num_iterations+1):
            # Start timing this training step
            step_start_time = time.time()
            
            # get the real lr
            current_lr = lr_cosine_schedule(
                it=it,
                max_learning_rate=args.lr,
                min_learning_rate=args.min_lr,
                warmup_iters=args.warmup_iters,
                cosine_cycle_iters=args.num_iterations,
            )

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # reset gradients
            optimizer.zero_grad()

            # get batch
            x_batch, y_batch = get_batch(
                dataset=training_dataset,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=str(device),
            )

            # forward
            model_outputs = model(x_batch)
            loss = cross_entropy_loss(
                inputs=model_outputs.view(-1, args.vocab_size),
                targets=y_batch.view(-1),
            )

            # backward
            loss.backward()
            grad_norm = gradient_clipping(
                parameters=model.parameters(),
                max_l2_norm=args.max_grad_norm,
            )
            optimizer.step()
            
            # Calculate step time
            step_time = time.time() - step_start_time

            # log
            if it % args.log_every == 0:
                metrics = {
                    'train/loss': loss.item(),
                    'train/learning_rate': current_lr,
                    'train/grad_norm': grad_norm,
                    'train/iteration': it,
                    'train/step_time': step_time,
                }
                run.log(metrics)
                
                if args.verbose:
                    print(f"Iter {it}: Loss = {loss.item():.4f}, LR = {current_lr:.6f}, Step time = {step_time*1000:.2f}ms")

            # eval
            if it % args.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    x_eval_batch, y_eval_batch = get_batch(
                        dataset=validation_dataset,
                        batch_size=args.batch_size,
                        context_length=args.context_length,
                        device=str(device),
                    )
                    eval_model_outputs = model(x_eval_batch)
                    eval_loss = cross_entropy_loss(
                        inputs=eval_model_outputs.view(-1, args.vocab_size),
                        targets=y_eval_batch.view(-1),
                    )
                    run.log({'eval/loss': eval_loss.item()})
                    if args.verbose:
                        print(f"Eval loss: {eval_loss.item():.4f}")
                model.train()

            # save
            if it % args.save_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{it:06d}.pt")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=it,
                    out=checkpoint_path
                )
                if args.verbose:
                    print(f"Saved checkpoint to {checkpoint_path}")


        # final checkpoint save
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_last.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=args.num_iterations,
            out=checkpoint_path
        )
        print(f"Training completed. Saved checkpoint to {checkpoint_path}")

    
if __name__ == "__main__":
    main()