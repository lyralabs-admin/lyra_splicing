#!/usr/bin/env python3
# Evaluate a trained Lyra splice classifier on a test set

import os
import sys
import argparse
import torch

# This block allows the script to be run directly from the lyra_splicing directory
if __name__ == '__main__' and __package__ is None:
    # To allow running this script directly from the repo root
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from models import LyraSeqTagger
from utils import eval_lyra_on_h5

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Lyra model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model .pt file.")
    parser.add_argument("--test_h5", type=str, required=True, help="Path to the test dataset H5 file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--center_bp", type=int, default=5000, help="Size of the center window (in bp) to evaluate.")
    
    # Model parameters (should match the trained model)
    parser.add_argument("--d_model", type=int, default=48, help="Model dimension")
    parser.add_argument("--d_state", type=int, default=48, help="SSM state dimension")
    parser.add_argument("--num_blocks", type=int, default=22, help="Number of Lyra blocks")
    
    args = parser.parse_args()

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Instantiate the model with the same architecture as during training
    model = LyraSeqTagger(
        d_input=4, 
        d_model=args.d_model, 
        d_state=args.d_state,
        dropout=0.0,  # Dropout is not used in evaluation
        num_blocks=args.num_blocks, 
        transposed=False
    ).to(device)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully. Total parameters: {num_params:,}")

    # --- Run Evaluation ---
    print(f"\nEvaluating on {args.test_h5}...")
    test_metrics = eval_lyra_on_h5(
        model, 
        args.test_h5, 
        device, 
        batch_size=args.batch_size, 
        center_bp=args.center_bp
    )

    # --- Print Results ---
    print("\n--- Final Test Metrics ---")
    print(f"Acceptor Top-k Accuracy: {test_metrics.get('acceptor topk acc', 0.0):.4f}")
    print(f"Donor Top-k Accuracy:    {test_metrics.get('donor topk acc', 0.0):.4f}")
    print(f"Average Top-k Accuracy:  {test_metrics.get('avg topk acc', 0.0):.4f}")
    print("--------------------------")

if __name__ == "__main__":
    main()
