from train import *
from infer import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GFlowNet model.")
    parser.add_argument('--infer', action='store_true', help="Run inference instead of training")
    parser.add_argument('--model_path', type=str, default=None, help="Path to the trained model for inference")

    args = parser.parse_args()

    if args.infer:     # --infer --model_path ./pretrained/10_3/checkpoint_epoch900.pt
        if not args.model_path:
            raise ValueError("You must provide --model_path when using --infer")
        print(f"Start inference with model: {args.model_path}")
        infer(args.model_path)
        print("Inference complete!")
    else:
        print("Start training GFlowNet FD model...")
        train()
        print("Training complete!")