import argparse
import os

import torch
import torch.onnx

from src.model import get_mobilenetv4


def parse_args():
    parser = argparse.ArgumentParser(description="Export best checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mobilenetv4_best.pth")
    parser.add_argument("--onnx", type=str, default="mobilenetv4.onnx")
    parser.add_argument("--onnx-sim", type=str, default="mobilenetv4_sim.onnx")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch axis")
    return parser.parse_args()


def export_onnx():
    args = parse_args()
    model = get_mobilenetv4(num_classes=38, pretrained=False)

    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("No checkpoint found. Exporting random initialized model for testing pipeline.")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting model to {args.onnx}...")
    export_kwargs = dict(
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    if args.dynamic_batch:
        export_kwargs["dynamic_axes"] = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(model, dummy_input, args.onnx, **export_kwargs)
    print("ONNX Export successful.")

    try:
        import onnx
        from onnxsim import simplify

        print("Simplifying ONNX model...")
        model_onnx = onnx.load(args.onnx)
        model_simp, check = simplify(model_onnx)
        if check:
            onnx.save(model_simp, args.onnx_sim)
            print(f"Simplified model saved to {args.onnx_sim}")
        else:
            print("Simplification check failed.")
    except ImportError:
        print("Warning: 'onnx' or 'onnx-simplifier' not installed. Skipping simplification.")
    except Exception as e:
        print(f"Simplification error: {e}")


if __name__ == "__main__":
    export_onnx()
