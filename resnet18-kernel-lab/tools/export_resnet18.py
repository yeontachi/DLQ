import os, json, argparse, numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

def save_bin(path, tensor):
    arr = tensor.detach().cpu().contiguous().float().numpy()
    arr.tofile(path)

def tensor_meta(name, t):
    shape = list(t.shape)
    kind = "param"
    layout = None
    if "conv" in name and "weight" in name and t.ndim == 4:
        layout = "OIHW" # (out, in, h, w)
        kind = "conv_weight"
    elif any(k in name for k in ["running_mean", "running_var"]):
        kind = "bn_buffer"; layout = "O" # 1D
    elif "bn" in name:
        kind = "bn_param"; layout = "O"
    elif name.endswith("fc.weight") and t.ndim == 2:
        kind = "fc_weight"; layout = "OI" # (out, in)
    elif name.endswith("fc.bias"):
        kind = "fc_bias"; layout = "O"
    else:
        layout = "auto"
    return {"shape": shape, "layout": layout, "kind": kind}

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    sd = m.state_dict()

    manifest = {
        "model":  "resnet18",
        "dtype":  "fp32",
        "layout": "NCHW",
        "version": 1,
        "preprocess": {
            "resize": 256, "center_crop": 224,
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225]
        },
        "tensors": {}
    }

    for name, t in sd.items():
        # 파일 경로: <out_dir>/<name>.bin (중첩 폴더 자동 생성)
        subdir = os.path.join(out_dir, os.path.dirname(name))
        os.makedirs(subdir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.bin")
        save_bin(path, t)
        meta = tensor_meta(name, t)
        meta["path"] = os.path.relpath(path, out_dir)  # manifest 기준 상대 경로
        manifest["tensors"][name] = meta
        print(f"saved: {name:30s}   {meta['shape']} -> {meta['path']}")

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nExport complete ->", os.path.join(out_dir, "manifest.json"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory (e.g., exports/resnet18/fp32)")
    args = ap.parse_args()
    main(args.out)