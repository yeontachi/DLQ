import torch
import torchvision.models as models
import numpy as np
import os
import argparse

def save_bin_f32(path, tensor):
    arr = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
    with open(path, "wb") as f:
        f.write(arr.tobytes())

@torch.no_grad()
def main(outdir):
    os.makedirs(outdir, exist_ok=True)

    model = models.resnet18(weights="IMAGENET1K_V1").eval()

    # C++에서 쓰는 input 그대로 로드
    x = np.fromfile(os.path.join(outdir, "sample_input.bin"), dtype=np.float32)
    x = torch.from_numpy(x).view(1,3,224,224)

    # conv1만 적용
    y_conv1 = model.conv1(x)  # shape [1,64,112,112]

    save_bin_f32(os.path.join(outdir, "py_conv1_before_bn.bin"), y_conv1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    main(args.outdir)
