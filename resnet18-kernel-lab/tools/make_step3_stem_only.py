import torch
import torchvision.models as models
import numpy as np
import argparse
import os

def save_bin_f32(path, tensor):
    arr = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
    with open(path, "wb") as f:
        f.write(arr.tobytes())

@torch.no_grad()
def main(outdir):
    os.makedirs(outdir, exist_ok=True)

    # 1. 모델 로드: 꼭 exports랑 같은 weight여야 한다.
    model = models.resnet18(weights="IMAGENET1K_V1").eval()

    # 2. C++에서 쓰고 있는 입력 그대로 로드
    x = np.fromfile(os.path.join(outdir, "sample_input.bin"), dtype=np.float32)
    x = torch.from_numpy(x).view(1,3,224,224)

    # 3. stem만 forward
    y = model.conv1(x)
    y = model.bn1(y)
    y = model.relu(y)
    y = model.maxpool(y)

    # 4. 저장
    save_bin_f32(os.path.join(outdir, "py_stem_after_pool.bin"), y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    main(args.outdir)
