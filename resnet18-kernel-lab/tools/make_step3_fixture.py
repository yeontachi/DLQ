import torch
import torch.nn as nn
import torchvision.models as models
import os
import numpy as np
import argparse

# 동일한 저장 형식: float32 little-endian binary dump
def save_bin_f32(path, tensor):
    arr = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
    with open(path, "wb") as f:
        f.write(arr.tobytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True,
                        help="exports/resnet18/fp32")
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # 1. 모델 로드 (fp32, eval)
    model = models.resnet18(weights=None)  # or weights='IMAGENET1K_V1' if you used pretrained before
    model.eval()

    # 2. 더미 입력 (Step2와 동일한 규약 사용)
    #    N=1, C=3, H=224, W=224
    #    만약 Step2 때 real 이미지 normalize 한 텐서를 썼다면
    #    똑같은 걸 여기서 로드해서 써도 됨.
    #
    #    지금은 일단 zeros or random으로 고정 (테스트 일관성용).
    #    실제로는 Step2에서 쓰던 input.bin과 동일한 텐서를 로드해서 써야 완전히 맞는다.
    #
    x = torch.randn(1,3,224,224, dtype=torch.float32)

    # 3. stem + block0만 forward 수동 구현
    # conv1 -> bn1 -> relu -> maxpool -> layer1[0]

    with torch.no_grad():
        out = model.conv1(x)
        out = model.bn1(out)
        out = model.relu(out)
        out = model.maxpool(out)
        out = model.layer1[0](out)  # basic block 0

    # 4. 저장
    # input (for --input)
    save_bin_f32(os.path.join(outdir, "sample_input.bin"), x)

    # expected output after block0 (for --expect)
    save_bin_f32(os.path.join(outdir, "sample_block0_out.bin"), out)

    print("wrote:")
    print(" ", os.path.join(outdir, "sample_input.bin"))
    print(" ", os.path.join(outdir, "sample_block0_out.bin"))

if __name__ == "__main__":
    main()
