import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse

def load_bin_f32(path, shape):
    arr = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(arr).view(*shape)

@torch.no_grad()
def main(outdir):
    # 입력 그대로 로드
    x = np.fromfile(os.path.join(outdir, "sample_input.bin"), dtype=np.float32)
    x = torch.from_numpy(x).view(1,3,224,224)

    # 우리가 C++에서 쓰는 conv1.weight.bin 그대로 로드
    w = load_bin_f32(os.path.join(outdir, "conv1.weight.bin"),
                     (64,3,7,7))

    # conv2d 직접 수행 (bias 없음)
    y = F.conv2d(
        x,
        w,
        bias=None,
        stride=2,
        padding=3
    )  # -> [1,64,112,112]

    # 결과 저장
    y_np = y.contiguous().numpy().astype(np.float32)
    y_np.tofile(os.path.join(outdir, "py_conv1_before_bn_MATCH_EXPORT.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    main(args.outdir)
