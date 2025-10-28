import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse

@torch.no_grad()
def main(outdir):
    # 1) CUDA 실행과 동일한 입력 로드
    x = np.fromfile(os.path.join(outdir, "sample_input.bin"), dtype=np.float32)
    x = torch.from_numpy(x).view(1,3,224,224)  # NCHW

    # conv1 설정 (ResNet18 stem)
    kH, kW = 7,7
    sH, sW = 2,2
    pH, pW = 3,3

    # 2) unfold == im2col
    # 결과 shape: [N, C*kH*kW, OH*OW] = [1,147,12544]
    patches = F.unfold(
        x,
        kernel_size=(kH,kW),
        dilation=1,
        padding=(pH,pW),
        stride=(sH,sW)
    )

    # 우리는 N=1만 다루고 있으므로 squeeze해도 된다
    patches = patches[0].contiguous()  # [147, 12544]

    # 3) 저장 (row-major로 그냥 dump)
    patches_np = patches.cpu().numpy().astype(np.float32)
    patches_np.tofile(os.path.join(outdir, "py_im2col.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    main(args.outdir)
