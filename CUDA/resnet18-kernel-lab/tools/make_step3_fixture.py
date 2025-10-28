import torch
import torchvision.models as models
import numpy as np

def save_bin_f32(path, tensor):
    arr = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
    with open(path, "wb") as f:
        f.write(arr.tobytes())

@torch.no_grad()
def dump_stem(outdir):
    model = models.resnet18(weights="IMAGENET1K_V1").eval()

    # C++에서 쓰는 sample_input.bin 다시 사용
    x = np.fromfile(f"{outdir}/sample_input.bin", dtype=np.float32)
    x = torch.from_numpy(x).view(1,3,224,224)

    # stem forward
    y = model.conv1(x)
    y = model.bn1(y)
    y = model.relu(y)
    y = model.maxpool(y)

    # 이걸 저장
    save_bin_f32(f"{outdir}/py_stem_after_pool.bin", y)

if __name__ == "__main__":
    dump_stem("exports/resnet18/fp32")
