# tools/bench_fp32_vs_torch_fast_nio.py
import os, time, ctypes, glob, random
import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as T
from PIL import Image

# ===== paths =====
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB  = os.path.join(ROOT, "build", "fp32", "libfc.so")

# ===== FC shared lib =====
fc = ctypes.cdll.LoadLibrary(LIB)
fc.fc_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
fc.fc_init.restype  = ctypes.c_int
fc.fc_cleanup.argtypes = []
fc.fc_cleanup.restype  = None
fc.fc_forward_batched.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
fc.fc_forward_batched.restype  = ctypes.c_int

def load_bin_f32(path, n=None):
    arr = np.fromfile(path, dtype=np.float32)
    if n is not None and arr.size != n:
        raise RuntimeError(f"size mismatch: {path} {arr.size} != {n}")
    return arr

def build_preprocess(weights):
    """
    torchvision 버전별 안전한 전처리 구성:
    - 우선 weights.transforms() 시도
    - 실패하면 ImageNet 기본 mean/std로 폴백
    """
    # 1) weights.transforms() 지원되면 그걸 사용
    try:
        tr = weights.transforms()
        return tr
    except Exception:
        pass

    # 2) meta에서 꺼내기 시도
    mean = None
    std  = None
    try:
        m = getattr(weights, "meta", {})
        if isinstance(m, dict):
            mean = m.get("mean", None)
            std  = m.get("std", None)
    except Exception:
        pass

    # 3) 최종 폴백: 고전적인 ImageNet 통계
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

@torch.inference_mode()
def run_backbone_resnet18(model, preprocess, paths, device="cuda"):
    """
    resnet18의 conv~layer4~avgpool까지를 명시적으로 호출해서
    GAP(B,512)를 반환. (버전 상관 없이 동일하게 동작)
    """
    imgs = [preprocess(Image.open(p).convert("RGB")) for p in paths]
    x = torch.stack(imgs, dim=0).to(device, non_blocking=True)

    t0 = time.time()
    # stem
    x = model.conv1(x)
    x = model.bn1(x)
    x = torch.relu(x)
    x = model.maxpool(x)
    # layers
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    # GAP
    x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))  # (B,512,1,1)
    gap = torch.flatten(x, 1)                              # (B,512)
    torch.cuda.synchronize()
    t1 = time.time()
    return gap, (t1 - t0) * 1000.0  # ms

@torch.inference_mode()
def main(
    manifest = os.path.join(ROOT, "exports", "resnet18", "fp32"),
    imagenet_dir = os.path.join(ROOT, "data", "imagenet_val", "ILSVRC2012_img_val"),
    limit = 500,
    batch = 128,
    device = "cuda",
):
    # 1) 모델 & 전처리
    weights = tv.models.ResNet18_Weights.IMAGENET1K_V1
    model = tv.models.resnet18(weights=weights).to(device).eval()
    preprocess = build_preprocess(weights)

    # 2) FC 가중치/바이어스 로드 + 초기화
    W = load_bin_f32(os.path.join(manifest, "fc.weight.bin"))  # 1000*512
    B = load_bin_f32(os.path.join(manifest, "fc.bias.bin"))    # 1000
    O, I = 1000, 512
    assert W.size == O*I and B.size == O
    rv = fc.fc_init(W.ctypes.data_as(ctypes.c_void_p),
                    B.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(O), ctypes.c_int(I))
    if rv != 0:
        raise RuntimeError("fc_init failed")

    # 3) 이미지 목록
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    img_paths = [p for p in glob.glob(os.path.join(imagenet_dir, "*")) if p.lower().endswith(exts)]
    if len(img_paths) == 0:
        # class-subdir 형식 처리
        for d in sorted(os.listdir(imagenet_dir)):
            dd = os.path.join(imagenet_dir, d)
            if os.path.isdir(dd):
                img_paths += [p for p in glob.glob(os.path.join(dd, "*")) if p.lower().endswith(exts)]
    random.seed(0)
    random.shuffle(img_paths)
    img_paths = img_paths[:limit]
    print(f"[INFO] images={len(img_paths)} under {imagenet_dir}")

    # 4) 루프: Torch 백본(GAP) + CUDA FC
    agree = 0
    total_torch_ms = 0.0
    total_cuda_ms  = 0.0

    for i in range(0, len(img_paths), batch):
        chunk = img_paths[i:i+batch]

        # 4-1) Torch 백본에서 GAP(B,512) 추출
        gap, torch_ms = run_backbone_resnet18(model, preprocess, chunk, device=device)
        Bcur = gap.shape[0]
        assert gap.shape[1] == I

        # 4-2) CUDA FC 실행 (디바이스 포인터 직접 전달)
        out = torch.empty((Bcur, O), device=device, dtype=torch.float32)
        dX_ptr   = ctypes.c_void_p(gap.data_ptr())
        dOut_ptr = ctypes.c_void_p(out.data_ptr())

        torch.cuda.synchronize()
        t0 = time.time()
        rv = fc.fc_forward_batched(dX_ptr, ctypes.c_int(Bcur), dOut_ptr)
        torch.cuda.synchronize()
        if rv != 0:
            raise RuntimeError("fc_forward_batched failed")
        cuda_ms = (time.time() - t0) * 1000.0

        # 4-3) 정확도 비교: Torch의 FC vs CUDA FC
        torch_logits = model.fc(gap)      # (B,O)
        top_torch = torch.argmax(torch_logits, dim=1)
        top_cuda  = torch.argmax(out, dim=1)
        agree += int((top_torch == top_cuda).sum().item())

        total_torch_ms += torch_ms
        total_cuda_ms  += cuda_ms

        done = i + Bcur
        print(f"[{done}/{len(img_paths)}] agree_top1={agree} ({agree/done*100:.2f}%) "
              f"torch={torch_ms/Bcur:.2f} ms/img, cudaFC={cuda_ms/Bcur:.3f} ms/img")

    n = len(img_paths)
    print("-----")
    print(f"images={n}")
    print(f"agree_top1={agree} ({agree/n*100:.2f}%)")
    print(f"torch_backbone_ms={total_torch_ms/n:.2f} ms/img")
    print(f"cuda_FC_ms={total_cuda_ms/n:.3f} ms/img")

    fc.fc_cleanup()

if __name__ == "__main__":
    main()
