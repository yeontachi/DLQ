# resnet18_feat.py
import os
import math
import argparse
from typing import Dict

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import numpy as np


def load_image_tensor(img_path: str | None, size: int = 224) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    if img_path and os.path.isfile(img_path):
        img = Image.open(img_path).convert("RGB")
    else:
        # 이미지가 없으면 단색 더미로 실행 (기능 테스트용)
        img = Image.new("RGB", (size, size), color=(200, 180, 160))
    return tfm(img).unsqueeze(0)  # [1,3,H,W]


def pick_topk_channels(feat_bchw: torch.Tensor, k: int = 16) -> torch.Tensor:
    """
    feat_bchw: [1, C, H, W]
    채널별 L2-노름 에너지 기준 Top-K 채널을 선택해 [K, H, W] 반환
    """
    assert feat_bchw.ndim == 4 and feat_bchw.size(0) == 1
    _, C, H, W = feat_bchw.shape
    f = feat_bchw[0]                              # [C,H,W]
    energy = f.view(C, -1).norm(p=2, dim=1)       # [C]
    k = min(k, C)
    top_idx = torch.topk(energy, k=k, dim=0).indices
    return f[top_idx]                              # [K,H,W]


def normalize_per_channel(feat_chw: torch.Tensor) -> torch.Tensor:
    """
    feat_chw: [C, H, W]
    각 채널을 0~1로 정규화
    """
    f = feat_chw.clone()
    C = f.shape[0]
    for i in range(C):
        x = f[i]
        mn, mx = x.min(), x.max()
        if (mx - mn) > 1e-12:
            f[i] = (x - mn) / (mx - mn)
        else:
            f[i].zero_()
    return f


def save_grid_from_features(feat_bchw: torch.Tensor, title: str,
                            save_dir: str = "./feat_out",
                            num: int = 16, nrow: int | None = None) -> str:
    """
    matplotlib 없이 그리드 저장 (PIL 사용)
    """
    os.makedirs(save_dir, exist_ok=True)
    # Top-K 선택 및 정규화
    top = pick_topk_channels(feat_bchw.detach().cpu(), k=num)   # [K,H,W]
    top = normalize_per_channel(top).unsqueeze(1)               # [K,1,H,W]

    # nrow 자동 추정(정사각형 그리드에 가깝게)
    if nrow is None:
        k = top.size(0)
        nrow = int(math.sqrt(k)) or 1

    grid = make_grid(top, nrow=nrow, padding=2)                 # [3,Hg,Wg]
    img = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    out_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    Image.fromarray(img).save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="", help="테스트 이미지 경로(없으면 더미 생성)")
    parser.add_argument("--save_dir", type=str, default="./feat_out", help="출력 저장 폴더")
    parser.add_argument("--topk", type=int, default=16, help="각 레이어에서 저장할 채널 개수")
    parser.add_argument("--use_pretrained", action="store_true",
                        help="ImageNet 사전학습 가중치 사용")
    args = parser.parse_args()

    # 모델 로드 (torchvision ResNet18)
    if args.use_pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights).eval()
    else:
        model = models.resnet18(weights=None).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 추출할 레이어 지정 (forward hook)
    # - stem과 각 stage의 블록 출력
    target_layers = [
        "conv1", "bn1", "relu", "maxpool",
        "layer1.0", "layer1.1",
        "layer2.0", "layer2.1",
        "layer3.0", "layer3.1",
        "layer4.0", "layer4.1",
    ]

    features: Dict[str, torch.Tensor] = {}

    def get_hook(name):
        def hook(_m, _inp, out):
            # 저장은 CPU로
            features[name] = out.detach().cpu()
        return hook

    # 모듈 경로로 객체 얻는 유틸
    def get_module_by_name(root: nn.Module, name: str) -> nn.Module:
        m = root
        for p in name.split("."):
            if p.isdigit():
                m = m[int(p)]
            else:
                m = getattr(m, p)
        return m

    # hook 등록
    hooks = []
    for nm in target_layers:
        mod = get_module_by_name(model, nm)
        h = mod.register_forward_hook(get_hook(nm))
        hooks.append(h)

    # 입력 텐서 준비 (이미지 없으면 더미 생성)
    x = load_image_tensor(args.img if args.img else None, size=224).to(device)

    # forward
    with torch.no_grad():
        _ = model(x)

    # hook 해제
    for h in hooks:
        h.remove()

    # 디버그: 어떤 키가 캡쳐되었는지
    print("[DEBUG] captured layers:", list(features.keys()))
    for k, v in features.items():
        print(f"[DEBUG] {k}: shape={tuple(v.shape)}")

    # 저장
    os.makedirs(args.save_dir, exist_ok=True)
    saved_paths = []
    for nm, feat in features.items():
        try:
            path = save_grid_from_features(
                feat_bchw=feat,
                title=f"{nm}_Top{args.topk}",
                save_dir=args.save_dir,
                num=args.topk,
                nrow=None
            )
            saved_paths.append(path)
            print(f"[Saved] {path}")
        except Exception as e:
            print(f"[WARN] failed to save {nm}: {e}")

    print("\n=== Summary ===")
    print("Save dir:", os.path.abspath(args.save_dir))
    if saved_paths:
        for p in saved_paths:
            print(" -", p)
    else:
        print("No files saved. Check hooks / inputs / permissions.")


if __name__ == "__main__":
    main()
