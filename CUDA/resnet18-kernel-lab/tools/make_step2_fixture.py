# tools/make_step2_fixture.py
import os, argparse, torch, numpy as np
from torchvision.models import resnet18, ResNet18_Weights

def main(manifest_dir):
    out_dir = os.path.join(manifest_dir, "fixtures")
    os.makedirs(out_dir, exist_ok=True)

    # 1) 모델과 가중치
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().float()
    sd = m.state_dict()

    # 2) 입력 텐서: 고정 시드로 재현성 있는 난수 입력 (1,3,224,224)
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        y = m.relu(m.bn1(m.conv1(x)))  # conv1 -> bn1 -> relu

    # 3) 저장
    x.cpu().numpy().tofile(os.path.join(out_dir, "input.bin"))
    y.cpu().numpy().tofile(os.path.join(out_dir, "expected.bin"))

    # 4) conv1/bn1 가중치도 이 디렉토리서 참조하므로, Step1에서 이미 export된 파일을 그대로 사용
    print("fixture saved:")
    print(" ", os.path.join(out_dir, "input.bin"))
    print(" ", os.path.join(out_dir, "expected.bin"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="exports/resnet18/fp32")
    args = ap.parse_args()
    main(args.manifest)
