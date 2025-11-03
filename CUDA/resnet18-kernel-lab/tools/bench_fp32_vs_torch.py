# tools/bench_fp32_vs_torch.py
import argparse, os, time, glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from subprocess import run, PIPE

def list_images(root, limit=None, shuffle=True):
    exts = ('*.JPEG','*.JPG','*.jpg','*.jpeg','*.png','*.bmp')
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(root, e))
    if not paths:
        for e in exts:
            paths += glob.glob(os.path.join(root, '**', e), recursive=True)
    if shuffle:
        import random; random.shuffle(paths)
    if not paths:
        raise FileNotFoundError(f'No images under {root}')
    return paths[:limit] if limit else paths

def preprocess_pil(img):
    w = ResNet18_Weights.DEFAULT
    mean, std = w.transforms().mean, w.transforms().std
    tfm = T.Compose([T.Resize(256), T.CenterCrop(224),
                     T.ToTensor(), T.Normalize(mean=mean, std=std)])
    return tfm(img).unsqueeze(0).cuda()

@torch.no_grad()
def torch_gap(model, x):
    m = model
    y = m.relu(m.bn1(m.conv1(x)))
    y = m.maxpool(y)
    y = m.layer1(y); y = m.layer2(y); y = m.layer3(y); y = m.layer4(y)
    gap = F.adaptive_avg_pool2d(y, (1,1)).flatten(1)  # 1x512
    return gap.squeeze(0).float().cpu().numpy()

def run_cuda_bin(bin_path, manifest, gap_vec, out_dir='out'):
    os.makedirs(out_dir, exist_ok=True)
    tmp_gap = os.path.join(out_dir, 'tmp_gap.bin')
    gap_vec.astype(np.float32).tofile(tmp_gap)
    cmd = [bin_path, '--manifest', manifest, '--gap', tmp_gap, '--save_logits']
    p = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f'CUDA bin failed:\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}')
    logits_path = os.path.join(out_dir, 'step8_logits.bin')
    logits = np.fromfile(logits_path, dtype=np.float32)
    if logits.size != 1000:
        raise RuntimeError(f'bad logits size {logits.size} (expect 1000). STDOUT:\n{p.stdout}')
    return logits, p.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--bin', required=True)
    ap.add_argument('--imagenet_dir', required=True)
    ap.add_argument('--limit', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--iters', type=int, default=1)
    args = ap.parse_args()

    device = 'cuda'
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()

    imgs = list_images(args.imagenet_dir, limit=args.limit, shuffle=True)
    print(f'[INFO] images={len(imgs)} under {args.imagenet_dir}')

    agree = 0; coses = []; torch_ms = []; cuda_ms = []

    for i, pth in enumerate(imgs, 1):
        img = Image.open(pth).convert('RGB')
        x = preprocess_pil(img)

        t0 = time.time()
        gap = torch_gap(model, x)

        with torch.no_grad():
            logits_t = model.fc(torch.from_numpy(gap).to(device).unsqueeze(0)) \
                             .squeeze(0).detach().cpu().numpy()
        
        t1 = time.time()

        c0 = time.time()
        logits_c, _ = run_cuda_bin(args.bin, args.manifest, gap, out_dir='out')
        c1 = time.time()

        torch_ms.append( (t1-t0)*1000.0 )
        cuda_ms.append( (c1-c0)*1000.0 )

        top_t = int(np.argmax(logits_t))
        top_c = int(np.argmax(logits_c))
        if top_t == top_c: agree += 1

        num = float((logits_t * logits_c).sum())
        den = float(np.linalg.norm(logits_t) * np.linalg.norm(logits_c) + 1e-12)
        coses.append(num/den)

        if i % 50 == 0:
            print(f'[${i}/{len(imgs)}] agree_top1={agree} ({agree*100.0/i:.2f}%) '
                  f'torch={np.mean(torch_ms[-50:]):.2f} ms, cuda={np.mean(cuda_ms[-50:]):.2f} ms, '
                  f'cos={np.mean(coses[-50:]):.4f}')

    print('-----')
    print(f'images={len(imgs)}')
    print(f'agree_top1={agree} ({agree*100.0/len(imgs):.2f}%)')
    print(f'torch_ms={np.mean(torch_ms):.2f}')
    print(f'cuda_ms={np.mean(cuda_ms):.2f}')
    print(f'speedup={np.mean(torch_ms)/np.mean(cuda_ms):.2f}x')
    print(f'cosine={np.mean(coses):.4f}')

if __name__ == '__main__':
    main()
