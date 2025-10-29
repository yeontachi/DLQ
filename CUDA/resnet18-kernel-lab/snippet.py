import numpy as np
ref = np.fromfile("exports/resnet18/fp32/fixtures_step8/logits.bin", dtype=np.float32)  # Torch
run = np.fromfile("out/step8_logits.bin", dtype=np.float32)  # (선택) 우리 코드가 저장하게 했다면
d = np.abs(ref - run)
print("max_abs:", d.max(), "mean_abs:", d.mean())
