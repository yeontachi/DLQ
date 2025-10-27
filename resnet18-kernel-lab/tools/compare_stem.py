import numpy as np

cuda_bin = "debug_stem_after_pool.bin"
torch_bin = "exports/resnet18/fp32/py_stem_after_pool.bin"

a = np.fromfile(cuda_bin, dtype=np.float32)
b = np.fromfile(torch_bin, dtype=np.float32)

print("cuda shape:", a.shape)
print("torch shape:", b.shape)

if a.shape != b.shape:
    print("SHAPE MISMATCH!")
else:
    diff = np.abs(a - b)
    print("max_abs:", diff.max())
    print("mean_abs:", diff.mean())
