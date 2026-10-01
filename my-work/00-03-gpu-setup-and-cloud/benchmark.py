import time

import torch

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    # Exercise 1: the lesson's benchmark, first GPU call (includes one-time CUDA warm-up)
    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    cold_time = time.time() - start
    print(f"GPU (first call): {cold_time:.3f}s  speedup {cpu_time / cold_time:.0f}x")

    # Same multiply again, now that the GPU is warmed up
    runs = 10
    start = time.time()
    for _ in range(runs):
        c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / runs
    print(f"GPU (warmed up):  {gpu_time:.4f}s  speedup {cpu_time / gpu_time:.0f}x")

    # Exercise 3: largest model that fits, at 2 bytes per parameter (fp16)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    print(f"\nVRAM: {vram_bytes / 1e9:.1f} GB ({vram_bytes / 2**30:.1f} GiB)")
    print(f"Max params at fp16 (2 bytes each): ~{vram_bytes / 2 / 1e9:.1f}B")
    print(f"Max params at fp32 (4 bytes each): ~{vram_bytes / 4 / 1e9:.1f}B")
