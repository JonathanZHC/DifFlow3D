# Exploratory benchmark (2026-10-02): runs inside the perceptive-safety-filter container with /tmp/real = ~/datasets/scene_flow/real (human pairs + pseudo GT). Monkeypatches the model; not wired into the pipeline.
import sys, time, os
sys.path.insert(0, "/opt/DifFlow3D")
import torch
from difflow3d.model.difflow import PointConvBidirection
from difflow3d.runtime.checkpoint import load_checkpoint
from difflow3d.runtime.inference import configure_fast_inference
N = int(os.environ.get("N", 8192)); it = tuple(int(x) for x in os.environ.get("IT", "4,2,2").split(","))
configure_fast_inference(True)
m = PointConvBidirection(iters=4, coarse_iters=it[0], middle_iters=it[1], fine_iters=it[2])
load_checkpoint(m, "/opt/DifFlow3D/checkpoints/model_difflow_355_0.0114.pth", strict=True); m.cuda().eval()
for l in m.modules():
    if isinstance(l, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)): l.track_running_stats = False
torch.manual_seed(0)
a = torch.randn(1, N, 3, device="cuda") * 0.25; b = a + 0.02 * torch.randn_like(a)
def step():
    eb = m.encode_frame(b, b)                      # streaming: one encode per frame
    return m.decode_pair(ea, eb, None, 0.1)
with torch.inference_mode():
    ea = m.encode_frame(a, a)
    for _ in range(10): step()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(30): step()
    torch.cuda.synchronize(); print(f"eager N={N} iters={it}: {(time.time()-t)/30*1e3:.2f} ms/frame (encode+decode)")
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        for _ in range(10): step()
        torch.cuda.synchronize()
    rows = sorted(p.key_averages(), key=lambda r: -r.device_time_total)
    tot = sum(r.device_time_total for r in p.key_averages() if r.device_time_total > 0 and r.key.startswith(("void", "ampere", "sm", "cutlass", "triton", "at::", "fused", "grouping", "furthest", "knn", "indexSelect", "elementwise", "vectorized", "reduce", "gather", "unrolled", "cunn", "topk", "bitonic", "radix", "gemm", "sgemm", "cudnn", "nchw", "bn", "batch_norm")))
    print("top CUDA kernels (ms per frame, 10 frames):")
    for r in rows[:22]:
        if r.device_time_total > 0: print(f"  {r.device_time_total/10/1e3:6.2f} ms  x{r.count//10:4d}  {r.key[:110]}")
