# Matched original-3D dilation baseline

User requested method, batch time and speedup relative to main in PR #29.
Measured the actual original SciPy 3D operation and both PR backends on the same
six frozen inputs, with rotated backend order. Input loading, kernel construction
and parity checks are excluded; CUDA transfers are included after a tiny warm-up.
One pass per method; shared workstation with concurrent workloads. Detailed
inputs, hashes, source snapshot, commands and timings remain in private storage.

| Method | Batch time | Speedup relative to main |
|---|---:|---:|
| Original full 3D SciPy dilation (main) | 2209.94 s | 1.00× |
| Separable CPU (three 1D passes) | 26.20 s | 84.34× |
| Separable CUDA (three 1D passes) | 2.95 s | 749.46× |

All masks matched main exactly. Replaced the headline comparison with these
matched timings and explained separability as successive X/Y/Z 1D dilations.
Kept the theoretical 300× work reduction distinct from measured performance.
No implementation change. Existing 126-test CUDA-enabled verification applies;
this additional run verifies parity on all six frozen masks. PR stays draft.
