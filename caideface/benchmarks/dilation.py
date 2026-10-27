"""Synthetic dilation comparison; no images or model weights required."""
import argparse
import json
import time

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from caideface.morphology import rectangular_dilation, validate_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, default=(96, 96, 64))
    parser.add_argument("--kernel", type=int, nargs=3, default=(14, 14, 14))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--pattern", choices=("random", "ellipsoid"), default="random")
    args = parser.parse_args()
    if min(*args.shape, *args.kernel, args.repeats) < 1:
        parser.error("Dimensions and repeats must be positive")
    rng = np.random.default_rng(42)
    if args.pattern == "random":
        mask = rng.random(args.shape) < 0.01
    else:
        # A centred solid ellipsoid with radii one third of each image dimension.
        axes = [(np.arange(n) - (n - 1) / 2) / (n / 3) for n in args.shape]
        mask = (axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2
                + axes[2][None, None, :] ** 2) <= 1
    structure = np.ones(args.kernel, dtype=np.uint8)
    methods = ["scipy", "cpu"] + (["cuda"] if args.cuda else [])
    startup = time.perf_counter()
    for method in methods:
        if method != "scipy":
            validate_device(method)
    startup = time.perf_counter() - startup
    def dilate(method):
        if method == "scipy":
            return binary_dilation(mask, structure=structure)
        tensor = torch.as_tensor(mask, device=method)
        return rectangular_dilation(tensor, structure).cpu().numpy()

    reference = dilate("scipy")
    timings = {method: [] for method in methods}
    for method in methods:
        dilate(method)
    for _ in range(args.repeats):
        for method in rng.permutation(methods):
            start = time.perf_counter()
            actual = dilate(method)
            elapsed = time.perf_counter() - start
            np.testing.assert_array_equal(actual, reference)
            timings[method].append(elapsed)
    print(json.dumps({"shape": args.shape, "kernel": args.kernel,
                      "seed": 42, "pattern": args.pattern,
                      "foreground_fraction": float(mask.mean()),
                      "preflight_seconds": startup,
                      "dilation_seconds": timings}, indent=2))


if __name__ == "__main__":
    main()
