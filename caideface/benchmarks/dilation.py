"""Synthetic dilation comparison; no images or model weights required."""
import argparse
import json
import time

import numpy as np

from caideface.morphology import rectangular_dilation, validate_dilation_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, default=(96, 96, 64))
    parser.add_argument("--kernel", type=int, nargs=3, default=(14, 14, 14))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if min(*args.shape, *args.kernel, args.repeats) < 1:
        parser.error("Dimensions and repeats must be positive")
    rng = np.random.default_rng(42)
    mask = rng.random(args.shape) < 0.01
    structure = np.ones(args.kernel, dtype=np.uint8)
    backends = ["scipy", "separable_cpu"] + (["cuda"] if args.cuda else [])
    startup = time.perf_counter()
    for backend in backends:
        validate_dilation_backend(backend)
    startup = time.perf_counter() - startup
    reference = rectangular_dilation(mask, structure)
    timings = {backend: [] for backend in backends}
    for backend in backends:
        rectangular_dilation(mask, structure, backend)
    for _ in range(args.repeats):
        for backend in rng.permutation(backends):
            start = time.perf_counter()
            actual = rectangular_dilation(mask, structure, backend)
            elapsed = time.perf_counter() - start
            np.testing.assert_array_equal(actual, reference)
            timings[backend].append(elapsed)
    print(json.dumps({"shape": args.shape, "kernel": args.kernel,
                      "seed": 42, "foreground_probability": 0.01,
                      "preflight_seconds": startup,
                      "dilation_seconds": timings}, indent=2))


if __name__ == "__main__":
    main()
