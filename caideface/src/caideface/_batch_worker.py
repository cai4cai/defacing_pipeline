"""Private subprocess entry point; GPU visibility is set before Python starts."""

import json
import os
from pathlib import Path
import sys


def run(config_path):
    os.umask(0o077)
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker requires exactly one usable CUDA device")
    torch.ones((16, 16), device="cuda").square().sum().item()
    from .pipeline import DefacePipeline
    result = DefacePipeline(**config["pipeline_options"]).run(config["input"], config["output"])
    reorient = result.get("reorient_log")
    skull = result.get("skull_strip_log")
    success = (reorient is not None and len(reorient) == 1 and bool(reorient["success"].all())
               and skull is not None and len(skull) == 1
               and all(skull[column].isin(["success", "skipped"]).all()
                       for column in ("brain_extract", "mask", "dilated"))
               and "failed_defacing" in result and not result["failed_defacing"]
               and len(list((Path(config["output"]) / "defaced").rglob("*_masked.nii.gz"))) == 1)
    # A swallowed stage exception must never become a successful process exit.
    (config_path.parent / "result.json").write_text(json.dumps({"success": bool(success), "stage_seconds": result["stage_seconds"]}))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
