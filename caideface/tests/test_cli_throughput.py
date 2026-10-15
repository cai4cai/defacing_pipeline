"""Do not report success when a stage catches and records an inference failure."""
import sys

import pandas as pd
import pytest

from caideface import cli


def test_serial_run_reports_skull_strip_failure(monkeypatch):
    class Pipeline:
        def __init__(self, **kwargs):
            assert kwargs["dilation_backend"] == "cuda"

        def run(self, *args, **kwargs):
            return {"skull_strip_log": pd.DataFrame([{
                "brain_extract": "success", "mask": "failed", "dilated": "failed",
            }]), "failed_defacing": []}

    monkeypatch.setattr(cli, "DefacePipeline", Pipeline)
    monkeypatch.setattr(sys, "argv", ["caideface", "run", "in", "out", "--modality", "mri",
                                      "--brainsfit", "fit", "--brainsresample", "resample",
                                      "--dilation-backend", "cuda"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_stage_log_success_and_failure():
    assert not cli._failed_stage_logs({})
    assert cli._failed_stage_logs({"reorient_log": pd.DataFrame({"success": [False]})})
    assert cli._failed_stage_logs({"skull_strip_log": pd.DataFrame()})
    assert not cli._failed_stage_logs({"skull_strip_log": pd.DataFrame([{
        "brain_extract": "skipped", "mask": "success", "dilated": "success",
    }])})
