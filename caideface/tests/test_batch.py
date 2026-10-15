"""Exercise real subprocess scheduling with synthetic workers, without MRI/GPU I/O."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from caideface.batch import resolve_gpu_devices, run_parallel


@pytest.mark.parametrize("devices,environ,expected", [
    (["0", "1"], {}, ["0", "1"]),
    (["0", "1"], {"CUDA_VISIBLE_DEVICES": "3,7"}, ["3", "7"]),
    (["1"], {"CUDA_VISIBLE_DEVICES": "GPU-12345678-1234-1234-1234-123456789abc,4"}, ["4"]),
])
def test_visibility(devices, environ, expected):
    assert resolve_gpu_devices(devices, environ) == expected


@pytest.mark.parametrize("devices,environ", [
    (["0", "0"], {}), (["0", "00"], {}), (["all"], {}), (["-1"], {}), ([], {}),
    (["0"], {"CUDA_VISIBLE_DEVICES": ""}),
    (["2"], {"CUDA_VISIBLE_DEVICES": "3,7"}),
    (["0"], {"CUDA_VISIBLE_DEVICES": "-1"}),
])
def test_invalid_visibility(devices, environ):
    with pytest.raises(ValueError):
        resolve_gpu_devices(devices, environ)


def inputs(tmp_path):
    source = tmp_path / "input"
    for name in ("a", "b", "c", "d"):
        (source / name).mkdir(parents=True)
        (source / name / "same.nii.gz").write_text("synthetic scheduler fixture")
    return source


def stub_workers(monkeypatch, sleep=0.08, failure=False):
    original = subprocess.Popen
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    code = '''import json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[1]).parent
(p/'stub.json').write_text(json.dumps({'gpu':os.environ['CUDA_VISIBLE_DEVICES'], 'threads':os.environ['OMP_NUM_THREADS'], 'pid':os.getpid(), 'start':time.monotonic()}))
time.sleep(float(sys.argv[2]))
(p/'result.json').write_text(json.dumps({'success':sys.argv[3]=='ok'}))
sys.exit(0 if sys.argv[3]=='ok' else 2)
'''

    def launch(command, **kwargs):
        config = command[-1]
        return original([sys.executable, "-c", code, config, str(sleep),
                         "fail" if failure else "ok"], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_disjoint_jobs_and_gpu_assignment(tmp_path, monkeypatch):
    source = inputs(tmp_path)
    stub_workers(monkeypatch, sleep=0.15)
    output = tmp_path / "output"
    result = run_parallel(source, output, {}, ["0", "1"], threads_per_worker=2)
    assert result["successful"] == result["completed"] == result["inputs"] == 4
    assert len({x["input"] for x in result["jobs"]}) == 4
    assert len({x["output"] for x in result["jobs"]}) == 4
    records = [json.loads(p.read_text()) for p in output.glob("jobs/*/stub.json")]
    assert {x["gpu"] for x in records} == {"0", "1"}
    assert {x["threads"] for x in records} == {"2"}
    assert len({x["pid"] for x in records}) == 4
    assert json.loads((output / "parallel_summary.json").read_text())["successful"] == 4
    with pytest.raises(FileExistsError):
        run_parallel(source, output, {}, ["0"])


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_shared_gpu_and_failure_propagation(tmp_path, monkeypatch):
    source = inputs(tmp_path)
    stub_workers(monkeypatch, failure=True)
    result = run_parallel(source, tmp_path / "out", {}, ["0"], workers_per_gpu=2)
    assert result["completed"] == 4
    assert result["successful"] == 0
    assert {x["exit_code"] for x in result["jobs"]} == {2}
    assert {x["gpu"] for x in result["jobs"]} == {"0"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_timeout_reaps_workers(tmp_path, monkeypatch):
    source = inputs(tmp_path)
    stub_workers(monkeypatch, sleep=30)
    result = run_parallel(source, tmp_path / "out", {}, ["0"], job_timeout=0.1)
    assert result["successful"] == 0
    assert all(x["timed_out"] and x["exit_code"] < 0 for x in result["jobs"])


def test_reject_overlap_and_invalid_options(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    source = inputs(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        run_parallel(source, source / "output", {}, ["0"])
    with pytest.raises(ValueError, match="positive integers"):
        run_parallel(source, tmp_path / "out", {}, ["0"], workers_per_gpu=0)
    with pytest.raises(ValueError, match="MRI only"):
        run_parallel(source, tmp_path / "out", {"modality": "ct"}, ["0"])
    with pytest.raises(ValueError, match="CUDA inference"):
        run_parallel(source, tmp_path / "out", {"device": "cpu"}, ["0"])


@pytest.mark.skipif(sys.platform != "linux", reason="Uses /proc to verify descendant termination")
def test_timeout_kills_descendant_process(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    (source / "scan.nii.gz").write_text("synthetic scheduler fixture")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    original = subprocess.Popen
    code = '''import pathlib,subprocess,sys,time
p=pathlib.Path(sys.argv[1]).parent
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
(p/'child_pid').write_text(str(child.pid))
time.sleep(30)
'''

    def launch(command, **kwargs):
        return original([sys.executable, "-c", code, command[-1]], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
    output = tmp_path / "out"
    result = run_parallel(source, output, {}, ["0"], job_timeout=0.5)
    assert result["successful"] == 0
    pid_file = next(output.glob("jobs/*/child_pid"))
    status = Path("/proc") / pid_file.read_text() / "status"
    for _ in range(20):
        try:
            state = next(line for line in status.read_text().splitlines() if line.startswith("State:"))
        except FileNotFoundError:
            return
        if "Z (zombie)" in state or "X (dead)" in state:
            return
        time.sleep(0.05)
    pytest.fail("Timed-out worker left a running descendant")
