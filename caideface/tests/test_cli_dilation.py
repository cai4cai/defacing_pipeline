"""Forward the shared device without changing serial CLI behaviour."""
import sys
import pytest
from caideface import cli


@pytest.mark.parametrize('command', ['run', 'skull-strip'])
@pytest.mark.parametrize('device', [None, 'cpu', 'cuda'])
def test_device_forwarding(monkeypatch, command, device):
    seen = {}
    class Pipeline:
        def __init__(self, **kwargs):
            seen.update(kwargs)
        def run(self, *args, **kwargs):
            return {}
    def strip(*args, **kwargs):
        seen.update(kwargs)
    monkeypatch.setattr(cli, 'DefacePipeline', Pipeline)
    monkeypatch.setattr(cli, 'skull_strip_batch', strip)
    argv = ['caideface', command, 'input', 'output', '--modality', 'mri']
    if command == 'run':
        argv += ['--brainsfit', 'fit', '--brainsresample', 'resample']
    if device:
        argv += ['--device', device]
    monkeypatch.setattr(sys, 'argv', argv)
    cli.main()
    assert seen['device'] == device
