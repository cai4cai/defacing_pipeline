"""Forward the optional backend without changing serial CLI behaviour."""
import sys
import pytest
from caideface import cli


@pytest.mark.parametrize('command', ['run', 'skull-strip'])
@pytest.mark.parametrize('backend', [None, 'separable_cpu', 'cuda'])
def test_backend_forwarding(monkeypatch, command, backend):
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
    if backend:
        argv += ['--dilation-backend', backend]
    monkeypatch.setattr(sys, 'argv', argv)
    cli.main()
    assert seen['dilation_backend'] == (backend or 'scipy')
