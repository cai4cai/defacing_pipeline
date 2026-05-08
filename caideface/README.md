# caideface

**MRI defacing pipeline with skull-stripping and affine registration** from the [cai4cai](https://cai4cai.ml/) research group (Contextual Artificial Intelligence for Computer Assisted Interventions).

This pipeline anonymises head MRI scans by removing facial features while preserving brain structures, as described in the paper *"A Generalisable Head MRI Defacing Pipeline: Evaluation on 2,566 Meningioma Scans"* ([arXiv:2505.12999](https://arxiv.org/abs/2505.12999)).

## Pipeline overview

The pipeline consists of three steps:

1. **Reorientation** -- Aligns NIfTI scans to LAS canonical orientation (MNI152 standard) using nibabel.
2. **Skull-stripping** -- Extracts brain masks using [HD-BET](https://github.com/MIC-DKFZ/HD-BET), then applies dynamic dilation to preserve peripheral brain structures.
3. **Registration & Defacing** -- Registers each scan to the MNI152 template using BRAINSFit (affine), warps a face mask into the scan's space, and applies it to remove facial features.

The MNI152 skull-stripped template and face mask are **bundled with the package**, so no additional downloads are needed.

## Requirements

### Python

- Python >= 3.9

### External tools (not pip-installable)

| Tool | Used in | Install |
|------|---------|---------|
| **BRAINSFit** & **BRAINSResample** | Step 3 | Bundled with [3D Slicer](https://www.slicer.org/) |

> **Note:** Step 1 (reorientation) no longer requires FSL -- it uses nibabel's orientation tools to reorient scans to LAS (equivalent to `fslreorient2std`).

#### Finding BRAINSFit and BRAINSResample

These executables are included with 3D Slicer. Common locations:

- **macOS**: `/Applications/Slicer.app/Contents/lib/Slicer-5.8/cli-modules/BRAINSFit`
- **Linux**: `/path/to/Slicer/lib/Slicer-5.8/cli-modules/BRAINSFit`

Replace `5.8` with your installed Slicer version if different. To verify the executables are found and working:

```bash
# Check they exist
ls /Applications/Slicer.app/Contents/lib/Slicer-5.8/cli-modules/BRAINSFit
ls /Applications/Slicer.app/Contents/lib/Slicer-5.8/cli-modules/BRAINSResample

# Check they run (should print usage/help info)
/Applications/Slicer.app/Contents/lib/Slicer-5.8/cli-modules/BRAINSFit --help
/Applications/Slicer.app/Contents/lib/Slicer-5.8/cli-modules/BRAINSResample --help
```

You can also build them from source via [BRAINSTools](https://github.com/BRAINSia/BRAINSTools).

## Installation

We recommend using a conda environment:

```bash
conda create -n caideface python=3.10 -y
conda activate caideface
pip install caideface
```

Or install from GitHub:

```bash
pip install "caideface @ git+https://github.com/cai4cai/defacing_pipeline.git#subdirectory=caideface"
```

Or install from source:

```bash
git clone https://github.com/cai4cai/defacing_pipeline.git
cd defacing_pipeline/caideface
pip install -e .
```

> **Note:** caideface requires `numpy<2` (enforced automatically). Some dependencies (HD-BET / nnU-Net) are not yet compatible with NumPy 2.x.

## Usage

### CLI -- Full pipeline

Run all three steps in one command:

```bash
caideface run ./input_nifti ./output \
  --brainsfit /path/to/BRAINSFit \
  --brainsresample /path/to/BRAINSResample
```

This creates three subdirectories under `./output`:
- `reoriented/` -- Step 1 outputs
- `hdbet/` -- Step 2 outputs (skull-stripped, masks, dilated)
- `defaced/` -- Step 3 outputs (final defaced scans)

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | auto-detected | `cpu` or `cuda` for HD-BET |
| `--no-tta` | on | Disable HD-BET test-time augmentation (faster but less accurate) |
| `--dilation-mm` | `14.0` | Brain mask dilation in mm |
| `--background` | `0` | Fill value for defaced regions (0 for MRI, -1024 for CT) |
| `--template` | bundled | Custom MNI152 skull-stripped template |
| `--face-mask` | bundled | Custom face mask in MNI152 space |
| `--steps` | `all` | Run specific steps: `reorient`, `skull_strip`, `deface` (comma-separated) |
| `-v` | off | Verbose/debug logging |

### CLI -- Individual steps

Run each step separately for more control:

```bash
# Step 1: Reorientation
caideface reorient ./raw_nifti ./reoriented

# Step 2: Skull-stripping
caideface skull-strip ./reoriented ./hdbet --device cpu

# Step 3: Registration & Defacing
caideface deface ./reoriented ./hdbet ./defaced \
  --brainsfit /path/to/BRAINSFit \
  --brainsresample /path/to/BRAINSResample
```

## Output structure

```
output/
├── reoriented/
│   ├── reorientation_log.csv
│   └── <subject>/<scan>.nii.gz
├── hdbet/
│   ├── hd_bet_log.csv
│   └── <subject>/
│       ├── hd_bet_<scan>.nii.gz           # Skull-stripped
│       ├── hd_bet_mask_<scan>.nii.gz      # Dilated brain mask
│       └── hd_bet_dilated_<scan>.nii.gz   # Dilated skull-stripped
└── defaced/
    ├── not_defaced_scans.csv              # Only if failures occurred
    └── <subject>/
        └── hd_bet_dilated_<scan>_masked.nii.gz  # Final defaced scan
```

## Existing transforms

If you have pre-computed registration transforms (e.g. from 3D Slicer), place a file named `Transform_to_template.txt` in the same directory as the dilated skull-stripped scan. The pipeline will use it instead of running BRAINSFit. Both plain 4x4 text matrices and ITK/Slicer transform formats are supported.

## Citation

If you use this tool, please cite:

```bibtex
@article{caideface2025,
  title={A Generalisable Head MRI Defacing Pipeline: Evaluation on 2,566 Meningioma Scans},
  year={2025},
  url={https://arxiv.org/abs/2505.12999}
}
```

If you use HD-BET (skull-stripping, Step 2), please also cite:

```bibtex
@article{Isensee2019,
  author={Isensee, F. and Schell, M. and Tursunova, I. and Brugnara, G. and Bonekamp, D. and Neuberger, U. and Wick, A. and Schlemmer, H. P. and Heiland, S. and Wick, W. and Bendszus, M. and Maier-Hein, K. H. and Kickingereder, P.},
  title={Automated brain extraction of multi-sequence MRI using artificial neural networks},
  journal={Human Brain Mapping},
  year={2019},
  pages={1--13},
  doi={10.1002/hbm.24750}
}
```

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE.md) file for details.
