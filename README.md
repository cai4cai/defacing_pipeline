# Defacing Pipeline

[![arXiv](https://img.shields.io/badge/arXiv-2505.12999-b31b1b.svg)](https://arxiv.org/abs/2505.12999)
[![PyPI](https://img.shields.io/pypi/v/caideface)](https://pypi.org/project/caideface/)

A head MRI defacing pipeline based on affine registration that removes facial features while preserving brain structures. Described in *"A Generalisable Head MRI Defacing Pipeline: Evaluation on 2,566 Meningioma Scans"* ([arXiv:2505.12999](https://arxiv.org/abs/2505.12999)).

<p align="center">
  <img src="Pipeline_smaller.svg" alt="Pipeline overview" width="100%">
</p>

## Table of Contents

- [Quick Start (pip)](#quick-start-pip)
- [Pipeline Overview](#pipeline-overview)
  - [Why Skull-Stripping?](#why-skull-stripping)
- [Using the Notebooks](#using-the-notebooks)
  - [Recommended: With Skull-Stripping](#recommended-with-skull-stripping)
  - [Legacy: Without Skull-Stripping](#legacy-without-skull-stripping)
- [Quality Assessment](#quality-assessment)
  - [3D Rendering](#step-4-3d-rendering-of-defaced-images)
  - [Manual Assessment](#step-5-manual-assessment)
  - [Manual Landmark Registration](#steps-6-7-manual-landmark-registration-fallback)
- [Requirements and Setup](#requirements-and-setup)
  - [Extracting BRAINSFit and BRAINSResample from 3D Slicer](#extracting-brainsfit-and-brainsresample-from-3d-slicer)
- [How to Cite](#how-to-cite)

---

## Quick Start (pip)

The fastest way to use the pipeline is via the [`caideface`](https://pypi.org/project/caideface/) Python package:

```bash
# MRI defacing only
pip install caideface

# MRI + CT defacing (includes TotalSegmentator)
pip install caideface[ct]
```

> **Note:** BRAINSFit and BRAINSResample are bundled with [3D Slicer](https://www.slicer.org/). See [Requirements and Setup](#requirements-and-setup) for how to locate or extract them.

### Deface MRI scans

```bash
caideface run ./input_nifti ./output \
  --modality mri \
  --brainsfit /path/to/BRAINSFit \
  --brainsresample /path/to/BRAINSResample
```

### Deface CT scans

```bash
caideface run ./input_nifti ./output \
  --modality ct \
  --brainsfit /path/to/BRAINSFit \
  --brainsresample /path/to/BRAINSResample
```

### Anonymise text reports

```bash
caideface anonymize ./reports ./anonymized_reports
```

All defacing commands create three subdirectories under `./output`:
- `reoriented/` -- reoriented scans
- `skullstripped/` -- skull-stripped scans and masks
- `defaced/` -- final defaced scans

For the full CLI reference (individual steps, all options, Python API, and output structure), see the [caideface package documentation](caideface/README.md).

---

## Pipeline Overview

The recommended pipeline consists of three steps:

1. **Reorientation** -- Aligns scans to LAS canonical orientation (MNI152 standard).
2. **Skull-stripping** -- Extracts a brain mask using [HD-BET](https://github.com/MIC-DKFZ/HD-BET), then dilates it (default 14 mm) to preserve peripheral brain structures. A new skull-stripped scan is generated using the dilated mask.
3. **Registration and Defacing** -- Registers each dilated skull-stripped scan to the bundled MNI152 template via affine registration ([BRAINSFit](https://github.com/BRAINSia/BRAINSTools)), warps the template face mask into the scan's space, and applies it to remove facial features.

The [template image](data/icbm152_ext55_model_sym_2020_nifti/icbm152_ext55_model_sym_2020/mni_icbm152_t1_tal_nlin_sym_55_ext.nii) is sourced from the [ICBM 152 Extended Nonlinear Atlases (2020)](https://nist.mni.mcgill.ca/icbm-152-extended-nonlinear-atlases-2020/). The [face mask](data/icbm152_ext55_model_sym_2020_nifti/icbm152_ext55_model_sym_2020/t1_mask.nii.gz) was created with [ITK-SNAP](http://www.itksnap.org) to remove facial features while retaining as much clinical information as possible.

### Why Skull-Stripping?

In the legacy pipeline, full head images are registered directly to the template. This works well for standard FOV images but often fails for reduced FOV scans (slabs), where the large anatomical differences between the input and template cause the affine registration to diverge.

By skull-stripping first, the registration focuses on brain anatomy only, which is far more consistent across subjects and FOV configurations. The brain mask dilation (14 mm by default) ensures that structures near the skull edge are preserved in the final defaced output.

---

## Using the Notebooks

> **Note:** Notebooks [1] and [2] must be run with the **3D Slicer Jupyter Notebook kernel**. To install the kernel, follow the instructions in the [SlicerJupyter](https://github.com/Slicer/SlicerJupyter) repository. Alternatively, they can be converted to Python scripts and run directly in 3D Slicer using the approach described [here](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#run-a-python-script-file-in-the-slicer-environment). Notebook [3] runs with a standard Python kernel.

### Recommended: With Skull-Stripping

The notebook [[1]deface_skullstripping.ipynb](%5B1%5Ddeface_skullstripping.ipynb) implements the full three-step pipeline described above:

1. **Reorientation** to MNI152 space
2. **Skull-stripping** with HD-BET and mask dilation
3. **Registration and defacing** using the dilated skull-stripped scans

Outputs are written to the [example_output_hdbet](data/example_output_hdbet) directory.

### Legacy: Without Skull-Stripping

The notebook [[1]deface_with_brainsfit.ipynb](%5B1%5Ddeface_with_brainsfit.ipynb) implements the original, simpler pipeline that registers full head images directly to the template -- without reorientation or skull-stripping.

In the second cell, set the paths to input images, output directories, and the BRAINSFit/BRAINSResample binaries. You can also provide a list of existing manual transforms for cases that previously failed:

```python
existing_transform_paths = [None, "./data/example_input_images/IXI002-Guys-0828-T2/Transform_to_template.txt"]
```

Outputs are written to the [example_output](data/example_output) directory.

> **Note:** This approach is less robust for reduced FOV images. Consider using the skull-stripping pipeline or the `caideface` CLI instead.

### Example Data

Both notebooks use two test images from the IXI dataset:
- [IXI002-Guys-0828-T1.nii.gz](data/example_input_images/IXI002-Guys-0828-T1/IXI002-Guys-0828-T1.nii.gz) (T1)
- [IXI002-Guys-0828-T2.nii.gz](data/example_input_images/IXI002-Guys-0828-T2/IXI002-Guys-0828-T2.nii.gz) (T2, includes a [manual transform](data/example_input_images/IXI002-Guys-0828-T2/Transform_to_template.txt))

---

## Quality Assessment

These steps are shared across both pipelines and are used to verify defacing quality and handle failures.

### 3D Rendering of Defaced Images

The notebook [[2]facemask_rendering_with_3DSlicer.ipynb](%5B2%5Dfacemask_rendering_with_3DSlicer.ipynb) creates GIF animations of the 3D rendering of each defaced image.


### Manual Assessment

The notebook [[3]loop_through_all_defaced_images_and_assess_manually.ipynb](%5B3%5Dloop_through_all_defaced_images_and_assess_manually.ipynb) loops through all defaced images for manual quality review.

For each image, the GIF rendering is displayed in MPV:

![IXI002-Guys-0828-T1_masked.gif](data%2Fexample_output%2Fdefaced_images_3d_visualization%2FIXI002-Guys-0828-T1%2FIXI002-Guys-0828-T1_masked.gif)

After closing MPV, you enter a rating for each image (e.g., `y` for success, `n` for failure). The special input `v` opens the original and defaced images side-by-side in ITK-SNAP. Results are saved to [example_output_manual_assessment.csv](data%2Fexample_output_manual_assessment.csv).

### Manual Landmark Registration (Fallback)

For cases where automatic registration failed, a manual affine transform can be created in 3D Slicer:

1. Open 3D Slicer and drag both the template and input images into the scene.
   ![data_module_after_data_import.png](docs%2Fscreenshots%2Fdata_module_after_data_import.png)

2. Right-click the input image and select **"Register this ..."**, then right-click the template and select **"Register \<input\> to this using ..." --> "Interactive Landmark Registration"**.
   ![interactive_landmark_registration_overview.png](docs%2Fscreenshots%2Finteractive_landmark_registration_overview.png)

3. Under **Landmarks**, click **"+Add"** and place at least 3 landmarks (e.g., left eye, right eye, 4th ventricle) in both the template and input views.
   ![landmark_1.png](docs%2Fscreenshots%2Flandmark_1.png)
   ![landmark_3.png](docs%2Fscreenshots%2Flandmark_3.png)

4. Under **Registration**, click **"Affine Registration"**, then under **Linear Registration** click **"Rigid"**. Verify the overlay is aligned.
   ![manual_registration_done.png](docs%2Fscreenshots%2Fmanual_registration_done.png)

5. Save the transform via **File --> Save Data**. Unselect all files except `Transform.h5`, change format to **Text (.txt)**, and save it alongside the input image as `Transform_to_template.txt`.
   ![saving_the_transform.png](docs%2Fscreenshots%2Fsaving_the_transform.png)

Re-run the defacing notebook (or `caideface`) with the manual transform in place. The pipeline will detect and use it automatically.

---

## Requirements and Setup

### For the `caideface` pip package

- Python >= 3.9
- BRAINSFit and BRAINSResample (bundled with [3D Slicer](https://www.slicer.org/))

Everything else is installed automatically via pip. See the [caideface README](caideface/README.md) for details.

### For the notebooks

The notebooks were tested on Ubuntu 20.04 with Python 3.11 and require:

| Tool | Purpose |
|------|---------|
| [BRAINSFit & BRAINSResample](https://github.com/BRAINSia/BRAINSTools) | Affine registration and resampling (tested with v5.4.0 / v5.7.0) |
| [3D Slicer](https://www.slicer.org/) (v5.2+) | 3D rendering (notebook [2]) and manual landmark registration (step 6) |
| [ITK-SNAP](http://www.itksnap.org) | Side-by-side comparison during manual assessment |
| [MPV](https://mpv.io/) | GIF viewing during manual assessment |
| [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/) | Reorientation (only for the skull-stripping notebook; not needed for `caideface`) |

### Extracting BRAINSFit and BRAINSResample from 3D Slicer

Building BRAINSTools from source can be challenging. The easiest way is to extract the binaries from a 3D Slicer installation:

**macOS:**
```bash
# Binaries are at:
/Applications/Slicer.app/Contents/lib/Slicer-<version>/cli-modules/BRAINSFit
/Applications/Slicer.app/Contents/lib/Slicer-<version>/cli-modules/BRAINSResample
```

**Linux (Ubuntu):**
1. Locate your 3D Slicer directory, e.g., `/home/user/3D_Slicer-5.6.2-linux-amd64`
2. Copy the binaries from `<Slicer_dir>/lib/Slicer-<version>/cli-modules/` to the [BRAINSTools](BRAINSTools) directory in this repository.
3. Register the shared library paths by creating `/etc/ld.so.conf.d/brainsfit.conf` with:
   ```
   <Slicer_dir>/lib/Slicer-<version>/cli-modules
   <Slicer_dir>/lib/Slicer-<version>
   <Slicer_dir>/lib
   ```
4. Run `sudo ldconfig` to update the shared library cache.

Replace `<Slicer_dir>` and `<version>` with your actual paths.

---

## How to Cite

If you use this pipeline or codebase in your work, please cite:

> Lorena Garcia-Foncillas Macias, et al. *A Generalisable Head MRI Defacing Pipeline: Evaluation on 2,566 Meningioma Scans*. 2025. Available at: [https://arxiv.org/abs/2505.12999](https://arxiv.org/abs/2505.12999)

If you use the skull-stripping step ([HD-BET](https://github.com/MIC-DKFZ/HD-BET)), please also cite:

> Isensee, F. et al. *Automated brain extraction of multi-sequence MRI using artificial neural networks*. Human Brain Mapping, 2019. DOI: [10.1002/hbm.24750](https://doi.org/10.1002/hbm.24750)

If you use CT defacing ([TotalSegmentator](https://github.com/wasserth/TotalSegmentator)), please also cite:

> Wasserthal, J. et al. *TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images*. Radiology: Artificial Intelligence, 5(5), 2023. DOI: [10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024)

If you use the text anonymisation (NER + HIPS) functionality, please also cite:

> Lorena Garcia-Foncillas Macias, Theodore Barfoot, Tom Vercauteren, Jonathan Shapey. *Evaluation of Named Entity Recognition for Automated Extraction of Present Tumor Size and Personal Names from Radiology Reports Using Spacy*. Journal of Neurological Surgery Part B: Skull Base, 86(S 01), 2025. DOI: [10.1055/s-0045-1803715](https://doi.org/10.1055/s-0045-1803715)

BibTeX:
```bibtex
@article{garcia2025defacing,
  title={A Generalisable Head MRI Defacing Pipeline: Evaluation on 2,566 Meningioma Scans},
  author={Garcia-Foncillas Macias, Lorena and Kujawa, Aaron and Elshalakany, Aya and Shapey, Jonathan and Vercauteren, Tom},
  year={2025},
  eprint={2505.12999},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@article{Wasserthal2023,
  author={Wasserthal, Jakob and Breit, Hanns-Christian and Meyer, Manfred T. and Pradella, Maurice and Hinck, Daniel and Sauter, Alexander W. and Heye, Tobias and Boll, Daniel T. and Cyriac, Joshy and Yang, Shan and Bach, Michael and Segeroth, Martin},
  title={TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images},
  journal={Radiology: Artificial Intelligence},
  volume={5},
  number={5},
  year={2023},
  doi={10.1148/ryai.230024}
}

@article{Isensee2019,
  author={Isensee, F. and Schell, M. and Tursunova, I. and Brugnara, G. and Bonekamp, D. and Neuberger, U. and Wick, A. and Schlemmer, H. P. and Heiland, S. and Wick, W. and Bendszus, M. and Maier-Hein, K. H. and Kickingereder, P.},
  title={Automated brain extraction of multi-sequence MRI using artificial neural networks},
  journal={Human Brain Mapping},
  year={2019},
  pages={1--13},
  doi={10.1002/hbm.24750}
}

@article{garcia2025ner,
  title={Evaluation of Named Entity Recognition for Automated Extraction of Present Tumor Size and Personal Names from Radiology Reports Using Spacy},
  author={Garcia-Foncillas Macias, Lorena and Barfoot, Theodore and Vercauteren, Tom and Shapey, Jonathan},
  journal={Journal of Neurological Surgery Part B: Skull Base},
  volume={86},
  number={S 01},
  year={2025},
  doi={10.1055/s-0045-1803715}
}
