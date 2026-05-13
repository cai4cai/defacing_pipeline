# Defacing pipeline
[![arXiv](https://img.shields.io/badge/arXiv-2505.12999-b31b1b.svg)](https://arxiv.org/abs/2505.12999)

This defacing pipeline is based on affine registration with BRAINSFit and resampling with BRAINSResample.
It was developed primarily for input images of brains with a reduced FOV (slabs) where the affine registration often fails.
To access the quality of the initial fully automatic defacing step, we provide a second script that creates 3D visualizations
of the defaced images in GIF format.
As a fallback solution for cases where the defacing failed, an input transform can be produced by selecting landmark pairs
in the template and the input image. This is done in 3D Slicer. 

Jump to the section [Alternative Registration and Defacing with HD-BET](#alternative-registration-and-defacing-with-hd-bet) to use the pipeline from [arXiv](https://arxiv.org/abs/2505.12999).


## Steps
1. Affine registration of the input image to a template image (that has a template face mask associated with it)
2. Transforming the template face mask to the input image space by inverting the affine transform
3. Applying the resampled template face mask to the input image to remove facial features
4. Creation of 3D rendering of the defaced image with 3D Slicer
5. Manually confirm the success of the defacing or mark as failed for each input image
6. If the defacing failed, manual landmark selection in 3D Slicer to produce a manual affine transformation
7. Repeat steps 2-4 based on the manual affine transformation

## Requirements
The code was tested on Ubuntu 20.04 with Python 3.11
1. BRAINSFit and BRAINSResample binaries from [BRAINSTools](https://github.com/BRAINSia/BRAINSTools) (Note: The binaries
are dependent on shared libraries that need to be installed as well) (The scripts were tested with BRAINSFit version: 5.4.0 and 
BRAINSResample version: 5.7.0). See [below](#extracting-brainsfit-and-brainsresample-binaries-and-shared-libraries-from-3d-slicer) for instructions on how to extract the binaries and shared libraries from
a 3D Slicer installation.
2. [3D Slicer](https://www.slicer.org/) (tested with version 5.2) with the Jupyter Notebook extension installed to create 3D renderings of the defaced images
and to perform landmark registration for failed cases
3. [ITK-SNAP](http://www.itksnap.org) and [MPV](https://mpv.io/) for viewing of the 3D renderings

## Example run
The example run is based on two test images to be defaced, 
a T1 image [IXI002-Guys-0828-T1.nii.gz](data/example_input_images/IXI002-Guys-0828-T1/IXI002-Guys-0828-T1.nii.gz)
and a T2 image [IXI002-Guys-0828-T2.nii.gz](data/example_input_images/IXI002-Guys-0828-T2/IXI002-Guys-0828-T2.nii.gz).
The T2 image also comes with a 
manual affine transform [Transform_to_template.txt](data/example_input_images/IXI002-Guys-0828-T2/Transform_to_template.txt) 
to explain how the pipeline deals with previously failed defacing attempts. How to create the manual transform is explained
further down.

The [template image](data/icbm152_ext55_model_sym_2020_nifti/icbm152_ext55_model_sym_2020/mni_icbm152_t1_tal_nlin_sym_55_ext.nii) used in this project is sourced from the [ICBM 152 Extended Nonlinear Atlases (2020)](https://nist.mni.mcgill.ca/icbm-152-extended-nonlinear-atlases-2020/).

The [template mask](data/icbm152_ext55_model_sym_2020_nifti/icbm152_ext55_model_sym_2020/t1_mask.nii.gz) was generated using [ITK-SNAP](http://www.itksnap.org) to remove facial features while retaining as much clinical information as possible.

All outputs are written to the [example_output](data/example_output) directory. If you want to rerun the example, 
delete the contents of the directory first, because the notebook will not overwrite existing files.

## Steps 1-3: Automatic defacing steps
These steps are performed by the notebook [[1]deface_with_brainsfit.ipynb](%5B1%5Ddeface_with_brainsfit.ipynb)
In the second cell the paths to the input images, and output directories need to be set. One also needs to set a list
of existing manual transforms (if none exist, set it to a list of None of the same length as the input images list). In
this example, we have no manual transform for the T1 image but we have one for the T2 image, so the list looks like this:
```python
existing_transform_paths = [None, "./data/example_input_images/IXI002-Guys-0828-T2/Transform_to_template.txt"]
```
Furthermore, the path to the BRAINSFit and BRAINSResample binaries need to be set in this cell. 

For each case the notebook will create an output directory, for example for the T1 image:
[IXI002-Guys-0828-T1](data/example_output/defaced_images/IXI002-Guys-0828-T1)

This directory contains intermediate output files and the defaced image:
[IXI002-Guys-0828-T1_masked.nii.gz](data/example_output/defaced_images/IXI002-Guys-0828-T1/IXI002-Guys-0828-T1_masked.nii.gz)


## Step 4: Create 3D rendering of the defaced images with 3D Slicer
This step is performed by the notebook [[2]facemask_rendering_with_3DSlicer.ipynb](%5B2%5Dfacemask_rendering_with_3DSlicer.ipynb).
In the third cell, it is required to set the path to the defaced images and the output directory.
The notebook will create a GIF animation of the 3D rendering of the defaced image. The GIF is saved in the output directory.

This script needs to be run with the 3D Slicer Jupyter Notebook kernel. To install the kernel, follow the instructions
[here](https://github.com/Slicer/SlicerJupyter). Alternatively, the notebook can be converted to a Python script and run
directly in 3D Slicer, using the approach described 
[here](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#run-a-python-script-file-in-the-slicer-environment).

## Step 5: Manual confirmation of the defacing
This step is performed by the notebook [[3]loop_through_all_defaced_images_and_assess_manually.ipynb](%5B3%5Dloop_through_all_defaced_images_and_assess_manually.ipynb).
The purpose is to efficiently go through all GIFs created in step 4 and manually assess the quality of the defacing for each 
image, and store the results in a CSV file, so that manual registration can be performed for the failed cases. 
In the second cell, the path to the defaced images, the 3D GIFs and the original images need to be set. The latter are 
used to compare the masked images with the original images. 

The code will loop through all defaced images and show the GIFs in MPV like this (this requires MPV to be installed and added to the
PATH): ![IXI002-Guys-0828-T1_masked.gif](data%2Fexample_output%2Fdefaced_images_3d_visualization%2FIXI002-Guys-0828-T1%2FIXI002-Guys-0828-T1_masked.gif) 
After assessing the quality of the defacing and closing MPV, the user is asked for an input string, 
which will be stored for each image in the CSV file. For example, this can be "y" for successful defacing and "n" for failed defacing.
The only special input is "v", which will open the original image and the defaced image in ITK-SNAP for a side-by-side comparison.

The output CSV file is [example_output_manual_assessment.csv](data%2Fexample_output_manual_assessment.csv)

## Step 6: Manual landmark selection in 3D Slicer
This step is performed manually in 3D Slicer. 
1. Open 3D Slicer.
1. Drag the template image onto the 3D view and confirm the "Add data into the scene" dialog. 
1. Drag the input image onto the 3D view and confirm the "Add data into the scene" dialog.
1. Under Modules, select the "Data" module: ![data_module_after_data_import.png](docs%2Fscreenshots%2Fdata_module_after_data_import.png)
1. Right-click on the input image and select "Register this ..."
1. Right-click on the template image and select "Register <input_image_name> to this using ..." --> "Interactive Landmark Registration". 
This will automatically open the Landmark Registration module. You should see a 3x3 matrix of images: ![interactive_landmark_registration_overview.png](docs%2Fscreenshots%2Finteractive_landmark_registration_overview.png)
The first row shows the template image views, the second row shows the input image views and the third row shows the
overlay of both images. 
1. In the Landmark Registration module, scroll to "Landmarks" and click "+Add". Then, place the landmark in the
template image, for example on the left eye by scrolling to the correct axial slice with the mouse wheel and placing the 
landmark with a left-click. The landmark will appear in the other views of the template image and in all views of the 
input image. First, make sure the landmark is placed correctly in the template image in all views, by dragging the landmark
to the correct position. Then, make sure the landmark is placed correctly in the input image in all views: ![landmark_1.png](docs%2Fscreenshots%2Flandmark_1.png)
1. Repeat the same for at least two more landmarks, for example on the right eye and a corner of the 4th ventricle: ![landmark_3.png](docs%2Fscreenshots%2Flandmark_3.png)
1. To create the landmark based transform, scroll down in the "Landmark Registration" module to "Registration" and click on "Affine Registration"
Then, scroll further down and under "Linear Registration" click on "Rigid" (even if already selected). If the landmarks are placed correctly, 
the overlaid images in row 3 should be registered. If not, try to place the landmarks again or add more landmarks: ![manual_registration_done.png](docs%2Fscreenshots%2Fmanual_registration_done.png)
1. Finally, to save the transform by clicking "File" --> "Save Data" and unselect all files except "Transform.h5". Change the File Format to "Text (.txt)" and select the correct
output directory. In this example, the first defacing script expects the transform to be called "Transform_to_template.txt" and to be 
located in the same folder as the input image (alternatively, paths can be adjusted in the script): ![saving_the_transform.png](docs%2Fscreenshots%2Fsaving_the_transform.png)

## Step 7: Repeat steps 2-4 with the manual affine transformation
Running the first notebook again with the manual transform should produce a successful defacing.
Make sure to delete previous outputs first, because the notebook will not overwrite existing files.


## Extracting BRAINSFit and BRAINSResample binaries and shared libraries from 3D Slicer
Building BRAINSTools from source can be challenging. An easier way to obtain the binaries and shared libraries is to 
extract them from a 3D Slicer installation. The instructions for Ubuntu are as follows:
1. Locate the 3D Slicer directory, for example `/home/user/3D_Slicer-5.6.2-linux-amd64`
2. Copy the binaries, `<Slicer_dir>/lib/Slicer-<version>/cli-modules/BRAINSFit` and 
`<Slicer_dir>/lib/Slicer-<version>/cli-modules/BRAINSResample` to the [BRAINSTools](BRAINSTools) directory contained in 
the defacing_pipeline repository, overwriting the existing binaries.
3. Now the Slicer paths that contain the shared libraries need to be registered in the system. This can be done by
creating a new `.conf` file (for example `brainsfit.conf`) in `/etc/ld.so.conf.d/` with the following paths as 
content:
```                                                          
<Slicer_dir>/lib/Slicer-<version>/cli-modules
<Slicer_dir>/lib/Slicer-<version>
<Slicer_dir>/lib
```
Make sure to replace `<Slicer_dir>` and `<version>` with the correct values.
4. Run `sudo ldconfig` to update the shared library paths.
5. Now the BRAINSFit and BRAINSResample binaries should be able to find the shared libraries and run successfully.


## Alternative Registration and Defacing with HD-BET

An alternative approach to the pipeline is outlined in the Jupyter notebook [[1]deface_skullstripping.ipynb]([1]deface_skullstripping.ipynb). This method involves several additional steps for improved defacing:

1. **Reorientation**: Scans are first reoriented to MNI152 space using the [FSL Installation Guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/).
2. **Skull-stripping**: Skull-stripping is then performed on these reoriented scans using [HD-BET GitHub repository](https://github.com/MIC-DKFZ/HD-BET).
3. **Mask Dilation**: The resulting binary brain mask from the skull-stripping process is dilated.
4. **New Skull-stripped Scan**: A new skull-stripped scan is generated using the dilated mask.

The registration and defacing steps follow as described earlier, but with the key difference that the input scans are the reoriented ones, and the dilated brain mask is used. This ensures that no brain structures are removed during the defacing process.

## How to Cite

If you use this pipeline or codebase in your work, please cite the following paper:

> Lorena Garcia-Foncillas Macias, et al. *A Generalisable Head MRI Defacing Pipeline: Evaluation on 2,566 Meningioma Scans*. 2025. Available at: [https://arxiv.org/abs/2505.12999](https://arxiv.org/abs/2505.12999)

If you use the skull-stripping step ([HD-BET](https://github.com/MIC-DKFZ/HD-BET)), please also cite:

> Isensee, F. et al. *Automated brain extraction of multi-sequence MRI using artificial neural networks*. Human Brain Mapping, 2019. DOI: [10.1002/hbm.24750](https://doi.org/10.1002/hbm.24750)

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
