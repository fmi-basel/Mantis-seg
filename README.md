<img width="4616" height="2094" alt="Asset 24@600x" src="https://github.com/user-attachments/assets/ed0fb55c-438e-4dcf-a974-93bdfe733c9c" />

<br> `Mantis-seg` is a Python package for 3D image segmentation and label-image processing, developed as part of the [Mantis ecosystem](https://github.com/fmi-basel/Mantis) for high-content, multiplexed, and volumetric microscopy image analysis.

`Mantis-seg` provides [Fractal](https://fractal-analytics-platform.github.io/) tasks for intensity-based segmentation, label-image generation, and post-processing of segmented objects. It operates on OME-Zarr datasets and supports region-of-interest (ROI)-based processing.

## Functionality

### 1. Intensity-based segmentation

The `segment_by_intensity_threshold` task segments microscopy images based on intensity thresholds, generating labeled regions from selected image channels. The resulting segmentations can be manually corrected to generate ground-truth annotations for training segmentation models, such as [RDCNet](https://github.com/fmi-basel/isit).

<img width="1324" height="391" alt="Screenshot 2026-11-19 at 15 38 44" src="https://github.com/user-attachments/assets/3b8d26cf-20f3-4e88-910c-badf9babd832" />

- **Input:** 3D microscopy image from an OME-Zarr dataset, with channel and intensity threshold parameters.
- **Output:** Segmentation label image and ROI table written to the OME-Zarr dataset.

### 2. Label expansion

The `expand_labels` task isotropically expands segmented objects in the XY plane or in all three dimensions (XYZ) into neighboring unlabeled regions, enabling the generation of expanded object masks.

<img width="1247" height="342" alt="Screenshot 2026-11-19 at 15 58 24" src="https://github.com/user-attachments/assets/79012a5b-d239-4338-a663-9dca874e0e34" />

- **Input:** Segmentation label image and expansion parameters.
- **Output:** Expanded label image and ROI table written to the OME-Zarr dataset.

### 3. Fusion of touching labels

The `fuse_touching_labels` task merges touching segmented objects to generate fused regions.

<img width="1242" height="345" alt="Screenshot 2026-11-19 at 16 03 10" src="https://github.com/user-attachments/assets/4c660bb1-9c26-44ec-8008-7aa4bf2d705d" />

- **Input:** Segmentation label image containing touching objects.
- **Output:** Label image with fused objects and ROI table written to the OME-Zarr dataset.

### 4. Cleanup of 3D child labels

The `cleanup_3d_child_labels` task post-processes 3D segmentation labels to remove small-volume debris among child objects within a parent object (e.g., an organoid or tissue) using a volume filter.

<img width="1240" height="335" alt="Screenshot 2026-11-19 at 16 31 04" src="https://github.com/user-attachments/assets/1a8d77ff-8ac6-42ee-9ae0-12805c76698f" />

- **Input:** 3D segmentation label image.
- **Output:** Cleaned 3D label image and ROI table written to the OME-Zarr dataset.

### 5. Projection of 3D segmentation labels to 2D 

The `convert_3d_to_mip` task projects 3D segmentation labels into 2D for downstream analysis and visualization in 2D.

<img width="1322" height="380" alt="Screenshot 2026-11-19 at 16 47" src="https://github.com/user-attachments/assets/dac6ed9d-b7c1-4ebe-9a83-bd8e4bc9e3e9" />

- **Input:** 3D label image (ZYX).
- **Output:** 2D projected label image (YX) and ROI table written to the corresponding 2D OME-Zarr dataset.

### 6. Build multiscale label image

The `build_label_image` task constructs the multiscale image pyramid and corresponding ROI table from an existing level-0 segmentation label image in an OME-Zarr dataset. This is useful when a segmentation task fails before completing pyramid generation and ROI table creation.

- **Input:** Existing level-0 segmentation label image.
- **Output:** Multiscale label-image pyramid and ROI table written to the OME-Zarr dataset.

### 7. In development: RDCNet segmentation

🚧 RDCNet segmentation is under development. The current implementation is available in the [ISIT repository](https://github.com/fmi-basel/isit). We are working on integrating and refactoring this functionality into `Mantis-seg`.

## Installation

Mantis-seg requires Python 3.11 or newer.

A Conda-based installation workflow is:

```bash
conda create -n mantis-seg python=3.11
conda activate mantis-seg

git clone https://github.com/fmi-basel/Mantis-seg.git
cd Mantis-seg
pip install -e .
```


## License

BSD 3-Clause License.
