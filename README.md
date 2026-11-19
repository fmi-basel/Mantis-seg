# Mantis-seg

Independent Fractal task package extracted from [Mantis-plex](https://github.com/fmi-basel/Mantis-plex).

## Tasks

- `build_label_image`
- `cleanup_3d_child_labels`
- `convert_3d_to_mip`
- `expand_labels`
- `fuse_touching_labels`
- `segment_by_intensity_threshold`

## Installation and task manifest

```bash
pip install -e ".[test]"
python -m mantis_seg.dev.task_info.create_manifest
pytest tests
```

The original Mantis-plex repository has not been modified. Internal helper modules are duplicated to avoid runtime dependencies on other Mantis packages.
