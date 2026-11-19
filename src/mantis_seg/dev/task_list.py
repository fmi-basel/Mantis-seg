from fractal_task_tools.task_models import CompoundTask, ParallelTask

AUTHORS = "Nicole Repina, Enrico Tagliavini, Tim-Oliver Buchholz, Joel Luethi"
DOCS_LINK = "https://github.com/fmi-basel/Mantis-seg"
INPUT_MODELS = [["fractal_tasks_core", "channels.py", "ChannelInputModel"]]
TASK_LIST = [
CompoundTask(
        name="Segment by Intensity Threshold",
        executable_init="fractal/init_select_many_rounds.py",
        executable="fractal/segment_by_intensity_threshold.py",
        meta_init={"cpus_per_task": 1, "mem": 1000},
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Segmentation",
        modality="HCS",
        tags=["Classical segmentation", "3D"],
        docs_info="file:task_info/segment_by_intensity_threshold.md",
    ),
ParallelTask(
        name="Expand Labels",
        executable="fractal/expand_labels.py",
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Image Processing",
        tags=["2D", "3D"],
        docs_info="file:task_info/expand_labels.md",
    ),
ParallelTask(
        name="Fuse Touching Labels",
        executable="fractal/fuse_touching_labels.py",
        meta={"cpus_per_task": 4, "mem": 16000},
        category="Image Processing",
        tags=["2D"],
        docs_info="file:task_info/fuse_touching_labels.md",
    ),
ParallelTask(
        input_types=dict(is_3D=True),
        output_types=dict(is_3D=False),
        name="Convert 3D Segmentation to MIP",
        executable="fractal/convert_3d_to_mip.py",
        meta={"cpus_per_task": 2, "mem": 8000},
        tags=[
            "Mixed modality",
            "3D to 2D workflows",
        ],
        docs_info="file:task_info/convert_3d_to_mip.md",
    ),
ParallelTask(
        name="Build Label Image",
        executable="fractal/build_label_image.py",
        meta={"cpus_per_task": 4, "mem": 36000},
        category="Image Processing",
        tags=["Mixed modality"],
        docs_info="file:task_info/build_label_image.md",
    ),
ParallelTask(
        name="Cleanup 3D Child Labels",
        executable="fractal/cleanup_3d_child_labels.py",
        meta={"cpus_per_task": 4, "mem": 12000},
        category="Image Processing",
        tags=["3D"],
        docs_info="file:task_info/cleanup_3d_child_labels.md",
    )
]
