# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-50, -50, -5, 50, 50, 3]
# For H2O we have 9 object classes
class_names = [
    "dontcare",
    "book",
    "espresso",
    "lotion",
    "spray",
    "milk",
    "cocoa",
    "chips",
    "capuccino ",
]
dataset_type = "H2ODataset"
data_root = "data/h2o/"
# Input modality for H2O dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False
)
file_client_args = dict(backend="disk")
train_pipeline = [
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='DEPTH',
    #     load_dim=3,
    #     use_dim=3,
    #     file_client_args=file_client_args),
    dict(type="LoadImageFromFileMono3D", to_float32=True),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    # Scene Preprocessing
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # dict(type='RandomShiftScale', shift_scale=(0.2, 0.4), aug_prob=0.3),
    # dict(type='AffineResize', img_scale=(1280, 384), down_ratio=4),
    dict(type="Pad", size_divisor=32),
    # Output formatting
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=[
            "img",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_bboxes",
            "gt_labels",
            "centers2d",
            "depths",
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFileMono3D", to_float32=True),
    dict(
        type="MultiScaleFlipAug",
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type="RandomFlip3D"),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
            dict(type="Collect3D", keys=["img"]),
        ],
    ),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type="LoadImageFromFileMono3D", to_float32=True),
    # dict(type='Pad', size_divisor=32),
    # Output formatting
    dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
    dict(type="Collect3D", keys=["img"]),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "label_split/pose_train.txt",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="Depth",
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "label_split/pose_train.txt",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="Depth",
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "label_split/pose_train.txt",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="Depth",
    ),
)
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=1, pipeline=eval_pipeline)
