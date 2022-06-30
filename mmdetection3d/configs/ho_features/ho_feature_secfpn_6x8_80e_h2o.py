_base_ = [
    "../_base_/datasets/h2o.py",
    "../_base_/models/ho_feature.py",
    # '../_base_/models/smoke.py',
    "../_base_/schedules/mmdet_schedule_1x.py",
    "../_base_/default_runtime.py",
]

# model settings
model = dict(
    backbone=dict(
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    )
)
# optimizer
optimizer = dict(lr=0.002, paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=1.0 / 3, step=[8, 11]
)
EPOCHS = 6
runner = dict(type="EpochBasedRunner", max_epochs=EPOCHS)
total_epochs = EPOCHS
evaluation = dict(interval=1)
