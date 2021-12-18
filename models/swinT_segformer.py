_base_ = [
    './_base_/upernet_swin.py', './_base_/pet.py',
    './_base_/base_runtime.py', './_base_/base_schdule.py'
]

MODEL_NAME='swinT_segformer_add_pascal_aug'

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='pretrain/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        _delete_=True,
        type='SegformerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(in_channels=384, num_classes=2))

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
         dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='cat-diary',
                name=MODEL_NAME,
                ))
    ])

