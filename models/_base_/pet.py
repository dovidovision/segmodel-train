# dataset settings
dataset_type = 'CustomDataset'
oxford_data_root = './data/'
pascal_data_root = './data/VOCdevkit/VOC2012'

classes = ['Backgroud','Cat']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

multi_scale = [(x,x) for x in range(112, 336+1, 32)]
img_scale = (224, 224)
# crop_size = (480, 480)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='SegCutMix',p=0.3,data_root=oxford_data_root,
                        img_dir='train/images',
                        ann_dir='train/annotations',
                        class_weight=[0,1],
                        min_pixel=100,
                        num_sampling=1
                        ),
    dict(type='MyAlbu'),
    dict(type='Resize', img_scale=multi_scale, multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


oxford_pet_train = dict(
        type=dataset_type,
        data_root=oxford_data_root,
        img_dir='train/images',
        ann_dir='train/annotations',
        pipeline=train_pipeline,
        classes=classes
        )

oxford_pet_val = dict(
        type=dataset_type,
        data_root=oxford_data_root,
        img_dir='val/images',
        ann_dir='val/annotations',
        pipeline=test_pipeline,
        classes=classes
        )

pascal_train=dict(
        type=dataset_type,
        data_root=pascal_data_root,
        img_dir='JPEGImages',
        ann_dir='CatSegmentationClass',
        split=['ImageSets/Segmentation/train.txt','ImageSets/Segmentation/val.txt'],
        pipeline=train_pipeline,
        classes=classes
        )

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=[pascal_train,oxford_pet_train],
    val=oxford_pet_val,
    test=oxford_pet_val
)




