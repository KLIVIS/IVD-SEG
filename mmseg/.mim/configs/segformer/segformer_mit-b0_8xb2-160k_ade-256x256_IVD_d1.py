_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/rsdds.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_100k_IVD.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=2))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-7,
        power=0.9,
        begin=0,
        end=100000,
        by_epoch=False,
    )
]
train_dataloader = dict(batch_size=16, num_workers=8)
val_dataloader = dict(batch_size=16, num_workers=8)
test_dataloader = val_dataloader
