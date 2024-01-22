_base_ = [
    '../../_base_/models/pspnet_r50-d8.py',
    '../../_base_/datasets/phonescreen.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_100k_IVD.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))

