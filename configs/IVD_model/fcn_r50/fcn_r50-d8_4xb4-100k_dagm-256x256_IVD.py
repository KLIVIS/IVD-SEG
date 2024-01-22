_base_ = [
    '/home/s414e2/wjc/mm_ivd/configs/_base_/models/fcn_r50-d8_IVD.py', '/home/s414e2/wjc/mm_ivd/configs/_base_/datasets/dagm.py',
    '/home/s414e2/wjc/mm_ivd/configs/_base_/default_runtime.py', '/home/s414e2/wjc/mm_ivd/configs/_base_/schedules/schedule_100k_IVD.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))
