_base_ = [
    '../../_base_/models/fcn_unet_s5-d16_IVD.py', '/home/s414e2/wjc/mm_ivd/configs/_base_/datasets/phonescreen.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_100k_IVD.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))
