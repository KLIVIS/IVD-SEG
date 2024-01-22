
## 仿照数据集 synapse.py
dataset_type = 'MVTecADMNISTDataset' # 数据集类型，自定义
data_root = 'IVD_data/IVD_noPavement' # 数据根路径

img_szie = (256,256)  # 训练时裁剪的大小

train_pipeline = [
    dict(type='LoadImageFromFile'),  # 第一个流程，从文件路径加载图像
    dict(type='LoadAnnotations'),   # 第2个流程，对于当前图像，加载它的标注图像
    dict(
        type='RandomChoiceResize',
        scales=[int(x * 0.1 * 256) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=768),
    dict(type='RandomCrop', crop_size=img_szie, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),      # 翻转图像和其标注图像的数据增广流程， 翻转图像的概率为0.5
    dict(type='PhotoMetricDistortion'),    # 光学上使用一些方法扭曲当前图像和其标注图像的数据增广流程
    dict(type='PackSegInputs')               # 打包用于语义分割的输入数据
]

test_pipeline = [
    dict(type='LoadImageFromFile'),                           # 第1个流程，从文件路径里加载图像
    dict(type='Resize', scale=img_szie, keep_ratio=True),
    dict(type='LoadAnnotations'),    # 加载数据集提供的语义分割标注
    dict(type='PackSegInputs')                               # 打包用于语义分割的输入数据
]

train_dataloader = dict(
    batch_size=64,                # 每一个GPU的batch size大小
    num_workers=8,               # 为每一个GPU预读取数据的进程个数
    persistent_workers=True,     # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='InfiniteSampler', shuffle=True),   # 训练时进行随机洗牌(shuffle)
    dataset=dict(  # 训练数据集配置
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='imgs/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),   # 验证时不进行随机洗牌(shuffle)
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='imgs/val',
            seg_map_path='annotations/val'),
        pipeline=test_pipeline))
test_dataloader =  dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='imgs/test',
            seg_map_path='annotations/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
