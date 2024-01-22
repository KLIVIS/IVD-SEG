# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes=2
data_preprocessor = dict(
    type='SegDataPreProcessor',      # 数据预处理的类型
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,              # 是否将图像从 BGR 转为 RGB
    pad_val=0,                    # 图像的填充值
    seg_pad_val=0)              # 'gt_seg_map'的填充值
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='experiment/resnet_pretrain/best_mIoU_iter_5250.pth',  # 加载使用 ImageNet 预训练的主干网络
    backbone=dict(
        type='ResNetV1c',                    # 主干网络的类别，
        depth=50,                            # 主干网络的深度，通常为 50 和 101
        num_stages=4,                        # 主干网络状态(stages)的数目
        out_indices=(0, 1, 2, 3),            # 每个状态(stage)产生的特征图输出的索引
        dilations=(1, 1, 2, 4),              # 每一层(layer)的空心率(dilation rate)
        strides=(1, 2, 1, 1),                # 每一层(layer)的步长(stride)
        norm_cfg=norm_cfg,                   # 归一化层(norm layer)的配置项
        norm_eval=False,                     # 是否冻结 BN 里的统计项
        style='pytorch',                     # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积
        contract_dilation=True),             # 当空洞率 > 1, 是否压缩第一个空洞层
    decode_head=dict(
        type='FCNHead',                      # 解码头(decode head)的类别。
        in_channels=2048,                    # 解码头的输入通道数
        in_index=3,                           # 被选择特征图(feature map)的索引
        channels=512,                       # 解码头中间态(intermediate)的通道数
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,   # 在分类层(classification layer)之前是否连接(concat)输入和卷积的输出
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,     # 解码过程中调整大小(resize)的 align_corners 参数
        loss_decode=dict( 
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),                     # train_cfg 当前仅是一个占位符
    test_cfg=dict(mode='whole'))          # 测试模式，可选参数为 'whole' 和 'slide'. 'whole': 在整张图像上全卷积(fully-convolutional)测试。 'slide': 在输入图像上做滑窗预测

