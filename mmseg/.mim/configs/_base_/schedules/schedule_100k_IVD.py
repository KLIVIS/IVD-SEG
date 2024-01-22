# optimizer
optimizer = dict(type='Adam', lr=0.0001,betas=(0.9,0.999),eps=1e-08,  weight_decay=0.0001)  # 将SGD更改为Adam
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',   # 调度流程的策略
        eta_min=1e-7,     # 训练结束时的最小学习率
        power=0.9,       # 多项式衰减 (polynomial decay) 的幂
        begin=0,         # 开始更新参数的时间步(step)
        end=50000,       # 停止更新参数的时间步(step)
        by_epoch=False)  # 是否按照 epoch 计算训练时间
]
# 40k iteration 的训练计划
train_cfg = dict(type='IterBasedTrainLoop', max_iters=50000, val_interval=50)  # 
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# 默认钩子(hook)配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),    # 记录迭代过程中花费的时间
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),  # 从'Runner'的不同组件收集和写入日志
    param_scheduler=dict(type='ParamSchedulerHook'),                         # 更新优化器中的一些超参数，例如学习率
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=50,
                    save_best=['mIoU'], rule='greater',max_keep_ckpts=2),   # 定期保存检查点(checkpoint), 
    sampler_seed=dict(type='DistSamplerSeedHook'),                            # 用于分布式训练的数据加载采样器
    visualization=dict(type='SegVisualizationHook', draw=True, interval = 5),  # 可视化 ground truth 和在模型测试和验证期间的预测分割结果,# interval 表示预测结果的采样间隔， 设置为 1 时，将保存网络的每个推理结果,默认为50
    earlystopping=dict( type='EarlyStoppingHook',  # 使用官方的 EarlyStoppingHook
            monitor='mIoU',  # 选择你要监测的指标，这里假设是 mIoU
            patience=200,  # 设置耐心值
            rule='greater',  # 选择是越大越好还是越小越好
))