WORK_DIRS=("experiment/DeepLabv3+_resnet50/AITEX2" "experiment/DeepLabv3+_resnet50/AITEX3"
           "experiment/DeepLabv3+_resnet50/BeanTech2" "experiment/DeepLabv3+_resnet50/BeanTech3" 
           "experiment/DeepLabv3+_resnet50/BSData2" "experiment/DeepLabv3+_resnet50/BSData3"
           "experiment/DeepLabv3+_resnet50/crackforest2" "experiment/DeepLabv3+_resnet50/crackforest3"
           "experiment/DeepLabv3+_resnet50/DAGM2" "experiment/DeepLabv3+_resnet50/DAGM3"
           "experiment/DeepLabv3+_resnet50/KolektorSDD_2" "experiment/DeepLabv3+_resnet50/KolektorSDD_3"
           "experiment/DeepLabv3+_resnet50/KolektorSDD2_2" "experiment/DeepLabv3+_resnet50/KolektorSDD2_3"
           "experiment/DeepLabv3+_resnet50/Magnetic_tile2" "experiment/DeepLabv3+_resnet50/Magnetic_tile3"
           "experiment/DeepLabv3+_resnet50/MVTecAD2" "experiment/DeepLabv3+_resnet50/MVTecAD3"
           "experiment/DeepLabv3+_resnet50/phonescreen2" "experiment/DeepLabv3+_resnet50/phonescreen3"
           "experiment/DeepLabv3+_resnet50/RSDDs2" "experiment/DeepLabv3+_resnet50/RSDDs3"
           "experiment/DeepLabv3+_resnet50/THC2" "experiment/DeepLabv3+_resnet50/THC3"
)

# 循环配置
for WORK_DIR in "${WORK_DIRS[@]}"; do
    # 寻找最新的权重文件
    LATEST_CHECKPOINT=$(ls -t ${WORK_DIR}/best_mIoU_iter_*.pth | head -1)

    # 寻找目标 .py 文件
    CONFIG_FILE=$(find ${WORK_DIR} -maxdepth 1 -type f -name "*.py" -print -quit)

    # 检查是否找到 .py 文件
    if [ -z "$CONFIG_FILE" ]; then
        echo "Error: No .py file found in ${WORK_DIR}"
        exit 1
    fi

    # 设置 GPU
    CUDA_VISIBLE_DEVICES=6  # 设置你想使用的 GPU

    # 运行测试命令
    python tools/test.py \
        ${CONFIG_FILE} \
        $LATEST_CHECKPOINT \
        --work-dir $WORK_DIR
done