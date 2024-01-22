WORK_DIRS=("experiment/unet/AITEX2" "experiment/unet/AITEX3"
           "experiment/unet/BeanTech2" "experiment/unet/BeanTech3" 
           "experiment/unet/BSData2" "experiment/unet/BSData3"
           "experiment/unet/crackforest2" "experiment/unet/crackforest3"
           "experiment/unet/DAGM2" "experiment/unet/DAGM3"
           "experiment/unet/KolektorSDD_2" "experiment/unet/KolektorSDD_3"
           "experiment/unet/KolektorSDD2_2" "experiment/unet/KolektorSDD2_3"
           "experiment/unet/Magnetic_tile2" "experiment/unet/Magnetic_tile3"
           "experiment/unet/MVTecAD2" "experiment/unet/MVTecAD3"
           "experiment/unet/phonescreen2" "experiment/unet/phonescreen3"
           "experiment/unet/RSDDs2" "experiment/unet/RSDDs3"
           "experiment/unet/THC2" "experiment/unet/THC3"
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