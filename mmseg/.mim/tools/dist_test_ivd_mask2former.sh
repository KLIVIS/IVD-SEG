WORK_DIRS=("experiment/Mask2former_swin_b/AITEX2" "experiment/Mask2former_swin_b/AITEX3"
           "experiment/Mask2former_swin_b/BeanTech2" "experiment/Mask2former_swin_b/BeanTech3" 
           "experiment/Mask2former_swin_b/BSData2" "experiment/Mask2former_swin_b/BSData3"
           "experiment/Mask2former_swin_b/crackforest2" "experiment/Mask2former_swin_b/crackforest3"
           "experiment/Mask2former_swin_b/DAGM2" "experiment/Mask2former_swin_b/DAGM3"
           "experiment/Mask2former_swin_b/KolektorSDD_2" "experiment/Mask2former_swin_b/KolektorSDD_3"
           "experiment/Mask2former_swin_b/KolektorSDD2_2" "experiment/Mask2former_swin_b/KolektorSDD2_3"
           "experiment/Mask2former_swin_b/Magnetic_tile2" "experiment/Mask2former_swin_b/Magnetic_tile3"
           "experiment/Mask2former_swin_b/MVTecAD2" "experiment/Mask2former_swin_b/MVTecAD3"
           "experiment/Mask2former_swin_b/phonescreen2" "experiment/Mask2former_swin_b/phonescreen3"
           "experiment/Mask2former_swin_b/RSDDs2" "experiment/Mask2former_swin_b/RSDDs3"
           "experiment/Mask2former_swin_b/THC2" "experiment/Mask2former_swin_b/THC3"
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