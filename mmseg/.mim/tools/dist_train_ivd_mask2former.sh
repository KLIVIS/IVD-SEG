CONFIG_FILES=(
            #   "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-beantech-256x256.py" "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-beantech-256x256.py"
            #   "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-bsdatamnist-256x256.py" "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-bsdatamnist-256x256.py"
            #   "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-crackforestmnist-256x256.py" "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-crackforestmnist-256x256.py"
            #   "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-dagm-256x256.py" "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-dagm-256x256.py"
            #   "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-kolektorsdd-256x256.py" 
              "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-kolektorsdd-256x256.py"
              "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-kolektorsdd2-256x256.py" "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-kolektorsdd2-256x256.py"
              "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-magnetic_tile-256x256.py" "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-magnetic_tile-256x256.py"
              "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-mvtecadmnist-256x256.py" "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-mvtecadmnist-256x256.py"
              "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-phonescreen-256x256.py" "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-phonescreen-256x256.py"
              "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-rsdds-256x256.py" "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-rsdds-256x256.py"
              "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-thc-256x256.py" "configs/IVD_model/mask2former/mask2former_swin-b-in22k-pre_100k-thc-256x256.py"
    

)

WORK_DIRS=(
        #    "experiment/Mask2former_swin_b/BeanTech2" "experiment/Mask2former_swin_b/BeanTech3"
        #    "experiment/Mask2former_swin_b/BSData2" "experiment/Mask2former_swin_b/BSData3"
        #    "experiment/Mask2former_swin_b/crackforest2" "experiment/Mask2former_swin_b/crackforest3"
        #    "experiment/Mask2former_swin_b/DAGM2" "experiment/Mask2former_swin_b/DAGM3"
        #    "experiment/Mask2former_swin_b/KolektorSDD_2" 
           "experiment/Mask2former_swin_b/KolektorSDD_3"
           "experiment/Mask2former_swin_b/KolektorSDD2_2" "experiment/Mask2former_swin_b/KolektorSDD2_3"
           "experiment/Mask2former_swin_b/Magnetic_tile2" "experiment/Mask2former_swin_b/Magnetic_tile3"
           "experiment/Mask2former_swin_b/MVTecAD2" "experiment/Mask2former_swin_b/MVTecAD3"
           "experiment/Mask2former_swin_b/phonescreen2" "experiment/Mask2former_swin_b/phonescreen3"
           "experiment/Mask2former_swin_b/RSDDs2" "experiment/Mask2former_swin_b/RSDDs3"
           "experiment/Mask2former_swin_b/THC2" "experiment/Mask2former_swin_b/THC3"
)

# 循环配置
for ((i=0; i<${#CONFIG_FILES[@]}; i++)); do
    CONFIG=${CONFIG_FILES[$i]}
    WORK_DIR=${WORK_DIRS[$i]}
    GPU="0,1"  # 设置要使用的 GPU，这里是卡2和卡3
    GPUS=${GPUS:-2}

    # 其他变量设置
    NNODES=1
    NODE_RANK=0
    PORT=29875
    MASTER_ADDR="127.0.0.1"

    # 运行训练命令
    PYTHONPATH="$(dirname $0)/.." \  # 将脚本所在目录的上一级目录添加到 Python 模块搜索路径，以确保 Python 能够找到脚本所依赖的模块。
    CUDA_VISIBLE_DEVICES=$GPU \
    python -m torch.distributed.launch \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        $(dirname "$0")/train.py \
        $CONFIG \
        --launcher pytorch \
        --work-dir $WORK_DIR \
        ${@:3}
    #测试运行命令
    LATEST_CHECKPOINT=$(ls -t ${WORK_DIR}/best_mIoU_iter_*.pth | head -1)
    python -m torch.distributed.launch \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        $(dirname "$0")/test.py \
        $CONFIG \
        $LATEST_CHECKPOINT \
        --launcher pytorch \
        --work-dir $WORK_DIR \
        ${@:3}
done