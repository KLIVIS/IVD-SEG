CONFIG_FILES=(
            #   "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_bsdatamnist-256x256.py"
            #   "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_crackforestmnist-256x256.py" "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_crackforestmnist-256x256.py"
            #   "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_dagm-256x256.py" "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_dagm-256x256.py"
            #   "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_kolektorsdd-256x256.py" "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_kolektorsdd-256x256.py"
            #   "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_kolektorsdd2-256x256.py" "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_kolektorsdd2-256x256.py"
              "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_magnetic_tile-256x256.py" "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_magnetic_tile-256x256.py"
              "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_mvtecadmnist-256x256.py" "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_mvtecadmnist-256x256.py"
              "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_phonescreen-256x256.py" "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_phonescreen-256x256.py"
              "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_rsdds-256x256.py" "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_rsdds-256x256.py"
              "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_thc-256x256.py" "configs/IVD_model/unet/unet-s5-d16_fcn_4xb4-100k_thc-256x256.py"
    

)

WORK_DIRS=(
        #    "experiment/unet/BSData3"
        #    "experiment/unet/crackforest2" "experiment/unet/crackforest3"
        #    "experiment/unet/DAGM2" "experiment/unet/DAGM3"
        #    "experiment/unet/KolektorSDD_2" "experiment/unet/KolektorSDD_3"
        #    "experiment/unet/KolektorSDD2_2" "experiment/unet/KolektorSDD2_3"
           "experiment/unet/Magnetic_tile2" "experiment/unet/Magnetic_tile3"
           "experiment/unet/MVTecAD2" "experiment/unet/MVTecAD3"
           "experiment/unet/phonescreen2" "experiment/unet/phonescreen3"
           "experiment/unet/RSDDs2" "experiment/unet/RSDDs3"
           "experiment/unet/THC2" "experiment/unet/THC3"
)

# 循环配置
for ((i=0; i<${#CONFIG_FILES[@]}; i++)); do
    CONFIG=${CONFIG_FILES[$i]}
    WORK_DIR=${WORK_DIRS[$i]}
    GPU="4,5"  # 设置要使用的 GPU，这里是卡2和卡3
    GPUS=${GPUS:-2}

    # 其他变量设置
    NNODES=1
    NODE_RANK=0
    PORT=29586
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