CONFIG_FILES=(
            #   "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_bsdatamnist-256x256.py"
            #   "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_crackforestmnist-256x256.py" "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_crackforestmnist-256x256.py"
            #   "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_dagm-256x256.py" "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_dagm-256x256.py"
            #   "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_kolektorsdd-256x256.py" "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_kolektorsdd-256x256.py"
            #   "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_kolektorsdd2-256x256.py" "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_kolektorsdd2-256x256.py"
            #   "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_magnetic_tile-256x256.py" "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_magnetic_tile-256x256.py"
            #   "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_mvtecadmnist-256x256.py" "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_mvtecadmnist-256x256.py"
            #   "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_phonescreen-256x256.py" "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_phonescreen-256x256.py"
              "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_rsdds-256x256.py" "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_rsdds-256x256.py"
              "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_thc-256x256.py" "configs/IVD_model/segformer/segformer_mit-b3_8xb2-100k_thc-256x256.py"
)

WORK_DIRS=(
        #    "experiment/SegFormer_b3/BSData3"
        #    "experiment/SegFormer_b3/crackforest2" "experiment/SegFormer_b3/crackforest3"
        #    "experiment/SegFormer_b3/DAGM2" "experiment/SegFormer_b3/DAGM3"
        #    "experiment/SegFormer_b3/KolektorSDD_2" "experiment/SegFormer_b3/KolektorSDD_3"
        #    "experiment/SegFormer_b3/KolektorSDD2_2" "experiment/SegFormer_b3/KolektorSDD2_3"
        #    "experiment/SegFormer_b3/Magnetic_tile2" "experiment/SegFormer_b3/Magnetic_tile3"
        #    "experiment/SegFormer_b3/MVTecAD2" "experiment/SegFormer_b3/MVTecAD3"
        #    "experiment/SegFormer_b3/phonescreen2" "experiment/SegFormer_b3/phonescreen3"
           "experiment/SegFormer_b3/RSDDs2" "experiment/SegFormer_b3/RSDDs3"
           "experiment/SegFormer_b3/THC2" "experiment/SegFormer_b3/THC3"
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
    PORT=29878
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