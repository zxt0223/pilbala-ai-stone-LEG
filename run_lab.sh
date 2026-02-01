#!/bin/bash
# -----------------------------------------------------------
# LEGNet Mask R-CNN 累加式消融实验全自动脚本
# 适配显卡: RTX 4090 (IDs: 4, 5, 6, 7)
# -----------------------------------------------------------

# 1. 基础配置
ROOT_DIR="/group/chenjinming/wyy/pytorch-pilipala-LEG"
DATA_PATH="${ROOT_DIR}/coco2017"
OUTPUT_ROOT="${ROOT_DIR}/output++lab"

# 指定GPU
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPU=4

# 训练参数 (4卡 4090, 24G显存)
# BatchSize=4 per GPU -> Total BS = 16
BATCH_SIZE=4
# 学习率: BS=16 对应 lr=0.02 (0.005 * 4)
LR=0.005
EPOCHS=24 # 跑24个epoch足够看清趋势

cd $ROOT_DIR

# 2. 定义实验顺序 (控制单一变量)
# key: 模式名称
# 顺序: Baseline -> +Scharr -> +LoG -> +Gaussian -> +LFEA (Full)
MODES=("baseline" "plus_scharr" "plus_scharr_log" "plus_scharr_log_gauss" "full")

# 3. 实验循环
echo "=========================================================="
echo "🚀 启动 LEGNet 累加式消融实验 (Cumulative Ablation)"
echo "   GPUs: $CUDA_VISIBLE_DEVICES | Total Batch: $(($BATCH_SIZE * $NUM_GPU))"
echo "   Output: $OUTPUT_ROOT"
echo "=========================================================="

# 创建总输出目录
mkdir -p "$OUTPUT_ROOT"

for MODE in "${MODES[@]}"
do
    # 定义该次实验的子目录
    EXP_DIR="${OUTPUT_ROOT}/${MODE}"
    LOG_FILE="${EXP_DIR}/train_log.txt"
    
    # 防止重复运行
    if [ -f "$LOG_FILE" ]; then
        echo "⚠️  跳过: $MODE (日志已存在)"
        continue
    fi
    
    mkdir -p "$EXP_DIR"
    
    echo " "
    echo "▶️  [Running] Mode: $MODE"
    echo "    Log -> $LOG_FILE"
    
    # 随机端口避免冲突
    PORT=$((29000 + $RANDOM % 1000))
    
    # 启动训练
    # 注意: --ablation-mode 参数对应我们在 legnet.py 里新写的逻辑
    torchrun --nproc_per_node=$NUM_GPU --master_port=$PORT train_multi_GPU.py \
        --data-path "$DATA_PATH" \
        --backbone "legnet" \
        --ablation-mode "$MODE" \
        --num-classes 1 \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --epochs $EPOCHS \
        --lr-steps 16 22 \
        --output-dir "$EXP_DIR" \
        --amp True \
        2>&1 | tee "$LOG_FILE"
        
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ [Success] Mode: $MODE Finished."
    else
        echo "❌ [Failed] Mode: $MODE Failed."
        exit 1
    fi
    
    # 稍微休息一下，清理显存
    sleep 10
done

echo " "
echo "🎉 所有消融实验已完成！"