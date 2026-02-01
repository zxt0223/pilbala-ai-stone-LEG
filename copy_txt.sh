#!/bin/bash

# 源目录（zxt_checkpoints的路径）
SOURCE_DIR="/group/chenjinming/wyy/pytorch-pilipala-LEG/zxt_checkpoints"
# 目标目录（用来存放最终的txt文件，可自定义）
DEST_DIR="/group/chenjinming/wyy/pytorch-pilipala-LEG/zxt_checkpoints_txts"

# 创建目标目录（如果不存在）
mkdir -p "$DEST_DIR"

# 遍历zxt_checkpoints下的所有子文件夹
for folder in "$SOURCE_DIR"/*/; do
    # 获取文件夹的名称（比如 "zxt_lefnet_baseline_run1"）
    folder_name=$(basename "$folder")
    
    # 遍历当前文件夹下的所有txt文件
    for txt_file in "$folder"*.txt; do
        # 只处理存在的txt文件（避免无txt时的报错）
        [ -f "$txt_file" ] || continue
        
        # 获取txt文件的原名称（比如 "zxt_det_results.txt"）
        txt_name=$(basename "$txt_file")
        
        # 新文件名：文件夹名_原文件名（比如 "zxt_lefnet_baseline_run1_zxt_det_results.txt"）
        new_name="${folder_name}_${txt_name}"
        
        # 复制并重命名到目标目录
        cp "$txt_file" "$DEST_DIR/$new_name"
        echo "已复制：$txt_file → $DEST_DIR/$new_name"
    done
done

echo "所有txt文件已处理完成，保存到：$DEST_DIR"