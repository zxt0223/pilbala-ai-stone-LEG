import os
import re
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
# 两个日志文件的路径
LOG_FILES = [
    # 第一阶段 (0-23 Epoch)
    "/group/chenjinming/wyy/pytorch-pilipala-LEG/output_best_20260201_182928/train_log.txt",
    # 第二阶段 (24-35 Epoch)
    "/group/chenjinming/wyy/pytorch-pilipala-LEG/output_best_20260201_182928_extended_36e/train_log_best.txt"
]

# 图片保存名称
OUTPUT_IMAGE = "combined_training_curve.png"
# ===========================================

def parse_log_files(file_paths):
    all_epochs = []
    all_losses = {} # Use dict to handle duplicates/overwrites
    all_maps = {}

    # 正则表达式
    # 匹配 Epoch 行: Epoch: [0]  [54/55] ... loss: 2.36 (2.32)
    # 我们取括号内的 smoothed loss
    re_epoch_loss = re.compile(r"Epoch:\s+\[(\d+)\]\s+.*loss:\s+[\d\.]+\s+\(([\d\.]+)\)")
    
    # 匹配 mAP 行: Average Precision ... IoU=0.50:0.95 ... = 0.679
    re_map = re.compile(r"Average Precision\s+\(AP\)\s+@\[\s+IoU=0\.50:0\.95\s+\|\s+area=\s+all\s+\|\s+maxDets=100\s+\]\s+=\s+(\d+\.\d+)")

    current_epoch = -1

    print(f"🚀 开始解析 {len(file_paths)} 个日志文件...")

    for fpath in file_paths:
        if not os.path.exists(fpath):
            print(f"⚠️  警告: 文件不存在 {fpath}")
            continue
        
        print(f"-> 正在读取: {os.path.basename(fpath)}")
        
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            # 1. 尝试匹配 Epoch 和 Loss
            match_loss = re_epoch_loss.search(line)
            if match_loss:
                ep = int(match_loss.group(1))
                loss_val = float(match_loss.group(2))
                
                # 记录该 Epoch 的 Loss (如果在同一个文件中出现多次，以后面的为准，代表该Epoch结束时的状态)
                current_epoch = ep
                all_losses[ep] = loss_val

            # 2. 尝试匹配 mAP (通常在 Epoch 结束后)
            if current_epoch != -1:
                match_map = re_map.search(line)
                if match_map:
                    map_val = float(match_map.group(1))
                    all_maps[current_epoch] = map_val
                    # 重置，防止匹配到错误的epoch
                    # current_epoch = -1 
                    # 注：不重置也可以，通常 mAP 紧跟在 Epoch 后面

    # 整理数据 (按 Epoch 排序)
    sorted_epochs = sorted(list(set(all_losses.keys()) | set(all_maps.keys())))
    
    final_epochs = []
    final_losses = []
    final_maps = []

    for ep in sorted_epochs:
        # 确保 Loss 和 mAP 都有才画 (或者根据需要处理缺失值)
        # 这里我们允许缺失，画图时会自动连线
        l = all_losses.get(ep, None)
        m = all_maps.get(ep, None)
        
        if l is not None and m is not None:
            final_epochs.append(ep)
            final_losses.append(l)
            final_maps.append(m)
    
    return final_epochs, final_losses, final_maps

def plot_curves(epochs, losses, maps):
    if not epochs:
        print("❌ 错误: 未提取到有效数据，请检查日志格式。")
        return

    print(f"📊 提取成功: 共 {len(epochs)} 个 Epoch 数据 (Epoch {epochs[0]} -> {epochs[-1]})")

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- 绘制 Loss (左轴) ---
    color_loss = '#e74c3c' # 红色
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Train Loss', color=color_loss, fontsize=12, fontweight='bold')
    
    line1 = ax1.plot(epochs, losses, color=color_loss, marker='o', markersize=4, 
                     linestyle='-', linewidth=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- 绘制 mAP (右轴) ---
    ax2 = ax1.twinx()  
    color_map = '#2980b9' # 蓝色
    ax2.set_ylabel('Val mAP (0.5:0.95)', color=color_map, fontsize=12, fontweight='bold')
    
    line2 = ax2.plot(epochs, maps, color=color_map, marker='s', markersize=4, 
                     linestyle='-', linewidth=2, label='Val mAP')
    ax2.tick_params(axis='y', labelcolor=color_map)

    # --- 合并图例 ---
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=11)

    # --- 标题和标注 ---
    plt.title('Training & Validation Curves (Combined)', fontsize=14, pad=20)
    
    # 标注最高点
    max_map = max(maps)
    max_idx = maps.index(max_map)
    max_epoch = epochs[max_idx]
    
    ax2.annotate(f'Best mAP: {max_map:.4f}', 
                 xy=(max_epoch, max_map), 
                 xytext=(max_epoch, max_map + 0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                 horizontalalignment='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\n✅ 曲线图已保存为: {os.path.abspath(OUTPUT_IMAGE)}")

if __name__ == "__main__":
    epochs, losses, maps = parse_log_files(LOG_FILES)
    plot_curves(epochs, losses, maps)