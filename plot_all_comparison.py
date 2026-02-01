import matplotlib.pyplot as plt
import os
import sys
#####################################################################################################            这几幅图
# =========================================================
# 1. 实验结果路径配置
#    注意：这里我把文件名改成了 "seg_results..." 
#    请务必检查你的文件夹里是否有对应的 seg_results 文件！
# =========================================================
EXPERIMENT_FILES = {
    # --- SOTA 对比组 ---
    "LEGNet (Ours)":    r"/group/chenjinming/wyy/pytorch-pilipala-LEG/zxt_checkpoints/legnet_full_run2/seg_results_20251230-221812.txt",
    "ResNet-50":        r"/group/chenjinming/wyy/pytorch-pilipala-LEG/zxt_checkpoints/resnet50_run2/seg_results_20251230-211138.txt",
    "ResNet-18":        r"/group/chenjinming/wyy/pytorch-pilipala-LEG/zxt_checkpoints/resnet18_run3/seg_results_20251231-021545.txt",
    
    # --- 消融实验组 ---
    "w/o Scharr":       r"/group/chenjinming/wyy/pytorch-pilipala-LEG/zxt_checkpoints/legnet_no_scharr_run1/seg_results_20251230-185231.txt",
    "w/o LFEA":         r"/group/chenjinming/wyy/pytorch-pilipala-LEG/zxt_checkpoints/legnet_no_lfea_run2/seg_results_20251231-003400.txt",
    "w/o LoG":          r"/group/chenjinming/wyy/pytorch-pilipala-LEG/zxt_checkpoints/legnet_no_log_run3/seg_results_20251231-054254.txt",
    "w/o Gaussian":     r"/group/chenjinming/wyy/pytorch-pilipala-LEG/zxt_checkpoints/legnet_no_gaussian_run2/seg_results_20251230-235941.txt"
}

# =========================================================
# 2. 分组定义
# =========================================================
GROUPS = {
    "SOTA_Comparison": ["LEGNet (Ours)", "ResNet-50", "ResNet-18"],
    "Ablation_Study":  ["LEGNet (Ours)", "w/o Gaussian", "w/o Scharr", "w/o LFEA", "w/o LoG"]
}

# =========================================================
# 3. 样式定义
# =========================================================
STYLES = {
    "LEGNet (Ours)":    {"color": "#d62728", "linestyle": "-",  "linewidth": 2.5}, # 鲜红
    "ResNet-50":        {"color": "#1f77b4", "linestyle": "--", "linewidth": 2.0}, # 蓝
    "ResNet-18":        {"color": "#2ca02c", "linestyle": "--", "linewidth": 2.0}, # 绿
    "w/o Scharr":       {"color": "#ff7f0e", "linestyle": ":",  "linewidth": 1.5}, # 橙
    "w/o LFEA":         {"color": "#9467bd", "linestyle": ":",  "linewidth": 1.5}, # 紫
    "w/o LoG":          {"color": "#17becf", "linestyle": ":",  "linewidth": 1.5}, # 青
    "w/o Gaussian":     {"color": "#7f7f7f", "linestyle": ":",  "linewidth": 1.5}  # 灰
}

def read_results_file(file_path):
    """
    读取 seg_results.txt 或 det_results.txt
    返回: epochs, mAP(列1), loss(倒数列2)
    """
    if not file_path or not os.path.exists(file_path):
        return None, None, None
    
    epochs, maps, losses = [], [], []
    print(f"Reading: {os.path.basename(file_path)}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip() or "epoch" not in line:
                continue
            try:
                parts = line.strip().split()
                # 解析 Epoch
                # 格式可能是 "epoch:0" 或 "epoch: 0"
                epoch_str = parts[0]
                if ":" in epoch_str:
                    epoch = int(epoch_str.split(':')[1])
                else:
                    epoch = int(epoch_str)
                
                # 解析 mAP (第2个元素，索引1)
                # 无论是 det 还是 seg，最重要的 mAP 都在这个位置
                map_val = float(parts[1])
                
                # 解析 Loss (倒数第2个元素)
                # 格式: ... loss learning_rate
                loss_val = float(parts[-2]) 
                
                epochs.append(epoch)
                maps.append(map_val)
                losses.append(loss_val)
            except Exception as e:
                # 调试时可打开
                # print(f"Skipping line in {os.path.basename(file_path)}: {e}")
                continue
                
    return epochs, maps, losses

def plot_single_group(data_cache, group_name, curve_keys, metric_type, save_dir="."):
    """
    绘图主函数
    metric_type: 'Mask mAP' 或 'Loss'
    """
    plt.figure(figsize=(10, 7))
    has_data = False
    
    for label in curve_keys:
        if label not in data_cache or data_cache[label] is None:
            print(f"  [Warn] Missing data for {label}")
            continue
        
        epochs, maps, losses = data_cache[label]
        style = STYLES.get(label, {})
        
        # 选择数据
        if metric_type == 'Loss':
            y_data = losses
            y_label = "Training Loss"
            title_suffix = "Loss Convergence"
            loc = 'upper right' # Loss 图例放右上
        else:
            y_data = maps
            y_label = "Mask mAP (IoU=0.50:0.95)" # 既然读的是seg文件，这里就是Mask mAP
            title_suffix = "Segmentation Performance"
            loc = 'lower right' # mAP 图例放右下
            
        if y_data:
            has_data = True
            plt.plot(epochs, y_data, label=label, **style)

    if not has_data:
        plt.close()
        return

    plt.title(f"{group_name}: {title_suffix}", fontsize=16)
    plt.ylabel(y_label, fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc=loc, frameon=True, fancybox=True, framealpha=0.7)
    plt.tight_layout()
    
    filename = f"{group_name}_{metric_type.replace(' ', '_')}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"  -> Saved: {filename}")
    plt.close()

def main():
    save_dir = "." 
    
    # 1. 读取数据
    data_cache = {}
    print(">>> Start Loading Data...")
    for label, path in EXPERIMENT_FILES.items():
        epochs, maps, losses = read_results_file(path)
        if epochs:
            data_cache[label] = (epochs, maps, losses)
        else:
            print(f"  [Error] Failed to load or empty file: {path}")
            data_cache[label] = None
    print("-" * 40)

    # 2. 批量绘图
    print(">>> Start Plotting...")
    for group_name, keys in GROUPS.items():
        # 画 Mask mAP
        plot_single_group(data_cache, group_name, keys, 'Mask mAP', save_dir)
        # 画 Loss
        plot_single_group(data_cache, group_name, keys, 'Loss', save_dir)

    print("\nAll Done! Check the generated .png files.")

if __name__ == "__main__":
    main()