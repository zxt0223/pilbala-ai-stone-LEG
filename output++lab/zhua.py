import os
import glob
import re
import matplotlib.pyplot as plt

# =================配置区域=================
EXP_CONFIG = {
    "Baseline": "/group/chenjinming/wyy/pytorch-pilipala-LEG/output++lab/baseline",
    "Full": "/group/chenjinming/wyy/pytorch-pilipala-LEG/output++lab/full",
    "Plus Scharr": "/group/chenjinming/wyy/pytorch-pilipala-LEG/output++lab/plus_scharr",
    "Plus Scharr Log": "/group/chenjinming/wyy/pytorch-pilipala-LEG/output++lab/plus_scharr_log",
    "Plus Scharr Gauss": "/group/chenjinming/wyy/pytorch-pilipala-LEG/output++lab/plus_scharr_log_gauss",
}

# =================数据解析函数=================

def find_file(directory, pattern):
    """在指定目录下寻找匹配模式的文件，返回最新的一个"""
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def parse_loss_log(filepath):
    """解析 Loss: 提取括号内的平均值"""
    if not filepath or not os.path.exists(filepath):
        return [], []
    epoch_data = {}
    with open(filepath, 'r') as f:
        for line in f:
            # 匹配 Epoch: [ 23] ... loss: ... (0.5015)
            match = re.search(r'Epoch:\s*\[\s*(\d+)\s*\].*?loss:.*?\((\d+\.\d+)\)', line)
            if match:
                ep = int(match.group(1))
                val = float(match.group(2))
                epoch_data[ep] = val 
    epochs = sorted(epoch_data.keys())
    losses = [epoch_data[e] for e in epochs]
    return epochs, losses

def parse_results_file(filepath):
    """解析 mAP: epoch:22 0.7241 ..."""
    if not filepath or not os.path.exists(filepath):
        return [], []
    epochs = []
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            # 匹配 epoch:22 后面的第一个浮点数
            match = re.search(r'epoch:(\d+)\s+([\d\.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                values.append(float(match.group(2)))
    return epochs, values

# =================通用绘图函数=================

def draw_single_metric(data_dict, metric_key, title, ylabel, filename):
    """
    通用绘图函数
    data_dict: 总数据字典
    metric_key: 'loss' 或 'det' 或 'seg'
    """
    plt.figure(figsize=(10, 6))
    has_data = False
    
    # 遍历每个实验配置
    for label, metrics in data_dict.items():
        x, y = metrics[metric_key]
        if x and y:
            # 绘图样式：点线图，稍微加一点透明度
            plt.plot(x, y, marker='o', markersize=4, label=label, linewidth=2, alpha=0.9)
            has_data = True
    
    if has_data:
        plt.title(title, fontsize=15, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 自动调整布局并保存
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f">>> {filename} 已成功保存！")
        plt.close() # 关闭画布，防止内存堆积
    else:
        print(f"!!! 警告: 没有为 {title} 找到任何数据，跳过绘图。")

# =================主逻辑=================

def plot_curves():
    # 设置风格，看起来更学术
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass # 如果样式不存在则使用默认

    plot_data = {}

    print("--- 开始解析数据 ---")
    for label, dir_path in EXP_CONFIG.items():
        print(f"正在读取: {label}")
        
        # 1. 解析 Loss
        log_file = find_file(dir_path, "train_log.txt")
        loss_x, loss_y = parse_loss_log(log_file)
        
        # 2. 解析 Det mAP
        det_file = find_file(dir_path, "det_results*.txt")
        det_x, det_y = parse_results_file(det_file)
        
        # 3. 解析 Seg mAP
        seg_file = find_file(dir_path, "seg_results*.txt")
        seg_x, seg_y = parse_results_file(seg_file)

        plot_data[label] = {
            "loss": (loss_x, loss_y),
            "det": (det_x, det_y),
            "seg": (seg_x, seg_y)
        }

    print("\n--- 开始绘图 ---")
    
    # 画 Loss
    draw_single_metric(plot_data, "loss", 
                       "Training Loss Curve", "Loss", "loss_curve.png")
    
    # 画 Detection mAP
    draw_single_metric(plot_data, "det", 
                       "Detection mAP (Box AP) Curve", "Box mAP", "det_map_curve.png")
    
    # 画 Segmentation mAP
    # $符号在matplotlib里会渲染成数学公式字体
    # 标题简洁，Y轴明确
    draw_single_metric(plot_data, "seg", "Validation Mask mAP", "Mask mAP", "seg_map_curve.png")    

if __name__ == "__main__":
    plot_curves()