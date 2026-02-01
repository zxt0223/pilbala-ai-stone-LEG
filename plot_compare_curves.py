import matplotlib.pyplot as plt
import os

# ================= 数据准备 =================
# 1. LEGNet 数据 (From Scratch, 10% Data)
legnet_data = [
    (0, 0.0031), (1, 0.0106), (2, 0.0251), (3, 0.0876), (4, 0.1208),
    (5, 0.1361), (6, 0.1551), (7, 0.1975), (8, 0.2444), (9, 0.2593),
    (10, 0.2809), (11, 0.3004), (12, 0.3085), (13, 0.2959), (14, 0.3080),
    (15, 0.3016), (16, 0.3071), (17, 0.3144), (18, 0.3056), (19, 0.3062)
]

def read_results(file_path):
    """读取 txt 文件中的 epoch 和 mAP"""
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping...")
        return []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("epoch:"):
                try:
                    parts = line.split()
                    epoch = int(parts[0].split(":")[1])
                    map_val = float(parts[1]) # mAP 0.5:0.95
                    data.append((epoch, map_val))
                except:
                    pass
    return data

# 2. 读取 ResNet 数据 (请确保这俩文件在当前目录下)
resnet18_data = read_results(r"/group/chenjinming/wyy/pytorch-pilipala-LEG/output_gpu7_legnet_10percent/det_results_resnet18_frac0.1.txt")
resnet50_data = read_results(r"/group/chenjinming/wyy/pytorch-pilipala-LEG/output_gpu7_legnet_10percent/det_results_resnet50_frac0.1.txt")

# ================= 开始绘图 =================
plt.figure(figsize=(10, 7))

# 提取 X, Y
def get_xy(data): 
    if not data: return [], []
    return [x[0] for x in data], [x[1] for x in data]

l_x, l_y = get_xy(legnet_data)
r18_x, r18_y = get_xy(resnet18_data)
r50_x, r50_y = get_xy(resnet50_data)

# 绘制曲线
# Line 1: LEGNet (红色实线，加粗)
plt.plot(l_x, l_y, color='#D62728', linestyle='-', linewidth=2.5, 
         marker='o', markersize=6, label='LEGNet (Ours, From Scratch)')

# Line 2: ResNet-50 (蓝色虚线)
if r50_x:
    plt.plot(r50_x, r50_y, color='#1F77B4', linestyle='--', linewidth=2, 
             label='ResNet-50 (ImageNet Pre-trained)')

# Line 3: ResNet-18 (绿色点线)
if r18_x:
    plt.plot(r18_x, r18_y, color='#2CA02C', linestyle=':', linewidth=2, 
             label='ResNet-18 (ImageNet Pre-trained)')

# 图表装饰
plt.title("Few-Shot Performance: Structural Priors vs. Pre-training", fontsize=15, fontweight='bold')
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("mAP (IoU=0.5:0.95)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
# Legend 放在右下角，避免遮挡曲线
plt.legend(fontsize=12, loc='lower right', frameon=True, shadow=True)

# === [关键修改] 优化峰值标注位置 ===
if l_y:
    peak_val = max(l_y)
    peak_idx = l_y.index(peak_val)
    peak_epoch = l_x[peak_idx]
    
    plt.annotate(f'Peak: {peak_val:.1%}', 
                 xy=(peak_epoch, peak_val), 
                 # 改动1: xytext=(0, 15) 表示在点正上方 15 个像素位置
                 xytext=(0, 15), 
                 textcoords='offset points', 
                 color='#D62728', 
                 fontweight='bold',
                 ha='center', # 水平居中
                 # 改动2: 加上箭头，让文字可以离得更远而不乱
                 arrowprops=dict(arrowstyle="->", color='#D62728', shrinkA=0, shrinkB=5))

plt.tight_layout()

# 保存
save_path = "few_shot_comparison_fixed.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")
plt.show()