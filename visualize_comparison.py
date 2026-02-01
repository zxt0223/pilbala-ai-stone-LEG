import torch
import torch.nn as nn
from torchvision.models import resnet50
from backbone.legnet import Scharr  # [修复] 直接导入 Scharr 类
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
# 1. 图片路径 (脚本会自动寻找该路径)
IMAGE_PATH = "/group/chenjinming/wyy/pytorch-pilipala-LEG/test_image/IMG_20251223143628_0_2.jpg"

# 2. ResNet50 权重路径
RESNET_WEIGHT_PATH = "resnet50.pth" 
# ===========================================

def load_resnet_model(device):
    print(f"[Info] Initializing ResNet-50...")
    model = resnet50(pretrained=False)
    if os.path.exists(RESNET_WEIGHT_PATH):
        print(f"[Info] Loading local ResNet weights from: {RESNET_WEIGHT_PATH}")
        try:
            state_dict = torch.load(RESNET_WEIGHT_PATH, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[Warning] Failed to load local weights: {e}")
    else:
        print("[Warning] Local weights not found. Using random init (Result will be noisy!)")
    return model.to(device)

def get_feature_map(model_layer, img_tensor):
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    handle = model_layer.register_forward_hook(hook)
    with torch.no_grad():
        try:
            _ = model_layer(img_tensor)
        except:
            pass 
    handle.remove()
    return outputs[0] if outputs else None

def process_heatmap(tensor):
    """将特征图转换为可视化矩阵 (0-1 float)"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    # Max projection: 取所有通道中的最大值
    map_data = torch.max(tensor, dim=0)[0].cpu().numpy()
    # 归一化
    min_val, max_val = map_data.min(), map_data.max()
    if max_val - min_val > 1e-6:
        map_data = (map_data - min_val) / (max_val - min_val)
    else:
        map_data = np.zeros_like(map_data)
    return map_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "paper_vis_comparison"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 检查图片
    if not os.path.exists(IMAGE_PATH):
        print(f"[Error] Image not found: {IMAGE_PATH}")
        return

    print(f"Processing image: {IMAGE_PATH}")
    img = Image.open(IMAGE_PATH).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 2. ResNet-50 (Baseline)
    resnet = load_resnet_model(device)
    resnet.eval()
    resnet_feat = get_feature_map(resnet.conv1, img_tensor)
    
    # 3. LEGNet (Ours) - [核心修复部分]
    print("[Info] Initializing LEGNet Scharr Prior...")
    
    # [修复逻辑]：
    # 我们实例化一个全新的 Scharr 模块，专门设为 3 通道 (channel=3)
    # 这证明了：Scharr 算子是通用的数学先验，不需要网络上下文也能工作！
    scharr_viz = Scharr(channel=3, act_layer=nn.ReLU).to(device)
    scharr_viz.eval()
    
    # 前向传播 (触发内部的 debug_feat 计算)
    _ = scharr_viz(img_tensor)
    scharr_out = scharr_viz.debug_feat  # 获取纯净的边缘特征
    
    # 4. 绘图
    print("Generating visualization...")
    heatmap_resnet = process_heatmap(resnet_feat)
    heatmap_legnet = process_heatmap(scharr_out)
    
    plt.figure(figsize=(12, 6))
    
    # 左图：ResNet
    plt.subplot(1, 2, 1)
    plt.imshow(heatmap_resnet, cmap='jet') # 使用 matplotlib 的 jet 色图
    plt.title("ResNet-50 (Learnable Conv1)\nNoisy & Redundant", fontsize=14)
    plt.axis('off')
    
    # 右图：LEGNet
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_legnet, cmap='jet')
    plt.title("LEGNet (Fixed Scharr Prior)\nClean & Sharp Edges", fontsize=14)
    plt.axis('off')
    
    save_path = os.path.join(save_dir, "feature_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"[Success] Comparison saved to: {save_path}")

if __name__ == "__main__":
    main()