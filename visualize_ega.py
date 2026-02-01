import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
""""
为了实现你在论文中提到的可视化效果（如 Scharr 提取边缘、Gaussian 抑制纹理），我为你编写了一个全新的可视化脚本 visualize_ega.py。

这个脚本比原来的更高级，它不是简单地抓取输出，而是直接深入到你代码中的 debug_feat 埋点（我在你的 legnet.py 中看到了这个埋点，非常有先见之明！），从而提取出最纯粹的算子响应图。
"""
# 引入你的模型定义
from backbone.legnet import legnet_fpn_backbone
from network_files import MaskRCNN

def normalize_map(tensor):
    """
    将特征图归一化到 0-255 并转为热力图
    """
    if tensor is None:
        return None
        
    # 如果是 [1, C, H, W] -> [C, H, W]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 核心步骤：最大值投影 (Max Projection)
    # 将多个通道的信息压缩成一张图，保留最强的响应
    map_data = torch.max(tensor, dim=0)[0].cpu().detach().numpy()
    
    # 归一化 (Min-Max Normalization)
    min_val = map_data.min()
    max_val = map_data.max()
    if max_val - min_val > 1e-6:
        map_data = (map_data - min_val) / (max_val - min_val)
    else:
        map_data = np.zeros_like(map_data)
        
    map_data = np.uint8(255 * map_data)
    
    # 应用 JET 颜色映射 (蓝=弱, 红=强)
    heatmap = cv2.applyColorMap(map_data, cv2.COLORMAP_JET)
    # 转回 RGB 供 matplotlib 显示
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

def visualize_ega_module(img_path, weights_path, output_dir="vis_results_ega"):
    """
    专门用于可视化 EGA 模块 (Scharr, LoG, Gaussian) 的脚本
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 构建模型
    print("Building LEGNet model...")
    # backbone
    backbone = legnet_fpn_backbone(pretrain_path="", ablation_mode="full")
    model = MaskRCNN(backbone, num_classes=2) # 假设2类(背景+石头)，这不影响特征提取
    
    # 2. 加载权重 (如果有)
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        ckpt = torch.load(weights_path, map_location=device)
        model_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(model_dict, strict=False)
    else:
        print("[Info] No weights found or provided. Using random init.")
        print("Note: Scharr/LoG/Gaussian are FIXED operators, so they work even without training!")

    model.to(device)
    model.eval()

    # 3. 注册 LoG 的钩子 (因为 LoGFilter 没有 debug_feat 埋点)
    log_features = {}
    def hook_log(module, input, output):
        log_features['LoG'] = output.detach()
    
    # 定位到 Stem 中的 LoG 卷积层
    # 路径: model -> backbone -> body -> Stem -> LoG -> LoG (卷积层)
    # 根据你的代码逻辑，Stem.LoG 是 LoGFilter 类，里面的 self.LoG 是卷积
    try:
        model.backbone.body.Stem.LoG.LoG.register_forward_hook(hook_log)
        print("Hook registered for LoG layer.")
    except AttributeError:
        print("[Warning] Could not find LoG layer path. Skipping LoG vis.")

    # 4. 读取图片
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    raw_img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    # 5. 前向传播
    print("Running inference...")
    with torch.no_grad():
        model(input_tensor)

    # 6. 提取特征图
    print("Extracting features...")
    results = {}
    
    # (a) 原始图
    results['Original'] = np.array(raw_img)

    # (b) LoG 响应 (从钩子获取)
    if 'LoG' in log_features:
        results['LoG Response (Stage 0)'] = normalize_map(log_features['LoG'])

    # (c) Scharr 响应 (从 debug_feat 获取)
    # 路径: Stage 0 -> Block 0 -> edge_extractor
    try:
        scharr_module = model.backbone.body.stages[0].blocks[0].edge_extractor
        if scharr_module and scharr_module.debug_feat is not None:
            results['Scharr Response (Stage 0)'] = normalize_map(scharr_module.debug_feat)
        else:
            print("Scharr debug feature is empty.")
    except Exception as e:
        print(f"Error accessing Scharr: {e}")

    # (d) Gaussian 响应 (从 debug_feat 获取)
    # 路径: Stage 3 (index 6) -> Block 0 -> gaussian
    # LWEGNet stages list: 0(S0), 1(DRFD), 2(S1), 3(DRFD), 4(S2), 5(DRFD), 6(S3)
    try:
        gaussian_module = model.backbone.body.stages[6].blocks[0].gaussian
        if gaussian_module and gaussian_module.debug_feat is not None:
            results['Gaussian Response (Stage 3)'] = normalize_map(gaussian_module.debug_feat)
        else:
            print("Gaussian debug feature is empty.")
    except Exception as e:
        print(f"Error accessing Gaussian: {e}")

    # 7. 绘图并保存
    plt.figure(figsize=(20, 5))
    for i, (name, img) in enumerate(results.items()):
        plt.subplot(1, len(results), i+1)
        plt.imshow(img)
        plt.title(name)
        plt.axis('off')
    
    save_file = os.path.join(output_dir, "EGA_Module_Visualization.png")
    plt.savefig(save_file, bbox_inches='tight')
    print(f"Result saved to {save_file}")
    plt.show()

if __name__ == "__main__":
    # === 配置区 ===
    img_path = r"D:\MASKRCNN_daima\111111mask_rcnn_B_up_pilibala\test_image\p2.jpg" # 换成你的图片路径
    weights_path = "save_weights1/model_84.pth"   # 换成你的权重路径
    
    visualize_ega_module(img_path, weights_path)