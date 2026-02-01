"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 导入你的模型
from backbone.legnet import legnet_fpn_backbone
from network_files import MaskRCNN

def normalize_feat(feat_map):
    """将特征图归一化到 0-1 之间，方便绘图"""
    min_val = feat_map.min()
    max_val = feat_map.max()
    norm = (feat_map - min_val) / (max_val - min_val + 1e-8)
    return norm

def main():
    # ================= 配置区域 =================
    # 1. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 权重路径 (请修改为你训练好的权重!)
    # 强烈建议使用 No LFEA 的权重，因为那是你的最佳模型
    weights_path = "./zxt_checkpoints/legnet_no_lfea_run3/model_24.pth" 
    
    # 3. 测试图片路径
    img_path = "./my_stone_image.jpg"  # 换成你的一张石头图片路径
    
    # 4. 模型模式 (必须与训练时一致，如果是 No LFEA 权重，这里就填 no_lfea)
    ablation_mode = "no_lfea" 
    # ===========================================

    print(f"Loading weights from: {weights_path}")
    
    # 创建模型
    backbone = legnet_fpn_backbone(pretrain_path="", ablation_mode=ablation_mode)
    model = MaskRCNN(backbone, num_classes=2) # 1类石头+背景
    
    # 加载权重
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        # 兼容 key
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    else:
        print(f"[Error] Weights not found: {weights_path}")
        return

    model.to(device)
    model.eval()

    # 读取图片
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
        
    original_img = Image.open(img_path).convert('RGB')
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    img = img.unsqueeze(0).to(device) # [1, 3, H, W]

    # 前向传播 (这会触发 legnet 里的 debug_feat 保存)
    print("Running inference...")
    with torch.no_grad():
        output = model(img)

    # === 提取并画图 ===
    
    # 1. 提取 Pure Scharr (Stage 0)
    try:
        # 获取埋点特征
        # 路径: backbone -> body -> stages[0] -> blocks[0] -> edge_extractor -> debug_feat
        # 注意：如果 ablation_mode="no_scharr"，edge_extractor 会是 None，这里会报错
        scharr_feat = model.backbone.body.stages[0].blocks[0].edge_extractor.debug_feat
        
        # 处理: [1, C, H, W] -> [H, W] (按通道求均值)
        scharr_map = scharr_feat.squeeze(0).mean(0).cpu().numpy()
        
        # 归一化并保存
        scharr_map = normalize_feat(scharr_map)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(scharr_map, cmap='jet') # jet 也就是热力图颜色
        plt.axis('off')
        plt.title('Pure Scharr Edge (Pre-Activation)')
        plt.savefig('vis_pure_scharr.png', bbox_inches='tight', pad_inches=0)
        print("Saved: vis_pure_scharr.png")
        
    except Exception as e:
        print(f"[Warning] Could not visualize Scharr: {e}")

    # 2. 提取 Pure Gaussian (Stage 3)
    try:
        # 路径: backbone -> body -> stages[3] -> blocks[0] -> gaussian -> debug_feat
        gaussian_feat = model.backbone.body.stages[3].blocks[0].gaussian.debug_feat
        
        gaussian_map = gaussian_feat.squeeze(0).mean(0).cpu().numpy()
        gaussian_map = normalize_feat(gaussian_map)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(gaussian_map, cmap='jet')
        plt.axis('off')
        plt.title('Pure Gaussian Heatmap')
        plt.savefig('vis_pure_gaussian.png', bbox_inches='tight', pad_inches=0)
        print("Saved: vis_pure_gaussian.png")
        
    except Exception as e:
        print(f"[Warning] Could not visualize Gaussian: {e}")

if __name__ == "__main__":
    main()
