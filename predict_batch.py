import os
import time
import torch
import numpy as np
import cv2
import random
import colorsys
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
import warnings

# 屏蔽警告
warnings.filterwarnings("ignore")

# 导入网络模块 (不再导入 draw_box_utils)
from network_files import MaskRCNN
from backbone.legnet import legnet_fpn_backbone

# ==============================================================================
# 【配置区】您指定的路径
# ==============================================================================
INPUT_DIR = r"/group/chenjinming/wyy/pytorch-pilipala-LEG/test_image" #/group/chenjinming/wyy/pytorch-pilipala-LEG/test_image_cropped_8parts
OUTPUT_DIR = r"/group/chenjinming/wyy/pytorch-pilipala-LEG/comparison_results_new"
WEIGHTS_PATH = r"/group/chenjinming/wyy/pytorch-pilipala-LEG/output_best_20260201_182928_extended_36e/model_35.pth"

# 绘图显示控制
DRAW_BOX = True        # 是否画方框
DRAW_SCORE = True      # 是否画分数
DRAW_MASK = True       # 是否画掩膜
COLOR_STYLE = 'random' # 'random' 彩色 or 'green' 单色
MASK_ALPHA = 0.45      # 掩膜透明度
# ==============================================================================

def generate_colors(num_colors):
    """生成区分度高的随机颜色列表 (RGB)"""
    colors = []
    for i in range(num_colors):
        h = i / num_colors
        s = 0.8 + (i % 2) * 0.1
        l = 0.5 + (i % 2) * 0.1
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    random.shuffle(colors)
    return colors

def custom_draw(img_pil, boxes, scores, masks, color_style='random'):
    """使用 OpenCV 自定义绘图，完全避开 PIL 的 getsize 报错问题"""
    image = np.array(img_pil) # 转 numpy (H, W, 3) RGB
    
    num_objs = len(boxes)
    if color_style == 'random':
        colors = generate_colors(num_objs)
    else:
        colors = [(0, 255, 0)] * num_objs

    # 1. 绘制 Mask (使用半透明叠加)
    if DRAW_MASK:
        mask_overlay = image.copy()
        for i in range(num_objs):
            color = colors[i]
            mask = masks[i] > 0.5
            mask_overlay[mask] = color
        image = cv2.addWeighted(image, 1 - MASK_ALPHA, mask_overlay, MASK_ALPHA, 0)

    # 2. 绘制 Box 和 Text
    for i in range(num_objs):
        x1, y1, x2, y2 = map(int, boxes[i])
        color = colors[i]
        
        if DRAW_BOX:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        if DRAW_SCORE:
            text = f"{int(scores[i]*100)}%"
            # 使用 OpenCV 获取文字大小，不再依赖 PIL
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(image, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
    return Image.fromarray(image)

def create_model(num_classes, box_thresh=0.5):
    """创建模型"""
    backbone = legnet_fpn_backbone(pretrain_path="", ablation_mode="full")
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     min_size=1000, max_size=1333,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model

def main():
    num_classes = 2  
    box_thresh = 0.5 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.exists(INPUT_DIR):
        print(f"[Error] 未找到输入文件夹: {INPUT_DIR}")
        return
        
    img_list = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(supported)]
    print(f"[*] 找到 {len(img_list)} 张图片待处理。")

    print(f"[*] 正在加载模型权重: {WEIGHTS_PATH}")
    model = create_model(num_classes=num_classes, box_thresh=box_thresh)
    
    try:
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False) 
        model.to(device)
        model.eval()
        print("[*] 权重加载成功！开始预测...")
    except Exception as e:
        print(f"[Error] 加载权重失败: {e}")
        return

    with torch.no_grad():
        for i, img_path in enumerate(img_list):
            file_name = os.path.basename(img_path)
            original_img = Image.open(img_path).convert('RGB')
            
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0).to(device)

            t_start = time.time()
            predictions = model(img)[0]
            t_end = time.time()
            
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_mask = predictions["masks"].to("cpu").numpy()
            predict_mask = np.squeeze(predict_mask, axis=1)

            if len(predict_boxes) == 0:
                print(f"  [{i+1}/{len(img_list)}] {file_name}: 未检测到目标。 (耗时: {t_end - t_start:.3f}s)")
                continue

            # 使用修复后的 OpenCV 绘制函数
            plot_img = custom_draw(original_img, 
                                   boxes=predict_boxes, 
                                   scores=predict_scores, 
                                   masks=predict_mask, 
                                   color_style=COLOR_STYLE)
            
            save_name = f"result_{file_name}"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            plot_img.save(save_path)
            
            print(f"  [{i+1}/{len(img_list)}] {file_name} -> 检测到 {len(predict_boxes)} 个目标, 已保存 (耗时: {t_end - t_start:.3f}s)")

    print(f"\n[*] 所有预测完成！结果保存在: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()