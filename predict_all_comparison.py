import os
import time
import json
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
from collections import OrderedDict

# 导入必要模块
from network_files import MaskRCNN
from backbone.legnet import legnet_fpn_backbone
from draw_box_utils import draw_objs

def create_model(num_classes, box_thresh=0.5, ablation_mode="full"):
    """创建标准的 LEGNet 模型 (2类)"""
    backbone = legnet_fpn_backbone(pretrain_path="", ablation_mode=ablation_mode)
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     min_size=1000, max_size=1333,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)
    return model

def find_best_weights(base_dir, mode_name):
    """
    自动寻找该模式下最新的权重文件
    策略：优先找 run1, run2, run3 中 epoch 最大的文件 (例如 model_23.pth)
    """
    # 假设路径结构: zxt_checkpoints/legnet_{mode}_run{i}/model_{epoch}.pth
    search_pattern = os.path.join(base_dir, f"legnet_{mode_name}_run*", "model_*.pth")
    files = glob.glob(search_pattern)
    
    if not files:
        return None
    
    # 按 epoch 数字排序
    def extract_epoch(path):
        try:
            # 兼容 model_10.pth 和 checkpoint_10.pth
            name = os.path.basename(path)
            return int(name.split("_")[-1].split(".")[0])
        except:
            return -1
            
    best_file = max(files, key=extract_epoch)
    return best_file

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 准备输出
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 2. 准备图片
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    img_list = []
    if os.path.isdir(args.input_dir):
        img_list = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(supported)]
    else:
        print(f"[Error] Input dir {args.input_dir} not found.")
        return

    # 3. 强制定义 Stone 标签
    category_index = {1: 'stone'}

    # 4. 定义要预测的消融模式
    ablation_modes = ["full", "baseline", "no_scharr", "no_gaussian", "no_lfea"]

    print(f"Found {len(img_list)} images. Starting prediction loop...")

    for mode in ablation_modes:
        print(f"\n>>> Processing Mode: {mode.upper()}")
        
        # 自动寻找权重
        weights_path = find_best_weights(args.checkpoints_dir, mode)
        if not weights_path:
            print(f"    [Skip] No weights found for mode '{mode}' in {args.checkpoints_dir}")
            continue
            
        print(f"    [Load] Weights: {weights_path}")

        # 创建模型 (num_classes=2: 1 stone + 1 background)
        model = create_model(num_classes=2, 
                             box_thresh=args.box_thresh, 
                             ablation_mode=mode)
        
        # 加载权重 (增加多GPU去前缀逻辑)
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            # --- [修复] 处理多GPU训练带来的 'module.' 前缀 ---
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # 去掉 'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
            # -----------------------------------------------

            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"    [Error] Loading weights failed: {e}")
            continue

        # 批量预测
        with torch.no_grad():
            for i, img_path in enumerate(img_list):
                file_name = os.path.basename(img_path)
                original_img = Image.open(img_path).convert('RGB')
                data_transform = transforms.Compose([transforms.ToTensor()])
                img = data_transform(original_img)
                img = torch.unsqueeze(img, dim=0).to(device)

                predictions = model(img)[0]
                
                predict_boxes = predictions["boxes"].to("cpu").numpy()
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()
                predict_mask = predictions["masks"].to("cpu").numpy()
                predict_mask = np.squeeze(predict_mask, axis=1)

                if len(predict_boxes) == 0:
                    print(f"    -> {file_name}: No objects detected.")
                    continue

                plot_img = draw_objs(original_img,
                                     boxes=predict_boxes,
                                     classes=predict_classes,
                                     scores=predict_scores,
                                     masks=predict_mask,
                                     category_index=category_index,
                                     line_thickness=3,
                                     font='arial.ttf',
                                     font_size=20)
                
                # 保存: full_stone.jpg
                save_name = f"{mode}_{file_name}"
                plot_img.save(os.path.join(args.output_dir, save_name))
                print(f"    -> Saved: {save_name}")

    print(f"\nAll Done! Results in: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict newly retrained LEGNet models")
    parser.add_argument('--input-dir', default='/group/chenjinming/wyy/pytorch-pilipala-LEG/test_image', help='测试图片文件夹')
    parser.add_argument('--output-dir', default='/group/chenjinming/wyy/pytorch-pilipala-LEG/comparison_results_new', help='结果保存路径')
    # 你的权重根目录
    parser.add_argument('--checkpoints-dir', default='./zxt_checkpoints', help='权重根目录') 
    # 建议设低一点，让Baseline暴露出更多误检，对比更强烈
    parser.add_argument('--box-thresh', type=float, default=0.3, help='置信度阈值')
    
    args = parser.parse_args()
    main(args)