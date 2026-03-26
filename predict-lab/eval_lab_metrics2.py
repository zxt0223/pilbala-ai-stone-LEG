import sys
import os
                                                          #兼容stone123
# ================= 配置区域 =================
# 指定物理 GPU ID (例如 5)
PHYSICAL_GPU_ID = 5 
# 强制屏蔽其他显卡，只让程序看到这张卡（此时它在程序内变成 cuda:0）
os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_GPU_ID)

import json
import glob
import csv
import cv2
import numpy as np
import torch
from PIL import Image, ImageFont
from torchvision import transforms
from tqdm import tqdm

# --------- 关键修复：给 Pillow 打补丁 (Fix for Pillow >= 10.0.0) ---------
# 新版 Pillow 移除了 getsize，这里手动加回去，兼容 draw_box_utils
if not hasattr(ImageFont.FreeTypeFont, 'getsize'):
    def getsize(self, text):
        left, top, right, bottom = self.getbbox(text)
        return right - left, bottom - top
    ImageFont.FreeTypeFont.getsize = getsize
    # 同时给默认字体也加上，以防万一
    ImageFont.ImageFont.getsize = getsize
    print("[System] 已应用 Pillow getsize 兼容性补丁")
# ---------------------------------------------------------------------

# 项目根目录
PROJECT_ROOT = "/group/chenjinming/wyy/pytorch-pilipala-LEG"
sys.path.append(PROJECT_ROOT)

# 权重和日志所在目录
WEIGHTS_DIR = "/group/chenjinming/wyy/pytorch-pilipala-LEG/output++lab/full"
ABLATION_MODE = "full"
DATA_DIR = "/group/chenjinming/Datas/test-img-json"
OUTPUT_ROOT = "/group/chenjinming/Datas/test-img-outputs"
# ===========================================

try:
    from backbone.legnet import legnet_fpn_backbone
    from network_files import MaskRCNN
    # 导入绘图工具
    from draw_box_utils import draw_objs
except ImportError:
    print(f"错误: 无法导入项目模块，请确保脚本运行在 {PROJECT_ROOT} 下")
    sys.exit(1)

# ------------------- 核心功能类 -------------------
class LabelMeLoader:
    def __init__(self, target_label_prefix="stone"):
        # 这里改为记录“前缀”
        self.target_label_prefix = target_label_prefix

    def get_gt_info(self, json_path, img_shape):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        h, w = img_shape
        masks = []
        for shape in data['shapes']:
            label = shape.get('label', '')
            
            # --- 关键修改：兼容 stone1, stone2, stone_big 等情况 ---
            # 只要标签是以 target_label_prefix (默认"stone") 开头，都算作目标
            if not label.startswith(self.target_label_prefix):
                continue
            
            points = np.array(shape['points'], dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            masks.append(mask)
            
        if not masks:
            return np.zeros((0, h, w), dtype=np.uint8), 0
            
        return np.stack(masks, axis=0), len(masks)

class ModelPredictor:
    def __init__(self, device):
        self.device = device
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.model = None

    def load_model(self, weights_path):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        try:
            backbone = legnet_fpn_backbone(pretrain_path="", ablation_mode=ABLATION_MODE)
        except TypeError:
            backbone = legnet_fpn_backbone(pretrain_path="")
            
        model = MaskRCNN(backbone, num_classes=2)
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model

    def predict(self, img_path, conf_thresh=0.5):
        image = Image.open(img_path).convert("RGB")
        w, h = image.size 
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)[0]
        
        scores = outputs['scores'].cpu().numpy()
        keep = scores >= conf_thresh
        
        boxes = outputs['boxes'][keep].cpu().numpy()
        classes = outputs['labels'][keep].cpu().numpy()
        scores = scores[keep]
        
        raw_masks = outputs['masks'][keep].cpu().numpy().squeeze(1)
        pred_masks_binary = (raw_masks > 0.5).astype(np.uint8)
        
        prediction = {
            "masks": pred_masks_binary,
            "boxes": boxes,
            "classes": classes,
            "scores": scores
        }
        
        return prediction, (h, w), image

class MetricCalculator:
    @staticmethod
    def compute_iou(mask1, mask2):
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return inter / union if union > 0 else 0

    @staticmethod
    def calculate(gt_masks, pred_masks, iou_thresh=0.1):
        num_gt = len(gt_masks)
        num_pred = len(pred_masks)
        metrics = {"coverage": 0.0, "repetition": 0.0, "fragmentation": 0.0, 
                   "gt_count": num_gt, "correct_count": 0}

        if num_gt == 0: return metrics

        gt_union = np.any(gt_masks, axis=0)
        pred_union = np.any(pred_masks, axis=0) if num_pred > 0 else np.zeros_like(gt_union)
        inter_area = np.logical_and(gt_union, pred_union).sum()
        metrics["coverage"] = inter_area / gt_union.sum() if gt_union.sum() > 0 else 0.0

        if num_pred == 0: return metrics

        hit_matrix = np.zeros((num_gt, num_pred), dtype=int)
        for i in range(num_gt):
            for j in range(num_pred):
                if MetricCalculator.compute_iou(gt_masks[i], pred_masks[j]) > iou_thresh:
                    hit_matrix[i, j] = 1

        gt_hits = np.sum(hit_matrix, axis=1)
        metrics["correct_count"] = np.sum(gt_hits > 0)
        metrics["fragmentation"] = np.sum(gt_hits >= 2) / num_gt

        pred_hits = np.sum(hit_matrix, axis=0)
        valid_preds = np.where(pred_hits > 0)[0]
        if len(valid_preds) > 0:
            covered_gts = set()
            for j in valid_preds:
                covered_gts.update(np.where(hit_matrix[:, j] > 0)[0])
            if len(valid_preds) > len(covered_gts):
                metrics["repetition"] = (len(valid_preds) - len(covered_gts)) / len(valid_preds)
        
        return metrics

class Visualizer:
    @staticmethod
    def draw_metrics(pil_img, metrics, save_path, model_name):
        img_cv = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        
        lines = [
            f"Model: {model_name}",
            f"GT Count: {metrics['gt_count']}",
            f"Correct: {metrics['correct_count']}",
            f"Coverage: {metrics['coverage']:.1%}",
            f"Repeat: {metrics['repetition']:.1%}",
            f"Frag: {metrics['fragmentation']:.1%}"
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        max_w = 0
        for line in lines:
            fw = cv2.getTextSize(line, font, font_scale, thickness)[0][0]
            if fw > max_w: max_w = fw
            
        start_x = w - max_w - 20
        start_y = h - 20 - (len(lines) * 35) + 35
        
        for i, line in enumerate(lines):
            y = start_y + i * 35
            cv2.putText(img_cv, line, (start_x, y), font, font_scale, (0,0,0), thickness+2)
            cv2.putText(img_cv, line, (start_x, y), font, font_scale, (0,0,255), thickness)
            
        cv2.imwrite(save_path, img_cv)

def get_top2_epochs_from_txt(dir_path):
    txt_files = glob.glob(os.path.join(dir_path, "seg_results_*.txt"))
    if not txt_files:
        print(f"[Error] 未在 {dir_path} 找到 seg_results_*.txt")
        return []
    
    latest_txt = max(txt_files, key=os.path.getmtime)
    print(f"\n[Info] 已锁定日志文件: {os.path.basename(latest_txt)}")
    
    epoch_scores = []
    with open(latest_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            if not parts[0].startswith("epoch:"): continue
            try:
                epoch_num = int(parts[0].split(":")[1])
                ap_score = float(parts[1]) 
                epoch_scores.append((epoch_num, ap_score))
            except Exception:
                continue
                
    epoch_scores.sort(key=lambda x: x[1], reverse=True)
    top2 = epoch_scores[:2]
    
    print(f"\n>>> 基于验证集 AP 筛选出的最佳 Epoch:")
    for rank, (ep, score) in enumerate(top2):
        print(f"  Rank {rank+1}: Epoch {ep} (AP: {score:.4f})")
        
    return top2

def main():
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)

    top2_epochs = get_top2_epochs_from_txt(WEIGHTS_DIR)
    if not top2_epochs: return

    # 因为前面设置了 CUDA_VISIBLE_DEVICES=5，所以这里用 cuda:0
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    print(f"[System] 使用设备: {device} (对应物理卡: {PHYSICAL_GPU_ID})")

    # 初始化 Loader，设置前缀为 "stone"
    loader = LabelMeLoader(target_label_prefix="stone")
    
    predictor = ModelPredictor(device)
    img_files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    if not img_files:
        print("[Error] No images found.")
        return

    category_index = {'1': 'stone'}

    print(f"\n>>> Phase 2: 开始生成 Top 2 模型的可视化报告...")
    comparison_report = []
    ranks_label = ["Best", "Second"]
    
    for i, (epoch_num, score) in enumerate(top2_epochs):
        rank_name = ranks_label[i]
        pth_name = f"model_{epoch_num}.pth"
        pth_path = os.path.join(WEIGHTS_DIR, pth_name)
        
        if not os.path.exists(pth_path): continue
            
        save_dir = os.path.join(OUTPUT_ROOT, f"full_{rank_name}_epoch{epoch_num}")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"  正在处理 [{rank_name}]: {pth_name}")
        predictor.load_model(pth_path)
        
        accum_final = {k: [] for k in ["coverage", "repetition", "fragmentation", "gt_count", "correct_count"]}
        
        for img_path in tqdm(img_files, desc=f"  Visualizing {rank_name}"):
            json_path = img_path.replace('.jpg', '.json')
            if not os.path.exists(json_path): continue
            
            # 1. 预测
            pred, shape, pil_img = predictor.predict(img_path)
            
            # 2. 计算指标 (传入兼容后的 mask 列表)
            gt_masks, _ = loader.get_gt_info(json_path, shape)
            res = MetricCalculator.calculate(gt_masks, pred['masks'])
            for k in accum_final: accum_final[k].append(res[k])
            
            # 3. 绘图
            try:
                viz_img = draw_objs(
                    image=pil_img,
                    boxes=pred['boxes'],
                    classes=pred['classes'],
                    scores=pred['scores'],
                    masks=pred['masks'],
                    category_index=category_index,
                    line_thickness=5,
                    font_size=20,
                    draw_boxes_on_image=True,
                    draw_masks_on_image=True
                )
            except Exception as e:
                print(f"[Warn] 绘图出错: {e}")
                viz_img = pil_img
            
            # 4. 保存
            Visualizer.draw_metrics(viz_img, res, 
                                  os.path.join(save_dir, f"eval_{os.path.basename(img_path)}"),
                                  f"{rank_name}-Ep{epoch_num}")
            
        avg = {k: np.mean(v) for k, v in accum_final.items()}
        comparison_report.append({
            "Rank": rank_name,
            "Weights": pth_name,
            "Val_AP": score,
            "Avg_Coverage": avg['coverage'],
            "Avg_Repetition": avg['repetition'],
            "Avg_Fragmentation": avg['fragmentation'],
            "Avg_Correct_Count": avg['correct_count'],
            "Avg_GT_Count": avg['gt_count']
        })

    final_csv = os.path.join(OUTPUT_ROOT, "fast_top2_report.csv")
    if comparison_report:
        with open(final_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=comparison_report[0].keys())
            writer.writeheader()
            writer.writerows(comparison_report)
        print(f"\n✅ 评估完成！结果已保存至: {final_csv}")

if __name__ == "__main__":
    main()