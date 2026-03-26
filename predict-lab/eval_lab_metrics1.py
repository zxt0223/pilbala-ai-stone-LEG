import sys
import os
import json
import glob
import csv
import cv2
import numpy as np
import torch
from datetime import datetime  # 新增时间处理模块
from PIL import Image, ImageFont
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= 配置区域 =================
# 指定物理 GPU ID
PHYSICAL_GPU_ID = 5 
os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_GPU_ID)

# --------- 0. 兼容性补丁 (Pillow >= 10.0) ---------
if not hasattr(ImageFont.FreeTypeFont, 'getsize'):
    def getsize(self, text):
        left, top, right, bottom = self.getbbox(text)
        return right - left, bottom - top
    ImageFont.FreeTypeFont.getsize = getsize

# 项目路径配置
PROJECT_ROOT = "/group/chenjinming/wyy/pytorch-pilipala-LEG"
sys.path.append(PROJECT_ROOT)

# 权重目录
WEIGHTS_DIR = "/group/chenjinming/wyy/pytorch-pilipala-LEG/output_best_20260201_182928_extended_36e"
ABLATION_MODE = "full"
DATA_DIR = "/group/chenjinming/Datas/test-img-json"

# >>> 修改点 1：输出目录加上时间戳，防止覆盖 <<<
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = f"/group/chenjinming/Datas/test-img-outputs/run_{current_time}"
# ===========================================

try:
    from backbone.legnet import legnet_fpn_backbone
    from network_files import MaskRCNN
    from draw_box_utils import draw_objs
except ImportError:
    print(f"错误: 无法导入项目模块，请确保脚本运行在 {PROJECT_ROOT} 下")
    sys.exit(1)

# ------------------- 1. 全 GPU 指标计算类 -------------------
class GPUMetricCalculator:
    @staticmethod
    def calculate(gt_masks_tensor, pred_masks_tensor, iou_thresh=0.1):
        device = gt_masks_tensor.device
        num_gt = gt_masks_tensor.shape[0]
        num_pred = pred_masks_tensor.shape[0]
        
        metrics = {"coverage": 0.0, "repetition": 0.0, "fragmentation": 0.0, 
                   "gt_count": num_gt, "correct_count": 0}

        if num_gt == 0: return metrics

        gt_flat = gt_masks_tensor.view(num_gt, -1).float()
        
        # Coverage
        gt_union = (gt_flat.sum(dim=0) > 0).float()
        union_area = gt_union.sum()
        
        if num_pred > 0:
            pred_flat = pred_masks_tensor.view(num_pred, -1).float()
            pred_union = (pred_flat.sum(dim=0) > 0).float()
            inter_area = (gt_union * pred_union).sum()
            metrics["coverage"] = (inter_area / union_area).item() if union_area > 0 else 0.0
        else:
            return metrics

        # IoU Matrix
        intersection = torch.mm(gt_flat, pred_flat.t())
        area_gt = gt_flat.sum(dim=1).view(num_gt, 1)
        area_pred = pred_flat.sum(dim=1).view(1, num_pred)
        union_matrix = area_gt + area_pred - intersection
        iou_matrix = intersection / (union_matrix + 1e-6)

        # Metrics
        hit_matrix = (iou_matrix > iou_thresh).float()

        gt_hits = hit_matrix.sum(dim=1) 
        metrics["correct_count"] = (gt_hits > 0).sum().item()
        metrics["fragmentation"] = ((gt_hits >= 2).sum() / num_gt).item()

        pred_hits = hit_matrix.sum(dim=0)
        valid_preds_idx = torch.where(pred_hits > 0)[0]
        if len(valid_preds_idx) > 0:
            sub_hit_matrix = hit_matrix[:, valid_preds_idx] 
            covered_gts_count = (sub_hit_matrix.sum(dim=1) > 0).sum().item()
            num_valid_preds = len(valid_preds_idx)
            if num_valid_preds > covered_gts_count:
                metrics["repetition"] = (num_valid_preds - covered_gts_count) / num_valid_preds

        return metrics

# ------------------- 2. 数据加载与处理 -------------------
class LabelMeLoader:
    def __init__(self, target_label="stone"):
        self.target_label = target_label

    def get_gt_tensor(self, json_path, img_shape, device):
        if not os.path.exists(json_path):
            return torch.zeros((0, img_shape[0], img_shape[1]), device=device, dtype=torch.uint8)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        h, w = img_shape
        masks = []
        for shape in data['shapes']:
            if shape['label'] != self.target_label: continue
            points = np.array(shape['points'], dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            masks.append(mask)
        
        if not masks: 
            return torch.zeros((0, h, w), device=device, dtype=torch.uint8)
        
        masks_np = np.stack(masks, axis=0)
        return torch.from_numpy(masks_np).to(device)

class InferenceDataset(Dataset):
    def __init__(self, img_dir):
        self.img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), img_path, image.size

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

# ------------------- 3. 模型预测 -------------------
class ModelPredictor:
    def __init__(self, device):
        self.device = device
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

    def predict_gpu(self, img_tensor, conf_thresh=0.5):
        inputs = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)[0]
        
        scores = outputs['scores']
        keep = scores >= conf_thresh
        
        pred_masks = outputs['masks'][keep].squeeze(1) 
        pred_masks_binary = (pred_masks > 0.5).type(torch.uint8)
        
        return {
            "masks": pred_masks_binary,
            "boxes": outputs['boxes'][keep],
            "classes": outputs['labels'][keep],
            "scores": scores[keep]
        }

# ------------------- 4. 可视化 & 搜索工具 -------------------
class Visualizer:
    @staticmethod
    def draw_metrics(pil_img, metrics, save_path):
        # 注意：这里去掉了 model_name 参数，因为不需要在图上写了
        img_cv = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        
        gt = metrics['gt_count']
        ok = metrics['correct_count']
        percent = (ok / gt) if gt > 0 else 0.0
        
        # >>> 修改点 2：去掉了 "Model: ..." 这一行 <<<
        lines = [
            f"GT: {gt} | OK: {ok} ({percent:.1%})",
            f"Cov: {metrics['coverage']:.1%}",
            f"Rep: {metrics['repetition']:.1%}",
            f"Frag: {metrics['fragmentation']:.1%}"
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        start_x = w - 380
        start_y = h - 200
        for i, line in enumerate(lines):
            y = start_y + i * 35
            cv2.putText(img_cv, line, (start_x, y), font, 0.7, (0,0,0), 4)
            cv2.putText(img_cv, line, (start_x, y), font, 0.7, (0,0,255), 2)
        cv2.imwrite(save_path, img_cv)

def find_best_models(dir_path):
    if not os.path.exists(dir_path):
        print(f"[Error] 路径不存在: {dir_path}")
        return []

    # 策略A: 找日志
    txt_files = glob.glob(os.path.join(dir_path, "seg_results_*.txt"))
    if txt_files:
        latest_txt = max(txt_files, key=os.path.getmtime)
        print(f"[Info] 依据日志文件: {os.path.basename(latest_txt)}")
        epoch_scores = []
        with open(latest_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2 or not parts[0].startswith("epoch:"): continue
                try:
                    ep = int(parts[0].split(":")[1])
                    sc = float(parts[1])
                    pth_full = os.path.join(dir_path, f"model_{ep}.pth")
                    if os.path.exists(pth_full):
                        epoch_scores.append((pth_full, sc))
                except: continue
        
        if epoch_scores:
            epoch_scores.sort(key=lambda x: x[1], reverse=True)
            return [(p, f"AP:{s:.4f}", r) for (p, s), r in zip(epoch_scores[:2], ["Best_AP", "2nd_AP"])]

    # 策略B: 找最新文件
    print("[Info] 未找到日志，使用文件时间排序...")
    pth_files = glob.glob(os.path.join(dir_path, "*.pth"))
    if not pth_files: return []
    pth_files.sort(key=os.path.getmtime, reverse=True)
    return [(p, "Latest", r) for p, r in zip(pth_files[:2], ["Newest", "2nd_Newest"])]

# ------------------- 主函数 -------------------
def main():
    # 创建带时间戳的输出根目录
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)
    print(f"[System] 结果将保存至: {OUTPUT_ROOT}")

    models_to_eval = find_best_models(WEIGHTS_DIR)
    if not models_to_eval:
        print(f"[Error] 在 {WEIGHTS_DIR} 未找到任何模型")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[System] 运行设备: {device} (Physical ID: {PHYSICAL_GPU_ID})")

    dataset = InferenceDataset(DATA_DIR)
    if len(dataset) == 0:
        print("[Error] 数据集目录为空")
        return
        
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, 
        num_workers=4, collate_fn=InferenceDataset.collate_fn, pin_memory=True
    )

    loader_tool = LabelMeLoader()
    predictor = ModelPredictor(device)
    category_index = {'1': 'stone'}

    # >>> 修改点 3：这里控制画多少张图，现在是 50 <<<
    VIS_LIMIT = 15 
    
    comparison_report = []

    for (pth_path, score_info, rank_name) in models_to_eval:
        pth_filename = os.path.basename(pth_path)
        # 子文件夹名
        save_dir = os.path.join(OUTPUT_ROOT, f"eval_{rank_name}_{pth_filename.replace('.pth','')}")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n>>> 正在评估: {pth_filename} [{score_info}]")
        predictor.load_model(pth_path)
        
        accum = {k: [] for k in ["coverage", "repetition", "fragmentation", "gt_count", "correct_count"]}
        
        for i, (img_tensors, img_paths, img_sizes) in enumerate(tqdm(data_loader, desc="Progress")):
            img_tensor = img_tensors[0] # Batch=1
            path = img_paths[0]
            w, h = img_sizes[0]
            
            # (A) GPU 推理
            pred_gpu = predictor.predict_gpu(img_tensor)
            
            # (B) GT 转 GPU
            json_path = path.replace('.jpg', '.json')
            gt_gpu = loader_tool.get_gt_tensor(json_path, (h, w), device)
            
            # (C) GPU 算指标
            res = GPUMetricCalculator.calculate(gt_gpu, pred_gpu['masks'])
            for k in accum: accum[k].append(res[k])
            
            # (D) 仅前 VIS_LIMIT 张画图
            if i < VIS_LIMIT:
                try:
                    orig_img = Image.open(path).convert("RGB")
                    viz_img = draw_objs(
                        image=orig_img,
                        boxes=pred_gpu['boxes'].cpu().numpy(),
                        classes=pred_gpu['classes'].cpu().numpy(),
                        scores=pred_gpu['scores'].cpu().numpy(),
                        masks=pred_gpu['masks'].cpu().numpy(),
                        category_index=category_index,
                        line_thickness=5, font_size=20,
                        draw_boxes_on_image=True, draw_masks_on_image=True
                    )
                    # 调用修改后的 draw_metrics（不传 model_name 也没事，或者传了也不显示）
                    Visualizer.draw_metrics(viz_img, res, 
                        os.path.join(save_dir, os.path.basename(path)))
                except Exception as e: 
                    pass

        # 汇总
        avg = {k: np.mean(v) for k, v in accum.items()}
        print(f"   结果 -> Cov: {avg['coverage']:.2%} | Correct: {avg['correct_count']:.2f}")
        
        comparison_report.append({
            "Rank": rank_name,
            "Weights": pth_filename,
            "Score_Info": score_info,
            "Avg_Coverage": avg['coverage'],
            "Avg_Repetition": avg['repetition'],
            "Avg_Fragmentation": avg['fragmentation'],
            "Avg_Correct_Count": avg['correct_count'],
            "Avg_GT_Count": avg['gt_count']
        })

    # 保存 CSV
    csv_path = os.path.join(OUTPUT_ROOT, "final_evaluation_report.csv")
    if comparison_report:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=comparison_report[0].keys())
            writer.writeheader()
            writer.writerows(comparison_report)
        print(f"\n✅ 完成！结果已保存至新的时间戳目录: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()