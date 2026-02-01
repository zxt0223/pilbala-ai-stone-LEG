import sys
import os
import json
import glob
import time
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ------------------- 路径配置 -------------------
# 项目根目录 (根据您的文件结构，脚本放在项目根目录下或手动添加路径)
PROJECT_ROOT = "/group/chenjinming/wyy/pytorch-pilipala-LEG"
sys.path.append(PROJECT_ROOT)

# 导入项目模块
try:
    from backbone.legnet import legnet_fpn_backbone
    from network_files import MaskRCNN
except ImportError:
    print(f"错误: 无法导入项目模块，请确保脚本运行在 {PROJECT_ROOT} 下或已正确设置 PYTHONPATH")
    sys.exit(1)

# 输入输出路径
DATA_DIR = "/group/chenjinming/Datas/test-img-json"
WEIGHTS_ROOT = "/group/chenjinming/wyy/pytorch-pilipala-LEG/output++lab"
OUTPUT_DIR = "/group/chenjinming/Datas/test-img-outputs"

# ------------------- 模块 1: LabelMe 数据加载器 -------------------
class LabelMeLoader:
    """负责加载 LabelMe JSON 并转换为 GT Masks"""
    def __init__(self, target_label="stone"):
        self.target_label = target_label

    def get_gt_masks(self, json_path, img_shape):
        """
        解析 JSON 返回二值掩码列表
        img_shape: (height, width)
        Returns: gt_masks (N, H, W) numpy array, uint8
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        h, w = img_shape
        masks = []
        
        for shape in data['shapes']:
            if shape['label'] != self.target_label:
                continue
            
            points = np.array(shape['points'], dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            # 填充多边形
            cv2.fillPoly(mask, [points], 1)
            masks.append(mask)
            
        if not masks:
            return np.zeros((0, h, w), dtype=np.uint8)
            
        return np.stack(masks, axis=0)

# ------------------- 模块 2: 模型预测器 -------------------
class ModelPredictor:
    """负责加载 LEGNet 模型并进行推理"""
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.model = None

    def load_model(self, weights_path, ablation_mode):
        print(f"正在加载模型: Mode={ablation_mode}, Weights={os.path.basename(weights_path)}")
        
        # 构建 LEGNet 模型 (注意 num_classes=2: 背景 + stone)
        # 假设 legnet_fpn_backbone 接受 ablation_mode 参数
        try:
            backbone = legnet_fpn_backbone(pretrain_path="", ablation_mode=ablation_mode)
        except TypeError:
            # 兼容旧接口或不同参数名，如果报错尝试不传 mode
            print("警告: legnet_fpn_backbone 不接受 ablation_mode，尝试默认加载")
            backbone = legnet_fpn_backbone(pretrain_path="")

        model = MaskRCNN(backbone, num_classes=2)
        
        # 加载权重
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # 处理可能的 DDP 'module.' 前缀
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict, strict=False) # strict=False 防止部分不匹配
        model.to(self.device)
        model.eval()
        self.model = model

    def predict(self, img_path, conf_thresh=0.5):
        image = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        h, w = image.size[1], image.size[0]
        
        with torch.no_grad():
            outputs = self.model(img_tensor)[0]
        
        # 过滤低置信度
        scores = outputs['scores'].cpu().numpy()
        keep_idxs = scores >= conf_thresh
        
        pred_masks = outputs['masks'][keep_idxs].cpu().numpy().squeeze(1) # (N, H, W) float
        pred_masks = (pred_masks > 0.5).astype(np.uint8) # 二值化
        
        return pred_masks, (h, w)

# ------------------- 模块 3: 指标计算器 -------------------
class MetricCalculator:
    """负责计算覆盖率、重复率、碎块率"""
    
    @staticmethod
    def compute_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    @staticmethod
    def calculate(gt_masks, pred_masks, iou_thresh=0.1):
        """
        :param gt_masks: (M, H, W)
        :param pred_masks: (N, H, W)
        :param iou_thresh: 判定命中的 IoU 阈值
        :return: (coverage, repetition, fragmentation)
        """
        num_gt = len(gt_masks)
        num_pred = len(pred_masks)
        
        # --- 1. 覆盖率 (Pixel Coverage) ---
        # 计算所有 GT 的并集区域被预测 Mask 并集区域覆盖的比例
        if num_gt == 0:
            return 0.0, 0.0, 0.0
            
        gt_union = np.any(gt_masks, axis=0)
        pred_union = np.any(pred_masks, axis=0) if num_pred > 0 else np.zeros_like(gt_union)
        
        intersection_area = np.logical_and(gt_union, pred_union).sum()
        gt_area = gt_union.sum()
        coverage_rate = intersection_area / gt_area if gt_area > 0 else 0.0

        if num_pred == 0:
            return coverage_rate, 0.0, 0.0

        # --- 构建关系矩阵 ---
        # hit_matrix[i, j] = 1 if GT_i matches Pred_j
        hit_matrix = np.zeros((num_gt, num_pred), dtype=int)
        for i in range(num_gt):
            for j in range(num_pred):
                iou = MetricCalculator.compute_iou(gt_masks[i], pred_masks[j])
                if iou > iou_thresh:
                    hit_matrix[i, j] = 1

        # --- 2. 分割碎块率 (Fragmentation Rate) ---
        # 定义: 单个 GT 被 >=2 个 Pred 覆盖 (且这些 Pred 互不包含，这里简化为命中数)
        # 例如: 一块大石头被预测成了两块小石头
        fragmented_count = 0
        for i in range(num_gt):
            hits = np.sum(hit_matrix[i, :])
            if hits >= 2:
                fragmented_count += 1
        fragmentation_rate = fragmented_count / num_gt

        # --- 3. 重复率 (Repetition Rate) ---
        # 定义: 有效预测数量中，属于"多余/重复"预测的比例
        # 简单逻辑: 总命中 Pred 数 - 唯一命中的 GT 数 / 总命中 Pred 数
        # 例如: 5个 Pred 命中了同一个 GT，那么有4个是重复的
        
        pred_hits_gt = np.sum(hit_matrix, axis=0) # 每个 Pred 命中了多少 GT
        valid_preds_indices = np.where(pred_hits_gt > 0)[0] # 有效的 Pred (至少命中一个 GT)
        num_valid_preds = len(valid_preds_indices)
        
        if num_valid_preds == 0:
            repetition_rate = 0.0
        else:
            # 统计这些有效 Pred 到底覆盖了多少个唯一的 GT
            covered_gt_indices = set()
            for j in valid_preds_indices:
                # 找到该 Pred 命中且 IoU 最大的 GT
                # (这里简化处理，直接看 hit_matrix)
                gts_hit_by_this_pred = np.where(hit_matrix[:, j] > 0)[0]
                for gt_idx in gts_hit_by_this_pred:
                    covered_gt_indices.add(gt_idx)
            
            num_unique_gts = len(covered_gt_indices)
            
            # 如果 10个 Pred 实际上只覆盖了 6个 GT，说明有 4个是重复/多余的
            # 若结果 < 0 (如1个Pred盖俩GT)，则重复率为0
            if num_valid_preds > num_unique_gts:
                repetition_rate = (num_valid_preds - num_unique_gts) / num_valid_preds
            else:
                repetition_rate = 0.0

        return coverage_rate, repetition_rate, fragmentation_rate

# ------------------- 主程序 -------------------
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    loader = LabelMeLoader()
    predictor = ModelPredictor()
    
    # 1. 获取所有测试图片
    img_files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    if not img_files:
        print(f"未找到图片: {DATA_DIR}")
        return
    print(f"找到 {len(img_files)} 张测试图片")

    # 2. 扫描 output++lab 下的所有实验文件夹
    exp_modes = [d for d in os.listdir(WEIGHTS_ROOT) if os.path.isdir(os.path.join(WEIGHTS_ROOT, d))]
    print(f"待评估实验模式: {exp_modes}")
    
    all_results = []

    for mode in exp_modes:
        exp_path = os.path.join(WEIGHTS_ROOT, mode)
        
        # 寻找最新的权重 (例如 model_23.pth)
        pth_files = glob.glob(os.path.join(exp_path, "model_*.pth"))
        if not pth_files:
            continue
        best_pth = max(pth_files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        # 加载模型
        try:
            predictor.load_model(best_pth, ablation_mode=mode)
        except Exception as e:
            print(f"加载模型失败 {mode}: {e}")
            continue
            
        # 遍历图片评估
        metrics_accum = {'cov': [], 'rep': [], 'frag': []}
        
        print(f"\n>>> 正在评估: {mode} (Weights: {os.path.basename(best_pth)})")
        for img_path in tqdm(img_files):
            json_path = img_path.replace('.jpg', '.json')
            if not os.path.exists(json_path):
                continue # 没有标注跳过
            
            # 预测
            pred_masks, shape = predictor.predict(img_path)
            
            # 获取 GT
            gt_masks = loader.get_gt_masks(json_path, shape)
            
            # 计算指标
            cov, rep, frag = MetricCalculator.calculate(gt_masks, pred_masks)
            
            metrics_accum['cov'].append(cov)
            metrics_accum['rep'].append(rep)
            metrics_accum['frag'].append(frag)
            
        # 统计该模式的平均值
        avg_cov = np.mean(metrics_accum['cov'])
        avg_rep = np.mean(metrics_accum['rep'])
        avg_frag = np.mean(metrics_accum['frag'])
        
        print(f"   [结果] 覆盖率: {avg_cov:.2%} | 重复率: {avg_rep:.2%} | 碎块率: {avg_frag:.2%}")
        
        all_results.append({
            "Mode": mode,
            "Weights": os.path.basename(best_pth),
            "Coverage": avg_cov,
            "Repetition": avg_rep,
            "Fragmentation": avg_frag
        })

    # 3. 保存最终报表
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n评估完成！结果已保存至: {csv_path}")
    print(df)

if __name__ == "__main__":
    main()