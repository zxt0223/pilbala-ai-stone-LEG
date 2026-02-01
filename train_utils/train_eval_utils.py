import math
import sys
import time

import torch

import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        mloss = (mloss * i + loss_value) / (i + 1)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

# 找到 train_one_epoch 函数，定位到 optimizer.zero_grad() 这一段
# 将其修改为如下内容：

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            # === 新增：在 step 之前先 unscale，然后进行梯度裁剪 ===
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) 
            # ====================================================
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            # === 新增：普通模式下的梯度裁剪 ===
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            # ==============================
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    det_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="bbox", results_file_name="det_results.json")
    seg_metric = EvalCOCOMetric(data_loader.dataset.coco, iou_type="segm", results_file_name="seg_results.json")
    
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        det_metric.update(targets, outputs)
        seg_metric.update(targets, outputs)
        metric_logger.update(model_time=model_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    det_metric.synchronize_results()
    seg_metric.synchronize_results()

    if utils.is_main_process():
        # 获取 COCO 原始统计信息 (AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100...)
        coco_info_det = det_metric.evaluate()
        coco_info_seg = seg_metric.evaluate()
        
        # === 核心增加：计算 Recall 和 F1 ===
        def calculate_extra_metrics(coco_stats):
            if coco_stats is None: return 0.0, 0.0
            # coco_stats[1] 是 AP50
            # coco_stats[8] 是 AR100 (通常作为 Recall)
            ap50 = coco_stats[1]
            recall = coco_stats[8] 
            
            # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
            # 这里用 AP50 近似 Precision，用 AR100 近似 Recall
            f1 = 0.0
            if (ap50 + recall) > 0:
                f1 = 2 * (ap50 * recall) / (ap50 + recall)
            return recall, f1

        r_det, f1_det = calculate_extra_metrics(coco_info_det)
        r_seg, f1_seg = calculate_extra_metrics(coco_info_seg)

        # 将 F1 和 Recall 追加到返回列表中
        # 列表结构: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl, Recall_Calc, F1_Calc]
        if coco_info_det is not None:
            coco_info_det = list(coco_info_det) + [r_det, f1_det]
        if coco_info_seg is not None:
            coco_info_seg = list(coco_info_seg) + [r_seg, f1_seg]
            
        print(f"\n[Extra Metrics] Det F1: {f1_det:.4f}, Seg F1: {f1_seg:.4f}")

        return coco_info_det, coco_info_seg
    else:
        return None, None