import os
import datetime
import argparse
import torch
import transforms
from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone, resnet18_fpn_backbone, legnet_fpn_backbone
from my_dataset_coco import CocoDetection
from train_utils import train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups

def create_model(num_classes, args):
    # 根据参数自动选择模型
    if args.backbone == "resnet50":
        backbone = resnet50_fpn_backbone(pretrain_path="resnet50.pth", trainable_layers=3)
    elif args.backbone == "resnet18":
        backbone = resnet18_fpn_backbone(pretrain_path="resnet18.pth", trainable_layers=3)
    elif args.backbone == "legnet":
        # pretrain_path="" 表示不加载预训练权重，证明结构优势
        backbone = legnet_fpn_backbone(pretrain_path="", 
                                       ablation_mode=args.ablation_mode,
                                       trainable_layers=3)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    model = MaskRCNN(backbone, num_classes=num_classes, min_size=1000, max_size=1333)
    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 结果文件名
    exp_tag = f"{args.backbone}_{args.ablation_mode}" if args.backbone == "legnet" else args.backbone
    if args.data_fraction < 1.0:
        exp_tag += f"_frac{args.data_fraction}"
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    det_results_file = os.path.join(args.output_dir, f"det_results_{exp_tag}.txt")

    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomColorJitter(brightness=0.3, contrast=0.3, prob=0.5),
            transforms.RandomGaussianBlur(prob=0.3)
        ]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # 加载数据
    train_dataset = CocoDetection(args.data_path, "train", data_transform["train"])
    
    # [核心修复] 在变为 Subset 之前，先提取出 collate_fn
    train_collate_fn = train_dataset.collate_fn

    # [小样本逻辑]
    if args.data_fraction < 1.0:
        total_len = len(train_dataset)
        subset_len = int(total_len * args.data_fraction)
        # 固定种子，确保对比公平
        g = torch.Generator()
        g.manual_seed(42)
        indices = torch.randperm(total_len, generator=g).tolist()[:subset_len]
        # 注意：这里 train_dataset 变成了 Subset 对象
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"!!! [Few-Shot] Using {args.data_fraction*100}% data: {subset_len} images !!!")

    train_sampler = None
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # [核心修复] DataLoader 中使用 train_collate_fn 而不是 train_dataset.collate_fn
    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                                        num_workers=nw, collate_fn=train_collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                        num_workers=nw, collate_fn=train_collate_fn)

    val_dataset = CocoDetection(args.data_path, "val", data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                  num_workers=nw, collate_fn=train_collate_fn)

    model = create_model(num_classes=args.num_classes + 1, args=args)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        lr_scheduler.step()
        det_info, _ = utils.evaluate(model, val_data_loader, device=device)

        # 写入结果
        with open(det_results_file, "a", encoding="utf-8") as f:
            result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            f.write("epoch:{} {}\n".format(epoch, '  '.join(result_info)))
            print("epoch:{} {}\n".format(epoch, '  '.join(result_info)))

        # 保存权重
        save_files = {'model': model.state_dict(), 'epoch': epoch}
        torch.save(save_files, os.path.join(args.output_dir, f"model_{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--data-path', default='./coco2017', help='dataset')
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    parser.add_argument('--output-dir', default='save_weights_robust', help='path where to save')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--lr', default=0.004, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--lr-steps', default=[10, 15], nargs='+', type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--amp", default=True, type=bool)
    
    # 核心新参数
    parser.add_argument('--backbone', default='legnet', help='legnet, resnet50, resnet18')
    parser.add_argument('--ablation-mode', default='full', help='full, no_scharr, no_lfea...')
    parser.add_argument('--data-fraction', default=1.0, type=float, help='data usage fraction (e.g. 0.1 for 10%)')

    args = parser.parse_args()
    main(args)
#nohup env CUDA_VISIBLE_DEVICES=2 python train_robustness.py --backbone resnet18 --ablation-mode full --data-fraction 0.1 --epochs 20 --batch_size 4 --output-dir output_gpu7_legnet_10percent > run_log-resnet18.txt 2>&1 &