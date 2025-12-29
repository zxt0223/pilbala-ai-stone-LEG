import time
import os
import datetime
import argparse

import torch
os.environ["NCCL_P2P_DISABLE"] = "1"

import transforms
from my_dataset_coco import CocoDetection
from backbone import resnet50_fpn_backbone, resnet18_fpn_backbone, legnet_fpn_backbone
from network_files import MaskRCNN
import train_utils.train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups, init_distributed_mode, save_on_master, mkdir

def create_model(num_classes, load_pretrain_weights=True, backbone_name="legnet", ablation_mode="full"):
    if backbone_name == "resnet50":
        # 如果需要官方预训练，请修改这里的路径或设为 None
        backbone = resnet50_fpn_backbone(pretrain_path="", trainable_layers=3)
        print("Using Backbone: ResNet50")
    elif backbone_name == "resnet18":
        # 如果需要官方预训练，请修改这里的路径
        backbone = resnet18_fpn_backbone(pretrain_path="", trainable_layers=3)
        print("Using Backbone: ResNet18")
    elif backbone_name == "legnet":
        # LEGNet 默认不加载预训练(从头训练)，如果有需要请填路径
        backbone = legnet_fpn_backbone(pretrain_path="", ablation_mode=ablation_mode)
        print(f"Using Backbone: LEGNet (Ablation Mode: {ablation_mode})")
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    model = MaskRCNN(backbone, num_classes=num_classes, min_size=1000, max_size=1333)

    if load_pretrain_weights and backbone_name == "resnet50":
        # 只有 ResNet50 有现成的 MaskRCNN COCO 权重可加载
        if os.path.exists("maskrcnn_resnet50_fpn_coco.pth"):
            weights_dict = torch.load("maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
            for k in list(weights_dict.keys()):
                if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    return model

def main(args):
    init_distributed_mode(args)
    
    device = torch.device(args.device)
    
    # 确保输出目录存在
    if args.output_dir:
        mkdir(args.output_dir)

    # 结果文件加上前缀 zxt 以区分
    det_results_file = os.path.join(args.output_dir, "zxt_det_results.txt")
    seg_results_file = os.path.join(args.output_dir, "zxt_seg_results.txt")

    print("Loading data")
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

    COCO_root = args.data_path
    train_dataset = CocoDetection(COCO_root, "train", data_transform["train"])
    val_dataset = CocoDetection(COCO_root, "val", data_transform["val"])

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    print("Creating model")
    model = create_model(num_classes=args.num_classes + 1, 
                         load_pretrain_weights=args.pretrain,
                         backbone_name=args.backbone,
                         ablation_mode=args.ablation_mode)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        utils.evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = utils.train_one_epoch(model, optimizer, data_loader,
                                              device, epoch, args.print_freq,
                                              warmup=True, scaler=scaler)
        lr_scheduler.step()

        det_info, seg_info = utils.evaluate(model, data_loader_test, device=device)

        if args.rank in [-1, 0]:
            with open(det_results_file, "a") as f:
                result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                f.write("epoch:{} {}\n".format(epoch, '  '.join(result_info)))

            if seg_info is not None:
                with open(seg_results_file, "a") as f:
                    result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                    f.write("epoch:{} {}\n".format(epoch, '  '.join(result_info)))

        if args.output_dir:
            save_files = {'model': model_without_ddp.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'args': args,
                          'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            save_on_master(save_files, os.path.join(args.output_dir, f'model_{epoch}.pth'))

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', default='/group/chenjinming/wyy/pytorch-pilipala-LEG/coco2017', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=50, type=int, metavar='N')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay')
    parser.add_argument('--lr-step_size', default=8, type=int)
    parser.add_argument('--lr-steps', default=[35, 45], nargs='+', type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--print-freq', default=50, type=int)
    parser.add_argument('--output-dir', default='multi_train_weights', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--test-only', action="store_true")
    parser.add_argument('--world-size', default=4, type=int)
    parser.add_argument('--dist-url', default='env://')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)
    parser.add_argument("--pretrain", type=bool, default=False)
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp")
    
    # 核心参数
    parser.add_argument('--backbone', default='legnet', help='backbone: resnet50, resnet18, legnet')
    parser.add_argument('--ablation-mode', default='full', help='Ablation mode')

    args = parser.parse_args()

    if args.output_dir:
        mkdir(args.output_dir)

    main(args)