import os
import datetime
import time

import torch
# from torchvision.ops.misc import FrozenBatchNorm2d

import transforms
from my_dataset_coco import CocoDetection
from my_dataset_voc import VOCInstances

# [核心修改 1] 导入 LEGNet backbone
from backbone.legnet import legnet_fpn_backbone
from network_files import MaskRCNN
from train_utils import train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from torch.utils.tensorboard import SummaryWriter  # [新增] TensorBoard


def create_model(num_classes, load_pretrain_weights=True):
    # [核心修改 2] 使用 LEGNet 构建 backbone
    # 这里的 pretrain_path 可以指向 LWEGNet_tiny.pth，如果没有则留空
    backbone = legnet_fpn_backbone(pretrain_path="LWEGNet_tiny.pth")

    # [核心修改 3] 增加 min_size=1000 以提升小目标检测能力
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     min_size=1000, max_size=1333)

    if load_pretrain_weights:
        # LEGNet 结构不同，不加载基于 ResNet 的 COCO 预训练权重
        # 除非你有专门针对 LEGNet 的 COCO 预训练权重
        pass

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    det_results_file = os.path.join(args.output_dir, f"det_results{now}.txt")
    seg_results_file = os.path.join(args.output_dir, f"seg_results{now}.txt")

    # [新增] 初始化 TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # [核心修改 4] 数据增强同步 (保持与多卡版本一致)
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),  # [新增]
            transforms.RandomColorJitter(brightness=0.3, contrast=0.3, prob=0.5), # [新增]
            transforms.RandomGaussianBlur(prob=0.3) # [新增]
        ]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    data_root = args.data_path

    # load train data set
    train_dataset = CocoDetection(data_root, "train", data_transform["train"])
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    val_dataset = CocoDetection(data_root, "val", data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    # create model num_classes equal background + classes
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)
    model.to(device)

    train_loss = []
    learning_rate = []
    val_map = []

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_steps,
                                                        gamma=args.lr_gamma)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    print("Start training")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # [新增] TensorBoard 记录 Loss
        writer.add_scalar('Train/Loss', mean_loss.item(), epoch)
        writer.add_scalar('Train/Learning_Rate', lr, epoch)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        det_info, seg_info = utils.evaluate(model, val_data_loader, device=device)

        # [新增] TensorBoard 记录 mAP
        writer.add_scalar('Val/Det_mAP_0.5:0.95', det_info[0], epoch)
        writer.add_scalar('Val/Det_mAP_0.5', det_info[1], epoch)
        if seg_info is not None:
            writer.add_scalar('Val/Seg_mAP_0.5:0.95', seg_info[0], epoch)
            writer.add_scalar('Val/Seg_mAP_0.5', seg_info[1], epoch)

        # write detection into txt
        with open(det_results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        # write seg into txt
        if seg_info is not None:
            with open(seg_results_file, "a") as f:
                result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

        val_map.append(det_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, os.path.join(args.output_dir, f'model_{epoch}.pth'))

    writer.close()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--data-path', default='./coco2017', help='dataset')
    # [核心修改 5] 默认 num_classes 为 1 (石头)
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    # 单卡 batch_size 建议设为 4 或 8 (取决于显存)
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-steps', default=[35, 45], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--pretrain", type=bool, default=False, help="load COCO pretrain weights.")
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)