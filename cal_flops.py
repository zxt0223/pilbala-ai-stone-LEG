""""
运行上面的脚本，你会得到类似这样的数字（假设）：

LEGNet: FLOPs = 150G, Params = 35M

ResNet50: FLOPs = 200G, Params = 44M
"""

import torch
from thop import profile
from thop import clever_format
from backbone.legnet import legnet_fpn_backbone
from network_files import MaskRCNN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 创建你的模型 (LEGNet)
    # 注意：这里我们只测 Backbone 或者 整个 Mask R-CNN 都可以
    # 通常论文里对比的是整个检测模型的开销
    backbone = legnet_fpn_backbone(pretrain_path="")
    model = MaskRCNN(backbone, num_classes=2) # 1类石头+背景
    model.to(device)
    model.eval()

    # 3. 创建一个虚拟输入 (Input Size 一般用 800x800 或者 1024x1024)
    # 遥感/大图通常用 1024x1024
    input_tensor = torch.randn(1, 3, 1024, 1024).to(device)

    print("=== Calculating FLOPs and Params ===")
    
    # 4. 计算 FLOPs (Macs) 和 Params
    # custom_ops 可以处理一些 thop 无法识别的算子，通常不用改
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    
    # 5. 格式化输出 (例如: 1.5G, 20M)
    flops_fmt, params_fmt = clever_format([flops, params], "%.3f")
    
    print(f"Input Size: {input_tensor.shape}")
    print(f"FLOPs (计算量): {flops_fmt}")
    print(f"Params (参数量): {params_fmt}")
    
    # ----------------------------------------------------
    # 额外：对比 ResNet50 (方便你填表)
    # ----------------------------------------------------
    print("\n=== Baseline (ResNet50) Reference ===")
    from backbone import resnet50_fpn_backbone
    r50_backbone = resnet50_fpn_backbone(pretrain_path="", trainable_layers=3)
    r50_model = MaskRCNN(r50_backbone, num_classes=2)
    r50_model.to(device)
    r50_flops, r50_params = profile(r50_model, inputs=(input_tensor, ), verbose=False)
    r50_flops_fmt, r50_params_fmt = clever_format([r50_flops, r50_params], "%.3f")
    print(f"ResNet50 FLOPs: {r50_flops_fmt}")
    print(f"ResNet50 Params: {r50_params_fmt}")

if __name__ == "__main__":
    main()