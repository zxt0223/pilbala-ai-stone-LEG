import os
import shutil

# ===================== 配置 =====================
root_dir = "/group/chenjinming/wyy/pytorch-pilipala-LEG/zxt_checkpoints"  # 根目录
output_dir = "/group/chenjinming/wyy/pytorch-pilipala-LEG/target_pths"     # 输出目录
list_file = os.path.join(output_dir, "found_pths_list.txt")                 # 清单文件路径

# 映射表：(子文件夹名, pth文件名, 备注)
model_pth_map = [
    # LegNetFull
    ("legnet_full_run2", "model_21.pth", "LegNetFull_Run2_Epoch21"),
    ("legnet_full_run3", "model_20.pth", "LegNetFull_Run3_Epoch20"),
    ("legnet_full_run1", "model_16.pth", "LegNetFull_Run1_Epoch16"),
    # LegNetBaseline
    ("legnet_baseline_run2", "model_23.pth", "LegNetBaseline_Run2_Epoch23"),
    ("legnet_baseline_run1", "model_21.pth", "LegNetBaseline_Run1_Epoch21"),
    ("legnet_baseline_run3", "model_22.pth", "LegNetBaseline_Run3_Epoch22"),
    # LegNetNoGaussian
    ("legnet_no_gaussian_run2", "model_23.pth", "LegNetNoGaussian_Run2_Epoch23"),
    ("legnet_no_gaussian_run1", "model_18.pth", "LegNetNoGaussian_Run1_Epoch18"),
    ("legnet_no_gaussian_run3", "model_21.pth", "LegNetNoGaussian_Run3_Epoch21"),
    # LegNetNoLFEA
    ("legnet_no_lfea_run2", "model_22.pth", "LegNetNoLFEA_Run2_Epoch22"),
    ("legnet_no_lfea_run1", "model_22.pth", "LegNetNoLFEA_Run1_Epoch22"),
    ("legnet_no_lfea_run3", "model_19.pth", "LegNetNoLFEA_Run3_Epoch19"),
    # LegNetNoLog
    ("legnet_no_log_run3", "model_20.pth", "LegNetNoLog_Run3_Epoch20"),
    ("legnet_no_log_run2", "model_20.pth", "LegNetNoLog_Run2_Epoch20"),
    ("legnet_no_log_run1", "model_19.pth", "LegNetNoLog_Run1_Epoch19"),
    # LegNetNoScharr
    ("legnet_no_scharr_run1", "model_21.pth", "LegNetNoScharr_Run1_Epoch21"),
    ("legnet_no_scharr_run2", "model_21.pth", "LegNetNoScharr_Run2_Epoch21"),
    ("legnet_no_scharr_run3", "model_21.pth", "LegNetNoScharr_Run3_Epoch21"),
    # ResNet18
    ("resnet18_run3", "model_22.pth", "ResNet18_Run3_Epoch22"),
    ("resnet18_run1", "model_22.pth", "ResNet18_Run1_Epoch22"),
    ("resnet18_run2", "model_15.pth", "ResNet18_Run2_Epoch15"),
    # ResNet50
    ("resnet50_run1", "model_12.pth", "ResNet50_Run1_Epoch12"),
    ("resnet50_run3", "model_15.pth", "ResNet50_Run3_Epoch15"),
    ("resnet50_run2", "model_15.pth", "ResNet50_Run2_Epoch15"),
]
# ===================================================

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

found_list = []  # 存储找到的文件信息
missing_list = []

# 查找并复制
for sub_folder, pth_name, remark in model_pth_map:
    src_pth = os.path.join(root_dir, sub_folder, pth_name)
    dst_pth = os.path.join(output_dir, f"{remark}.pth")

    if os.path.exists(src_pth):
        shutil.copy2(src_pth, dst_pth)
        found_list.append(f"{remark}\t{src_pth}\t{dst_pth}")
        print(f"✅ 找到并复制：{src_pth} → {dst_pth}")
    else:
        missing_list.append(f"{remark}\t{src_pth}")
        print(f"❌ 未找到：{src_pth}")

# 写入清单文件
with open(list_file, "w", encoding="utf-8") as f:
    f.write("模型信息\t原始路径\t复制后路径\n")
    f.write("="*80 + "\n")
    for line in found_list:
        f.write(line + "\n")
    if missing_list:
        f.write("\n\n未找到的文件：\n")
        f.write("="*80 + "\n")
        for line in missing_list:
            f.write(line + "\n")

print("\n" + "="*60)
print(f"✅ 已生成清单文件：{list_file}")
print(f"✅ 找到 {len(found_list)} 个文件，未找到 {len(missing_list)} 个文件")