import os
import shutil

# 定义原始文件夹路径
source_path = '/group/chenjinming/wyy/pytorch-pilipala-LEG/zxt_checkpoints'

# 定义目标文件夹路径（可自定义）
target_path = '/group/chenjinming/wyy/pytorch-pilipala-LEG/zxt_checkpoints_txt'

# 如果目标文件夹不存在，则创建
os.makedirs(target_path, exist_ok=True)

# 遍历原始文件夹下的所有子文件夹
for folder_name in os.listdir(source_path):
    folder_path = os.path.join(source_path, folder_name)
    # 确保当前路径是一个文件夹
    if os.path.isdir(folder_path):
        # 遍历子文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # 检查文件是否为 txt 文件
            if os.path.isfile(file_path) and file_name.lower().endswith('.txt'):
                # 构造新的文件名（子文件夹名.txt）
                new_file_name = f"{folder_name}.txt"
                new_file_path = os.path.join(target_path, new_file_name)
                # 复制文件到目标文件夹
                shutil.copy2(file_path, new_file_path)
                print(f"已复制文件: {file_path} -> {new_file_path}")

print("所有 txt 文件提取完成！")