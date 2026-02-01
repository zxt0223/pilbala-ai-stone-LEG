import shutil
import os

# 定义源目录和目标目录
source_dirs = [
    '/group/chenjinming/wyy/pytorch-pilipala-stone/stone1/all-stone',
    '/group/chenjinming/wyy/pytorch-pilipala-stone/stone2/qie-fen-stone'
]
target_dir = '/group/chenjinming/wyy/pytorch-pilipala-stone/labelme_images'

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 移动文件
for source_dir in source_dirs:
    if os.path.exists(source_dir):
        # 遍历源目录中的所有文件
        for filename in os.listdir(source_dir):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            
            # 确保是文件而不是目录
            if os.path.isfile(source_file):
                # 如果目标文件已存在，可以选择重命名或跳过
                if os.path.exists(target_file):
                    # 方法1：重命名（添加后缀）
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(target_file):
                        new_filename = f"{base}_{counter}{ext}"
                        target_file = os.path.join(target_dir, new_filename)
                        counter += 1
                
                # 移动文件
                shutil.move(source_file, target_file)
                print(f"移动: {source_file} -> {target_file}")
    else:
        print(f"警告: 源目录不存在: {source_dir}")

print("文件移动完成！")