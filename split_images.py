import os
from PIL import Image

def split_each_image():
    # ==========================================
    # 路径配置
    # ==========================================
    input_dir = r"/group/chenjinming/wyy/pytorch-pilipala-LEG/test_image"
    # 为了避免和原图混淆，输出到带 split 标识的新文件夹
    output_dir = r"/group/chenjinming/wyy/pytorch-pilipala-LEG/test_image_cropped_8parts"

    # 定义裁剪网格：2行 * 4列 = 8份
    # (如果是竖长图，可以改为 rows=4, cols=2)
    rows = 2
    cols = 4

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_dir):
        print(f"[错误] 找不到输入文件夹: {input_dir}")
        return

    # 获取所有图片
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
    print(f"[*] 找到 {len(img_files)} 张图片，开始逐张拆分...")

    for file_name in img_files:
        img_path = os.path.join(input_dir, file_name)
        base_name, ext = os.path.splitext(file_name)
        
        try:
            img = Image.open(img_path)
            w, h = img.size
            
            # 计算每一份的基础宽度和高度
            part_w = w // cols
            part_h = h // rows
            
            count = 1
            for i in range(rows):
                for j in range(cols):
                    # 计算裁剪框的 [左, 上, 右, 下] 坐标
                    left = j * part_w
                    upper = i * part_h
                    
                    # 关键：如果是最后一列或最后一行，右/下边界直接取到图片边缘 w/h
                    # 这样可以防止图片尺寸除以4除不尽时，漏掉最边缘的几个像素
                    right = w if j == cols - 1 else (j + 1) * part_w
                    lower = h if i == rows - 1 else (i + 1) * part_h
                    
                    # 裁剪并保存
                    crop_img = img.crop((left, upper, right, lower))
                    save_name = f"{base_name}_part{count}{ext}"
                    save_path = os.path.join(output_dir, save_name)
                    
                    crop_img.save(save_path)
                    count += 1
                    
            print(f"  -> {file_name} 成功拆分为 8 份")
            
        except Exception as e:
            print(f"  [错误] 处理 {file_name} 时发生异常: {e}")

    print(f"\n[*] 所有图片拆分完成！结果保存在: {output_dir}")

if __name__ == '__main__':
    split_each_image()