import os
import cv2
# 设置母文件夹路径
parent_folder = r"E:\KIRI\ME_1015"

# 获取母文件夹中的所有子文件夹
subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

# 遍历每个子文件夹
for subfolder in subfolders:
    print(subfolder)
    images_folder = os.path.join(subfolder, "images_meta")

    # 获取images文件夹中的所有图片文件
    image_files = [f.path for f in os.scandir(images_folder) if f.is_file() and (f.name.endswith('.JPEG'))]

    # 遍历每个图片文件
    for i, image_file in enumerate(image_files):
        image = cv2.imread(image_file)
        if image is not None:
            # 将图片转换为PNG格式
            png_file = os.path.join(images_folder, f"{i:04d}.png")
            cv2.imwrite(png_file, image)

            # 可选：如果要删除原始图片文件，可以添加以下行：
            # os.remove(image_file)
    break
print("转换完成")