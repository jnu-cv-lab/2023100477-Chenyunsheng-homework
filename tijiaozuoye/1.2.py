import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 【核心修改】统一获取脚本所在目录 (zuoye 文件夹)
# ==========================================
# 获取 1.py 文件的绝对路径所在的文件夹
script_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 构建读取路径 (确保读到的是 zuoye 里的 1.jpg)
img_path = os.path.join(script_dir, "1.jpg")

# 2. 构建保存路径 (确保存到 zuoye 里，不会跑到上一级)
gray_save_path = os.path.join(script_dir, "gray_image.jpg")
crop_save_path = os.path.join(script_dir, "crop_image.jpg")

# ==========================================
# 任务1：读取图片
# ==========================================
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"❌ 找不到图片！请确认 '1.jpg' 在文件夹：{script_dir} 中")

# 任务2：输出图像信息
height, width, channels = img.shape
dtype = img.dtype
print(f"图像尺寸：宽度={width}，高度={height}")
print(f"通道数：{channels}")
print(f"数据类型：{dtype}")

# 任务3：显示原图
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# 任务4：灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8, 6))
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()

# 任务5：保存灰度图 (使用绝对路径保存)
cv2.imwrite(gray_save_path, gray)
print(f"✅ 灰度图已保存至: {gray_save_path}")

# 任务6：裁剪左上角 (使用绝对路径保存)
crop = img[0:100, 0:100]
cv2.imwrite(crop_save_path, crop)
print(f"✅ 裁剪图已保存至: {crop_save_path}")