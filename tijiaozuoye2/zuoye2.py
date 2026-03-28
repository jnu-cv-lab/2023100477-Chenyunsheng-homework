import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util
import math
import os

# 获取当前 .py 文件所在的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前 .py 文件所在的文件夹（最关键！）
SAVE_FOLDER = os.path.dirname(current_file_path)

moon=data.moon()
camera=data.camera()

noisy_camera = util.random_noise(camera, mode='gaussian', var=0.01)

def image(img):
    img = (img - img.min()) * (255.0 / (img.max() - img.min()))
    return img.astype(np.uint8)

img_moon = image(moon)
img_camera = image(camera)
img_noisy = image(noisy_camera)

def prepare_image(img):
    img = (img - img.min()) * (255.0 / (img.max() - img.min()))
    return img.astype(np.uint8)

img_moon = prepare_image(moon)
img_camera = prepare_image(camera)
img_noisy = prepare_image(noisy_camera)

#1. 自行实现的模块：直方图均衡化 
def equ(image):
    L = 256
    # 1. 计算直方图
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    total_pixels = image.size
    # 2. 计算累积分布
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    # 3. 计算映射
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * (L - 1) / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    # 4. 映射图像
    equ_img = cdf[image]
    return equ_img

#2. 定量评价指标 (满足作业要求)
def cal(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr, mse

#3. 图像处理主流程
def process_and_display(img, title_prefix):
    #   步骤1：直方图均衡化对比  
    img_manual = equ(img)
    img_cv_eq = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    #步骤2：滤波处理 (去噪/平滑)  
    # 均值滤波
    img_mean = cv2.blur(img, (5, 5))
    #高斯滤波
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    #中值滤波 (对椒盐噪声效果好，对高斯噪声也有一定效果)
    img_median = cv2.medianBlur(img, 5)

    #步骤3：锐化 (使用拉普拉斯算子)  
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    img_sharpen = cv2.convertScaleAbs(img - laplacian)

    #步骤4：组合处理 (滤波 -> 均衡)  
    #先高斯滤波去噪，再进行直方图均衡化
    img_filtered_then_eq = cv2.equalizeHist(img_gaussian)

    #步骤5：定量评价 (以原图和手动均衡化为例)  
    psnr_manual, mse_manual = cal(img, img_manual)
    psnr_clahe, mse_clahe = cal(img, img_clahe)

    #步骤6：可视化结果  
    plt.figure(figsize=(16, 12))

    #第一行：原图与直方图均衡化对比
    plt.subplot(3, 3, 1), plt.imshow(img, cmap='gray'), plt.title(f'Original {title_prefix}')
    plt.subplot(3, 3, 2), plt.imshow(img_manual, cmap='gray'), plt.title(f'Manual HE (PSNR: {psnr_manual:.2f})')
    plt.subplot(3, 3, 3), plt.imshow(img_cv_eq, cmap='gray'), plt.title('OpenCV Equalize')
    plt.subplot(3, 3, 4), plt.imshow(img_clahe, cmap='gray'), plt.title(f'CLAHE (PSNR: {psnr_clahe:.2f})')

    #第二行：滤波与锐化
    plt.subplot(3, 3, 5), plt.imshow(img_median, cmap='gray'), plt.title('Median Filter')
    plt.subplot(3, 3, 6), plt.imshow(img_gaussian, cmap='gray'), plt.title('Gaussian Filter')
    plt.subplot(3, 3, 7), plt.imshow(img_sharpen, cmap='gray'), plt.title('Sharpening')

    #第三行：组合处理
    plt.subplot(3, 3, 8), plt.imshow(img_filtered_then_eq, cmap='gray'), plt.title('Gaussian -> Equalize')

    #直方图展示
    plt.subplot(3, 3, 9)
    plt.hist(img.flatten(), 256, [0, 256], color='gray', alpha=0.5, label='Original')
    plt.hist(img_manual.flatten(), 256, [0, 256], color='red', alpha=0.5, label='Manual')
    plt.legend(loc='upper left'), plt.title('Histogram Comparison')

    plt.suptitle(f'Image Processing Results: {title_prefix}', fontsize=16)
    plt.tight_layout()
    
    filename = f"{title_prefix}_result.png"
    full_path = os.path.join(SAVE_FOLDER, filename)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    
    # 打印出来让你看到存在哪了
    print("✅ 图片已保存到：")
    print(full_path)

    plt.show()
    plt.close()  # 防止内存堆积

#4. 执行处理
print("Processing Moon Image (Low Contrast)...")
process_and_display(img_moon, "Moon")

print("Processing Noisy Image...")
process_and_display(img_noisy, "Noisy camera")