import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------
# 关键：获取当前 .py 文件所在的目录
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------
# 1. 创建测试图（矩形、圆、平行线、垂直线）
# --------------------------
def create_test_image(size=500):
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    # 矩形
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 0), 2)
    # 圆
    cv2.circle(img, (400, 400), 50, (0, 0, 0), 2)
    # 平行线
    for i in range(50, 500, 50):
        cv2.line(img, (50, i), (450, i), (0, 0, 0), 1)
    # 垂直线（和水平线垂直）
    for i in range(50, 500, 50):
        cv2.line(img, (i, 50), (i, 450), (0, 0, 0), 1)
    return img

# --------------------------
# 2. 相似变换（缩放+旋转+平移）
# --------------------------
def similarity_transform(img):
    rows, cols = img.shape[:2]
    # 构造相似变换矩阵（缩放+旋转+平移）
    angle = 30  # 旋转角度
    scale = 0.8 # 缩放比例
    center = (cols//2, rows//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 平移
    M[0, 2] += 20
    M[1, 2] += 20
    res = cv2.warpAffine(img, M, (cols, rows))
    return res, M

# --------------------------
# 3. 仿射变换（含剪切，保持平行）
# --------------------------
def affine_transform(img):
    rows, cols = img.shape[:2]
    # 三个点确定仿射变换
    pts1 = np.float32([[50,50], [450,50], [50,450]])
    pts2 = np.float32([[100,100], [400,150], [150,400]])
    M = cv2.getAffineTransform(pts1, pts2)
    res = cv2.warpAffine(img, M, (cols, rows))
    return res, M

# --------------------------
# 4. 透视变换（不保持平行）
# --------------------------
def perspective_transform(img):
    rows, cols = img.shape[:2]
    # 四个点确定透视变换
    pts1 = np.float32([[50,50], [450,50], [50,450], [450,450]])
    pts2 = np.float32([[100,80], [400,120], [80,420], [420,380]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    res = cv2.warpPerspective(img, M, (cols, rows))
    return res, M

# --------------------------
# 5. 透视畸变校正（A4纸照片校正）
# --------------------------
def correct_perspective(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图片，请检查路径")
        return None
    rows, cols = img.shape[:2]
    # 手动选取四个角点
    pts_src = np.float32([
        [50, 50],      # 左上
        [cols-50, 80], # 右上
        [30, rows-30], # 左下
        [cols-30, rows-50] # 右下
    ])
    # 目标点（校正后的A4纸矩形）
    pts_dst = np.float32([
        [0, 0],
        [cols, 0],
        [0, rows],
        [cols, rows]
    ])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    corrected = cv2.warpPerspective(img, M, (cols, rows))
    return corrected

# --------------------------
# 主程序运行
# --------------------------
if __name__ == "__main__":
    # 创建测试图
    test_img = create_test_image()

    # 应用三种变换
    sim_img, M_sim = similarity_transform(test_img)
    aff_img, M_aff = affine_transform(test_img)
    per_img, M_per = perspective_transform(test_img)

    # ======================
    # 保存到 .py 文件所在目录
    # ======================
    cv2.imwrite(os.path.join(BASE_DIR, "original.png"), test_img)
    cv2.imwrite(os.path.join(BASE_DIR, "similarity.png"), sim_img)
    cv2.imwrite(os.path.join(BASE_DIR, "affine.png"), aff_img)
    cv2.imwrite(os.path.join(BASE_DIR, "perspective.png"), per_img)

    # 显示结果
    plt.figure(figsize=(12, 10))
    plt.subplot(221), plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(222), plt.imshow(cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB)), plt.title('Similarity Transform')
    plt.subplot(223), plt.imshow(cv2.cvtColor(aff_img, cv2.COLOR_BGR2RGB)), plt.title('Affine Transform')
    plt.subplot(224), plt.imshow(cv2.cvtColor(per_img, cv2.COLOR_BGR2RGB)), plt.title('Perspective Transform')
    plt.show()

    # 打印变换矩阵
    print("相似变换矩阵:\n", M_sim)
    print("仿射变换矩阵:\n", M_aff)
    print("透视变换矩阵:\n", M_per)