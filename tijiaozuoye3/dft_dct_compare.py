import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# ===================== 自动获取当前文件所在目录 =====================
# 这行代码会自动找到当前.py文件所在的文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
# 图片保存路径（和代码文件在同一目录）
save_path = os.path.join(current_dir, "result.png")

# 1. 生成一维测试信号
N = 8
n = np.arange(N)
f = 50 + 20*np.sin(2*np.pi*1*n/N) + 10*np.cos(2*np.pi*2*n/N)
f = f.astype(np.float32)

# 2. 计算 DFT
F = np.fft.fft(f)

# 3. 计算 DCT
F_dct = cv2.dct(f.reshape(N, 1))

# 4. 画图
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.stem(np.abs(F))
plt.title("DFT 频谱")

plt.subplot(2,1,2)
plt.stem(F_dct)
plt.title("DCT 频谱")

plt.tight_layout()

# ===================== 自动保存图片到代码同级目录 =====================
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✅ 图片已保存到：{save_path}")

# 显示图片
plt.show()

# 5. 输出结果
print("\n原始信号：")
print(f)

print("\nDFT 结果：")
print(F)

print("\nDCT 结果：")
print(F_dct)