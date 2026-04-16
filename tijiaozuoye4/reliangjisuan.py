import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== 1. 工具函数（最终课件匹配版） =====================
def compute_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """计算图像梯度幅值（Sobel算子，归一化尺度）"""
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, scale=1/8)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, scale=1/8)
    return np.sqrt(sobel_x**2 + sobel_y**2)

def fft_rms_freq(block: np.ndarray) -> float:
   
    h, w = block.shape
    fft = np.fft.fft2(block)
    fft_shift = np.fft.fftshift(fft)
    power_spectrum = np.abs(fft_shift)**2
    # 计算归一化频率坐标
    u = np.fft.fftfreq(w, d=1.0)
    v = np.fft.fftfreq(h, d=1.0)
    u_shift = np.fft.fftshift(u)
    v_shift = np.fft.fftshift(v)
    uu, vv = np.meshgrid(u_shift, v_shift)
    freq_radial = np.sqrt(uu**2 + vv**2)
    # 计算均方根频率（二阶矩）
    total_power = np.sum(power_spectrum)
    if total_power < 1e-6:
        return 0.0
    return np.sqrt(np.sum(power_spectrum * freq_radial**2) / total_power)

def gradient_rms_freq(block: np.ndarray) -> float:
    
    grad_mag = compute_gradient_magnitude(block)
    grad_sq = grad_mag**2
    E_grad_sq = np.mean(grad_sq)
    var_I = np.var(block)
    if var_I < 1e-6:
        return 0.0
    # 严格按课件公式解f_rms
    f_rms = np.sqrt(E_grad_sq / (4 * np.pi**2 * var_I))
    return np.clip(f_rms, 0, 0.5)

def fft_95_energy_max_freq(block: np.ndarray) -> float:
   

    h, w = block.shape
    fft = np.fft.fft2(block)
    fft_shift = np.fft.fftshift(fft)
    power_spectrum = np.abs(fft_shift)**2
    u = np.fft.fftfreq(w, d=1.0)
    v = np.fft.fftfreq(h, d=1.0)
    u_shift = np.fft.fftshift(u)
    v_shift = np.fft.fftshift(v)
    uu, vv = np.meshgrid(u_shift, v_shift)
    freq_radial = np.sqrt(uu**2 + vv**2)
    # 按频率从小到大累加能量，找到95%能量对应的最高频率
    power_flat = power_spectrum.flatten()
    freq_flat = freq_radial.flatten()
    sort_idx = np.argsort(freq_flat)
    power_sorted = power_flat[sort_idx]
    freq_sorted = freq_flat[sort_idx]
    total_energy = np.sum(power_sorted)
    if total_energy < 1e-6:
        return 0.0
    cumulative_energy = np.cumsum(power_sorted)
    threshold = 0.95 * total_energy
    idx_95 = np.argmax(cumulative_energy >= threshold)
    return freq_sorted[idx_95]

# ===================== 2. 主流程（课件匹配版） =====================
def main(block_size: int = 32):

    # 自动获取同级目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "1.jpg")
    save_path = os.path.join(current_dir, "result_course_match.png")

    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"未找到1.png，请放在代码同级目录！")
    H, W = img.shape
    print(f"图像尺寸：{H}x{W}，分块大小：{block_size}x{block_size}")

    # 分块补零
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    H_pad, W_pad = img_pad.shape
    num_blocks_h = H_pad // block_size
    num_blocks_w = W_pad // block_size
    print(f"补零后尺寸：{H_pad}x{W_pad}，分块数量：{num_blocks_h}x{num_blocks_w}")

    # 遍历所有块
    fft_rms_freqs = []
    grad_rms_freqs = []
    fft_max_freqs = []
    positions = []

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            y1, y2 = i*block_size, (i+1)*block_size
            x1, x2 = j*block_size, (j+1)*block_size
            block = img_pad[y1:y2, x1:x2]
            # 计算FFT均方根频率（课件核心）+ 95%能量最高频率（作业要求）
            f_rms_fft = fft_rms_freq(block)
            f_max_fft = fft_95_energy_max_freq(block)
            # 计算梯度法均方根频率（课件核心）
            f_rms_grad = gradient_rms_freq(block)
            # 保存结果
            fft_rms_freqs.append(f_rms_fft)
            grad_rms_freqs.append(f_rms_grad)
            fft_max_freqs.append(f_max_fft)
            positions.append((i, j))

    # 转numpy数组
    fft_rms_freqs = np.array(fft_rms_freqs)
    grad_rms_freqs = np.array(grad_rms_freqs)
    fft_max_freqs = np.array(fft_max_freqs)

    # ===================== 一致性分析（课件匹配版） =====================
    # 1. 核心：课件要求的「均方根频率」对比（两种方法的理论对应项）
    corr_rms = np.corrcoef(fft_rms_freqs, grad_rms_freqs)[0, 1]
    mae_rms = np.mean(np.abs(fft_rms_freqs - grad_rms_freqs))
    rmse_rms = np.sqrt(np.mean((fft_rms_freqs - grad_rms_freqs)**2))
    # 2. 作业要求：「95%能量最高频率」vs 梯度法近似（f_max ≈ 2*f_rms）
    grad_max_approx = np.clip(2 * grad_rms_freqs, 0, 0.5)
    corr_max = np.corrcoef(fft_max_freqs, grad_max_approx)[0, 1]
    mae_max = np.mean(np.abs(fft_max_freqs - grad_max_approx))
    rmse_max = np.sqrt(np.mean((fft_max_freqs - grad_max_approx)**2))

    print("\n===== 课件匹配版计算结果 =====")
    print("【核心：课件要求的均方根频率对比（理论对应项）】")
    print(f"相关系数：{corr_rms:.4f}")
    print(f"平均绝对误差：{mae_rms:.6f}")
    print(f"均方根误差：{rmse_rms:.6f}")
    print(f"FFT f_rms均值：{np.mean(fft_rms_freqs):.6f}")
    print(f"梯度f_rms均值：{np.mean(grad_rms_freqs):.6f}")
    print("\n【作业要求：95%能量最高频率 vs 梯度近似】")
    print(f"相关系数：{corr_max:.4f}")
    print(f"平均绝对误差：{mae_max:.6f}")
    print(f"均方根误差：{rmse_max:.6f}")

    # ===================== 可视化（课件匹配版） =====================
    # 构建频率热力图
    fft_rms_map = np.zeros((num_blocks_h, num_blocks_w))
    grad_rms_map = np.zeros((num_blocks_h, num_blocks_w))
    fft_max_map = np.zeros((num_blocks_h, num_blocks_w))
    for idx, (i, j) in enumerate(positions):
        fft_rms_map[i, j] = fft_rms_freqs[idx]
        grad_rms_map[i, j] = grad_rms_freqs[idx]
        fft_max_map[i, j] = fft_max_freqs[idx]

    plt.figure(figsize=(18, 12))

    # 1. 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image", fontsize=12)
    plt.axis("off")

    # 2. FFT均方根频率（课件核心）
    plt.subplot(2, 3, 2)
    im1 = plt.imshow(fft_rms_map, cmap="jet", vmin=0, vmax=0.2)
    plt.colorbar(im1, shrink=0.8)
    plt.title("FFT RMS Frequency (Course Core)", fontsize=12)
    plt.axis("off")

    # 3. 梯度法均方根频率（课件核心）
    plt.subplot(2, 3, 3)
    im2 = plt.imshow(grad_rms_map, cmap="jet", vmin=0, vmax=0.2)
    plt.colorbar(im2, shrink=0.8)
    plt.title("Gradient RMS Frequency (Course Core)", fontsize=12)
    plt.axis("off")

    # 4. 均方根频率散点图（课件核心对比）
    plt.subplot(2, 3, 4)
    plt.scatter(fft_rms_freqs, grad_rms_freqs, s=8, alpha=0.7)
    max_val = max(fft_rms_freqs.max(), grad_rms_freqs.max())
    plt.plot([0, max_val], [0, max_val], "r--", label="y=x")
    plt.xlabel("FFT RMS Frequency", fontsize=10)
    plt.ylabel("Gradient RMS Frequency", fontsize=10)
    plt.title(f"RMS Freq Corr = {corr_rms:.4f}", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    # 5. 95%能量最高频率散点图（作业要求）
    plt.subplot(2, 3, 5)
    plt.scatter(fft_max_freqs, grad_max_approx, s=8, alpha=0.7)
    max_val_max = max(fft_max_freqs.max(), grad_max_approx.max())
    plt.plot([0, max_val_max], [0, max_val_max], "r--", label="y=x")
    plt.xlabel("FFT 95% Max Frequency", fontsize=10)
    plt.ylabel("Gradient Approx Max Frequency", fontsize=10)
    plt.title(f"Max Freq Corr = {corr_max:.4f}", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    # 6. 均方根频率误差热力图
    plt.subplot(2, 3, 6)
    error_map = fft_rms_map - grad_rms_map
    im3 = plt.imshow(error_map, cmap="coolwarm", vmin=-0.05, vmax=0.05)
    plt.colorbar(im3, shrink=0.8)
    plt.title("RMS Frequency Error Map (FFT - Grad)", fontsize=12)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n课件匹配版结果已保存到：{save_path}")

# ===================== 运行 =====================
if __name__ == "__main__":
    main(block_size=32)