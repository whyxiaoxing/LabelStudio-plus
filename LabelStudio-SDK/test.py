"""nvmath-python 最小示例：FFT 谱方法求导 + 矩阵乘法求方差。

本机实测：nvmath-python 1.0.0 / numpy 2.2.6 / RTX 5060 Ti。
参数统一在下面的「参数区」改写，不走命令行传参。
"""

from __future__ import annotations

import numpy as np

import nvmath
import nvmath.fft
import nvmath.linalg

# ------------------------------- 参数区 -------------------------------------
N = 10240                         # 采样点数
X_MIN, X_MAX = 0.0, 10.0 * np.pi  # 求导区间
EXECUTION = 'cuda'               # 'cuda' 走 GPU；'cpu' 需另装 nvmath-python[cpu]（MKL/FFTW）
SEED = 0                         # 随机样本种子
# ----------------------------------------------------------------------------


def spectral_derivative(f: np.ndarray, dx: float, execution: str = EXECUTION) -> np.ndarray:
    """FFT 谱方法求一阶导数：d/dx f ≈ IFFT(i·k·FFT(f))。

    Args:
        f: 均匀网格上的一维实信号。
        dx: 网格间距。
        execution: 'cuda' 或 'cpu'。

    Returns:
        与 f 同长度的一阶导数。
    """
    n = f.size
    # 各模态波数 k = 2π·j / (n·dx)，j 为 fftfreq 给出的整数频率
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    f_hat = nvmath.fft.fft(f.astype(np.complex128), execution=execution)
    # nvmath 的 ifft 与 cuFFT 语义一致，不做 1/n 归一化，需手动除
    df_hat = (1j * k) * f_hat
    return (nvmath.fft.ifft(df_hat, execution=execution) / n).real


def mean_variance(x: np.ndarray, execution: str = EXECUTION) -> tuple[float, float]:
    """用 nvmath.linalg.matmul 求均值与总体方差：Var = E[x²] - E[x]²。

    Args:
        x: 一维样本。
        execution: 'cuda' 或 'cpu'。

    Returns:
        (均值, 总体方差)。
    """
    n = x.size
    cols = np.stack([x, x * x], axis=1)  # (n, 2)，两列为 [x, x²]
    ones = np.ones((1, n), dtype=x.dtype)
    sums = np.asarray(nvmath.linalg.matmul(ones, cols, execution=execution))  # (1, 2) 行和
    s1, s2 = float(sums[0, 0]), float(sums[0, 1])
    mean = s1 / n
    return mean, s2 / n - mean * mean


def main() -> None:
    print(f'nvmath {nvmath.__version__} / numpy {np.__version__} / execution={EXECUTION}')

    # 1) 求导：f(x) = sin(x)，解析导数 (sin x)' = cos x
    x = np.linspace(X_MIN, X_MAX, N, endpoint=False)
    f = np.sin(x)
    df = spectral_derivative(f, x[1] - x[0])
    print(f'[求导] N={N}  最大误差 |df - cos x| = {np.max(np.abs(df - np.cos(x))):.3e}')

    # 2) 方差：随机样本，与 numpy 参考值对比
    sample = np.random.default_rng(SEED).normal(loc=3.0, scale=2.0, size=N)
    mean, var = mean_variance(sample)
    print(f'[方差] mean = {mean:.6f}  (numpy {np.mean(sample):.6f})')
    print(f'       var  = {var:.6f}  (numpy {np.var(sample):.6f})')


if __name__ == '__main__':
    main()
