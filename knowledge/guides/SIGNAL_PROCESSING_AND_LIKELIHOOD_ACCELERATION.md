# 信号处理与引力波似然计算加速指南

<!-- 2026-09-03 CST: Guide covering frequency-domain methods, NUFFT, and fast likelihood algorithms -->

## 1. 引力波信号处理与匹配滤波基础

### 1.1 时域与频域数据分析的权衡
* **时域生成（Time Domain）**：
  * 直观反映引力波动力学演化过程；
  * 但对于长期 EMRI（例如持续 1 年、采样间隔 $\Delta t = 15\text{ s}$），时域包含 **$N = 2,103,877$ 个采样点**。
  * 时域卷积和滤波计算量巨大（$\mathcal{O}(N^2)$ 或 FFT 下 $\mathcal{O}(N \log N)$）。
* **频域生成（Frequency Domain）**：
  * 引力波探测器的噪声通常在频域是平稳不相关的（协方差矩阵为对角阵）；
  * 匹配滤波与似然函数天然在频域以标量内积形式表述：
    $$\langle d | h \rangle = 4 \, \text{Re} \int_0^\infty \frac{\tilde{d}(f) \, \tilde{h}^*(f)}{S_n(f)} \, df$$

---

## 2. 频域 EMRI 波形加速：多音分解与 Fast and Fourier

### 2.1 驻相近似（Stationary Phase Approximation, SPA）
对于缓慢啁啾的模式，每个谐波模式 $(m, k, n)$ 的瞬时频率随时间单调演化：
$$f_{mkn}(t) = \frac{1}{2\pi} \frac{d\Phi_{mkn}}{dt} = \frac{1}{2\pi} (m \Omega_\phi(t) + k \Omega_\theta(t) + n \Omega_r(t))$$
由驻相条件 $f = f_{mkn}(t_s)$，频域傅里叶变换积分的主导贡献集中在驻相时间 $t_s(f)$ 附近：
$$\tilde{h}_{mkn}(f) \approx A_{mkn}(t_s) \sqrt{\frac{2\pi}{|\ddot{\Phi}_{mkn}(t_s)|}} \exp\left( -i \left( 2\pi f t_s - \Phi_{mkn}(t_s) \pm \frac{\pi}{4} \right) \right)$$

### 2.2 多音分解（Multivoice Decomposition）
* EMRI 的时域信号是由上千个不同的独立单色或慢啁啾谐波（"Voices"）叠加而成的。
* **物理特征**：在给定的较短时间窗口内，每一个 voice 的频带极窄，但在一年尺度上，每个 voice 会在频域扫过一段带宽。
* **加速思想**：无需为全部模式分配全尺寸高采样率频域网格。各模式在其各自的非零频段内独立解析生成，然后再投影或插值求和，极大地减少内存和零填充计算开销。

---

## 3. 非均匀快速傅里叶变换（NUFFT 与 cuFINUFFT）

在模式求和或时频转换中，采样点通常不是均匀分布在 Nyquist 网格上的（例如轨道步进是非等距的自适应步长）：
* **Type-1 NUFFT（非均匀到均匀）**：输入为非均匀时间点 $t_j$，输出为均匀频域网格 $f_k$。
* **Type-2 NUFFT（均匀到非均匀）**：输入为均匀频域系数，输出到非均匀时间或空间点。
* **Type-3 NUFFT（非均匀到非均匀）**：时频两端皆非均匀。

### 3.1 核心算法原理（以 Type-1 为例）
1. **卷积扩散（Spreading）**：利用截断的快速衰减紧支集卷积核（如平滑的指数平方根核 $\psi(x) = e^{-\beta \sqrt{1 - (2x/w)^2}}$ 或 Kaiser-Bessel 核），将非均匀样本点分配到邻近的几个过采样（Oversampled）均匀网格点上。
2. **标准 FFT 运算**：在规则过采样网格上执行标准 1D/2D FFT（利用 cuFFT）。
3. **频域解卷积（Deconvolution）**：在频域除以解析卷积核的连续傅里叶变换，消除扩散引起的平滑失真。

### 3.2 cuFINUFFT 在 GPU 上的极致加速启示
* **分块与共享内存重排（Sub-problem binning）**：将非均匀点按空间局部性划分进小块（Bins），使每个 CUDA Block 可以将核函数值和局部网格载入 Shared Memory。
* **计算与内存合并**：避免全局显存的离散随机原子写操作（Atomic Add）。

---

## 4. 引力波参数估计与似然评估加速算法

在 LISA 科学分析中，马尔可夫链蒙特卡洛（MCMC）或嵌套采样（Nested Sampling）需要评估 **$10^6 \sim 10^8$ 次似然函数**。传统计算每次耗时几毫秒都会导致单次采样耗费数天。

对高斯白噪声，对数似然函数为：
$$\ln \mathcal{L}(\theta) = -\frac{1}{2} \langle d - h(\theta) | d - h(\theta) \rangle = \langle d | h(\theta) \rangle - \frac{1}{2} \langle h(\theta) | h(\theta) \rangle - \frac{1}{2} \langle d | d \rangle$$

### 4.1 外差似然（Heterodyned Likelihood / Relative Binning）
* **基本原理**：给定一个靠近后验峰值的参考波形 $h_0(f)$，在邻域参数 $\theta$ 下的目标波形 $h(f; \theta)$ 与 $h_0(f)$ 相比，其比值是一个极其平缓的慢变函数：
  $$r(f; \theta) = \frac{h(f; \theta)}{h_0(f)}$$
* **分箱汇总（Summary Statistics）**：
  将整个频域划分为数百个粗分箱（Frequency Bins $[f_{\min}^{(b)}, f_{\max}^{(b)}]$），在每个分箱内将慢变比值 $r(f)$ 展开为一阶泰勒级数：
  $$r(f; \theta) \approx r_0^{(b)}(\theta) + r_1^{(b)}(\theta) (f - f_c^{(b)})$$
* **离线-在线分解（Offline-Online Decomposition）**：
  包含真实数据 $d$ 和参考波形 $h_0$ 的稠密积分：
  $$A_0^{(b)} = 4 \int_{f_{\min}^{(b)}}^{f_{\max}^{(b)}} \frac{\tilde{d}(f) \tilde{h}_0^*(f)}{S_n(f)} df, \quad A_1^{(b)} = 4 \int_{f_{\min}^{(b)}}^{f_{\max}^{(b)}} \frac{\tilde{d}(f) \tilde{h}_0^*(f)(f - f_c^{(b)})}{S_n(f)} df$$
  只需在 MCMC 开始前计算一次（离线）！
  在 MCMC 采样中（在线），似然评估退化为对数百个分箱的简单加权和：
  $$\langle d | h(\theta) \rangle \approx \sum_b \left( r_0^{(b)}(\theta) A_0^{(b)} + r_1^{(b)}(\theta) A_1^{(b)} \right)$$
  **速度提升可达 $10^2 \sim 10^4$ 倍！**

### 4.2 降阶求积（Reduced Order Quadrature, ROQ）
* 利用奇异值分解（SVD）或贪心算法（Greedy Algorithm）从波形流形中提取出低维正交基矢量 $\{e_i(f)\}_{i=1}^N$（$N \sim 100$）。
* 寻找经验插值节点（Empirical Interpolation Method, EIM），用少量特定的频点数值直接精确重构整个内积积分，完全免除全频段积分。

---

## 5. LISA 时间延迟干涉测量（TDI）与 GPU 仪器响应

引力波并非直接以 $h_+, h_\times$ 进入数据记录仪，而是通过 LISA 三个绕日旋转的干涉卫星手臂产生的相位差测量。
由于三个激光干涉手臂不等长（动态漂移达数千公里），激光频率噪声比引力波信号大数个数量级。必须使用 **时间延迟干涉测量（TDI, Time Delay Interferometry）**（如 $X, Y, Z$ 或正交信道 $A, E, T$）对激光相位差进行时延组合以完全对消激光相位抖动。

### 5.1 fastlisaresponse 架构借鉴
FEW 团队开发的 `fastlisaresponse`（已保存在本地参考工程中）采用了如下 GPU 协同方案：
1. **轨道位置与时延预计算**：将 LISA 星座的岁差、轨道运动在 CPU/GPU 上用高阶切比雪夫多项式或样条拟合。
2. **GPU 上的内联投影（In-situ Projection）**：时域波形生成后直接留在 GPU 显存（CuPy 数组），紧接着通过自定义 CUDA 内核直接计算手臂时间投影，避免显存到主机内存（D2H）的回传。
3. **频域 TDI 转换**：直接利用时延定理 $\mathcal{F}\{h(t - L/c)\} = \tilde{h}(f) e^{-2\pi i f L/c}$，在频域只需与标量传递函数相乘即可完成探测器响应，这与 FEW 的频域模型具有天然的亲和性。
