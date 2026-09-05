# 经典课本核心理论深度精要与定理推导

<!-- 2026-09-03 CST: Deep theoretical synthesis of standard textbook theorems, derivations, and formalisms -->

本指南从本地归档的 18 部经典教材与专著中，深度凝练了 FEW 项目底层最核心的三大支柱理论体系，将数百页大部头课本的核心定理、公式推导与算法逻辑沉淀于此。

---

## 第一篇：广义相对论、Kerr 测地线与黑洞微扰论（基于 Carroll、Poisson、Barack & Pound）

### 1.1 Kerr 时空度规与第一积分
在 Boyer-Lindquist 坐标 $(t, r, \theta, \phi)$ 下，质量为 $M$、自旋参数为 $a = J/M$ 的 Kerr 黑洞度规线元为：
$$ds^2 = -\left(1 - \frac{2Mr}{\rho^2}\right) dt^2 - \frac{4Mar\sin^2\theta}{\rho^2} dt d\phi + \frac{\rho^2}{\Delta} dr^2 + \rho^2 d\theta^2 + \left(r^2 + a^2 + \frac{2Ma^2r\sin^2\theta}{\rho^2}\right) \sin^2\theta d\phi^2$$
其中：
$$\Delta(r) = r^2 - 2Mr + a^2, \quad \rho^2(r, \theta) = r^2 + a^2 \cos^2\theta$$

#### 四个守恒荷与测地线第一积分
质量为 $\mu$ 的粒子在弯曲时空中沿类时测地线运动，具有四个独立的守恒量（Carter 1968）：
1. **静止质量**：$g_{\mu\nu} u^\mu u^\nu = -1$；
2. **比能量（Energy per unit mass）**：$E = -u_t$（时间平移对称 Killing 矢量生成）；
3. **轴向比角动量（Angular momentum per unit mass）**：$L_z = u_\phi$（轴对称 Killing 矢量生成）；
4. **Carter 常数（Carter Constant）**：$Q$（由二阶 Killing-Yano 张量生成，表征非赤道面振荡自由度）。

引入 **Mino 时间（Mino Time）** $d\lambda = d\tau / \rho^2$，四维测地线方程实现完全解耦：
$$\left(\frac{dr}{d\lambda}\right)^2 = R(r), \quad \left(\frac{d\theta}{d\lambda}\right)^2 = \Theta(\theta)$$
$$\frac{dt}{d\lambda} = T_r(r) + T_\theta(\theta), \quad \frac{d\phi}{d\lambda} = \Phi_r(r) + \Phi_\theta(\theta)$$
其中 $R(r)$ 是四次多项式，其四个根决定了轨道的远地点 $r_{\max}$、近地点 $r_{\min}$ 以及视界分界线。

### 1.2 绝热辐射反作用与双时间尺度展开（Two-Timescale Expansion）
根据 Poisson 与 Barack & Pound 的微扰教材体系，EMRI 动力学严格遵循微扰展开：
$$x^\alpha(t) = X_0^\alpha(t, \tilde{t}) + \eta X_1^\alpha(t, \tilde{t}) + \mathcal{O}(\eta^2)$$
其中：
* **快时间尺度（Fast timescale）**：$t \sim M$，代表轨道周期（几分钟到几十分钟）；
* **慢时间尺度（Slow timescale）**：$\tilde{t} = \eta t \sim M/\eta$，代表绝热旋进寿命（几个月到数年）。

在**绝热极限（Leading Adiabatic Order）**下，轨道在每个快时间步上表现为完全的测地线，而守恒量沿慢时间演化：
$$\frac{dJ_i}{dt} = -\langle \mathcal{F}_i \rangle_{t} + \mathcal{O}(\eta^2)$$
其中 $J_i \in \{E, L_z, Q\}$，通量 $\mathcal{F}_i$ 是无穷远引力波与视界吸积通量的时间平均。FEW 通过预计算的高维样条网格精确给出了这一导数关系。

### 1.3 Teukolsky 标量场主方程
引力摄动由 Newman-Penrose 标量 $\psi_4 = -C_{\alpha\beta\gamma\delta} n^\alpha \bar{m}^\beta n^\gamma \bar{m}^\delta$ 描述。自旋加权为 $s = -2$ 的 Teukolsky 主方程为：
$$\left[\frac{(r^2+a^2)^2}{\Delta} - a^2\sin^2\theta\right] \frac{\partial^2 \psi_4}{\partial t^2} + \frac{4Mar}{\Delta} \frac{\partial^2 \psi_4}{\partial t \partial \phi} + \left[\frac{a^2}{\Delta} - \frac{1}{\sin^2\theta}\right] \frac{\partial^2 \psi_4}{\partial \phi^2} - \Delta^2 \frac{\partial}{\partial r}\left(\frac{1}{\Delta} \frac{\partial \psi_4}{\partial r}\right) - \frac{1}{\sin\theta}\frac{\partial}{\partial \theta}\left(\sin\theta \frac{\partial \psi_4}{\partial \theta}\right) - 4\left[\frac{M(r^2-a^2)}{\Delta} - r - i a \cos\theta\right] \frac{\partial \psi_4}{\partial t} - 4\left[\frac{a(r-M)}{\Delta} + \frac{i\cos\theta}{\sin^2\theta}\right] \frac{\partial \psi_4}{\partial \phi} + (4\cot^2\theta - 2) \psi_4 = 4\pi \Sigma T_4$$
通过变量分离 $\psi_4 = \sum_{lm\omega} R_{lm\omega}(r) S_{lm\omega}(\theta) e^{-i(\omega t - m\phi)}$，角向方程解为**自旋加权长椭球谐函数（Spin-Weighted Spheroidal Harmonics）**，远场退化为自旋加权球谐函数（SWSH），这构成了 FEW 模式分解的理论根基。

---

## 第二篇：统计信号处理与引力波匹配滤波（基于 Jaranowski & Krolak、Thrane & Talbot）

### 2.1 平稳高斯噪声与内积定理
设探测器应变输出为 $d(t) = h(t; \theta_{\text{true}}) + n(t)$，其中 $n(t)$ 为平稳零均值高斯白噪声，其自相关函数满足：
$$E[n(t) n(t')] = C_n(t - t')$$
频域协方差矩阵为对角阵：
$$E[\tilde{n}(f) \tilde{n}^*(f')] = \frac{1}{2} S_n(f) \delta(f - f')$$
根据奈曼-皮尔逊引理（Neyman-Pearson Lemma），使假警报概率固定时检测概率最大的最优线性滤波器是**匹配滤波器（Matched Filter）**，其频域核为：
$$K(f) \propto \frac{\tilde{h}(f)}{S_n(f)}$$

### 2.2 费雪信息矩阵（Fisher Information Matrix）与参数估计误差界
在信噪比 $\rho \gg 1$ 的高斯近似下，似然函数对真实参数 $\theta_0$ 的泰勒展开由**费雪矩阵 $\Gamma_{ij}$** 控制：
$$\Gamma_{ij} = \left\langle \frac{\partial h}{\partial \theta^i} \;\middle|\; \frac{\partial h}{\partial \theta^j} \right\rangle$$
根据**克拉美-罗下界（Cramér-Rao Lower Bound, CRLB）**，任何无偏估计量 $\hat{\theta}^i$ 的协方差矩阵满足：
$$\text{Cov}(\hat{\theta}^i, \hat{\theta}^j) \ge (\Gamma^{-1})_{ij}$$
* **波形误差容限**：如果波形建模误差引入的系统偏差 $\Delta h$ 满足 $\langle \Delta h | \Delta h \rangle \le 1$，则参数估计偏差 $\Delta \theta^i \le \sqrt{(\Gamma^{-1})_{ii}}$，即系统误差低于高斯统计涨落！
* 这就是 Lindblom 判据中 $\mathcal{M} \le 1/(2\rho^2)$ 的统计物理本源。

---

## 第三篇：现代 GPU 体系结构与高性能数值分析（基于 Kirk & Hwu、Higham、NVIDIA 规范）

### 3.1 IEEE 754 浮点体系与舍入算术
任何浮点数 $x \in \mathbb{F}$ 表示为：
$$x = (-1)^s \times (1.m_1 m_2 \dots m_p)_2 \times 2^{e - 	ext{bias}}$$
* **FP32（单精度）**：1 位符号，$p=23$ 位尾数，8 位指数。机器精度 $\epsilon = 2^{-24} \approx 5.96 \times 10^{-8}$，Unit in the Last Place (ULP) $\approx 1.19 \times 10^{-7}$；
* **FP64（双精度）**：1 位符号，$p=52$ 位尾数，11 位指数。机器精度 $\epsilon = 2^{-53} \approx 1.11 \times 10^{-16}$，ULP $\approx 2.22 \times 10^{-16}$。

#### 浮点基本定理与无误差变换（EFT）
对任意浮点操作 $\text{fl}(a \odot b)$，标准 IEEE-754 保证：
$$\text{fl}(a \odot b) = (a \odot b)(1 + \delta), \quad |\delta| \le \mathbf{u} = \frac{1}{2} \epsilon$$
**Dekker-Knuth TwoSum 定理**：存在纯浮点算法，使得加法舍入误差 $e = a + b - \text{fl}(a+b)$ 能够被精确计算出来，且无任何下溢丢失：
$$s = \text{fl}(a + b), \quad e = \text{fl}((a - \text{fl}(s - (s - a))) + (b - (s - a)))$$
这使得 GPU 可以利用两个 FP32 变量组合为 **Double-Single（48位尾数）**，兼具 FP32 的极高吞吐量与接近 FP64 的绝对数值精度。

### 3.2 GPU 内存层次与延迟隐藏机制
GPU 与 CPU 架构的根本差异在于**延迟隐藏（Latency Hiding）**而非**延迟优化（Latency Minimization）**：
* **CPU**：依靠巨大的 L1/L2/L3 缓存和复杂的分支预测器降低单指令延迟；
* **GPU**：通过成千上万个并发线程（Warp-level Context Switching）填补显存加载的数百周期停顿。

**对 FEW 的加速铁律**：
1. **合并访存（Memory Coalescing）**：相邻线程必须访问相邻内存。列优先（Column-Major）与行优先（Row-Major）的不匹配会直接导致显存带宽浪费 80% 以上。
2. **避免寄存器溢出（Register Spilling）**：每个 SM 的寄存器文件容量固定（RTX 2080 Ti 为 64 KB/SM）。若核函数使用的局部变量过多导致寄存器溢出到本地显存（Local Memory / DRAM），内核性能将产生悬崖式下跌。
3. **消除隐式同步**：在 host-device 间，尽量用流管道（Streams）调度，杜绝使用全局屏障同步。
