# 经典教材、权威专著与官方大部头手册导读索引

<!-- 2026-09-03 CST: Curated textbook index covering General Relativity, Black Hole Perturbation, Signal Processing, and CUDA Architecture -->

本目录汇集了引力波物理、黑洞微扰论、信号处理与 GPU 并行计算的 **18 部经典教材、研究生级课程专著与官方大部头技术手册**（总计 35.8 MB，全部经过 SHA-256 完整性校验并归档于 `downloads/` 目录，由 `.gitignore` 排除主工程版本库）。

清单及数字指纹详见 [`MANIFEST.tsv`](MANIFEST.tsv)。

---

## 1. 广义相对论、引力波与黑洞微扰论经典教材 (GR & Black Hole Physics)

| 书名 / 专著名称 | 作者与机构 | 规模 | 在 FEW 中的核心指导价值 |
| :--- | :--- | :--- | :--- |
| **《Spacetime and Geometry: Lecture Notes on General Relativity》** | **Sean M. Carroll** (Addison-Wesley 经典教材原型) | 238 页 | 现代广义相对论入门圣经。系统涵盖微分流形、曲率张量、爱因斯坦场方程、Schwarzschild 与 Kerr 度规推导、引力波弱场四极辐射。 |
| **《The Motion of Point Particles in Curved Spacetime》** | **Eric Poisson** (Living Reviews in Relativity 专著) | 196 页 | **EMRI 自力与辐射反作用奠基教材**。系统推导了点质量粒子在弯曲背景时空中的测地线运动、引力辐射反作用（Radiation Reaction）、MiSaTaQuNo 方程与正则化方法。 |
| **《Self-force and Radiation Reaction in General Relativity》** | **Leor Barack & Adam Pound** (权威专著) | 137 页 | **黑洞微扰与自力理论的当代前沿教科书**。详述一阶引力自力（1SF）、二阶自力（2SF）、绝热（Adiabatic）近似与后绝热（Post-Adiabatic）轨道演化。 |
| **《Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries》** | **Luc Blanchet** (Living Reviews 专著) | 160 页 | **后牛顿（PN）理论大百科全书**。详细给出了 1PN 至 4PN 能量通量、角动量通量及轨道相位展开公式，FEW 中 5PN 轨迹模型的核心理论源头。 |
| **《Gravitational Wave Physics》** | **Scott Hughes** (MIT 物理系研究生教材讲义) | 94 页 | **FEW 主创团队经典教程**。以清晰直观的物理推导阐述了引力波偏振态、辐射机制、天体物理源以及引力波探测器的物理原理。 |
| **《The Basics of Gravitational-Wave Theory》** | **Eanna Flanagan & Scott Hughes** | 60 页 | 理论物理界公认最优雅简明的引力波理论入门讲义，推导了规范不变辐射形式与能量动量张量。 |
| **《Quasinormal Modes of Black Holes and Black Branes》** | **Emanuele Berti, Vitor Cardoso, Andrei Starinets** | 110 页 | **黑洞准正则模式（QNM）专著**。推导了 Kerr 背景下的 Teukolsky 径向与角向方程、分离变量法以及铃宕（Ringdown）模式求解。 |
| **《Gravitational Lensing and Black Holes》** | **Valerio Bozza** | 80 页 | 详尽讨论了强引力场下的测地线偏折、偏振输运与高阶轨道效应。 |

---

## 2. 引力波信号处理、数据分析与贝叶斯推断教材 (Data Analysis & DSP)

| 书名 / 专著名称 | 作者与机构 | 规模 | 在 FEW 中的核心指导价值 |
| :--- | :--- | :--- | :--- |
| **《Gravitational-Wave Data Analysis: Formalism and Sample Applications》** | **Piotr Jaranowski & Andrzej Krolak** (Living Reviews 专著) | 80 页 | **引力波信号处理与数据分析经典教材**。详细推导了平稳高斯噪声、匹配滤波内积、最优信噪比（SNR）、参数估计协方差矩阵（Fisher 矩阵）与假设检验。 |
| **《An Introduction to Bayesian Inference in Gravitational-Wave Astronomy》** | **Eric Thrane & Christopher Talbot** (教程级专著) | 76 页 | **引力波贝叶斯推断与参数估计实战教科书**。深入剖析了 MCMC 采样、嵌套采样（Nested Sampling）、似然函数构建、先验分布设定与贝叶斯证据计算。 |
| **《Basics of Gravitational Wave Data Analysis》** | **Jolien Creighton** (威斯康星大学 / LAL 核心作者) | 23 页 | 浓缩了引力波数据预处理、窗函数截断、功率谱密度估计与匹配滤波流水线设计规范。 |
| **《Gravitational Waves from Merging Compact Binaries: How Accurately Can One Extract Parameters?》** | **Curt Cutler & Eanna Flanagan** (经典专著) | 35 页 | 引力波费雪信息矩阵（Fisher Information Matrix）与参数估计误差界的开山之作。 |

---

## 3. 并行计算、CUDA 体系结构与高性能库官方大部头教材 (CUDA & HPC)

| 书名 / 专著名称 | 作者与机构 | 规模 | 在 FEW 中的核心指导价值 |
| :--- | :--- | :--- | :--- |
| **《CUDA C++ Programming Guide》** | **NVIDIA 官方权威完整手册** | 350+ 页 | **GPU 编程圣经**。涵盖线程层级（Grid/Block/Warp）、统一内存（Unified Memory）、硬件流式多处理器（SM）执行机制、异步流与并发模型。 |
| **《CUDA C++ Best Practices Guide 13.1》** | **NVIDIA 官方性能优化指南** | 118 页 | **极限性能优化全攻略**。深入讲解内存访问合并（Coalescing）、寄存器溢出防范、共享内存 Bank Conflict 规避、分支发散消除。 |
| **《CUBLAS Library User Guide》** | **NVIDIA 官方线性代数手册** | 150+ 页 | 稠密矩阵乘法（GEMM）、批处理 GEMM（`cublasGemmBatchedEx`）与混合精度矩阵运算的高级使用规范。 |
| **《CUSPARSE Library User Guide》** | **NVIDIA 官方稀疏代数手册** | 200+ 页 | 稀疏矩阵存储格式（CSR/COO/BSR）与高效求解三对角矩阵（Thomas 算法 GPU 实现），是加速 FEW 三次样条求解的绝佳参考。 |
| **《Turing GPU Tuning Guide》** | **NVIDIA 官方微架构指南** | 20 页 | 针对本地使用的 **RTX 2080 Ti（SM 7.5）** 专属调优：Tensor Core 架构、独立的 FP32/INT32 数据通路、L1 缓存/共享内存统一架构配置。 |
| **《Floating Point and IEEE 754 Compliance for NVIDIA GPUs》** | **NVIDIA 官方数值精度白皮书** | 28 页 | 深入剖析 CUDA 中单双精度舍入模式（RN/RZ/RU/RD）、FMA 融合乘加指令对精度的影响、以及不同硬件算力单元的截断特征。 |

---

## 4. 重点章节查阅与应用指南

1. **若需要研究轨道步进器（DOP853）与引力自力的误差累积**：
   * 查阅 Poisson《Motion of Point Particles》第 3 章与 Barack & Pound《Self-force》第 4 章。
2. **若需要优化波形匹配度（Match）与噪声加权内积**：
   * 查阅 Jaranowski & Krolak《Gravitational-Wave Data Analysis》第 2-3 章与 Thrane & Talbot 第 2 章。
3. **若需要消除 CUDA 内核延迟并引入混合精度/Tensor Core**：
   * 查阅 NVIDIA《CUDA Programming Guide》第 5 章（硬件实现机制）与《CUDA Best Practices Guide》第 9 章（指令吞吐与显存合并）。
