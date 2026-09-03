# FEW 在 Apple Silicon 上的 CPU 适配：原理、实现与验证

<!-- 2026-09-03 17:18 CST (linux): Add the user-requested 1–3 page,
undergraduate-readable Apple Silicon CPU adaptation report. Every implementation
and measurement claim is tied to pinned code, primary documentation, or accepted
dual-host evidence; references are outside the stated body length. -->

> 正文按常见 A4 技术报告排版约 2–3 页；文末参考文献与代码索引不计入正文篇幅。

## 摘要

FastEMRIWaveforms（FEW）把极端质量比旋近（EMRI）波形拆成轨道演化、振幅计算、插值和谐波求和等模块；这种模块化设计使热点可以逐个替换，而不必改动物理模型。[FEW 的框架论文][R1]给出了这一总体结构。本次 Apple Silicon **CPU** 适配的核心不是把 CUDA 代码机械翻译为 ARM 指令，也不是使用 GPU，而是保持 `float64/complex128` 科学语义，把三类热点交给 macOS 自带的 Accelerate BLAS/LAPACK，并用 Grand Central Dispatch（GCD）并行化写入互不重叠的循环。M3 Pro 上的确定性测试、关闭加速的 A/B 基线、独立 wheel 测试，以及 Ubuntu CPU/CUDA 重放共同构成验收依据。[E1][E2]

## 1. 为什么需要适配，为什么这样做

Apple Silicon 使用 arm64 指令集。原 FEW CPU 后端虽然能把共享的 `.cu` 文件复制为 `.cxx` 后用普通 C++ 编译器构建，[C2]但其中部分矩阵乘法仍是三重标量循环，样条求解依赖外部 LAPACKE/Fortran 运行时。代码“能编译”不等于“利用了 M 芯片”：标量循环难以充分利用向量单元、缓存分块和经过调优的矩阵内核；外部 Homebrew LAPACK 还曾导致未解析的 Fortran 符号。[E1]

Accelerate 是 Apple 面向 CPU 的系统数值框架。Apple 明确说明其 BLAS/LAPACK 会为运行时处理器选择合适实现，并以高性能和低能耗为目标。[R2][R3] 因此，保持数学操作不变而替换实现是风险最低的路径：

\[
C_{ij}=\sum_{k}A_{ik}B_{kj}
\]

仍然是同一个 FP64 矩阵乘法，只是从 FEW 自己的逐元素循环换成 `cblas_dgemm`；复数投影同理换成 `cblas_zgemm`。三次对角样条方程仍由高斯消元求解，只是从 LAPACKE 包装层换成 Accelerate 提供的 `dgtsv_`。Netlib 对 `DGTSV` 的定义正是求解一般三对角线性系统 \(AX=B\)。[R4] 这种“等价算子替换”比改变精度或重写物理公式更容易验证。

第二类加速来自任务级并行。若不同谐波模式或样条区间写入不同输出区域，它们可以同时执行；但同一采样点内部的模式累加次序必须保留，否则浮点加法不满足结合律，可能改变长波形。Apple 的 `dispatch_apply_f` 正是同步并行 `for`：迭代可并发，函数返回前等待全部迭代完成，并要求工作函数可重入。[R5] 这与 FEW 的“区间间并行、区间内保持原求和顺序”相匹配。[C5][C6]

## 2. 具体实现

**第一步：建立可回退的平台开关。** CMake 新增 `FEW_USE_APPLE_ACCELERATE=AUTO|ON|OFF`：`AUTO` 在 Apple 平台解析为 `ON`，在其他平台为 `OFF`；非 Apple 平台强制 `ON` 会立即报错。[C1] 这一步有两个目的：普通 Mac 构建自动走系统库；`OFF` 又能生成标量参考版本做同机 A/B。子模块只在 CPU 目标上定义 `FEW_USE_APPLE_ACCELERATE=1` 和 `ACCELERATE_NEW_LAPACK=1`，并用 `find_library(... Accelerate REQUIRED)` 链接系统 framework；CUDA 目标不继承这些设置。[C2] 因而默认源码安装只需 `pip install .`，需要基线时显式传入：

```bash
pip install . \
  --config-settings=cmake.define.FEW_USE_APPLE_ACCELERATE=OFF
```

该命令及默认行为也写入了项目安装说明。[C8]

**第二步：替换密集线性代数热点。** ROMAN 神经网络的实数层从标量 \(O(mnk)\) 循环换为列主序 `cblas_dgemm`，偏置与 LeakyReLU 仍沿用 FEW 原逻辑。[C3] 随后的复数降阶基投影使用 `cblas_zgemm`。由于 FEW 的复数类型是为了兼容 CUDA 自定义的 `complex<double>`，代码先用 `static_assert` 检查它与 Accelerate 复双精度类型的大小、对齐，再进行指针解释；这一步防止“数值类型相同、二进制布局不同”的 ABI 错误。[C3] Apple 的 BLAS 文档确认 CBLAS 支持列主序接口，并说明 `ACCELERATE_NEW_LAPACK` 选择新版接口；本实现保留 32 位维度，因此不启用 ILP64。[R3]

**第三步：去掉不必要的外部 LAPACKE 依赖。** CPU 三次样条需要批量解三对角系统。Apple 路径直接调用 `dgtsv_`，尺寸、右端项数和 `info` 均使用 `__LAPACK_int`；非 Apple CPU 仍调用原 `LAPACKE_dgtsv`，CUDA 仍调用 cuSPARSE。[C4] CMake 因此只在非 Accelerate 路径查找或下载 LAPACKE，[C2]解决了 Mac 对 Homebrew LAPACK 和 gfortran 运行时的偶然依赖，同时不改变 Linux 构建。

**第四步：只并行化能证明独立的循环。** `global.h` 封装了一个 C++14 `few_apple_parallel_for`，内部使用 `dispatch_apply_f(..., DISPATCH_APPLY_AUTO, ...)`。[C5] 二维振幅插值按模式并行，每个任务只写 `z[mode, :]`；AAK 和时域波形按样条区间并行，每个区间只写自己的采样范围。[C6] 首次运行发现每个波形任务约 540 KiB 的临时数组可能超过 dispatch worker 栈，随后将这些缓冲区改为任务私有堆内存；5000 模式分块和单采样点求和顺序均保持不变。[C4][E1] 这既消除栈溢出，也避免任务间共享可写缓冲区造成数据竞争。

## 3. 怎样证明“更快但没有算错”

验证分三层，而不是只看一次运行结果。

1. **算子级测试。** 固定随机种子 `20260901`，用 NumPy 分别计算 FP64 实数网络层和 `complex128` 投影参考值，Accelerate 结果采用 `rtol=atol=2×10⁻¹⁴` 比较。[C7] 同时构建 `OFF` 标量版本，避免把两个调用同一系统 BLAS 的上层库误当成独立参考。
2. **模块与波形级 A/B。** 同一台 M3 Pro、同一数据和单 BLAS 线程下，比较关闭/开启 Accelerate+GCD；记录完整模块时间而非只报微内核。结果如下。[E1]

   | 工作负载 | 原标量 CPU | Accelerate/GCD | 加速比 | 数值结果 |
   | --- | ---: | ---: | ---: | --- |
   | 实数网络层 `(1000,128,128)` | 11.440 ms | 0.1455 ms | 78.6× | 最大绝对误差 0 |
   | 复数 ROM 投影 `(1000,32,384)` | 8.880 ms | 0.5763 ms | 15.4× | 最大绝对误差 `2.082×10⁻¹⁷` |
   | 1000 点 ROMAN 振幅 | 1.4013 s | 0.05558 s | 25.2× | 相对 L2 `5.985×10⁻¹⁶` |
   | 128 点二维振幅插值 | 0.06602 s | 0.01687 s | 3.9× | 逐位相同 |
   | 0.01 年 Schwarzschild 波形 | 0.05121 s | 0.03506 s | 1.46× | mismatch 0 |
   | 短 AAK 波形 | 0.003352 s | 0.002394 s | 1.40× | 逐位相同 |

   微内核比完整波形快得更多是正常的：完整流水线还含轨迹、Python 调度、数据读取和不能并行的部分，符合 Amdahl 对串行部分限制总体加速的分析；因此不能把 78.6× 宣称为整个 FEW 的加速比。[R6]
3. **跨主机独立复算。** Mac 生成带数据文件 SHA256、形状和随机种子的参考 artifact；Ubuntu 用当前源码分别在 CPU 与 CUDA 12.x 上重算六类工作负载。[C9] Ubuntu CPU 相对 Mac Accelerate 的 ROMAN 振幅相对 L2 为 `1.143×10⁻¹⁵`，Schwarzschild 波形为 `5.817×10⁻¹⁵` 且 mismatch 为 0；Mac 快速套件及更广的非高内存套件也全部通过。[E1][E2] 独立 wheel 只链接 Accelerate、`libc++` 与 `libSystem`，说明没有把 Homebrew LAPACK/gfortran 间接带回。[E1]

## 4. 结论与边界

这次适配已实现一个可维护的 Apple Silicon FP64 CPU 基线：原生 arm64 构建、系统 Accelerate 线性代数、无 OpenMP 依赖的 GCD 并行，以及可关闭、可重放、可跨主机核验的验证链。主要收益集中在 ROMAN 矩阵计算和可独立分区的插值/求和；收益大小随模型和波形长度变化，不能用单一倍数概括。

还应明确两条边界。第一，Accelerate 是 **M 芯片 CPU** 路径，不是 Metal GPU；后来加入的 `force_backend="metal"` 是另一条显式、混合精度受控的 GPU 路径。第二，当前证据证明的是实现一致性和所测工作负载的工程正确性，不等于覆盖所有 EMRI 参数、所有探测器响应和参数估计偏差。后续 CPU 优化应先做端到端 profiling，再考虑 vDSP/vForce；任何改变浮点精度或归约顺序的方案，都必须重新经过波形级误差和跨主机验证，而不能仅凭内核吞吐量决定。[R1][E2]

## 参考文献与代码索引（不计入正文篇幅）

### 文献与官方文档

- <a id="R1"></a>**[R1]** Katz, M. L. et al., “FastEMRIWaveforms: New tools for millihertz gravitational-wave data analysis,” 2021. [arXiv:2104.04582](https://arxiv.org/abs/2104.04582).
- <a id="R2"></a>**[R2]** Apple, [Accelerate framework overview](https://developer.apple.com/documentation/accelerate).
- <a id="R3"></a>**[R3]** Apple, [BLAS library](https://developer.apple.com/documentation/accelerate/blas-library)；包含 CBLAS、列主序、新 LAPACK 与 LP64/ILP64 接口说明。
- <a id="R4"></a>**[R4]** Reference LAPACK, [`DGTSV` routine](https://www.netlib.org/lapack/explore-html/d1/db3/dgtsv_8f.html).
- <a id="R5"></a>**[R5]** Apple, [`dispatch_apply_f`](https://developer.apple.com/documentation/dispatch/dispatch_apply_f).
- <a id="R6"></a>**[R6]** Amdahl, G. M., “Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities,” 1967. [DOI:10.1145/1465482.1465560](https://doi.org/10.1145/1465482.1465560).

### 固定提交的项目代码与验证证据

以下链接固定到同步提交 `e2967b675ad0a7d8decc7dca544c0fea46438efd`，避免后续行号变化改变本文依据。

- <a id="C1"></a>**[C1]** [`FEW_USE_APPLE_ACCELERATE` 的 AUTO/ON/OFF 解析](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/CMakeLists.txt#L54-L137).
- <a id="C2"></a>**[C2]** [Accelerate framework 与编译定义](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/CMakeLists.txt#L282-L314)、[CPU 矩阵扩展目标](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/CMakeLists.txt#L441-L468)和[样条扩展的 LAPACKE 回退](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/CMakeLists.txt#L485-L518).
- <a id="C3"></a>**[C3]** [`cblas_dgemm`、`cblas_zgemm` 与复数 ABI 检查](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/matmul.cu#L193-L342).
- <a id="C4"></a>**[C4]** [`dgtsv_`](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/interpolate.cu#L233-L253)和[任务私有堆缓冲区](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/interpolate.cu#L465-L510).
- <a id="C5"></a>**[C5]** [`dispatch_apply_f` 的 C++14 封装](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/global.h#L17-L39).
- <a id="C6"></a>**[C6]** [按振幅模式并行](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/AmpInterp2D.cu#L89-L122)、[按 AAK 样条区间并行](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/gpuAAK.cu#L508-L548)与[按时域样条区间并行](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/interpolate.cu#L765-L815).
- <a id="C7"></a>**[C7]** [Accelerate FP64/complex128 单元测试](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/tests/test_apple_accelerate.py#L15-L70).
- <a id="C8"></a>**[C8]** [Apple CPU 默认构建与显式关闭命令](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/README.md#L170-L193).
- <a id="C9"></a>**[C9]** [双主机验证协议](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/validation/README.md#L1-L53).
- <a id="E1"></a>**[E1]** [Mac 实现、A/B 性能、wheel 和回归证据](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/collaboration/mac/HANDOFF.md#L5-L123).
- <a id="E2"></a>**[E2]** [Ubuntu CPU/CUDA 独立复算结果](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/collaboration/linux/HANDOFF.md#L75-L102).

[R1]: #R1
[R2]: #R2
[R3]: #R3
[R4]: #R4
[R5]: #R5
[R6]: #R6
[C1]: #C1
[C2]: #C2
[C3]: #C3
[C4]: #C4
[C5]: #C5
[C6]: #C6
[C7]: #C7
[C8]: #C8
[C9]: #C9
[E1]: #E1
[E2]: #E2
