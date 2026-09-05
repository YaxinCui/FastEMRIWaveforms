# FEW Apple Silicon CPU 加速测试报告与验收方案

<!-- 2026-09-03 18:00 CST (linux): Add the user-requested evidence-based test
report for the Apple Silicon FP64 Accelerate/GCD CPU path. The report separates
completed measurements from proposed coverage and distinguishes engineering
regression limits from detector/SNR-dependent scientific accuracy. -->

## 摘要与结论

本报告回答四个问题：Apple Silicon CPU 加速究竟改了什么；应测试哪些层次和参数；怎样构造公平、可复现的测试；测得的数值误差能否用于 EMRI 模拟。测试对象是 FEW 的 **FP64/complex128 CPU 路径**：密集矩阵运算和三对角求解交给 Apple Accelerate，互不重叠的模式或样条区间交给 Grand Central Dispatch（GCD）并行；它不是 Metal GPU 路径。[R1][R2][C1][C2]

当前证据支持以下结论：

1. 在已测 M3 Pro 上，实数 GEMM、复数 ROM 投影、ROMAN 振幅、二维插值和完整波形分别获得不同程度的加速；不能把最快微内核的 `78.6×` 当成整个 FEW 的加速比。[E1]
2. 同机标量/Accelerate 算子误差约为 FP64 舍入量级；跨主机的 Schwarzschild、AAK 和全表 Kerr 测试也通过了预先写入验证器的尺度归一化误差与 `1×10⁻¹⁰` 波形 mismatch 门限。[C5][C6][C7][E2]
3. 因而，**就已覆盖的实现一致性而言，这些 CPU 加速误差可以接受**。但这不等价于“所有 LISA 科学分析已经验证”：最终科学判断还依赖 LISA 噪声谱、探测器响应、信号信噪比（SNR）、时间/相位最大化以及参数偏差研究。[R5]

## 1. 测试对象：不是测一块芯片，而是测一条计算链

FEW 将波形构造拆成轨道演化、振幅生成、模式选择、角函数和模式求和等模块。[R6] Apple CPU 适配只替换实现热点，未改变 EMRI 物理方程：

```text
物理参数
  → 自适应轨迹与相位（CPU）
  → HDF5/ROM 振幅插值（CPU；Accelerate/GCD 热点）
  → 模式选择与角函数（CPU）
  → 样条与模式求和（CPU；GCD 可并行区间）
  → h₊ + i h×
```

Accelerate 版本把标量矩阵循环替换为 `cblas_dgemm` 和 `cblas_zgemm`，[C3]把 CPU 样条中的三对角方程交给 `dgtsv_`，[C4]并用 `dispatch_apply_f` 运行可证明独立的迭代。[C4] Apple 将 Accelerate 定义为利用向量处理能力的 CPU 数值框架；`dispatch_apply_f` 允许迭代并发，但要求工作函数可重入。[R1][R2] 因此测试必须同时覆盖：数学等价、数据布局/ABI、并发独立性、完整流水线和运行时间。

## 2. 测试目标与判定问题

| 目标 | 要回答的问题 | 失败意味着什么 |
| --- | --- | --- |
| 构建正确性 | `AUTO/ON/OFF` 是否按平台解析？arm64 wheel 是否只链接预期系统库？ | 适配不可部署或意外依赖 Homebrew/Fortran |
| 算子正确性 | `dgemm/zgemm/dgtsv` 是否保持 FP64、形状、列主序和复数 ABI？ | 最底层数学替换错误 |
| 并发正确性 | 每个 GCD 任务是否写独立区域？重复运行是否稳定？ | 数据竞争、越界或归约次序变化 |
| 模块正确性 | ROMAN、二维振幅插值、样条与求和是否与标量参考一致？ | 错误位于组合逻辑而非单算子 |
| 端到端正确性 | Schwarzschild、AAK、Kerr 波形是否保持形状、有限值与重叠？ | 加速改变最终模拟输出 |
| 性能收益 | 冷启动、热运行、不同长度和批量下是否真实加速？ | 只优化了微内核或调度开销吞噬收益 |
| 资源与稳态 | 峰值内存、线程数、温度稳定后吞吐是否可接受？ | 移动设备上可能内存不足或热降频 |
| 跨主机可复现性 | Mac CPU、Linux CPU、CUDA 在相同数据/输入下差异来自哪一层？ | 无法区分实现问题与独立轨迹积累 |

这里的核心原则是“逐层冻结输入”。如果直接比较两台机器各自生成的一年波形，差异同时包含自适应轨迹、插值和求和；若先冻结求和输入，则剩余差异才可归因于求和内核。[C9][E2]

## 3. 测试范围

### 3.1 构建与打包

在同一台 Apple Silicon Mac 上建立两个互相隔离的 wheel：

```bash
# 加速候选：只测 CPU，显式排除 Metal
python -m pip wheel . --no-deps -w /tmp/few-accelerate \
  --config-settings=cmake.define.FEW_USE_APPLE_ACCELERATE=ON \
  --config-settings=cmake.define.FEW_WITH_METAL=OFF

# 标量参考：同一源码、编译器和 Python，仅关闭 Accelerate
python -m pip wheel . --no-deps -w /tmp/few-scalar \
  --config-settings=cmake.define.FEW_USE_APPLE_ACCELERATE=OFF \
  --config-settings=cmake.define.FEW_WITH_METAL=OFF
```

依据是项目提供的 `AUTO|ON|OFF` 开关和 Apple-only 防误配逻辑。[C1] 应记录 Git 提交、macOS/Xcode/Python/NumPy/SciPy 版本、芯片型号与核心数、wheel SHA256；用 `file`/`lipo -info` 检查 arm64，用 `otool -L` 确认本次 CPU 扩展只引入 Accelerate、`libc++` 和 `libSystem`。随后在空白虚拟环境安装 wheel 并运行测试，防止源码目录或 Homebrew 库偶然“帮忙”。这套独立 wheel 检查已在 M3 Pro 上成功执行过。[E1]

### 3.2 算子层

1. **实数网络层：** 固定种子生成 `float64` 输入、权重和偏置；比较 `matrix @ weights + bias` 及 LeakyReLU。现有单元测试使用 `(m,k,n)=(37,11,19)` 和 `rtol=atol=2×10⁻¹⁴`。[C5]
2. **复数 ROM 投影：** 将网络的实部/虚部组合成 `complex128`，再与复数变换矩阵相乘；测试形状 `(31,7,23)`，并检查 dtype、大小和对齐。[C3][C5]
3. **三对角求解：** 构造有已知解、对角占优的系统，分别让 Accelerate `dgtsv_` 和参考 LAPACK/NumPy 求解；除输出误差外，还检查 `info=0`。Netlib 将 `DGTSV` 定义为求解一般三对角系统 \(AX=B\)。[R8]
4. **GCD 循环：** 使用 1、少于核心数、约 `3×` 核心数和远多于核心数的迭代规模；检查任务只写自身切片，重复至少 100 次，并用 Address/Thread Sanitizer 的独立调试构建排查越界和竞争。Apple 文档说明只有迭代独立且函数可重入时才适合 `dispatch_apply_f`，并提醒调度本身存在开销。[R2]

正确性的小矩阵负责发现索引和布局错误；性能测试另用接近真实工作负载的 `(1000,128,128)` 实数 GEMM 与 `(1000,32,384)` 复数投影，不能用微小矩阵推断生产速度。[E1]

### 3.3 模块层

| 模块 | 最小覆盖 | 扩展覆盖 | 主要风险 |
| --- | --- | --- | --- |
| `RomanAmplitude` | 128 点与 1000 点 `(p,e)` 网格 | 1、16、128、1000、5000 点批量 | BLAS 尺寸、缓存、批量摊销 |
| Schwarzschild 二维振幅 | 128 点网格 | 域内、网格边缘、近分离轨道 | 插值索引、GCD 模式写入 |
| 轨迹/DOP853 | 既有短轨迹 | `0.001/0.01/0.1/0.25/1` 年 | 自适应分支与相位积累 |
| 时域求和 | 短 Schwarzschild/AAK | 模式数、采样率、时长阶梯 | 区间任务、求和顺序、栈/堆 |
| 频域路径 | 现有一年轻/慢测试 | 与时域一致参数的交叉检查 | FFT/SPA 与频率网格 |

现有跨主机验证器已经覆盖固定种子 `20260901`、ROMAN、二维振幅、`0.01` 年 Schwarzschild 波形与短 AAK 波形，并绑定四个数据文件的大小和 SHA256。[C6] 扩展测试不能只抽“容易”的参数点：应从模型自己的合法域查询函数中按三层抽样——域中央、边缘附近、接近 separatrix 但仍合法——同时覆盖低/中/高偏心率以及负/零/正自旋；无效输入应验证为明确异常，而不是 NaN。

### 3.4 全表 Kerr 与资源上限

`ZNAmps_l10_m10_n55_DS2Outer.h5` 为 5,089,095,248 字节。全表测试应在单独进程进行，先核验数据 SHA256，再用同一个模型对象完成：

- 四个 `(a,p,e)` 点的全部 6993 个振幅模式；
- 五个已登记的目标模式；
- 一条 `T=0.001` 年、`dt=15 s` 的 2104 点 Kerr 波形；
- 每个输出重复两次，检查有限值、形状、逐位重复性、时间和峰值 RSS。[C7]

现有测试发现第三个上游 fixture 的“声明模式”和期望值不对应，因此只把它排除在“对上游常数”的断言之外，但仍把五个 Mac 实际值全部纳入跨主机比较。[C7] 这说明测试本身也可能有错；正确做法是记录例外、保留覆盖，而不是为了全绿删除数据点。

### 3.5 端到端科学覆盖

工程验收至少包括 Schwarzschild eccentric、PN5 AAK 和 Kerr eccentric equatorial 三条模型路径。Kerr 模型论文给出的适用范围达到 |`a`| ≤ 0.999、`e < 0.9`、`p < 200`；测试点仍须由当前代码的合法域检查确认。[R7] 建议建立小型分层网格：

- 自旋：逆行、零自旋、顺行和接近高自旋；
- 偏心率：近圆、中等和接近模型上界；
- 初始轨道：远离、居中、接近 separatrix；
- 时长：短烟雾测试、0.01/0.1 年性能测试、1 年相位积累测试；
- 观察参数：至少两组天空方向/初相位，并保留完全相同的一组作后端对照。

这个网格不是要穷举连续参数空间，而是覆盖最可能放大插值、模式数量和相位误差的方向。若用于发布，应再采用拉丁超立方或分层随机采样补足内部区域，并把随机种子写进 artifact。

## 4. 如何构建公平的性能测试

### 4.1 控制变量

- 两个 wheel 必须来自同一提交、同一编译器和依赖版本；唯一变量是 `FEW_USE_APPLE_ACCELERATE`。
- 第一组设置 `VECLIB_MAXIMUM_THREADS=1`，分离 BLAS 内部线程与外层 GCD；第二组使用系统默认线程，回答普通用户的实际体验。
- 固定电源模式，关闭无关重负载，记录是否接电、电量、环境温度和 macOS 低电量模式；长测试前后记录温度/频率或至少记录运行序号，以识别热降频。
- 每个测试放在独立进程，随机交错 Accelerate/标量顺序，避免总让某一路径享受冷机器或文件缓存。

### 4.2 冷启动和热运行必须分开

冷启动包括 Python 导入、动态库加载、HDF5 读取、模型构造和首次内存分配；热运行复用模型与缓存。二者回答不同问题，必须分别报告。全表 Kerr 已显示首次与热运行差异可达数量级，因此只报“最快一次”没有意义。[E1]

计时使用 `time.perf_counter()`，它是 Python 为短时间测量提供的最高可用分辨率、单调性能计数器。[R4] 每个案例建议：5 次预热、至少 30 次计时；报告 P10/P50/P90、均值、标准差和原始样本。核心比较量为

\[
S=\frac{\operatorname{median}(t_{\mathrm{scalar}})}
        {\operatorname{median}(t_{\mathrm{accelerate}})}.
\]

用 bootstrap 给出 `S` 的 95% 置信区间；只有下界仍大于 1，才能称该案例稳定加速。发布回归建议同时设置“不得比已冻结基线慢 10% 以上”的告警，但该百分比是工程预算，必须按同型号机器维护，不能跨 M1/M2/M3/M4 直接比较。

### 4.3 性能结果如何解释

微内核加速比描述矩阵核；模块加速比还包含激活函数、内存布局和 Python 调度；端到端加速比还包含轨迹、数据读取和串行部分。因此速度应按“算子—模块—波形”三级报告。当前 M3 Pro 结果正体现这一规律：[E1]

| 案例 | 标量 CPU | Accelerate/GCD | 加速比 | 数值差异 |
| --- | ---: | ---: | ---: | --- |
| 实数网络层 | 11.440 ms | 0.1455 ms | 78.6× | 最大绝对误差 0 |
| 复数 ROM 投影 | 8.880 ms | 0.5763 ms | 15.4× | 最大绝对误差 `2.082×10⁻¹⁷` |
| 1000 点 ROMAN 振幅 | 1.4013 s | 0.05558 s | 25.2× | 相对 L2 `5.985×10⁻¹⁶` |
| 128 点二维振幅 | 0.06602 s | 0.01687 s | 3.9× | 逐位相同 |
| 0.01 年 Schwarzschild 波形 | 0.05121 s | 0.03506 s | 1.46× | mismatch 0 |
| 短 AAK 波形 | 0.003352 s | 0.002394 s | 1.40× | 逐位相同 |

这些数据证明“热点有效”，但仍应按上述 30 次、交错顺序和置信区间协议重跑，才能形成稳定的版本性能基线。

## 5. 误差指标：每个数字分别回答什么

设参考数组为 \(x\)，候选数组为 \(y\)，差为 \(\delta=y-x\)。

1. **逐位相同**：字节完全一致，适合检查重复性；它不证明参考算法本身正确。
2. **最大绝对误差**：\(\max_i|\delta_i|\)，能找到最坏样本，但数值尺度变化时难比较。
3. **归一化最大误差**：
   \[
   E_\infty=\frac{\max_i|\delta_i|}{\max_i|x_i|}.
   \]
   它适合不同幅值案例的回归门限，但可能被一个峰值主导。
4. **相对 L2 误差**：
   \[
   E_2=\frac{\|y-x\|_2}{\|x\|_2}.
   \]
   它衡量全数组平均能量差异，但可能稀释少数局部尖峰。
5. **逐元素混合容差**：NumPy `assert_allclose` 检查
   \(|y_i-x_i|\leq atol+rtol|x_i|\)。[R3] `atol` 保护接近零的元素，`rtol` 约束相对尺度；二者都必须显式写入，不能沿用库默认值。
6. **波形 mismatch**：FEW 当前回归函数计算归一化重叠 \(\gamma=\langle x,y\rangle/(\|x\|\|y\|)\)，再取 \(1-\gamma\)。[C8] 该实现是平坦权重且不搜索时间/相位，只适合严格回归。
7. **科学 mismatch**：用探测器噪声加权内积
   \[
   \langle a|b\rangle=4\operatorname{Re}\int_0^\infty
   \frac{\tilde a(f)\tilde b^*(f)}{S_n(f)}\,df,
   \]
   并按分析目的对时间、相位等无关自由度最大化。它回答“探测器是否能区分两个波形”，不能由单点最大误差替代。[R5]

因此一次测试应同时记录形状、dtype、有限值、逐位重复性、`max_abs`、`E∞`、`E2` 和至少一种 mismatch。只看一个指标容易误判。

## 6. 验收门限及其依据

下表中的数值是**当前项目实现回归门限**，不是 FEW 物理模型精度声明。它们已经硬编码在共享验证器中，并由 Mac artifact 与 Ubuntu CPU/CUDA 重放验证。[C6][C7]

| 层次/输出 | 归一化最大误差与相对 L2 | 额外门限 | 解释 |
| --- | ---: | ---: | --- |
| 实数网络层、复数投影 | `≤5×10⁻¹³` | dtype/shape 必须一致 | 给 FP64 跨库舍入留裕量 |
| ROMAN、二维振幅 | `≤5×10⁻¹²` | 全部有限 | 多层运算后的累计门限 |
| Schwarzschild 波形 | `≤5×10⁻¹¹` | 平坦 mismatch `≤1×10⁻¹⁰` | 完整短波形回归 |
| AAK 波形 | `≤5×10⁻⁹` | 平坦 mismatch `≤1×10⁻¹⁰` | CPU 与 CUDA Bessel 实现不同的已知例外 |
| 全表 Kerr 振幅 | `≤5×10⁻¹¹` | 已知 fixture `atol=1×10⁻⁹` | 6993 模式和目标模式 |
| 全表 Kerr 短波形 | `≤5×10⁻¹⁰` | 平坦 mismatch `≤1×10⁻¹⁰` | 大数据端到端验收 |

同机算子单元测试使用更严格的 `rtol=atol=2×10⁻¹⁴`，因为它只隔离一次数学替换。[C5] AAK 的 `5×10⁻⁹` 不是“降低整个项目精度”：Linux CUDA 使用 libdevice `jn`，CPU 使用历史 Bessel 近似，实测相对 L2 为 `1.034×10⁻⁹`，但波形 mismatch 仅 `8.882×10⁻¹⁶`；该放宽只属于 AAK 数组指标。[E2]

任何新测试若失败，应先判断它违反的是：数据/输入身份、形状/dtype、逐元素门限，还是波形门限。不能通过继续放宽容差来“修复”未知原因；只有能指出不同数学实现、量化其影响并保持独立 mismatch 门限时，才可登记例外。

## 7. 这些误差能否被模拟接受？

### 7.1 对软件实现回归：可以接受

已完成的 Mac Accelerate 对标量 CPU 比较中，ROMAN 相对 L2 为 `5.985×10⁻¹⁶`；Mac 对 Ubuntu CPU 的 ROMAN、Schwarzschild 波形和 AAK 相对 L2 分别为 `1.143×10⁻¹⁵`、`5.817×10⁻¹⁵`、`4.981×10⁻¹⁶`，波形 mismatch 为 0。[E1][E2] 全表 Kerr 的 Mac/Ubuntu CPU 短波形相对 L2 为 `2.750×10⁻¹¹`、归一化最大误差为 `6.732×10⁻¹¹`、平坦 mismatch 为 0；所有门限通过。[E2]

这些量远小于当前 Kerr FEW 模型相对“无误差绝热波形”约 `10⁻⁵` 的 LISA 加权模型 mismatch。[R7] 两者使用的指标与参考对象不同，不能简单相减，但数量级对比说明 CPU 后端实现误差目前不是主导误差源。

### 7.2 对探测和参数估计：需要按 SNR 重新判断

Lindblom、Owen 与 Brown 给出的充分精度条件以噪声加权波形差 \(\langle\delta h|\delta h\rangle\) 为基础；用于精确参数测量时，常用充分条件是该量小于 1。[R5] 对主要为波形形状正交误差的小偏差，可近似得到

\[
\mathcal M_{\mathrm{impl}}\lesssim\frac{1}{2\rho^2},
\]

其中 \(\rho\) 是信号 SNR。例：\(\rho=100\) 时约为 `5×10⁻⁵`，\(\rho=1000\) 时约为 `5×10⁻⁷`。当前 `1×10⁻¹⁰` 工程 mismatch 门限比这两个示例更严，但只有在使用同一 LISA/TDI 响应、同一噪声谱和正确的时间/相位处理后，才能把这种比较升级为科学结论。

### 7.3 为什么较大的点误差仍可能有极小 mismatch

一年波形在 Mac 与 x86_64 上独立重算时，曾出现 `1.7693×10⁻⁵` 的归一化最大差和 `6.7103×10⁻⁶` 的相对 L2；但平坦、相位优化和近似 LISA 噪声加权 mismatch 的最坏值约为 `2.5122×10⁻¹¹`。[E2] 原因是两台主机分别执行自适应 DOP853，微小舍入差会改变后续轨迹和相位；局部复波形可写成

\[
h=Ae^{-i\phi},\qquad \frac{\delta h}{h}\approx
\frac{\delta A}{A}-i\,\delta\phi .
\]

最大点误差捕捉最坏时刻，而 mismatch 衡量整体方向；二者本来就回答不同问题。冻结相同求和输入后，CPU/CUDA 的最坏相对 L2 回到 `5.29×10⁻¹⁶`，证明较大差异主要来自上游独立轨迹，而非求和内核。[C9][E2]

因此应采用两级结论：

- **工程接受：** 固定输入后通过严格数组门限；独立端到端重算通过预定 mismatch 门限；当前已达到。
- **科学接受：** 在目标 LISA 响应、噪声 PSD、SNR 和参数网格上，通过噪声加权 mismatch，并用 Fisher 或 Bayesian 注入恢复确认参数偏差低于预算；当前尚未完成全面网格。

## 8. 推荐的执行顺序与失败定位

1. 校验分支、提交、数据 SHA256 和两个 wheel；任一不一致则停止，避免比较不同输入。
2. 运行 Accelerate `ON` 的单元测试和完整快速套件；确保候选本身健康。
3. 用固定种子运行 `OFF`/`ON` 算子 A/B；若失败，先查列主序、维度和复数 ABI。
4. 运行 ROMAN、二维插值、样条和求和模块 A/B；若仅 GCD 失败，查任务写区间、任务私有缓冲区和求和次序。
5. 运行短 Schwarzschild/AAK/Kerr 波形；先检查形状、dtype、有限值，再判断数组误差与 mismatch。
6. 运行 0.1–1 年案例；若差异只在独立轨迹中增长，则传输冻结的轨迹/样条/求和输入重新比较，不要直接归罪于 Accelerate。
7. 在性能测试中分开冷/热运行、单 BLAS 线程/默认线程和短/长任务；只有正确性全部通过后才解释速度。
8. 最后进行 LISA PSD/TDI 和参数恢复测试；只有这一层可以支持“科学分析可接受”的推广声明。

## 9. 最终判定与尚未覆盖的工作

Apple Silicon 的 FP64 Accelerate/GCD CPU 适配已通过当前算子、模块、短/长波形、全表 Kerr、独立 wheel 和 Ubuntu CPU/CUDA 一致性验证；已测误差可以作为 FEW 模拟的工程实现误差被接受。它的优势是保持 `float64/complex128`，因此适合做 Mac 上的可信基线，也适合校验后续 Metal 混合精度路径。

仍需补足三类发布级证据：第一，按同一协议在不同 M 系列芯片上建立基线，而不是把 M3 Pro 倍数外推；第二，在合法参数域进行分层随机网格和长时间压力测试；第三，以完整 LISA/TDI 响应、目标 SNR 和参数注入恢复形成科学误差预算。完成这三项后，才能把“已测实现可接受”提升为“对目标科学任务普遍可接受”。

## 参考文献与固定代码依据

### 文献与官方文档

- <a id="R1"></a>**[R1]** Apple, [Accelerate framework](https://developer.apple.com/documentation/accelerate)：Accelerate 是 CPU 上的高性能、低能耗向量与线性代数框架。
- <a id="R2"></a>**[R2]** Apple, [`dispatch_apply_f`](https://developer.apple.com/documentation/dispatch/dispatch_apply_f)：同步并行循环、完成等待和可重入要求。
- <a id="R3"></a>**[R3]** NumPy, [`numpy.testing.assert_allclose`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose)：`atol + rtol × |reference|` 判据。
- <a id="R4"></a>**[R4]** Python, [`time.perf_counter`](https://docs.python.org/3/library/time.html#time.perf_counter)：短时测量的高分辨率单调性能计数器。
- <a id="R5"></a>**[R5]** Lindblom, Owen & Brown, “Model Waveform Accuracy Standards for Gravitational Wave Data Analysis,” 2008, [arXiv:0809.3844](https://arxiv.org/abs/0809.3844).
- <a id="R6"></a>**[R6]** Katz et al., “FastEMRIWaveforms: New tools for millihertz gravitational-wave data analysis,” 2021, [arXiv:2104.04582](https://arxiv.org/abs/2104.04582).
- <a id="R7"></a>**[R7]** Chapman-Bird et al., “The Fast and the Frame-Dragging,” 2025, [arXiv:2506.09470](https://arxiv.org/abs/2506.09470).
- <a id="R8"></a>**[R8]** Reference LAPACK, [`DGTSV`](https://www.netlib.org/lapack/explore-html/d1/db3/dgtsv_8f.html).

### 当前分支的固定代码与证据

以下链接固定到提交 `e2967b675ad0a7d8decc7dca544c0fea46438efd`。

- <a id="C1"></a>**[C1]** [`FEW_USE_APPLE_ACCELERATE=AUTO|ON|OFF`](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/CMakeLists.txt#L54-L137).
- <a id="C2"></a>**[C2]** [Apple CPU 目标的 Accelerate 编译与链接](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/CMakeLists.txt#L282-L314).
- <a id="C3"></a>**[C3]** [`dgemm/zgemm` 与复数 ABI 检查](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/matmul.cu#L193-L342).
- <a id="C4"></a>**[C4]** [`dgtsv_` CPU 路径](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/interpolate.cu#L233-L253)与[`dispatch_apply_f` 封装](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/cutils/global.h#L17-L39).
- <a id="C5"></a>**[C5]** [Apple FP64/complex128 单元测试](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/tests/test_apple_accelerate.py#L1-L70).
- <a id="C6"></a>**[C6]** [六工作负载双主机验证器](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/validation/dual_host_consistency.py#L25-L310).
- <a id="C7"></a>**[C7]** [5.09 GB 全表 Kerr 验证器](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/validation/high_memory_kerr_consistency.py#L25-L429).
- <a id="C8"></a>**[C8]** [FEW 平坦权重 overlap/mismatch 实现](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/src/few/utils/utility.py#L16-L95).
- <a id="C9"></a>**[C9]** [轨迹/DOP853 分层复现诊断](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/validation/trajectory_reproducibility.py#L1-L590).
- <a id="E1"></a>**[E1]** [Mac 构建、性能、全表与回归证据](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/collaboration/mac/HANDOFF.md#L3-L222).
- <a id="E2"></a>**[E2]** [Ubuntu CPU/CUDA、全表与长波形复算证据](https://github.com/YaxinCui/FastEMRIWaveforms/blob/e2967b675ad0a7d8decc7dca544c0fea46438efd/collaboration/linux/HANDOFF.md#L46-L430).

[R1]: #R1
[R2]: #R2
[R3]: #R3
[R4]: #R4
[R5]: #R5
[R6]: #R6
[R7]: #R7
[R8]: #R8
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
