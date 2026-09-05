# FEW 多层次问题审计：从科学目标到可复现实验

<!-- 2026-09-04 11:35 CST (linux): Create the first design-first problem
audit for the deep-optimization branch. The audit separates model scope,
verified defects, engineering debt, and research hypotheses; source code is
used as supporting evidence rather than as the organizing principle. -->

<!-- 2026-09-04 12:05 CST (linux): Reclassify the high-mode and trajectory
algorithm findings after the phase-two literature review. The new evidence is
kept within its demonstrated circular-Kerr-flux and NIT domains rather than
being generalized to all FEW strain models. -->

<!-- 2026-09-04 18:39 CST (linux): Restore this audit byte-for-byte from the
deep-optimization stash onto codex/cuda-mixed-precision-5x, then add only this
lineage note.  Branch-specific status at the end is historical; see the phase-
four empirical update for current CUDA evidence. -->

## 0. 结论先行

FEW 当前最核心的问题不是“某个 CUDA kernel 还不够快”，而是缺少一份贯穿
**科学用途、物理近似、离线数据、在线波形、探测器响应和参数估计**的统一契约。
没有这个契约，下面几类误差和性能数字很容易被混为一谈：

1. 物理模型相对真实 EMRI 的误差；
2. 离线通量/振幅数据和插值带来的模型实现误差；
3. ODE、样条、模式截断和浮点精度带来的数值误差；
4. CPU、CUDA、Metal 之间的后端实现差异；
5. 未经 LISA 噪声加权的数组差异与真正影响搜索、SNR 和参数估计的误差。

因此，优化 FEW 应遵循下面的依赖顺序，而不是从硬件峰值算力倒推：

```text
科学用途与允许偏差
  -> 物理近似和参数域
  -> 离线数据及代理模型
  -> 数值算法与逐阶段误差预算
  -> 在线计算图和硬件放置
  -> 软件接口、测试、发布和可复现证据
```

本审计不把所有限制都叫作“缺陷”。它使用四种分类：

- **范围边界**：模型有意不描述的物理，若用户越界使用才构成风险；
- **确认问题**：当前设计或测试中可由源码、论文或实测直接证明的问题；
- **证据缺口**：方案可能正确，但现有证据不足以支持广泛结论；
- **研究方向**：值得探索，但不能提前承诺效果的假设。

严重度也必须和用途绑定：`P0` 表示会阻止相应科学用途或使结论不可审计，
`P1` 表示显著影响可靠性/性能/可维护性，`P2` 表示中长期能力建设。

## 1. 应当区分的十二个层次

| 层次 | 首要设计问题 | 典型输出 |
| --- | --- | --- |
| L0 科学任务 | 是搜索、快速注入、精确 SNR，还是无偏参数估计？ | 用途档位与验收阈值 |
| L1 物理系统 | 黑洞、次级天体、轨道和环境包含哪些自由度？ | 物理参数空间 |
| L2 近似层级 | 绝热、后绝热、自作用力和过渡区保留到什么阶？ | 动力学方程 |
| L3 波形表示 | 用哪些模式、基底、时域/频域表示及截断？ | 模式振幅与相位 |
| L4 离线模型 | 通量/振幅网格如何生成、压缩、插值和验证？ | HDF5、ROM/ROMAN |
| L5 数值模拟 | ODE、根求解、样条、求和和精度如何组合？ | 离散轨迹与波形 |
| L6 探测器/统计 | 如何进入 LISA TDI、PSD 内积、SNR 和似然？ | 可观测量与似然 |
| L7 计算算法 | 计算图的关键路径、并行度、访存与批处理是什么？ | 算法复杂度与剖析 |
| L8 硬件执行 | CPU/CUDA/Metal 各执行哪些阶段、用什么精度？ | 后端执行计划 |
| L9 软件实现 | API、状态、错误处理、线程安全和扩展边界如何定义？ | 可维护的软件契约 |
| L10 验证发布 | 测试是否覆盖正确的参数域、硬件和科学指标？ | 可审计验收证据 |
| L11 数据治理 | 数据、代码、环境和结果能否重建和长期追踪？ | 来源、版本、校验和 |

把“可微分、Agent 调用、服务化”作为横跨 L7--L11 的使用方式，而不是替代物理
准确性的独立层。一个运行很快或能自动求导的错误模型仍然是错误模型。

## 2. L0：科学目标与产品定义

### P0-01：一个“准确”标签覆盖了不同用途

**分类：确认问题 / 设计缺口。** FEW 同时被用于方法开发、搜索/识别、模拟数据、
SNR 研究和参数估计，但这些用途允许的误差完全不同。最新 Kerr 赤道偏心模型论文
明确指出：其相对“无误差绝热波形”在大部分参数域可达到约 `1e-5` 的
LISA 加权 mismatch，同时又明确说明它尚未包含真实信号无偏分析所需的后绝热修正。
两句话并不冲突，因为参照物不同。

问题在于软件接口没有要求用户声明用途，也没有随结果返回“物理阶次、参数域、
数据版本、模式策略、数值容差、后端精度和探测器假设”。因此同一条波形可能被
不加区分地称为“LISA 准确”。

**应做的验证/改进：** 定义至少三个用途档位：

- `exploration/search`：强调吞吐，允许更宽的模式/精度近似；
- `science-forecast`：必须使用 LISA 响应与 PSD 加权指标；
- `unbiased-inference`：除工程误差外，还必须满足所选 SNR 下的模型偏差门槛，
  当前绝热模型应明确标记“不具备完整物理条件”。

每次生成都应能导出 machine-readable provenance，而不是只返回裸数组。

### P0-02：没有端到端、可分配的误差预算

**分类：确认问题。** 当前资料分别给出了 ODE 容差、插值误差、数组相对误差、
平坦内积 mismatch 和部分 LISA 加权结果，但没有一张误差账本说明各阶段允许消耗
多少预算，以及误差是否相关。优化者因此可能把 `FP32` 的逐元素误差和论文中的
物理模型 mismatch 直接比较，这是量纲和参照物均不同的比较。

**应做的验证/改进：** 为每种用途固定以下误差分解并保存交叉项：

```text
真实信号
 -> 物理近似
 -> 离线高精度数据
 -> 插值/代理模型
 -> 轨迹/相位离散化
 -> 模式选择与求和
 -> 后端浮点实现
 -> LISA 响应和似然
```

验收既要有冻结输入的组件误差，也要有独立构造的端到端误差；两者不能互相替代。

## 3. L1--L2：物理系统与近似层级

### P0-03：全相对论快速模型仍未覆盖 generic Kerr

**分类：范围边界。** 当前高保真快速族覆盖 Schwarzschild 偏心轨道和 Kerr
赤道偏心轨道；generic Kerr（同时偏心且倾斜）的公开快捷路径仍主要依赖 PN5-AAK
类近似。真实 generic Kerr 的三频结构会产生更多边带和更高维的通量/振幅数据。

这不是现有赤道模型的 bug，但它阻止把 FEW 当作完整的 EMRI 波形族。最新 FEW
论文也把进动轨道、随后 generic 偏心进动轨道列为未来扩展。

### P0-04：绝热阶足以做很多研究，但不足以保证真实信号无偏推断

**分类：范围边界，且对 `unbiased-inference` 是阻断项。** FEW 的相对论快速模型
使用轨道平均通量驱动慢演化。最新 FEW 论文明确指出，达到 LISA 所需的亚弧度相位
准确性最终需要第一后绝热动力学，其中会用到一阶和二阶自作用力信息以及次级天体
自旋贡献。这里不能写成“绝热模型毫无用处”：论文同样说明它能支持探测算法、
科学前景和大量绝热层面的研究。

**正确策略：** 给模型标注物理阶次；先为已有的圆 Schwarzschild 1PA 数据建立
可插拔 forcing-term 接口和对照实验，再扩展轨道复杂度。不能用减小浮点误差来补偿
缺失的物理项。

### P1-05：共振问题依赖轨道族，不能一概而论

**分类：范围边界。** generic 倾斜轨道会经过径向--极向瞬态共振，可能产生守恒量
跳跃和后续相移；但 Schwarzschild 偏心及 Kerr 赤道偏心模型没有独立的极向振荡，
不能把 generic `r-theta` 共振直接指责为赤道模型内部的 bug。FEW 论文指出赤道偏心
共振更弱并在其绝热模型阶次之外，而 generic 扩展需要检测共振面并施加预计算跳跃。

### P1-06：模型在 transition-to-plunge 前终止，不包含完整 IMR

**分类：范围边界。** FEW 在 separatrix 外设置缓冲并终止绝热积分，因为该区域更适合
transition-to-plunge 模型。它没有描述 plunge、merger 和 ringdown。原有蓝图所称
“轨道频率趋于无穷导致傅里叶崩溃”是不正确的论证；真正问题是近 separatrix 的
绝热慢演化失效及需要匹配过渡模型。

对典型 EMRI 参数估计，论文认为 merger-ringdown 的贡献可能较小；对较低质量比的
IMRI，它可能更重要。因此优先级必须由目标源族和 SNR 决定。

### P1-07：次级自旋、环境效应、非 Kerr 偏离和记忆尚非标准组成

**分类：研究方向/范围边界。** 这些项有重要科学价值，但“真空 Kerr”本身是明确的
基准假设，不应被称为错误。真正不足是缺少统一的 perturbation/plugin contract，
使研究者难以组合额外 forcing、频率修正、振幅修正和相应 provenance。

## 4. L3--L4：波形表示、离线数据与代理模型

### P0-08：参数域边缘的可靠性不是均匀的

**分类：已发表的证据缺口。** Kerr 模型的标称域可到 `|a|<=0.999`、`e<=0.9`、
`p<=200`，但论文认为最稳健区域约为 `|a|<=0.998`、初始 `e<=0.85`，并报告高自旋、
高偏心尾部具有更大的插值误差。这说明“参数合法”不等于“精度均匀”。

**应做的验证/改进：** 输出局部精度标签；用自适应网格、边缘专用网格或误差指示器
处理难区；把训练/插值域、稳健域和仅可计算域分开暴露。

### P1-09：离线成本被隐藏，数据产品却缺少完整模型卡

**分类：确认问题。** 在线波形约百毫秒的代价来自巨大的离线计算。Kerr 论文报告
615,810 个通量点和约 605,000 CPU 小时。当前注册表有 URL、标签和 SHA-256，
这是优点；但还缺少统一记录生成器版本、物理约定、目标精度、输入域、网格构造、
独立验证集、许可证和逐区域误差统计的机器可读 model card。

如果只知道文件哈希，就能证明“字节相同”，却不能证明“这些字节适合当前用途”。

### P1-10：5.09 GB Kerr 振幅表的表示面向批量预计算，而非移动端在线访问

**分类：确认的性能/部署问题。** 当前主线实现初始化时读取完整系数数组。已有双机
实测记录表明完整模型约需 `6.47 GiB` 进程 RSS，CUDA 内存池曾保留约 `5.20 GiB`；
而一次调用只使用一个或相邻两个自旋切片。这使移动设备、多个 worker 和 Agent
按需调用的冷启动成本不合理。

`gemini/cuda-mixed-precision` 的惰性切片/LRU 原型方向正确，但仍缺少并发安全、
缓存抖动、跨进程、冷/热访问和派生分块文件的完整证据。惰性加载解决容量与启动，
不自动等于热计算加速。

### P1-11：固定模式上限不是天然错误，但需要边界收敛证据

**分类：有限域已证实的风险 + FEW 广泛域证据缺口。** Kerr 表采用
有限的 `l,m,n` 范围；固定截断是任何快速模型的正常设计。2026 年的快速
EMRI 系统误差研究已在其**圆 Kerr 通量**设置中给出直接证据：为达到该研究的
精度目标，`a >= 0.9` 时需要 `l_max >= 30`，而样条边界条件误差可沿轨迹累积；
其 Chebyshev 表示也用更少的离线点达到了目标。

但这不能直接判定 FEW 当前 `l<=10` **辐射振幅表**在赤道偏心全域失败：
通量截断、strain 模式截断、轨道族和验收指标不同。因此必须分别建立
`flux-convergence` 与 `strain-convergence` 账本，再把局部误差转换为轨迹相位、
TDI mismatch 和 SNR/参数偏差风险。

### P1-12：ROMAN 的主要风险是校准和域外行为，不是“没有时间记忆”

**分类：纠正旧结论。** ROMAN 映射的是绝热分解中某个瞬时测地轨道的振幅系数，
其输入为轨道参数而非完整时间序列，因此 MLP 无“上一时刻记忆”不是原则性缺陷。
真正的问题是：网络训练数据、损失、保留验证集、最坏误差、域外检测、权重版本与
不确定度没有以模型卡形式随发行物完整暴露。

FP32/FP16 可否使用也必须以这种局部校准和最终波形误差决定，而不是以网络类型决定。

## 5. L5：数值模拟与误差传播

### P0-13：长时间相位误差是关键状态量，但跨平台验证容易混合两种问题

**分类：确认问题。** 自适应 DOP853 的最后位差异可能改变接受/拒绝步序列，随后影响
样条结点和累计相位。此前 Mac/Linux 独立轨迹出现约 `1.77e-5` 量级端到端差异，
而冻结同一求和输入后的 CPU/CUDA/Metal 差异远小得多，说明“独立构造可复现性”和
“同输入 kernel 正确性”必须分别报告。

**应做的验证/改进：** 轨迹诊断模式应记录接受/拒绝步、步长、状态、导数、频率、
相位和第一个分叉点；再用容差扫描、固定结点参照和独立求解器决定科学影响。

### P1-14：误差控制集中在局部 ODE 容差，未统一控制最终观测误差

**分类：设计缺口。** 一个统一的绝对容差作用于尺度不同的轨道变量和相位变量，
并不直接等价于最终 noise-weighted mismatch。轨迹插值、振幅插值、模式选择和求和
各自也有误差，当前没有运行时的组合估计器。

长期方向不是盲目把所有步骤设成 FP64/更小容差，而是建立敏感度驱动的自适应预算：
对相位和强模式严，对弱振幅及不敏感区域可松，并以最终观测指标闭环。

### P1-15：边界处理混合了 clamp、断言、异常和停止规则

**分类：确认的软件/数值问题。** 参数映射和插值涉及舍入修正、边界 clamp、
`assert`、`ValueError` 与 separatrix 缓冲。Python 在 `-O` 下会删除 `assert`，所以
用户输入和物理域校验不应依赖断言。边界行为还应区分“数学上可算”“数据域内”
和“科学上已验证”。

### P1-16：模式选择误差不是单个功率阈值可以全局概括的

**分类：确认问题。** 最新 FEW 论文指出当前在线选择需要先计算完整振幅集合，成本
较高，且不是 mismatch 意义下的全局最优；高 SNR 时较激进的阈值可造成参数偏差。
因此阈值必须依赖 PSD、源参数、SNR 和用途，而不是固定的“通常很好”。

## 6. L6：探测器响应与统计推断

### P0-17：裸 `h+ / hx` 一致不等于 LISA 数据产品一致

**分类：确认的验收缺口。** FEW 核心输出源/SSB 波形；完整二代 TDI、时变星座响应
通常由 `fastlisaresponse` 等外部组件完成。最新论文的参数估计明确使用了外部
fastLISAresponse、二代 TDI 和特定轨道近似。

Apple/CUDA 优化若只比较平坦内积或源波形，能证明工程一致性，却不能单独证明
对 LISA 观测和似然无影响。正式科学门槛至少要包含：

- 相同响应配置下的 TDI `A/E/T` 或 `X/Y/Z`；
- 明确 PSD 和频带的加权内积；
- 时间/相位对齐规则；
- SNR 偏差与代表性 injection-recovery；
- 阈值随目标 SNR 的缩放。

### P1-18：性能目标应从“每条波形”扩展到“每个有效似然样本”

**分类：设计缺口。** 参数估计成本还包含响应、FFT/NUFFT、内积、似然和采样器，
只优化振幅 kernel 可能不会改变总分析时间。应同时测量 cold latency、warm latency、
批吞吐、显存、能耗，以及完整 likelihood evaluations/s。

## 7. L7--L8：计算图、并行算法与硬件执行

### P0-19：关键路径是 CPU 串行轨迹加后端求和，不是单一神经网络

**分类：已发表且已实测。** 最新 FEW 论文测得约 90% 在线成本来自轨迹积分和波形
求和；代表性案例中轨迹约 `0.10 s`、振幅约 `0.02 s`、TD/FD 求和约 `0.15/0.60 s`。
因此 ROMAN FP32 即使大幅提高局部 GEMM 吞吐，对一年波形的端到端收益也可能很小。
此前 RTX 2080 Ti 的 `cupy_fp64` 实验正体现了这一点：短波形约 `1.23x`，一年案例
只有约 `1.026x`。

这不是说轨迹“绝对不可并行”。单条自适应时间推进存在顺序依赖，但仍可通过批量
多源、右端向量化、parallel-in-time、代理轨迹或硬件友好的求解器探索。更重要的
是，近同一变换（NIT）已在相邻自作用力模型中表明，**消去不必逐周期解析的快时标**
可比直接加速 ODE kernel 获得更大的算法级收益；针对轨道共振的部分平均/全平均
切换方案在其研究范围内报告至少两个数量级加速。这些结果尚未集成或验证于
FEW，但已足以把 `NIT overlap study` 排在精度下沉或并行时间原型之前。
每种方法都需要独立精度证明。

### P1-20：当前 GPU 原生路径存在过度同步和短生命周期资源

**分类：确认的工程问题。** 主线 CUDA ROMAN 路径逐层创建/销毁 cuBLAS handle 并
同步；时间域求和还创建多条 stream，却在循环内进行设备级同步，削弱了并发意义。
同 FP64 的 CuPy 路径已证明移除部分调度开销能加速小/中批量。这是比立即降精度更
低风险的优化方向。

### P1-21：模式选择发生在完整振幅计算之后，节省不到上游工作

**分类：确认的算法问题。** 选择过程需要完整振幅功率，主要减少最终求和，而不能
避免大部分振幅生成。可研究分层筛选：廉价上界/代理排名 -> 候选模式精确振幅 ->
误差校正；但必须防止遗漏弱而统计上重要的模式。

### P1-22：后端抽象把数组所有权、加速器和 kernel 能力混成一个选择

**分类：确认的架构问题。** 实际最优方案已经是混合的：CPU FP64 轨迹、CUDA/Metal
并行 kernel、可能的 FP32 振幅和高精度最终累计。单个 `force_backend` 不能充分表达
每阶段的数据位置、精度、异步行为和 fallback。

目标应是 per-operation execution plan：

```text
operation -> implementation -> input/output location -> dtype/accumulation
          -> workspace/cache -> synchronization -> fallback -> error contract
```

### P1-23：混合精度尚缺“精度策略”，目前只有零散 dtype 实验

**分类：证据缺口。** 合理的候选是相位/频率/最终累计保留 FP64，ROMAN 隐层或部分
插值尝试 FP32，取消/求和敏感处采用 FP64、补偿求和或 double-single。但真正选择
必须由硬件、批量、参数位置和误差预算决定。

`gemini/cuda-mixed-precision` 已实现 ROMAN `cupy_fp32` 原型，但提交内的结构化 JSON
仍是 `cupy_fp64` 测量；提交信息中的 FP32 mismatch 尚无同等可复现证据。因此目前
不能宣称 FP32 加速已经通过。

### P2-24：移动端/Agent 使用需要容量可控和并发可控，而不只是单次更快

**分类：设计缺口。** 移动设备和 Agent 工作负载通常短、突发、重复且可能并发。
需要模型预热、可取消调用、内存上限、缓存策略、结构化错误、确定性选项和资源回收。
5 GB eager load、隐式网络下载和全局/实例可变状态都会放大运维风险。

## 8. L9：软件实现与接口

### P1-25：生成器是可变状态对象，重复/并发调用契约不清晰

**分类：确认问题。** 波形基类会在调用时更新 `inspiral_kwargs`、`end_time`、
`num_modes_kept`、缓冲区和样条缓存。这样便于交互式使用，却使同一实例的并发调用、
异常后重用和服务化行为难以推理。

**改进：** 把不可变 model configuration、可复用 resource context 和单次 call state
分开；明确实例是否 thread-safe，并提供显式 session/workspace。

### P1-26：隐式下载让首次调用不可预测

**分类：确认的运维问题。** 文件管理器支持哈希校验和离线失败模式，这是良好基础；
但默认首次调用可能下载数 GB 数据。当前下载直接写目标文件，缺少显式的容量预检、
可恢复断点、临时文件后原子重命名以及清晰的模型准备阶段。

对 notebook 尚可，对作业调度、移动端和 Agent 服务不够可靠。应提供 `few prepare`
生成锁定清单，并让生产调用默认不发生网络 I/O。

### P1-27：错误处理与 API 约定仍有技术债

**分类：确认问题。** 物理输入校验中仍有 `assert`；部分 native CUDA 失败路径使用
`exit(0)`，这会把错误报告为成功退出并杀死宿主进程；还有未实现分支和 TODO 行为。
库代码应返回结构化异常和上下文，不应终止 Python/Agent 进程。

### P2-28：可微分路径有价值，但不等于必须重写整个 FEW

**分类：研究方向。** 当前 Python/Cython/C++/CUDA、离散模式选择和自适应 ODE 不能
直接形成通用自动微分图，这限制 HMC、梯度优化和 Fisher/Jacobian 的成本。但“改用
JAX/PyTorch 就能把数月 MCMC 变几小时”是未经 FEW 端到端实验证实的承诺。

更稳妥的路线是先定义导数契约，比较有限差分、切线/伴随 ODE、解析局部导数和可微
代理的准确性与成本，再决定哪些阶段值得重写。

## 9. L10：测试、验证与发布

### P0-29：测试通过不代表关键物理断言被测试到

**分类：确认缺陷。** 当前测试中至少存在以下直接可见的覆盖漏洞：

- Kerr 振幅的五个参考点在循环中计算，但断言位于循环外，实际只检查最后一点；
- 非物理输入测试在同一个 `assertRaises` 中先触发负距离异常，后续负参数循环不可达；
- 一个 mismatch 测试的第二个断言再次比较 `x0/x1`，没有使用刚构造的 `x2`；
- noise-weighted mode-selection 测试生成了 `ls_nw/ms_nw/...`，构造比较波形时却继续
  使用旧的未加权模式数组，不能证明噪声加权选择有效。

这些问题说明测试数量不能替代覆盖语义审计。

### P0-30：公开 CI 没有执行 CUDA 数值路径

**分类：确认问题。** 当前 GitHub Actions 的公开矩阵覆盖多个 Ubuntu/macOS/Python
组合，但构建明确使用 `FEW_WITH_GPU=OFF`；没有真实 NVIDIA GPU job。CUDA wheel
可以发布而核心数值路径没有持续硬件回归门槛。

应建立至少一条受控 GPU CI/nightly：安装后端插件、运行冻结输入、组件对照、短/长
波形和显存泄漏测试，并记录 GPU/driver/runtime/toolkit/架构。

### P1-31：高内存测试与常规 CI 的边界不清晰

**分类：确认问题。** 5.09 GB Kerr 文件同时带 `testfile` 和 `high_memory` 标签；
公共缓存任务按 `testfile` 预取，而常规矩阵未统一禁用 `high_memory`。主线 Kerr 振幅
测试本身也未显式标注高内存。结果是常规 CI 可能承担大下载、缓存和内存开销。

应该把小型结构/切片 fixture、完整科学验收、性能基准分成不同工作流。

### P1-32：跨后端阈值需要物理来源和统计覆盖

**分类：证据缺口。** 当前双机工作已经区分了归一化最大误差、L2 和 mismatch，
并解释了 AAK CPU/CUDA 特殊函数实现差异，这是良好进展。但测试点仍少，许多阈值
是工程经验值，尚未覆盖参数域边缘、SNR 分层、TDI/PSD 和参数偏差。

### P1-33：缺少性能回归、内存回归和能耗回归门槛

**分类：确认的验证缺口。** 单次手工 benchmark 能证明某台机器上的一次结果，
不能防止未来 commit 恶化。应保存原始重复样本、置信区间和环境，建立宽松但稳定的
regression envelope；移动端还应加入能耗和热降频观察。

## 10. L11：数据治理、协作与可复现性

### P0-34：优化分支的基线和证据已经发生分叉

**分类：当前项目状态问题。** `deep-optimization` 当前根在上游 `master` 的
`47e4fea4`，并非 `codex/apple-silicon-dual-host` 或任一 CUDA 混合精度分支。
它复制了知识库和实测报告，却不包含相应 Apple/FP64/FP32/lazy-loading 源码。
因此本分支适合做独立问题审计，但不能直接运行那些原型；任何实验结论必须注明
具体 commit/branch，之后再决定以 cherry-pick、merge 或重做方式建立统一基线。

### P1-35：现有若干探针未纳入版本与 provenance 管理

**分类：确认问题。** 工作区有五个未跟踪 probe；它们没有 Linux/CST 文件注释，
部分依赖当前分支不存在的 `cuda_roman_mode`，并包含未经依据的“LISA threshold”文字。
在清理前它们只能算草稿，不能作为验收证据。为避免破坏他人工作，本审计不修改或
删除这些文件。

### P1-36：结果文件应可验证语义，而不只是哈希

**分类：设计缺口。** JSON/NPZ 加 SHA-256 能验证传输一致性，但报告还应包含：代码
commit、数据文件哈希、依赖锁、编译选项、设备、随机种子、完整参数、计时范围、
同步位置、参考实现和通过规则。验证器应重新计算通过状态，而不是信任 JSON 中的
`passed: true`。

## 11. 需要明确撤回或降级的旧判断

现有 `NEXT_GEN_EMRI_SIMULATION_BLUEPRINT.md` 更像头脑风暴，以下表述不应作为结论：

1. **“FEW 用工程技巧掩盖基础物理缺失”**：不公允。FEW 明确定位为快速绝热框架，
   并公开讨论物理边界；真正问题是用途标签和未来物理扩展尚未产品化。
2. **“plunge 时轨道频率趋于无穷”**：错误。问题是绝热近似/数据域在过渡区失效，
   FEW 在 separatrix 外终止。
3. **“固定 `lmax=10` 必然导致高偏心严重能量丢失”**：缺证据。应做模式收敛和
   LISA 加权验证，且现有论文已经给出大范围约 `1e-5` 的绝热模型结果。
4. **“ROMAN 没有时间记忆，所以不能表示尾效应”**：概念混淆。它拟合的是瞬时
   轨道振幅映射；应审计的是训练误差、域外检测和不确定度。
5. **“全量 JAX/PyTorch 重写必然解决参数估计”**：研究假设。自动微分有价值，但
   离散选择、自适应积分、显存和梯度准确性仍需验证。
6. **“神经 ODE 一次前向传播可准确并行生成十年轨迹”**：未经证明，不能作为路线
   承诺；可作为代理模型实验候选。

## 12. 根因排序与行动顺序

### 第一优先级：先建立可判定的科学契约

1. 定义三种用途档位和各自的 reference、PSD/TDI、SNR 与误差门槛；
2. 建立逐阶段 error-budget ledger 和 provenance schema；
3. 明确当前模型族在 `unbiased-inference` 上的后绝热/generic 覆盖缺口；
4. 修正四个已识别的空测试，建立参数域分层采样。

### 第二优先级：修复在线计算的结构性成本

1. 将 5 GB Kerr 数据改为可验证、线程安全的惰性/分块 provider；
2. 消除 cuBLAS handle/stream 的短生命周期和设备级过度同步；
3. 以 trajectory + summation 为主要剖析对象，并测完整 likelihood；
4. 设计廉价候选模式筛选，避免“全算后再删”；
5. 让 precision 成为逐操作策略，并以 FP64 reference 和科学门槛选择。

### 第三优先级：扩展模型能力

1. generic Kerr 相对论振幅/通量；
2. 1PA forcing、次级自旋和共振插件；
3. transition-to-plunge 及面向 IMRI 的 merger-ringdown；
4. 局部不确定度/OOD 指示；
5. 导数契约及可微 likelihood 路径。

## 13. 第一批可证伪实验

| 实验 | 要回答的问题 | 必须保存的证据 |
| --- | --- | --- |
| E1 误差账本 | 哪个阶段首先耗尽科学预算？ | 各阶段独立/组合扰动、TDI PSD mismatch、SNR |
| E2 参数域地图 | 高自旋/高偏心/近边界哪里失效？ | 分层采样、独立数据、局部误差和失败类型 |
| E3 轨迹复现 | Mac/Linux 首次何处分叉，是否影响科学量？ | 步序列、状态/频率/相位、容差扫描 |
| E4 Kerr data provider | lazy/chunk/cache 是否真正减少容量且不拖慢扫描？ | 冷/热、多自旋、多进程、RSS/VRAM/I/O |
| E5 CUDA 调度 | 持久 handle、少同步、graph 哪个产生端到端收益？ | kernel timeline、同步数、批量曲线、waveform/s |
| E6 混合精度 | 哪些操作可降精度，哪些必须补偿？ | dtype ledger、边界网格、TDI mismatch、速度/能耗 |
| E7 模式选择 | 能否在计算完整振幅前淘汰候选？ | recall、漏失 SNR、参数偏差、总耗时 |
| E8 inference loop | 波形更快是否真的让推断更快？ | 完整响应+似然吞吐、有效样本/s、后验偏差 |

每个实验都必须允许失败；如果只设计能证明方案“成功”的指标，它不是科学验证。

## 14. 证据来源

### 主要论文与项目资料

- [FEW v2 Kerr eccentric-equatorial model and validation](https://arxiv.org/html/2506.09470)
  — 参数域、绝热范围、插值误差、性能分解、模式选择缺点和未来物理扩展。
- [FEW framework paper](https://arxiv.org/abs/2104.04582)
  — 模块化架构、Schwarzschild 相对论模型、AAK、模式约简和 GPU 加速。
- [Rapid fully relativistic EMRI templates](https://arxiv.org/abs/2008.06071)
  — 原始 ROMAN/快速全相对论 Schwarzschild 偏心波形。
- [Adiabatic generic-Kerr analytical waveforms](https://arxiv.org/abs/2111.05288)
  — generic Kerr 绝热建模、共振修正及其局限。
- [Importance of transient resonances](https://arxiv.org/abs/1608.08951)
  — generic EMRI 共振对轨迹、相位和可探测性的影响。
- [Model waveform accuracy standards](https://arxiv.org/abs/0809.3844)
  — 噪声加权误差与探测/测量准确性条件。
- [Systematic errors in fast relativistic EMRI waveforms](https://arxiv.org/abs/2509.08875)
  — 圆 Kerr 通量的模式截断、插值边界、长相位和参数偏差的直接研究。
- [Fast self-forced inspirals](https://arxiv.org/abs/1802.05281)
  与 [resonance-aware NIT evolution](https://arxiv.org/abs/2405.21072)
  — 快/慢时标平均化及共振附近部分平均切换的算法证据。
- [LISA waveform modelling review](https://arxiv.org/abs/2311.01300)
  — LISA 波形工作组对模型阶次、用途和未解问题的综合背景。

### 本仓库与双机证据

- [`FEW_ARCHITECTURE_AND_APPLE_ADAPTATION.md`](FEW_ARCHITECTURE_AND_APPLE_ADAPTATION.md)
  — 已有的计算图、四类误差预算、5 GB 数据结构和 Apple 路由审计。
- [`MIXED_PRECISION_PLAN.md`](MIXED_PRECISION_PLAN.md)
  — 逐阶段精度候选和门槛。
- [`FEW_PHASE2_CROSS_DISCIPLINARY_HYPOTHESES.md`](FEW_PHASE2_CROSS_DISCIPLINARY_HYPOTHESES.md)
  — 三条跨层断层、74 个可证伪问题及十项优先实验。
- [`FEW_PHASE3_ADVERSARIAL_FAILURE_AUDIT.md`](FEW_PHASE3_ADVERSARIAL_FAILURE_AUDIT.md)
  — 进一步审计约定、有效域、循环验证、调用历史、并发和运行时完整性中的
  68 个“平滑但可能错误”反例。
- [`collaboration/linux/cuda_mixed_compute_probe.json`](../collaboration/linux/cuda_mixed_compute_probe.json)
  — RTX 2080 Ti 同 FP64 CuPy 调度实验。
- [`collaboration/linux/HANDOFF.md`](../collaboration/linux/HANDOFF.md)
  与 [`collaboration/mac/HANDOFF.md`](../collaboration/mac/HANDOFF.md)
  — CPU/CUDA/Metal、完整 Kerr 表和跨主机复现记录。
- 当前主线实现：
  [`waveform/base.py`](../src/few/waveform/base.py)、
  [`trajectory/integrate.py`](../src/few/trajectory/integrate.py)、
  [`amplitude/ampinterp2d.py`](../src/few/amplitude/ampinterp2d.py)、
  [`cutils/matmul.cu`](../src/few/cutils/matmul.cu)、
  [`tests/`](../tests/)、
  [CI workflow](../.github/workflows/run-tests.yml)。

## 15. 当前审计边界

这是一份设计层问题地图，不声称所有问题已经完成动态复现。已确认的测试覆盖漏洞、
CUDA 资源生命周期、CI 配置和分支拓扑来自当前 `deep-optimization@23eef90c`
的静态证据；性能数字来自注明 commit/设备的既有双机报告和 FEW 论文。

第二阶段文献审计把高模式/插值边界的部分风险提升为有限域直接证据，并把
NIT、共振切换、transition/plunge、global fit 污染与按条件混合精度加入待验证
队列；详见第二阶段报告。下一轮应优先完成 E1（误差账本 schema）与四个测试
漏洞的最小复现，然后再决定
是否把当前分支建立到 Apple/CUDA 统一代码基线上。未经这个选择，不应在本分支直接
修改生产 kernel。

第三阶段又发现若干当前源码可直接确认的静态风险：相对论模型的质量比有效域检查
不一致、部分非有限输入可穿过比较、极化/探测器测试没有数值断言、模式选择 warning
与实际状态可能不一致、异常后对象状态和 pickle 自定义配置可能不守恒。它们的动态
影响和修复优先级以第三阶段 Y1--Y12 为准，不能只凭静态阅读宣布科学结果已经错误。
