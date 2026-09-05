# FEW 第二阶段跨学科问题发现：从波形函数到多保真科学系统

<!-- 2026-09-04 11:57 CST (linux): Create the phase-two cross-disciplinary
brainstorm from source-verified physics, numerical analysis, reduced-order
modeling, heterogeneous computing, LISA analysis, and research-software
literature. Every proposed problem is evidence-ranked and paired with a way to
falsify it; production source and pre-existing untracked probes remain
untouched. -->

<!-- 2026-09-04 18:39 CST (linux): Restore this hypothesis backlog from the
deep-optimization stash onto the current codex CUDA branch.  Its old branch
boundary is retained as provenance; phase four records present measurements. -->

## 0. 新结论

第一阶段把 FEW 拆成十二层；第二阶段进一步发现，真正决定下一步路线的不是层数，
而是三条贯穿各层的断层：

1. **语义断层**：`accurate`、`fast` 和 `GPU accelerated` 没有绑定到参照模型、
   科学任务、SNR、观测时长、TDI/PSD 与硬件工作负载；
2. **参数空间几何断层**：高自旋、近 separatrix、共振面、模式开启/消失和通量小量
   区域并不是均匀的欧氏矩形，统一网格、统一误差和统一精度可能在错误的位置花钱；
3. **工作负载断层**：FEW 的基本产品是“一次生成一条源波形”，而实际需求可能是
   半相干搜索、批量候选筛选、逐周 global fit、相邻参数似然、梯度、注入恢复或
   移动端短调用。这些任务的最佳计算图不同。

因此，FEW 的长期形态更适合被理解为一个**多保真波形系统**，而不是一个只有
“CPU 版”和“GPU 版”的函数库：

```text
science policy
  -> model policy (0PA / 1PA / kludge / relativistic / transition / response)
  -> data policy  (dataset version / local error / cache / provenance)
  -> execution policy (CPU / CUDA / Metal / precision / batch / async)
  -> evidence policy (reference / metric / threshold / uncertainty)
```

性能路线也应按杠杆从大到小排列：

```text
改变数学问题（平均化、频域、似然压缩）
  > 改变表示（坐标、稀疏模式、ROM、数据布局）
  > 改变调度（批处理、缓存、异步、持久资源）
  > 改变算术精度（FP64/FP32/补偿/双单精度）
```

这不是说低层优化不重要，而是只有前三级确定后，才知道哪些运算值得降精度。

## 1. 本轮如何约束头脑风暴

### 1.1 证据等级

| 标记 | 含义 | 可以支持什么结论 |
| --- | --- | --- |
| `A` | 直接的 FEW/EMRI 论文、数据或本仓库实测 | 可陈述已观察到的问题及其限定范围 |
| `B` | LISA 官方/工作组要求或相邻 GW 一手研究 | 可建立设计约束，不能自动证明 FEW 已有缺陷 |
| `C` | 数值分析、HPC、科研软件的一手方法证据 | 可提出工程候选，必须在 FEW 上复验 |
| `R` | 当前仓库源码、测试、CI 或协作记录 | 可确认实现事实，不能单独证明科学影响 |
| `H` | 本轮新假设或跨领域推论 | 只能进入实验队列，不能写成结果 |

每个问题必须至少回答：影响哪种科学任务、当前证据是什么、什么实验能推翻它。
如果没有可证伪条件，它只是口号。

### 1.2 一个重要更新

第一阶段曾把“高阶模式不足”整体归为证据缺口。2026 年发表的快速相对论 EMRI
系统误差研究已经给出更强、但范围明确的证据：在其**圆 Kerr 通量**研究中，
`a >= 0.9` 时需要 `l_max >= 30` 才满足所研究的精度目标，并发现样条边界条件可产生
可积累的低端误差；Chebyshev 表示用更少点达到了目标精度。

这不能直接证明 FEW 当前 `l<=10` **振幅表**在赤道偏心全域不合格，因为通量截断、
辐射振幅截断、轨道族和验收目标不同。正确结论是：**模式上限与插值边界已经成为
有直接证据的风险，必须逐数据产品、逐参数域验证，不能再以全局常数处理。**

## 2. 断层一：真值、用途与多保真契约

| ID | 问题或假设 | 依据 | 可证伪/改进实验 |
| --- | --- | --- | --- |
| Q01 | FEW 没有单一“真值”；0PA、1PA、自作用力、Teukolsky、EOB、NR 各自在不同区域充当 reference。若报告只写 `reference=CPU`，只能证明实现一致，不能证明物理准确。 | `A/B`：1PA、transition 和 LISA 波形综述 | 为每个模型建立 reference ladder，声明在哪个参数域、物理阶次和观测量上比较 |
| Q02 | 同一波形不应只有一个 accuracy badge。探测、半相干搜索、参数估计、GR 检验和 global fit 的门槛不同。 | `B`：LISA WAV.1 明确要求按任务制定不同标准 | 用同一批误差扰动分别跑 search efficiency、TDI mismatch、参数偏差和残差污染；检验门槛是否真的不同 |
| Q03 | “工程误差小于物理误差”仍不够，因为误差可能同方向累积；需要相关性和灵敏度，而不只是独立上界相加。 | `A/C/H`：长相位积累与数值误差理论 | 对 flux、轨迹、振幅、求和逐项注入随机/结构化扰动，测交叉项和最终 posterior bias |
| Q04 | 多保真模型之间缺少显式的嵌套关系。搜索用 kludge、跟踪用 0PA、精估用 1PA 时，参数定义与相位约定若不一致，升级模型可能跳到另一个似然模态。 | `B/H`：LISA 搜索与精估需求不同 | 对同一 injection 做逐级 warm-start，追踪最大似然点、模态映射和参数语义 |
| Q05 | 误差不一定只应被“压到零”；已知残余模型误差可以作为带结构的 nuisance/discrepancy 进入推断。当前 FEW 没有输出可供边缘化的误差基或协方差。 | `A/H`：快速模型系统误差可导致偏差 | 从独立误差样本构建低秩 discrepancy basis，比较忽略、硬阈值和边缘化三种 posterior |
| Q06 | 独立实现比同实现跨后端更能发现共同模式错误。CPU/CUDA/Metal 三者共享相同数据和公式时，三方一致仍可能一起错。 | `R/C` | 建立至少一个来自 pybhpt、独立 Teukolsky/self-force 或解析极限的 oracle，而非只做 FEW 后端互比 |
| Q07 | 模型输出缺少“我为什么选择这个模型”的可审计记录。自动 fallback 会让结果可运行却不可解释。 | `R/H` | 让 policy resolver 返回决策树：请求用途、可用数据、域检查、fallback 原因和被放宽的门槛 |
| Q08 | 版本兼容不仅是 API 兼容；同一函数名更换数据表、模式规则或 TDI 约定会改变科学语义。 | `B/R`：FAIR4RS 与 LISA conventions | 定义 semantic model ID，将物理模型、数据、接口约定和默认参数共同版本化 |

## 3. 断层二：物理阶次、快慢时间尺度与区域切换

| ID | 问题或假设 | 依据 | 可证伪/改进实验 |
| --- | --- | --- | --- |
| Q09 | 0PA 的通量与插值误差会成为未来 1PA 的地基误差，不能期待加入高阶物理后自动消失。 | `A`：2026 系统误差论文明确讨论 0PA 误差向 1PA 传播 | 在一个已有 1PA 的圆 Schwarzschild 子域中主动扰动 0PA flux，测 1PA 输出偏差 |
| Q10 | “增加一个 1PA forcing 数组”可能过度简化问题。最新工作把 1PA 表述为六维相空间上的规范不变 pseudo-Hamiltonian，并处理非局域性；FEW 需要先定义相空间/规范数据契约。 | `A`：2025 pseudo-Hamiltonian 框架 | 用最小圆轨道模型实现两种等价规范表示，验证输出观测量而非中间 self-force 分量一致 |
| Q11 | generic Kerr 的主要困难不只是表格多一维，而是三频、相位、动作变量、共振面和更多模式共同增加；直接把 `(a,p,e,x)` 做稠密张量网格可能指数膨胀。 | `A/C`：generic EMRI 与 ROM 维数灾难 | 比较张量网格、稀疏/自适应网格和低秩表示在 held-out 强场区域的误差—字节曲线 |
| Q12 | 轨迹加速的首选问题可能不是“怎样并行 DOP853”，而是“怎样不解析每个快轨道周期”。近恒等变换（NIT）已展示数量级更大的算法收益。 | `A`：NIT 论文报告毫秒轨迹和至少两个数量级的共振方案加速 | 在 FEW 可重叠子域对相同 forcing 比较现有平均轨迹、osculating geodesic 与 NIT 的 phase/runtime |
| Q13 | 完全平均化在共振处失效，因此未来轨迹器天然是 hybrid dynamical system：远离共振用全平均，附近用部分平均，之后再切回。 | `A`：2024 NIT resonance switching | 实现只记录不改变结果的 resonance monitor，测候选轨迹穿越次数、驻留时间和潜在切换误差 |
| Q14 | 共振不能只表示为状态上的瞬时 jump；文献指出它是在约 `O(epsilon^-1/2)` 周期内的平滑演化，jump 是远场摘要。不同用途需要不同表示。 | `A` | 对同一共振比较 step jump、resolved partial averaging 和无共振模型的频率/相位/TDI 差异 |
| Q15 | transition-to-plunge 不是普通 ODE 终止条件，而是不同渐近展开的匹配问题；连续性阶次本身是接口要求。 | `A`：2PLT 工作展示 inspiral/transition composite matching | 定义 `transition_state`，逐项检查轨道频率及其导数、相位和模式在拼接点的连续阶次 |
| Q16 | merger/ringdown 不能只在最大振幅处拼接。2026 偏心 Kerr 研究显示接近 light-ring 的锚点更稳健，并需处理球谐—扁球谐混合和正/逆旋 QNM beating。 | `A` | 对若干偏心/高自旋轨道扫描 attachment rule，比较 Teukolsky reference 的多模式 mismatch |
| Q17 | 次级自旋不是独立的小开关，它会改变进动、相位变量和 1PA 数据需求；“先加 spin 参数、以后补数据”可能制造无效 API。 | `A`：2025 spinning 1PA 模型有明确适用限制 | 先定义最小支持域和退化极限；验证 spin→0 连续回到无自旋模型，再扩域 |
| Q18 | 环境效应、非 Kerr 偏离和自作用力可能在相位上退化。插件化可以提高扩展性，但若没有可辨识性分析，会产生物理上不可区分的参数。 | `A/H`：FEW 非真空研究已显示忽略盘可偏置参数 | 对 correction basis 做 Fisher/posterior 相关性图，只有可区分或可合理边缘化的项才进入公共参数表 |

## 4. 断层三：参数坐标、模式和离线数据几何

| ID | 问题或假设 | 依据 | 可证伪/改进实验 |
| --- | --- | --- | --- |
| Q19 | `(a,p,e,x_I)` 是自然输入，但未必是最佳插值坐标。接近 separatrix 或极端自旋时，固定 `p` 间距可能对应非常不同的物理变化率。 | `A/H` | 比较原坐标、距 separatrix 的归一化坐标、频率/动作坐标的导数动态范围和 held-out 误差 |
| Q20 | 一个全域坐标系可能不存在。不同轨道族或近边界区域可能需要 chart/patch，并在重叠区做一致性检查。 | `H` | 训练单一全域插值器与分区 atlas，在同字节预算下比较边缘最大误差和拼接误差 |
| Q21 | 均匀网格按参数体积分配离线成本，却不按科学先验、误差曲率或 likelihood 访问频率分配。605,000 CPU 小时的下一份数据集不应沿用未证明的采样策略。 | `A/H` | 用已有查询/推断轨迹构造访问分布，对均匀、误差驱动和任务驱动采样做成本—风险比较 |
| Q22 | 样条的局部性利于增量修补，但边界条件会产生隐藏系统误差；Chebyshev 更省点但全局耦合、更新成本和高维扩展需实测。 | `A/C`：2026 系统误差研究 | 在相同离线点数下比较 spline、分块 Chebyshev、稀疏格点；分别测最大 flux、相位和 posterior 偏差 |
| Q23 | “全局最大相对误差”在通量过零、模式很弱或复振幅相消时会病态。数据模型需要绝对+相对+条件数/物理权重的组合指标。 | `C/H` | 找出接近零点与符号变化区，比较相对误差告警和实际相位/TDI 灵敏度是否一致 |
| Q24 | 通量模式截断与辐射振幅模式截断不是同一问题。把 `l_max>=30` 的通量结论直接套到 FEW `l<=10` 振幅表会犯类别错误。 | `A` | 分别做 flux-convergence 和 strain-convergence，记录各自对轨迹与探测器波形的影响 |
| Q25 | 固定模式集合浪费了参数相关稀疏性，但动态集合会引入离散不连续和不可微点。需要带滞回或上界保证的模式政策。 | `A/H` | 扫描相邻参数，检查模式集合跳变是否让波形/梯度不连续；比较 hysteresis 与连续权重 relaxation |
| Q26 | 当前“先计算全模式振幅再选择”把稀疏性用得太晚。可先用廉价上界或层级代理筛选，但漏失风险必须由 detector-weighted 指标定义。 | `A/H` | 构建 conservative upper-bound selector，报告 recall、漏失 SNR、计算节省和最坏参数偏差 |
| Q27 | ROMAN/插值器应输出局部可信度，而不是只有全局训练误差。训练域内部也可能有孔洞，几何域检查不能替代误差校准。 | `C/H`：ROM 综述强调训练集与事后验证 | 设计 withheld islands、边界带和对抗采样；校准 error predictor 的 coverage，而不只测平均误差 |
| Q28 | 5 GB HDF5 的根因不只是 eager load，而是物理查询轴与磁盘 chunk 轴可能不一致。盲目 lazy loading 可能变成随机 I/O 放大器。 | `R/C`：HDF5 官方 chunk/cache 指南 | 收集真实 query trace，比较原布局、按自旋 chunk、派生 shard 和内存映射的冷/热/P95/RSS |
| Q29 | 多 worker 共享数据时，进程复制、HDF5 handle、page cache、CuPy pool 和 Apple unified memory 会形成不同的真实容量；单进程 RSS 不能代表服务容量。 | `C/H` | 在 1/2/4/8 worker 下测总 RSS、私有/共享页、VRAM、首次缺页和吞吐，建立 admission limit |
| Q30 | 数据哈希只能确认字节；数据产品还缺少生成代码、配置、物理约定、独立验证集、许可证和区域误差的可机读 model card。 | `A/R/C`：FAIR4RS | 从任一振幅/flux 文件出发，测试另一主机能否仅凭 manifest 重建来源、语义和验收，不要求重算全部数据 |

## 5. 数值误差不是一个 `dtype`

| ID | 问题或假设 | 依据 | 可证伪/改进实验 |
| --- | --- | --- | --- |
| Q31 | 自适应 ODE 的局部 truncation error 与最终相位/似然灵敏度不一致；统一 `atol/rtol` 可能过度保护不敏感变量，同时低估相位敏感方向。 | `A/C` | 计算终态/TDI 对各状态变量的切线灵敏度，和传统 norm controller 的步点分布比较 |
| Q32 | Mac/Linux 的不同步序列不一定是“错误”；需要区分 bitwise reproducibility、轨迹等价、波形等价和推断等价四级目标。 | `R/C` | 同时输出四级判据，检查提高确定性所付的性能成本是否与用途相称 |
| Q33 | resonance、separatrix 和 plunge 是事件面；若事件定位误差没有预算，ODE 主体再精确也可能在错误时刻换模型或停止。 | `A/H` | 对事件 root tolerance 做扫描，测切换时刻、相位和最终 mismatch 的灵敏度 |
| Q34 | dense output 与重采样可能独立制造误差或混叠。轨迹步点正确不代表均匀波形采样充分，尤其是高偏心近星点 burst。 | `A/C` | 用过采样 reference 检查不同 `dt`、样条阶数、窗口和 FD/TD 路径的 aliasing/TDI mismatch |
| Q35 | 巨大未折叠相位进入 `sin/cos` 时，argument reduction 与 libm/backend 差异可支配末位；简单提高累加精度不一定解决。 | `R/C` | 冻结相位并扫描幅度数量级，比较 CPU/CUDA/Metal 特殊函数与高精度 argument reduction |
| Q36 | 模式求和的条件数随时间、视角和参数变化；平均误差掩盖强相消时刻。精度策略应看局部 cancellation indicator。 | `C/H` | 保存 `sum(abs(terms))/abs(sum(terms))` 或稳定替代指标，与 FP32/DS/FP64 误差相关性比较 |
| Q37 | FMA、fast-math、flush-to-zero 与复数实现会产生系统性而非随机误差，单次 CPU 对照不足以发现敏感路径。 | `R/C` | 建立 compiler/math-mode factorial experiment，并用 stochastic arithmetic 估计有效位数分布 |
| Q38 | 浮点误差根因非局部且不均匀，手工猜测“相位 FP64、振幅 FP32”只是起点。Herbgrind/Verificarlo 类动态诊断可帮助定位，但需验证能否覆盖 Cython/CUDA 边界。 | `C/H` | 先在 CPU 小型冻结 kernel 上做 shadow/stochastic run，判断告警是否预测最终 waveform error |

## 6. 探测器、global fit 与推断工作负载

| ID | 问题或假设 | 依据 | 可证伪/改进实验 |
| --- | --- | --- | --- |
| Q39 | TDI、星座轨道、时标、单位、坐标系和索引约定都是波形语义的一部分，不是生成后的普通后处理。 | `B/R`：LDC、PyTDI、Typed LISA Toolkit | 建立 conventions fixture，跨 fastlisaresponse/LDC 逐通道验证同一源的单位、延迟方向和时标 |
| Q40 | 固定 PSD 的 mismatch 不能覆盖非平稳噪声、缺口、校准误差和混合前景。科学验收应包含噪声模型集合。 | `B/H` | 在多 PSD、数据缺口和响应扰动下重复 mode/precision 决策，看最优策略是否稳定 |
| Q41 | LISA 是 source-rich global fit；单源无噪声注入不能测量一个近似波形对其他源和残余噪声估计的污染。 | `B`：LISA global-analysis 工作 | 将受控 FEW 近似注入含重叠源的数据，测目标源偏差、邻源偏差和 residual PSD |
| Q42 | global fit 会随着观测数据增长而持续更新；只支持“从头生成固定一年”会浪费先前计算。 | `B/H` | 设计周级 append/extend 原型，测复用轨迹、模式和响应缓存后相对全重算的收益与一致性 |
| Q43 | 搜索不一定需要整年相干精确波形。LISA 工作包明确指出半相干策略可放宽探测模型要求；把 inference 标准强加给 search 会浪费算力。 | `B` | 比较分段长度、模型阶次和搜索召回率，建立“最便宜且保持 detection efficiency”的策略 |
| Q44 | 同一内禀轨迹/模式可服务多个距离、相位、偏振、天空位置和响应配置。若 API 过早合成为裸 strain，会丢失缓存和批处理机会。 | `C/H`：模式分解与相邻 GW 推断实践 | 把生成拆为 intrinsic modes 与 extrinsic projection，测多视角/多响应批量速度、内存和数值一致性 |
| Q45 | 参数估计通常连续访问邻近参数；无状态逐次调用忽略了轨迹、活跃模式、插值切片和 workspace 的局部性。 | `H` | 回放真实 MCMC/优化 query trace，对无缓存、精确键缓存和近邻增量策略测 hit rate 与偏差 |
| Q46 | 可微 FEW 的价值在 Fisher、优化和 HMC，但离散模式选择、自适应步长和事件切换会制造不光滑梯度。只证明 forward match 不足。 | `B/C/H`：ripple 展示相邻 GW 模型收益 | 在解析极限/有限差分 oracle 上验证 JVP/VJP；扫描模式和事件边界，报告梯度失真与采样结果 |
| Q47 | ROQ、relative binning、heterodyne 和 neural proposal 加速的是 likelihood，不应把它们的近似误差算进 waveform backend 指标后就消失。 | `A/C` | 分离 waveform、response、likelihood-compression 三份误差预算，再测组合 posterior coverage |
| Q48 | 神经推断器需要可校验的 fallback。相邻研究用 importance sampling 的有效样本效率暴露 OOD；FEW/EMRI 方案也需要类似“失败可见”机制。 | `C/H` | 对边界、共振、异常噪声和模型错配输入测 coverage/ESS；低 ESS 时自动转入精确 likelihood |

## 7. 计算图、硬件与混合精度

| ID | 问题或假设 | 依据 | 可证伪/改进实验 |
| --- | --- | --- | --- |
| Q49 | kernel FLOP/s 不是主要目标；算法消除、表示压缩、I/O 和同步可能比精度下降产生更大收益。 | `A/C/R`：NIT 与既有 Amdahl 实测 | 对每个候选同时报告 operation count、bytes、launch/sync、端到端耗时和科学误差 |
| Q50 | cold latency、warm latency、吞吐和尾延迟是四个不同目标。移动 Agent 关心 cold/P95，MCMC 关心 steady batch throughput。 | `R/H` | 建立四套固定 workload，不再用一个平均“waveform time”代表所有场景 |
| Q51 | 可并行轴不只有时间样本：source、parameter proposal、mode、detector channel、sky angle 和 Monte Carlo realization 都可批处理。最佳轴取决于任务。 | `H` | 对每个轴画 batch-size—速度—内存曲线，避免把单条长波形结论外推到多源推断 |
| Q52 | 持久 cuBLAS handle、stream、workspace、CUDA graph 或 Metal pipeline cache 可能降低短调用开销，但它们引入线程安全和生命周期契约。 | `C/R`：CUDA Graph/BLAS 官方约束 | 构造单线程、多线程、多实例和 fork 后测试，测复用收益、竞争、泄漏和销毁安全 |
| Q53 | 异步只有在 API 不立即同步/拷回时才有意义。若 Python 层每阶段需要 host 结果，更多 stream 只增加复杂度。 | `C/R` | 画真实 timeline 并统计 host wait/device sync；要求候选减少关键路径而不只是增加 overlap 图形 |
| Q54 | Apple unified memory 减少显式拷贝，不等于数据移动免费；5 GB 工作集可能触发缓存、缺页和系统内存压力。 | `C/R/H` | 在 Mac 用 resident set、page fault、GPU counter 和不同访问顺序测实际迁移成本 |
| Q55 | 精度必须是 `storage × compute × accumulation × transcendental` 四元组；只写 `FP32` 无法解释误差和硬件映射。 | `C`：混合精度综述与 CUDA IEEE-754 | 每个实验自动导出 dtype ledger，并用相同 ledger 在 CUDA/Metal 上比较 |
| Q56 | 降精度适合低条件数、误差可校正的操作；相位、事件判定、强相消和边界索引可能需要高精度或补偿。策略应动态依赖条件，而非只依赖模块名。 | `C/H` | 建立敏感度 atlas，用 FP64 shadow 抽样验证动态 precision policy 的误报/漏报 |
| Q57 | RTX 2080 Ti 的 Tensor Core 优势只有在矩阵尺寸、布局和批量足够时才能兑现；ROMAN 小 batch 可能更受 launch/packing 限制。 | `R/C` | 将层融合、batch 扩展、FP32/TF32/FP16 与纯 FP64 分开测，包含转换和同步成本 |
| Q58 | 能耗和热稳定性可能改变移动端最佳方案。短 benchmark 的 Metal 优势不保证持续 Agent 服务下不降频。 | `H` | 在固定电量/温度条件下运行 30–60 分钟负载，报告波形/J、P95 latency 和热降频 |

## 8. 软件产品、并发与失败语义

| ID | 问题或假设 | 依据 | 可证伪/改进实验 |
| --- | --- | --- | --- |
| Q59 | 一个对象同时承担 immutable model config、共享资源和单次调用状态，阻碍线程安全、缓存复用和失败恢复。 | `R` | 定义三层对象并做行为等价 PoC；用并发相同/不同参数调用验证无状态污染 |
| Q60 | `force_backend` 太粗；需要 capability negotiation，按 operation 选择实现并明确 fallback，而不是假装所有后端等价。 | `R/H` | 输出 capability matrix，故意移除某能力，验证 resolver 的拒绝/降级和 provenance |
| Q61 | 数据准备应是显式、可恢复、可审计阶段。生成期间隐式下载会让超时、空间不足和网络失败混入数值 API。 | `R/C` | 设计 `few data plan/prepare/verify` dry-run，覆盖容量预检、`.part`、断点、原子安装和离线模式 |
| Q62 | 物理扩展需要 typed correction interface：作用于 actions/frequencies/phases/amplitudes 的位置、阶次、单位、域和导数都要声明。普通 Python callback 太弱。 | `H` | 用一个盘 torque 和一个 1PA toy correction 实现同一接口，检查组合顺序和退化极限是否明确 |
| Q63 | 库不应 `exit(0)` 或依赖优化时消失的 `assert`。错误必须跨 C++/CUDA/Python 保留状态、设备和物理参数上下文。 | `R` | 故障注入 CUDA allocation、坏 HDF5、域外参数，验证宿主进程存活且错误可机读 |
| Q64 | 长波形与 global fit 需要流式、可取消和进度可见；一次性分配/返回数组不利于服务和容错。 | `H` | 实现 chunk iterator 原型，测取消延迟、峰值内存、chunk 边界一致性和与整数组路径的误差 |
| Q65 | Agent 调用需要资源配额：最大数据下载、RSS/VRAM、运行时、并发、网络和 fallback 必须在执行前估计。 | `H` | 让 planner 对一组请求预测资源，比较预测与实测并测试超限拒绝，不允许进程被 OOM 杀死 |
| Q66 | 输出裸 `ndarray` 不足以携带物理/数值语义；但把所有 metadata 塞进数组对象也会破坏生态。需要轻量 result envelope 与显式 raw-array escape hatch。 | `H` | 设计兼容原 API 的 result schema，验证序列化、NumPy/CuPy 互操作和旧用户迁移成本 |

## 9. 验证、CI 与科研软件治理

| ID | 问题或假设 | 依据 | 可证伪/改进实验 |
| --- | --- | --- | --- |
| Q67 | reference fixture 由同一代码路径生成时只能做回归，不能做验证；测试应标注 oracle independence。 | `R/C` | 为每个基准记录生成器和代码血缘，至少保留解析极限、独立实现或高精度 oracle 之一 |
| Q68 | 示例点断言不足以覆盖高维边界。物理不变量、对称性、连续极限和尺度律可形成 property/metamorphic tests。 | `R/H` | 加入 `e→0`、`a→0`、距离缩放、相位平移、共轭/模式对称等属性并做随机域扫描 |
| Q69 | 随机均匀采样会错过薄的共振/边界失效区；测试采样应按几何 strata 和误差历史主动更新。 | `A/H` | 比较均匀、Latin hypercube、边界强化和 adversarial sampler 每 CPU 小时发现的问题数 |
| Q70 | CI 需要四级：快速接口、CPU 数值、真实加速器、离线科学验收。把所有测试塞进 PR CI 会过慢，把 GPU 全移出 CI 又会失真。 | `R/H` | 给现有测试标注成本/数据/设备/科学级别，构造分层矩阵并验证发布门是否覆盖关键路径 |
| Q71 | 性能回归不是一个固定毫秒阈值；需要重复样本、环境指纹、置信区间和显著性/效应量。 | `C/H` | 在稳定环境积累基线分布，以相对 envelope 和最小有意义效应判定，避免偶然噪声报警 |
| Q72 | FAIR 不仅适用于代码，也适用于模型、权重、HDF5、fixture 和报告。每个版本需要持久 ID、丰富 metadata、许可证与依赖。 | `C/R`：FAIR4RS | 用 FAIR4RS checklist 审计一个完整模型 release，统计机器可发现/可重建字段缺口 |
| Q73 | 分支复制文档而不复制代码会产生“证据悬空”。报告必须绑定 commit、数据 hash 和可运行命令，合并时重做 lineage audit。 | `R` | 自动检查报告引用的 commit/object/文件是否在当前 ancestry 可达，失败则标记 external evidence |
| Q74 | HDF5/权重属于可执行数值供应链的一部分；哈希防传输损坏，但不能防恶意/畸形输入、许可证缺失和资源耗尽。 | `C/H` | 对 parser 做大小/shape/dtype 上限、临时文件和坏数据故障注入；发布时生成 SBOM/data BOM |

## 10. 单项问题之间更危险的组合

孤立评估会低估下面这些组合风险：

1. **高自旋 × 网格边界 × 模式截断 × 长相位积累**：每项局部误差不大，组合后可能
   在最有价值的 golden EMRI 区域产生参数偏差；
2. **动态模式选择 × 自动微分**：forward 波形可连续，但离散集合变化会使梯度跳变；
3. **lazy HDF5 × 多 worker × GPU memory pool**：单调用节省内存，服务总容量反而因
   每 worker cache 和派生数组变差；
4. **FP32 振幅 × 强相消 × fast-math**：平均点误差通过，但特定视角/时刻的求和条件
   数放大误差；
5. **NIT 平均化 × 共振 × transition-to-plunge**：三个区域各自正确，不代表切换时的
   相位变量、规范和连续阶次一致；
6. **模型 fallback × global fit**：某些 proposal 静默退到 kludge，会在似然面制造
   非物理台阶；
7. **缓存 × 相邻参数复用 × provenance**：近邻复用加速明显，但若缓存键漏掉数据
   版本、模式政策或响应约定，会返回语义错误的波形；
8. **可微代理 × OOD × HMC**：错误梯度可能比无梯度更危险，因为采样器会高效地
   前往错误区域；
9. **bitwise 强制 × 跨硬件性能**：为完全复现固定求和顺序，可能失去并行性，却不
   一定改善科学可复现性；
10. **大数据自动下载 × Agent 自动重试**：失败重试可造成磁盘/网络放大，应让准备
    阶段幂等且受配额控制。

## 11. 一个供验证的五平面设计草图

这不是实现承诺，而是用来检验现有模块边界是否足够的思想实验：

```text
┌──────────────────────────────────────────────────────────┐
│ Science plane: task, SNR, duration, response, threshold │
└───────────────────────┬──────────────────────────────────┘
                        v
┌──────────────────────────────────────────────────────────┐
│ Model plane: domain + 0PA/1PA + resonance + transition  │
│              modes + correction graph + uncertainty     │
└───────────────────────┬──────────────────────────────────┘
                        v
┌──────────────────────────────────────────────────────────┐
│ Data plane: versioned providers, local error, cache, BOM│
└───────────────────────┬──────────────────────────────────┘
                        v
┌──────────────────────────────────────────────────────────┐
│ Execution plane: per-op backend/dtype/batch/async plan  │
└───────────────────────┬──────────────────────────────────┘
                        v
┌──────────────────────────────────────────────────────────┐
│ Evidence plane: reference ladder, metrics, provenance,  │
│                 CI status, uncertainty and warnings     │
└──────────────────────────────────────────────────────────┘
```

一个节点只有同时声明以下内容，才可以被 policy 自动选择：

```text
physics order and gauge/convention
parameter domain and boundary behavior
input/output units and coordinates
data dependencies and versions
local error evidence and OOD behavior
backend, dtype, accumulation and synchronization
derivative availability and smoothness
fallback and failure semantics
```

这会把“后端选择”从布尔开关提升为可以审计的执行计划，同时避免把软件抽象当成
物理模型本身。

## 12. 优先实验：先用最小成本改变路线判断

| 顺序 | 实验 | 为什么先做 | 退出条件/交付物 |
| --- | --- | --- | --- |
| X1 | **reference ladder + science contract** | 没有它，后续所有误差和速度都不可比较 | 一份 machine-readable schema；三个用途档位；至少一个完整实例 |
| X2 | **复现 2026 flux/interpolation systematics 的最小子集** | 它可能改变数据网格、模式上限和插值器路线 | 圆 Kerr 高自旋小网格；spline/Chebyshev；flux→phase→TDI 指标 |
| X3 | **NIT overlap study** | 可能比 GPU ODE 获得数量级更大的算法收益 | 与 FEW 重叠 forcing 子域；runtime/phase；明确不适用区 |
| X4 | **resonance/transition event monitor** | 在不改结果前提下测未来 hybrid solver 的实际触发范围 | 代表性 population 的事件图、驻留时间和切换敏感度 |
| X5 | **真实 query trace 驱动的数据布局实验** | 决定 5 GB 文件应 lazy、shard、rechunk 还是常驻 | cold/warm/P95/RSS，多 worker，至少三种访问模式 |
| X6 | **mode convergence 双账本** | 分清 flux 模式和 strain 模式，避免错误外推 | 高自旋/高偏心分层图；轨迹与探测器影响分别报告 |
| X7 | **precision sensitivity atlas** | 决定 CUDA/Metal 真正可降精度的操作 | dtype 四元组、随机算术/FP64 shadow、相消指标和端到端误差 |
| X8 | **global-fit contamination toy problem** | 单源 mismatch 可能漏掉最重要的系统影响 | 两类重叠源+噪声，报告目标/邻源偏差和 residual PSD |
| X9 | **intrinsic-mode reusable API PoC** | 可能直接改善多视角、响应和相邻似然吞吐 | 与旧 API 行为等价；批量收益；缓存键完整性测试 |
| X10 | **四级 CI 与 FAIR release audit** | 防止结果继续散落在主机、分支和大文件中 | 测试矩阵、model/data BOM、commit/环境/许可证/阈值 provenance |

X1--X6 在任何大规模源码重构前完成；X7 的结果才决定混合精度实现；X8--X10
决定 FEW 是否能从快速函数扩展为可靠的 LISA/Agent 基础设施。

## 13. 暂时不应承诺的路线

- 不承诺“全量 GPU 化”：单条轨迹有顺序依赖，但 NIT、批量和多保真可能改变问题；
- 不承诺“FP32/FP16 一定更快”：小批量、转换、访存和同步可能吞掉理论吞吐；
- 不承诺“lazy HDF5 一定更快”：它首先解决容量，性能取决于 chunk 与 query trace；
- 不承诺“把 `l_max` 全局升到 30”：已有证据针对特定圆 Kerr 通量问题；
- 不承诺“JAX 重写即可 HMC”：必须验证事件、离散模式与长相位导数；
- 不承诺“神经网络给出 uncertainty 就可信”：必须做 coverage、OOD 和精确重加权；
- 不承诺“一项 mismatch 门槛适合所有任务”：LISA 工作包明确区分任务；
- 不把三后端一致称为物理正确：需要独立理论/数值 oracle。

## 14. 新增证据如何改变原问题地图

| 原判断 | 本轮更新 |
| --- | --- |
| 轨迹串行是主要硬件障碍 | 仍是现实现状，但 NIT 表明先消除快尺度可能优于 parallel-in-time/GPU 移植 |
| 共振是未来物理插件 | 进一步明确为全平均/部分平均切换的 hybrid solver 与事件接口问题 |
| plunge 只是缺少末段模型 | 进一步明确需渐近匹配、连续阶次和 ringdown anchor/mode-mixing 契约 |
| 固定高阶模式缺少证据 | 更新为有特定圆 Kerr 通量证据，但不可外推到所有 FEW 振幅/轨道域 |
| spline 是成熟基础设施 | 新证据显示边界条件可积累成系统误差，且 Chebyshev 值得直接比较 |
| 混合精度靠模块分层 | 更新为按局部条件数、相消、事件与特殊函数的动态 operation policy |
| LISA 响应是端到端验收末端 | 更新为 conventions、global fit、残余污染和增量观测都属于产品语义 |
| provenance 主要为复现实验 | 更新为多保真模型解析、自动 fallback、Agent 缓存安全的运行时必要条件 |

## 15. 本轮一手资料与工程索引

### 物理、渐近展开和模型系统误差

- [Systematic errors in fast relativistic EMRI waveforms](https://arxiv.org/abs/2509.08875)
  — 直接研究 fast offline/online 模型的 flux 截断、插值、相位与参数偏差。
- [Second-order self-force waveforms](https://arxiv.org/abs/2112.12265)
  — 展示圆、无自旋 1PA 波形可在毫秒级生成，并明确 0PA/1PA 相位层级。
- [Invariant pseudo-Hamiltonian 1PA framework](https://arxiv.org/abs/2507.08081)
  — 给出六维相空间、规范不变和非局域性处理的 1PA 设计依据。
- [Spinning-primary/precessing-secondary 1PA waveforms](https://arxiv.org/abs/2510.16113)
  — 展示自旋扩展及其明确的小自旋/小失配适用限制。
- [Fast self-forced inspirals with NIT](https://arxiv.org/abs/1802.05281)
  与 [NIT treatment of resonances](https://arxiv.org/abs/2405.21072)
  — 支持“先改变多时间尺度算法，再优化 ODE kernel”的候选路线。
- [Self-force transition-to-plunge framework](https://arxiv.org/abs/2405.00170)
  — 支持把末端处理建模为渐近匹配而非简单 cutoff。
- [Eccentric-equatorial Kerr ringdown model](https://arxiv.org/abs/2603.19413)
  — 支持 light-ring 相关锚点、模式混合和 QNM beating 风险。
- [BHPT/NR surrogate across mass ratios](https://arxiv.org/abs/2204.01972)
  — 提供多保真校准和完整 IMR 代理模型的相邻案例。

### LISA、推断与工作负载

- [LISA waveform modelling review](https://arxiv.org/abs/2311.01300)
  — LISA Waveform Working Group 对各源模型能力和未解挑战的综合综述。
- [LISA Data Challenge documentation](https://lisa.pages.in2p3.fr/LDC/)
  与 [Typed LISA Toolkit waveform distinction](https://lisa-apc.pages.in2p3.fr/typed-lisa-toolkit/api/toplevel.html)
  — 区分 raw 与 projected waveform，并提供响应/数据约定背景。
- [LISA global analysis](https://arxiv.org/abs/2403.15318)
  — 支持把持续增长数据、重叠源和残余噪声纳入性能/误差目标。
- [ripple differentiable waveforms](https://arxiv.org/abs/2302.05329)
  — 为可微波形的相邻收益提供证据，但不证明 FEW 可直接照搬。
- [Neural importance sampling](https://arxiv.org/abs/2210.05686)
  — 展示用精确 likelihood 重加权和 ESS 暴露 OOD/失败的相邻范式。

### 数值、数据和工程

- [Mixed-precision numerical methods survey](https://arxiv.org/abs/2007.06674)
  — 支持按算法条件与校正机制选择精度，而非全局换 dtype。
- [Herbgrind](https://arxiv.org/abs/1705.10416) 与
  [Verificarlo](https://arxiv.org/abs/1509.01347)
  — 提供非局部浮点根因与随机算术诊断思路。
- [Reduced-order and surrogate models for GW](https://arxiv.org/abs/2101.11608)
  — 覆盖训练集、greedy、维数灾难和事后验证。
- [HDF5 chunking guidance](https://support.hdfgroup.org/documentation/hdf5-docs/advanced_topics/chunking_in_hdf5.html)
  — 说明 chunk/cache 必须按访问模式选择，错误布局会放大 I/O。
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
  — 明确同步、固定资源、线程安全与 graph replay 的约束。
- [FAIR4RS](https://www.nature.com/articles/s41597-022-01710-x)
  — 支持软件、版本和 metadata 的持久标识与可重用性要求。
- [pybhpt](https://github.com/znasipak/pybhpt)、
  [Fast Self-Forced Inspirals](https://github.com/BlackHolePerturbationToolkit/Fast_Self-Forced_Inspirals)
  与 [transition-to-plunge ancillary code](https://github.com/gcompere/SelfForceFrameworkForTransitionToPlungeWaveforms)
  — 可作为独立 oracle、数据生成和算法 PoC 的工程入口。

上述新增 PDF 已记录在 [`library/MANIFEST.tsv`](library/MANIFEST.tsv)；二进制位于被
Git 忽略的 `library/downloads/`，可按 URL 与 SHA-256 在 Mac 重建。

## 16. 当前边界

本报告是**假设 backlog**，不是实现完成声明。`A/B` 证据能证明问题在相应论文范围
内存在；把它映射到 FEW 当前模型仍需 X1--X10。尤其不能把 2026 圆 Kerr 通量结论
直接当作赤道偏心振幅表的失败判决，也不能把其他项目的 NIT/JAX/代理速度当作 FEW
速度。

当前 `deep-optimization` 仍建立在 upstream `master@47e4fea4` 的文档基线上，不含
Apple/Metal 或 CUDA mixed-compute 的生产源码血缘。本轮只丰富知识库与问题定义。
