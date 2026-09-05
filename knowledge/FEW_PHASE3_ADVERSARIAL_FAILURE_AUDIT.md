# FEW 第三阶段反例审计：当结果看起来正常时，它还能怎样错

<!-- 2026-09-04 13:20 CST (linux): Create a third-pass adversarial audit from
current-branch source evidence and source-verified literature. It searches for
plausible-but-wrong outputs, circular validation, history-dependent likelihoods,
coordinate/convention failures, and operational faults. Production source/tests
and pre-existing untracked probes remain untouched. -->

<!-- 2026-09-04 18:39 CST (linux): Restore this adversarial audit from the
deep-optimization stash onto the current codex CUDA branch without rewriting
its historical claims.  Use the phase-four update for current evidence. -->

## 0. 新结论

前两轮找到了层次问题和三条跨层断层。第三轮故意反过来问：

> 如果程序不崩溃、波形看起来平滑、CPU/GPU 也一致，结论还能怎样是错的？

答案集中在四个更深的结构性问题：

1. **约定即模型**：质量是源坐标还是红移质量、输出是 source/SSB/TDI 哪一层、
   复应变是 `h+ - i hx` 还是其他约定，都会改变科学结果，但不一定改变数组的
   “正常外观”。
2. **有效域不是一个矩形**：它是物理阶次、参数坐标、数据版本、软件路径、
   硬件能力和使用任务的交集。“输入通过 `if` 检查”不等于“输出在已验证域内”。
3. **优化策略是似然的一部分**：模式切换、动态精度、近邻缓存和 fallback 若依赖
   历史或资源压力，同一参数点可能得到不同似然，进而破坏 MCMC/HMC 的前提。
4. **验证也会过拟合**：同一代码生成 fixture、同一似然模型做 self-injection、
   反复调到通过固定 benchmark，都可以让错误模型通过所有“自洽”检查。

因此还需要两个与数值精度并列的正确性条件：

```text
call-history invariance: 给定完整语义输入，结果不应依赖之前调用
validation independence: 参照的物理、代码、数据或统计信息必须至少有一项独立
```

## 1. 审计方法与证据等级

每个候选问题使用五个反例检查：

1. 能否输出平滑、有限、可画图，但物理语义错误？
2. CPU/CUDA/Metal 能否一致，但因共用公式/数据而一起错？
3. 单点/平均指标能否通过，但尾部、边界或人群结论失败？
4. 参数等价变换、事件、模式或 fallback 能否使函数不连续/不可微？
5. 一次失败、取消或 OOM 能否污染后续调用？

证据标记沿用第二阶段：`A` 为 FEW/EMRI 直接证据，`B` 为 LISA/GW
一手需求或相邻实证，`C` 为数值/HPC/统计方法，`R` 为当前仓库静态证据，
`H` 为待验证假设。`R` “确认”的是实现事实，不自动确认其科学影响。

## 2. 约定、坐标与等价类

| ID | 新问题 | 证据 | 最小可证伪实验 |
| --- | --- | --- | --- |
| Q75 | API 的 `m1/m2` 只写“太阳质量”，未在结果中强制记录 source-frame 还是 detector-frame/redshifted mass。LISA EMRI 科学研究明确把内秉参数称为红移质量，需距离/宇宙学才能反推真实质量。 | `B/R` | 用同一 source mass 在两个红移生成明确转换的 reference，检查 FEW 调用者能否不经手工约定得到同一 detector-frame 相位/振幅 |
| Q76 | 代码使用 `M=m1+m2` 和约化质量 `mu=m1*m2/M`；自作用力展开通常以中心体/小天体质量与质量比组织阶次。在有限质量比下，这种变量选择等价于某种默认 resummation，需成为模型语义。 | `A/R/H` | 固定物理系统，比较 central/small-mass 与 total/reduced-mass 两种展开在 `q→0` 和有限 `q` 的阶次差 |
| Q77 | `source`/`detector` 标签同时覆盖源极化、SSB 旋转和 AAK 长波近似；注释又明确说后者“不是 TDI”。两值枚举无法表达实际观测层级。 | `R` | 让 API 输出 `source polarization → SSB polarization → one-way link → TDI` 的 typed stages，与现接口交叉检查 |
| Q78 | 时变 LISA 响应需要绝对起始 epoch 与时标；FEW 核心输入主要给相对 `T/dt`。同一源在不同星座位相启动不是同一 detector waveform。 | `B/R` | 在 fastlisaresponse/LDC 中仅改 epoch，测各 TDI 通道变化；确认 result envelope 能否唯一重建时间轴 |
| Q79 | SSB 极化旋转处有“尚不理解为何需要”的源码 TODO，而 detector-frame 测试只执行函数，没有数值断言。符号、偏振角和 `h×` 约定尚无独立 oracle。 | `R` | 与 LDC/fastlisaresponse 建立若干解析方位的极化 golden vectors，断言逐分量数值而非不崩溃 |
| Q80 | 天球极点的方位角本来无定义；AAK 路径把 `qS/qK` 自动移到 `1e-6`，这是坐标补丁而非物理操作，并会引入人为不连续。 | `R` | 用 Cartesian unit vectors/quaternion 表示旋转，绕极点小圆扫描新旧波形的连续性与坐标不变性 |
| Q81 | `xI<0` 会触发 `a,xI,qK,phiK,Phi_phi0` 联合变换。这是参数等价类的 canonicalization；对采样器却可能成为多对一映射，导致先验双计数、Jacobian 丢失或 `xI=0` 梯度断裂。 | `R/H` | 在变换前后比较波形不变性；再用对称先验检查归一化、可逆性和边界有限差分 |
| Q82 | 反向积分、`flip_output`、初相位翻转和时间/相位对齐共同构成一个 gauge 契约。分别测试“曲线看起来相同”不能保证与似然的时间原点一致。 | `R/H` | 构造 forward/backward/flip 群合律 metamorphic test，连同绝对 epoch 进入 TDI 后检查等价 |
| Q83 | `T` 使用恒星年，`dt` 使用秒，输出长度通过 `int(T*YRSID_SI/dt)` 向下取整。实际终止时间、最后一个样本与零填充语义未在结果中显式返回。 | `R` | 扫描 `T/dt` 正好整除与边界两侧，断言 sample count、首末时刻、FFT 频率栅格和跨工具对齐 |
| Q84 | 裸 `float` 输入承载 `Gpc`、太阳质量、秒、年和无量纲坐标；裸 complex array 又依赖 `h+ - i hx`。Agent 或跨库组装时，单位/符号错误不会被 dtype 发现。 | `R/H` | 在不破坏 NumPy 快路径前提下加 typed boundary adapter，对 Mpc/Gpc、year/s 和两种 complex convention 做故障注入 |
| Q85 | TDI 输入/输出可表示相位、频率或分数频率；文献已显示对频率量直接套用标准时延组合会由臂长变化产生不可接受残余。“通道名相同”不等于观测量相同。 | `B` | 为每个 TDI fixture 记录 observable/unit/delay convention，复现有无 Doppler scaling 的激光噪声残余 |

## 3. 物理有效域与数据束

| ID | 新问题 | 证据 | 最小可证伪实验 |
| --- | --- | --- | --- |
| Q86 | Schwarzschild/Kerr 相对论基类主要检查 `m1>=m2`，因而 `m1=m2` 在静态逻辑上可通过；只有 Pn5AAK 显式警告 `m2/m1>1e-4`。一个摄动 EMRI 模型可对非 EMRI 输入返回平滑但无物理保证的数组。 | `R/A` | 逐模型扫描 `q=1e-7...1`，记录 reject/warn/result；将软件限制与论文校准域做机器可读差集 |
| Q87 | 多处检查使用 `<`/`>`，未统一先做 `isfinite`；`NaN` 对大小比较为 false，可穿过质量、距离、时间或角度检查，直到更深层才生成无上下文的 NaN/异常。 | `R` | 对所有公开标量输入注入 `NaN/±Inf/-0.0`，要求在分配大数组或启动 GPU 前结构化拒绝 |
| Q88 | `a==0`、`abs(xI)==1` 等精确浮点比较与 `a→1e-6`、极点角移动等自动 clamp 并存：有的微小偏差被拒绝，有的却被静默改成另一个物理系统。 | `R` | 为每个边界声明 reject/canonicalize/clip 策略，保存 requested/effective 参数并做两侧连续扫描 |
| Q89 | 轨迹通量表、振幅表、谐波基和探测器响应各有有效域。当前轨迹停止主要参考 flux/separatrix 边界，不保证整条轨迹都在振幅/响应验证域交集内。 | `R/H` | 在积分前做 domain-intersection plan，与现有“边跑边失败”比较失败点、费用和误报/漏报 |
| Q90 | flux HDF5、amplitude HDF5、ROMAN 权重、基矩、归一化和模式索引是一个数值产品束。逐文件 hash 正确仍可能组合出语义不兼容的新旧混合束。 | `R/H` | 为全束定义一个 signed model manifest 和交叉维度/约定检查，故意交换一个旧文件确认必须在数值调用前失败 |
| Q91 | 耗散 flux 与辐射振幅都来自 Teukolsky 模式，但在 FEW 中由不同数据产品供应轨迹和 strain。两者可各自插值很准，却不满足同一模式截断下的能量/角动量 balance。 | `A/R/H` | 在小型高精度子域从振幅模式重建 flux，与轨迹表做 mode-by-mode 及总量平衡账本 |
| Q92 | 环境、次级自旋、非 Kerr 修正和 1PA 并非必然可线性相加；两个各自 `O(epsilon)` 的 correction 会产生阶次可能同样重要的交叉项，而插件顺序可能不对易。 | `A/H` | 对两个 toy correction 比较 `A∘B`、`B∘A`、线性和与含交叉项 reference，以阶次计数决定允许的组合 |
| Q93 | 先验支持集若大于模型验证域，“域外拒绝”会隐式改变先验归一化；“fallback”则把一个先验分成多个物理模型。两者都不是普通异常处理。 | `B/H` | 显式计算 prior mass 在各有效域的比例，比较 reject/truncate/fallback 对 evidence 和 posterior 归一化的影响 |
| Q94 | 几何上在数据包围盒内不代表有局部训练支持。近极端自旋 surrogate 研究显示外推可否接受依赖探测器灵敏度和训练域，不是一个通用布尔值。 | `B/C` | 保留整块 withheld islands 和边界带，按科学任务校准 coverage，不用随机行拆分作唯一证据 |
| Q95 | 轨迹提前触发 separatrix/数据边界后，求和器可以零填充到请求时长。下游若不知道“物理信号已结束”与“请求观测窗”的差别，FFT/似然会把模型终止解释为真实静默。 | `R/H` | 返回 termination reason/time 和 valid mask，对零填充、截断、taper 三种处理比较谱泄漏与似然 |
| Q96 | 仓库内存在 generic-Kerr 示例类，其数据路径包含工作目录硬编码与 TODO，但不在 stock waveform 列表中。“可 import”、“实验性”和“可用”的边界需机器可读，否则 Agent 可以组装到未发布路径。 | `R` | 生成 public capability inventory，从每个可导入组件尝试构造，要求返回 stable/experimental/internal 及所需数据 |

## 4. 数值路径、事件和不可微面

| ID | 新问题 | 证据 | 最小可证伪实验 |
| --- | --- | --- | --- |
| Q97 | 观测角由 `arccos(-dot(R,S))` 生成，当前未把浮点内积 clip 到 `[-1,1]`。近对齐单位向量的末位越界可将完全有效的几何变成 NaN。 | `R` | 生成近平行/反平行角并在不同 libm 上扫描，检查 clip 前后有限性与角度误差 |
| Q98 | 共振、数据边界、separatrix 和 plunge 的事件时刻随参数变化。可微积分器若只对选中的平滑路径求导，而不对 event time/guard/reset 求导，梯度在事件面附近错误。 | `C/A` | 用带参数事件时间的 toy EMRI 对比解析、有限差分、连续 adjoint 和 event-aware discrete adjoint |
| Q99 | 有限差分步长可跨过 canonicalization、模式集变化、ODE 接受步分叉或终止事件。“对某个步长与 AD 接近”不是导数收敛证据。 | `C/R/H` | 对每个参数做多步长 convergence plot，同时标记模式/步序列/事件结构是否改变 |
| Q100 | `ELQ↔pex` Jacobian 、距 separatrix 坐标变换和 PN-normalized flux 都可能在某些区域病态。只看输出误差不能区分数据误差与坐标变换放大。 | `A/R/H` | 计算变换 Jacobian 的局部条件数地图，将 flux 扰动在变换前后传播并与实际轨迹误差相关 |
| Q101 | ODE 右端有路径用 NaN 表示越界，其他路径的 NaN 则表示数值失败。若不保留产生者和原因，后续自适应减步、停止或报错可能处理错类。 | `R` | 用 typed status 替代内部 NaN sentinel 做对照原型，故障注入 domain exit、溢出、坏数据和用户 NaN |
| Q102 | `run_inspiral` 在积分前设置 `generating_trajectory=True`，正常返回前才恢复，未使用 `finally`。积分异常可将对象留在错误模式，后续调用依赖历史。 | `R` | 在积分中途故障注入，捕获异常后复用同对象，与新对象的状态/输出比较 |
| Q103 | mode selector 对非 list 且 `include_minus_mkn=False` 时警告“已 override 为 True”，但静态路径没有实际赋值；后续可在 `mode_arr=None` 上取索引。这是警告语义与状态不一致。 | `R` | 构造 `threshold/all/None × include_minus=False` 最小矩阵，断言 effective policy 和返回模式，不只断言 warning |
| Q104 | `mode_selection_threshold` 在该路径未明显检查 finite 或 `[0,1]`；公式中又对 `(1-threshold)` 平方，负值或大于 1 的值可返回“合法”但无语义的模式集。 | `R` | 对边界、域外、NaN/Inf 做 API property test，拒绝后再检查阈值单调性 |
| Q105 | TD/FD 路径的窗、零填充、单/双边谱、FFT 归一化和起止样本只要有一项不同，即使同一物理波形也会表现出假 mismatch 或 SNR 差。 | `A/R/C` | 从同一冻结 TD 信号建立 Parseval/逆变换 golden test，把窗和 padding 放入 fixture identity |
| Q106 | 复振幅的相位有 branch cut；直接对 amplitude/phase 分解后插值或学习，可在越过 `±pi` 或振幅过零时制造虚假不连续。分别报告 amplitude 和 phase RMSE 可能比直接复数误差更差。 | `C/H` | 标出模式零点/分支切割，比较 Re/Im、log-amplitude/unwrapped-phase 和局部复平面表示 |

## 5. 验证的循环论证与统计陷阱

| ID | 新问题 | 证据 | 最小可证伪实验 |
| --- | --- | --- | --- |
| Q107 | `test_detector_wave.py` 的主路径生成两种 waveform 后丢弃返回值，没有数值、shape、finite 或约定断言。它只能证明当前样例没有抛异常。 | `R` | 加独立极化/SSB golden vectors、距离缩放和旋转不变性，确认故意改符号时测试会失败 |
| Q108 | pickle 测试只在反序列化后调用，未与序列化前的配置/输出比较；源码 `__reduce__` 还自述为会丢自定义 args/kwargs 的 band-aid。分布式 worker 可静默回到默认模型。 | `R` | 对每个非默认实例对比 pickle 前后 capability、文件、后端、阈值、dtype ledger 和输出 |
| Q109 | CPU/CUDA/Metal 的同输入一致可排除 kernel 分叉，却无法发现共享的质量约定、极化符号、数据表或模式索引错误。 | `R/C` | 每个验收表增加 `oracle_lineage`，至少一列为解析极限、独立 Teukolsky/self-force 实现或独立 LISA 响应 |
| Q110 | Simulation-based calibration 能验证“生成器+似然+采样器”的计算自洽性；若 injection 和 recovery 共用同一 FEW 近似，排名图再完美也不检验模型相对真实信号的偏差。 | `C/H` | 将测试分成 self-SBC 和 cross-model SBC；后者用更高保真/独立模型注入并分类计算失效与模型失配 |
| Q111 | 如果反复查看同一参数网格并调阈值，该网格已成为训练集而不是测试集；同时尝试许多 kernel/精度候选只报最快者，会产生 winner's curse。 | `C/H` | 预注册门槛与候选数，保留未见参数、新 workload 和第二台硬件作最终 holdout |
| Q112 | 平均/中位 mismatch 可以掩盖薄的共振带、近边界和强相消时刻的灾难尾部。应报告受分布测度的 failure probability 与条件分位数，而不只是最大值或均值。 | `A/C/H` | 对科学先验、均匀几何和边界强化三种 measure 分别报 P50/P95/P99.9/失败率与置信区间 |
| Q113 | posterior 变化很小可能因为先验或噪声主导，不是波形近似很准；反之，很小 mismatch 也可沿某个高 Fisher 灵敏方向产生明显偏差。 | `A/B` | 同时报告 prior-to-posterior information gain、bias/statistical-error ratio 和 waveform-error projection，并扫描 SNR |
| Q114 | 快速模型若在参数空间非均匀地改变检测效率，即使每个已检测事件的 posterior 看起来可接受，selection function 也会偏置黑洞质量函数和事件率。 | `B/H` | 用两种保真度对同一 population 注入做 search/recovery，将效率差带入层次人群推断 |
| Q115 | 单事件内小于 `1 sigma` 的同向偏差可在多事件组合中积累，甚至制造虚假非 GR 迹象；高 SNR “golden source”可能更脆弱。 | `B` | 对独立和相关模型误差生成递增 catalog，画 P-P plot 与组合 GR/人群参数偏差随事件数的缩放 |
| Q116 | global fit 中灵活的噪声/前景模型可以吸收波形残差，让 residual PSD “看起来正常”，代价是噪声、邻源或校准参数被偏置。残差白化不是独立通过证据。 | `B/H` | 在固定与可变噪声模型下注入结构化 waveform error，同时追踪源、噪声、前景和残差参数 |
| Q117 | 用 Gaussian-process/discrepancy basis 边缘化模型误差可减少偏差，但若训练差异不包含真实误差方向，额外方差只是不正确的安全感。 | `B/C` | 用已见误差、withheld 模型差和结构外误差三层 injection 检查 posterior coverage 与方差校准 |
| Q118 | ROMAN 误差估计器/OOD 分类器若与主模型共用训练分布和表示，可能对同一未知区域同时过度自信。可靠 fallback 需要独立覆盖或保守上界，不只是另一个网络。 | `C/H` | 设计 withheld islands、坐标变形和对抗搜索，测试不确定度 coverage 而非分类 AUROC |

## 6. 调用历史、并发与运行时可靠性

| ID | 新问题 | 证据 | 最小可证伪实验 |
| --- | --- | --- | --- |
| Q119 | waveform 调用会 `update` 实例的 `inspiral_kwargs`，所以本次参数可成为下次默认值。在 notebook 里是意外状态，在多线程服务里则是请求间污染。 | `R` | 对 `A(config1)→B(default)` 与新实例 `B(default)` 做结果/内部状态对照，再交错并发 A/B |
| Q120 | 缓存键若只包含轨道参数，漏掉 model/data hash、后端、精度、模式政策、epoch、response/PSD 或软件版本，会把旧语义结果当作新结果。 | `R/H` | 定义 semantic cache key schema，逐字段扰动并确认只有经证明的不变性允许 cache hit |
| Q121 | 近邻参数缓存或自适应精度若使用滞回/LRU/当前资源来决定返回近似，似然可依赖访问顺序。这不只是可复现问题，还可破坏 Markov 链转移核所假设的目标密度。 | `C/H` | 对同一参数集随机重排调用，要求结果逐点等价；再用 toy MCMC 测 detailed balance/posterior |
| Q122 | 现有 `__reduce__` 策略可丢失自定义构造参数；多进程运行时“可 pickle”不代表 worker 拿到同一模型。 | `R` | 为序列化内容加 semantic identity，在 spawn 子进程中与主进程做双向 identity/数值对照 |
| Q123 | h5py 官方明确建议每个 reader process 独立打开文件；在打开 HDF5 后 `fork` 会继承内部状态并可出问题。惰性常驻 handle 与服务器 pre-fork 模型有直接冲突。 | `C/R` | 覆盖 `open-before-fork`、`open-after-spawn`、多 reader 和反复销毁，验证数值、死锁、fd 泄漏和 RSS |
| Q124 | HDF5 默认构建不必线程安全；其 thread-safe 构建也可用全局锁串行化 API。“多线程不崩溃”和“并行读有吞吐收益”必须分开验证。 | `C` | 记录 HDF5/h5py build flags，在 1/2/4/8 threads 下测安全性、锁等待、IOPS 和 P95，与多进程分别比较 |
| Q125 | CUDA/Metal 调用在 OOM、取消或异步 kernel 失败后，memory pool、stream、event 或对象状态可能不再可复用。简单 catch-and-retry 可用受污染上下文再次返回错误。 | `C/H` | 在各阶段故障注入 allocation/kernel/copy/cancel，比较复用与全新 session，建立明确 reset/poison 状态机 |
| Q126 | capability fallback 若在不同主机因数据、内存、dtype 或驱动而选择不同模型，即使 API 返回成功，Mac 与 Ubuntu 也已不在计算同一个科学对象。 | `R/H` | 对每个 fallback 故障注入，要求 result 中包含 requested/effective plan 与改变的科学门槛；默认不允许静默改模型 |
| Q127 | 文件 hash 不覆盖内存/计算中的瞬发或永久 GPU 故障。故障注入研究显示控制/调度单元错误可形成 silent data corruption，NaN/Inf 检查只覆盖很小一类后果。 | `C` | 长运行中抽样重算、保存不变量/输出 checksum，记录 ECC/XID/硬件健康，对关键阶段做软件故障注入 |
| Q128 | 异步 API 返回的 device view 若在用户消费前被 workspace 复用，或 Python 对象析构先于 kernel 完成，可产生时序相关错误。强制全局同步可修复，却抹掉异步性能。 | `C/H` | 定义 result/future 所有权和 stream/event dependency，用延迟消费、及时 GC 与 workspace 压力测试 |
| Q129 | 自动调优若依据当前温度、内存压力或背景负载选算法/精度，调优结果可在一次推断中漂移。调优可改调度，但不应未记录地改数值政策。 | `C/H` | 冻结 numerical plan 后单独允许 launch/batch tuning，连续热运行检查 output identity 与 latency 漂移 |
| Q130 | 多 worker 同时根据“当前空闲内存”通过 admission，之后可集体超额并 OOM；单请求资源预测正确不代表服务容量正确。 | `H` | 用中央 token/semaphore 和乐观无锁 admission 做突发对照，测 OOM、队列 P95、取消和公平性 |

## 7. 离线数据、学习系统与技术决策

| ID | 新问题 | 证据 | 最小可证伪实验 |
| --- | --- | --- | --- |
| Q131 | 网格相邻点/同一轨迹的样本高度相关；随机行拆分会让验证集与训练集几乎重合，夸大 surrogate/interpolator 泛化。 | `C/H` | 比较 random-row、blocked-region、leave-one-spin-band-out 和 leave-one-trajectory-out 误差分布 |
| Q132 | active learning 依靠当前 error/acquisition model 选下一点；若它对某种分支、共振或尖峰结构盲区，可在错误区域永远不采样。 | `C/H` | 保留固定比例的空间覆盖/随机探索，用隐藏窄峰函数与真实高曲率区检验发现率 |
| Q133 | OOD 不只在几何边界外；新的物理修正、数据生成器版本或噪声/响应分布都可把原域内点变成 semantic OOD。 | `B/C/H` | 分别施加 covariate、label/physics 和 convention shift，不确定度必须能拒绝至少定义的 shift 类型 |
| Q134 | HDF5 压缩、FP32 存储、系数量化或分块低秩的误差不应按字节均匀分配。强模式、相位灵敏区、边界和强相消项的一 bit 价值不同。 | `C/H` | 在固定字节预算下比较均匀量化与灵敏度加权量化，用 TDI/参数偏差而非系数 RMSE 决胜 |
| Q135 | 一份 query trace 可优化 chunk/cache/shard，也可过拟合某个 sampler、先验或单源路径。部署到 population/global-fit 后的访问几何可完全不同。 | `C/H` | 将 trace 按 search、MCMC、population、cold Agent 分层，保留一类 workload 作数据布局 holdout |
| Q136 | “与表格节点一致”只验证在线插值；离线 Teukolsky/self-force 生成器本身的分辨率、边界条件和模式截断误差需要独立账本。 | `A/C` | 使用高精度独立节点分解 generator error 与 interpolator error，再注入轨迹测交叉项 |
| Q137 | 磁盘文件 hash 验证之后，页缓存、host/device 拷贝、长驻 GPU 权重和 workspace 仍可损坏。长服务只在启动时验证文件不足以建立端到端数据完整性。 | `C/H` | 为不变权重/切片保存廉价运行时指纹，抽样重传/重算并做 bit-flip 故障注入 |
| Q138 | 更新模型时若先安装 registry，再下载权重/HDF5，中途失败可留下跨版本束。原子文件重命名不够；需要原子的束级 activation pointer。 | `R/H` | 在每个安装阶段 kill -9/空间耗尽，重启后必须看到完整旧束或完整新束，不能混合 |
| Q139 | 在同一主机、数据尺寸和热缓存上反复选最快实现，会把计时噪声、编译器特例和大小特化当成算法优越性。 | `C/H` | 分离 tuning/validation workloads，预定最小效应量，用多大小、cold/warm、不同主机和重复分布复验 |
| Q140 | 新数据表、训练和跨主机验收有巨大离线成本。如果模型使用量不足、很快被更高物理阶次取代，单次在线加速不一定偿还总成本或能耗。 | `H` | 建立 lifecycle cost：生成+验证+分发+存储+在线调用+迁移，对调用量/模型寿命做 break-even 曲线 |
| Q141 | 失败的优化和未通过的参数域若只留在对话/临时脚本，Mac/Linux/未来 Agent 会反复消耗成本，甚至只传播最终“成功”幸存者。 | `R/H` | 为每个候选记录 tested domain、negative result、退出理由和证据 hash，搜索新实验是否命中已知反例 |
| Q142 | 长周期参数推断中途更新 FEW、数据表或 response/PSD 会把一条链变成多个目标分布的混合。“使用最新版”对运行中分析是错误策略。 | `R/C` | 分析启动时冻结 semantic bundle ID；故意更新环境，必须拒绝继续或开启有明确 bridge/reweight 的新分析 |

## 8. 十条可能静默通过的故障链

1. **质量约定链**：source mass 当 detector mass → 相位平滑 → 后端一致
   → 宇宙学/质量推断偏置。
2. **非 EMRI 链**：`m1>=m2` 通过 → 摄动模型返回有限数组 → 画图正常
   → 用户将计算外推到 IMRI/等质量。
3. **极化链**：角度 canonicalization/极点 clamp → TODO 旋转 → 只测不崩溃
   → `h×` 符号或偏振角错误进入 TDI。
4. **可微链**：参数扰动跨过模式/事件面 → AD 只对当前分支求导 →
   局部梯度看似精确 → HMC/Fisher 走向错误方向。
5. **历史似然链**：近邻缓存/动态精度滞回 → 访问顺序改变返回值 →
   同一参数不再对应固定似然 → MCMC 验证前提失效。
6. **自洽验证链**：同一 FEW 生成 injection/reference → CPU/GPU 一致 →
   self-SBC 均匀 → 共同物理/约定错误仍完全隐藏。
7. **终止链**：轨迹进入未覆盖域 → 提前停止 → 零填充到请求 `T` →
   FFT/噪声模型将突然消失解释为物理信号。
8. **数据束链**：每个文件 hash 通过 → 权重/基/归一化版本不匹配 →
   网络输出有限且稳定 → 整个 amplitude 模型系统性偏移。
9. **pre-fork 链**：主进程惰性打开 HDF5 → fork worker 继承库状态 →
   低负载正常 → 并发时偶发错读/卡死。
10. **catalog 链**：每个源的小同向偏差 → 单源小于统计误差 →
    多源组合后系统量增长 → 人群或 GR 检验得到高显著性假结论。

## 9. 优先级：先堵住“平滑但错”

| 顺序 | 实验 | 为什么现在做 | 通过条件 |
| --- | --- | --- | --- |
| Y1 | **semantic input/result contract** | 质量、单位、frame、epoch、complex/TDI 约定是所有测试的前提 | 机器可读 schema + 一个端到端实例 + 约定故障注入 |
| Y2 | **validity firewall** | 非 EMRI、NaN/Inf 和无效阈值可在昂贵调用前拒绝 | 公开输入矩阵、requested/effective 参数、逐模型校准域 |
| Y3 | **state-purity/exception replay** | 历史污染会让任何 benchmark 和似然不可信 | 调用顺序置换不变；所有故障后复用对象等价于新对象 |
| Y4 | **frame/polarization oracle** | 当前 detector test 无数值断言 | 独立 golden vectors，覆盖极点、对齐、正/逆行与 TD/FD |
| Y5 | **domain and model-bundle preflight** | 避免跑到一半才发现 amplitude/response 越界或文件束不兼容 | 运行前输出交集域、估算资源和唯一 bundle ID |
| Y6 | **flux–amplitude balance ledger** | 将两个各自准确但彼此不一致的数据产品暴露出来 | 独立小域 mode-by-mode 与总量 balance，误差传播到相位/TDI |
| Y7 | **discontinuity/gradient atlas** | 模式、事件和 canonicalization 会决定 AD/HMC 是否成立 | 多步长导数收敛 + 分支标记 + event-time derivative |
| Y8 | **independent cross-model calibration** | self-injection 不能发现共同模型错误 | 至少一个独立 oracle/cross-model SBC，区分计算与物理失效 |
| Y9 | **selection/catalog toy study** | 单源 mismatch 不足以保护人群与 GR 结论 | 检测效率图 + catalog P-P plot + 相关误差扫描 |
| Y10 | **concurrency chaos matrix** | lazy HDF5、pickle、CUDA context 与 worker 模式存在交互 | fork/spawn/thread、OOM/cancel/crash 后无死锁、泄漏或语义漂移 |
| Y11 | **runtime integrity and health** | 启动文件 hash 不覆盖长运行故障 | 抽样重算 + 不变量 + 设备健康 + fault-injection detection rate |
| Y12 | **held-out decision protocol** | 防止在同一样例上同时过拟合精度和速度 | 预注册指标，保留参数/workload/hardware holdout，公开所有候选含负结果 |

Y1--Y4 不需要先改加速 kernel，却决定后面的速度和误差是否有意义。

## 10. 本轮新证据

### EMRI/LISA 语义与观测量

- [LISA EMRI science study](https://arxiv.org/abs/1703.09722) — 明确区分红移内秉质量与
  需由光度距离辅助推回的真实质量，并说明 EMRI 人群价值。
- [Frequency-domain TDI adaptation](https://arxiv.org/abs/2103.06976) — 表明 TDI
  的观测量单位与时变臂长缩放不是可忽略的实现细节。

### 模型误差与统计验证

- [Gaussian-process marginalization of waveform uncertainty](https://arxiv.org/abs/1412.3657)
  与 [GPR waveform surrogate](https://arxiv.org/abs/1903.09204) — 为显式模型误差分布提供相邻方法。
- [Simulation-based calibration](https://arxiv.org/abs/1804.06788) — 验证复杂贝叶斯计算自洽性；
  本报告进一步区分 self-SBC 与独立模型验证。
- [Population bias from waveform error](https://arxiv.org/abs/1504.02767) 与
  [accumulating errors in GR tests](https://arxiv.org/abs/2210.04769) — 支持“单事件可接受”
  不保证 catalog/人群/强引力结论可接受。
- [Near-extremal-spin surrogate extrapolation](https://arxiv.org/abs/2208.02927) — 表明
  surrogate 外推门槛必须与训练域和探测器任务共同声明。

### 事件、并发与硬件可靠性

- [Differentiable parameter optimization with state-dependent events](https://arxiv.org/abs/2605.05395)
  — 说明 event time、guard/reset 和固定事件顺序是梯度有效性的明确条件。
- [h5py parallel HDF5 guidance](https://docs.h5py.org/en/stable/mpi.html) — 建议每个
  reader process 独立打开文件，避免打开后 fork。
- [HDF5 thread-safety design](https://portal.hdfgroup.org/documentation/hdf5/latest/thread-safe-lib.html)
  — 显示 thread-safe 构建的全局序列化约束。
- [GPU control-unit permanent-fault study](https://arxiv.org/abs/2306.10856) 与
  [NVIDIA GPU memory diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/reference/diagnostics/plugins/memory.html)
  — 支持将运行时数据完整性与启动文件 hash 分开。

新增的 10 份 PDF 已记录在 [`library/MANIFEST.tsv`](library/MANIFEST.tsv)；本地文献库
现共 65 份、171,399,038 字节，二进制文件仍由 Git 忽略。

### 当前仓库静态证据

- [`waveform/waveform.py`](../src/few/waveform/waveform.py) — frame/canonicalization/极化旋转及 TODO。
- [`waveform/base.py`](../src/few/waveform/base.py) — 质量/距离缩放与 per-call kwargs 持久化。
- [`utils/baseclasses.py`](../src/few/utils/baseclasses.py) — 逐模型有效域、质量比警告差异和角度约定。
- [`trajectory/integrate.py`](../src/few/trajectory/integrate.py) 与
  [`trajectory/ode/base.py`](../src/few/trajectory/ode/base.py) — 轨迹状态、NaN sentinel 与 pickle band-aid。
- [`utils/modeselector.py`](../src/few/utils/modeselector.py) — 阈值、正负模式政策与 warning/state 不一致候选。
- [`tests/test_detector_wave.py`](../tests/test_detector_wave.py) — 当前 frame/pickle 测试只验证可执行性。

## 11. 当前边界

本轮新增 Q75--Q142，共 68 个反例候选。其中 Q79、Q86--Q88、Q97、
Q102--Q104、Q107--Q108、Q119、Q122 有当前分支直接静态证据；
“对科学结论已造成多大影响”仍需 Y1--Y12 动态验证。

报告没有把所有 warning、TODO 或域边界都称为生产 bug。它们被记录的原因是：
在高 SNR、长相位、global fit、梯度推断或 Agent 服务条件下，这些边界可从
“一般技术债”变成静默改变结论的故障链。

当前分支仍为 `deep-optimization@23eef90c`，不包含 Apple/Metal 或 CUDA
mixed-compute 的生产源码血缘。本轮只增加知识、问题定义与可证伪实验；
没有修改生产源码或测试。
