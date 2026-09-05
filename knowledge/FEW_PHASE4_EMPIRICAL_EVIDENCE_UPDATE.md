# FEW 第四阶段问题审计：用当前主线和 5× 原型校正问题地图

<!-- 2026-09-04 18:35 CST (linux): Add a current-branch evidence update after
reviewing the three stashed audits against master@47e4fea4, the 5.09 GB Kerr
data product, RTX 2080 Ti profiling, CPU/CUDA regression tests, and current
primary literature.  This document changes evidence status; it does not claim
that the listed scientific questions are resolved. -->

## 0. 本轮最重要的新认识

前三轮的 142 个候选问题大体成立，但今天的实现与测量把几个问题从“猜测”升级成了
可量化事实，也否证了若干过于简单的说法：

1. **混合精度确实可能超过 5×，但只在合适工作负载上。** 一年、约 210 万点的
   Kerr 波形由 `152.4895 ms` 降到 `26.1811 ms`，同进程加速 `5.824×`；约 2 千和
   2 万点的短任务只有 `1.692×`、`1.822×`。因此“FEW 加速多少倍”不是单一数字。
2. **真正有效的改动不是把所有 FP64 改成 FP32。** 获胜路径保留 FP64 轨迹、相位
   样条与相位约化，用 FP32 保存/插值振幅、计算三角函数和块内累计，并利用
   `+m/-m` 共轭结构减少工作量。算法融合与数据表示至少和 dtype 同样重要。
3. **高吞吐通过不等于数值或科学等价。** 七个物理区域的平坦内积 mismatch 都小于
   `3.234e-12`，模式集合完全一致；但严格逐点误差为 `7.73e-7`--`5.18e-6`，没有
   通过既有 `5e-10` 门槛。两种指标回答不同问题。
4. **高偏心比本轮高自旋点更敏感。** `e0=0.8` 是七点扫描的最坏项。这支持按局部
   参数/条件数选择精度，而不是按模块固定“振幅一律 FP32”。
5. **当前 5× 后的新瓶颈已经转移。** 同步阶段剖析中，编译求和 kernel 约
   `6.40 ms`，CPU 轨迹约 `15.48 ms`。继续只优化 GPU 求和的 Amdahl 上限迅速降低，
   下一轮更应研究轨迹算法、批量多源和端到端 likelihood。
6. **测试和数据治理问题会直接污染精度判断。** 五点振幅测试实际只断言最后一点；
   `(2,2,0,0)` 参考值与未修改 master 的 FP64 结果也相差约 `0.408`。若只看绿色测试，
   这个矛盾会完全不可见。

所以，当前 FEW 的根问题可以浓缩为：

```text
没有一个把“科学用途—物理模型—数据产品—数值政策—硬件计划—验收证据”
绑定成同一语义对象的端到端契约。
```

## 1. 十二层问题的当前证据状态

| 层次 | 已确认问题 | 尚不能下的结论 | 下一份决定性证据 |
| --- | --- | --- | --- |
| L0 科学任务 | `fast/accurate/GPU` 未绑定时长、SNR、PSD/TDI、cold/warm 或用途 | 平坦 mismatch 小不能自动批准参数估计 | search、forecast、unbiased inference 三档契约 |
| L1 物理系统 | 相对论快速模型仍不是完整 generic Kerr/环境/次级自旋族 | 范围边界不是现有模型“错误” | 每个模型的机器可读支持域与退化极限 |
| L2 近似阶次 | 当前 Kerr 快速模型是绝热波形；不能靠更高浮点精度补回 1PA 物理 | 本轮后端误差远小于平坦 mismatch，不代表远小于真实模型误差 | 独立 1PA/Teukolsky reference ladder |
| L3 波形表示 | 模式先全算后选择；模式阈值不是全局 mismatch 最优 | 圆 Kerr flux 的 `lmax>=30` 结论不能直接套到偏心 strain 表 | flux/strain 两份模式收敛账本 |
| L4 离线数据 | 5.09 GB 表缺少生成版本、原始数据、许可证、局部误差等内嵌 model card | SHA256 只证明字节身份，不证明科学适用性 | 可重建 data bundle、blocked holdout 与逐区误差 |
| L5 数值模拟 | ODE、样条、事件、模式和求和没有统一误差预算；高偏心放大 FP32 误差 | 一个逐点阈值或一个 mismatch 都不能单独代表所有用途 | 分阶段扰动、相消指标、相位/TDI/参数偏差传播 |
| L6 探测器统计 | 当前混合精度只验证源波形平坦内积，没有 LISA TDI/PSD | `mismatch<1e-10` 不是完整 LISA 验收 | 固定响应/epoch/PSD 的 A/E/T 或 X/Y/Z 验证 |
| L7 计算算法 | 5× 来自精度、共轭融合和减少工作共同作用；轨迹已成最大阶段 | 峰值 FP32/FP64 比不能预测端到端加速 | operation/byte/launch/sync 账本与批量曲线 |
| L8 硬件执行 | 一年任务过 5×，短任务未过；结果只在 RTX 2080 Ti/cc7.5 验证 | 不能外推到 M3、Ampere/Hopper、不同驱动或持续热负载 | 第二 CUDA 架构、Mac、能耗/热稳定 holdout |
| L9 软件实现 | 可变调用状态、`assert` 输入校验、native `exit(0)`、粗粒度 backend 都有静态证据 | 尚未量化每项对用户科学结果的发生率 | 状态纯度、异常重放、并发与故障注入矩阵 |
| L10 验证发布 | 公开 CI 不跑真实 CUDA；存在空断言/错断言；Schwarz CUDA 路径实测失败 | CPU 测试通过不能证明发布 CUDA wheel 正确 | 四级 CI：接口、CPU、真实 GPU、离线科学门 |
| L11 数据治理 | 下载超时留下 2.32 GB 同名残片；版本隔离缓存会重复大文件 | 单机手工清理不解决安装/升级原子性 | `.part`+续传+容量预检+bundle 原子激活 |

## 2. 第四轮新增问题 Q143--Q166

证据标记沿用前三轮：`A` 为 FEW/EMRI 一手结果，`C` 为数值/HPC 方法证据，`R` 为
当前仓库或本机实测，`H` 为仍需验证的推论。

| ID | 问题 | 证据 | 最小可证伪实验 |
| --- | --- | --- | --- |
| Q143 | 速度比是 workload 函数；只报一年波形的 5.82× 会误导短调用/Agent 用户 | `R` | 固定 cold/warm、点数、模式数、batch 的性能曲面和 P95 |
| Q144 | 当前候选在高偏心区逐点误差约为普通区域数倍，统一混合精度政策不稳健 | `R` | 对 `a,e,Δp,视角,模式` 分层随机扫描并训练保守 fallback 指示器 |
| Q145 | 平坦 overlap 对幅度微扰不敏感，可与 `1e-6` 逐点误差同时出现 | `R/C` | 加入 LISA PSD/TDI、SNR 偏差、Fisher 投影和 injection-recovery |
| Q146 | `+m/-m` 融合依赖精确的共轭/符号约定；共享实现的后端一致可能一起错 | `R` | 用解析对称性和独立逐模式复数求和作 oracle，覆盖非零 `k,n`/逆行/相消 |
| Q147 | FP32 把 GPU 常驻表从约 4.95 GiB 降至 2.52 GiB，但 cold 构造仍为秒级 | `R` | 单独报告进程启动、文件校验、HDF5 读、H2D、首波形与 steady state |
| Q148 | 下载器直接以 `wb` 写最终文件；超时会留下看似正式的截断文件 | `R` | 网络中断/磁盘满/kill 注入；要求最终路径只出现完整校验后的文件 |
| Q149 | 文件缓存按 FEW 版本分目录，同一 5 GB 哈希可重复占盘并重复下载 | `R` | 内容寻址对象库+版本清单 PoC，测升级、回滚和并发准备 |
| Q150 | 当前 Kerr 五点振幅测试的断言在循环外，实际上只检验最后一点 | `R` | 修正测试结构，并先解决当前第三点 master/reference 冲突 |
| Q151 | `(2,2,0,0)` 表格参考与 untouched master FP64 差约 `0.408` | `R` | 用原始生成器或独立 Teukolsky 代码重算该点，审计模式/相位/数据版本 |
| Q152 | 数据预处理脚本首行导入不存在的 `AmpInterpKerrEqEcc`，无法按仓库原样重跑 | `R` | 在干净环境执行预处理 smoke test，并锁定 raw-data manifest |
| Q153 | 大 HDF5 根属性只有 `lmax/mmax/nmax`，没有生成 commit、约定、误差或许可证 | `R` | 定义 data model card schema，并验证文件+sidecar 的语义完整性 |
| Q154 | master Kerr CUDA 曾因 SciPy 接收 CuPy 数组失败；公开 CPU CI 没发现 | `R` | 真实 NVIDIA nightly，从安装 wheel 到 Kerr 长/短波形全路径 |
| Q155 | Kerr--Schwarz CUDA 回归仍在 ROMAN normalization 的 CuPy→SciPy 转换失败 | `R` | 把 host/device 边界显式化，并要求两个相对论模型的 CUDA 对照通过 |
| Q156 | 当前分支为探索保留多个负收益 kernel 变体，扩大 ABI、编译和维护面 | `R` | 收敛前按证据表保留/删除，只让最终政策进入公共 API |
| Q157 | 候选是在同一主机和主要基准上迭代选出的，存在 benchmark overfitting/winner's curse | `C/R` | 预注册门槛，在未见参数、第二 GPU 和不同 workload 上盲测一次 |
| Q158 | 七点扫描证明覆盖方向，不足以估计薄边界区的失败概率 | `C/R` | 科学先验、边界强化和对抗采样分别报告 P50/P95/P99.9/失败率 |
| Q159 | 当前误差只比较相同轨迹上的后端路径，未分离数据生成、插值和物理阶次误差 | `R` | 建立 oracle ladder：独立数据点→FP64 插值→轨迹→strain→TDI |
| Q160 | 一年生产计时为 26.18 ms，插桩计时约 30.12 ms，instrumentation 本身改变关键路径 | `R/C` | Nsight/CUPTI 外部 timeline 与轻/重插桩交叉校准 |
| Q161 | GPU 求和降到 6.40 ms 后，15.48 ms CPU 轨迹构成下一阶段 Amdahl 下限 | `R` | 比较批量多轨迹、NIT 重叠子域、代理轨迹与仅移植 RHS 的收益/相位 |
| Q162 | 论文中 circular-flux `1e-6` 误差可接受的结论不能直接批准本次 eccentric-strain FP32 | `A/R` | 在论文相同 SNR/时长/PSD/推断设置与偏心扩展上分别复验 |
| Q163 | 自动模式集合相同只能排除离散选择分叉，不能排除每个模式误差相关叠加 | `R/C` | 保存逐模式误差、相位、功率和相消条件数，再与总 strain 关联 |
| Q164 | 单 GPU 重复 bitwise 不能证明跨编译器、架构和 fast-math 的确定性 | `C/R` | 驱动/toolkit/arch/math-mode factorial test，区分 bitwise 与科学复现 |
| Q165 | 性能报告没有能耗和温度；移动端“快”可能以降频、内存压力或电池为代价 | `H` | 30--60 分钟持续负载，报告 waveform/J、P95、温度和 page fault |
| Q166 | 5× 原型加快了源波形，却未证明每个有效后验样本或 global-fit 迭代加快 | `H` | 计入 response、FFT、PSD inner product、likelihood 与 sampler 的有效样本/s |

### 2.1 低成本动态复现结果

<!-- 2026-09-04 18:55 CST (linux): Upgrade selected phase-three static
findings using the untouched master wheel.  The probe deliberately stops at
contract behavior and does not infer downstream scientific impact. -->

[`multilayer_failure_probe.json`](../collaboration/linux/multilayer_failure_probe.json)
在 `master@47e4fea4` 的独立 wheel 上确认：

- Kerr `sanity_check_init` 接受 `m1=NaN`、`m2=NaN` 和 `m1=m2`，但能拒绝负 `m2`；
  这把 Q86--Q87 从纯静态阅读升级为前置校验行为事实。
- mode-policy helper 接受 `-0.1`、`1.1`、`NaN` 和 `Infinity` 阈值。七次调用都打印
  “override include-minus to True”，但返回的 effective 值仍是 `False`；Q103--Q104
  得到动态证据。
- 在 `run_inspiral` 内注入可控异常后，`generating_trajectory` 停留在 `True`，没有由
  `finally` 恢复；Q102 得到动态证据。
- 当前预处理脚本所导入的 `AmpInterpKerrEqEcc` 符号不存在；Q152 可在不接触原始数据
  前稳定复现。
- AST 检查给出参考点循环为第 62--66 行，紧随其后的断言在第 67 行且不属于循环；
  Q150 不再只是人工缩进判断。

这些结果证明的是“接口允许或状态残留”，尚未证明它们已在某次已发表分析中造成
错误。下一步应以 validity firewall、异常后重放和 full-call property tests 量化影响。

## 3. 被实测否证或必须降级的说法

1. **“FP64 换 FP32 就会接近 32×”——否证。** 32 是特定 GPU 的峰值吞吐比，
   不包含访存、轨迹、HDF5、模式选择、launch、转换与最终累计。本轮长任务为 5.82×，
   短任务不足 2×。
2. **“混合精度只有 dtype 选择”——否证。** 单独的 FP32 相位、递推和 warp reduction
   都没有达到目标；共轭代数融合才把核心 kernel 从约 14.75 ms 降至 6.40 ms。
3. **“高自旋一定是最坏点”——本轮不支持。** 七点扫描里 `e0=0.8` 最差；这不证明
   全域高偏心永远最差，只说明风险排序要靠地图而非直觉。
4. **“后端 mismatch 很小，所以科学安全”——必须降级。** 这里只是未加权源波形，
   尚未经过 LISA 响应、PSD、SNR 与参数偏差。
5. **“测试通过说明五个参考点正确”——否证。** 循环外断言和第三点冲突证明测试
   语义本身需要审计。
6. **“lazy/loading 或 FP32 解决了移动端部署”——必须降级。** 容量降低不等于 cold
   latency、并发、热稳定、下载可靠性和 Agent 配额已经解决。

## 4. 更深的跨层故障链

### 4.1 快速但不可用于推断

```text
FP32 振幅/块累计
 -> 高偏心局部误差增大
 -> 平坦 overlap 仍极好
 -> 未经过 LISA PSD/Fisher 方向
 -> 被误标为“科学等价”
 -> 高 SNR posterior 可能沿敏感方向偏移
```

### 4.2 数据完整但语义错误

```text
SHA256 正确
 -> HDF5 缺少生成/约定 model card
 -> 测试只断言最后一个参考点
 -> 某模式参考与当前数据版本不匹配
 -> CPU/CUDA/Metal 共用数据而一致
 -> 共同错误被当成跨平台正确
```

### 4.3 Agent 自动重试放大失败

```text
首次调用隐式下载 5 GB
 -> 5 秒 read timeout
 -> 最终路径留下 2.32 GB 残片
 -> 版本缓存未复用已有同哈希文件
 -> Agent 重试再次下载/占盘
 -> 计算请求变成不可预测的网络与存储故障
```

### 4.4 优化后瓶颈转移却继续优化旧热点

```text
求和 kernel 降至 6.40 ms
 -> CPU 轨迹约 15.48 ms 成为最大阶段
 -> 继续追求更低精度求和
 -> 科学误差上升但端到端收益趋近饱和
 -> 忽略 NIT/批量/likelihood 级算法杠杆
```

## 5. 当前优先级：不是继续盲目降精度

| 顺序 | 工作包 | 通过条件 |
| --- | --- | --- |
| Z1 | **语义与用途契约** | 每条结果声明模型阶次、域、数据 hash、requested/effective execution plan、frame/epoch/单位 |
| Z2 | **独立 reference 修复** | 重算矛盾振幅点；五点测试真正逐点断言；oracle lineage 非 FEW 自循环 |
| Z3 | **LISA 科学门** | 固定 fastlisaresponse、二代 TDI、PSD、SNR；报告 mismatch、SNR 与参数偏差 |
| Z4 | **precision sensitivity atlas** | 全域分层/对抗采样；高偏心与相消指标；保守 FP64 fallback 的 coverage |
| Z5 | **真实 GPU CI** | CUDA wheel 安装、Kerr/Schwarz、CPU 对照、显存/重复调用、真实数据哈希全部持续通过 |
| Z6 | **原子数据供应链** | 显式 `plan/prepare/verify`；容量预检、断点、`.part`、校验后原子激活、内容寻址复用 |
| Z7 | **状态与失败语义** | 调用顺序不变；异常/OOM/取消后对象可验证地复用或进入 poisoned 状态；禁止 `exit(0)` |
| Z8 | **新瓶颈研究** | 在重叠物理子域比较 NIT/批量轨迹；以 phase/TDI/runtime 决策，不只测 RHS kernel |
| Z9 | **API/实现收敛** | 负收益实验留报告但退出公共 ABI；最终 fast policy 可审计、可回退、默认仍 FP64 |
| Z10 | **端到端 workload** | cold/P95、batch、能耗及完整 likelihood/effective-sample throughput |

Z1--Z5 是把今天的性能里程碑变成可信科学功能的最短路径。Z6--Z10 决定它能否成为
Mac、CUDA、服务和 Agent 都可持续使用的工程系统。

## 6. 一手证据与本仓库证据

### 论文

- [Kerr eccentric-equatorial FEW model, Phys. Rev. D 112 (2025)](https://arxiv.org/abs/2506.09470)
  — `|a|<=0.999`、`e<0.9`、`p<200`、约 100 ms 硬件加速及大部分域约 `1e-5`
  LISA mismatch；同时明确这是绝热模型。
- [Systematic errors in fast relativistic EMRI waveforms, PRD 113 (2026)](https://arxiv.org/abs/2509.08875)
  — 离线 flux 截断、插值、Chebyshev 与参数偏差；其 `1e-6` 结论限于四年圆 Kerr
  flux/SNR `O(100)` 设置，不能直接外推到本轮偏心 strain 精度。
- [Fast inspirals and orbital resonances](https://arxiv.org/abs/2405.21072)
  — NIT 与共振区部分平均切换在其 self-force toy-model 范围报告至少两数量级加速，
  支持改变数学问题优先于继续压低单个 kernel。
- [Original FEW framework paper](https://arxiv.org/abs/2104.04582)
  — 模块化、ROMAN、模式约简、GPU 和后验偏差的基础设计与早期范围。

### 当前分支的可复验材料

- [`CUDA_MIXED_PRECISION_5X_PLAN.md`](CUDA_MIXED_PRECISION_5X_PLAN.md)
  — 性能契约、精度分层和已达到/未达到的门槛。
- [`mixed32_kerr_probe.json`](../collaboration/linux/mixed32_kerr_probe.json)
  — 同进程短/中/一年性能、内存、误差、源与数据哈希。
- [`mixed32_accuracy_sweep.json`](../collaboration/linux/mixed32_accuracy_sweep.json)
  — 七区域模式集合、逐点/L2/平坦 mismatch 和五个参考点。
- [`pipeline_stage_probe.json`](../collaboration/linux/pipeline_stage_probe.json)
  — 轨迹、模式选择、样条、kernel 和不可归因时间。
- [`multilayer_failure_probe.json`](../collaboration/linux/multilayer_failure_probe.json)
  — untouched master 上的非有限/等质量输入、模式政策、异常状态、预处理导入与测试
  AST 结构复现。
- [`files/manager.py`](../src/few/files/manager.py) 与
  [`files/registry.yml`](../src/few/files/registry.yml) — 下载、版本缓存和哈希行为。
- [`amplitude-data-preparation.py`](../dataset-preprocessing/KerrEccEq/amplitude-data-preparation.py)
  — 当前离线振幅表整理脚本及其缺少可执行/语义供应链的证据。
- [`tests/test_amplitudes.py`](../tests/test_amplitudes.py)、
  [`tests/test_detector_wave.py`](../tests/test_detector_wave.py) 与
  [`tests/test_mode_selector.py`](../tests/test_mode_selector.py) — 已确认的测试语义缺口。

## 7. 当前审计边界

本文件把今天可复验的结果加入前三轮问题地图，不把 `H` 类假设写成已发生事故。
七个参数点不是全域统计，平坦 mismatch 不是 LISA 验收，RTX 2080 Ti 不是所有 CUDA
架构，master FP64 也只是实现 oracle 而非真实信号。反过来，严格逐点门槛未过也不等于
候选在所有科学用途都不可接受；是否可接受必须由 Z1--Z4 的任务化科学证据决定。

当前分支仍为 `codex/cuda-mixed-precision-5x@47e4fea4`，Ubuntu 持有编辑锁；源码、
测试、知识和报告尚未提交。前三轮审计从 stash 原样迁入以保留历史血缘，本文件才是
对当前 CUDA 原型的增量状态说明。
