# 本地参考工程与高性能加速源码索引

<!-- 2026-09-03 CST: Curated reference repositories for GPU acceleration, precision arithmetic, and scientific computing -->

本目录克隆并归档了与 FEW 引力波加速、信号处理和数值精度高度相关的优秀开源工程代码。此目录已被 `.gitignore` 排除，不会增加主工程 Git 仓库负担。

---

## 1. 工程清单与核心价值

| 参考工程 | 仓库地址 | 主要语言 | 核心解决问题 / 对 FEW 的借鉴价值 |
| :--- | :--- | :--- | :--- |
| **`fastlisaresponse`** (`lisa-on-gpu`) | [mikekatz04/lisa-on-gpu](https://github.com/mikekatz04/lisa-on-gpu) | Python / CuPy / CUDA | **LISA 探测器时延干涉（TDI）GPU 端即时响应计算**。展示了如何通过 CuPy 和自写 CUDA 内核完成卫星轨道岁差、时间延迟插值与多通道 TDI 投影，实现波形生成到探测器响应的零拷贝传输。 |
| **`QD`** | [BL-highprecision/QD](https://github.com/BL-highprecision/QD) | C++ / CUDA | **高精度双单（Double-Double / Double-Single）和四单（Quad-Double）算法库**。提供了无误差浮点变换（TwoSum, TwoProd）、高精度初等函数（sin/cos/sqrt/exp）的工业级模板实现，是我们在 GPU 上进行高精度补偿求和的核心参考。 |
| **`sleef`** | [shibatch/sleef](https://github.com/shibatch/sleef) | C / SIMD | **极致性能的可向量化初等数学函数库**。针对 AVX-512、ARM NEON 和 GPU 提供了 1.0 ULP / 3.5 ULP 精度保障的紧凑三角函数与多项式求值内核，对 FEW 底层自旋加权球谐函数与相位三角函数加速具有重大指导意义。 |

---

## 2. 关键代码路径与查阅建议

### 2.1 `fastlisaresponse/fastlisaresponse/`
* `response.py`: 继承自 CuPy / NumPy 统一后端的探测器响应类基类，展示了与 FEW 相同风格的 `BackendLike` 双后端抽象。
* GPU 上的时间重采样与切比雪夫多项式逼近实现。

### 2.2 `QD/include/qd/`
* `dd_inline.h` / `dd_real.h`: 双精度模拟（106-bit 尾数）基础算术运算，包括 `two_sum`、`quick_two_sum`、`two_prod` 的纯 inline C++ 实现。
* 可直接将算法思想移植为 CUDA `__device__` 内部函数用于模式求和。

### 2.3 `sleef/src/libm/`
* 各类高精度矢量化 `sin`, `cos`, `exp`, `log` 的有理逼近和多项式展开表。
