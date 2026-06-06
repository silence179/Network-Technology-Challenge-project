# OTCP 论文详解：创新点、算法设计与代码实现

> 本文件后续大部分内容用于解释 OTCP 原理；当前工作区里的实际实验入口和集成状态，以本页下方“当前仓库集成状态（2026-05）”小节为准。

> **OTCP** — *Orbit-Predictive Topology-Aware Content Placement*  
> IEEE LCN 2026 投稿论文

---

## 当前仓库集成状态（2026-05）

当前 `algorithms/code/` 目录有两种使用方式：

1. 原生 OTCP/cache 评测入口：`python -m code.experiment ...`
2. 原始实验桥接入口：通过 `algorithms/code/project_experiment_bridge.py` 接入 `experiment1_results/experiment1_cache_routing.py`、`experiment2_results/experiment2_topology_stability.py`、`experiment3_results/experiment3_uav_relay.py`

需要特别区分：

- Experiment 1 是原生内容/缓存场景，新增 8 个方法和原始 3 个方法的命中率/回源率/完成时延最直接可比
- Experiment 2、3 中新增 8 个方法通过“持续流 -> 固定 `content_id`”映射进入原始脚本，因此更适合看成功率、恢复能力、可服务性，不宜把绝对时延当成完全同构的控制策略指标
- 如果目的是展示 OTCP 自己最严格的论文口径结果，仍应优先运行 `python -m code.experiment --mode main`

当前桥接进原始实验的 8 个方法是：

- `nocache`
- `lce_lru`
- `greedy`
- `madrl`
- `submod`
- `spacecache`
- `myopic`
- `otcp`

---

## 1. 研究背景与动机

### 1.1 场景描述

研究场景是 **LEO 卫星-无人机-地面站三层异构网络** 中的缓存放置问题：

```
            ☆ SAT_0   ☆ SAT_1   ☆ SAT_2  ...  ☆ SAT_99
           / 560km 轨道 \   (ISL: 100 Mbps)
          ↕ 卫星-地面链路 (20 Mbps) ↕
    🛩 UAV_0  🛩 UAV_1  ...  🛩 UAV_9      📡 GS (origin)
```

- **100 颗 LEO 卫星** — 560km 轨道，~96 分钟周期，7.6 km/s
- **10 架 UAV** — 内容请求发起者
- **1 个地面站 (GS)** — 内容源服务器
- 每颗卫星有**有限缓存**（C_cap = 20 个对象）

### 1.2 核心问题

UAV 从网络获取内容。若附近卫星已缓存该内容，延迟极低（单向传输）；否则需**多跳回源**到 GS，延迟极高（往返传输）。

**现有方案的不足**：
- **反应式**（LCE-LRU）— 只在内容经过时缓存，不主动放置 → 命中率低
- **贪心**（Greedy-Popular）— 每个节点独立放热门内容 → 大量冗余
- **强化学习**（MADRL-Cache）— 各节点独立 Q-learning，收敛慢、无全局协调
- **覆盖预测**（SpaceCache+）— 利用覆盖信息放置，但无时间前瞻
- **单步最优**（Myopic）— 只看当前拓扑 → 不利用轨道可预测性

### 1.3 关键观察

> **LEO 卫星轨道是确定性的、完全可预测的。**

卫星未来任意时刻的精确位置都可通过轨道力学计算。这意味着可以预知：
- 未来哪些卫星会服务于哪些 UAV
- 哪些卫星将离开服务范围（缓存失效）
- 哪些新卫星将进入范围（提前放置内容）

---

## 2. 核心创新点

### 创新点 1：扩展协作缓存集 (K' > K)

**传统方法**：只在 UAV 当前最近的 K=3 颗卫星上放内容。

**OTCP**：扩展到 **K'=8 颗卫星**，在更大范围内协作放置。

```
传统 (K=3):     UAV → [SAT_a, SAT_b, SAT_c]        → 3×20 = 60 缓存位
OTCP (K'=8):    UAV → [SAT_a, ..., SAT_h]           → 8×20 = 160 缓存位
```

更多节点参与 → 更多缓存容量 → 可覆盖更多内容 → 命中率↑

### 创新点 2：可控冗余约束 (R_max = 2)

**ILP 中的约束**：

$$\sum_{c \in \mathcal{K}'} x_{c,f}^{(\tau)} \leq R_{\max} = 2 \quad \forall f, \tau$$

每个内容在同一时间步最多被 **2 个节点** 缓存。这在保证多样性的同时允许关键内容的战略性冗余备份。

### 创新点 3：逐节点延迟加权 (δ_c)

**目标函数**：

$$\max \sum_{\tau=0}^{H} \gamma^\tau \sum_{f=1}^{F} \lambda_f \sum_{c \in \mathcal{K}'} \delta_c^{(\tau)} \cdot x_{c,f}^{(\tau)}$$

其中 δ_c 是节点 c 的延迟节省量：$\delta_c = d_{\text{miss}} - d_{\text{hit},c}$

靠近 UAV 的卫星 δ_c 大（放内容收益高），LP 自然优先在近处放置热门内容。

### 创新点 4：LP-Preserving 舍入

LP 松弛得到连续值 x ∈ [0,1]。OTCP 使用 **LP-Preserving 舍入**：

```python
# 1. 收集所有 LP 值 > 0.01 的 (节点, 内容) 对
triples = [(val * (1.0 + λ_f), c, f) for c, f if sol[x(c,f,0)] > 0.01]

# 2. 按 LP值×流行度 降序排列
triples.sort(reverse=True)

# 3. 贪心分配，同时尊重 R_MAX + 容量约束
for _, c, f in triples:
    if placed_count[f] < R_MAX and node_count[c] < C_cap:
        assign(c, f)
```

优先保留 LP 的全局最优决策（包含 H 步前瞻信息），而非用流行度覆盖 LP 结果。

### 创新点 5：滚动视野求解 (Rolling-Horizon)

每个时间步执行：
1. 预测未来 H=5 步拓扑
2. 构建多步 LP → 求解 → 取 **τ=0 层** 的放置方案
3. 下一步重新规划（MPC 风格）
4. 折扣因子 γ=0.9 自动降低远期权重

---

## 3. 代码架构

### 3.1 核心代码结构

```
paper/code/                          # 论文实验代码
│
├── config.py                        # 全局参数配置
├── experiment.py                    # 实验主控（6 种模式, 8 方案）
├── common/                          # 公共基础设施
│   ├── data_loader.py               # CSV 轨迹加载（Pandas）
│   ├── topology.py                  # cKDTree 拓扑构建 + 缓存节点选择
│   ├── metrics.py                   # Zipf 请求生成 + 路径度量
│   ├── popularity.py                # EWMA 流行度追踪
│   ├── network_utils.py             # ECEF 距离/延迟/带宽/仰角
│   └── visualization.py             # Matplotlib 字体配置
│
├── olcp/                            # OTCP 核心算法
│   ├── solver.py                    # 稀疏 LP + HiGHS 求解 + LP-Preserving 舍入
│   └── router.py                    # 全放置节点搜索路由
│
├── baselines/                       # 7 个基线方案
│   ├── no_cache.py                  # 无缓存（纯 Dijkstra 回源）
│   ├── lce_lru.py                   # LCE-LRU（经典 NDN 缓存）
│   ├── greedy_popular.py            # 贪心流行度放置
│   ├── drl_actor_critic.py          # MADRL-Cache [Zhong et al., TCCN 2020]
│   ├── spacecache_plus.py           # SpaceCache+ [Fang et al., INFOCOM 2024]
│   ├── submodular_greedy.py         # 子模贪心 + 多样性保证
│   └── myopic.py                    # 单步 LP（H=0, K'=8）
│
└── ns/                              # 网络仿真
    ├── experiment.py                # NS 实验主控
    └── simulator.py                 # SimPy DES 仿真器
```

### 3.2 config.py — 全局参数

| 参数 | 值 | 说明 |
|------|------|------|
| SAT_DIR | `sat_trace_100/` | 100 颗卫星轨迹目录 |
| K | 3 | 路由缓存节点数 |
| K' | 8 | LP 扩展集大小 |
| H | 5 | 规划视野步数 |
| C_cap | 20 | 单节点缓存容量 |
| R_max | 2 | 可控冗余上限 |
| M_bud | 3 | 每步迁移预算 |
| F | 100 | 内容目录大小 |
| α | 1.5 | Zipf 偏度 |
| γ | 0.9 | 折扣因子 |

### 3.3 solver.py — OTCP LP 求解器（核心）

#### 变量索引

```
x[c,f,τ] = c×F×T + f×T + τ        # 放置决策变量
m[c,f,τ] = n_x + c×F×H + f×H + τ  # 迁移指示变量
总变量数 = C×F×T + C×F×H           # 典型: 8×100×6 + 8×100×5 = 8800
```

#### LP 约束

| 约束 | 公式 | 含义 |
|------|------|------|
| 容量 | Σ_f x[c,f,τ] ≤ C_cap | 每节点每步最多 20 项 |
| 可控冗余 | Σ_c x[c,f,τ] ≤ R_max=2 | 每内容每步最多放 2 份 |
| 迁移指示 | -m + x(τ) - x(τ-1) ≤ 0 | 追踪新增内容 |
| 迁移预算 | Σ_f m[c,f,τ] ≤ M_bud=3 | 每节点每步最多新增 3 项 |
| 初始迁移 | Σ_{新增} x[c,f,0] ≤ M_bud | 从当前状态到 τ=0 |

约束矩阵大小约 5000×8800，非零元素仅 ~50000（密度 ~0.1%），使用稀疏矩阵 + HiGHS 内点法求解。

### 3.4 experiment.py — 实验主控

8 方案对比：`['nocache', 'lce_lru', 'greedy', 'madrl', 'submod', 'spacecache', 'myopic', 'otcp']`

每步流程：
1. 构建未来 H 步拓扑快照
2. 各方案运行放置决策
3. 生成 Zipf 请求 → 路由 → 记录命中/延迟/流量
4. 更新流行度追踪器

6 种实验模式：`main | ablation | scale | zipf | capacity | all`

### 3.5 其他模块

| 文件 | 功能 |
|------|------|
| data_loader.py | 从 CSV 加载卫星/UAV 轨迹 |
| topology.py | cKDTree 空间索引，O(N log N) 拓扑构建 |
| metrics.py | Zipf 请求采样，路径完成时间/流量计算 |
| popularity.py | EWMA 流行度追踪（decay=0.95） |
| network_utils.py | ECEF 距离、仰角、链路带宽 |
| router.py | OTCP/Myopic 路由：搜索所有已放置节点 |

---

## 4. 实验结果

### 4.1 主实验（N=100, α=1.5, C_cap=20, 8 方案对比）

| 方案 | 延迟 (s) | σ (s) | 流量 (GB) | 命中率 | 多样性 |
|------|----------|-------|-----------|--------|--------|
| No-Cache | 26.9 | 12.0 | 39.06 | 0.0% | 0.0 |
| LCE-LRU | 6.7 | 4.4 | 15.36 | 84.1% | 0.0 |
| Greedy-Pop | 8.0 | 6.4 | 16.53 | 76.9% | 19.0 |
| MADRL-Cache | 6.6 | 5.2 | 15.27 | 81.9% | 83.3 |
| Submod-Greedy | 5.9 | 3.1 | 11.98 | 92.7% | 66.7 |
| SpaceCache+ | 6.0 | 3.2 | 12.33 | 91.4% | 59.3 |
| Myopic-Opt | 5.8 | 3.4 | 11.56 | 94.1% | 83.0 |
| **OTCP** | **5.5** | **2.6** | **10.05** | **99.9%** | **99.4** |

**关键分析**：
1. **OTCP vs No-Cache** — 延迟 ↓79.6%，命中率 99.9%
2. **OTCP vs Myopic** — 命中率 +5.8pp，证明多步前瞻的价值
3. **OTCP vs SpaceCache+** — 命中率 +8.5pp，LP 全局优化优于覆盖贪心
4. **OTCP vs MADRL-Cache** — 命中率 +18.0pp，确定性最优优于试错学习
5. **OTCP 多样性 99.4/100** — 近乎完美的内容目录覆盖

### 4.2 消融实验（规划视野 H）

| H | 命中率 |
|---|--------|
| 0 (Myopic) | 94.4% |
| 1 | 96.2% |
| 3 | 98.5% |
| 5 (默认) | **99.9%** |

**核心发现**：多步时间前瞻是性能的主要驱动力，即使 H=0 仍达 94.4%（超越所有基线），H≥5 后收益饱和。

### 4.3 可扩展性（25~200 颗卫星）

OTCP 在所有规模下稳定领先。LP 求解时间随卫星数近似线性增长。

### 4.4 Zipf 敏感性 (α=1.1~2.2)

- α↑ → 需求集中 → 所有方案命中率 ↑
- OTCP 在低 α（需求分散）时优势最大：分散需求下协作规划更关键

### 4.5 容量敏感性 (C_cap=5~40)

- C_cap↑ → 所有方案命中率 ↑
- OTCP 在低 C_cap 时优势最大：受限容量下 LP 全局优化更重要


## 5. 论文结构

| 章节 | 文件 | 核心内容 |
|------|------|----------|
| §1 Introduction | introduction.tex | 动机、3 个贡献 |
| §2 Related Work | related_work.tex | LEO 缓存综述 + 方案对比表 |
| §3 System Model | system_model.tex | 三层网络模型 + ILP 建模 + NP-hard 证明 |
| §4 OTCP Algorithm | proposed_method.tex | LP-Preserving 舍入 + (1-1/e) 近似 |
| §5 Evaluation | evaluation.tex | 8 方案对比 + 图表 + 消融/敏感性 |
| §6 Conclusion | conclusion.tex | 总结与展望 |

---

## 6. 实验图表

| 图 | 文件 | 内容 |
|----|----|------|
| Fig.1 | experiment_results.png | 主实验 4 子图（延迟/流量/命中率/回源率） |
| Fig.2 | convergence.png | 8 方案命中率收敛曲线（滑窗 10 步） |
| Fig.3 | delay_cdf.png | 延迟累积分布函数 |
| Fig.4 | cache_diversity.png | 缓存内容多样性时序图 |
| Fig.5 | ablation_horizon.png | H=0,1,3,5 消融对比 |
| Fig.6 | scale_comparison.png | 25/50/100/200 卫星规模对比 |
| Fig.7 | zipf_sensitivity.png | Zipf α=1.1~2.2 敏感性 |
| Fig.8 | capacity_sensitivity.png | C_cap=5~40 敏感性 |

---

## 7. 项目文件结构

```
S3/
├── README.md                        # 本文件
├── s3.py                            # S3 仿真入口
│
├── paper/                           # 论文源文件
│   ├── main.tex                     # 主 LaTeX 文件（IEEE LCN 格式）
│   ├── bilingual.tex                # 双语版（xelatex 编译）
│   ├── main.pdf                     # 编译输出
│   ├── bilingual.pdf                # 双语版 PDF
│   ├── main.bbl                     # 参考文献编译输出
│   │
│   ├── sections/                    # 论文各章节
│   │   ├── introduction.tex
│   │   ├── related_work.tex
│   │   ├── system_model.tex
│   │   ├── proposed_method.tex
│   │   ├── evaluation.tex
│   │   └── conclusion.tex
│   │
│   ├── references/
│   │   └── refs.bib                 # 参考文献数据库
│   │
│   ├── figures/                     # 实验图表（9 张 PNG）
│   │
│   ├── code/                        # 实验代码（见§3）
│   │   ├── config.py, experiment.py, leoem_exp.py, leoem_ns_exp.py
│   │   ├── common/                  # 公共模块
│   │   ├── olcp/                    # OTCP 核心算法
│   │   ├── baselines/               # 7 个基线
│   │   └── ns/                      # SimPy DES 仿真
│   │
│   ├── results/                     # 实验结果 JSON
│   │   ├── metrics.json             # 主实验
│   │   ├── ns_metrics.json          # NS 实验
│   │   ├── leoem_metrics.json       # LeoEM 跨平台验证
│   │   ├── ablation_metrics.json    # 消融实验
│   │   ├── scale_metrics.json       # 可扩展性
│   │   ├── zipf_metrics.json        # Zipf 敏感性
│   │   └── capacity_metrics.json    # 容量敏感性
│   │
│   └── LeoEM/                       # LeoEM 仿真器（子项目）
│       ├── route_stage/             # 路由阶段（utility.py 被导入）
│       ├── precomputed_paths/       # 预计算 Starlink ISL 路径
│       ├── constellation_params/    # 星座参数
│       ├── emulation_stage/         # 网络仿真阶段
│       └── StarPerf_MATLAB_stage/   # MATLAB 性能分析
│
├── algorithms/                      # 独立算法实现
├── traces/                          # 卫星/UAV 轨迹数据
│   ├── uav_trace_full.csv
│   ├── sat_trace/                   # 默认轨迹
│   ├── sat_trace_50/                # 50 卫星轨迹
│   ├── sat_trace_100/               # 100 卫星轨迹
│   ├── sat_trace_150/               # 150 卫星轨迹
│   └── sat_trace_200/               # 200 卫星轨迹
│
├── test/                            # 算法测试 & 复现
├── outputs/                         # 基准测试输出
├── experiment1_results/             # 实验 1 结果
└── docs/                            # 文档
```

---

## 8. 编译 & 运行

### 编译论文

```bash
cd paper/
# 主论文（需编译 4 次以解析引用）
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
# 双语版
xelatex bilingual.tex && xelatex bilingual.tex
```

### 运行实验

```bash
cd paper/code/
python experiment.py           # 主实验（默认配置）
python leoem_exp.py            # LeoEM 跨平台验证（路由阶段）
python leoem_ns_exp.py --steps 200  # LeoEM 网络仿真验证
```

---

## 9. 总结：为什么 OTCP 有效

| 设计决策 | 效果 | 数据支撑 |
|----------|------|----------|
| 扩展集 K'=8 | 160 位 vs 60 位，覆盖全目录 | Myopic→OTCP: +5.8pp 命中率 |
| 可控冗余 R_max=2 | 多样性 + 战略性复制 | diversity: 99.4/100 |
| Per-node δ_c | 近处优先放热门内容 | avg delay: 5.5s（最低） |
| LP 全局优化 | 比贪心做更好的权衡 | vs SpaceCache+: +8.5pp |
| LP-Preserving 舍入 | 整数化保留全局最优信息 | 99.9% 命中率 |
| 稀疏矩阵+HiGHS | 毫秒级求解 | ~155ms/步，远小于 6s 决策间隔 |
| 滚动视野 H=5 | 利用轨道可预测性 | H=0→5: +5.5pp 命中率 |
