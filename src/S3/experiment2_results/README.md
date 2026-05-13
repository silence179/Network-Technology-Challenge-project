# 实验二：动态拓扑稳定性验证（关键加分项）

## 文档状态（2026-05）

当前版本的 `experiment2_topology_stability.py` 已将新增 8 个内容/缓存 baseline 接入原始实验，最终会在同一份 `metrics.json` 中输出 11 个方法：

- 原始稳定性方法：`baseline1`、`baseline2`、`your_method`
- 桥接接入方法：`nocache`、`lce_lru`、`greedy`、`madrl`、`submod`、`spacecache`、`myopic`、`otcp`

这里的新增方法通过 `algorithms/code/project_experiment_bridge.py` 适配进拓扑稳定性场景：每条 `UAV -> GS_01` 持续流会映射到一个固定 `content_id`，再用当前/未来拓扑快照求放置并解析路径。因此，`success_rate`、`route_flaps`、`avg_recovery_steps` 是当前版本最适合跨族比较的指标；`avg_delay_ms` 与 `jitter_ms` 仍保留各自实现的原生语义。

为避免新增算法在该场景里因为“固定 5 条流 + 容量足够”而全部收敛到同一结果，当前版本额外引入了：

- 时变热点内容窗口：每条流不再永远固定一个 `content_id`
- 共享热点周期：多个 UAV 会阶段性竞争同一批热门内容
- 业务时限：超过时限的内容服务记为失败，从而把 hit / miss 的差异反映到成功率和恢复指标上

## 实验目标

验证"内容-拓扑协同路由"在**动态变化网络**中的稳定性优势，核心命题：

> **你的方法在 '变化网络' 中更稳** — 面对卫星快速移动、UAV 掉线、链路随机中断等拓扑扰动，稳定性感知路由能显著降低路由振荡、减少抖动、加快恢复。

---

## 实验设置

### 拓扑扰动设计（时间相关性故障模型）

| 扰动类型 | 实现方式 | 参数 |
|----------|---------|------|
| 卫星快速移动 | 使用真实 100 卫星轨迹 CSV（sat_trace_100），拓扑每步自然变化 | — |
| UAV 能耗掉线 | 基于能量模型：UAV 能量持续消耗，低能量时高概率退出 | `UAV_DROPOUT_PROB = 0.08` |
| UAV 重连 | 掉线 UAV 以概率重新上线（能量恢复） | `UAV_REJOIN_PROB = 0.35` |
| 链路相关性故障 | 故障概率与历史相关：`prob *= (1 + 0.6 × past_failures)`；故障后进入 8 步脆弱期（3× 乘数） | `LINK_FAILURE_BASE = 0.05` |
| 扰动起始步 | 前 15 步不施加扰动（系统稳定后再测试） | `PERTURBATION_START = 15` |

### 仿真参数

| 参数 | 值 |
|------|-----|
| 时间步数 | 200 步 |
| 通信任务 | 5 条 UAV → GS_01 持续流 |
| 随机种子 | 42（可复现） |

---

## 对比算法

### Baseline 1: 含噪 Dijkstra 全局重算（模拟分布式路由收敛抖动）
- **策略**: 每步全量重算最短路径，但边权增加 ±8% 随机噪声模拟分布式路由协议的收敛不一致
- **缺陷**: 噪声导致路由振荡 → 频繁路径切换；每次路径变更有 55% 概率丢包（SDN 流表更新延迟）
- **无稳定性意识**: 不保留历史信息，不区分稳定链路与脆弱链路

### Baseline 2: 响应式重路由
- **策略**: 沿用当前路径直到断裂，才被迫重算（plain Dijkstra 无噪声）
- **缺陷**: 2 步检测 + 重算延迟期间全部丢包；无预测能力，不提前感知链路风险
- **无备选路径**: 路径断裂后从零开始寻路，恢复缓慢

### Your Method: 稳定性感知 + 预维护备选路径 + 主动切换
- **链路稳定性追踪**: 维护每条链路近 10 步的稳定性评分（指数衰减 α=0.85，有历史相关惩罚）
- **K-Shortest 备选路径**: 维护 K=8 条候选路径，选路权重 `delay × (1.8 − 0.8 × stability)`
- **主动健康监测**: 当当前路径最弱链路稳定性 < 0.50 时，主动切换到更稳定路径（要求改善 ≥0.05）
- **延迟就近备选**: 切换时优先选择延迟接近当前路径的备份（60% 稳定性权重 + 40% 延迟接近度）
- **即时故障切换**: 路径断裂时零延迟切换到预缓存备选路径 → 恢复时间最短

### 新增 8 个内容放置 baseline（桥接接入）

新增方法来自 `algorithms/code/`：

- `nocache`
- `lce_lru`
- `greedy`
- `madrl`
- `submod`
- `spacecache`
- `myopic`
- `otcp`

它们不是原始实验里的“路径控制器”，而是通过桥接层把持续流映射为内容请求，并把求得的放置结果还原成当前步可用路径。因此这部分结果更适合解释“在动态拓扑下是否还能持续服务、是否更容易恢复”，而不是替代原始三方法的控制策略分析。

---

## 评估指标

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| **Route Flaps** | 路由重构次数 | 活跃路径发生变化的次数 |
| **Success Rate** | 任务完成率 | 数据包成功送达比例 |
| **Jitter** | 时延抖动 | Gap-aware 计算：中断恢复后延迟跳变被加权放大 |
| **Recovery Time** | 中断恢复时间 | 从路径失效到重新建立可用路径的步数 |

---

## 当前集成结果（11 方法）

当前 `metrics.json` 的核心摘要如下。为了避免混淆，这里优先展示跨族最稳妥的 3 个指标：

| 方法 | Route Flaps | Success Rate | Avg Recovery (steps) |
|------|-------------|--------------|----------------------|
| Baseline1 | 653 | 36.5% | 3.32 |
| Baseline2 | 385 | 45.2% | 3.41 |
| Your Method | 339 | 72.7% | 2.64 |
| NoCache | 0 | 0.0% | 0.00 |
| LCE-LRU | 152 | 16.2% | 6.31 |
| Greedy | 147 | 15.5% | 6.35 |
| MADRL | 331 | 34.4% | 3.12 |
| Submod | 577 | 60.9% | 2.32 |
| SpaceCache | 509 | 54.5% | 2.47 |
| Myopic | 597 | 63.4% | 2.39 |
| OTCP | 471 | 76.1% | 2.71 |

### 当前版本的读法

1. 在原始三方法里，`your_method` 仍然是最稳的一条主线：成功率最高、恢复也最快。
2. 在桥接接入的新增方法里，`otcp` 现在明显高于其他新增方法，`myopic/submod/spacecache` 构成第二梯队，`lce_lru/greedy` 在动态热点 + 时限约束下明显吃亏。
3. `nocache` 在这个版本里被清晰压成了 0% 成功率，说明“固定流 + 超大容忍度”造成的结果塌缩已经被消掉了。
4. `avg_delay_ms` 与 `jitter_ms` 在原始三方法和新增八方法之间不是严格同构的时延定义；如果要看绝对时延解释，请直接查看 `metrics.json` 并结合桥接语义阅读。

---

## 代码架构

```
experiment2_topology_stability.py
│
├── 基础设施
│   ├── load_traces() / get_nodes()
│   ├── build_topology_graph()
│   └── ecef_distance(), propagation_delay_ms(), ...
│
├── 扰动引擎 — PerturbationEngine
│   ├── UAV 能耗模型（能量消耗/恢复/掉线）
│   ├── 时间相关性链路故障（历史加权概率 + 脆弱期）
│   └── 链路恢复管理
│
├── 稳定性追踪器 — StabilityTracker
│   ├── 滑动窗口链路稳定性评分（指数衰减）
│   ├── 历史故障惩罚
│   └── 路径综合稳定性（min×0.7 + mean×0.3）
│
├── 原始 3 方法 + 桥接接入的 8 个内容放置方法
│   ├── AlgoBaseline1  — 含噪全局 Dijkstra + 丢包模型
│   ├── AlgoBaseline2  — 保持直到断裂 + 2步恢复延迟
│   └── AlgoYourMethod — 稳定性加权K最短路 + 主动监测 + 即时切换
│
├── 桥接层
│   └── project_experiment_bridge.py — 内容放置结果到路径稳定性指标的映射
│
├── 实验驱动
│   └── run_experiment()  — 主循环 200 步
│
└── 输出
    ├── _per_flow_jitter() — Gap-aware 抖动计算
    ├── plot_results()     — 2×2 柱状图 + 时延时序图
    └── print / save       — 控制台 + metrics.json
```

---

## 运行方式

```bash
cd experiment2_results
python experiment2_topology_stability.py
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `experiment2_comparison.png` | 四指标对比柱状图 |
| `experiment2_delay_timeline.png` | 时延随时间变化时序图 |
| `metrics.json` | 当前集成版 11 方法指标数据（JSON） |
