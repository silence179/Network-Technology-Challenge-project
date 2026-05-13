# S3 Topology Project (Reorganized)

本项目已按功能重新整理目录，并将默认入口切换为优化版本。

当前仓库有三类实验入口：

- 原始场景实验：`experiment0_results/` 到 `experiment3_results/`
- 路由横评：`algorithms/s3_routing_core.py` 与 `algorithms/benchmark_algorithms.py`
- OTCP / cache 放置实验：`algorithms/code/experiment.py`

当前代码状态下：

- Experiment 0 仍保持 A/B/C 三种建模深度的 reality gap 验证
- Experiment 1 已原生接入新增 8 个内容/缓存 baseline，与原始 3 个方法一起比较
- Experiment 2、3 也已通过 `algorithms/code/project_experiment_bridge.py` 将新增 8 个 baseline 接入原始脚本
- `run_all_project_experiments.py` 仍可用于项目级统一重跑

## 目录结构

```text
s3/
├─ s3.py                          # 主入口（已替换为 optimized）
├─ README.md
├─ algorithms/                    # 路由基线 + cache/placement baseline
│  ├─ s3_routing_core.py
│  ├─ s3_optimized.py
│  ├─ s3_hypatia.py
│  ├─ s3_lsr.py
│  ├─ s3_madrl.py
│  ├─ s3_otcp.py
│  ├─ s3_ftrl.py
│  ├─ s3_dtn_cgr.py
│  ├─ benchmark_algorithms.py     # 七路由算法横评脚本
│  └─ code/                       # OTCP/cache baseline 套件 + 实验桥接层
├─ experiment0_results/           # Experiment 0: reality gap validation
├─ experiment1_results/           # Experiment 1: cache-routing synergy
├─ experiment2_results/           # Experiment 2: topology stability
├─ experiment3_results/           # Experiment 3: UAV relay
├─ figures/                       # OTCP 与其他实验图表
├─ results/                       # OTCP 与总控清单结果
├─ run_all_project_experiments.py # 项目级全量实验入口
├─ traces/                        # 全部轨迹数据
│  ├─ sat_trace/
│  ├─ sat_trace_50/
│  ├─ sat_trace_100/
│  ├─ sat_trace_150/
│  ├─ sat_trace_200/
│  ├─ uav_trace/
│  └─ uav_trace_full.csv
├─ outputs/                       # 所有输出文件
│  ├─ output*/
│  ├─ output_sat_trace_*/
│  └─ benchmark_results/
├─ docs/                          # 文档与历史脚本
│  ├─ Optimization_Readme.md
│  └─ benchmark_legacy.py
└─ test/                          # 复现与实验参考代码
```

## 默认行为变更

- 根目录 `s3.py` 已改为优化算法入口，相当于运行 `optimized`。
- 算法脚本默认从 `traces/` 读取数据。
- 新输出统一写入 `outputs/`。

## 快速开始

### 1) 跑默认优化版

```bash
python s3.py
```

### 2) 跑某个算法（示例：Hypatia）

```bash
python algorithms/s3_hypatia.py
```

### 3) 七路由算法横评

```bash
python algorithms/benchmark_algorithms.py
```

可选参数示例：

```bash
python algorithms/benchmark_algorithms.py traces/sat_trace_100 --max-steps 300 --save-outputs
```

### 4) 原始实验 1/2/3（已集成 11 方法）

```bash
python experiment1_results/experiment1_cache_routing.py
python experiment2_results/experiment2_topology_stability.py
python experiment3_results/experiment3_uav_relay.py
```

说明：

- 这三个脚本都会输出包含原始 3 个方法和新增 8 个 baseline 的 `metrics.json`
- Experiment 1 是原生内容/缓存场景，跨方法指标最直接可比
- Experiment 2、3 中新增 baseline 通过桥接层适配到原始场景，成功率/恢复类指标最适合跨族比较

### 5) OTCP / cache baseline 主实验

```bash
python -m code.experiment --mode main --sat-dir traces/sat_trace_100 --max-steps 100
```

说明：该命令需要在 `algorithms/` 目录内执行；若在项目根目录统一调度，推荐直接使用总控脚本。

### 6) 全量重跑项目实验

```bash
python run_all_project_experiments.py
```

只重跑新增 cache baseline 套件：

```bash
python run_all_project_experiments.py --only otcp --otcp-mode all --otcp-max-steps 100
```

## 主要输出位置

- 原始实验集成结果：
  - `experiment1_results/metrics.json`
  - `experiment2_results/metrics.json`
  - `experiment3_results/metrics.json`
- 原始实验图表：各自实验目录下的 `experiment*_comparison.png` / `timeline` 图
- 单算法输出：`outputs/output_<sat_count>/`（例如 `output_25`、`output_50`）
- 横评汇总：`outputs/benchmark_results/`
  - `algorithm_benchmark.csv`
  - `algorithm_benchmark.json`
  - `algorithm_benchmark.png`
- OTCP/cache baseline 图表：`figures/`
- OTCP/cache baseline 指标与总控清单：`results/`
  - `metrics.json`
  - `project_experiment_manifest.json`

## 备注

- 若使用 Windows 且命令 `python` 指向不一致，请改用你本机解释器完整路径。
- `test/` 目录保留为对照与复现材料，不参与当前主流程。
- `algorithms/code/experiment.py` 仍是 OTCP/cache baseline 的原生评测入口；若要看最严格的内容放置指标，优先用它。
- `experiment2_results/` 和 `experiment3_results/` 中的新增 baseline 是通过桥接层适配到原始场景的，因此成功率、可恢复性、可服务性更适合作为跨族主比较项；绝对时延/流量仍保留各自实现的原生语义。
