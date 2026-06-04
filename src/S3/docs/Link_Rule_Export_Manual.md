# Link / Rule 导出操作手册

## 1. 适用场景

这份手册对应当前 S3 路由主流程，目标是：

- 使用新的轨迹场景 `traces/sat_trace_40`
- 使用新的 UAV 轨迹目录 `traces/uav_trace_50`
- 在 10 分钟窗口内，把每架 UAV 建模成“每 10 秒发送一张 10+ MB 照片”
- 生成并导出 `links` 和 `rules` 给下一个模块使用

当前代码已支持：

- `--uav-file` 传单个 CSV 文件
- `--uav-file` 直接传一个目录，自动拼接目录下所有 UAV CSV 分片
- `--traffic-model photo` 周期照片流量模型
- `--output-tag` 为输出目录命名，避免覆盖旧实验结果

## 2. 输入数据约定

当前新场景输入如下：

- 卫星轨迹目录：`traces/sat_trace_40`
- UAV 轨迹目录：`traces/uav_trace_50`

其中：

- `sat_trace_40` 下是按时间分片的卫星 CSV
- `uav_trace_50` 下是按时间分片的 UAV CSV
- 代码会按文件名排序后拼接全部 CSV，再按 `time_ms` 建立时间线

## 3. 流量模型说明

### 3.1 照片上报模型

执行参数：

- `--traffic-model photo`
- `--photo-interval-s 10`
- `--photo-size-mb 12`

表示：

- 每架在线 UAV 在 `t = 0, 10, 20, ..., 590s` 发起一次上报
- 每次上报目标节点为 `GS_01`
- 每次上报等价为一张 `12 MB` 照片

在规则文件中，这个业务会被换算成：

$$
req\_bw\_mbps = \frac{photo\_size\_mb \times 8}{photo\_interval\_s}
$$

对 `12 MB / 10 s`，即：

$$
req\_bw\_mbps = \frac{12 \times 8}{10} = 9.6
$$

因此，下游模块如果按带宽请求处理，可以直接读取规则里的 `req_bw_mbps`。

### 3.2 控制流

默认 legacy 模式下还会生成 `CTRL_FLOW`。

如果只想导出照片业务，建议加：

```bash
--no-ctrl-flow
```

## 4. 推荐运行命令

### 4.1 单算法导出（推荐先做）

以优化算法为例：

```bash
python algorithms/s3_optimized.py traces/sat_trace_40 --uav-file traces/uav_trace_50 --traffic-model photo --photo-interval-s 10 --photo-size-mb 12 --no-ctrl-flow --max-steps 6000 --output-tag sat40_uav50_photo10s
```

说明：

- `6000` 步对应 10 分钟、100 ms 时间粒度
- 每 100 步正好是 10 秒，因此照片请求节奏和时间线对齐
- 输出目录会写到：`outputs/output_sat40_uav50_photo10s/`

### 4.2 七算法统一横评

```bash
python algorithms/benchmark_algorithms.py traces/sat_trace_40 --uav-file traces/uav_trace_50 --traffic-model photo --photo-interval-s 10 --photo-size-mb 12 --no-ctrl-flow --max-steps 6000 --save-outputs --output-dir outputs/benchmark_sat40_uav50_photo10s
```

说明：

- 每个算法自己的 `links` / `rules` 仍写在各自 `outputs/output_*` 目录里
- benchmark 目录额外汇总算法对比表和对比图

## 5. links 是怎么生成的

`links` 由 `algorithms/s3_routing_core.py` 中的 `compute_topology()` 生成。

每个时间步的生成过程是：

1. 读取当前 `time_ms` 对应的 UAV 位置和对齐到秒级的 SAT 位置
2. 用 `cKDTree` 在 `5000 km` 距离阈值内查找候选邻居
3. 对 SAT-GS / SAT-UAV 链路做仰角过滤，低于 `10°` 的链路丢弃
4. 根据节点类型设置链路带宽：
   - SAT-SAT: `100 Mbps`
   - SAT-UAV: `20 Mbps`
   - SAT-GS: `20 Mbps`
5. 计算每条链路的：
   - `distance_km`
   - `delay_ms`
   - `jitter_ms`
   - `bw_mbps`
   - `max_queue_pkt`
6. 将本时间步所有链路写入当前 chunk

### 5.1 links 输出文件

文件命名规则：

- `topology_links_<start_ms>_<end_ms>.csv`

例如：

- `topology_links_0_59900.csv`

### 5.2 links 分块规则

系统按 `60000 ms` 为一个 chunk 保存一次，也就是每分钟一个文件：

- `0 ~ 59999 ms`
- `60000 ~ 119999 ms`
- `...`

10 分钟场景一般会得到 10 组 `links` 文件。

### 5.3 links 关键字段

| 字段 | 含义 |
|------|------|
| `time_ms` | 当前时间步 |
| `src` / `dst` | 链路两端节点 |
| `direction` | 当前固定为 `BIDIR` |
| `distance_km` | 节点间距离 |
| `delay_ms` | 传播时延 |
| `jitter_ms` | 当前模型里的简化抖动 |
| `bw_mbps` | 链路带宽 |
| `max_queue_pkt` | 按带宽-时延积估计的队列上限 |
| `type` | 链路类型，如 `SAT-SAT`、`SAT-UAV` |
| `status` | 当前是否可用，默认 `UP` |
| `lifetime_ms` | 链路生命周期，当前固定 `60000` |

## 6. rules 是怎么生成的

`rules` 由 `run_simulation()` 主循环在每个时间步生成：

1. 先调用 `build_flow_requests()` 生成本时间步的业务请求
2. 对每条请求调用对应路由算法的 `plan_route()`
3. 只有当路由成功且路径长度至少为 2 时，才会生成一条规则
4. 规则由 `make_rule()` 组装并写入当前 chunk

### 6.1 photo 模式下的规则生成时机

在 `--traffic-model photo` 模式下：

- 只有当 `time_ms` 是 `10000 ms` 的整数倍时，才会为每架 UAV 生成一条照片上报请求
- 如果该时间点没有可达路径，则：
  - `Flow Requests` 会增加
  - `Failed Routes` 会增加
  - 但 `rules` 文件中不会出现对应规则

因此，出现“有 links 文件，但 rules 为空”并不表示导出失败，而是表示该时间窗内业务请求没有成功路径。

### 6.2 rules 输出文件

文件命名规则：

- `routing_rules_<start_ms>_<end_ms>.json`

每个文件结构如下：

```json
{
  "meta": {
    "chunk_id": 0
  },
  "rules": [
    {
      "time_ms": 10000,
      "node": "UAV_01",
      "dst_cidr": "10.0.0.1/32",
      "action": "replace",
      "next_hop": "SAT_12345",
      "next_hop_ip": "10.1.2.3",
      "algo": "Optimized",
      "req_bw_mbps": 9.6,
      "traffic_type": "periodic_photo",
      "photo_size_mb": 12.0,
      "photo_interval_s": 10.0,
      "debug_info": "..."
    }
  ]
}
```

### 6.3 rules 关键字段

| 字段 | 含义 |
|------|------|
| `time_ms` | 规则生成时刻 |
| `node` | 下发规则的源节点 |
| `dst_cidr` | 目标地址前缀 |
| `action` | 当前固定为 `replace` |
| `next_hop` / `next_hop_ip` | 下一跳节点及 IP |
| `algo` | 产生该规则的路由算法 |
| `req_bw_mbps` | 本次业务请求需要的带宽 |
| `traffic_type` | 当前业务类型，如 `periodic_photo` |
| `photo_size_mb` | 照片大小 |
| `photo_interval_s` | 照片发送周期 |
| `debug_info` | 路由算法附加说明 |

## 7. 输出目录结构

如果使用：

```bash
--output-tag sat40_uav50_photo10s
```

则输出目录为：

```text
outputs/
└─ output_sat40_uav50_photo10s/
   ├─ links/
   │  ├─ topology_links_0_59999.csv
   │  ├─ topology_links_60000_119999.csv
   │  └─ ...
   ├─ rules/
   │  ├─ routing_rules_0_59999.json
   │  ├─ routing_rules_60000_119999.json
   │  └─ ...
   └─ metrics.json
```

`metrics.json` 里会额外记录：

- `UAV Trace`
- `Traffic Model`
- `Photo Interval (s)`
- `Photo Size (MB)`
- `Include CTRL Flow`
- `Output Root`

这些字段适合下游模块做场景对账。

## 8. 导出到下一个模块的建议

建议直接按整个输出目录导出，而不是只拷单个文件：

1. 保留 `links/` 和 `rules/` 的分钟级 chunk 文件名
2. 同时保留根目录的 `metrics.json`
3. 下游模块按 `time_ms` 和 chunk 时间窗对齐读取

推荐最小导出集合：

- `outputs/output_sat40_uav50_photo10s/links/`
- `outputs/output_sat40_uav50_photo10s/rules/`
- `outputs/output_sat40_uav50_photo10s/metrics.json`

## 9. 常见问题

### 9.1 为什么有 links，没有 rules？

因为 `links` 表示“这个时间步网络里存在什么链路”，而 `rules` 表示“这个时间步有哪些业务请求成功找到了路径”。

如果照片请求在该时间点不可达，就会出现：

- `Flow Requests > 0`
- `Failed Routes > 0`
- `rules` 文件为空数组

### 9.2 怎么确认 photo 模式真的生效了？

检查输出目录下的 `metrics.json`，应至少看到：

- `"Traffic Model": "photo"`
- `"Photo Interval (s)": 10.0`
- `"Photo Size (MB)": 12.0`

还可以打开任一 `rules` 文件，检查规则项中是否包含：

- `"traffic_type": "periodic_photo"`
- `"photo_size_mb": 12.0`
- `"photo_interval_s": 10.0`