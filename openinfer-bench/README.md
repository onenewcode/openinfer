# openinfer-bench 工作原理

`openinfer-bench` 是一个**模型无关的 GPU kernel 基准测试骨架**。它不负责跑完整个模型，也不负责定义某个模型有哪些算子；它只负责把“如何稳定测一段 CUDA 工作负载”“如何把一次次测量汇总成报告”“如何给 kernel call 生成稳定身份”这些通用逻辑抽出来，供 `openinfer-qwen3-4b`、`openinfer-qwen35-4b` 等模型 crate 复用。

## 一句话先理解

可以把它理解成：

- 模型 crate 决定“我要测哪些 op、怎么准备输入、怎么发起 kernel”；
- `openinfer-bench` 决定“怎么测得稳、怎么统计、怎么把结果组织成可比较的报表”。

它是基准测试的“公共方法论层”，不是某个模型自己的 benchmark。

## 它在系统里的位置

典型使用路径是：

```text
模型 crate 的 report / bench 工具
  -> 准备某个 KernelCall 对应的输入张量
  -> 调 openinfer-bench::measure_loop(...)
  -> 拿到 LatencyStats / MeasuredCall
  -> 再用 accumulate / RollupRow / CallSiteRow 汇总成报告
```

所以它站在：

- 上游：模型自定义的“如何发起这个算子”
- 下游：统一的时延统计与报表结构

之间。

## 这个 crate 解决了什么问题

如果没有它，每个模型 crate 都要重复写一遍：

- CUDA event 计时循环；
- warmup；
- 均值 / 方差 / p50 / p95 / p99 计算；
- “这个 op 的输入输出 shape + attrs 组成什么唯一 key”；
- “多次调用如何汇总成 per-op / per-call-site 报表”。

这些逻辑本身没有模型特异性，所以被抽到这里。

## 核心组成

### 1. `measure_loop`：统一的 GPU 计时循环

`measure_loop(...)` 是这个 crate 最核心的入口。

它的做法是：

1. 先做固定次数 warmup；
2. 用 CUDA event 在同一条 stream 上把一次 launch 包起来；
3. 收集每轮耗时；
4. 最后转成 `LatencyStats`。

这意味着它测的是：

- 真正的 GPU kernel / stream 工作耗时；
- 不是 wall-clock 的粗粒度时间；
- 也不是把一堆 launch 混在一起的模糊统计。

对 kernel bench 来说，这种测法比普通 CPU 定时更接近我们真正关心的对象。

### 2. `LatencyStats`：统一的统计口径

`LatencyStats` 把一次样本集合压成一组统一字段：

- `mean_us`
- `stddev_us`
- `min_us`
- `p50_us`
- `p95_us`
- `p99_us`
- `max_us`

这样不同模型 crate 输出的报告就可以天然对齐，而不是每个地方都自定义一套统计列。

同时它还支持 `zero(...)`，用来表示“这个调用在逻辑上存在，但这里故意不计时或它本质是 no-op”。这对某些单 rank collectives 或条件路径特别有用。

### 3. `bench_key`：给一次 kernel call 一个稳定身份

`bench_key(...)` 会把一个 `KernelCall` 的关键信息序列化成稳定 JSON：

- op 名称
- inputs
- outputs
- attrs

这背后的目的不是“好看”，而是为了：

- 跨次运行比较同一个 kernel 形状；
- 让 benchmark snapshot / report 具有可匹配性；
- 避免只靠一个模糊的 op 名导致不同 shape 被错误归并。

### 4. shape / attr 访问辅助函数

像 `axis(...)`、`input(...)`、`output(...)`、`attr_usize(...)` 这些函数，本质上是在帮上层 report 代码更安全地读取 `KernelCall` 元数据。

也就是说，这个 crate 不只测时，还承担了“如何从模型提供的调用描述里读出维度语义”的辅助工作。

### 5. rollup：把逐调用数据折叠成报告

`Accum`、`accumulate(...)`、`RollupRow`、`CallSiteRow` 这一组类型，负责把细粒度结果组织成更适合人读的汇总视图：

- 每个 op 总共耗了多少时间；
- 每个 call site 平均一次耗时多少；
- 各项在整个测量集里的占比是多少。

这也是为什么它既能服务单算子 bench，也能服务更高层的 model report。

## 它刻意不做什么

`openinfer-bench` 的边界很清楚：

- 不负责知道 Qwen3 或 Qwen3.5 的模型结构；
- 不负责构造真实权重；
- 不负责决定某个模型有哪些 kernel call；
- 不负责服务端压测；
- 不负责 correctness。

这些事情都留给上层：

- 模型 crate 自己定义 call schedule；
- `openinfer-kernels` 提供底层 op；
- 服务级别 benchmark 由别的工具或 bin 负责。

## 为什么它值得单独做成 crate

因为 openinfer 的优化工作很依赖“同一种测量方法跨模型复用”：

- Qwen3 要做 kernel report；
- Qwen3.5 要做 op bench；
- 以后 DeepSeek / Kimi 也会需要类似基础设施。

如果每条模型线都各写一套，最后最难比较的反而不是 kernel，而是“测量方法本身是不是一致”。

所以 `openinfer-bench` 的真正价值是：**把基准测试中的“方法学差异”降到最低**。
