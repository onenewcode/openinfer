# openinfer-kimi-k2 工作原理

`openinfer-kimi-k2` 是 Kimi-K2.6 的模型线 crate。它负责把 Kimi 的 MLA、MoE、INT4 Marlin 专家、TP/DP/EP 并行拓扑、调度与权重打包全部组织成一条可服务的 text-only 引擎。

## 一句话先理解

这条 crate 的本质不是“又一个 decoder-only 模型”，而是：

- attention 不是普通 full attention，而是 MLA；
- MLP 不是普通 dense MLP，而是大规模 MoE；
- 并行不是单一 TP，而是会组合 TP / DP / EP；
- 专家执行还依赖 Marlin INT4 与 DeepEP / NCCL 之类的通信后端。

所以它更像一台“面向大规模 MoE + 多并行拓扑的专用 serving 引擎”。

## 它在架构里的位置

```text
server
  -> openinfer-kimi-k2::launch / start_engine
      -> runner
          -> scheduler / worker / load_balancer / executor
              -> openinfer-kernels
              -> openinfer-kv-cache
              -> openinfer-sample
```

和别的模型线一样，上层看到的是 `EngineHandle`；但内部真正的核心是 `runner/`。

## 为什么 `runner/` 是这个 crate 的灵魂

`runner/` 不是一个小工具目录，而是 Kimi-K2 运行时的主体：

- `scheduler.rs`：请求调度
- `worker.rs`：rank worker 真正执行 decode / prefill
- `executor.rs`：执行编排
- `load_balancer.rs`：DP 方向的请求分配
- `moe_deepep.rs` / `moe_nccl.rs`：不同 EP 后端
- `affinity.rs`：线程 / 设备绑定

也就是说，这个 crate 把“怎么让 8 卡上的 MoE 系统真正跑起来”放在 `runner` 里集中实现，而不是把逻辑散落到 server 或 root runtime。

## Kimi 在这里最特殊的三个点

### 1. MLA 而不是普通 KV attention

Kimi 的 attention 不是 Qwen3 那种标准 paged full attention，而是 MLA 路径：

- query / kv 的投影与吸收方式不同；
- paged cache 存的也不是同样的形态；
- decode 时的 kernel 组合与普通 full attention 不一样。

这也是为什么它需要大量 Kimi 专用 kernel surface。

### 2. MoE 是主路径，不是旁支

这条模型线的核心热点之一就是 MoE：

- routing
- dispatch
- expert GEMM
- combine

这些都不是附加功能，而是每步 decode 都要经过的主路径。所以 crate 里专门有：

- `moe_deepep.rs`
- `moe_nccl.rs`
- 权重打包与专家布局相关的 `weights/package.rs`

### 3. 拓扑策略属于模型 crate 自己

`KimiLaunchOptions` 很能体现这条线的设计原则：

- `tp_size`
- `dp_size`
- `ep_backend`
- `cuda_graph`

这些不是让 server 任意拼的自由参数，而是由 `launch(...)` 在模型 crate 内部解释和验证。也就是说，**并行拓扑策略被视为模型语义的一部分**，而不是外层框架随便组合的开关。

## 权重层为什么也很重要

`weights/` 在这里不只是“读 safetensors”：

- `manifest.rs`
- `package.rs`
- `load.rs`
- `context.rs`

这些模块承担了更重要的事情：把 checkpoint 里的原始权重，转换成 runtime 真正能高效消费的布局，尤其是 Marlin INT4 专家权重。

对 Kimi 来说，权重“怎么打包”本身就是运行时性能的一部分。

## 它是怎么跑请求的

从高层看，请求路径是：

1. `launch(...)` 根据 TP/DP/EP 选定并行布局；
2. `runner` 拉起对应 worker；
3. scheduler 接收请求并把它们分配到合适 rank / DP lane；
4. worker 执行 prefill / decode；
5. attention 走 MLA，MoE 走对应 EP 后端；
6. 采样结果通过统一事件流返回。

这里最关键的一点是：Kimi 的“模型逻辑”和“并行执行逻辑”高度耦合，所以不能简单照搬 Qwen3 的单引擎思路。

## kernel report 为什么也放在这里

这个 crate 同时带着：

- `kimi_kernel_report`
- `kimi_model_report`

原因很自然：Kimi 的性能瓶颈很依赖具体模型路径、MoE backend、拓扑配置。如果把报告逻辑完全抽离到别处，很多上下文会丢掉。

所以它和 `openinfer-bench` 的关系是：

- `openinfer-bench` 提供通用基准骨架；
- `openinfer-kimi-k2` 提供 Kimi 专属的测量语义与路径。

## 最重要的心智模型

理解 `openinfer-kimi-k2` 时，最好不要把它想成“一个模型 + 一些优化”，而要想成：

- 一个 MLA attention 引擎；
- 一个 MoE 专家调度与通信系统；
- 一个由 TP/DP/EP 拓扑驱动的分布式 serving 运行时；
- 一个把权重打包、执行、报告都收在模型线内部的专用系统。

这才是它和其他 crate 最大的不同。
