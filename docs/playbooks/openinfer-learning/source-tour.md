# OpenInfer Source Tour

> **TL;DR:** 这份文档不是“把所有文件列一遍”，而是告诉你**哪些文件先看、为什么看、看到什么程度就够**。第一次读 openinfer，建议按 `openinfer-server` -> `openinfer-vllm-frontend` -> `openinfer-core` -> `openinfer-qwen3-4b` -> `openinfer-kernels` 的顺序走，最后再回头看 Qwen3.5 / DeepSeek / Kimi。

## 如何使用这份源码导图

每个子系统我都用同一套问题来拆：

- 它为什么存在？
- 第一轮应该先看哪些文件？
- 每个文件负责什么？
- 哪些文件第一轮可以先跳过？
- 它和别的子系统的边界是什么？

如果你第一次读代码，照着“建议阅读顺序”走就够了。

## 子系统 1：`openinfer-server`

### 为什么存在

`openinfer-server` 是服务进程入口和顶层分发层。它负责：

- 读 CLI 参数；
- 识别模型类型；
- 装载模型引擎；
- 启动 frontend 服务。

它**不负责**真正的模型执行。

### 第一轮先看哪些文件

1. `openinfer-server/src/main.rs`
2. `openinfer-server/src/config.rs`
3. `openinfer-server/src/server_engine.rs`
4. `openinfer-server/src/vllm_frontend.rs`

### 文件职责表

| 文件 | 作用 | 第一轮要读到什么程度 |
| --- | --- | --- |
| `openinfer-server/src/main.rs` | 进程入口、启动顺序、并发加载 engine/frontend | 看懂启动流程即可 |
| `openinfer-server/src/config.rs` | CLI 参数与参数校验 | 看懂有哪些全局开关 |
| `openinfer-server/src/server_engine.rs` | 模型识别与 `launch(...)` 分发 | 必须细读 |
| `openinfer-server/src/scheduler.rs` | 对共享 engine 类型的 re-export 适配 | 知道存在即可 |
| `openinfer-server/src/vllm_frontend.rs` | 重新导出 frontend crate | 知道 server 通过它启动前端 |
| `openinfer-server/src/bin/bench_serving/*` | benchmark harness | 非主线，后读 |

### 第一轮可以先跳过什么

- `openinfer-server/src/bin/bench_serving/*`
- `openinfer-server/src/sampler.rs`

这些对理解请求主链路不是第一优先级。

### 和别的层的边界

- 往下：它调用模型 crate 的 `launch(...)`
- 往上：它提供一个 HTTP/CLI 启动壳
- 不越界：它不理解 attention、KV、scheduler 策略、MoE 路径

## 子系统 2：`openinfer-vllm-frontend`

### 为什么存在

这个 crate 让 openinfer 可以复用 vLLM/OpenAI-compatible 的前端层，而不必自己重写一套聊天路由、tokenizer、SSE 输出协议。

它最关键的工作是 bridge：

- `EngineCoreRequest` -> `GenerateRequest`
- `TokenEvent` -> frontend outputs

### 第一轮先看哪些文件

1. `openinfer-vllm-frontend/src/lib.rs`
2. `openinfer-vllm-frontend/src/bridge.rs`
3. `openinfer-vllm-frontend/README.md`

### 文件职责表

| 文件 | 作用 | 第一轮要读到什么程度 |
| --- | --- | --- |
| `openinfer-vllm-frontend/src/lib.rs` | `serve(...)` / `serve_model_with_lora_routes(...)` 等入口 | 看懂怎么把 `EngineHandle` 接进来 |
| `openinfer-vllm-frontend/src/bridge.rs` | 真正的请求/事件双向翻译层 | 必须细读 |
| `openinfer-vllm-frontend/src/lora.rs` | LoRA 路由与控制面 | 如果你关心 LoRA 再读 |
| `openinfer-vllm-frontend/src/sampling_guard.rs` | 采样参数防线 | 知道存在即可 |
| `openinfer-vllm-frontend/src/wire.rs` | wire-level 类型适配 | 只在追输出结构时再读 |
| `openinfer-vllm-frontend/src/bridge/tests.rs` | bridge 行为测试 | 非常值得第二轮再读 |

### 第一轮可以先跳过什么

- `lora.rs`
- `sampling_guard.rs`
- `wire.rs`

先抓住 bridge 主线就够了。

### 和别的层的边界

- 往下：它只通过 `EngineHandle` / `GenerateRequest` / `TokenEvent` 与引擎交互
- 往上：它承接 HTTP / OpenAI-compatible 语义
- 不越界：它不拥有模型状态，不做真正推理

## 子系统 3：`openinfer-core`

### 为什么存在

`openinfer-core` 是共享 runtime helpers 和统一契约出口，不是“所有模型实现都在这里”。

它主要持有：

- 引擎契约 re-export
- sampler
- parallel config
- tensor / device helpers
- KV / page / CUDA graph helpers
- weight loader
- shared ops 包装

### 第一轮先看哪些文件

1. `openinfer-core/src/lib.rs`
2. `openinfer-core/src/engine.rs`
3. `openinfer-core/src/sampler.rs`
4. `openinfer-core/src/parallel.rs`
5. `openinfer-core/src/kv_pool.rs`
6. `openinfer-core/src/page_pool.rs`
7. `openinfer-core/src/cuda_graph.rs`
8. `openinfer-core/src/weight_loader.rs`
9. `openinfer-core/src/ops.rs`

### 文件职责表

| 文件 | 作用 | 第一轮要读到什么程度 |
| --- | --- | --- |
| `openinfer-core/src/lib.rs` | 模块地图 | 快速扫一遍 |
| `openinfer-core/src/engine.rs` | 重新导出 `openinfer-engine` 契约 | 知道真定义在别处 |
| `openinfer-core/src/sampler.rs` | 共享 `SamplingParams` | 必须读 |
| `openinfer-core/src/parallel.rs` | TP/DP/EP 等模型无关并行拓扑 | 必须读 |
| `openinfer-core/src/kv_pool.rs` | 共享 KV 资源接口与状态 | 必须读 |
| `openinfer-core/src/page_pool.rs` | 底层 page allocator | 结合 `kv_pool.rs` 看 |
| `openinfer-core/src/cuda_graph.rs` | 图捕获公共帮助 | 如果关心 graph 再细读 |
| `openinfer-core/src/weight_loader.rs` | shard/safetensors 加载辅助 | 接新模型时必须回来看 |
| `openinfer-core/src/ops.rs` | 共享 operator 入口 | 知道有这层即可 |
| `openinfer-core/src/tensor.rs` | device tensor 抽象 | 真追 operator 时再深读 |
| `openinfer-core/src/ffi.rs` | FFI glue | 第一轮可略 |
| `openinfer-core/src/logging.rs` | 共享日志入口 | 第一轮可略 |
| `openinfer-core/src/cpu_topology.rs` | CPU/NUMA 拓扑 | 多卡/性能问题时再读 |

### 第一轮可以先跳过什么

- `ops/call_trace.rs`
- `ops/traced.rs`
- `cpu_topology.rs`
- `ffi.rs`

除非你正在看 trace/perf/FFI 路径，否则先别被这些辅助文件带跑偏。

### 和别的层的边界

`openinfer-core` 的价值不在“越多越好”，而在“足够共性”。

一个很实用的判断标准是：

- 如果第二个模型也明显会复用，考虑放这里；
- 如果只是当前模型方便，先留在模型 crate。

## 子系统 4：`openinfer-engine`

### 为什么单独存在

`openinfer-engine` 是更轻量的契约 crate：

- `EngineLoadOptions`
- `GenerateRequest`
- `TokenEvent`
- `TokenSink`
- `EngineHandle`

`openinfer-core` 通过 `pub use openinfer_engine::engine::*;` 把它暴露出来。

### 为什么值得看

因为它能让你在不陷入 KV / tensor / CUDA graph 细节时，先把“统一引擎语言”看明白。

### 第一轮要看哪些点

- `GenerateRequest` 的字段
- `TokenEvent` 的变体
- `TokenSink` 的 shared-channel + cancel-flag 设计
- `EngineHandle` 的角色

## 子系统 5：`openinfer-qwen3-4b`

### 为什么它是第一样本

Qwen3 是默认 feature，也是最完整、最标准、最不容易让新手迷路的路径：

- 有清晰的 `launch(...)` / `start_engine(...)`
- 有独立 scheduler / executor
- 有 prefill / decode / unified path
- 有 prefix cache / offload / LoRA 等真实 serving 复杂性
- 但没有 MoE / MLA / 大规模多卡拓扑带来的噪音

### 第一轮先看哪些文件

1. `openinfer-qwen3-4b/src/lib.rs`
2. `openinfer-qwen3-4b/src/scheduler.rs`
3. `openinfer-qwen3-4b/src/scheduler/plan.rs`
4. `openinfer-qwen3-4b/src/scheduler/resolve.rs`
5. `openinfer-qwen3-4b/src/scheduler/effects.rs`
6. `openinfer-qwen3-4b/src/executor.rs`
7. `openinfer-qwen3-4b/src/weights.rs`
8. `openinfer-qwen3-4b/src/prefill.rs`
9. `openinfer-qwen3-4b/src/batch_decode.rs`
10. `openinfer-qwen3-4b/src/unified_forward.rs`
11. `openinfer-qwen3-4b/src/batch_decode_buffers.rs`

### 文件职责表

| 文件 | 作用 | 第一轮要读到什么程度 |
| --- | --- | --- |
| `src/lib.rs` | 外部接口、launch policy、runtime exports | 必须细读 |
| `src/config.rs` | 模型配置解释 | 知道形状从哪来 |
| `src/weights.rs` | 权重加载与运行时模型构建 | 第二优先级 |
| `src/scheduler.rs` | 主调度循环、请求状态、线程模型 | 必须细读 |
| `src/scheduler/plan.rs` | Prefill/Decode/Unified 计划选择与执行调用 | 必须细读 |
| `src/scheduler/resolve.rs` | executor 结果 -> StepEffects | 必须细读 |
| `src/scheduler/effects.rs` | 发 `TokenEvent`、推进 active/pending 状态 | 必须细读 |
| `src/executor.rs` | GPU 执行编排、phase boundary、结果组装 | 必须细读 |
| `src/prefill.rs` | prefill 路径实现 | 第二轮看 |
| `src/batch_decode.rs` | decode 路径实现 | 第二轮看 |
| `src/unified_forward.rs` | prefill+decode 共存时的统一前向 | 第二轮看 |
| `src/batch_decode_buffers.rs` | decode buffer 组织 | 追性能时再深读 |
| `src/lora.rs` | LoRA runtime 相关 | 关心 LoRA 时再读 |
| `src/kernel_plan.rs` | kernel plan/report tooling 入口 | 追 operator 时再读 |
| `src/batch_decode_trace.rs` | trace 辅助 | 诊断时再读 |

### 第一次不要做什么

- 不要一上来钻 `prefill.rs` 和 `batch_decode.rs` 的每个 operator 细节；
- 不要先看 trace / report 工具；
- 不要先看 bench 二进制。

先把 scheduler/executor 边界看清楚再往下钻。

## 子系统 6：`openinfer-qwen35-4b`

### 它主要教你什么

Qwen3.5 教你的不是“如何启动一个模型”，而是：

- shared contract 不变；
- 但内部执行形态可以大变；
- hybrid attention / recurrent state / Triton AOT 都会把复杂性留在模型 crate。

### 优先看哪些文件

- `openinfer-qwen35-4b/src/lib.rs`
- `openinfer-qwen35-4b/src/scheduler.rs`
- `openinfer-qwen35-4b/src/recurrent.rs`
- `openinfer-qwen35-4b/src/recurrent_state.rs`
- `openinfer-qwen35-4b/src/batch_decode_graph.rs`

### 为什么不适合做第一样本

因为它已经把你拉进：

- recurrent state
- hybrid attention
- build-time Triton AOT
- graph slot compaction

这些复杂性适合作为“第二阶段对比”，不适合作为第一天入门。

## 子系统 7：`openinfer-deepseek-v2-lite` 与 `openinfer-deepseek-v4`

### 它们主要教你什么

它们教你的是：

- 固定多卡拓扑也应该是模型层语义；
- MoE / EP 路径可以通过模型 crate 自己决定运行时组织；
- server 不应该硬编码这些策略。

### 看哪些文件

#### DeepSeek-V2-Lite

- `openinfer-deepseek-v2-lite/src/lib.rs`
- `openinfer-deepseek-v2-lite/src/engine.rs`
- `openinfer-deepseek-v2-lite/src/runtime.rs`
- `openinfer-deepseek-v2-lite/src/weights.rs`

#### DeepSeek-V4

- `openinfer-deepseek-v4/src/lib.rs`
- `openinfer-deepseek-v4/src/direct.rs`
- `openinfer-deepseek-v4/src/direct/scheduler.rs`
- `openinfer-deepseek-v4/src/runtime/*`
- `openinfer-deepseek-v4/src/weights.rs`

### 第一次读的目标

不是要一下子吃透 MoE，而是看清：

- 它们依然从 `launch(...)` 接 server；
- 它们依然返回同一个 `EngineHandle`；
- 真正复杂的拓扑与运行时组织，被留在模型 crate 内部。

## 子系统 8：`openinfer-kimi-k2`

### 为什么它重要

Kimi-K2 是这个仓库里最能说明“复杂性必须留在模型 crate”的样本：

- MLA attention
- MoE
- Marlin INT4 experts
- TP/DP/EP 拓扑
- runner/worker/load_balancer 运行时组织

### 为什么它不适合第一样本

因为它的复杂性不是“文件多”，而是多个复杂维度同时叠加。第一次读它，很容易失去对主路径的感觉。

### 正确读法

先读：

- `openinfer-kimi-k2/src/lib.rs`
- `openinfer-kimi-k2/src/runner.rs`
- `docs/models/kimi-k2/source-layout.md`

然后再按需要分叉到：

- `src/runner/scheduler/`
- `src/runner/executor/`
- `src/runner/worker/`
- `src/weights/`

### 你真正要学到的东西

不是每个文件在做什么，而是：

- 为什么一个复杂模型线最终会长成一个专用运行时；
- 为什么这种复杂性不该被强行抽回共享层。

## 子系统 9：`openinfer-kernels`

### 为什么它重要

`openinfer-kernels` 不是“放 `.cu` 文件的目录”，而是整个仓库的底层构建 owner。

`build.rs` 负责：

- 检测或解析 SM 目标
- 安排 nvcc 任务
- 触发 Triton AOT / TileLang / CuTe DSL 路径
- 处理不同模型线的 feature-gated kernel build

### 第一轮先看什么

- `openinfer-kernels/build.rs`
- `docs/subsystems/kernels/openinfer-kernels-boundary.md`

### 第一轮重点读什么

- `workspace_root()` / `crate_root()` 之类路径组织
- `detect_sm_targets()` 和 SM 归一化逻辑
- 不同 feature 如何映射到不同的代码生成路径
- 为什么默认 Qwen3 可以不依赖 Python，而 Qwen3.5 不行

### 第一轮不要做什么

- 不要一上来读所有 CUDA/Triton 具体实现；
- 不要把 operator 细节和系统架构混成一个问题。

## 一张最终阅读顺序表

| 轮次 | 读什么 | 目标 |
| --- | --- | --- |
| 第 1 轮 | `openinfer-server` + `openinfer-vllm-frontend` | 看懂入口与桥接 |
| 第 2 轮 | `openinfer-engine` + `openinfer-core` | 看懂统一契约与共享积木 |
| 第 3 轮 | `openinfer-qwen3-4b` | 看懂标准模型线闭环 |
| 第 4 轮 | `openinfer-kernels` | 看懂底层 build owner |
| 第 5 轮 | `qwen35` / `deepseek*` / `kimi` | 看懂复杂性如何逐步增长 |

## 如果你只有半天

只看这些：

1. `openinfer-server/src/server_engine.rs`
2. `openinfer-vllm-frontend/src/bridge.rs`
3. `openinfer-engine/src/engine.rs`
4. `openinfer-qwen3-4b/src/lib.rs`
5. `openinfer-qwen3-4b/src/scheduler.rs`
6. `openinfer-qwen3-4b/src/executor.rs`
7. `openinfer-kernels/build.rs`

这七个入口已经足够让你知道这个仓库是怎么工作的。

下一步建议看：`docs/playbooks/openinfer-learning/model-comparison.md`，然后回头看 `docs/playbooks/openinfer-learning/subsystem-invariants.md`。

