# OpenInfer Request Lifecycle

> **TL;DR:** 想真正读懂 openinfer，最稳的主线不是“按 crate 一个个看”，而是跟着一条请求走完它的一生：server 解析参数并识别模型，frontend bridge 把前端请求翻译成 `GenerateRequest`，`EngineHandle` 把请求送进模型引擎，模型 scheduler 决定这一步是 prefill、decode 还是 unified，executor 在 GPU 上执行，最后 `TokenEvent` 被 bridge 折叠回前端输出流。

## 为什么先看请求生命周期

因为这个仓库的目录很容易让人误判：

- 看到 `openinfer-server`，以为推理逻辑在这里；
- 看到 `openinfer-core`，以为所有核心模型逻辑都抽到了这里；
- 看到 `openinfer-kernels`，以为这里才是系统主线；
- 看到各模型 crate，反而不知道它们和 server 的关系。

一旦你按请求生命周期读，边界会非常清楚：

- server 负责入口与分发；
- frontend bridge 负责协议翻译；
- engine contract 负责统一语言；
- 模型 crate 负责调度与执行；
- kernels/comm 负责底层能力。

## 总时序图

```text
Client / HTTP request
  -> openinfer-server/src/main.rs
  -> openinfer-server/src/server_engine.rs
  -> openinfer-vllm-frontend/src/bridge.rs
  -> GenerateRequest { prompt_tokens, params, token_tx, ... }
  -> EngineHandle::submit(...)
  -> model scheduler thread
  -> execution plan: Prefill / Decode / Unified
  -> model executor on GPU
  -> TokenEvent::Scheduled / Token / Finished / Error / Rejected
  -> openinfer-vllm-frontend bridge reduces burst
  -> frontend SSE / OpenAI-compatible output
```

下面按阶段拆。

## 阶段 0：进程启动和模型装载

### 入口文件

- `openinfer-server/src/main.rs`
- `openinfer-server/src/config.rs`
- `openinfer-server/src/server_engine.rs`

### 这阶段做了什么

`openinfer-server/src/main.rs` 不是“推理逻辑”，而是整个服务的启动器。它主要做四件事：

1. 初始化日志；
2. 解析 CLI 参数；
3. 检测模型类型并校验参数；
4. 在一个 blocking 线程里 load engine，与 frontend 初始化并行进行。

一个最关键的片段是：

```rust
let model_type = detect_model_type(&args.model_path)?;
args.validate(model_type)?;
let engine_load = tokio::task::spawn_blocking(move || -> anyhow::Result<EngineHandle> {
    load_engine(&args, model_type)
});
```

这说明两件事：

- server 知道“它是什么模型”；
- server 不直接自己 new 某个 executor，而是把启动交给模型 crate。

### `detect_model_type(...)` 真正在做什么

位置：`openinfer-server/src/server_engine.rs`

`detect_model_type(...)` 读取 `model_path/config.json`，根据稳定字段判断模型线：

- `model_type == "deepseek_v2"` 且满足 Lite 的形状约束 -> `DeepSeekV2Lite`
- `model_type == "deepseek_v4"` -> `DeepSeekV4`
- `model_type == "kimi_k25"` 或嵌套 `text_config.model_type == "kimi_k2"` -> `KimiK2`
- `text_config` 存在 -> `Qwen35`
- 否则默认走 Qwen3

这就是 **server 的职责边界**：它只回答“这属于哪条模型线”，不回答“这条模型线该怎么跑”。

### `load_engine(...)` 为什么重要

同样在 `openinfer-server/src/main.rs`。

`load_engine(...)` 是一个很干净的纯分发函数：

- DeepSeek-V4 -> `openinfer_deepseek_v4::launch(...)`
- DeepSeek-V2-Lite -> `openinfer_deepseek_v2_lite::launch(...)`
- Kimi-K2 -> `openinfer_kimi_k2::launch(...)`
- Qwen3 -> `openinfer_qwen3_4b::launch(...)`
- Qwen3.5 -> `openinfer_qwen35_4b::launch(...)`

你要特别注意，这里没有：

- attention 分支；
- scheduler 细节；
- CUDA Graph 具体实现；
- KV cache 结构；
- MoE / TP / DP / EP 的运行时组织。

这些都故意被留在模型 crate 内。

## 阶段 1：frontend bridge 把前端请求翻译成 `GenerateRequest`

### 入口文件

- `openinfer-vllm-frontend/src/lib.rs`
- `openinfer-vllm-frontend/src/bridge.rs`

### 为什么要有 bridge

openinfer 的本地引擎契约是：

- 输入：`GenerateRequest`
- 输出：`TokenEvent`

但前端复用的是 vLLM/OpenAI-compatible 的 engine-core 接口。于是 bridge 的职责就是双向翻译：

- 前端请求 -> `GenerateRequest`
- `TokenEvent` 流 -> frontend 期待的输出批次

### 请求翻译的关键代码

位置：`openinfer-vllm-frontend/src/bridge.rs`

```rust
let token_tx = TokenSink::new(tag.clone(), event_tx.clone(), Arc::clone(&cancelled));
self.handle.submit(GenerateRequest {
    request_id: Some(request_id),
    queued_at_unix_s: Some(request.arrival_time),
    prompt_tokens,
    params: convert_sampling(&sampling_params),
    max_tokens: sampling_params.max_tokens as usize,
    lora_adapter: lora_adapter_from_sampling_params(&sampling_params)?,
    token_tx,
    logprobs: requested_logprobs(&sampling_params),
    echo: false,
})?
```

从这段你应该读出：

- front-end 层已经把 prompt 变成 token ids；
- sampling 参数在这里被转换成内部 `SamplingParams`；
- bridge 给每个请求创建一个带 `RequestTag` 的 `TokenSink`；
- 请求提交后，真正的引擎已经不关心 HTTP/OpenAI 语义，只关心 token 和事件。

### `TokenSink` 为什么关键

`TokenSink` 定义在 `openinfer-engine/src/engine.rs`，并通过 `openinfer-core::engine` 暴露。

它不是“每个请求一个独立 channel”的老式设计，而是：

- 所有请求共享一个 tagged output channel；
- 每个请求自己的 `TokenSink` 只负责给事件打 `RequestTag`；
- frontend bridge 再按 tag demux。

这带来的系统意义是：**每步只需要一个共享消费者，而不是 N 个请求各自 wakeup。**

## 阶段 2：`GenerateRequest` 穿过统一引擎契约

### 核心文件

- `openinfer-core/src/engine.rs`
- `openinfer-engine/src/engine.rs`

`openinfer-core/src/engine.rs` 现在基本是重新导出：

```rust
pub use openinfer_engine::engine::*;
```

所以要理解契约本体，直接读 `openinfer-engine/src/engine.rs` 即可。

### 你要重点看哪些类型

#### `EngineLoadOptions`

定义了模型启动时的共通选项：

- `enable_cuda_graph`
- `enable_prefill_profile`
- `device_ordinals`
- `parallel_config`
- `ep_backend`
- `seed`

这说明：

- 共享层可以表达“设备、并行、采样种子、是否图捕获”这些共性；
- 但共享层不负责决定某个模型是否支持它们。

#### `GenerateRequest`

这是真正的请求契约。它包含：

- `request_id`
- `queued_at_unix_s`
- `prompt_tokens`
- `params`
- `max_tokens`
- `lora_adapter`
- `token_tx`
- `logprobs`
- `echo`

读这个结构体时，你要明白：

- 这是一个**引擎视角**的请求，不是一个 HTTP 视角的请求；
- 它已经摆脱了 chat/openai 风格的高层语义；
- 模型层拿到这个结构体后，就只关心如何调度和执行。

#### `TokenEvent`

`TokenEvent` 定义了引擎能吐回来的全部事件：

- `Scheduled`
- `Token`
- `PromptTokens`
- `Finished`
- `Error`
- `Rejected`

为什么这很重要？因为你之后在所有模型 scheduler 里，看到的几乎都是“何时 emit 哪种 event”。

#### `EngineHandle`

`EngineHandle` 是上层握着的“提交句柄”，不是引擎本体。它的意义是：

- 统一所有模型的提交面；
- 屏蔽底下是单 submit channel 还是 command channel；
- 给 frontend / bench / tests 一个统一入口。

## 阶段 3：模型 crate 的 `launch(...)` 把 server 意图转成具体启动策略

### 最适合看的样本：Qwen3

位置：`openinfer-qwen3-4b/src/lib.rs`

Qwen3 的 `launch(...)` 很值得当模板，因为它最清楚地演示了“server-facing policy layer”这个概念。

它主要做这些事：

- 根据 `tp_size` 推导 `device_ordinals`；
- 处理 LoRA 与 CUDA Graph 的互斥；
- 组装 `EngineLoadOptions`；
- 处理 KV offload / prefix cache / max_prefill_tokens；
- 决定调用 `start_engine_with_lora_control(...)` 还是 `start_engine_with_offload(...)`。

也就是说：server 给它的是“用户想要什么”；模型 crate 决定的是“这个模型实际上怎么接受这些要求”。

### 对比：Qwen3.5 / DeepSeek / Kimi

- `openinfer-qwen35-4b/src/lib.rs`：Qwen3.5 只支持单卡，所以 `launch(...)` 很瘦，但它明确把“只支持一个设备”的约束留在模型层。
- `openinfer-deepseek-v2-lite/src/lib.rs`：固定 EP2，明确忽略 CUDA Graph。
- `openinfer-deepseek-v4/src/lib.rs`：固定 MP8，明确忽略 CUDA Graph，并允许 prefill profiling。
- `openinfer-kimi-k2/src/lib.rs`：把 TP/DP/EP 拓扑组合解释收在模型 crate，而不是让 server 硬编码。

这一步之后，server 的工作就结束了。接下来，控制权进入模型引擎内部。

## 阶段 4：scheduler 接收请求并决定本步执行计划

### 最值得跟的文件

- `openinfer-qwen3-4b/src/scheduler.rs`
- `openinfer-qwen3-4b/src/scheduler/plan.rs`
- `openinfer-qwen3-4b/src/scheduler/resolve.rs`
- `openinfer-qwen3-4b/src/scheduler/effects.rs`

### Qwen3 scheduler 的线程模型

`scheduler.rs` 的头部注释就已经把模型说得很清楚：

- 专用 GPU 线程；
- frontend handler 把 `GenerateRequest` 送进 channel；
- scheduler 线程独占 model / batch buffers / KV 资源；
- 它每一轮选择 prefill、decode 或 unified step。

关键内部状态分两类：

- `PendingRequest`：已接收但还没完成 prefill 的请求；
- `ActiveRequestState`：已经在 decode 中的请求。

### `PendingRequest` 和 `ActiveRequestState` 的区别

`PendingRequest` 持有：

- 完整 prompt tokens
- 采样参数
- token sink
- echo / logprobs / prefill chunk 相关状态
- prefix cache / prefetch / prefill progress 状态

`ActiveRequestState` 则持有：

- 最近一个 token
- 已生成 token 数
- max_tokens
- prompt_len
- 采样参数
- token sink

这个拆分很关键：它让 scheduler 清楚地区分“还在补 prompt 的请求”和“已经进入逐 token decode 的请求”。

### 每一轮 scheduler 如何决定做什么

真正的 truth table 在 `openinfer-qwen3-4b/src/scheduler/plan.rs`：

```rust
pub(super) fn build_next_plan(
    have_active: bool,
    pending: Vec<PendingRequest>,
) -> Option<ExecutionPlan>
```

策略非常清楚：

- 只有 pending -> `Prefill`
- 只有 active -> `Decode`
- 两者都有 -> `Unified`
- 都没有 -> idle

也就是说，batch 形成策略并不是散在很多地方，而是收在 `plan.rs` 里。

### 为什么还有 `resolve.rs` 和 `effects.rs`

这两个文件是 Qwen3 scheduler 很值得学的拆分：

- `resolve.rs`：把 executor 结果翻译成“这一步应该发生哪些效果”；
- `effects.rs`：真正把这些效果应用到 active/prefilling 队列，并 emit `TokenEvent`。

这让代码结构变成：

```text
executor result
  -> resolve_step(...)
  -> StepEffects
  -> apply_effects(...)
```

好处是：

- 计划、执行、结果解释、状态应用被分层；
- 很容易看清 `Scheduled` / `Token` / `Finished` 是在哪一步被发出去的；
- scheduler 逻辑不会退化成一个巨型 `match + side effects` 大函数。

## 阶段 5：executor 真正驱动 prefill / decode / unified forward

### 最重要的文件

- `openinfer-qwen3-4b/src/executor.rs`
- `openinfer-qwen3-4b/src/prefill.rs`
- `openinfer-qwen3-4b/src/batch_decode.rs`
- `openinfer-qwen3-4b/src/unified_forward.rs`
- `openinfer-qwen3-4b/src/batch_decode_buffers.rs`

### executor 在系统里的角色

如果说 scheduler 回答的是“这一步做什么”，那么 executor 回答的是“怎么在 GPU 上做这一步”。

它主要持有：

- 模型与 device context
- batch decode buffers
- KV cache / prefix cache / offload 相关对象
- sampling / logits 提取辅助
- worker lanes / step command / result 组装逻辑

### 为什么 executor 不是一个 `generate()` 函数

看 `PrefillStepItem` / `DecodeStepItem` 这类结构体，你会发现 executor 并不是“输入 prompt，输出文本”的玩具接口，而是严格的 phase boundary：

- prefill 路径有自己的 step item 和 result；
- decode 路径有自己的 step item 和 result；
- unified 路径则同时吃 prefill request 和 decode request。

这也是 openinfer 适合做 continuous batching 的关键。

### Qwen3 executor 里最值得注意的点

#### 1. 它理解 chunked prefill

`PrefillStepItem` 里有：

- `cached_tokens`
- `chunk_budget`
- `chunk_start`
- `chunk_tokens`

这说明 executor 不只是“跑一次前向”，它还理解：

- prefix-cache hit 的前缀不必重算；
- 大 prompt 可能被切成多个 chunk；
- 每一步只 forward 当前 chunk。

#### 2. 它理解 logprobs / echo 这种服务层需求

`build_prefill_request_results(...)` / `build_decode_request_results(...)` 里会按请求决定是否提取 logprob，以及 echo prompt logprobs。

这说明服务层语义并没有全都留在 server；其中一部分与 logits 直接相关的语义必须在 executor 完成。

#### 3. 它理解 stop token / length limit 的区别

scheduler 的 `resolve.rs` 与 `effects.rs` 再加 executor 的结果结构一起，保证了：

- EOS stop 与 length stop 能分开表达；
- length stop 需要先 emit 最后一个 token 再 `Finished`；
- stop token 则可能直接 `Finished` 而不 emit stop token。

## 阶段 6：`TokenEvent` 从模型引擎回流到 frontend

### Qwen3 路径里事件在哪里发出

关键文件：`openinfer-qwen3-4b/src/scheduler/effects.rs`

你会看到几种典型 emit：

- `TokenEvent::Scheduled`
- `TokenEvent::PromptTokens`
- `TokenEvent::Token`
- `TokenEvent::Finished`

以及遇到不合法请求或运行中异常时的：

- `TokenEvent::Rejected`
- `TokenEvent::Error`

注意：这些 event 不是 frontend 自己猜出来的，而是模型 scheduler 明确发出的运行时事实。

### frontend bridge 如何处理事件 burst

关键文件：`openinfer-vllm-frontend/src/bridge.rs`

bridge 做的不是“每来一个 event 就立刻回一个 HTTP chunk”那么简单。它会：

1. 从共享 token channel drain 一个 ready burst；
2. 按 `RequestTag` 把 event 分桶；
3. 对每个请求调用 `reduce_request(...)`；
4. 把这一 burst 折叠成一批 frontend outputs；
5. 一次性发回。

这个设计的意义是：

- 减少 N 请求并发时的消息风暴；
- 保持每个请求内部事件顺序；
- 在 frontend 层把 `Scheduled + Token + Finished` 折叠成更适合上层消费的输出语义。

### `reduce_request(...)` 真正做了什么

你至少要知道它处理了这些情况：

- 只收到 `Scheduled` -> 暂存 metadata，不立刻输出；
- 收到 `Token` -> 累加 token ids / logprobs；
- 收到 `Finished` -> 标记 terminal，并附 finish reason；
- 收到 `Error` / `Rejected` -> 转成 error finish 语义；
- `PromptTokens` 当前被 bridge 有意忽略（prompt logprobs 暂未通过这条路径上送）。

所以从前端视角看，`TokenEvent` 不是直接暴露给 HTTP 层，而是先被 bridge reduction 了一次。

## 阶段 7：取消、拒绝、失败分别在哪一层发生

### 取消（cancel）

取消不是通过“关闭独立请求 channel”，而是通过 `TokenSink` 背后的 `cancelled` flag。

这意味着：

- 每个请求共享输出通道；
- 某个请求 abort 时，只翻自己的 flag；
- scheduler 下次 emit 时发现 `send()` 失败或 `is_closed()`，就把该请求退休。

这是共享输出通道设计下很关键的配套机制。

### 拒绝（rejected）

拒绝通常发生在 scheduler admission 之前或之时。

典型原因：

- prompt 太长；
- 请求理论上永远塞不进 KV 容量；
- sampling 参数非法；
- LoRA adapter 不存在；
- 模型 feature 没开。

Qwen3 里可以在 `scheduler.rs` 和 `resolve/effects` 周边看到很多 `TokenEvent::Rejected` 路径；Kimi 的 `runner/scheduler/lifecycle.rs` 也把 unschedulable 请求判断收得比较清楚。

### 失败（error）

失败是“本来被接受了，但运行时出错”。

比如：

- decode 路径中途出错；
- worker lane 执行失败；
- 后端通信或 kernel 调用失败。

这时发的是 `TokenEvent::Error`，而不是 `Rejected`。

这两者对服务层和使用者语义完全不同：

- `Rejected` = 根本没被调度成功；
- `Error` = 已经开始处理但中途失败。

## 阶段 8：为什么 Qwen3 是读这条主线的最佳样本

因为 Qwen3 同时拥有：

- server-facing `launch(...)`
- scheduler / executor 分层
- prefill / decode / unified path
- prefix cache / offload / LoRA 等真实 serving 复杂性
- 相比 Kimi / DeepSeek 更低的架构噪音

它足够真实，又没有一下子把你扔进 MLA、MoE、TP/DP/EP 的深水区。

## 一条建议的源码跟读顺序

### 第一次跟读

1. `openinfer-server/src/main.rs`
2. `openinfer-server/src/server_engine.rs`
3. `openinfer-vllm-frontend/src/bridge.rs`
4. `openinfer-engine/src/engine.rs`
5. `openinfer-qwen3-4b/src/lib.rs`
6. `openinfer-qwen3-4b/src/scheduler.rs`
7. `openinfer-qwen3-4b/src/scheduler/plan.rs`
8. `openinfer-qwen3-4b/src/scheduler/resolve.rs`
9. `openinfer-qwen3-4b/src/scheduler/effects.rs`
10. `openinfer-qwen3-4b/src/executor.rs`

### 第二次跟读

1. `openinfer-qwen35-4b/src/lib.rs`
2. `openinfer-deepseek-v2-lite/src/lib.rs`
3. `openinfer-deepseek-v4/src/lib.rs`
4. `openinfer-kimi-k2/src/lib.rs`

第二次的目标不是再跟一遍完整时序，而是对比：

- 哪些部分仍然走同一个统一契约；
- 哪些复杂性被各模型 crate 自己接住了。

## 读完这篇后你该带走什么

如果你只能带走三句话，就带走这三句：

1. **server 只做入口和分发，不做模型内部策略。**
2. **`GenerateRequest` / `TokenEvent` 是整个系统的统一语言。**
3. **真正的运行时复杂性在模型 crate 的 scheduler/executor 里。**

下一步建议看：`docs/playbooks/openinfer-learning/source-tour.md`，再配合 `docs/playbooks/openinfer-learning/subsystem-invariants.md` 一起读会更稳。

