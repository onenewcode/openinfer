# OpenInfer Subsystem Invariants

> **TL;DR:** 这份文档不解释“代码怎么写”，而解释**哪些设计约束不能轻易破**。如果你准备改 scheduler、executor、KV、bridge、engine contract，这些不变量比具体实现细节更重要，因为它们定义了这个系统为什么还能继续扩模型、扩前端、扩并行，而不在每次修改后整体散架。

## 为什么要单独写“不变量”

源码导读告诉你“现在是怎么做的”，但不变量告诉你“为什么不能随便改”。

这类仓库最容易出的问题不是编译错误，而是：

- 看起来改动很小，但把边界搞坏了；
- 看起来性能 patch 很局部，但破坏了服务语义；
- 看起来抽象更统一，但把模型专属复杂性污染到了共享层。

所以这份文档的目标是：给后续改代码的人一个“先验护栏”。

## 不变量 1：server 只做入口与分发，不做模型内部策略

### 约束内容

`openinfer-server` 负责：

- 读配置；
- 识别模型；
- 把请求交给 frontend / engine；
- 调用各模型 crate 的 `launch(...)`。

它**不负责**：

- 决定具体 attention 路径；
- 决定某模型支持哪些 graph / cache / offload / EP 约束；
- 决定某模型的运行时线程/worker/拓扑组织。

### 证据

- `openinfer-server/src/main.rs` 的 `load_engine(...)` 是纯分发函数。
- `openinfer-server/src/server_engine.rs` 只做 `config.json` 识别和 feature-gated dispatch。
- Qwen3 / Qwen3.5 / DeepSeek / Kimi 的 `launch(...)` 都在模型 crate 自己的 `src/lib.rs`。

### 如果你破坏它，会发生什么

- server 会开始长出模型专属 if/else；
- 新模型接入会越来越依赖 server 改动；
- 模型线无法独立演进，导致“加一个模型 = 改所有层”。

### 一个自检问题

> 我现在想加的这段逻辑，是所有模型都需要，还是只有这条模型线需要？

如果答案是后者，它大概率不该进 server。

## 不变量 2：`GenerateRequest` / `TokenEvent` 是统一语言

### 约束内容

所有真实模型引擎对上层暴露的提交 / 输出契约，统一收敛到：

- 输入：`GenerateRequest`
- 输出：`TokenEvent`
- 入口：`EngineHandle`

这意味着：

- frontend bridge 不需要理解每条模型线的内部状态机；
- bench / test / sim 可以复用统一请求/事件接口；
- 模型线的复杂性可以藏在内部，而不是扩散到调用者。

### 证据

- `openinfer-engine/src/engine.rs` 定义了 `GenerateRequest`、`TokenEvent`、`TokenSink`、`EngineHandle`。
- `openinfer-core/src/engine.rs` 直接 re-export 这套契约。
- `openinfer-vllm-frontend/src/bridge.rs` 把前端请求翻译成 `GenerateRequest`，再把事件流折叠回 frontend output。

### 子约束：terminal event 语义不能乱

至少要维持这几个区别：

- `Rejected`：没被接纳进入正常执行
- `Error`：接纳后运行中出错
- `Finished`：正常终止（`Stop` / `Length`）

一旦把这些语义混了，frontend、bench、调试工具都会误判系统行为。

## 不变量 3：shared token channel + per-request tag 是性能与结构双重约束

### 约束内容

`TokenSink` 背后不是“每请求一个输出 channel”，而是：

- engine-wide shared token stream
- per-request `RequestTag`
- frontend bridge 统一 demux
- per-request cancel flag

### 证据

- `openinfer-engine/src/engine.rs` 中 `TokenStreamSender` / `TokenStreamReceiver`、`RequestTag`、`TokenSink` 的定义。
- `openinfer-vllm-frontend/src/bridge.rs` 里 `dispatch_burst(...)` 和 `reduce_request(...)` 的按 tag 分桶逻辑。

### 不能轻易改成 per-request channel 的原因

- N 个请求意味着 N 个 sleeping consumer / wakeup；
- bridge 已经围绕“shared burst -> bucket -> reduce”建立了结构；
- cancel 语义现在通过 per-request flag 模拟旧的“receiver dropped”效果。

### 结论

如果你要改输出分发模型，必须同时重新证明：

- 性能不会倒退；
- cancel 语义不会变脏；
- frontend bridge 的 burst reduction 仍然成立。

## 不变量 4：scheduler 与 executor 的职责必须分层

### 约束内容

- scheduler 决定：谁在这一步跑、这一步是 prefill / decode / unified、何时退休请求
- executor 决定：这一步在 GPU 上怎么执行、用什么 buffer/state/operator 组合

### 证据

- `openinfer-qwen3-4b/src/scheduler/plan.rs` 负责计划选择与执行调用。
- `openinfer-qwen3-4b/src/scheduler/resolve.rs` 把结果翻译成 effects。
- `openinfer-qwen3-4b/src/scheduler/effects.rs` 应用状态变化并 emit events。
- `openinfer-qwen3-4b/src/executor.rs` 持有执行态、phase item、结果结构和 worker lane 组织。

### 为什么这条边界不能塌

如果 scheduler 开始直接塞满底层 operator 细节，或者 executor 开始主导 admission / lifecycle：

- unified step 会变得难以推理；
- `Scheduled` / `Token` / `Finished` 的服务语义会被埋进实现细节；
- chunked prefill / prefix cache / cancellation 等跨阶段逻辑更难维护。

## 不变量 5：Qwen3 scheduler 的计划选择是真正的批处理策略中心

### 约束内容

Qwen3 当前的 truth table 是：

- pending only -> `Prefill`
- active only -> `Decode`
- both -> `Unified`
- none -> idle

### 证据

- `openinfer-qwen3-4b/src/scheduler/plan.rs::build_next_plan(...)`

### 为什么它重要

这不是一个“无关紧要的小函数”，而是整个连续批处理策略的中心。你改它，相当于改了：

- 新请求何时进入 GPU；
- 活跃 decode 是否会与 prefill 融合；
- scheduler 在混合负载下的吞吐和延迟形态。

### 修改前必须回答的问题

- 为什么新策略在 mixed load 下更合理？
- 会不会把原先的事件语义打乱？
- 是否破坏 chunked prefill / prefix cache 的配合？

## 不变量 6：`Scheduled` 语义是一条请求只发一次，而且时间戳有明确含义

### 约束内容

对 Qwen3 路径，`TokenEvent::Scheduled`：

- 只在请求第一次真正进 GPU 工作时发送；
- `scheduled_at_unix_s` 在 forward 之前打点，而不是之后；
- `cached_tokens` 代表 prefix-cache hit 的已知值。

### 证据

- `openinfer-qwen3-4b/src/scheduler/resolve.rs`：`req.prefill_pos == 0` 时才 push `ScheduledEffect`
- `openinfer-qwen3-4b/src/scheduler/plan.rs`：`scheduled_at_unix_s` 在 executor 运行前获取
- `openinfer-qwen3-4b/src/scheduler/effects.rs`：真正 emit `TokenEvent::Scheduled`

### 为什么这很重要

因为 frontend / bench 用它拆分：

- queue time
- prefill time
- prefix-cache usage

如果你改坏这里，性能指标会直接失真。

## 不变量 7：length stop 和 stop-token stop 不是一回事

### 约束内容

- stop-token 终止：可能不 emit stop token，而是直接 `Finished { Stop }`
- length limit 终止：通常要先 emit 最后一个 token，再 `Finished { Length }`

### 证据

- `openinfer-qwen3-4b/src/scheduler/resolve.rs` 区分 `Finish` 与 `EmitAndFinish`
- `openinfer-qwen3-4b/src/scheduler/effects.rs` 里分别处理 `Finish` 和 `EmitAndFinish`

### 为什么这不能混

因为上层语义不同：

- stop-token 是模型语义终止
- length stop 是服务限制终止

如果你把二者混为一类，客户端行为、基准统计和 correctness case 都会变得不可信。

## 不变量 8：prefix cache / chunked prefill / unified step 是一起成立的，不是独立 patch

### 约束内容

Qwen3 的 serving 路径里：

- prefix cache 决定部分 prompt 不必重算；
- chunked prefill 允许超长 prompt 分多步 forward；
- unified step 允许 pending prefill 和 active decode 共存。

这三者不是互不相关的开关，而是联动设计。

### 证据

- `PendingRequest` 里同时持有 `prefetch_offered`、`prefill_pos`、`step_chunk`、`cached_tokens`
- `take_prefill_chunks(...)` 管 chunk 选择
- `resolve_prefill_outputs(...)` 处理 continue-prefill 与 scheduled/prefix-cache event 语义
- executor 的 `PrefillStepItem` 带 `chunk_budget` / `chunk_start` / `chunk_tokens`

### 修改前必须回答的问题

- 改动会不会让 chunk continuation 破坏 FIFO？
- prefix-cache hit 是否还会在 first chunk truthfully 报告？
- unified step 下 active decode 的语义是否仍然成立？

## 不变量 9：模型复杂度应尽量留在模型 crate，不往 `openinfer-core` 回流

### 约束内容

- Qwen3.5 的 recurrent/hybrid complexity 留在 `openinfer-qwen35-4b`
- DeepSeek-V4 的 direct engine / MP8 complexity 留在 `openinfer-deepseek-v4`
- Kimi-K2 的 runner/worker/load_balancer/TP-DP-EP complexity 留在 `openinfer-kimi-k2`

### 为什么这是长期可扩展性的核心

如果这些复杂性都被抽回 `openinfer-core`：

- core 会失去“共享积木”的边界；
- 任何新模型都要先理解所有旧模型特例；
- 项目会从“多模型共享底座”退化成“一个巨型框架 + 很多分支”。

## 不变量 10：feature gate 不只是编译优化，也是架构边界

### 约束内容

模型线 feature gate 的作用不仅是“少编一点”，而是：

- 避免默认路径背上特殊依赖；
- 把 Python/Triton/多卡/特定硬件要求收在明确边界内；
- 让 server 的支持矩阵可见且明确。

### 证据

- `openinfer-server/Cargo.toml` 的各模型 feature
- `openinfer-qwen35-4b` 需要 `qwen35-4b` feature 才会启用 Triton AOT 路径
- `openinfer-kimi-k2`、`openinfer-deepseek-v4` 也是明确 feature-gated

### 实践含义

如果一个新模型需要特殊工具链或硬件，优先考虑 feature gate，而不是让默认路径隐式背锅。

## 不变量 11：frontend bridge 的职责是 reduction，而不是重新定义引擎语义

### 约束内容

bridge 可以：

- bucket 一个 burst
- reduce 一个请求的一串 event
- 把低层事件折叠成前端输出对象

但它不应该：

- 发明新的引擎生命周期语义；
- 猜测模型内部状态；
- 掩盖 `Rejected` / `Error` / `Finished` 的差异。

### 证据

- `openinfer-vllm-frontend/src/bridge.rs::reduce_request(...)` 只是 fold event stream，不重新解释模型内部计划。

### 为什么这条边界重要

如果 bridge 开始“脑补”低层语义，那么：

- 不同模型线 event 细节稍变，前端就可能 silently 失真；
- 调试时很难判断是引擎错了，还是 bridge 错了。

## 不变量 12：任何“补全文档”都应该沿着主路径和边界来写，而不是堆命令

### 约束内容

对于这类仓库，真正高价值文档应优先覆盖：

- 主路径：请求怎么走
- 边界：哪层负责什么
- 不变量：哪些设计约束不能乱动
- 模板：新模型怎么接入

而不是只堆：

- 一长串运行命令
- 孤立的 operator 笔记
- 没验证过的环境假设

### 为什么把这一条也写成不变量

因为文档本身就是这个项目的协作基础设施。如果文档不沿着主路径和边界组织，后续人还是会在代码里迷路。

## 改代码前的一张自查表

在你动以下子系统之前，先自问：

### 改 server 前

- 这真的是 server 的职责吗？
- 还是应该下沉到模型 crate `launch(...)`？

### 改 engine contract 前

- 新字段 / 新事件是所有模型都需要吗？
- frontend / bench / tests 是否都能承受这个改动？

### 改 scheduler 前

- 我是在改 batch policy，还是只是在改实现细节？
- 会不会破坏 `Scheduled` / `Finished` 语义？

### 改 executor 前

- 我是在改 phase boundary，还是只是局部 operator 组织？
- 这个复杂性真的值得共享吗？

### 改 bridge 前

- 我是在做 reduction，还是在偷偷重定义引擎语义？
- shared token channel 的性能前提还成立吗？

## 最后一句话

如果你只能记住一句不变量，就记住这句：

> **统一契约向上稳定，模型复杂性向下收敛。**

这几乎就是 openinfer 现在还能继续长的根本原因。

