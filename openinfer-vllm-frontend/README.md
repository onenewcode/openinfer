# openinfer-vllm-frontend 工作原理

`openinfer-vllm-frontend` 是 openinfer 的 **HTTP / OpenAI 兼容前端桥接层**。它本身不做模型推理；它做的事是：**把 openinfer 的本地 `EngineHandle` 伪装成 vLLM 前端期望的 engine-core transport，让现成的 vLLM/OpenAI 路由、tokenizer、chat template、SSE 输出层都能直接复用。**

如果只记一句话，可以这样理解：

- `openinfer-engine` 负责“本地引擎协议”；
- `vllm-server` 负责“HTTP/OpenAI 前端”；
- `openinfer-vllm-frontend` 负责把这两套世界接起来。

## 它在系统里的位置

这个 crate 处在“server 与本地模型引擎之间”的桥接位置：

```text
HTTP client
  -> vllm-server 路由 / tokenizer / chat template
  -> openinfer-vllm-frontend
       -> LocalEngineBridge
       -> EngineHandle / TokenEvent
  -> 本地 openinfer 模型引擎
```

所以它不是另一个 server，也不是模型 runtime；它更像一个**本地版 engine-core adapter**。

## 这个 crate 做的三件大事

从代码结构看，它主要负责三件事：

### 1. 启动前端并等待本地引擎接入

`serve(...)` / `serve_model_on_host(...)` 负责：

- 提前启动 vLLM/OpenAI 前端；
- 在前端加载 tokenizer / chat template 的同时等待 engine future；
- engine ready 后启动本地 bridge；
- 只有 bridge 注册成功后，HTTP 才真正 bind 对外可服务。

这就是为什么它能把启动时间压短：**前端初始化和引擎加载并行进行**，而不是完全串行。

### 2. 把本地 `EngineHandle` 翻译成 vLLM engine-core transport

这部分主要在 `bridge.rs` 的 `LocalEngineBridge`。

它做的事是：

- 伪造一个本地 IPC namespace；
- 用 ZMQ socket 连上 vLLM 前端期望的 input/output transport；
- 把前端发来的 `EngineCoreRequest` 翻译成 openinfer 的 `GenerateRequest`；
- 再把本地引擎吐回来的 `TokenEvent` 聚合成 `EngineCoreOutputs` 发回前端。

换句话说，它把：

```text
EngineCoreRequest <-> EngineHandle / TokenEvent
```

这两套协议粘在了一起。

### 3. 守住 openinfer 不支持的参数边界

这部分主要在 `sampling_guard.rs`。

因为 vLLM/OpenAI 前端暴露的参数面，比 openinfer 当前真实支持的要宽，所以如果直接放行，很多参数会“看似成功，实际被静默忽略”。

这个 guard 的职责就是：

- 把会被静默降级的参数提前拒绝；
- 把超过 servable cap 的 `max_tokens` 提前拒绝；
- 确保前端行为是 fail-closed，而不是 silently degraded。

## `serve()` 为什么能在引擎没加载完时先启动

这是这个 crate 最关键的设计之一。

`serve()` 接收的不是现成 `EngineHandle`，而是：

```rust
impl Future<Output = Result<EngineHandle>>
```

这意味着：

- 前端可以先开始加载 tokenizer / backend / router；
- engine future 可以并行加载模型和 GPU 资源；
- 等 engine future resolve 后，再把 `LocalEngineBridge` 接上去。

但有一个很重要的约束：

- **HTTP 端口只有在 bridge 注册成功后才真正对外 ready**

所以并发启动并没有改变 readiness 语义，只是把两个耗时阶段重叠了。

## `LocalEngineBridge` 到底在翻译什么

`LocalEngineBridge` 是这整个 crate 的核心。

它主要做两种翻译：

### 1. 请求方向：`EngineCoreRequest -> GenerateRequest`

当前端通过 engine-core 协议发来请求时，bridge 会：

- 解析 `EngineCoreRequest`
- 取出 `prompt_token_ids`
- 把 sampling params 转成 openinfer 的 `SamplingParams`
- 解析额外的 LoRA adapter 参数
- 创建一个带 `RequestTag` 的 `TokenSink`
- 最后通过 `EngineHandle::submit()` 送进本地引擎

所以 bridge 在这里像一个“协议降落器”：把 vLLM 那边的 wire request 落成本地引擎能吃的请求对象。

### 2. 输出方向：`TokenEvent -> EngineCoreOutputs`

本地引擎吐出来的是 `TokenEvent` 流，而前端期待的是 `EngineCoreOutputs`。

bridge 做的事包括：

- 按 request tag 把共享事件流 demux 回每个请求；
- 把一串 `TokenEvent` 折叠成一个或多个 `EngineCoreOutput`；
- 把 `Scheduled` 事件变成 engine-core 侧的 prefill stats / timing events；
- 把 `Finished` / `Error` / `Rejected` 转成对应的 terminal output。

也就是说，bridge 不只是“中转”，它还负责把 openinfer 本地事件流整理回 engine-core 的输出语义。

## 为什么输出要做 burst coalescing

`dispatch_burst()` 这段逻辑很重要。

本地引擎现在把所有请求的 token event 都发到一个共享 channel 里。bridge 每次收到一条事件时，不是立刻发一个 ZMQ message，而是：

- 把当前 ready 的 burst 全部 drain 出来；
- 按 request 分桶；
- 每个 request 的一批事件折叠成最多一个 output；
- 再把整个 burst 一次性打成一个 `EngineCoreOutputs` 发出去。

这样做的收益是：

- 降低 per-token / per-request 的 IPC 发送次数；
- 把以前 N 个请求一轮 step 可能产生的 N 条输出消息，压缩成 1 条；
- 减少 bridge / frontend 之间的 wakeup 和 ZMQ 消息数量。

所以这个 bridge 不是傻转发，而是做了一层**输出侧批处理**。

## 为什么 sampling guard 很关键

前端能表达的参数面很宽，比如：

- `presence_penalty`
- `frequency_penalty`
- `min_p`
- `use_beam_search`
- `logit_bias`
- `structured_outputs`
- `stop_token_ids`
- `prompt_logprobs`

但 openinfer 当前真实支持的并没有这么多。

如果不拦：

- 请求会在 API 层“成功”；
- 但引擎层悄悄忽略这些参数；
- 用户拿到的结果看起来像成功，实际语义已经变了。

所以 `sampling_guard` 的职责就是：

- 在统一的入口 parse body；
- 找出第一个不支持的参数；
- 用 OpenAI 风格错误直接拒绝。

可以把它理解成：**前端兼容面很宽，但 openinfer 只公开自己真正支持的那一块。**

## `ServableCap` 是怎么和前端一起工作的

`ServableCap` 是一个小结构，但非常关键。

它表达的是：

- 这台引擎最终能服务的最大生成长度是多少；
- 这个值在 engine 真正 ready 前还没法确定；
- 但 guard 又需要在处理 HTTP 请求时读到它。

所以这里用了一个 `OnceLock<Option<u32>>`：

- engine future resolve 后 set 一次；
- router/guard 启用后只读；
- 如果 guard 读到 unset，直接 loud reject，而不是带着未知 cap 继续服务。

这保证了：

- 前端可以提前启动；
- 但请求校验仍然基于最终引擎能力，而不是拍脑袋。

## LoRA 路由为什么放在这个 crate

LoRA 路由放在这里很合理，因为它本质上还是“前端/控制面扩展”。

`lora.rs` 做了两类事情：

### 1. 控制路由

- `/v1/load_lora_adapter`
- `/v1/unload_lora_adapter`

它们会调用 `EngineHandle` 的控制接口，把请求发给底层引擎。

### 2. OpenAI 路由扩展

- `/v1/models`
- `/v1/completions`
- `/v1/chat/completions`

这里会把 LoRA adapter 名字作为额外参数塞进 sampling params，让 bridge 再把它降到本地引擎请求。

所以 LoRA 在这个 crate 里的定位不是“模型实现”，而是：**给现有前端增加 adapter-aware 的控制面和路由层。**

## `wire.rs` 为什么存在

`wire.rs` 负责一些细小但很重要的协议转换：

- vLLM sampling params -> openinfer `SamplingParams`
- 本地 `FinishReason` -> engine-core `FinishReason`
- 本地 `TokenLogprob` -> wire `PositionLogprobs`
- 从 `extra_args` 里解 LoRA adapter

这层很重要，因为 bridge 的职责不只是“连通”，还要**把两边字段语义对齐**。

例如 `ignore_eos` 的判定就不能简单看 stop token 集合，而要按前端 lowering 后的真实语义来推。

## 从一次请求的角度看完整路径

把整个 crate 串起来，一次请求大致是：

```text
1. HTTP 请求进入 vLLM/OpenAI 路由
2. sampling_guard 检查 unsupported params / servable cap
3. vllm-server 生成 EngineCoreRequest
4. LocalEngineBridge 收到 request
5. bridge 把它翻译成 GenerateRequest 并 submit 到 EngineHandle
6. 本地引擎开始跑，并发回 TokenEvent
7. bridge 把事件 burst 聚合成 EngineCoreOutputs
8. vllm-server 再把它转成 SSE / OpenAI 响应流返回给客户端
```

如果只记一句话，可以把 `openinfer-vllm-frontend` 理解成：

**“拿 vLLM 的前端壳子，套在 openinfer 的本地引擎协议外面。”**

## 这个 crate 刻意没有做的事

为了看清边界，也值得明确它**没有**做什么：

- 不做真实模型推理；
- 不做 scheduler；
- 不管理 KV cache；
- 不定义引擎契约；
- 不决定某个模型具体支持哪些算子。

它专注做的是：**前端兼容、协议桥接、参数守卫、以及 LoRA 路由扩展。**

