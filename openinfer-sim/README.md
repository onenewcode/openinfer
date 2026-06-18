# openinfer-sim 工作原理

`openinfer-sim` 是一个**CPU-only 的模拟引擎 crate**。它不加载任何真实模型权重，也不做真实 GPU 推理；它做的事是：**用一套可配置的时间模型，伪造一个“看起来像真实引擎”的 `EngineHandle`，并通过现有的 vLLM/OpenAI 前端对外提供服务。**

如果只记一句话，可以这样理解：

- `openinfer-engine` 定义请求/事件协议；
- `openinfer-vllm-frontend` 负责 HTTP/OpenAI 兼容入口；
- `openinfer-sim` 在中间伪装成一个“会按 TTFT/TPOT 出 token 的引擎”。

## 它在系统里的位置

它的角色不是模型实现，而是**前端验证和 benchmark harness**：

```text
HTTP / OpenAI 请求
  -> openinfer-vllm-frontend
  -> openinfer-sim start_engine()
  -> 模拟 EngineHandle
  -> 按设定的时间参数吐 TokenEvent
```

所以它的典型用途是：

- 验证 frontend 行为；
- 跑 `vllm bench serve` 这类压测；
- 隔离“前端开销”和“真实模型/GPU 开销”。

## 它到底模拟了什么

`openinfer-sim` 只模拟四件事：

- **base TTFT**：首 token 的固定基础延迟
- **prefill throughput**：prompt 越长，TTFT 越大
- **TPOT**：后续 token 的固定间隔
- **fallback token id**：空 prompt 时用什么 token 占位

配置体 `SimulatedEngineConfig` 对应的就是这四个量：

- `base_ttft_ms`
- `prefill_tokens_per_ms`
- `tpot_ms`
- `fallback_token_id`

它并不模拟：

- 真实 logits
- 真实采样
- 真实 KV cache
- 真实 scheduler 资源竞争

所以它更像一个**延迟模型**，而不是“迷你 LLM”。

## TTFT 是怎么模拟的

这个 crate 用一个非常直接的公式模拟 TTFT：

```text
TTFT = base_ttft_ms + prompt_tokens / prefill_tokens_per_ms
```

也就是说：

- prompt 越长，TTFT 越大；
- 但增长方式不是按真实 kernel，而是按一个固定的“prefill 吞吐率”近似。

这刚好够前端和 benchmark 使用，因为它们通常只需要：

- 有一个随 prompt 长度变化的首 token 延迟；
- 不需要关心这个延迟背后是 attention、MoE 还是 CUDA Graph。

## TPOT 是怎么模拟的

后续 token 的生成更简单：每个 token 间隔一个固定的 `tpot_ms`。

流程是：

- 第一个 token 之前，先 sleep 一次 TTFT；
- 之后每生成一个新 token，如果不是第一个，就 sleep 一次 TPOT。

所以输出节奏就是：

```text
prompt 到来
  -> 等 TTFT
  -> 第 1 个 token
  -> 等 TPOT
  -> 第 2 个 token
  -> 等 TPOT
  -> 第 3 个 token
  ...
```

这让上层可以稳定观察：

- SSE/streaming 行为是否正确；
- bench 工具测到的 TTFT / TPOT 是否符合预期；
- 前端是否有额外开销。

## 为什么它能直接复用真实前端

`openinfer-sim` 的关键不是“自己造了一套前端”，而是：

- 它返回的是一个真实的 `EngineHandle`
- 发送的也是标准 `TokenEvent`

也就是说，在 `openinfer-vllm-frontend` 看来：

- 它根本不需要知道底下是 Qwen3 还是真模拟器；
- 只要事件流契约一致，前端逻辑就可以完全复用。

这正是把模拟器做成 crate，而不是做成测试 mock 的最大价值：**它跑的是完整前端路径。**

## `start_engine()` 实际做了什么

`start_engine(config)` 是这个 crate 的真正入口。

它做的事情很简单：

1. 创建一个 `submit_tx/submit_rx`
2. `tokio::spawn` 一个后台循环，不断接收 `GenerateRequest`
3. 每来一个请求，再 `tokio::spawn` 一个任务去跑 `run_simulated_request`
4. 返回 `EngineHandle::new(submit_tx)`

所以这不是一个“单线程串行假引擎”，而是一个：

- 收请求的总循环
- 每个请求各自异步处理

的轻量模拟器。

这让它足够像真实 serving 环境，至少在“多请求并发 + streaming 输出”这一层是像的。

## `run_simulated_request()` 的核心流程

这个函数几乎就是模拟器的全部行为。

它按顺序做：

1. 发 `TokenEvent::Scheduled`
2. 如果 `echo=true`，先发 `PromptTokens`
3. 如果 `max_tokens > 0`，先 sleep TTFT
4. 逐个 token：
   - 必要时 sleep TPOT
   - 发 `TokenEvent::Token`
5. 最后发 `TokenEvent::Finished`

这个顺序非常重要，因为它对前端来说就是“引擎的真实行为”。

换句话说，`openinfer-sim` 的模拟重点不在“token 内容像不像模型”，而在于：

- 事件顺序像不像真实引擎；
- 延迟分布像不像一个可配置的 serving 引擎。

## 为什么 token 内容是假的也没关系

`fake_token_id()` 的逻辑非常简单：

- 如果 prompt 不为空，就循环复用 prompt token
- 如果 prompt 为空，就返回 `fallback_token_id`

看起来很“假”，但这是刻意的：

- 前端测试通常不在意语义是否合理；
- benchmark 也主要关心时延、吞吐、流式行为；
- 只要 token 能稳定地产生，事件流就能被完整驱动。

所以这里的 token id 设计目标不是“自然语言合理”，而是“稳定、可预测、零模型依赖”。

## logprobs 是怎么处理的

当请求里 `logprobs > 0` 时，模拟器会给每个 token 附一个最小 `TokenLogprob`：

- `logprob = 0.0`
- `top_logprobs = []`

这说明它只是满足“这个接口字段存在”的契约，而不是模拟真实 logits 分布。

因此它足够做：

- 前端格式兼容验证；
- 某些 client 对 logprobs 字段存在性的测试；

但并不适合做：

- 任何基于真实概率分布的正确性验证。

## CLI 二进制是怎么接前端的

`src/main.rs` 里，`openinfer-sim` 作为一个可执行程序时，会：

1. 解析命令行参数
2. 构造 `SimulatedEngineConfig`
3. 调 `start_engine(config)` 拿到 `EngineHandle`
4. 把这个 handle 交给 `openinfer_vllm_frontend::serve(...)`

这意味着：

- `openinfer-sim` 自己不实现 HTTP server；
- 它只是把“模拟出来的引擎”接到真实的 frontend 上。

所以当你跑：

```text
cargo run -p openinfer-sim --release -- ...
```

你其实是在启动：

- 一个真实的 OpenAI/vLLM 风格服务端入口
- 只不过底下挂的是模拟引擎，不是真模型

## 为什么它对前端 profiling 很有用

因为它把“模型开销”几乎剥离掉了。

真实模型路径里，前端看到的总延迟混着：

- tokenizer
- frontend bridge
- scheduler
- GPU compute
- KV / sampling / network

而 `openinfer-sim` 能把其中“模型/GPU compute”替换成一个已知、可控的时间函数。

于是当你测到：

- TTFT 比设定值多很多；
- TPOT 比设定值有额外抖动；
- CPU profile 主要花在 IPC / SSE / 分配；

你就能更有把握地说：这是**前端路径**的问题，不是模型本身的问题。

## 从一次请求的角度看完整流程

把它串起来，一次请求在这个 crate 里的路径大概是：

```text
1. 前端构造 GenerateRequest
2. 通过 EngineHandle::submit() 送进模拟引擎
3. 后台循环收到请求
4. spawn 一个 request task
5. 先发 Scheduled
6. 按需发 PromptTokens
7. sleep(TTFT)
8. 每个 token 间隔 sleep(TPOT) 并发 Token
9. 最后发 Finished
```

所以如果只记一句话，可以把 `openinfer-sim` 理解成：

**“拿真实引擎协议 + 真实前端路径，套一个可配置延迟模型，专门用来测服务层而不是测模型。”**

## 这个 crate 刻意没有做的事

为了看清边界，也值得明确它**没有**做什么：

- 不加载权重；
- 不做真实 tokenization 推理；
- 不做真实 sampling；
- 不做真实 KV cache / scheduler 资源模型；
- 不给性能结论提供真实模型依据。

它的使命很明确：让你在**没有 GPU、没有模型、或者不想让模型开销干扰分析**时，仍然能完整跑通 openinfer 的 frontend/serving 路径。
