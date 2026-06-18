# openinfer-engine 工作原理

`openinfer-engine` 是 openinfer 的**最小引擎契约层**。它不负责真正推理，也不负责 GPU kernel；它只定义一件事：**上层如何把请求交给引擎、引擎如何把 token 事件流式吐回来、控制平面如何和引擎交互。**

如果只记一句话，可以把它理解成：

- `openinfer-engine` 定义“协议”；
- 具体模型 crate 定义“实现”。

## 这个 crate 在整体架构里的位置

它处在非常靠上的抽象层，但本身非常轻：

```text
frontend / server / bench / sim
  └─ EngineHandle / GenerateRequest / TokenEvent
       └─ 真实模型引擎 or 模拟引擎
```

也就是说：

- 前端、服务层、benchmark harness、模拟器，都靠这套类型和引擎打交道；
- 但这套 crate 本身并不知道底下跑的是 Qwen3、Qwen3.5、Kimi，还是一个 CPU-only simulator。

这正是它“lightweight engine contract”的含义。

## 这个 crate 主要定义了哪几件事

从 `src/lib.rs` 看，它只暴露三个模块：

- `engine`
- `sampler`
- `parallel`

可以分别理解成：

- **`engine`**：请求/事件/控制面契约
- **`sampler`**：请求里带的采样参数
- **`parallel`**：模型无关的并行拓扑描述

## `GenerateRequest`：一次生成请求长什么样

`GenerateRequest` 描述的是“上层交给引擎的一次生成任务”。

它包含的核心信息有：

- `request_id`
- `prompt_tokens`
- `params: SamplingParams`
- `max_tokens`
- `lora_adapter`
- `token_tx: TokenSink`
- `logprobs`
- `echo`

从这些字段可以看出，这个契约层故意只关心引擎真正需要知道的东西：

- 输入 token 是什么；
- 采样怎么做；
- 最多生成多少；
- 输出事件发到哪里；
- 是否带 LoRA；
- 是否要 echo prompt / logprobs。

它并不关心：

- 具体模型结构；
- 具体 scheduler；
- 具体 kernel；

这些都留给下层引擎实现。

## `TokenEvent`：为什么输出被设计成事件流

引擎不是一次性返回一个大结果，而是不断吐 `TokenEvent`。

这反映了 openinfer 的真实工作方式：请求会经历排队、调度、逐 token 生成、结束或报错，所以事件流比“返回一个 completion struct”更贴近运行时事实。

`TokenEvent` 主要有这些变体：

- `Scheduled`
- `Token`
- `PromptTokens`
- `Finished`
- `Error`
- `Rejected`

它们分别表达的是：

- **`Scheduled`**：请求什么时候真正进了调度器，以及 prompt/cached token 统计
- **`Token`**：正常生成出的一个 token
- **`PromptTokens`**：需要把 prompt token 自身也回传的场景（例如 echo）
- **`Finished`**：正常结束，带 finish reason
- **`Error`**：运行中失败
- **`Rejected`**：还没跑就被 admission / 参数校验之类的逻辑拒绝

这个设计最大的好处是：**上层不用知道引擎内部状态机，只需要按事件消费结果。**

## `TokenSink`：为什么不是每个请求一个 channel

这是这个 crate 里最值得理解的一点之一。

表面上看，`GenerateRequest` 里只带了一个 `TokenSink`。但 `TokenSink` 背后的设计，其实是在解决“多请求并发下的事件分发开销”问题。

现在的设计是：

- 整个引擎只有一个共享输出 channel：`TokenStreamSender`
- 每个请求自己的 `TokenSink` 只是在发事件时，自动打上 `RequestTag`
- 前端在另一端按 tag demux

也就是说，设计从：

```text
每个请求一个 channel
```

变成了：

```text
所有请求共享一个 channel
  + 每条事件带 request tag
```

这样做的原因很实际：

- 每请求一个 channel，会带来 N 个 sleeping consumer / wakeup
- 共享 channel 则把 fan-out 成本压成 1 个接收者

所以 `TokenSink` 的本质不是“普通 sender”，而是**带 request 身份的事件发射器**。

## 为什么取消不是靠关 channel，而是靠 `cancelled` flag

旧式设计里，常见做法是：

- 请求取消 = 对应 receiver drop 掉

但在共享 channel 模型下，这样做不成立，因为所有请求共用一个 receiver。

所以 `TokenSink` 改成了：

- engine-wide channel 是否关闭：看 `tx.is_closed()`
- 某个请求是否取消：看 `cancelled: Arc<AtomicBool>`

这意味着：

- 一个请求取消，不会把别的请求的输出通道也一起关掉；
- scheduler 仍然可以通过 `send()` / `is_closed()` 的返回值，把这个请求当成“消费者没了”，在下一个 emit 时自然退休。

也就是说，`TokenSink` 通过一个 per-request cancel flag，模拟出了“每请求独立关闭”的语义，但保留了共享输出通道的性能收益。

## `EngineHandle`：为什么它只是 handle，不是 engine 本体

`EngineHandle` 是上层握在手里的“提交句柄”，而不是引擎本身。

它内部主要包了：

- `submit_tx`
- 或者 `command_tx`
- 可选的 `join_handle`
- 额外的只读元数据，如 `servable_len`、`kv_capacity`

这说明它的角色非常明确：

- 负责把请求/控制命令送进引擎；
- 负责在最后一个 handle drop 时做线程清理；
- 负责向上层暴露少量“引擎能力信息”。

但它并不参与真正的推理循环。

## `EngineHandle` 为什么支持两种通道模式

从构造函数能看出来，`EngineHandle` 支持两类引擎后端：

- 只有 `submit_tx`：只支持 generate
- 有 `command_tx`：支持统一的 `EngineCommand`

后者能处理：

- `Generate`
- `Control(LoadLoraAdapter / UnloadLoraAdapter / ListLoraAdapters)`

这说明这个契约从一开始就考虑了：

- 不是所有引擎都支持动态 LoRA 控制；
- 但支持控制面的引擎，也不应该为 generate 再另外搞一套完全不同的 handle。

所以 `EngineHandle` 对控制面支持是 **opt-in** 的。

## LoRA 控制平面是怎么抽象的

LoRA 控制不是直接暴露“调用某个 executor 方法”，而是封装成：

- `EngineControlRequest`
- `EngineControlError`
- `EngineControlResult`

并通过 `oneshot` 取回结果。

这种设计有两个优点：

1. 上层看到的是明确的异步控制请求，而不是依赖引擎实现细节；
2. 控制面和生成面共用同一套 engine handle，而不会泄漏内部线程/状态结构。

所以可以把它理解成：**在 token 流之外，再给引擎加一条小型控制 RPC 通道。**

## `KvCapacity`：为什么契约层要知道 block 容量

`KvCapacity` 只包含两件事：

- `total_blocks`
- `block_size`

它看起来很小，但表达的是一个很重要的事实：

- scheduler 实际按 block 分配 KV，不是按 token 精确分配；
- 一个请求占多少 KV 容量，必须按 `ceil(tokens / block_size)` 算；

所以 `KvCapacity` 的作用是让上层在**不深入引擎内部**的前提下，也能大致判断：

- 一个请求是否根本不可能放进这台引擎；
- 一个 batch 的 footprint 会不会超出池子上限。

这对 benchmark harness、上层 admission 预估、诊断都很有用。

## `SamplingParams` 为什么很小

`SamplingParams` 只放了最核心的采样参数：

- `temperature`
- `top_k`
- `top_p`
- `ignore_eos`

它的设计重点不是“把 OpenAI / vLLM 的所有采样参数一把塞进来”，而是保留当前引擎层真正承诺支持的最小集合。

其中 `is_greedy()` 也很关键，它把“这次是否可以直接走 argmax”这件事收敛成一个公共判定：

- `temperature < 1e-5`
- 或 `top_k == 1`

这让上层 scheduler / executor / sampler 路径可以共享“greedy 还是 sampling”这一判断，而不是各处重新写一套。

## `ParallelConfig` 解决的是什么问题

`parallel.rs` 里的 `ParallelConfig` 是模型无关的并行拓扑描述。

它主要关心三件事：

- `tp_world`
- `dp_world`
- 推导出来的 `ep_world`

同时还能给出：

- 某个 global rank 的 `tp_rank`
- 某个 global rank 的 `ep_rank`
- 某个 `dp_rank` 对应的 `tp_group`

这说明它的定位不是“执行 collective 的后端”，而是：**用一个纯数据结构，把并行拓扑的信息稳定传下去。**

这正适合放在 engine contract 层，而不是放进具体模型 crate。

## 从外部视角看一次请求是怎么流动的

把这些类型串起来，一次请求的逻辑路径大致是：

```text
1. 上层构造 GenerateRequest
2. 给它挂一个 TokenSink
3. 调 EngineHandle::submit()
4. 引擎线程收到请求并开始调度
5. 调度/执行过程中不断 send(TokenEvent)
6. 前端按 RequestTag demux 事件
7. 请求结束时收到 Finished / Error / Rejected
```

如果需要控制 LoRA，则是另一条路径：

```text
1. 上层调用 EngineHandle::load_lora_adapter() / unload / list
2. handle 发送 EngineCommand::Control(...)
3. 引擎控制面处理后经 oneshot 返回结果
```

可以看到，`openinfer-engine` 的重点不是“引擎内部怎么实现”，而是“外部怎样稳定地驱动引擎，并消费它的结果”。

## 为什么这个 crate 很小反而很重要

这个 crate 的代码量不大，但它非常关键，因为它定义了多个子系统的共同语言：

- 真实模型引擎和模拟引擎都要说这套语言；
- 服务层和 benchmark harness 都要消费这套语言；
- LoRA 控制平面也要挂在这套语言上。

一旦这个层级定义清楚，底下的模型实现就可以自由演进，而不会频繁打破上层。

## 这个 crate 刻意没有做的事

为了看清边界，也值得明确它**没有**做什么：

- 不做模型推理；
- 不做 scheduler；
- 不做 GPU kernel 调用；
- 不做 KV cache 管理；
- 不做 HTTP/OpenAI 协议适配。

这些都属于更下层或更上层的职责。`openinfer-engine` 只负责定义最小而稳定的引擎契约。
