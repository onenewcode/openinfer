# openinfer-server 工作原理

`openinfer-server` 是 openinfer 的**组合入口 crate**。它自己不拥有某个模型的核心推理逻辑；它做的事是：**读 CLI、识别模型类型、根据 cargo feature 选择对应模型 crate 启动引擎，再把引擎挂到 `openinfer-vllm-frontend` 上对外提供服务。**

如果只记一句话，可以这样理解：

- 模型 crate 负责“怎么推理”；
- `openinfer-vllm-frontend` 负责“怎么对外说 OpenAI/vLLM 协议”；
- `openinfer-server` 负责“把哪一个模型引擎接到哪一个前端配置上”。

## 它在整体架构里的位置

它是整个 workspace 面向用户启动的主入口：

```text
CLI / binary
  -> openinfer-server
       ├─ config 解析参数
       ├─ detect_model_type
       ├─ launch 对应模型 crate
       └─ 挂到 openinfer-vllm-frontend
```

所以它的角色更像“composition root”或“组装层”，而不是模型 runtime 本身。

## 这个 crate 做的四件大事

### 1. 解析 CLI 并做参数校验

`config.rs` 定义了启动时的大部分参数面：

- `model_path`
- `served_model_name`
- `port`
- `cuda_graph`
- `enable_lora`
- `lora_modules`
- `device_ordinal`
- `tp_size`
- `dp_size`
- `ep_backend`
- `kv_offload`
- `no_prefix_cache`
- `max_prefill_tokens`

这些参数里有些是通用的，有些只对特定模型有效。

因此 `Args::validate(model_type)` 的职责就是：

- 在识别出模型类型之后，再做“这个参数对这类模型是否合法”的校验；
- 避免把本来只支持 Qwen3 的选项误传给别的模型路径。

所以这里的 CLI 不是纯粹的参数收集器，而是第一层 **model-aware 配置守门员**。

### 2. 识别当前模型属于哪一条模型线

`server_engine.rs` 里的 `detect_model_type()` 会读 `config.json`，根据内容判断：

- DeepSeek-V2-Lite
- DeepSeek-V4
- Kimi-K2
- Qwen3
- Qwen3.5

它不只是“认字符串”，还带 feature-gate 语义：

- 如果配置像 DeepSeek-V4
- 但当前构建没有开 `deepseek-v4`
- 那么它会明确报错，让你重新带 feature 构建

这意味着 `openinfer-server` 做的是：

- **运行时探测模型线**
- **结合编译时 feature 决定能不能启动**

这正是这个 binary crate 必须负责的事情。

### 3. 根据模型类型把请求分发到正确的模型 crate

真正的启动分发在 `main.rs` 里的 `load_engine()`。

它的思路很明确：

- `openinfer-server` 自己不实现模型；
- 每个模型 crate 自己暴露 `launch(...)`；
- server 只负责把 CLI 里的相关参数翻译成对应模型 crate 的 launch options。

例如：

- DeepSeek-V4 -> `openinfer_deepseek_v4::launch(...)`
- Kimi-K2 -> `openinfer_kimi_k2::launch(...)`
- Qwen3 -> `openinfer_qwen3_4b::launch(...)`
- Qwen3.5 -> `openinfer_qwen35_4b::launch(...)`

所以这里的核心原则是：

**server 负责挑 crate；模型 crate 负责自己的启动策略。**

这也是为什么代码里专门写了注释：“Pure dispatch”。

### 4. 把引擎挂到前端服务层

引擎起来之后，`openinfer-server` 还要决定：

- 是普通服务路径；
- 还是带 LoRA routes 的服务路径。

然后把 engine handle 交给：

- `openinfer::vllm_frontend::serve(...)`
- 或 `serve_model_with_lora_routes(...)`

所以 server 的最后一步其实是：

- 把本地引擎接到统一前端；
- 让用户最终看到的是一个 OpenAI/vLLM 风格服务端，而不是裸 engine。

## 为什么 engine load 放到 `spawn_blocking`

这是这个 crate 里很关键的一个优化点。

模型加载本身通常包含：

- 读配置
- mmap / 读 safetensors
- CUDA 上下文初始化
- H2D 上传权重

这些都可能很重，而且很多并不是标准 async IO。

所以 `main.rs` 里用了：

```rust
tokio::task::spawn_blocking(move || load_engine(...))
```

这样做的好处是：

- 前端的 tokenizer / chat template 初始化可以和 engine load 并行；
- 不会把 tokio runtime 的 async 执行线程长时间卡死；
- 最终 readiness 仍然由前端 bridge 注册时刻决定，不会提早暴露端口。

所以这里的核心目标是：**缩短 startup-to-ready，而不改变 ready 语义。**

## 为什么 LoRA 路径和普通路径不同

`enable_lora` 为真时，代码走的是“先拿到 handle，再建带 LoRA 路由的 router”的顺序。

原因是：

- LoRA routes 不是纯静态路由；
- 它们在构建时就需要 `EngineHandle`，因为后续 `/v1/load_lora_adapter` 等控制路由要直接调用 handle 上的控制接口。

所以 LoRA 路径不能完全像普通路径那样，把 engine future 延后到 bridge 内部再 resolve；它必须更早拿到 handle。

这也是为什么：

- 普通路径可以最大化 frontend / engine 并发启动；
- LoRA 路径则更偏顺序化一些。

## 为什么 `config.rs` 里有很多“看起来像模型细节”的参数

乍看之下，`openinfer-server` 的 CLI 好像知道很多模型细节，比如：

- `max_loras`
- `max_lora_rank`
- `kv_offload_host_gib`
- `deepseek_prefill_profile`
- `ep_backend`

这并不矛盾，因为 server 是用户进入系统的统一入口。

这些参数放在这里，不代表 server 拥有这些能力；更准确地说：

- server 负责把用户意图收集起来；
- 再把这些意图下发到真正拥有能力的模型 crate。

所以 `config.rs` 是“**统一入口的配置面**”，不是“统一实现层”。

## `vllm_frontend.rs` 为什么只是 re-export

`openinfer-server/src/vllm_frontend.rs` 只是：

```rust
pub use openinfer_vllm_frontend::*;
```

这说明一个历史演进点：

- 以前 bridge/fronted 逻辑可能在 server crate 里；
- 现在已经拆成独立 `openinfer-vllm-frontend` crate；
- server 这里只保留兼容 re-export。

从架构角度看，这种拆分是健康的：

- server 留在“组合层”
- frontend bridge 变成独立、可单独复用的 crate

## `scheduler.rs` / `sampler.rs` 为什么只是再导出

`openinfer-server/src/scheduler.rs` 和 `sampler.rs` 本身几乎不持有逻辑，只是做 re-export。

这进一步说明：

- server crate 并不想重新实现调度或采样；
- 它更多是一个“对外入口 + 兼容门面”的角色。

也就是说，`openinfer-server` 的中心不是业务逻辑密集模块，而是**把别处的能力组装成一个可启动、可服务的程序**。

## 从用户启动到服务 ready 的完整路径

把这几个模块串起来，一次典型启动大致是：

```text
1. 用户运行 `openinfer ...`
2. `config.rs` 解析 CLI
3. `detect_model_type()` 读取 `config.json` 判断模型线
4. `Args::validate()` 校验该模型线下哪些参数合法
5. `spawn_blocking(load_engine)` 启动模型加载
6. 同时开始准备前端服务
7. `load_engine()` 调对应模型 crate 的 `launch(...)`
8. 得到 `EngineHandle`
9. 把 handle 交给 `openinfer-vllm-frontend`
10. bridge 注册完成后，HTTP 才真正对外 ready
```

所以如果只记一句话，可以把 `openinfer-server` 理解成：

**“识别你给的模型属于哪条模型线，然后启动那条线自己的引擎，再统一挂到 OpenAI/vLLM 前端上。”**

## 为什么它是 feature-gated binary

`Cargo.toml` 里每条模型线都是 feature-gated：

- `qwen3-4b`
- `qwen35-4b`
- `deepseek-v4`
- `deepseek-v2-lite`
- `kimi-k2`

这背后的目的很明确：

- 默认构建尽量轻；
- 模型依赖链按需打开；
- 像 Qwen3.5 这种带 Triton/Python 的路径，不强迫所有用户都构建。

所以 `openinfer-server` 不是“一个永远包含所有模型”的重型二进制，而是“一个 feature-selectable 的统一入口”。

## 这个 crate 刻意没有做的事

为了看清边界，也值得明确它**没有**做什么：

- 不实现模型 forward；
- 不实现模型 scheduler；
- 不拥有 frontend bridge 核心逻辑；
- 不实现采样算法；
- 不拥有 kernel 或 tensor 层。

它专注做的是：**配置入口、模型识别、feature-gated 分发、以及服务层组装。**

