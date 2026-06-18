# openinfer-vllm-support 工作原理

`openinfer-vllm-support` 是一个很小但很实用的桥接 crate：它专门解决 **“openinfer 这边想加载 tokenizer，而 tokenizer 解析逻辑其实在 vLLM 生态那边”** 这个问题。

如果只记一句话，可以这样理解：

- `vllm-text` / `vllm-tokenizer` 知道怎么解析 Hugging Face / tokenizer 文件；
- `openinfer-vllm-support` 把这套能力包装成 openinfer 侧可以同步调用的小工具；
- 重点不是“功能多”，而是“边界清晰，避免 runtime/async 使用出错”。

## 它在系统里的位置

这个 crate 的位置非常窄：

```text
openinfer 侧调用者
  -> openinfer-vllm-support::load_tokenizer(...)
  -> vllm-text / vllm-tokenizer
```

它不负责前端路由，也不负责模型推理；它只是一个**tokenizer 解析支持层**。

## 为什么需要单独一个 crate

从代码看，这个 crate 只暴露了几个函数：

- `load_tokenizer(model_id: &str)`
- `tokenizer_from_source(source: &TokenizerSource)`

看起来不多，但它解决了两个很实际的问题：

### 1. 把 vLLM tokenizer 能力隔离出来

openinfer 某些路径需要 tokenizer，但不想把整个 vLLM 前端逻辑都拉进来。

所以这里把“解析 tokenizer source 并创建 tokenizer”这件事单独抽出来，形成一个更小的依赖面。

### 2. 把 async tokenizer 解析包装成同步安全接口

`ResolvedModelFiles::new(model_id)` 是 async 的，但很多 openinfer 调用点希望：

- 在同步上下文里直接拿 tokenizer；
- 而不是把整个调用链都改成 async。

于是这个 crate 的核心价值，就变成了：**提供一个同步可用、但不会误用 Tokio runtime 的 tokenizer 加载门面。**

## `load_tokenizer()` 到底做了什么

这个函数是主要入口。它的逻辑其实很清晰：

1. 先检查当前线程里是不是已经有 active Tokio runtime
2. 如果有，直接报错
3. 否则 resolve model files
4. 再根据 `TokenizerSource` 创建具体 tokenizer

所以它不是一个“无脑同步包装”，而是一个**带边界保护的同步入口**。

## 为什么要拒绝在 active Tokio runtime 里调用

这是这个 crate 最关键的设计点。

`load_tokenizer()` 内部最终会走到：

```rust
runtime.block_on(...)
```

而在 Tokio 已经运行的上下文里再 `block_on` 一个 runtime，是非常容易出问题的：

- 轻则 panic
- 重则把调用方搞成很难排查的嵌套 runtime 事故

所以这个 crate 的态度非常明确：

- 如果已经在 active Tokio runtime 里，就直接拒绝；
- 不去做“也许能跑、也许会炸”的模糊行为。

这是一种典型的 fail-closed 设计。

## `resolve_model_files()` 为什么要自己维护一个 runtime

它内部用了一个：

- `OnceCell<Mutex<Runtime>>`

也就是说，整个进程里：

- 只会懒创建一个 current-thread Tokio runtime；
- 后续每次 resolve tokenizer files 都复用它；
- 通过 `Mutex` 确保同步调用不会并发冲进同一个 runtime。

这里有两个设计动机：

### 1. 避免每次调用都新建 Tokio runtime

如果每次 `load_tokenizer()` 都临时 build 一个 runtime：

- 开销更大；
- 生命周期也更分散。

### 2. 保持同步入口简单

对调用方来说，它只看到：

```rust
let tokenizer = load_tokenizer(model_id)?;
```

内部是否需要 runtime、怎么复用 runtime，都被这个 crate 吞掉了。

所以 `OnceCell + Mutex<Runtime>` 的本质，是把 async resolver 封装成一个**进程级同步服务**。

## `TokenizerSource` 是怎么落成具体 tokenizer 的

当 model files resolve 完成后，代码会根据 `TokenizerSource` 分三种情况：

- `HuggingFace(path)` -> `HuggingFaceTokenizer`
- `Tiktoken(path)` -> `TiktokenTokenizer`
- `Tekken(path)` -> `TekkenTokenizer`

这一步由 `tokenizer_from_source(...)` 完成。

所以这个 crate 并不发明 tokenizer 格式；它只是：

- 识别上游解析出来的 tokenizer source 类型；
- 构造对应的 tokenizer 实例；
- 最终统一返回 `DynTokenizer`。

也就是说，它做的是**多种 tokenizer 后端的轻量工厂层**。

## 为什么返回 `DynTokenizer`

调用者通常不想关心自己拿到的是：

- HuggingFaceTokenizer
- TiktokenTokenizer
- TekkenTokenizer

它只想要一个统一的“能 tokenize / detokenize 的对象”。

所以这里返回的是 `DynTokenizer`，也就是：

- 把多种 tokenizer 实现擦平成一个统一动态接口

这样上层调用点就不必为 tokenizer 类型分支。

## 这个 crate 的价值不在“多”，而在“稳”

这个 crate 代码很少，但很有边界价值，因为它把几件容易出错的事情统一处理掉了：

- 模型文件解析是 async
- openinfer 有同步调用点
- tokenizer 后端不止一种
- runtime 嵌套有风险

而它给出的策略非常明确：

- 同步调用可以，但只能在没有 active Tokio runtime 时；
- 内部 runtime 只建一次并复用；
- tokenizer 类型由 source 自动分派。

所以它更像一个“安全适配层”，不是复杂业务层。

## 从调用者视角看一次加载流程

完整流程大概是：

```text
1. 调用 load_tokenizer(model_id)
2. 检查当前是否已有 active Tokio runtime
3. 如果有 -> 直接报错
4. 如果没有 -> 复用或创建内部 current-thread runtime
5. block_on ResolvedModelFiles::new(model_id)
6. 取出 tokenizer source
7. tokenizer_from_source(...) 构造具体 tokenizer
8. 返回 DynTokenizer
```

如果只记一句话，可以把 `openinfer-vllm-support` 理解成：

**“把 vLLM 侧的 tokenizer 解析能力，安全地封装成 openinfer 可同步调用的小桥。”**

## 这个 crate 刻意没有做的事

为了看清边界，也值得明确它**没有**做什么：

- 不做 HTTP/frontend；
- 不做模型推理；
- 不管理请求生命周期；
- 不自己实现 tokenizer 算法；
- 不暴露一整套 vLLM backend。

它只专注在一个窄而关键的问题上：**把 tokenizer 解析能力安全地接进 openinfer。**

