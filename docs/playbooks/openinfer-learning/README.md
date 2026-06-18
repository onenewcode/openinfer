# OpenInfer Learning

> **TL;DR:** 这套资料的目标不是告诉你“openinfer 有哪些 crate”，而是把你带到**能真正读源码、定位问题、接入新模型**的状态。建议顺序是：先看 `README.md` 建立心智模型，再看 `request-lifecycle.md` 跟一遍请求主链路，再看 `source-tour.md` 做系统化源码导读，最后根据目标选 `model-comparison.md` 或 `new-model-integration.md`。

## 这个文件夹怎么用

```text
docs/playbooks/openinfer-learning/
  README.md                # 短入口：你应该怎么学、先看什么
  request-lifecycle.md     # 一次请求如何穿过 server -> bridge -> engine -> scheduler -> executor
  source-tour.md           # 按 crate / 文件的源码阅读地图
  model-comparison.md      # Qwen3 / Qwen3.5 / DeepSeek / Kimi 的架构差异与复杂度对照
  new-model-integration.md # 接入新模型时该改哪些文件、哪些层、哪些验证闭环
  task.md                  # 这轮文档建设记录
```

如果你不知道先看哪份，直接按下面的阅读顺序：

1. `docs/playbooks/openinfer-learning/README.md`
2. `docs/playbooks/openinfer-learning/request-lifecycle.md`
3. `docs/playbooks/openinfer-learning/source-tour.md`
4. `docs/playbooks/openinfer-learning/model-comparison.md`
5. `docs/playbooks/openinfer-learning/new-model-integration.md`

## 一张够用的总图

```text
HTTP/OpenAI request
  -> openinfer-server
  -> openinfer-vllm-frontend bridge
  -> GenerateRequest / EngineHandle
  -> per-model scheduler
  -> per-model executor
  -> openinfer-core shared runtime helpers
  -> openinfer-kernels (+ openinfer-comm when needed)
  -> CUDA kernels / collectives
  -> TokenEvent stream
  -> bridge reduces events into frontend outputs
```

把项目先压成四层，就不会迷路：

```text
Layer 1: Serving / Frontend
  openinfer-server
  openinfer-vllm-frontend
  openinfer-vllm-support

Layer 2: Shared Runtime Contract / Helpers
  openinfer-core
  openinfer-engine
  openinfer-sample
  openinfer-kv-cache
  openinfer-kv-offload

Layer 3: Per-Model Engines
  openinfer-qwen3-4b
  openinfer-qwen35-4b
  openinfer-deepseek-v2-lite
  openinfer-deepseek-v4
  openinfer-kimi-k2

Layer 4: Kernel / Comm / Build
  openinfer-kernels
  openinfer-comm
```

最重要的一句只有一句：**共享层只收稳定共性，模型层自己持有策略和复杂性。**

## 三种读法

### 1. 我是第一次接触这个仓库

看：

1. `README.md`
2. `request-lifecycle.md`
3. `openinfer-server/src/server_engine.rs`
4. `openinfer-qwen3-4b/src/lib.rs`

读完后你应该能回答：

- server 怎么识别模型？
- 为什么 server 调的是各模型 crate 的 `launch(...)`？
- 为什么输出是 `TokenEvent` 而不是字符串？
- 为什么 Qwen3 是最好的第一阅读样本？

### 2. 我准备开始读源码甚至改代码

看：

1. `request-lifecycle.md`
2. `source-tour.md`
3. `openinfer-core/src/engine.rs`
4. `openinfer-qwen3-4b/src/scheduler.rs`
5. `openinfer-qwen3-4b/src/executor.rs`
6. `openinfer-kernels/build.rs`

目标不是背下所有函数，而是知道：

- 请求在哪一层被构造、分发、调度、执行、回流；
- 哪些文件是主路径，哪些文件只是辅助；
- 哪些逻辑该改在共享层，哪些逻辑该留在模型 crate。

### 3. 我准备接一个新模型

看：

1. `README.md`
2. `request-lifecycle.md`
3. `model-comparison.md`
4. `new-model-integration.md`
5. 对照 `openinfer-qwen3-4b/src/lib.rs`
6. 再对照 `openinfer-deepseek-v4/src/lib.rs` 或 `openinfer-kimi-k2/src/lib.rs`

目标是建立两种判断：

- 这是“扩已有模型线”，还是“新建模型 crate”；
- 这是“共享层问题”，还是“模型专属复杂性”。

## 四个最该先看的代码入口

### 入口 1：server 总分发口

- `openinfer-server/src/main.rs`
- `openinfer-server/src/server_engine.rs`

这是最上层入口。你会看到：

- CLI 参数是怎么被读取和验证的；
- `detect_model_type(...)` 怎么根据 `config.json` 判断模型线；
- `load_engine(...)` 怎么只做**纯分发**，不做模型内部策略决策。

### 入口 2：共享引擎契约

- `openinfer-core/src/engine.rs`
- `openinfer-engine/src/engine.rs`

这是整个系统的“统一语言”。这里定义了：

- `EngineLoadOptions`
- `GenerateRequest`
- `TokenEvent`
- `TokenSink`
- `EngineHandle`

如果你不先看这层，后面所有 scheduler/executor 的路径都会像散点。

### 入口 3：默认模型线

- `openinfer-qwen3-4b/src/lib.rs`
- `openinfer-qwen3-4b/src/scheduler.rs`
- `openinfer-qwen3-4b/src/executor.rs`

Qwen3 是最标准的单卡 dense serving 闭环，几乎能教会你这个仓库最关键的 80%。

### 入口 4：kernel build owner

- `openinfer-kernels/build.rs`

这个文件决定：

- 需要编哪些 CUDA/Triton/TileLang 路径；
- SM 目标怎么检测；
- feature gate 怎么映射到底层构建行为。

## 当前最推荐的学习路线

### Route A：1 小时建立总图

1. `README.md`
2. `request-lifecycle.md`
3. `openinfer-server/src/server_engine.rs`
4. `openinfer-qwen3-4b/src/lib.rs`

### Route B：半天进入源码主线

1. `request-lifecycle.md`
2. `source-tour.md`
3. `openinfer-core/src/engine.rs`
4. `openinfer-qwen3-4b/src/scheduler.rs`
5. `openinfer-qwen3-4b/src/executor.rs`
6. `openinfer-kernels/build.rs`

### Route C：准备做扩展

1. `source-tour.md`
2. `model-comparison.md`
3. `new-model-integration.md`
4. 选一条模板模型线做对照阅读

## 每条模型线最适合学什么

| 模型线 | 最值得学的点 | 第一入口 | 适不适合作为第一样本 |
| --- | --- | --- | --- |
| Qwen3 | 单卡 dense serving 的标准闭环 | `openinfer-qwen3-4b/src/lib.rs` | 是 |
| Qwen3.5 | hybrid attention、recurrent state、Triton AOT | `openinfer-qwen35-4b/src/lib.rs` | 第二阶段再看 |
| DeepSeek-V2-Lite | 固定 EP2 + 小型 MoE 的接法 | `openinfer-deepseek-v2-lite/src/lib.rs` | 作为 MoE 入门可看 |
| DeepSeek-V4 | 固定 MP8、多 rank direct engine | `openinfer-deepseek-v4/src/lib.rs` | 不适合第一样本 |
| Kimi-K2 | MLA + MoE + TP/DP/EP 组合后的专用运行时 | `openinfer-kimi-k2/src/lib.rs` | 不适合第一样本 |

## 读源码时最容易掉进去的坑

- 一上来就从 `openinfer-kernels` 开始看，跳过 scheduler/executor 主线。
- 把 `openinfer-core` 当成“所有模型逻辑都应该抽进去”的地方。
- 把 `openinfer-server` 误解成真正做推理的地方。
- 把 Kimi-K2 当成第一阅读样本，结果从第一天就淹没在并行拓扑和 MoE 细节里。
- 把“能跑起来”误认为“接入完成”，忘记 correctness gate、serving gate 和文档闭环。

## 下一步去哪里

- 想一口气跟完请求主链路：看 `docs/playbooks/openinfer-learning/request-lifecycle.md`
- 想做系统化源码导读：看 `docs/playbooks/openinfer-learning/source-tour.md`
- 想知道不同模型线到底差在哪：看 `docs/playbooks/openinfer-learning/model-comparison.md`
- 想真正开始接新模型：看 `docs/playbooks/openinfer-learning/new-model-integration.md`

