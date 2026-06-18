# OpenInfer New Model Integration Guide

> **TL;DR:** 在 openinfer 里接一个新模型，真正要做的不是“再写一个 `weights.rs`”，而是把一整条闭环接通：server 能识别它、模型 crate 能把 server 意图解释成自己的启动策略、scheduler/executor 能跑通请求生命周期、shared runtime 只吸收真正的共性、底层 kernels/comm 只在必要时扩，最后还要补齐 correctness、serving、性能和文档证据。

## 先建立两个总原则

### 原则 1：server 只识别和分发，不决定模型内部策略

server 层该做的是：

- 识别模型；
- 调用对应模型 crate 的 `launch(...)`；
- 把 CLI / 顶层配置转发下去。

server **不该**做的是：

- 根据模型名手写 attention 路径；
- 决定某模型是否支持 CUDA Graph；
- 决定固定设备拓扑；
- 决定 LoRA / prefix cache / offload / EP backend 的模型内约束。

### 原则 2：共享层只收稳定共性，复杂性留在模型 crate

能进 `openinfer-core` 的，应该是这些东西：

- `GenerateRequest` / `TokenEvent` 这类统一契约；
- `SamplingParams`；
- `ParallelConfig`；
- `kv_pool` / `page_pool` / `cuda_graph` 等共享积木；
- `weight_loader` 这类跨模型都会用的辅助。

不该为了“看起来统一”而硬抽进去的，是这些：

- 某模型特有的 attention 状态；
- 某模型特有的 scheduler policy；
- 某模型特有的 worker/runner 组织；
- 某模型特有的通信或 buffer 生命周期。

## 第 0 步：先判断你接入的是哪一类模型

### 类型 A：同家族变体

特征：

- attention 形态不变；
- scheduler/executor 结构大概率不变；
- 差异主要在配置、形状、权重、少量 kernel 参数。

最常见动作：**扩已有模型 crate。**

优先模板：`openinfer-qwen3-4b`

### 类型 B：新的模型线，但 shared contract 和总体 runtime 边界还能沿用

特征：

- 需要新 crate；
- 需要自己的 `config` / `weights` / `scheduler` / `executor`；
- 但不需要改 server 的总体服务方式。

优先模板：`openinfer-qwen35-4b`

### 类型 C：新家族，伴随新 attention / 新并行 / 新通信 / 新运行时组织

特征：

- attention / KV / execution shape 明显变化；
- 多卡、多并行维度、MoE、通信层成为主路径；
- 模型 crate 可能长出 `runner/worker/load_balancer` 一类专用组织。

优先模板：`openinfer-deepseek-v4` 或 `openinfer-kimi-k2`

## 第 1 步：决定“扩已有 crate”还是“新建模型 crate”

### 继续扩已有 crate 的信号

- 只是同一家族的小变体；
- 配置和权重布局变化有限；
- 当前 scheduler/executor 的状态模型仍成立；
- 不需要新增一条完全不同的前向执行家族。

### 必须新建模型 crate 的信号

- attention 形态变了；
- KV 组织变了；
- decode 路径需要新的状态类型；
- 需要新的通信/并行拓扑；
- 继续塞进旧 crate 会制造很多模型专属 if/else。

### 一个实用判断句

> 如果你已经开始想“在 Qwen3 的 scheduler 里加一个 `if my_model { ... } else { ... }`”，很可能就该新建 crate 了。

## 第 2 步：工作区与 feature gate 接入

### 你通常要改哪些文件

#### 工作区根

- `Cargo.toml`

你要做的事：

- 把新 crate 加进 `[workspace].members`
- 把新 crate 加进 `[workspace.dependencies]`

### server crate

- `openinfer-server/Cargo.toml`

你要做的事：

- 新增 optional dependency
- 新增同名 feature
- 如果模型 crate 本身还依赖 feature-gated kernel 路径，把 server feature 连到模型 crate feature

### 参考模式

```toml
[features]
default = ["qwen3-4b"]
qwen3-4b = ["dep:openinfer-qwen3-4b"]
qwen35-4b = ["dep:openinfer-qwen35-4b", "openinfer-qwen35-4b/qwen35-4b"]
deepseek-v4 = ["dep:openinfer-deepseek-v4", "openinfer-deepseek-v4/deepseek-v4"]
```

### 设计纪律

- 默认 feature 只放最稳、依赖最少的主线；
- 需要 Python/Triton、专用通信、特殊硬件的路径显式 feature gate；
- server 只声明“我能分发到这条模型线”，不声明模型内部复杂策略。

## 第 3 步：让 server 能识别这个模型

### 要改的文件

- `openinfer-server/src/server_engine.rs`

### 典型改动点

1. `ModelType` 增加一个分支
2. `fmt::Display` 增加展示名
3. `detect_model_type(...)` 增加 `config.json` 识别逻辑
4. `load_engine(...)` 增加对应 `launch(...)` 分发

### 最小骨架

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelType {
    #[cfg(feature = "my-model")]
    MyModel,
}

pub fn detect_model_type(model_path: impl AsRef<Path>) -> Result<ModelType> {
    let config_path = model_path.as_ref().join("config.json");
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

    if json
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|value| value == "my_model")
    {
        #[cfg(feature = "my-model")]
        return Ok(ModelType::MyModel);
        #[cfg(not(feature = "my-model"))]
        anyhow::bail!("MyModel support is feature-gated; rebuild openinfer-server with --features my-model");
    }

    anyhow::bail!("unsupported model config")
}
```

### 识别层的经验规则

- 优先用 `config.json` 稳定字段，不要靠目录名猜；
- 如果同一 `model_type` 下需要区分变体，参考 `openinfer-deepseek-v2-lite/src/lib.rs` 里的 `probe_config_json(...)`；
- 如果识别很复杂，把模型专属 probe 放回模型 crate，而不是让 server 胀成百科全书。

## 第 4 步：先设计模型 crate 的外部接口

### 你至少要有这些文件

```text
openinfer-my-model/
  Cargo.toml
  src/
    lib.rs
    config.rs
    weights.rs
    scheduler.rs
    executor.rs
```

更复杂时再加：

- `prefill.rs`
- `batch_decode.rs`
- `recurrent.rs`
- `runner/`
- `weights/`
- `kernel_plan.rs`
- `tests/`

### `lib.rs` 的最低目标

`lib.rs` 要承担三层职责：

1. 暴露模型线的 server-facing `launch(...)`
2. 暴露模型线的 model-facing `start_engine(...)`
3. 把模型自己的模块边界收起来

### 推荐骨架

```rust
mod config;
mod weights;
mod scheduler;
mod executor;

use std::path::Path;
use anyhow::Result;
use openinfer_core::engine::{EngineHandle, EngineLoadOptions, EpBackend};

pub fn launch(model_path: &Path, device_ordinal: usize, cuda_graph: bool) -> Result<EngineHandle> {
    start_engine(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: cuda_graph,
            enable_prefill_profile: false,
            device_ordinals: vec![device_ordinal],
            parallel_config: None,
            ep_backend: EpBackend::Nccl,
            seed: 42,
        },
    )
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    let model = weights::MyModel::load(model_path, &options)?;
    scheduler::start(model, options.seed)
}
```

### 为什么要分 `launch(...)` 和 `start_engine(...)`

- `launch(...)`：解释 server 传下来的意图
- `start_engine(...)`：真正把模型拉起来

不要把这两个层次糊成一个函数，否则 server-facing policy 和 model runtime bootstrap 会很快纠缠在一起。

## 第 5 步：决定模型启动策略放在哪里

### 应该放在模型 crate 的策略

- 是否支持 CUDA Graph
- 固定设备拓扑
- TP/DP/EP 怎么组合
- LoRA 是否与图捕获互斥
- prefix cache / offload 是否支持
- 模型专属 admission 约束

### 不应该放在 server 的策略

- 如果 server 里开始出现“这个模型不支持 graph，所以……”这类细节分支，通常说明边界放错了。

### 现成参考

- `openinfer-qwen3-4b/src/lib.rs`：LoRA 与 graph 互斥、TP 推导 device ordinals、offload 选项
- `openinfer-qwen35-4b/src/lib.rs`：单卡限制
- `openinfer-deepseek-v2-lite/src/lib.rs`：固定 EP2，并忽略 graph
- `openinfer-deepseek-v4/src/lib.rs`：固定 MP8，并忽略 graph
- `openinfer-kimi-k2/src/lib.rs`：TP/DP/EP 由模型层解释

## 第 6 步：在模型 crate 内放对模块边界

### 推荐的职责分布

| 文件 | 应该负责什么 | 不该负责什么 |
| --- | --- | --- |
| `config.rs` | 解析模型 config 与形状 | 启动线程、调度请求 |
| `weights.rs` | 读权重、构建运行时权重表示 | 直接处理 HTTP / frontend 语义 |
| `scheduler.rs` | 请求生命周期、批处理、退休、事件发射 | 真正做所有底层 kernel 调用 |
| `executor.rs` | prefill/decode/unified 执行编排 | 识别模型类型、做 CLI 分发 |
| `prefill.rs` / `batch_decode.rs` | 阶段性算子组合 | 高层 admission policy |
| `runner/` | 专用多卡运行时组织 | server 顶层路由 |

### 一个很实用的自检

如果某个文件已经同时做了：

- batch formation
- kernel launch
- client event emission
- runtime resource retirement

那它很可能已经该拆了。

## 第 7 步：先想清楚 scheduler / executor 的关系，再写实现

### 一个最低心智模型

- scheduler：决定**谁在这一步跑、跑哪种计划、何时退休**
- executor：决定**这一步在 GPU 上怎么执行**

### 为什么这条边界重要

因为它让你能支持：

- prefill / decode 分离
- unified step
- chunked prefill
- prefix cache
- cancellation / rejection / error 的清晰事件语义

### 极简伪代码骨架

```rust
pub fn start(model: MyModel, seed: u64) -> Result<EngineHandle> {
    let (submit_tx, mut submit_rx) = tokio::sync::mpsc::unbounded_channel();
    let join_handle = std::thread::spawn(move || {
        let mut scheduler = Scheduler::new(model, seed)?;
        while let Some(req) = submit_rx.blocking_recv() {
            scheduler.enqueue(req);
            scheduler.step_until_idle()?;
        }
        Ok::<_, anyhow::Error>(())
    });
    Ok(EngineHandle::new_with_join_handle(submit_tx, join_handle))
}
```

这段只是骨架，但它提醒你：

- 上层拿到的是统一 `EngineHandle`
- 请求通过 `GenerateRequest` 进入
- scheduler 是真正的运行时控制中心

## 第 8 步：决定哪些能力复用 `openinfer-core`

### 优先复用的东西

- `openinfer_core::engine`
- `openinfer_core::sampler`
- `openinfer_core::parallel`
- `openinfer_core::weight_loader`
- `openinfer_core::kv_pool`
- `openinfer_core::page_pool`
- `openinfer_core::cuda_graph`
- `openinfer_sample`

### 不要急着共享的东西

- 只在一个模型里成立的 KV/state 组织
- 只在一个模型里成立的 scheduler 计划
- 只在一个模型里成立的 runtime worker 拓扑
- 只在一个模型里成立的 buffer 生命周期

### 实用判断句

> 如果这段逻辑离开当前模型 crate 会变得更抽象、更难懂、但并没有第二个模型马上复用，那先别抽。

## 第 9 步：决定是否要扩 `openinfer-kernels`

### 只用现有 kernel surface 的信号

- 算法路径没变；
- 只需要新 shape、新参数或新调度；
- 主要工作仍在模型 crate 的 executor/prefill/decode 组织层。

### 需要扩 `openinfer-kernels` 的信号

- 需要新的 attention / MoE / quant / sampling 算子；
- 需要新的 build-time codegen；
- 需要新的 feature-gated kernel surface；
- 需要新增 CUDA/Triton/TileLang 构建路径。

### 你通常会碰哪些文件

- `openinfer-kernels/build.rs`
- `openinfer-kernels/Cargo.toml`
- 对应 feature-gated kernel 模块或 codegen 脚本

### 原则

- kernel crate 是能力 owner；
- 但“什么时候走哪条 kernel 路径”通常仍由模型 crate 决定。

## 第 10 步：决定是否要接 `openinfer-comm`

### 什么时候不用接

- 单卡路径；
- 简单 TP 且已有实现足够；
- 模型当前目标只是先跑通最小闭环。

### 什么时候才考虑接

- EP / all-to-all / RDMA 是主路径；
- MoE 路由和专家交换已经进入性能或正确性关键路径；
- 模型不接通信层就根本不成立。

### 原则

不要为了“以后可能多卡”提前把简单模型设计成通信优先。只有当通信确实是模型语义的一部分时，才把它接进来。

## 第 11 步：最小验证闭环必须长什么样

### 1. 识别闭环

证明 server 能认出这个模型：

- `config.json` 能被正确识别；
- feature 没开时会给出清晰错误；
- 变体 probe 会拒绝不支持的形状。

### 2. 启动闭环

证明 `launch(...)` / `start_engine(...)` 真能把模型拉起来：

- 设备和拓扑解释正确；
- 不支持的开关显式 warning 或 reject；
- 返回的是统一 `EngineHandle`。

### 3. 请求闭环

证明至少能完成最小请求生命周期：

- `GenerateRequest` 可 submit；
- scheduler 能消费；
- executor 能执行；
- 能收到 `Scheduled` / `Token` / `Finished` 或 terminal event。

### 4. correctness gate

优先走 logits/teacher-forced 路径，而不是脆弱的 exact text：

- 参考 `docs/subsystems/correctness/logits-golden-gate.md`
- 参考 `docs/playbooks/accuracy-parity-playbook.md`

最低要求是：

- 有外部 truth source；
- 能检测回归；
- 不依赖“刚好这台卡上字符串一样”。

### 5. serving / e2e gate

证明不是“库能跑”，而是“服务能跑”：

- frontend 请求能真正打通；
- `TokenEvent` 能回流到输出层；
- rejection / error / finished 语义都正确。

### 6. baseline gate

哪怕先不追极致性能，也要留一个最小基线：

- 让后续优化有比较对象；
- 避免“修正确性时 silently 把性能打穿”。

## 第 12 步：文档闭环别忘了

### 你通常要补哪些文档

- `docs/models/<line>/...`：模型线自己的设计、精度、性能、状态文档
- `docs/index.md`：路由表
- 如影响共享边界：
  - `docs/subsystems/runtime/...`
  - `docs/subsystems/kernels/...`
- 如果形成了通用开发方法：
  - `docs/playbooks/...`

### 为什么这一步属于“接入完成”的一部分

因为这个仓库的协作方式就是围绕 `docs/` 跑的。没有文档闭环，后续开发者就只能再走一遍试错。

## 一份文件级接入清单

### 工作区 / server 层

- [ ] `Cargo.toml`：workspace 成员与依赖
- [ ] `openinfer-server/Cargo.toml`：optional dependency + feature gate
- [ ] `openinfer-server/src/server_engine.rs`：`ModelType` / detect / load dispatch
- [ ] 如需要参数：`openinfer-server/src/config.rs`

### 新模型 crate

- [ ] `src/lib.rs`：`launch(...)` / `start_engine(...)`
- [ ] `src/config.rs`
- [ ] `src/weights.rs`
- [ ] `src/scheduler.rs`
- [ ] `src/executor.rs`
- [ ] `src/prefill.rs` / `src/batch_decode.rs` / `src/unified_forward.rs`（按需要）
- [ ] `tests/...` 或 integration tests

### shared runtime / kernels / comm

- [ ] 判断是否需要改 `openinfer-core`
- [ ] 判断是否需要扩 `openinfer-kernels`
- [ ] 判断是否需要接 `openinfer-comm`

### 验证

- [ ] probe / detection
- [ ] startup
- [ ] request lifecycle
- [ ] correctness gate
- [ ] serving/e2e gate
- [ ] baseline gate

### 文档

- [ ] `docs/models/<line>/...`
- [ ] `docs/index.md`
- [ ] 必要时更新 subsystem / playbook 文档

## 最后给一句判断边界的话

如果你在接新模型时发现自己必须大量改：

- `openinfer-server` 的内部逻辑，或者
- `openinfer-core` 里和当前模型强耦合的代码

那通常不是“模型太特殊”，而是**边界正在被放错**。

正确方向通常是：

- server 保持纯分发；
- shared runtime 只收共性；
- 模型 crate 自己对自己的复杂性负责；
- kernels/comm 只在必要时扩能力。

按这个原则接模型，你补的不是一次性代码，而是未来还能继续长下去的结构。

