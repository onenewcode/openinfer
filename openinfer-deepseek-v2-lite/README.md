# openinfer-deepseek-v2-lite 工作原理

`openinfer-deepseek-v2-lite` 是 DeepSeek-V2-Lite 的 EP2 模型线 crate。它的重点不是做一个通用 MoE 框架，而是把“2 卡专家并行、正确性优先、可做 attribution 的实验性服务路径”落成一条可验证的实现。

## 一句话先理解

这条 crate 可以理解成：

- 它不是主打生产吞吐的成熟 serving 引擎；
- 它是一条 feature-gated 的 EP2 路径；
- 重点在于把 HF / host-staged / NCCL 三种路径做成可比较、可归因、可验证的基线。

所以它既是模型执行器，也是一个“MoE EP 实验台”。

## 它在系统里的位置

```text
server / tests / attribution bin
  -> openinfer-deepseek-v2-lite::start_engine / launch
      -> engine / runtime
          -> host-staged backend 或 NCCL backend
              -> 2 GPU EP2 执行
```

和其他模型线一样，它对外暴露 `EngineHandle`；但内部最重要的分叉是 backend：

- host-staged：作为正确性基线
- NCCL：作为更接近目标高性能路径的实现

## 这个 crate 的核心模块怎么分工

### 1. `engine.rs`

北向入口，负责把模型路径、设备、backend 选择等装配成真正可运行的引擎。

### 2. `runtime/`

真正的模型执行逻辑：

- generation
- layers
- moe
- routing
- backend
- readiness

它定义了“DeepSeek-V2-Lite 一步生成到底怎么推进”。

### 3. `host_ops.rs` / `nccl_backend.rs`

这是它最重要的一组对照面：

- `host_ops` / host-staged 路径更像 correctness oracle
- `nccl_backend` 则是通信优化方向

工程上持续保留 host-staged 的意义，是让 NCCL 改动有一个本地、可重复对比的真值参考。

### 4. `attribution.rs`

这不是装饰，而是这条模型线的核心能力之一。

它负责把 decode 过程拆成：

- call site
- section
- timing / counters

从而让这条 EP2 路径不仅“能跑”，还“知道慢在哪里”。

## 它是怎么工作的

理解这条 crate，关键是抓住三层状态：

1. **模型状态**：权重、配置、device
2. **运行时状态**：KV / 路由 / token 生成上下文
3. **并行后端状态**：host-staged 或 NCCL 的 dispatch / combine 语义

请求进来后：

1. 引擎把 prompt 喂给 runtime；
2. runtime 做 routing，决定 token/row 要去哪些专家；
3. backend 执行跨 GPU 的 dispatch / combine；
4. 返回 logits 和生成结果；
5. attribution / readiness 逻辑记录这次路径是否图捕获友好、各阶段耗时如何。

所以它不只是“把模型放到两张卡上”，而是把 EP2 的执行、验证、归因都绑在了一起。

## 为什么它特别强调“readiness”和“attribution”

因为这条模型线当前的角色不是“默认生产路径”，而是“向更成熟 EP serving 靠近”的中间阶段。

因此除了跑通以外，更重要的是回答：

- 当前 NCCL 路径还卡在哪里？
- 是否具备 CUDA Graph 捕获条件？
- 和 host-staged 的差异到底来自哪一段？

`DecodeGraphReadinessReport`、attribution bin、状态文档，本质上都在服务这个目标。

## 它刻意不做什么

这个 crate 明确不声称：

- 已有成熟连续批处理 serving；
- 已与 vLLM 生产级别等价；
- 已抽象出通用多节点 EP 框架。

它保留“DeepSeek-V2-Lite EP2 专用实现”这个边界，反而让代码和实验结论更可信。

## 最重要的心智模型

把 `openinfer-deepseek-v2-lite` 看成：

- 一个带模型语义的 EP2 执行器；
- 一个带 host-staged / NCCL 对照面的并行实验台；
- 一个把 correctness、attribution、graph-readiness 合在一起的验证载体。

这样理解它，比把它看成“另一个普通模型 crate”更接近真实作用。
