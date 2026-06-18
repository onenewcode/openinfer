# OpenInfer First-Week Exercises

> **TL;DR:** 这不是“做题集”，而是把学习路径变成一组能真正带你读代码、验证理解、形成直觉的小练习。每个练习都围绕当前 learning folder 的主线展开：请求时序、源码导图、模型对照、新模型接入。完成这一周后，你不一定已经会改所有代码，但你应该已经知道**问题该落在哪一层、文件该从哪看起、复杂度该如何判断**。

## 如何使用这份练习

- 先配合 `docs/playbooks/openinfer-learning/README.md` 使用。
- 每天做 1 组，不追求速度，追求“能用代码路径解释清楚”。
- 建议边看边写自己的答案，不要只在脑子里想。
- 这份练习主要是阅读 / 理解型，不要求你必须跑 GPU 命令。

## Day 1：建立系统总图

### 目标

回答：**这个项目分几层？每层负责什么？**

### 阅读材料

- `docs/playbooks/openinfer-learning/README.md`
- `docs/index.md`
- `openinfer-server/src/main.rs`
- `openinfer-server/src/server_engine.rs`

### 练习 1：写出四层模型

不用看答案，自己写一遍：

- Serving / Frontend 层有哪些 crate？
- Shared Runtime 层有哪些 crate？
- Per-Model 层有哪些 crate？
- Kernel / Comm / Build 层有哪些 crate？

### 练习 2：说明 server 的边界

用 5 句话以内解释：

- `openinfer-server` 做什么
- `openinfer-server` 不做什么

如果你写出来的答案里提到了 attention kernel、KV cache 细节、MoE 路由策略，多半已经越界了。

### 练习 3：追一个函数

找到：

- `detect_model_type(...)`
- `load_engine(...)`

回答：

- 哪个函数负责识别模型？
- 哪个函数负责纯分发到各模型 crate？
- 哪些逻辑被刻意留在模型 crate？

## Day 2：跟一条请求走完主链路

### 目标

回答：**一个前端请求是如何变成引擎请求，再变成输出事件的？**

### 阅读材料

- `docs/playbooks/openinfer-learning/request-lifecycle.md`
- `openinfer-vllm-frontend/src/bridge.rs`
- `openinfer-engine/src/engine.rs`

### 练习 1：手画时序图

不看文档，自己画一条 8~10 步的 ASCII 时序图，至少包含：

- server
- bridge
- `GenerateRequest`
- `EngineHandle`
- scheduler
- executor
- `TokenEvent`
- frontend output

### 练习 2：解释 `GenerateRequest`

打开 `openinfer-engine/src/engine.rs`，解释下面字段分别解决什么问题：

- `prompt_tokens`
- `params`
- `max_tokens`
- `token_tx`
- `logprobs`
- `echo`

### 练习 3：区分三类 terminal 语义

解释下面三种事件分别代表什么：

- `Finished`
- `Error`
- `Rejected`

并回答：

- 为什么不能把 `Rejected` 当作一种正常 `Finished`？
- 为什么 `Error` 和 `Rejected` 要分开？

## Day 3：读懂 shared token channel 设计

### 目标

回答：**为什么不是每个请求一个输出 channel？**

### 阅读材料

- `docs/playbooks/openinfer-learning/request-lifecycle.md`
- `openinfer-engine/src/engine.rs`
- `openinfer-vllm-frontend/src/bridge.rs`

### 练习 1：解释 `TokenSink`

回答：

- `TokenSink` 背后真正连的是哪种 channel？
- `RequestTag` 是做什么的？
- 为什么需要 per-request `cancelled` flag？

### 练习 2：解释 bridge reduction

定位并说明：

- `dispatch_burst(...)`
- `reduce_request(...)`

你要说清楚：

- 它们为什么先 bucket 再 reduce？
- 为什么这比“每个请求一个 task”更适合当前架构？

### 练习 3：思考题

如果你把这条路径改成“每个请求一个输出 channel”，你预期会影响：

- 性能
- cancel 语义
- frontend bridge 结构

至少各写一句。

## Day 4：Qwen3 为什么是第一样本

### 目标

回答：**Qwen3 为什么最适合作为第一次深入源码的模型线？**

### 阅读材料

- `docs/playbooks/openinfer-learning/source-tour.md`
- `openinfer-qwen3-4b/src/lib.rs`
- `openinfer-qwen3-4b/src/scheduler.rs`
- `openinfer-qwen3-4b/src/executor.rs`

### 练习 1：找边界

在 Qwen3 里分别指出：

- `launch(...)` 在哪里
- `start_engine(...)` 在哪里
- scheduler 主循环在哪
- executor phase boundary 在哪类结构里最明显

### 练习 2：解释 scheduler / executor 分工

用你自己的话解释：

- scheduler 决定什么
- executor 决定什么

要求不超过 8 句，但必须提到：

- pending vs active
- prefill / decode / unified
- GPU 执行 vs 生命周期控制

### 练习 3：找一个不变量

从 `docs/playbooks/openinfer-learning/subsystem-invariants.md`（如果已经存在）或你自己的理解里，写出一个你认为最关键的 Qwen3 路径不变量，并说明为什么。

## Day 5：模型线复杂度对照

### 目标

回答：**为什么不能直接从 Kimi-K2 开始学？**

### 阅读材料

- `docs/playbooks/openinfer-learning/model-comparison.md`
- `openinfer-qwen35-4b/src/lib.rs`
- `openinfer-deepseek-v2-lite/src/lib.rs`
- `openinfer-deepseek-v4/src/lib.rs`
- `openinfer-kimi-k2/src/lib.rs`

### 练习 1：填对照表

自己做一个小表，至少对比这 5 条模型线：

- attention 形态
- 是否需要特殊并行拓扑
- 是否适合作为第一样本
- 适合拿来学什么

### 练习 2：模板选择题

假设你现在要接一个新模型，分别判断首选模板：

1. 单卡 dense，小改配置与权重
2. hybrid attention，仍是单卡
3. 固定 2 卡的小型 MoE
4. MLA + MoE + TP/DP/EP

### 练习 3：一句话总结

分别用一句话总结：

- Qwen3
- Qwen3.5
- DeepSeek-V2-Lite
- DeepSeek-V4
- Kimi-K2

要求每句都说出“这条模型线最大的代表性复杂度是什么”。

## Day 6：把“接新模型”落到文件级

### 目标

回答：**如果今天真要接一条新模型线，我该先改哪些文件？**

### 阅读材料

- `docs/playbooks/openinfer-learning/new-model-integration.md`
- `openinfer-server/Cargo.toml`
- `openinfer-server/src/server_engine.rs`
- `Cargo.toml`

### 练习 1：写一个文件级清单

假设你要加 `openinfer-my-model`，写出你预期要改的文件列表，至少包含：

- workspace 根
- server Cargo/features
- server model detection
- 新模型 crate `src/lib.rs`
- scheduler / executor / weights / config
- tests
- docs/index.md

### 练习 2：写最小 `launch(...)` 骨架

不用运行，只要写出一个最小骨架，至少包含：

- `EngineLoadOptions`
- `device_ordinals`
- `EpBackend`
- `seed`
- 调用 `start_engine(...)`

### 练习 3：判断哪些逻辑不该进 `openinfer-core`

列出至少 3 类“模型专属复杂性”，并解释为什么不要为了统一把它们塞进 shared runtime。

## Day 7：做一次完整复盘

### 目标

证明你已经不只是“看过文档”，而是能自己定位入口和边界。

### 练习 1：闭卷复述主链路

不看文档，回答：

- 一个请求从 HTTP 到 `TokenEvent` 回流，最关键的 6 个文件是什么？
- 它们各自扮演什么角色？

### 练习 2：选一条后续路线

从下面三条里选一条，并写下你下周要看的文件：

- 深入 Qwen3 主线
- 深入 Kimi-K2 复杂运行时
- 真正准备接一个新模型

### 练习 3：写一份自己的“项目地图”

用 200~300 字写一份你自己的 openinfer 项目地图，要求必须包含：

- 四层结构
- `GenerateRequest` / `TokenEvent`
- scheduler / executor
- 一个你认为最重要的边界判断

## 如何判断自己这一周是否有效

如果做完这一周，你已经能稳定回答下面这些问题，就算有效：

- 为什么 server 不该决定模型内部策略？
- 为什么 `GenerateRequest` / `TokenEvent` 是统一语言？
- 为什么 Qwen3 适合作为第一样本？
- 为什么 Kimi-K2 不能作为第一样本？
- 新模型接入至少要补哪几类闭环？

## 建议的输出方式

如果你真的在学，建议在仓库外另写一个学习笔记，按这几个标题记录：

- 今天我追了哪条路径
- 我确定了哪些边界
- 我还没搞懂什么
- 如果明天让我改代码，我会先改哪层

这会比“读完就关掉文档”有用得多。

