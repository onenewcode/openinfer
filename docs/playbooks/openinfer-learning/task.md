# Build OpenInfer Learning Guides

> **TL;DR:** 为 openinfer 补一组面向新开发者的中文学习文档：总览知识图谱、代码导读、以及新模型接入指南；本任务文档记录准备、执行和复盘。

## Preparation

- **Read**:
  - `docs/index.md` — 先确认仓库现有知识按 domain 组织，新的学习材料应该落在 `docs/playbooks/`。
  - `docs/playbooks/developer-onboarding.md` — 已有环境与基础运行手册，新的文档不该重复安装步骤，而应补“如何理解项目”。
  - `docs/subsystems/runtime/runtime.md` — 共享 runtime 与 per-model crate 的边界，是解释整体架构的核心。
  - `docs/subsystems/scheduler/scheduler.md` — scheduler 是请求生命周期的中心，需要放进学习路径主线。
  - `docs/subsystems/kernels/openinfer-kernels-boundary.md` — kernel 层不是孤立 CUDA 目录，而是整个架构分层的一环。
  - `docs/models/qwen3/model-crate.md` — Qwen3 是默认模型线，最适合作为第一次读代码的起点。
  - `docs/playbooks/accuracy-parity-playbook.md` — 新模型接入不能只讲“跑起来”，还要讲“怎么证明是对的”。
  - `openinfer-server/src/server_engine.rs` — 模型检测与 launch 分发入口。
  - `openinfer-core/src/lib.rs` — 共享 runtime 模块表。
  - `openinfer-engine/src/engine.rs` — 最清楚地解释了 `EngineHandle` / `GenerateRequest` / `TokenEvent` 的契约心智模型。
  - `openinfer-qwen3-4b/src/lib.rs` — 单卡 dense 模型接入模板。
  - `openinfer-qwen35-4b/src/lib.rs` — 混合注意力 + feature gate 的模型接入模板。
  - `openinfer-deepseek-v4/src/lib.rs` — 固定 MP8 拓扑、模型自行决定 launch 策略的模板。
  - `openinfer-deepseek-v2-lite/src/lib.rs` — 固定 EP2 拓扑的较小 MoE 模型模板。
  - `openinfer-kimi-k2/src/lib.rs` — 最复杂模型线的运行时分工说明。
  - `openinfer-kernels/build.rs` — build-time kernel 编译和 feature gate 的真实入口。
- **Relevant history**:
  - `docs/playbooks/developer-onboarding.md` — 现有 onboarding 偏“怎么跑”，缺少“如何读懂”。
  - `docs/subsystems/runtime/runtime.md` — 明确了“不为每个新模型复制一套 runtime”这一长期原则。
  - `openinfer-kimi-k2/src/runner.rs` — 展示了当模型复杂到 MLA + MoE + TP/DP/EP 时，为什么运行时逻辑必须留在模型 crate 内部。
- **Plan**:
  1. 新增 `docs/playbooks/openinfer-learning/README.md`，用知识图谱 + 学习阶段说明整个项目该怎么学。
  2. 新增 `docs/playbooks/openinfer-learning/request-lifecycle.md`，按请求生命周期和 crate 边界做代码导读。
  3. 新增 `docs/playbooks/openinfer-learning/new-model-integration.md`，写清楚“接入一个新模型”时该改哪些层、复用哪些层、避免哪些坏模式，并附代码骨架。
  4. 更新 `docs/index.md`，把新文档加入 playbooks 路由表。
  5. 回填本任务文档的执行日志与复盘，并做一次一致性检查。
- **Risks / open questions**:
  - 工作区里存在用户自己的未提交改动和未跟踪 `README.md`，本任务必须只触碰 `docs/`，不能顺手整理别处。
  - 文档规则要求写进文档的命令必须经过验证；因此新文档以代码路径、结构图和代码片段为主，避免引入本轮未验证的 GPU 命令。

## Execution Log

### Step 1: 设计文档集合与读者路径
- 把目标拆成三份文档，而不是一份超长大杂烩：
  - 学习总览：回答“先学什么、再学什么”。
  - 架构导读：回答“请求如何穿过这些 crate”。
  - 新模型接入：回答“我要改哪些层，最小闭环是什么”。
- 结果：成功，结构足够清晰，也符合用户“多个文档”的要求。

### Step 2: 编写学习总览与架构导读
- 新增 `docs/playbooks/openinfer-learning/README.md`。
- 新增 `docs/playbooks/openinfer-learning/request-lifecycle.md`。
- 内容重点放在：
  - 共享层 / 模型层 / kernel 层 / 验证层的边界；
  - 默认入口应从 Qwen3 开始；
  - 用真实代码路径解释请求生命周期。
- 结果：成功。

### Step 3: 编写新模型接入指南
- 新增 `docs/playbooks/openinfer-learning/new-model-integration.md`。
- 用已有模型线提炼了三类接入模式：
  - 同家族变体；
  - 新模型线但仍可复用现有 runtime 骨架；
  - 新注意力 / 新并行 / 新通信 / 新 kernel 家族。
- 添加了服务器 feature gate、模型检测、`launch`/`start_engine`、scheduler/executor、accuracy gate 的代码骨架。
- 结果：成功。

### Step 4: 更新路由表并做一致性检查
- 更新 `docs/index.md` 的 playbooks 区域，加入三份新文档和本任务文档。
- 结果：成功。

### Step 5: 收敛入口并整合到单独文件夹
- 根据用户反馈，把四份分散在 `docs/playbooks/` 下的文档整理到 `docs/playbooks/openinfer-learning/`。
- 新增 `docs/playbooks/openinfer-learning/README.md` 作为更短的统一入口；后续再把细节下沉并拆成更深的源码级子文档。
- 结果：成功，入口从多个顶层文件收敛成一个文件夹。

### Step 6: 规划方案 B 扩写结构
- 根据用户反馈“当前仍然不够详细”，评估现有四份文档的深度缺口。
- 确认采用方案 B：保持一个文件夹，但扩成 `README.md` + `request-lifecycle.md` + `source-tour.md` + `model-comparison.md` + `new-model-integration.md`。
- 先写扩写设计文档，再等待用户确认后执行。
- 结果：成功。

### Step 7: 按方案 B 扩写到源码级深度
- 重写 `README.md`，让它从长篇概览收敛成短入口，但保留清晰阅读顺序。
- 新增 `request-lifecycle.md`，按 server -> bridge -> engine contract -> scheduler -> executor -> TokenEvent 回流讲完整请求时序。
- 新增 `source-tour.md`，为 `openinfer-server`、`openinfer-vllm-frontend`、`openinfer-core`、`openinfer-engine`、`openinfer-qwen3-4b`、`openinfer-kernels` 建源码导图。
- 新增 `model-comparison.md`，对比 Qwen3 / Qwen3.5 / DeepSeek-V2-Lite / DeepSeek-V4 / Kimi-K2 的复杂度与模板价值。
- 扩写 `new-model-integration.md`，把原则升级为文件级改动清单与验证闭环。
- 删除过渡性的旧概览文档与设计草稿，避免文件夹再次分散。
- 结果：成功。

### Unexpected
- 用户在上一轮中断了任务，然后追加了“分成多个文档，并且要有代码”的新要求；因此这轮把交付从“单次回答”升级成了“成套文档”。
- 随后用户又要求“不要太繁琐，并整合到一个文件夹中”，所以交付被进一步收敛成一个统一目录 + 一个短入口文件。

## Debrief

- **Outcome**:
  - 已把学习材料收敛到 `docs/playbooks/openinfer-learning/`，并进一步扩成一个短入口 README + 三份深度学习文档 + 一份接入实战文档，覆盖“怎么学 openinfer”“怎么跟完整请求时序”“怎么做源码导读”“怎么比较各模型线”“怎么接新模型”。
- **Pitfalls encountered**:
  - 仓库现有 onboarding 更偏运行步骤，不宜和学习导图混在一起；如果混成一个文档，新人会很难区分“环境问题”和“架构问题”。
  - 这个仓库对“命令必须验证”的要求很严格，所以学习文档不能随手堆一堆 cargo 命令，必须以结构和代码路径为主。
- **Lessons learned**:
  - 这类“帮助新人理解项目”的任务，最好的组织方式不是按功能列文件，而是按“心智模型 -> 请求路径 -> 扩展路径”三层展开。
  - 新模型接入文档必须强调边界：server 负责分发，model crate 负责策略，kernel crate 负责算子，不要把策略散进所有共享层。
- **Follow-ups**:
  - 如果后续用户愿意，还可以再补一份 `docs/playbooks/openinfer-first-week-exercises.md`，把阅读路径变成可打卡的小练习。
  - 如果后续需要更深入，还可以继续补一份 `subsystem-invariants.md`，专门讲 scheduler / executor / KV / bridge 的设计不变量。

