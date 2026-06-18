# openinfer-kv-offload 工作原理

`openinfer-kv-offload` 不是另一套 KV cache，而是一个进程内桥接层：上接 `openinfer-kv-cache` / kvbm 的逻辑 block 与 GPU `KvBuffer`，下接 `pegaflow-core` 的 host-tier 卸载能力，把“把已 seal 的 KV block 存到 host tier”和“把 host-tier 前缀 block 恢复回 GPU page”这两条路径封装成 `save` / `query` / `load` / `flush` 接口；当前用于 Qwen3-4B 单卡 dense prefix offload。

## 一句话先理解

如果只想先抓住核心，可以把它理解成：

- `openinfer-kv-cache` / kvbm 决定“哪些 block 值得复用、它们的内容 hash 是什么、当前请求已经命中了多少前缀”；
- `pegaflow-core` 负责“把某些 block 从 GPU 搬到 host，或者再从 host 搬回 GPU”；
- `openinfer-kv-offload` 站在中间，把 openinfer 的 page-first GPU KV 布局翻译成 pegaflow 能搬的格式，并把搬运动作包装成上层 executor 能直接调用的 API。

它解决的不是“怎么管理 KV cache”，而是“怎么把已经管理好的 KV block 搬进搬出更深的存储层”。

## 它在系统里的位置

`openinfer-kv-offload` 只负责**搬运**，不负责**决策**。

- **它不拥有逻辑 KV cache**：哪些 block 属于哪个请求、哪些 block 已 seal、哪些 hash 代表同一段前缀，这些都由 `openinfer-kv-cache` / kvbm 管。
- **它不拥有模型执行**：什么时候 prefill、什么时候 decode、何时尝试 prefetch，也都由每个模型自己的 executor / scheduler 决定。
- **它只做桥接**：把 openinfer 的 GPU KV 表示法翻译成 pegaflow 能理解的注册/拷贝形式，并暴露同步/异步的 save/load 原语。

当前接线可概括成：

```text
Qwen3Executor / scheduler
  ├─ 逻辑层：BlockPool / RequestKv / SequenceHash
  ├─ 物理 GPU 层：KvBuffer（page-first fused buffer）
  └─ openinfer-kv-offload::OffloadEngine
       └─ pegaflow-core::PegaEngine
            └─ host pinned memory tier（dense v1 实际使用的就是这一层）
```

所以这个 crate 更接近“KV 数据平面的适配器”，不是新的 cache manager，也不是新的 prefix-cache 策略层。

## 当前实现边界

这份 crate 目前故意只覆盖一个很窄、但已经在线路上可用的场景：

- **单 GPU / 单 rank**：`TP_RANK` / `PP_RANK` / `WORLD_SIZE` 都固定为 0/1。
- **dense prefix offload**：针对 Qwen3-4B 这种 full-attention paged KV。
- **连续前缀恢复**：GPU 命中的前缀在前，CPU tier 命中的延续前缀在后；只支持一个连续区间，不支持“GPU 命中一段、CPU 命中一段、再 GPU 命中一段”的交错。
- **host tier 优先**：dense v1 实际依赖的是 pegaflow 的 host pinned memory；`PrefetchStatus::Loading` 在这里被当成 miss，因为当前路径不依赖 SSD/RDMA 的异步回填。

这也是为什么 `openinfer-kv-offload` 的 API 很小：它暴露的是当前这条数据通路真正需要的几个动作，而不是试图抽象出一个通用的多层 tiering 框架。

## 它到底在做哪两件事

从 executor 的视角看，这个 crate 只负责两件事：

### 1. SAVE：把 GPU 上已经稳定的 KV block 存到 host tier

冷请求跑完某个 step 后，只要有新的 full block seal 完成，就可以把这些 block 异步存到 host tier，供后续请求复用。

```text
GPU full block seal 完成
  -> 取出 (page_id, content_hash)
  -> OffloadEngine::save(...)
  -> pegaflow 异步做 GPU -> CPU 拷贝
  -> host tier 以后可被 query 命中
```

### 2. LOAD：把 host tier 命中的前缀 block 恢复回 GPU

新请求进来时，先看 GPU prefix cache 命中了多少；再拿剩余的连续前缀 hash 去问 host tier；如果 host tier 也命中，就先把那部分 block 恢复回 GPU，再让正常的 `match_and_add_prefix()` 把这段前缀吃进去。

```text
新请求到来
  -> 先看 GPU 已命中多少前缀
  -> 再查 host tier 是否还能接着命中
  -> 命中则异步 load 回 GPU 预留页
  -> 注册进 kvbm registry
  -> prefill 前重新 match_and_add_prefix()
```

这也是它名字里 “offload” 的真实含义：不是直接替代推理路径，而是给现有的 prefix reuse 多加一层更深的存储。

## 依赖的两类输入

### 1. 逻辑输入：block id 和 content hash

offload tier 必须按“内容”而不是“物理页号”寻址，否则跨请求前缀复用就不成立。

openinfer 侧提供这部分输入的是 `RequestKv` / `BlockPool`：

- `RequestKv::prompt_block_hashes()` 给出完整 prompt full blocks 的内容 hash。
- `RequestKv::assigned_block_hashes()` 给出当前请求已分配 block 的 `(page_id, content_hash)`。
- `BlockPool::probe_prefix()` 给出：
  - GPU 已命中的前缀长度；
  - 接下来可以拿去查 host tier 的 hash 列表；
  - 一组 `ImmutableBlock` pin，防止 load 期间被 eviction。

这里的 hash 是 kvbm 的 `SequenceHash`，最后打包成 16 字节 key 交给 offload tier。这个 crate 并不关心 hash 是怎么算出来的，只要求它满足一件事：**同一段 prompt 前缀，无论是哪次请求算出来，hash 都必须一样**。只有这样，host tier 里存的老 block 才能被新请求查到。

### 2. 物理输入：page-first GPU `KvBuffer`

`openinfer-kv-cache::KvBuffer` 拥有实际 GPU 内存。它的布局是 page-first fused buffer：

```text
page 0:
  layer 0: [K | V]
  layer 1: [K | V]
  ...
page 1:
  layer 0: [K | V]
  layer 1: [K | V]
  ...
```

布局参数由 `KvLayout` 提供：

- `kv_block_len`: 一段 K 或 V 的元素数
- `layer_stride`: 同一 layer 内 `[K|V]` 的跨度
- `page_stride`: 一个 page 覆盖所有 layers 的跨度

对 `openinfer-kv-offload` 来说，关键事实是：

- **同一 page 内，同一 layer 的 K 和 V 是连续的**；
- **但同一 layer 的相邻 pages 并不连续**，它们之间隔着整个 `page_stride`。

这直接决定了注册方式：每层可以看成 pegaflow 的一个“layer”，每个 block 的拷贝大小是这一层在一个 page 里的整段 `[K|V]`，而同层相邻 blocks 的步长则是 `page_stride`。

如果换句话说，就是：

- 对 openinfer 来说，它看到的是“一个大 fused buffer”；
- 对 pegaflow 来说，它更容易理解成“每层各有一串 block，只是这一串 block 在内存里是按 stride 跳着放的”。

## 初始化：`OffloadEngine::new` 做了什么

`OffloadEngine::new` 只做一次，通常在 executor 初始化阶段完成。它做三件事：

1. 建一个很小的内嵌 tokio runtime；
2. 建 `pegaflow_core::PegaEngine`；
3. 把 `KvBuffer` 注册给 pegaflow。

其中最核心的是第三步的布局翻译。

### `Registration::from_buffer` 如何翻译 page-first 布局

`Registration::from_buffer` 把 fused `KvBuffer` 变成 pegaflow 的“每层一条 strided registration”：

- `layer_names`: `0..num_layers`
- `data_ptrs[layer]`: 这一层在 fused buffer 里的起始地址
- `bytes_per_block[layer]`: 这一层一个 page 内 `[K|V]` 整段的字节数
- `block_stride_bytes[layer]`: 相邻 pages 对应这层数据的跨度，即 `page_stride * sizeof(bf16)`
- `segments[layer] = 1`
- `kv_stride_bytes[layer] = 0`

这里 `segments=1` 很重要：虽然语义上有 K 和 V，但它们在当前布局里本来就是挨在一起的，所以对 pegaflow 来说它就是“一块连续数据”，不需要走 K/V split 注册路径。真正需要额外描述的是“下一块同层数据在多远的位置”，这就是 `block_stride_bytes`。

可以把这个映射想成：

```text
openinfer fused page-first buffer
  └─ 视角切换
pegaflow layer 0: ptr = base + layer0_off, copy N bytes, next block += page_stride_bytes
pegaflow layer 1: ptr = base + layer1_off, copy N bytes, next block += page_stride_bytes
...
```

一旦注册成功，后面的 save/load 都不再需要重新解释布局，只传 block id 和 hash 即可。

## save 路径：为什么能安全地把 block 存出去

save 的目标是：把**已经 seal、且内容稳定**的 KV block 异步送到 pegaflow 的 host tier。

整体调用链：

```text
Qwen3Executor::save_sealed_blocks
  -> RequestKv::assigned_block_hashes()
  -> RequestKv::assigned_block_guards()
  -> OffloadEngine::save(...)
       -> runtime.spawn(...)
            -> PegaEngine::batch_save_kv_blocks_from_ipc(...)
```

### 为什么只保存 sealed blocks

`RequestKv::assigned_block_hashes()` 只返回已经注册完成的 full blocks。当前还在写的 partial tail block：

- 没有稳定 hash；
- 内容也还没 finalize；
- 即便提前存出去，后续也无法作为 prefix 复用。

所以 save 的粒度天然就是“seal 完的 block”。这不是实现细节，而是这个设计能正确复用的前提。

### 为什么要 `saved_cursor`

一个请求可能经历多次 prefill/decode step。每个 step 结束后再看 `assigned_block_hashes()`，前面已经保存过的 blocks 仍然会在列表里。

`Qwen3Executor.saved_cursor` 用来记住“这个请求已经保存到第几个 sealed block 了”，这样每次只增量保存新 seal 的部分，避免重复 D2H。

### 为什么 `save` 需要 `keep_alive`

`OffloadEngine::save` 是 fire-and-forget：函数返回时，真实 D2H 拷贝通常还没完成。

如果这时源 block 被释放，物理 page 可能被别的请求复用并覆写，结果 pegaflow 保存下来的就不是旧 hash 对应的数据，而是后来者的数据，属于静默腐蚀。

所以 `save` 额外接收一个 `keep_alive` 负载；Qwen3 这里传入的是 `Vec<KvBlockGuard>`，它把对应 blocks pin 住，直到异步 save 真正完成才 drop。

这可以理解成一句更直白的话：**save 发出去以后，源 page 还不能马上被别人拿去重用**。

### `save` 和 `save_blocking` 的区别

- `save`：异步 best-effort，失败只记日志，不影响推理正确性；因为 host tier 是 cache，save 丢了只是少一次未来命中。
- `save_blocking`：同步等待 D2H 捕获完成，用在必须确认“现在已经安全交接”的场景。

### save 的时序约束

`openinfer-kv-offload` 不拥有模型 compute stream，所以它**无法**自己保证“你现在读到的 KV 已经写完”。

因此 contract 是：调用者只能在 producing step 已经同步之后调用 `save`。Qwen3 的接线满足这一点，因为它在 `apply_prefill` / `apply_decode` 后、step 的 token readback 已完成时触发 save。

## query + load 路径：host tier -> GPU

load 的目标是：在新请求到来时，把 host tier 中命中的 prefix blocks 提前恢复到 GPU page，随后让正常的 prefix matching 把这些 blocks 吞进请求自己的 KV 状态。

整体流程分成四段。

### 1. 先探测 GPU 前缀，再决定要不要查 CPU tier

`BlockPool::probe_prefix()` 先做两件事：

- 计算当前 prompt 在 GPU prefix cache 里已经命中了多少完整 blocks；
- 返回“剩下哪些 full-block hash 可以继续去 CPU tier 查”。

它还会持有一组强引用，把 GPU 命中的 blocks pin 住，防止 query/load 期间被 eviction。

### 2. `query` 只问“CPU tier 还能接上多少连续前缀”

`OffloadEngine::query(req_id, block_hashes)` 会调用 pegaflow 的 prefix lookup。

返回值是：

- `num_blocks`：CPU tier 连续命中的 block 数；
- `lease`：这些命中 blocks 的临时所有权。

这里 lease 的含义很重要：query 不是单纯“数一下命中了几个”，而是“把这些命中的 host blocks 暂时 pin 给这个请求”，直到后续 `load` 消耗掉，或者显式 `release_query_lease` 释放。

如果当前路径收到 `PrefetchStatus::Loading`，dense v1 直接把它当 miss 处理，不在 admission 上等待更深 tier。

### 3. 先预留 GPU 落点，再提交 load

query 命中之后，并不能立刻 load；还必须先在 GPU `BlockPool` 里预留足够的空 page。

Qwen3 的流程是：

1. 检查 `available_blocks - reserve_floor` 是否足够；
2. `reserve_loaded_blocks(num_blocks)` 申请一组 `MutableBlock`；
3. 取出它们的 `page_ids()`；
4. 调 `OffloadEngine::load(lease, dst_page_ids)`。

`load` 自己不阻塞，它只返回一个 `LoadHandle`。底层 pegaflow 在自己的 worker 上执行 CPU -> GPU DMA，完成后通过 oneshot 回传结果。

也就是说，`openinfer-kv-offload` 只负责“把 host block 搬回某几个 page id”，并不负责“这些 page id 在逻辑上属于哪个请求”——那个归属关系仍然由 kvbm 在 commit 阶段建立。

### 4. load 完成后，再把这些 blocks 注册进 prefix cache

load 落地并不意味着请求已经自动拥有这些 blocks。它们此时只是“GPU 上写好了数据的物理页”。

真正把它们变成可复用 prefix 的动作在 `settle_prefetch`：

- 拿到 `LoadReservation`
- 对每个 loaded block 调 `stage(hash, block_size)`
- 用 `BlockManager::register_block(...)` 注册进 kvbm registry
- 把这些新注册的 blocks 并入 `PrefixProbe.held`

这样一来，下一次该请求正式 `new_request().match_and_add_prefix()` 时，就会把“GPU 原生命中 + CPU 刚恢复回来的 continuation”看成一个连续前缀，一次性吞掉。

这是整个设计里最关键的一层解耦：

- `load` 只恢复“字节”；
- `commit_loaded_blocks` 才恢复“逻辑身份”；
- `match_and_add_prefix()` 最终把这些逻辑 block 吸收到请求自己的序列状态里。

## 为什么 load 是异步的

当前设计故意让 load 不阻塞 admission 主路径：

- `begin_kv_prefetch()` 只负责提交 query/load；
- scheduler 每 tick 用 `drain_ready_prefetch()` 轮询；
- 真需要等的时候才 `wait_ready_prefetch()` 阻塞等一个。

这使得 CPU-tier restore 能以“后台预取”的方式并入现有 scheduler，而不把所有 admission 都变成同步 IO。

`LoadHandle` 本身也只提供两个操作：

- `poll()`：非阻塞检查
- `wait()`：阻塞直到完成

这和当前需求是完全对齐的；crate 没有引入更复杂的 future/stream 接口。

## 为什么 query lease 和 prefix pin 这么重要

这一层最容易被误解的地方，是它不只是“查到了就去拷贝”，还必须在查到以后把那批 block 临时 pin 住。

原因很简单：命中不等于已经消费。

- query 命中时，host tier 里的 block 只是“准备给你用”；
- load 还没完成时，GPU 目标 page 只是“预留好了”；
- commit 完成但请求还没开始 prefill 时，prefix 也只是“刚注册回 registry”。

这几段空窗如果不 pin 住，block 可能在你真正消费前就被别的路径回收或替换掉。

## 失败路径和资源回收

这个 crate 最容易出错的地方不是 happy path，而是“命中了但最后没 load 成”的清理。

### lease 必须显式释放

`query()` 返回的 `QueryLeaseId` 只是一个 token；它没有 Drop 语义。也就是说：

- query 成功；
- 但后面因为 GPU page 不够、或者 load 提交失败；
- 如果不手动 `release_query_lease(lease)`，

那批 host-tier blocks 会一直被 pegaflow pin 到 TTL 超时。

因此 Qwen3 接线在两个地方都显式释放 lease：

- `reserve_loaded_blocks()` 失败时；
- `load()` 提交失败时。

### prefetch commit 前也要 pin 住

`PrefixProbe` 里维护的 `held` 不只是初始 GPU-hit blocks；CPU restore 成功并 register 之后，它还会继续持有那些新注册 blocks，直到请求真正 `match_and_add_prefix()`。

这是为了封住“刚注册进 prefix cache，但正式请求还没 consume”这段窗口，避免中间被 eviction 掉。

## 为什么需要内嵌 tokio runtime

`pegaflow-core` 的 save/query flush 接口是 async，但 openinfer 的 scheduler/executor 主体是同步线程模型。

所以 `OffloadEngine` 直接内嵌一个很小的 tokio runtime，把异步世界包在 crate 内部：

- save 用 `runtime.spawn(...)`
- query / save_blocking / flush_saves 用 `runtime.block_on(...)`

`assert_outside_runtime()` 的存在就是为了保护这个边界：这些 `block_on` 入口必须从同步线程调用，不能在另一个 tokio runtime 里套娃。

换句话说，这个 crate 的设计目标之一就是：**不要把 async 传染到模型 executor 的主控制流**。

## 从请求生命周期看一遍完整流程

把上面的零散细节串起来，一个“冷请求保存 + 热请求恢复”的完整生命周期大致是：

```text
第一次请求（冷）
  1. prefill / decode 正常运行
  2. 某些 full block seal 完成
  3. save_sealed_blocks() 取出 (page_id, hash)
  4. OffloadEngine::save() 异步搬到 host tier

第二次请求（热）
  5. probe_prefix() 先看 GPU 前缀命中了多少
  6. query() 再看 host tier 能否继续接上
  7. reserve_loaded_blocks() 先预留 GPU 目标页
  8. load() 异步把 host block 搬回这些页
  9. settle_prefetch() + commit_loaded_blocks() 注册回 kvbm
 10. match_and_add_prefix() 把恢复的前缀吞进请求状态
 11. 只对剩余 tail 做真正 prefill
```

如果只记住一句话，那就是：**它让“GPU 没保住的前缀”还能从 host tier 找回来，并重新伪装成普通的 GPU prefix hit。**

## Qwen3 实际调用链

Qwen3 目前是这套 crate 的首个真实使用者。其调用链可以概括成：

```text
启动:
  Qwen3Executor::single
    -> build_offload(...)
    -> OffloadEngine::new(...)

冷请求执行:
  execute_prefill / execute_decode
    -> apply_prefill/apply_decode
    -> save_sealed_blocks(...)
    -> OffloadEngine::save(...)

热请求 admission:
  begin_kv_prefetch(...)
    -> BlockPool::probe_prefix(...)
    -> OffloadEngine::query(...)
    -> reserve_loaded_blocks(...)
    -> OffloadEngine::load(...)

prefetch 完成:
  drain_ready_prefetch / wait_ready_prefetch
    -> settle_prefetch(...)
    -> commit_loaded_blocks(...)

正式 prefill:
  schedule_prefill_chunk(...)
    -> RequestKv::match_and_add_prefix(...)
```

可以看到 `openinfer-kv-offload` 没有侵入模型前向逻辑；它只在“seal 后保存”和“prefill 前恢复”两个边界点介入。

## 两个测试分别保证什么

### `openinfer-kv-offload/tests/cpu_roundtrip.rs`

这个测试只关心 crate 本身的字节路径是否正确：

- 真正分配一个 page-first `KvBuffer`
- 往若干 source blocks 写入可区分的模式
- save 到 pegaflow host tier
- load 到另一组 destination blocks
- 逐 layer / segment / block 校验字节完全一致

它验证的是：

- `Registration::from_buffer` 的布局映射正确；
- `block_stride_bytes` 的 strided registration 正确；
- `save/query/load` 这条原始数据通路没把字节写到错误位置。

### `openinfer-qwen3-4b/tests/kv_offload_cpu_hit.rs`

这个测试关心系统接线是否正确：

- 第一个请求冷算并 save sealed blocks；
- 清空 GPU inactive cache；
- 第二个同前缀请求从 CPU tier restore；
- restore 后的 first-token logits 必须与冷算一致到 bf16 floor。

它验证的是：

- query / lease / load / commit 的时序；
- `begin_kv_prefetch()` 到 `match_and_add_prefix()` 的衔接；
- “GPU 命中一段 + CPU 命中续段” 组合前缀不会错位。

前者证明“字节搬对了”，后者证明“恢复回来的 KV 真能继续推理，而且数值没歪”。

## 这个 crate 刻意没有做的事

为了看清边界，也值得明确它**没有**做什么：

- 没有抽象统一的 per-model residency policy；
- 没有处理 recurrent/SSM state；
- 没有处理 sparse active-set gather；
- 没有把 TP / 多 rank 的 offload 编排做成通用方案；
- 没有把 pegaflow 更深层的 SSD/RDMA 取回逻辑暴露到 scheduler API。

这些都属于上层调度策略或未来更泛化的数据平面工作；`openinfer-kv-offload` 当前只承担已经跑通、且被 Qwen3 真实需要的那一小段桥接职责。

