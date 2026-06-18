# openinfer-kv-cache 工作原理

`openinfer-kv-cache` 是 openinfer 的 paged KV 基础层：它把“逻辑上的 block 生命周期与前缀复用”放进 `BlockPool` / `RequestKv`，把“GPU 上 page-first KV 内存”放进 `KvBuffer` / `KvLayout`，再用 `KvView` 把某个请求当前这一步真正可见的 page 表交给 attention kernel。它本身不做模型前向，但决定了请求的 KV 如何分配、复用、增长、释放，以及如何安全地映射到 GPU 上。

## 一句话先理解

如果只先抓核心，可以把这个 crate 理解成：

- `BlockPool` 负责回答“这个请求现在拥有哪些 KV block、还能不能继续分配、有没有前缀可以复用”；
- `KvBuffer` 负责回答“这些 block 在 GPU 内存里到底放在哪”；
- `KvView` 负责回答“这次 prefill / decode 这一步，kernel 应该看到哪些 page、序列长度是多少”。

也就是说，`openinfer-kv-cache` 解决的不是“怎么算 attention”，而是“attention 要读写的 KV 状态该怎么组织”。

## 它在系统里的位置

这个 crate 站在模型 executor 和更底层 kernel 之间：

```text
executor / scheduler
  ├─ RequestKv / BlockPool
  ├─ KvView
  └─ KvCacheManager
       ├─ BlockPool（逻辑层）
       └─ KvBuffer（GPU 物理层）
            └─ openinfer-kernels / FlashInfer 读取 page table
```

从职责上看，它刻意把 KV cache 拆成两半：

- **逻辑层**：谁拥有 block、block 什么时候 seal、哪些 block 可以拿来做 prefix cache；
- **物理层**：GPU 上有一大块 page-first buffer，某个 block id 对应这块 buffer 的哪个 page；

这样做的好处是：像 Qwen3 这种 full-attention 模型，可以直接用 `KvCacheManager = BlockPool + KvBuffer`；而像 Kimi-K2 这种 MLA 模型，也可以继续复用 `BlockPool` / `RequestKv`，但自己实现物理 buffer。

## 这个 crate 里的三层抽象

### 1. `KvLayout`：先定义一页 KV 长什么样

`KvLayout` 是最底层的几何定义，它不分配显存，只回答一个问题：一个 page 内，各层 K/V 数据怎么排布。

它的几个关键字段是：

- `page_size`：每个 block/page 能装多少 token
- `num_layers`
- `num_kv_heads`
- `head_dim`
- `kv_block_len`：一段 K 或 V 的元素数
- `layer_stride`：同一层里 `[K|V]` 这整段的跨度
- `page_stride`：整个 page 覆盖所有层的总跨度

对 full-attention 模型，它描述的是 page-first 布局：

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

这个定义看起来简单，但后面所有组件都依赖它：

- `KvBuffer` 用它决定要分配多大显存；
- `KvViewDesc` 用它把 page table 交给 kernel；
- `openinfer-kv-offload` 用它把 fused buffer 映射到 pegaflow 的 strided registration。

### 2. `KvBuffer`：真正的 GPU KV 内存

`KvBuffer` 只做一件事：在 GPU 上分配一整块连续的 bf16 显存，按 `KvLayout` 的 page-first 几何组织起来。

它有几个很重要的特点：

- **它自己不分配 block**：它只是“物理存储”，不管哪个请求现在用到哪一页；
- **它是 fused buffer**：所有 pages、所有 layers 都放在一大块 `CudaSlice<bf16>` 里；
- **它的 device pointer 稳定**：buffer 创建后地址不变，所以其他层可以安全地长期引用它。

所以 `KvBuffer` 更像“显存仓库”，而不是“请求级 KV 状态”。

## 逻辑层：`BlockPool` 为什么是核心

真正决定 KV cache 行为的，其实是 `BlockPool`。

`BlockPool` 内部包了一层 kvbm 的 `BlockManager<()>`，再加上一些 openinfer 自己的约束。它本质上做三件事：

- 管理 block 的生命周期与可用数量；
- 做 prefix cache 匹配；
- 给请求返回可供调度和释放的 block 句柄。

### padding block 为什么一上来就保留

`BlockPool::new` 创建后，会先保留一个 padding block，并把它永久注册出去。

这么做的目的不是为了业务语义，而是为了 CUDA Graph / paged attention 的元数据稳定性：有些 padding row 需要一个永远存在的 page id，kernel 才能安全构造固定形状的 page table。

因此：

- 总 block 数至少要 >= 2；
- 真正能分给请求的 block 数是 `total_blocks - 1`。

### 为什么 `BlockPool` 不拥有 GPU 内存

这点很重要：`BlockPool` 只知道 block id，不知道 block 对应的 GPU 地址。

这样设计是为了让“逻辑生命周期”和“物理布局”彻底解耦：

- full-attention：block id 对应 `KvBuffer` 的某个 page；
- MLA：block id 也可以对应另一种双 buffer 物理布局；
- 将来换其他物理形式，也不必重写 prefix cache / request 生命周期。

## `RequestKv`：一个请求自己的 KV 状态机

`RequestKv` 是每个请求的逻辑 KV 视角，它内部包的是 kvbm 的 `SchedulableSequence<()>`。

如果说 `BlockPool` 是全局的 KV 资源池，那么 `RequestKv` 就是“这个请求目前走到哪里了，它拥有哪些 block，它下一步该怎么长”。

它的生命周期可以浓缩成：

```text
new_request
  -> match_and_add_prefix（可选）
  -> schedule_prefill
  -> prefill_view
  -> apply_prefill / apply_prefill_chunk
  -> schedule_decode
  -> decode_view
  -> apply_decode
  -> ...循环...
  -> release
```

## prefix cache 是怎么接进来的

`RequestKv::match_and_add_prefix()` 是这个 crate 最有价值的入口之一。

它会把 prompt 的 full blocks 去和 `BlockPool` 里已注册的 blocks 做匹配，如果前缀命中，就直接跳过这部分 prefill。

但它有一个刻意保留的约束：

- **永远至少留 1 个 prompt token 不缓存**

原因是首个生成 token 仍然需要一次真正的 prefill forward 来产出。如果把整个 prompt 都完全“吃干抹净”，这次请求就没机会生成 first token 了。

所以 prefix cache 的复用边界不是“能复用多少就复用多少”，而是“最多复用到最后一个仍保留首 token 生成语义的位置”。

## 调度阶段：为什么要先 schedule，再 build view

`RequestKv` 里的 `schedule_prefill()` / `schedule_decode()` 负责向 `BlockPool` 申请 block。

这个顺序很关键：

1. 先 schedule，决定本 step 需要哪些 block；
2. 再 build `KvView`，把“这一步真正该看到的 page 表”交给 kernel；
3. forward 跑完后，再 `apply_prefill()` / `apply_decode()` 把这批 block 正式注册成有效 KV。

也就是说：

- **schedule 阶段**解决“这一步能不能跑、需要先占哪些 block”；
- **apply 阶段**解决“这一步真的跑完了，把哪些 block 变成已写好的 KV”。

这种拆分能保证：

- 还没真正算出来的 block，不会过早注册进 prefix cache；
- decode 跨 block 边界时，可以提前占好下一页，避免半路 OOM。

## `KvView`：为什么要有这层视图

`KvView` 看起来很轻，但它实际上承担了一个很关键的安全边界：告诉 kernel，这一步到底有哪些 page 是有效的。

它只保存三样东西：

- `page_indices`
- `seq_len`
- `page_size`

但这正是 paged attention kernel 真正需要的核心元数据。

### 为什么不能直接把请求持有的所有 block 都交给 kernel

因为 `RequestKv::page_indices()` 代表的是“这个请求当前持有的所有物理页”，而不一定等于“这一步真正可见的序列长度”。

尤其在 decode 场景里，kvbm 可能会**提前分配下一页**。如果你把原始 block 列表不加裁剪地交给 kernel，kernel 会误以为序列更长，于是：

- 读到还没写过的垃圾 KV；
- 或者直接越界。

所以 `step_page_indices(new_tokens)` 的语义不是“列出我有多少页”，而是：

- 假设这一步会 append `new_tokens`
- 那么当前 forward 结束后可见的序列长度是多少
- 为这个精确长度截出刚好够用的 page 列表

这就是 `KvView::new()` 里那个强断言的意义：page 数必须精确覆盖 `seq_len`，不能多也不能少。

## prefill view 和 decode view 的区别

`RequestKv` 会构造两种视图：

- `prefill_view(prompt_len)`：表示“从当前 `kv_position()` 开始，再 append 一段 prompt 后”的可见状态；
- `decode_view()`：表示“在当前 KV 后再 append 一个新 token”的可见状态。

它们的共同点是：

- 都返回一个不可变 `KvView`；
- 都要求 page row 精确覆盖本 step 的目标序列长度；

区别只在于这一步 append 的 token 数不同：

- prefill 是一段；
- decode 永远是 1 个。

## 为什么 `KvViewDesc` 还要再包一层

`KvView` 还不是 kernel 最终看到的东西。真正交给 kernel 的是 `KvViewDesc`：

- 它把 `KvView` 的 page table
- 和 `KvBuffer` 的物理 buffer
- 以及 `KvLayout`

打包成一个 kernel-facing metadata bundle。

这样 kernel 侧就能同时拿到：

- 页表：该读哪些 page id
- 几何：每页内部怎么解释
- 基址：显存真正在哪里

这个分层让 `KvView` 保持逻辑轻量，而 `KvViewDesc` 只在真正下沉到 kernel 之前才拼装物理细节。

## offload 为什么也复用这个 crate

虽然 `openinfer-kv-offload` 是另一个 crate，但它几乎完全建立在 `openinfer-kv-cache` 提供的逻辑之上。

这里最关键的几个接口是：

- `RequestKv::prompt_block_hashes()`：给 host tier query 用的内容键
- `RequestKv::assigned_block_hashes()`：save sealed blocks 时记录 `(page_id, hash)`
- `RequestKv::assigned_block_guards()`：异步 save 时 pin 住源 block
- `BlockPool::probe_prefix()`：先看 GPU prefix hit，再决定 CPU tier 该查什么
- `reserve_loaded_blocks()` / `commit_loaded_blocks()`：给 CPU -> GPU restore 提供落点并注册回 registry

所以可以把 `openinfer-kv-offload` 看成是：

- 用 `openinfer-kv-cache` 维护的逻辑 block 身份和 page id；
- 再给这些 block 多加一个 host-tier 的存取通道。

## LoRA 为什么会影响 prefix cache

`BlockPool::new_request()` 里会把 `lora_name` 折成一个 salt，加进 block hash 链。

这意味着：

- 同一段 token，如果跑在 base model 和某个 LoRA adapter 下；
- 即使文本完全一样；
- 它们得到的 block hash 也不同。

这样做是为了防止不同权重下算出来的 K/V 被静默复用。换句话说，prefix cache 不是“按 token 文本”复用，而是“按 token 文本 + 当前 adapter 身份”复用。

## 为什么 Kimi 也能复用这套逻辑

Qwen3 用的是完整的 `KvCacheManager = BlockPool + KvBuffer`，但 Kimi-K2 在 scheduler 里也直接复用了 `BlockPool` / `RequestKv`。

这正好说明这套分层的意义：

- `BlockPool` / `RequestKv` 负责的是 block 生命周期、prefix cache、请求级增长与释放；
- `KvBuffer` 只是 full-attention 的一种物理落地方式；

所以只要模型也需要“按 block 管理 KV”和“按前缀复用”，就能复用逻辑层，即使它的物理 KV 布局完全不同。

## 从一次请求的角度看完整流程

把这些组件串起来，一个请求在这个 crate 里的完整路径大致是：

```text
1. executor 调 BlockPool::new_request() 创建 RequestKv
2. 可选：match_and_add_prefix() 直接吃掉已缓存前缀
3. schedule_prefill() 为 suffix 申请 block
4. prefill_view() 构造本 step 精确 page table
5. kernel 根据 KvViewDesc 写入 KvBuffer
6. apply_prefill() / apply_prefill_chunk() 注册新 block，推进 kv_position
7. decode 时 schedule_decode()，必要时提前拿下一页
8. decode_view() 构造“当前 + 1 token”的精确 page table
9. apply_decode() 注册新 token，继续推进
10. 请求结束后 release() 归还 block
```

如果只记一句话，那就是：`openinfer-kv-cache` 让“请求的 KV 逻辑状态”和“GPU 上的 paged KV 物理布局”既能配合，又不互相绑死。

## 这个 crate 刻意没有做的事

为了看清边界，也值得明确它**没有**做什么：

- 不负责模型前向；
- 不负责采样；
- 不负责 scheduler 的 admission 策略；
- 不负责 host/SSD/RDMA offload；
- 不负责 MLA 这类非 full-attention 布局的物理 buffer 实现。

这些要么在更上层的 executor/scheduler，要么在其他专门 crate 中完成。`openinfer-kv-cache` 当前专注在 paged KV 的逻辑/物理桥梁本身。

