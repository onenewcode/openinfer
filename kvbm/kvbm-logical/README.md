# kvbm-logical 工作原理

`kvbm-logical` 是 KVBM（KV Block Manager）的**逻辑层** crate。它不负责真正的 GPU kernel，也不直接拥有某个模型的 forward；它负责的是“KV block 在生命周期里应该处于什么状态、如何注册、如何匹配、如何被回收”的这一套规则。

## 一句话先理解

可以把它理解成：

- 物理层关心“KV 数据到底放在 GPU、CPU 还是别的 tier”
- 模型层关心“prefill / decode 什么时候需要这些 KV”
- `kvbm-logical` 关心“一个 block 现在是可写、已完成、已注册、还是弱引用，以及它怎样安全地流转”

所以它解决的是 **KV block 的逻辑生命周期与复用语义**。

## 它在系统里的位置

大致关系是：

```text
模型 executor / scheduler
  -> kvbm-logical
      -> block 状态机 / registry / pool
  -> 物理存储层（GPU / CPU / disk 等）
```

这个 crate 之所以叫 `logical`，就是因为它刻意不把自己绑定到某一种具体存储介质。

## 最重要的设计：type-state 生命周期

这个 crate 的核心不是某个函数，而是它用类型状态把 block 生命周期写进了 API：

```text
MutableBlock<T> -> CompleteBlock<T> -> ImmutableBlock<T> <-> WeakBlock<T>
```

直觉上可以这样理解：

- `MutableBlock`：刚分配出来，还能写
- `CompleteBlock`：内容已经完整，且带上了 `SequenceHash`
- `ImmutableBlock`：已经注册进 registry，可以参与前缀匹配与缓存复用
- `WeakBlock`：不拥有 block，只是一个可升级的弱引用

这个设计的意义在于：很多“本来要靠注释约束”的生命周期规则，被提前变成了编译期约束。

## 它到底在管理什么

### 1. block 状态转换

一个 block 不能随便从“可写”跳到“可复用”。

它必须经历：

1. 分配
2. 写满 / 完成
3. 计算并绑定 `SequenceHash`
4. 注册进 registry
5. 之后才能被别的请求按前缀匹配拿到

`kvbm-logical` 的价值，就是把这个流程变成显式、类型安全的状态机。

### 2. registry

一旦 block 进入 `ImmutableBlock`，它就不只是“一块内存”，而是可被查找的缓存对象。

registry 负责：

- 用 `SequenceHash` 建索引
- 支持前缀匹配
- 处理重复注册 / 去重
- 维持“逻辑上有哪些 block 处于可复用状态”

所以 registry 更像 KV block 的“逻辑目录”。

### 3. pool

pool 负责 block 的回收与再利用。

文档里能看到至少两类重要池子：

- reset pool
- inactive pool

大致语义是：

- reset pool：可以重新写入的新鲜 block
- inactive pool：内容还在、但当前没有强引用持有，可被缓存复用或驱逐

这让 block 的 drop 行为不只是“释放内存”，而是“回到合适的池中，等待下一次逻辑用途”。

## `SequenceHash` 为什么重要

这个 crate 的前缀复用不是按“物理地址”匹配，而是按“逻辑内容 + 位置血缘”匹配。

`SequenceHash` 的存在，是为了保证：

- 同一段 prompt 前缀，在不同请求里能映射到同一逻辑 block
- block 匹配基于内容身份，而不是某次运行时的偶然 page id

没有这一层，prefix cache 很难做成真正跨请求的逻辑复用。

## `WeakBlock` 为什么单独存在

`WeakBlock` 的意义是：你有时需要“记住一个 block”，但又不想因此阻止它被驱逐。

所以它提供一种非拥有引用：

- 不增加 block 的强生命周期
- 但在 block 还活着时，仍可以尝试 upgrade 回 `ImmutableBlock`

这在缓存系统里很重要，因为“可引用”和“必须保活”不是同一件事。

## 最重要的边界

`kvbm-logical` 不负责：

- 真正的数据搬运
- GPU kernel
- 模型 forward
- 某个具体 tier 的内存布局

它负责的是：**让 KV block 从“可写数据”变成“可复用缓存对象”的逻辑过程可证明、可组合、可回收。**

下面保留原始英文 README 里的更细 API 和示例，方便对照具体类型与用法。

# kvbm-logical

Logical block lifecycle management for KVBM (KV Block Manager). Manages KV cache blocks for LLM inference through a type-safe state machine, registry, and pool system.

## Block Lifecycle

Blocks follow a compile-time enforced state machine via the type-state pattern:

```text
MutableBlock<T> → CompleteBlock<T> → ImmutableBlock<T> ⇄ WeakBlock<T>
   (Reset)           (Staged)          (Registered)       (Non-owning)
```

- **MutableBlock** — Allocated from the reset pool, writable. Drop returns to the reset pool.
- **CompleteBlock** — Staged with a `SequenceHash` but not yet registered. Drop returns to the reset pool.
- **ImmutableBlock** — Registered in the block registry. Strong-ref prevents eviction. Drop moves to the inactive pool for caching.
- **WeakBlock** — Non-owning reference that does not prevent eviction. Upgradeable back to `ImmutableBlock` via two-phase lookup.

The type parameter `T: BlockMetadata` is a marker for the storage tier (e.g. GPU, CPU, disk).

## Usage

```rust,no_run
use kvbm_logical::{
    BlockManager, BlockRegistry, MutableBlock, CompleteBlock, ImmutableBlock, WeakBlock,
    SequenceHash,
    manager::FrequencyTrackingCapacity,
};

# fn main() {
// Any Clone + Send + Sync + 'static type satisfies BlockMetadata.
#[derive(Clone)]
struct G2; // CPU tier marker

// Build a registry with TinyLFU frequency tracking.
let tracker = FrequencyTrackingCapacity::Medium.create_tracker();
let registry = BlockRegistry::builder()
    .frequency_tracker(tracker)
    .build();

// Build the block manager with an LRU eviction backend.
let manager = BlockManager::<G2>::builder()
    .block_count(1024)
    .block_size(16)
    .registry(registry)
    .with_lru_backend()
    .build()
    .expect("failed to build block manager");

// Allocate mutable blocks from the reset pool.
let mut blocks: Vec<MutableBlock<G2>> = manager
    .allocate_blocks(2)
    .expect("not enough blocks available");

// Stage a block with a pre-computed sequence hash, producing a CompleteBlock.
// SequenceHash wraps a positional lineage hash computed from token data.
let seq_hash_0 = SequenceHash::new(42, None, 0);
let complete: CompleteBlock<G2> = blocks
    .remove(0)
    .stage(seq_hash_0, manager.block_size())
    .expect("block size should match");

// Register the staged block, producing an ImmutableBlock.
let immutable: ImmutableBlock<G2> = manager.register_block(complete);

// Prefix-match registered blocks by sequence hash.
let matched: Vec<ImmutableBlock<G2>> = manager.match_blocks(&[seq_hash_0]);
assert_eq!(matched.len(), 1);

// Downgrade to a WeakBlock (does not prevent eviction).
let weak: WeakBlock<G2> = immutable.downgrade();

// Upgrade back to ImmutableBlock if the block hasn't been evicted.
if let Some(restored) = weak.upgrade() {
    assert_eq!(restored.sequence_hash(), seq_hash_0);
}

// RAII: dropping an ImmutableBlock moves it to the inactive pool for caching.
{
    let temporary = manager.match_blocks(&[seq_hash_0]);
    // `temporary` dropped here → block returns to inactive pool
}

// Introspect pool state.
let available = manager.available_blocks();
let total = manager.total_blocks();
# }
```

## Prometheus Metrics

All metrics carry a `pool` label identifying the storage tier.

### Counters

| Name | Description |
|------|-------------|
| `kvbm_allocations_total` | Total blocks allocated from pools |
| `kvbm_allocations_from_reset_total` | Total blocks allocated from the reset pool |
| `kvbm_evictions_total` | Total blocks evicted from inactive pool |
| `kvbm_registrations_total` | Total blocks registered (CompleteBlock → ImmutableBlock) |
| `kvbm_duplicate_blocks_total` | Total duplicate blocks created (Allow policy) |
| `kvbm_registration_dedup_total` | Total block registrations deduplicated (Reject policy) |
| `kvbm_stagings_total` | Total MutableBlock → CompleteBlock transitions |
| `kvbm_match_hashes_requested_total` | Total hashes requested in match_blocks calls |
| `kvbm_match_blocks_returned_total` | Total blocks returned from match_blocks calls |
| `kvbm_scan_hashes_requested_total` | Total hashes requested in scan_matches calls |
| `kvbm_scan_blocks_returned_total` | Total blocks returned from scan_matches calls |

### Gauges

| Name | Description |
|------|-------------|
| `kvbm_inflight_mutable` | Current MutableBlocks held outside pool |
| `kvbm_inflight_immutable` | Current ImmutableBlocks held outside pool |
| `kvbm_reset_pool_size` | Current reset pool size |
| `kvbm_inactive_pool_size` | Current inactive pool size |
