# openinfer-sample 工作原理

`openinfer-sample` 是 openinfer 的**共享采样层**。它不负责模型前向，也不负责产生 logits；它做的事是：**把一块 logits arena 变成“每一行下一 token 是什么”，并在需要时把某一行 logits 转成 host 侧 logprobs。**

如果只记一句话，可以这样理解：

- `openinfer-kernels` 负责低层 GPU 采样 primitive；
- `openinfer-engine` 定义 `SamplingParams` / `TokenLogprob` 这类协议类型；
- `openinfer-sample` 站在中间，负责“采样策略”和“共享 scratch 管理”。

## 它在系统里的位置

这个 crate 处在“模型 executor 与 kernel 采样 primitive 之间”的位置：

```text
模型 executor
  -> openinfer-sample
       ├─ select_batch(...)
       ├─ token_logprob_from_row(...)
       └─ SampleScratch
  -> openinfer-kernels::ops
```

所以它的价值不是“再写一遍 kernel”，而是把采样这扇门统一起来，让不同模型都用同一套策略层。

## 这个 crate 解决的两个问题

它只做两个模型无关的工作：

### 1. 从 batched logits 里选出下一个 token

也就是：

```text
一块 logits arena
  -> 每一行一个 next token id
```

### 2. 把一行 logits 变成 host 侧 logprobs

也就是：

```text
一行 full-vocab logits
  -> sampled token 的 logprob + top-k logprobs
```

这两个问题看起来简单，但如果每个模型 crate 各写一遍，很容易出现：

- greedy / sampling 分流逻辑不一致；
- top-k tie-breaking 不一致；
- logprob 精度不一致；
- scratch buffer 管理各写各的。

`openinfer-sample` 的意义，就是把这些统一掉。

## `select_batch` 到底在做什么

`select_batch(...)` 是这个 crate 的核心入口。

它输入：

- `logits: &HiddenStates`
- `params: &[&SamplingParams]`
- `seed`
- `scratch`

输出：

- `Vec<u32>`：每一行一个 token id

但它内部不是“对每一行单独采样一次”，而是故意做成**批处理**：

- greedy 行一起走一遍 batched argmax；
- 非 greedy 行一起走一遍 batched FlashInfer sampling；

这样可以避免最坏的退化：

```text
for row in rows {
    单独调一次采样 kernel
}
```

这也是它存在的一个核心理由：**保证 token 选择仍然是 batched 的，而不是被上层不小心写回逐行采样。**

## 为什么要分 greedy 路径和 sampling 路径

不是所有请求都需要真实采样。

对很多请求来说，只要满足下面任一条件，就等价于 argmax：

- `temperature < 1e-5`
- `top_k == 1`

这就是 `SamplingParams::is_greedy()` 的判定。

因此 `openinfer-sample` 做的第一件事，就是把一批 rows 分成两类：

- **greedy rows**
- **non-greedy rows**

然后分别处理。

这么做有两个好处：

1. greedy 行不需要走 softmax / rejection sampling，成本更低；
2. 语义更稳定，尤其是 top tie / 近 greedy 场景更可预测。

## 什么叫 “effectively greedy”

这个 crate 还有一个很重要的判断：`effectively_greedy(...)`。

它表示：

- 虽然参数形式上不是显式 greedy；
- 但从概率语义看，最终 nucleus 里其实只会剩下一个 token；
- 那它也应该走 argmax，而不是走真正 sampling。

当前使用的规则是：

```text
top_p <= 1 / vocab_size
```

原因是 softmax 的最大概率一定 `>= 1 / vocab_size`，所以当 `top_p` 小到这个程度时，nucleus 最终只会包含 argmax token。

这件事非常重要，因为如果这种“实际上只有一个 token 会留下”的行仍然走 sampler：

- 在 bf16 tie 场景下，可能会因为随机过程选中不同 top token；
- 本来应该 deterministic 的请求反而变得不稳定。

所以这个 crate 在这里做的是**语义层面的去随机化**。

## `SampleScratch` 为什么必须独立成一个结构

采样不是只要一个输入 logits 就够了，内部还需要很多临时 device buffer：

- greedy row indices
- argmax partial values / indices
- top-1 values
- argmax 输出
- FlashInfer sampling scratch

`SampleScratch` 就是把这些 buffer 一次性分配好，然后跨 decode steps 复用。

它的重要性在于两点：

### 1. decode 路径需要 pointer-stable scratch

如果每步临时重新分配：

- 会有额外分配开销；
- 也不利于图捕获/稳定地址场景。

所以这套 scratch 被设计成 allocate-once、step-by-step reuse。

### 2. vocab 维度是 scratch 结构的一部分

每个 buffer 的大小都依赖 vocab 宽度，因此 `SampleScratch` 会把 `vocab` 记下来，并在 `select_batch` 时校验：

- logits 的 vocab 宽度
- scratch 创建时的 vocab 宽度

必须完全一致。

这避免了一个很危险的问题：如果 vocab 不一致，最坏情况不是“结果有点错”，而是 kernel 直接越界。

## greedy 行具体怎么选 token

对 greedy 行，`select_batch` 并不是逐行取 argmax，而是：

1. 先把 greedy row 的索引压成一个 `row_indices` 向量；
2. 把这个向量 H2D；
3. 调一次 batched indexed argmax；
4. 再 D2H 回 token id；
5. 按原行号放回输出数组。

所以 greedy 路径的本质是：

**从“全 arena 所有行”里，挑一批指定行，一次性做 batched argmax。**

这比逐行调用 argmax kernel 成本低很多。

## non-greedy 行具体怎么选 token

对 non-greedy 行，`select_batch` 会构造一组 `BatchSamplingRow`：

- 哪一行
- temperature
- top_k
- top_p

然后把这些 row 一起交给 `gpu_sample_batch_into(...)`。

所以 sampling 路径不是“每行调一次 sampler”，而是：

- 先 compact 出所有非 greedy 行；
- 再做一次 batched sampling pass。

这保证了不同采样参数的行，仍然可以合并进一个统一调用。

## 为什么这个 crate 还负责 logprob 计算

采样只是“选 token”，但很多 OpenAI/vLLM 接口还要求：

- sampled token 的 logprob
- top-k 候选 token 的 logprobs

这部分由 `token_logprob_from_row(...)` 负责。

它的输入是一整行 full-vocab logits，输出是：

- `TokenLogprob { logprob, top_logprobs }`

这部分看似简单，但这个 crate 做了两个重要统一：

### 1. 用 f64 累加 exp

log-softmax 里的分母是全 vocab 的 exp 求和。对 100k+ vocab 来说，如果全程用 f32：

- 数值精度会明显变差。

所以这里会：

- 先找 max
- 再用 `f64` 累加 exp
- 最后回到 `f32`

这是一种更稳的 host 侧 logprob 计算方式。

### 2. top-k tie-breaking 统一

不同模型自己写 top-k 逻辑时，tie-breaking 很容易不一致。

这里统一采用：

- 高 logprob 优先；
- tie 时按较小 token id 靠前。

这保证了：

- 输出顺序 deterministic；
- 不同模型不会因为同值 tie 的排序差异而让测试/行为漂移。

## 为什么它对 `row` 做成泛型

`token_logprob_from_row<T>` 的一个很实用的设计是：

- 行元素类型不强制写死成 `f32`
- 只要求 `T: Copy + Into<f32>`

这样不同模型就可以按自己已有的数据形态直接喂：

- Qwen：host 上常常已经是 `f32`
- Kimi：可能直接是 `bf16`

这样就避免了：

- 为了算 logprobs 先额外做一遍 widening copy

所以这个泛型不是抽象炫技，而是减少无意义数据搬运。

## 为什么 Kimi-K2 只部分复用这个 crate

Kimi-K2 的非 greedy 采样和 logprobs 计算都可以复用这里，但它的 greedy 路径不能完全复用。

原因在于：

- Kimi 的 greedy 不是在完整 vocab 上选 argmax；
- 它先做 shard-local argmax；
- 再把 top-1 logit 参与跨 rank 的 DP reduction；

而 `select_batch` 的假设是：

- 输入是一整行完整 vocab logits；
- 直接在完整 vocab 上做 next-token 选择。

所以 Kimi 只能复用：

- `gpu_sample_batch_into`
- `token_logprob_from_row`

但不能完全复用“whole-vocab greedy path”。

这恰恰说明这个 crate 的边界很清楚：**它只处理模型无关的采样逻辑，不把分布式 vocab-sharding 协议也塞进来。**

## 从一次 decode step 的角度看完整流程

把整个 crate 串起来，一次 decode step 大概是：

```text
1. executor 拿到一块 logits arena
2. 按每行的 SamplingParams 把 rows 划成 greedy / non-greedy
3. greedy rows 走一次 batched indexed argmax
4. non-greedy rows 走一次 batched sampling pass
5. 拼回 Vec<u32> 作为每行 next token
6. 若请求需要 logprobs，再对对应行调用 token_logprob_from_row(...)
```

如果只记一句话，可以把 `openinfer-sample` 理解成：

**“统一管理 next-token 选择这扇门：哪一行该 argmax，哪一行该 sampling，scratch 怎么复用，logprob 怎么统一算。”**

## 这个 crate 刻意没有做的事

为了看清边界，也值得明确它**没有**做什么：

- 不产生 logits；
- 不做模型前向；
- 不做分布式 vocab-shard greedy reduction；
- 不拥有底层 CUDA sampling kernel；
- 不管理请求级 scheduler。

这些分别属于模型 executor、分布式路径和 `openinfer-kernels`。`openinfer-sample` 当前专注的，就是共享采样策略层本身。

