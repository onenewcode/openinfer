# openinfer-comm 工作原理

`openinfer-comm` 是 openinfer 里面向**跨 rank 数据移动**的公开通信表面。你可以把它理解成：它不是直接提供 CUDA kernel、也不是直接提供 RDMA Verbs 实现，而是试图定义一层更窄、更稳定的 OpenInfer 侧接口，让上层调度器只依赖“我需要一次 EP all-to-all”这种语义，而不依赖底层到底是 `pplx-garden` 风格实现、RDMA、CUDA 还是别的硬件细节。

## 一句话先理解

这个 crate 做的是：

- 向上，给 OpenInfer 暴露一个较稳定的 EP backend 抽象；
- 向下，把真实硬件实现藏在 `backend::*` 和各个 wrapper crate 后面；
- 默认构建时故意保持“无硬件依赖、失败关闭（fail-closed）”。

所以它的重点不是“现在已经有很强的通信能力”，而是“先把公开契约收窄并固定住”。

## 它在系统里的位置

```text
OpenInfer 调度器 / 模型运行时
  -> openinfer-comm
      -> backend::rdma（未来/硬件 feature）
          -> openinfer-comm-* wrapper crates
              -> CUDA / GDRCopy / libibverbs / all-to-all kernels
```

这里有一个非常重要的设计选择：**OpenInfer 外部代码原则上只该看见 `openinfer-comm`，不该直接依赖底下那些 wrapper crate。**

## 它当前真正提供的东西

虽然 README 里反复强调它还是 skeleton，但从工程边界看，它已经明确了这些概念：

- `EpAllToAll`：一次 EP all-to-all 调用的抽象接口
- `EpBackend` / `EpBackendBuilder`：后端构建入口
- `EpTopology`：world size、rank、expert / hidden / token 这些拓扑信息
- `DispatchPlan` / `CombinePlan`：本次调用的路由描述
- `DispatchHandle` / `CombineHandle` / `Poll`：异步 in-flight 句柄与轮询面
- `Error` / `Result`：对外错误契约

这说明它的目标不是暴露底层所有能力，而是把 OpenInfer 真正需要的那组“最小通信语义”先定义清楚。

## 为什么它要 fail-closed

当前它故意让 `EpBackendBuilder::build()` 在默认模式下返回错误，而不是返回一个半成品对象，原因很关键：

- 不想让调用方拿到一个“方法里全是 `todo!()`”的 backend
- 不想让 skeleton 形态误导别人以为已经有可用实现
- 先锁定契约，再逐步接真后端

也就是说，这个 crate 当前最重要的成果不是“能跑”，而是“公开 API 的边界已经被认真设计过，并且不会假装自己已完成”。

## 为什么默认构建不碰硬件头文件

这是整个 `openinfer-comm` 体系的核心哲学之一：

- 默认 `cargo check -p openinfer-comm` 应该能在没有 CUDA / GDRCopy / RDMA Verbs 的机器上通过
- 真正的硬件依赖，放到 feature 后面
- 更底层的 wrapper crate 再去做 `*-sys` 绑定和硬件实现

这样做的好处是：

- 契约设计和 API 审查不被硬件环境绑死
- 工作区基础 CI 更容易跑
- 上层 OpenInfer 可以先依赖通信抽象，而不是马上被硬件细节污染

## 它和 `openinfer-comm/crates/*` 的关系

你可以把关系理解成两层：

- `openinfer-comm`：面向 OpenInfer 的正式公开表面
- `openinfer-comm/crates/*`：底层实现拼图（CUDA 包装、RDMA fabric、Torch 胶水、all-to-all kernel、Python bridge 等）

这些子 crate 很重要，但它们不是想给 OpenInfer 其他模块直接依赖的。它们更像 backend adapter 的零件箱。

## 最重要的心智模型

理解 `openinfer-comm` 时，最重要的是接受它当前的定位：

- 它现在首先是**契约 crate**
- 其次才会成长为可用 backend 的装配入口

也就是说，今天看它时，重点不是“功能多少”，而是“它把通信问题切出了怎样一条清晰的公开边界”。

下面保留原始英文 README，继续记录 skeleton 状态、公共类型表面和 feature 约束，方便对照实现细节。

# openinfer-comm

Skeleton comm-backend surface for **OpenInfer**: a narrow, hardware-free
trait that OpenInfer's request scheduler will use to drive cross-rank
data movement (EP all-to-all first; future data-movement surfaces later).

**Status: skeleton.** This crate currently exposes the *shape* of the
initial public API. The default-feature build compiles and the type
surface is reviewable, but there is **no usable backend yet** — see
[Status](#status) below. The crate is in the workspace so its contract
can be reviewed before the hardware adapter is wired in.

The crate is structured so that the default-feature build does not
require any hardware-class system header (CUDA SDK, GDRCopy, RDMA
Verbs). This lets OpenInfer's main CI lane `cargo check -p openinfer-comm`
on a barebones development machine. Hardware backends live behind
feature flags and only compile in when the matching feature is on.

## Status

This is a **skeleton PR**. Concretely:

- The default-feature build compiles and the public types exist.
- `EpBackendBuilder::build` is **fail-closed in both feature modes**:
  - default-off: returns `Error::BackendUnavailable` (no hardware
    backend feature active).
  - `hw-rdma`: returns `Error::Unimplemented` (the `RdmaBackend`
    adapter exists as a private type but its wiring is not landed
    yet).
- As a result, no caller can obtain an `EpBackend` whose trait methods
  would panic. The `EpAllToAll` trait methods on `RdmaBackend` are
  `todo!()` placeholders that are not reachable through the public
  builder.
- Trait method signatures, plan / handle / buffer field shape, and
  `EpTopology` field set are marked `#[non_exhaustive]` and may evolve
  in follow-up PRs. Treat this crate as a contract *shape under
  review*, not as a frozen API.

The wiring PR that turns this into a working backend will:

1. Remove the `Error::Unimplemented` branch in `EpBackendBuilder::build`
   and replace it with real `RdmaBackend` construction.
2. Replace the `todo!()` bodies on `RdmaBackend`'s `EpAllToAll` impl
   with translation onto the wrapper-crate API.
3. Add an integration test that exercises the public trait surface.

## Public type surface (skeleton shape)

| Type                                  | Purpose                                                      |
| ------------------------------------- | ------------------------------------------------------------ |
| `EpAllToAll` (trait, object-safe)     | Per-call dispatch / combine / poll / release entry points.   |
| `EpBackend`, `EpBackendBuilder`       | Builder for the future active backend (fail-closed today).   |
| `EpTopology`                          | World size + rank + expert/dim/token sizing.                 |
| `DispatchPlan`, `CombinePlan`         | Per-call routing descriptors.                                |
| `SendBuf<'a>`, `RecvBuf<'a>`          | Opaque views over caller-owned device buffers.               |
| `DispatchHandle`, `CombineHandle`     | In-flight op tokens; finalized through `EpAllToAll::poll`.   |
| `AnyHandle`, `Poll`                   | Polling surface.                                             |
| `Error`, `Result`                     | Public error type; backend errors erased via `Box<dyn ...>`. |

All non-exhaustive types are marked `#[non_exhaustive]`; adding fields
is a non-breaking change. Method signatures and field sets are
explicitly subject to revision in the wiring PR.

## Feature flags

- `default = []` — pure-Rust surface. Pulls only `thiserror`. No
  wrapper-crate dependency, no `*-sys` link probe, no CUDA / GDRCopy /
  Verbs header lookup.
- `hw-rdma` — compiles the `crate::backend::rdma` module. Enabling
  `hw-rdma` **transitively activates the CUDA subsystem** on the
  underlying wrapper crates (`cuda-lib`, `torch-lib`, `a2a-kernels`)
  because the EP all-to-all path needs both CUDA kernels and RDMA
  Verbs. There is no `hw-rdma`-only path that omits CUDA. In this
  skeleton PR, even with `hw-rdma` on, `EpBackendBuilder::build` still
  returns `Error::Unimplemented`; the feature exists so the wrapper
  crates' build chain is exercised, not so a usable backend is
  produced.

The `HW_RDMA_ENABLED` constant is a diagnostic / build-system signal,
not part of the stable API. Code that needs to react to backend
availability should call `EpBackendBuilder::build` and dispatch on the
returned `Result`.

## Public-surface invariants for this skeleton

These are the properties the skeleton aims to preserve as the contract
matures. They are written in to constrain follow-up PRs:

1. **No wrapper-crate types in the default surface.** The `EpAllToAll`
   trait and the plan / handle / buffer / error / builder types must
   not reference any type from a wrapper crate (`p2p-all-to-all`,
   `fabric-lib`, `cuda-lib`, `torch-lib`, `a2a-kernels`, `cuda-sys`,
   `cudart-sys`, `gdrapi-sys`, `libibverbs-sys`). Backend-specific
   errors are erased through `Error::Backend { source: Box<dyn
   std::error::Error + Send + Sync> }`.
2. **Backends are not re-exported.** Implementation modules live under
   `crate::backend::*` and are only compiled when the matching feature
   is on. They must not be re-exported through this crate's public
   namespace; the only way to obtain a backend is through
   `EpBackendBuilder::build`.
3. **Diagnostic markers are not stable API.** `HW_RDMA_ENABLED` and any
   similar `*_ENABLED` constants exist for runtime / build-system
   introspection only.
4. **Fail-closed before wiring.** While the public surface is in
   skeleton form, `EpBackendBuilder::build` returns `Err` in all
   feature modes; no caller can obtain a backend whose trait methods
   would panic.

## Wrapper crates are *not* OpenInfer's public API

This repository contains several upstream-derived wrapper crates
(`p2p-all-to-all`, `fabric-lib`, `cuda-lib`, `torch-lib`, `a2a-kernels`,
`python-ext`, plus their `*-sys` siblings). They are hardware
implementation packages reached only through `openinfer-comm`
adapters. Their names, types, and feature flags are **not** part of
OpenInfer's API contract and may evolve as the upstream and adapter
layers change.

OpenInfer code (and any code outside this crate) should depend on
`openinfer-comm` only. Direct use of wrapper-crate types from outside
this crate's `backend::*` adapter modules is unsupported.

## Usage sketch

Today, every call to `build` returns an error. The fail-closed result is
intentional — callers must dispatch on the `Result`. **Do not call
`build().unwrap()`** in this PR or in any caller during the skeleton
window:

```rust
use openinfer_comm::{EpBackendBuilder, EpTopology};

// `EpTopology` is `#[non_exhaustive]`; outside this crate the constructor
// is the only stable way to obtain one.
let topology = EpTopology::new(
    /* world_size     */ 1,
    /* rank           */ 0,
    /* num_experts    */ 0,
    /* hidden_dim     */ 0,
    /* max_num_tokens */ 0,
);

match EpBackendBuilder::new().topology(topology).build() {
    Ok(_backend) => {
        // Reached once the wiring PR lands. Currently unreachable;
        // do NOT unwrap build() here — the fail-closed Err is intended.
    }
    Err(e) => {
        // BackendUnavailable on default-off, Unimplemented on hw-rdma
        // until the wiring PR lands.
        eprintln!("EP backend not yet available: {e}");
    }
}
```

## License & provenance

See the top-level `LICENSE` and `NOTICE.md`. The hardware backend is
being adapted from upstream `pplx-garden`; this crate adds the
OpenInfer-facing public surface skeleton and the feature-gating that
keeps the default build hardware-free.
