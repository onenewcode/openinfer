# openinfer-cupti 工作原理

`openinfer-cupti` 是一个很薄的 Rust <-> CUPTI 桥接层。它不自己做 profiler UI，也不自己维护一整套 tracing 系统；它的职责只有一个：把“一段 GPU 工作负载在某个 CUDA context 下跑一次，并把指定 CUPTI metrics 读回来”这件事，包装成 Rust 代码能直接调用的函数。

## 一句话先理解

可以把它理解成：

- Rust 侧知道“我想测哪一段工作负载”；
- C/C++ 侧更适合直接对接 CUPTI；
- `openinfer-cupti` 把两边接起来，让 Rust 可以安全地把 prepare/launch 回调交给 CUPTI 侧执行。

它不是通用 profiling 框架，而是“在 OpenInfer 里按需测一个 range 的计数器读数”的最小封装。

## 它在系统里的位置

它的调用形态大致是：

```text
Rust bench / report 代码
  -> profile_range_with_prepare(...)
      -> FFI 调 C 层 openinfer_cupti_profile_range(...)
          -> CUPTI 配置 range profiler
          -> 回调 Rust 提供的 prepare / launch
          -> 读出 metric values
  -> Rust 侧拿回 Vec<f64>
```

所以它站在：

- 上层：kernel report、性能分析工具
- 下层：CUPTI C API

之间。

## 入口函数是怎么工作的

核心入口是 `profile_range_with_prepare(...)`。

它接收这些关键信息：

- `context`：当前 CUDA `CUcontext`
- `device_index`：当前设备编号
- `range_name`：这段 profile 的名字
- `metric_names`：要采哪些 CUPTI 指标
- `prepare`：可选预处理回调
- `launch`：真正发起 GPU 工作负载的回调

这背后的执行模型是：

1. Rust 先把字符串参数转成 `CString`；
2. 把 metric 名字数组整理成 C 兼容指针；
3. 构造一个 `CallbackState`，里面保存 Rust 闭包和错误槽位；
4. 把 `prepare_trampoline` / `launch_trampoline` 这样的 C ABI trampoline 函数传给底层 FFI；
5. 底层 C/C++ 代码在合适时机回调这些 trampoline；
6. trampoline 再回到原本的 Rust 闭包；
7. 执行成功则返回 metric 数组，失败则把 FFI 错误和回调错误拼成 Rust 错误对象。

## 为什么需要 trampoline

Rust 闭包不能直接当成 C 函数指针传给 CUPTI，所以这里用了经典 FFI 技巧：

- 真正传给 C 的，是 `unsafe extern "C"` trampoline；
- trampoline 收到一个 `userdata` 指针；
- 再把这个指针还原成 `CallbackState`；
- 最后间接调用 Rust 闭包。

这样就能把 Rust 世界里的：

- 准备输入
- 发起 kernel
- 记录错误

嵌进 C 风格回调模型里。

## `prepare` 和 `launch` 为什么分开

这个分层很重要。

- `prepare` 用来做每次采样前需要的准备动作，例如重置缓冲区、拷贝输入、清理状态；
- `launch` 则只负责真正发起要测的 GPU 工作负载。

这样做的好处是：

- 有些 profile 需要把“准备阶段”和“被测阶段”区分开；
- 可以更精确地控制被 CUPTI 计量的那一小段范围；
- 上层工具也更容易复用同一套 profiling 外壳。

## 错误处理是怎么做的

这个 crate 不假设错误只会来自一个地方。

错误可能来自：

- FFI / CUPTI 本身；
- Rust 回调闭包内部；
- 字符串参数转换。

因此它的策略是：

- 底层 FFI 提供一块 error buffer；
- Rust 回调错误则写进 `CallbackState.error`；
- 最后根据返回状态把两类错误合并成 `CuptiProfilerError`。

这样做的目的，是避免出现“CUPTI 失败了，但真正原因其实是 Rust 回调先报错”的诊断盲区。

## 它刻意保持很薄

`openinfer-cupti` 没有试图抽象出一个庞大的 profiler 框架，因为在 openinfer 里它的任务非常具体：

- 只在需要时包一个 range；
- 只把 metric 值读回来；
- 把复杂度留在调用方和底层 C 实现。

所以它不负责：

- 可视化；
- 长时间 trace 会话管理；
- 多 range 编排；
- metric 命名规范化；
- 模型级报告格式。

这些都属于更高层工具。

## 你应该怎么理解它的价值

如果说 `openinfer-kernels` 负责“把 kernel 跑起来”，
那么 `openinfer-cupti` 负责的是“在不离开 Rust 工作流的前提下，把 kernel 的硬件计数器读回来”。

它的价值不在于功能多，而在于：

- 把 CUPTI 这种偏底层、偏 C 的接口压缩成一个简单 Rust API；
- 让模型 crate / kernel report 工具可以直接把 profiling 集成进现有测试和分析流程。
