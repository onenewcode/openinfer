# openinfer-build 工作原理

`openinfer-build` 不是运行时 crate，而是给其他 crate 的 `build.rs` 用的一层“构建期基础设施”。它把“去哪里找 CUDA / 第三方头文件和库”“什么时候该触发重编译”“同一套探测逻辑如何在多个 crate 里复用”这些脏活收口到一个地方，避免 `openinfer-kernels`、`openinfer-cupti` 之类的 crate 各自复制一份 build script 工具函数。

## 一句话先理解

可以把它理解成：

- 运行时 crate 负责推理；
- `build.rs` 负责把 CUDA/C++ 代码编译进 Rust 包；
- `openinfer-build` 负责给这些 `build.rs` 提供“找环境、找头文件、找库、发 cargo 指令”的通用零件。

它解决的是“怎么把依赖找对、把编译配置说清楚”，不是“怎么执行推理”。

## 它在系统里的位置

当前最直接的使用者是：

- `openinfer-kernels/build.rs`：编译 CUDA、FlashInfer、Triton AOT 产物时，需要解析 CUDA toolkit 布局、拼 include/lib 搜索路径、控制 `rerun-if-changed`。
- `openinfer-cupti/build.rs`：需要定位 CUPTI / CUDA 相关头文件和库。

也就是说，依赖关系大致是：

```text
Cargo build
  -> crate/build.rs
      -> openinfer-build
           -> 解析环境变量 / 默认安装目录
           -> 输出 cargo:rerun-if-* / cargo:rustc-link-search=*
```

所以这个 crate 是“构建系统的共享工具箱”。

## 它主要做什么

### 1. 统一“找安装根目录”的逻辑

`find_package(...)` 做的事情很朴素，但非常重要：

- 先看指定环境变量，例如某个依赖是否通过 `FOO_HOME` 指向了自定义安装目录；
- 再按约定去默认目录里探测；
- 通过一组 `check_files` 判断这个目录到底是不是想要的包根目录；
- 找到后把“根目录 + 实际命中的文件”一起返回。

这样上层 build script 不需要自己写一堆“如果在 `/usr/local/...` 找不到，就再去 `targets/x86_64-linux/...` 里找”的分支。

### 2. 抽象 CUDA toolkit 的多种目录布局

`CudaToolkit` 是这个 crate 里最核心的抽象。

它把 CUDA toolkit 视为一组构建期资产：

- `nvcc` 在哪里；
- 哪些 `include` 目录真的存在；
- 哪些 `lib` / `lib64` 目录真的存在；
- 某个头文件最终应该从哪个 include 目录取。

这背后的原因是：CUDA 的安装形态并不统一。

- 传统安装常见于 `/usr/local/cuda`
- conda 环境可能只提供部分头文件/库
- NVIDIA HPC SDK 的 CUDA 和 math libraries 还会分布在兄弟目录
- `targets/<arch>/include` / `targets/<arch>/lib` 这种布局在某些机器上才存在

`CudaToolkit::discover()` / `CudaToolkit::from_root(...)` 的价值，就是把这些差异折叠成统一接口，让上层只需要问：

- 头文件目录在哪？
- link search path 应该怎么打？
- 要不要走 stub 库？

### 3. 输出 Cargo 能理解的增量构建信号

`emit_rerun_if_changed_files(...)` 会递归扫描源码目录，把匹配扩展名的文件都打印成：

```text
cargo:rerun-if-changed=...
```

这样一来：

- 改了 `.cu` / `.h` / Triton 生成脚本，Cargo 会重跑 build script；
- 没改相关文件，就不会无意义地重编译整个 CUDA 侧产物。

这件事看起来只是“体验优化”，但对 CUDA/Triton 工程非常关键，因为重编译成本高。

## 设计重点：它只做“发现与声明”，不做“编译策略”

这个 crate 的边界很明确：

- 它会告诉上层“CUDA 在哪”“哪些目录存在”“需要监听哪些文件变化”；
- 但它不决定某个 `.cu` 文件该用什么 nvcc 参数；
- 也不决定是否启用 Triton、TileLang、FlashInfer；
- 更不会拥有任何运行时 CUDA 上下文。

这些策略仍然属于调用它的具体 `build.rs`，尤其是 `openinfer-kernels/build.rs`。

换句话说：

- `openinfer-build` 负责“找路”；
- 各 crate 的 `build.rs` 负责“怎么走这条路”。

## 你读这个 crate 时应该抓住的主线

如果你以后要改构建逻辑，最重要的是理解这条分层：

1. `openinfer-build` 提供可复用的环境探测与 cargo 指令输出；
2. `openinfer-kernels/build.rs` 等上层脚本基于这些工具决定具体编译动作；
3. 最终产物才会被运行时 crate 通过 FFI 调用。

所以它不是推理引擎的一部分，而是“让推理引擎能被正确编译出来”的公共地基。
