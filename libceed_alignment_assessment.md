# Reed 与 libCEED 对齐程度评估（基于当前代码）

本文档在 **`design_mapping.md` 概念对照** 之外，单独给出截至当前仓库实现的 **对齐度评估**，便于迁移 libCEED 示例、做路线图或对外说明。评估依据为 `reed`、`reed-core`、`reed-cpu`、`reed-wgpu` 源码与集成测试；**不绑定** libCEED 某一具体发行版号（上游 API 以 `Ceed*` / `ceed-gallery-list.h` 为准）。

---

## 1. 结论摘要

| 层级 | 与 libCEED 的大致对齐度 | 一句话 |
|------|-------------------------|--------|
| 对象模型与资源路由 | **高** | `Reed<T>` / `Backend`、`/cpu/self`、`/gpu/wgpu`（可选 feature）、CUDA/HIP 占位与 `design_mapping` 一致。 |
| Vector / Restriction（CPU + WGPU） | **高** | `VectorTrait`、`ElemRestrictionTrait`、offset / strided、`elem_restriction_at_points`、`*_ceed_int_*` 工厂与 libCEED 语义一致并有测试；WGPU 路径覆盖 f32 offset/strided gather/scatter 及 f64 gather。 |
| Basis（CPU） | **高** | 张量 H1 Lagrange（P≥2, dim=1..3, Gauss/GaussLobatto）+ simplex H1（P1–P3, Line/Triangle/Tet）；`Interp/Grad/Div/Curl/Weight` 全模式及转置；Div/Curl 为 **H1 向量上的微分**，非 Nédélec/RT 独立元。 |
| Basis（WGPU） | **中高** | `f32` 上 `Interp/Grad/Div/Curl/Weight` 全模式及转置均有 GPU 路径（含 `WgpuSimplexBasis`）；标量 `Weight` 转置复用 `Interpᵀ` 核；`f64` / 无 runtime 回落 CPU；WASM 上 `WgpuBasis` 无 `BasisTrait` impl，张量基创建回落 `CpuBackend`。 |
| QFunction（CPU gallery） | **高** | 31 个 interior 命名 gallery + `q_function_interior` / `q_function_exterior` + `QFunctionCategory` 元数据；涵盖 Mass/Poisson/Identity/Scale/Vector*/VecDot 的 Build 与 Apply；`QFunctionContext` 命名字段 LE 读写；多个 gallery 核实现 `apply_operator_transpose`（伴随）。 |
| QFunction（WGPU device） | **中高** | `try_device_q_function_by_name` 提供 17+ 种 `f32` WGSL 设备端 gallery QFunction，含 Mass/Poisson/Identity/Scale/Vector*/VecDot 的 Build/Apply 及 **transpose/adjoint pipeline**；`f64` 回落 CPU gallery；`WgpuBackend` 在非 WASM 上实现 `Backend::try_device_q_function_by_name`。 |
| Operator（CPU） | **高** | `OperatorBuilder` + `CpuOperator`：`apply/apply_add`、`apply_with_transpose(Adjoint)`（离散伴随，含 `apply_operator_transpose_with_primal` 扩展）、多 active 场 `apply_field_buffers*`、被动/`None` 槽；稠密 `linear_assemble*` / `linear_assemble_add`、自建 CSR `linear_assemble_csr_matrix*`、`CeedMatrix` 句柄 set/add 装配、FDM 替代路径（小 `n` 稠密逆 + Jacobi 近似逆）在单全局 active 空间上可用；依赖 `apply_operator_transpose` 与标量 `Weight` 约定。 |
| Operator（WGPU hybrid） | **中** | 整条算子仍在 CPU 编排；但可通过 `CpuOperator` + WGPU restriction + WGPU basis + **WGPU device QFunction** 构成混合路径，`apply` 及 `apply_with_transpose` 已有与全 CPU 路径的数值一致性集成测。 |
| CompositeOperator | **高** | 加法型组合与对角装配与 libCEED 子集一致；支持 `apply_field_buffers*`（含 `Adjoint`）对子算子路径求和；`CompositeOperatorBorrowed` 支持借用子算子。 |
| WASM | **中偏低** | `OperatorTrait` / `QFunctionTrait` 等在 `wasm32` 上裁剪（无 `Send + Sync`）；WGPU basis 在 WASM 上无 `BasisTrait` 实现（张量基回落 CPU）；WGPU restriction/vector 在 WASM 上仍可用。 |

**总体**：在 **主机 CPU 离散算子迁移** 方向上，Reed 已覆盖 libCEED 教学/示例中最常见的 **restriction + tensor/simplex H1 basis + interior gallery QFunction + operator apply / transpose（伴随约束内）** 路径，并具备 **Reed 侧 CSR 装配与 SpMV**、**`CeedMatrix` 句柄语义（dense/CSR set+add）** 与 **FDM 替代路径（稠密逆 + Jacobi）**；**设备端完整算子**、**面元专用 quadrature / exterior 全语义**、**与 libCEED 托管式 `CeedMatrix` 的 1:1 后端生命周期**、**libCEED 原生张量 FDM**、**libCEED 全部 gallery / OCCA 后端** 等仍为明显缺口。

---

## 1.1 CPU 对齐发布清单（快速判断）

以下清单仅面向 **`/cpu/self`** 路径，帮助快速回答"CPU 后端能否作为 libCEED 迁移目标发布"：

| 项目 | 状态 | 说明 |
|---|---|---|
| 向量 / restriction / basis 基础路径 | ✅ **已对齐（子集）** | `VectorTrait`、`ElemRestrictionTrait`（offset/strided/at-points）、Lagrange/Simplex basis（P1–P3）与常见 libCEED 示例语义对齐。 |
| QFunction（interior + context） | ✅ **已对齐（子集）** | `QFunctionField`、`apply(ctx,...)`、gallery 名称解析（31 个 interior 名称）、context 字节布局与 LE 读写路径稳定。 |
| Operator 前向/累加 apply | ✅ **已对齐（子集）** | `apply` / `apply_add`、`check_ready`、多 active 场 `apply_field_buffers*`。 |
| Operator 离散伴随（Adjoint） | ⚠️ **条件对齐** | 依赖 `QFunctionTrait::apply_operator_transpose`（及可选 `apply_operator_transpose_with_primal`）；向量 `Weight` 等高级情形仍有边界。 |
| 线装配（Diagonal / AddDiagonal） | ✅ **已对齐（子集）** | `linear_assemble_diagonal` / `linear_assemble_add_diagonal` 已稳定。 |
| 线装配（Dense / CSR，set + add） | ✅ **已对齐（子集）** | `linear_assemble_symbolic` / `linear_assemble` / `linear_assemble_add` 与 `linear_assemble_csr_matrix` / `_add`；测试覆盖 Mass/Poisson。 |
| `CeedMatrix` 句柄装配 | ✅ **已对齐（子集）** | `linear_assemble_ceed_matrix` / `linear_assemble_add_ceed_matrix` 支持 dense 与 CSR 两种存储。 |
| FDM inverse 形状 (`CeedOperatorCreateFDMElementInverse`) | ⚠️ **API 对齐，实现替代** | 采用小 `n` 全局稠密逆（`CpuFdmDenseInverseOperator`，`n ≤ 256`）+ Jacobi 近似逆（`CpuFdmJacobiInverseOperator`），非 libCEED 原生 tensor-FDM。 |
| `CeedMatrix` 对象级 1:1 语义 | ⚠️ **部分对齐** | 已有 `CeedMatrix` 句柄（dense/CSR + symbolic/numeric 状态）与 `CpuOperator` 的 set/add 装配；仍非 libCEED 后端托管矩阵对象的完整 1:1 模型。 |
| 复合算子（加法） | ✅ **已对齐（子集）** | `CompositeOperator*` 的 apply/diag-add 行为稳定；矩阵装配/FDM 在复合上显式 `Err`。 |
| `CompositeOperatorBorrowed` | ✅ **已对齐（子集）** | 借用子算子的复合路径，行为与 `CompositeOperator` 对称。 |

**发布建议（CPU）**：若目标是"迁移主机离散算子与常见示例工作流"，可按 **高对齐** 口径发布；若目标是"libCEED 全 API 逐项等价"，则仍需补 `CeedMatrix` 后端托管语义与原生 tensor-FDM。

---

## 2. 分维度说明

### 2.1 顶层与后端

- **对齐**：资源字符串（`/cpu/self`、`/cpu/self/ref`、`/gpu/wgpu`、`/gpu/wgpu/ref`）、`Reed::init`、`Backend` trait（Send + Sync 在非 WASM 目标，WASM 上放宽）、`reed_core::Backend` 与 `reed_wgpu::Backend` 分层。
- **部分对齐**：`/gpu/cuda`、`/gpu/hip` 可解析与报告（`ReedBackendRequest` 枚举），**无执行实现**（占位，返回 `BackendNotSupported`）。
- **差异**：无 libCEED 式 OCCA / 多 vendor 运行时枚举；后端矩阵由 Rust feature（`wgpu-backend`、`parallel`）+ 资源串表达。
- **WASM 支持**：`reed-wasm-runner` crate 提供 JS 入口点（`init_wgpu`、`run_example`）；`Backend` trait 在 WASM 上放宽 `Send + Sync` 以适配 `wgpu::Device` 非线程安全特性。

### 2.2 类型与枚举

- **对齐**：`EvalMode`（含 `Weight`）、`QuadMode`、`TransposeMode`、`NormType`、`ElemTopology`、`MemType` 等与 libCEED 概念对应；`ElemTopology` 包含 Pyramid/Prism 等占位枚举值。
- **差异**：Reed 明确提供 `CeedInt`（`i32`）与 `CeedSize`（`usize`）双别名；与 libCEED 单一整型策略仍非 1:1，但通过 `*_ceed_int_*` 工厂覆盖常见绑定桥接；`QFunctionCategory`（`Interior` / `Exterior`）为 Reed 扩展的元数据分类。

### 2.3 `ElemRestriction`

- **对齐（CPU）**：`NoTranspose` / `Transpose` 与 gather/scatter 语义；strided；`elem_restriction_at_points` 与 offset 实现一致（集成测覆盖）；`assembled_csr_pattern` 实现于 `CpuElemRestriction`（offset layout 且 `ncomp==1` 时）。
- **对齐（WGPU）**：offset / strided 的 GPU gather/scatter（f32）；f64 offset/strided gather 有 GPU 路径（WGSL bitcast）；f64 scatter 仍走 CPU 回落。
- **风险点**：与 libCEED 示例混用 `int64` 缓冲区时，仍需通过 **`elem_restriction_ceed_int_*`** 等工厂显式收窄到 `CeedInt`。

### 2.4 `Basis`

**CPU（`LagrangeBasis` / `SimplexBasis`）**

- **高对齐**：`Interp` / `Grad` 及转置；Gauss / GaussLobatto；`q_weights` / `q_ref`。
- **高对齐**：`Div` / `Curl`（含转置与离散伴随恒等式类测试）；语义为 **H1 向量笛卡尔分量上的算子**，**非** libCEED 中独立 H(div)/H(curl) 元的 Nédélec/RT 基。
- **高对齐**：**标量** `EvalMode::Weight` 的 **转置** 与 `Interp` 转置同构（basis + 算子伴随路径已接）；**向量 `Weight`** 仍不支持。
- **高对齐**：Simplex 线/三角/四面体 **P1–P3**（含预计算的 P3 系数数据）；张量积 Lagrange 覆盖 dim=1..3、P≥2（覆盖 libCEED 常见示例维数组合）。
- `#[cfg(feature = "parallel")]` 下的 rayon 并行 evaluator（阈值 `PAR_MIN_ELEMS_PER_TASK = 128`）。

**WGPU（`WgpuBasis` / `WgpuSimplexBasis`）**

- **中高**：`f32` 上 `Interp/Grad/Div/Curl/Weight` 全模式（前向 + 转置）与 CPU 对齐，有 WGSL compute pipeline 及与 CPU 基础对照的集成测（20+ 项）；**标量 `Weight`+transpose** 走 `Interpᵀ` 核（GPU 路径）；否则回落 `LagrangeBasis` CPU。
- **中高**：`WgpuSimplexBasis` 支持 Tri/Tet P1–P2 的 Interp/Grad/Div/Curl/Weight 全模式 GPU 求值（simplex grad 矩阵经格式重排后复用 `WgpuBasis` 的 WGSL kernel）。
- **限制**：`f64` 所有模式回落 CPU；`wasm32` 上 **`BasisTrait` 未为 `WgpuBasis` / `WgpuSimplexBasis` 实现**（`#[cfg(not(target_arch = "wasm32"))]`），张量/单形基创建由 `WgpuBackend` 委托 `CpuBackend` 工厂。

### 2.5 `QFunction`

**CPU gallery**

- **高对齐（能力子集）**：`QFunctionField`、`apply(ctx,…)`、`context_byte_len`；`QFunctionContext` 命名字段与 LE 读写；gallery **`q_function_by_name`** 与 `QFUNCTION_INTERIOR_GALLERY_NAMES`（31 项）同步自检。
- **高对齐**：Gallery 核覆盖 Mass1/2/3DBuild、MassApply、Poisson1/2/3DBuild、Poisson1/2/3DApply、Identity、IdentityScalar、Scale、ScaleScalar、Vec2Dot、Vec3Dot、Vector2/3MassApply、Vector2/3Poisson1/2/3DApply，以及 AtPoints 别名（MassApplyAtPoints、ScaleAtPoints、IdentityAtPoints、Poisson2DApplyAtPoints、MassApplyInterpTimesWeightAtPoints）。
- **中（B）**：**interior / exterior 闭包**：`q_function_interior` / **`q_function_exterior`** + **`QFunctionTrait::q_function_category`**（**`QFunctionCategory`**）；与 libCEED interior/exterior **注册分类** 对齐；**执行路径相同**，无独立面 quadrature 或句柄类型。
- **中**：命名 gallery 均为 **interior** 语义；**无** `CeedQFunctionCreateInteriorByName` 的 C 级独立句柄。
- **中**：`ClosureQFunction` **默认无** `apply_operator_transpose` → 用于 **算子 `Adjoint`** 时需 gallery 或自研 `QFunctionTrait`。
- **伴随之持**：`MassApply`、`Poisson2DApply` 等多个 gallery 核实现 `apply_operator_transpose`（含 `Vec2Dot`/`Vec3Dot` 的 transpose-accumulate）。
- **Reed 扩展**：`MassApplyInterpTimesWeight`（及 `MassApplyInterpTimesWeightAtPoints`）用于 **被动 `Weight` 槽** 与伴随测试；非 libCEED 注册名。

**WGPU device QFunction**

- **中高**：`try_device_q_function_by_name` 在 `WgpuBackend`（非 WASM）上实现，调用 `try_create_device_q_function_f32` 返回 GPU 实现的 `Box<dyn QFunctionTrait<f32>>`；覆盖 17+ gallery 名称（MassApply/1/2/3DBuild、Poisson1/2/3DApply/Build、Identity/Scalar、Scale/Scalar、Vector2/3 MassApply、Vector2/3 Poisson1/2/3DApply、Vec2Dot/Vec3Dot、MassApplyInterpTimesWeight），含对应的 **transpose/adjoint** WGSL compute pipeline。
- **限制**：仅 `f32`（运行时 `TypeId` 检查）；`f64` / 未识别名称返回 `None`（调用侧回落 CPU gallery）；device QFunction 由用户手动选入算子（`OperatorBuilder` 尚不自动从 backend 获取 device QFunction）。
- **与 CpuOperator 集成**：`CpuOperator` + WGPU restriction + WGPU basis + WGPU device QFunction 的混合路径已有 `apply` 与 `apply_with_transpose` 集成测数值验证。

**QFunctionContext 设备同步**

- `GpuRuntime` 提供 `sync_qfunction_context_to_buffer` / `write_qfunction_context_to_buffer` 方法将 host `QFunctionContext` 字节写入 GPU buffer。

### 2.6 `Operator` / `CompositeOperator`

**`CpuOperator`（`OperatorBuilder`）**

- **高对齐（子集）**：单 active 输入 + 单 active 输出下的 `apply` / `apply_add`；`check_ready`；非对称 build 的 `active_input_global_len` / `active_output_global_len` 与文档说明。
- **高对齐（子集）**：**离散伴随** — `apply_with_transpose(Adjoint)` 与 `apply_field_buffers_with_transpose(Adjoint)`，在 **qfunction 实现 `apply_operator_transpose`**（或覆写 `apply_operator_transpose_with_primal`）且满足 **单缓冲或命名场映射**、**basis/Weight 约定** 时工作；伴随执行缓存前向 qp 输入 (`last_forward_q_inputs`) 供 `apply_operator_transpose_with_primal` 使用。
- **明确约束（相对 libCEED 一般性）**：
  - 非线性或 **依赖 active 场前向值** 的 qp 核伴随 **不在 v1 范围**（与 `design_mapping` 一致）。
  - **向量场 `EvalMode::Weight`** 在算子伴随中仍不支持。
  - `CompositeOperator*` 已支持 `apply_field_buffers*`（含 `Adjoint`）对子算子路径求和；若子算子需要命名缓冲，单缓冲 `apply*` 路径会返回提示错误并引导改用命名缓冲接口。
- **已修正行为（与 libCEED 文档习惯对齐）**：命名 **`Adjoint` 的 `outputs` 仅需各 active 输入场**；**被动 / `None` 输入槽** 不要求出现在 `outputs`（实现与 `reed_core::OperatorTrait` 文档一致）。
- **Reed 扩展（内存管理）**：**`CpuOperator::dense_linear_assembly_n`** / **`dense_linear_assembly_numeric_ready`** 查询稠密槽状态；**`clear_dense_linear_assembly`** 释放稠密装配槽（**不影响** `apply` / CSR 装配）。
- **部分对齐（稠密 + 自建 CSR + `CeedMatrix` 句柄 + FDM API）**：**`CpuOperator::linear_assemble_symbolic` / `linear_assemble`** 写入 **列主序稠密 `n×n`**（`Mutex` 缓冲，`n` 次 `apply`）；**`linear_assemble_add`** 在已有槽上 **累加列**；**非线性** `apply` 下 **不保证** 为全局 Jacobian。**`csr_sparsity_from_offset_restriction`** / **`assembled_csr_pattern`** / **`linear_assemble_csr_matrix`** / **`linear_assemble_csr_matrix_add`** / **`CsrMatrix::mul_vec`**：与 libCEED **稀疏拓扑 + 数值（含累加）** 概念对齐。并已提供 **`CeedMatrix` 句柄**（dense/CSR，symbolic/numeric 状态）与 **`linear_assemble_ceed_matrix` / `linear_assemble_add_ceed_matrix`**。**`linear_assemble_add_diagonal`**：对齐 **`CeedOperatorLinearAssembleAddDiagonal`**（`CpuOperator` / **`CompositeOperator*`** / **`CpuFdmDenseInverseOperator`**）。**`operator_create_fdm_element_inverse`**：小 `n`（`n ≤ FDM_DENSE_MAX_N = 256`）下 **`Ok(CpuFdmDenseInverseOperator)`**（创建时在本地缓冲按 `n` 次前向 `apply` 组装规范 Jacobian \(A\)，**不读取也不改写** 稠密槽；全局稠密 \(A^{-1}\)，**非** libCEED 张量 FDM）；另有 **`operator_create_fdm_element_inverse_jacobi`**（`CpuFdmJacobiInverseOperator`）提供结构化近似逆。**`operator_supports_assemble`**：`LinearCsrNumeric` 与稠密线装同为 **`active_global_dof_len` 有定义**（**set** 与 **add** 共用）；**复合算子**对 **`LinearCsrNumeric`** 与 **`FdmElementInverse`** 恒 **`false`**（对应 **`linear_assemble_csr_matrix`** / **`linear_assemble_csr_matrix_add`** / **`operator_create_fdm_element_inverse`** 为 **`Err`**）；`FdmElementInverse` 在 `n ≤ FDM_DENSE_MAX_N` 时于 **`CpuOperator`** 为 `true`。稠密 **`linear_assemble_symbolic` / `linear_assemble` / `linear_assemble_add`** 在复合上 **`Err`**。

**`CompositeOperator` / `CompositeOperatorBorrowed`**

- **高**：加法、`apply_add*`、`linear_assemble_diagonal`、`linear_assemble_add_diagonal`、单缓冲与命名缓冲 **`Adjoint` 为子算子伴随之和**；与 libCEED 组合模式的部分子集一致；`CompositeOperatorBorrowed` 支持对 `&dyn OperatorTrait<T>` 子算子的借用组合，行为与 `CompositeOperator` 对称。

### 2.7 WGPU 与「整条算子在设备上」

- **无 `WgpuOperator`**：算子编排（restriction → basis → QFunction → basisᵀ → restrictionᵀ）完全在 CPU 端 `CpuOperator` 中执行，无 `WgpuOperator` 类型。
- **混合算子路径（已测）**：`CpuOperator` + `/gpu/wgpu` 下 `WgpuVector` / `WgpuElemRestriction` / `WgpuBasis` / **WGPU device QFunction** 可在单个 `apply` 中混合 GPU 与 CPU 对象；已有与全 CPU 路径的 **`apply` 及 `apply_with_transpose` 数值一致性集成测**（`tests/integration.rs`：`test_wgpu_hybrid_mass_operator_apply_matches_cpu`、`test_wgpu_hybrid_mass_operator_transpose_matches_cpu`，feature `wgpu-backend`）。
- **WGPU 基础能力**：`GpuRuntime` 持有 ~40+ WGSL compute pipeline，覆盖：
  - Vector 操作（set_value、scale、axpy）
  - Restriction（offset/strided gather、scatter、f64 gather）
  - Basis（Interp/Grad/Div/Curl/Weight 前向与转置、post-processing）
  - QFunction（~20+ gallery 核的 Build/Apply/Transpose，含 pointwise_mul、dot、mass_build、poisson_build/apply、identity/scale copy/accumulate、mass_apply_qp 及其 transpose）
- **QP 核独立调度**：`GpuRuntime` 提供 `dispatch_mass_apply_qp_f32` / `dispatch_mass_apply_qp_transpose_accumulate_f32` 及 `*_host` 便捷方法（含 upload/readback）。
- **差距**：qp 数据与 `QFunctionContext` 的 **设备驻留**（非每步 upload）、与 restriction/basis 管线的 **device 端自动衔接**（无 host round-trip）；混合路径仍需 CPU 端的 gather/basisᵀ/scatter 编排。

### 2.8 WASM

- **Operator / QFunction / Backend**：`reed_core` 在 `wasm32` 上 **弱化** `Send + Sync` 等约束（全部 5 个核心 trait 拆分为两个 `#[cfg]` 版本）；与 native 非同一 trait 定义。
- **WGPU Basis**：见 §2.4，`WgpuBasis` / `WgpuSimplexBasis` 在 WASM 上 **无** `BasisTrait` impl；张量 H1 Lagrange 基在 WASM 上由 `WgpuBackend` 委托 `CpuBackend` 工厂创建 `LagrangeBasis`。
- **WGPU Vector / Restriction**：WASM 上仍可用（`WgpuVector`、`WgpuElemRestriction` 不依赖 `BasisTrait` 的 wasm cfg）。
- **对齐策略**：以 **主机 + 可选 wgpu** 为主；与 libCEED wasm 示例同构时需对照 **附录 A** 能力矩阵。

---

## 3. Gallery 名称覆盖（相对 `ceed-gallery-list.h`）

- **`QFUNCTION_LIBCEED_MAIN_GALLERY_NAMES`**：`crates/cpu/src/lib.rs` 中 **18** 个条目，与上游 `gallery/ceed-gallery-list.h` **顺序与名称** 对齐（Identity、Identity to scalar、Mass1/2/3DBuild、MassApply、Vector3MassApply、Poisson1/2/3DApply/Build、Vector3Poisson1/2/3DApply、Scale、Scale (scalar)）。
- **`QFUNCTION_INTERIOR_GALLERY_NAMES`**：**31** 个条目（18 个 libCEED main + Reed 扩展：`Vec2Dot`、`Vec3Dot`、`IdentityScalar`、`ScaleScalar`、`Vector2MassApply`、`Vector2Poisson1/2DApply`、`MassApplyInterpTimesWeight`，及 AtPoints 迁移别名 `MassApplyAtPoints`、`MassApplyInterpTimesWeightAtPoints`、`ScaleAtPoints`、`IdentityAtPoints`、`Poisson2DApplyAtPoints`）。
- **WGPU device QFunction 名称覆盖**：`try_create_device_q_function_f32` 覆盖 17+ 个 gallery 名称（所有 Build/Apply 核 + Identity/Scalar/Scale + Vector* + Vec*Dot + MassApplyInterpTimesWeight），与 CPU gallery 高度重叠；仅支持 `f32`。
- **对齐度**：与 libCEED **体积** interior 示例常用核 **高度重叠**；**不声称**与上游 header 中每一条注册名 1:1 完备（上游会随版本增减）。
- **迁移注意**（已在 `design_mapping` 强调）：
  - **`Vector3Poisson2DApply`**：Reed 侧 `qdata` **4 分量/点** 与部分 libCEED 注册描述 **3 分量** 的打包差异 — 迁移须核对布局。

---

## 4. 测试作为对齐证据（当前仓库）

下列测试类别支撑上表判断（非穷举）；`tests/integration.rs` 含 **83** 个测试函数，其中 **52** 个由 `#[cfg(feature = "wgpu-backend")]` 门控：

- **Restriction**：`elem_restriction` vs `elem_restriction_at_points`、strided、`*_ceed_int_*`、`csr_sparsity_from_offset_restriction` / `csr_sparsity_from_offset_lnodes`（`tests/integration.rs`）。
- **Basis（CPU）**：Lagrange / Simplex 恒等式、div/curl 伴随、`Weight` 转置与 `Interp` 转置一致性、Simplex P3 工厂验证（`reed-cpu` unit + `reed` integration）。
- **Basis（WGPU）**：20+ 项 CPU vs WGPU 对照测（Interp/Grad/Div/Curl/Weight 前向与转置，覆盖 Lagrange 1D/2D/3D 及 Simplex Tri/Tet P1/P2）；`reed-wgpu` 内 basis 单元测（`wgpu_weight_transpose_matches_interp_transpose_f32` 等）。
- **Operator**：Mass / Poisson 对称伴随、`MassApplyInterpTimesWeight` 单缓冲与 **命名缓冲 `Adjoint`**、多 active 场 `apply_field_buffers`、`CompositeOperator` / `CompositeOperatorBorrowed`；**稠密 `LinearAssemble*` / `linear_assemble_add`**、**CSR `linear_assemble_csr_matrix_add`**、**`dense_linear_assembly_n` / `dense_linear_assembly_numeric_ready` / `clear_dense_linear_assembly`**；CSR matvec vs `apply` 数值对照。
- **QFunction**：**exterior 闭包分类**（`test_qfunction_exterior_closure_reports_exterior_category`）；gallery **interior 分类**；**WGPU device QFunction** 单测（`reed-wgpu` 内 `qfunction_device.rs`，需 adapter 时跳过）。
- **WGPU hybrid**：`CpuOperator` + WGPU 对象混合路径的 `apply` / `apply_with_transpose` 数值一致性（`test_wgpu_hybrid_mass_operator_apply_matches_cpu`、`test_wgpu_hybrid_mass_operator_transpose_matches_cpu`）。
- **FDM**：`operator_create_fdm_element_inverse`（稠密逆）+ `operator_create_fdm_element_inverse_jacobi`（Jacobi 近似）；FDM 创建不修改 dense 槽的副作用校验。
- **CeedMatrix**：`CeedMatrix` dense col-major 与 CSR symbolic/numeric 状态、`linear_assemble_ceed_matrix` / `linear_assemble_add_ceed_matrix` 集成测。
- **Benchmarks**：`benches/cpu_backend.rs`、`benches/wgpu_backend.rs`、`benches/backend_compare.rs`（Criterion 框架，覆盖 vector/restriction/basis/operator）。

---

## 5. 对齐度分级（建议用法）

| 等级 | 含义 | 典型用途 |
|------|------|----------|
| **A** | 语义与路径与 libCEED 常见示例 **等价或可机械迁移** | 1D/2D/3D Poisson/Mass + tensor Lagrange + offset restriction；CPU gallery 核 MassApply/PoissonApply 等 |
| **B** | 功能有，但 **API 形状或类型细节不同** 或 **仅 CPU** 或 **需显式 opt-in** | 多向量 `apply_field_buffers`、`*_ceed_int_*`、闭包 QFunction、**稠密** `LinearAssemble*` / **`linear_assemble_add`**、**`OperatorTrait::linear_assemble_csr_matrix`** / **`linear_assemble_csr_matrix_add`**、**自建 CSR**、**WGPU device QFunction**（`try_device_q_function_by_name`，手动选入算子） |
| **C** | **部分** 与 libCEED 同名或同概念，**语义子集或扩展** | `Vector2*` gallery、Reed 扩展 `MassApplyInterpTimesWeight`、`QFunctionCategory::Exterior` 元数据（无独立面 quadrature） |
| **D** | **未实现** 或 **仅占位** | CUDA/HIP 执行、整条 Operator on GPU（无 `WgpuOperator`）、**libCEED 托管 `CeedMatrix` / 原生张量 FDM**、面元专用 exterior **全语义**（Reed 仅有 **元数据** `QFunctionCategory::Exterior`） |

可将迁移中的每个 libCEED 调用映射到上表某一格，再决定是否需要上层胶水代码。

---

## 6. 建议的后续对齐优先级（与 `design_mapping` §8.1 一致，略作压缩）

1. **WGPU 算子端到端**：实现 `WgpuOperator` 将 restriction → basis → QFunction → basisᵀ → restrictionᵀ 全链路驻留 GPU（最大缺口）。当前 device QFunction 能力已就绪但编排仍在 CPU。
2. **Device QFunction 自动集成**：`OperatorBuilder` / `CpuOperator` 自动从 `Backend::try_device_q_function_by_name` 获取 device QFunction 替代 CPU gallery 核，无需用户手动选入。
3. **整型与 `CeedInt` 策略**：是否在公共 API 层固定 `i32` 索引 + `usize` 尺寸文档化，减少与 libCEED C 示例的摩擦（**当前落盘约定见附录 B**）。
4. **WASM 能力矩阵**：为 `wasm32` 单独维护「支持 / 不支持」表，与 libCEED wasm 路径期望对齐（**见附录 A**）。
5. **Gallery 缺口**：按上游 `ceed-gallery-list.h` 做 diff，补缺或显式标注「不计划支持」。
6. **Exterior 全语义**：面元专用 quadrature、独立 face restriction 迭代、exterior gallery 名称注册。

---

## 7. 与 `design_mapping.md` 的关系

- **`design_mapping.md`**：长期维护的 **概念 ↔ API 映射表** 与约定说明。
- **本文档**：基于 **当前实现快照** 的 **对齐度与风险** 评估，可随大版本或里程碑更新；不必逐行重复映射表。
- **附录 A**：`wasm32` 目标下各 trait / 工厂与 WGPU 路径的 **能力矩阵**（与 `reed_core` / `reed_wgpu` 中 `cfg` 一致）。
- **附录 B**：与 libCEED `CeedInt` 互操作的 **整型桥接** 与 Reed 侧 **索引类型** 约定摘要。

若二者冲突，以 **源码与测试** 为准，并应回写修正 `design_mapping.md`。

---

## 附录 A. WASM（`target_arch = "wasm32"`）能力矩阵

依据 `reed_core` 与 `reed_wgpu` 源码中的 `#[cfg(target_arch = "wasm32")]` 拆分整理；**非 wasm** 列表示桌面/原生线程模型下的对照。

| 能力域 | 非 wasm（native） | wasm32 | 说明 |
|--------|-------------------|--------|------|
| `reed_core::Backend` | `Send + Sync` | 无 `Send + Sync` | 浏览器中 `wgpu::Device` 非 `Send + Sync`，故后端工厂 trait 在 wasm 上放宽边界。 |
| `VectorTrait` / `ElemRestrictionTrait` / `BasisTrait`（对象侧） | 多为 `Send + Sync` | 无 `Send + Sync` | 与 `reed_core` 中各 trait 的 wasm 变体一致；便于持有 `dyn` 后端对象。 |
| `QFunctionTrait` / `QFunctionClosure` | trait 与闭包要求 `Send + Sync` | 不要求 `Send + Sync` | `qfunction.rs`：wasm 上闭包可为单线程捕获。 |
| `OperatorTrait` | `Send + Sync` | 无 `Send + Sync` | `CpuOperator` 在 wasm 上仍实现完整算子 API（含 `apply_field_buffers` 与伴随相关路径），仅 trait 对象边界不同。 |
| `WgpuBackend::create_vector` | `WgpuVector` | `WgpuVector` | wasm 上仍走 WebGPU 缓冲路径。 |
| `WgpuBackend::create_elem_restriction` / `create_strided_elem_restriction` | `WgpuElemRestriction` | `WgpuElemRestriction` | GPU gather/scatter 路径在 wasm 上仍可用（受 WebGPU 实现与浏览器策略约束）。 |
| `WgpuBackend::create_basis_tensor_h1_lagrange` | `WgpuBasis`（`BasisTrait`） | **CPU `LagrangeBasis`（`CpuBackend` 工厂）** | `WgpuBasis` / `WgpuSimplexBasis` 的 `BasisTrait` 实现 **仅** `#[cfg(not(target_arch = "wasm32"))]`（`basis.rs`、`basis_simplex.rs`）；wasm 上 `WgpuBackend` 委托 cpu 后端创建张量 Lagrange / simplex 基。 |
| `WgpuBackend::try_device_q_function_by_name` | `Ok(Some(...))` for f32 gallery names | 未实现 device QFunction | WASM `reed_core::Backend` impl 中不含 `try_device_q_function_by_name` override，走默认 `None`。 |
| WGPU 相关集成测 / basis 单测 | 默认运行 | 多处 `#[cfg(not(target_arch = "wasm32"))]` 跳过 | `basis.rs`、`basis_simplex.rs`、`runtime.rs`、`qfunction_device.rs` 中测试。 |

**迁移提示**：在 wasm 上若资源选 `/gpu/wgpu`，张量 H1 Lagrange **基求值在 CPU**，restriction/vector 仍可走 WGPU；与 libCEED「全对象同后端」的想象不完全一致，上层若需统一性能模型应显式分支或固定用 CPU 后端。`reed-wasm-runner` crate 提供独立 WASM 入口，通过 `WgpuBackendWrap` 解决 orphan rule 问题。

---

## 附录 B. 整型与 `CeedInt` 桥接（当前约定）

| 概念 | Reed 运行时类型 | 与 libCEED 的桥 |
|------|-----------------|----------------|
| 全局长度、单元数、`elemsize`、`ncomp`、`lsize`、多项式阶等 **尺寸** | `CeedSize`（`usize`） | C 示例里常为 `CeedInt`；从 Rust 调用 Reed 时使用 `CeedSize` / `usize`。 |
| restriction **偏移**、**strided 三整数** | `&[CeedInt]` / `[CeedInt; 3]`（`CeedInt = i32`） | 与 32 位索引的 GPU / WGSL 路径一致；超大网格需自行确认不溢出 `CeedInt`。 |
| 自 **i64 / `int64_t` 绑定** 迁入 | `Reed::elem_restriction_ceed_int_offsets`、`elem_restriction_at_points_ceed_int_offsets`、`strided_elem_restriction_ceed_int_strides` | `reed.rs` 内 `ceed_int_*_to_i32`：每项必须落入 `i32`，否则 `InvalidArgument`。 |

**未决（与 §6.3 一致）**：是否在文档层统一写死「Reed 公共 API 以 `CeedSize` + `CeedInt` 为规范，与 libCEED 64 位 `CeedInt` 的差异由上述 `*_ceed_int_*` 入口吸收」，并在更多工厂上对称提供 `*_ceed_int_*` 变体，仍属产品化决策。

---

## 8. 修订历史

| 日期 | 说明 |
|------|------|
| 2026-04-21 | 首版：综合当前 `reed` 工作区算子伴随、`Weight`、WGPU basis、命名缓冲 `Adjoint` 与 gallery 状态撰写。 |
| 2026-04-21 | 增补附录 A（WASM 能力矩阵）、附录 B（`CeedInt` / 尺寸约定）；§6 与 §7 交叉引用。 |
| 2026-04-21 | §2.7：记录 WGPU 混合 `CpuOperator` 前向与 CPU 栈对齐的集成测；`design_mapping` §4.5 同步。 |
| 2026-04-21 | 混合算子：增补 `apply_with_transpose`（Forward / Adjoint）与 CPU 交叉验证的集成测；`design_mapping` §4.5 同步。 |
| 2026-04-21 | WGPU：`GpuRuntime` 增加 `MassApply` 标量 `f32` qp 前向/转置 compute 与单测；§2.7、`design_mapping` §8 / §8.1 同步。 |
| 2026-04-21 | WGPU：补充 `mass_apply_qp_*_host` 主机往返 API、集成测；文档同步。 |
| 2026-04-21 | CPU：`QFUNCTION_LIBCEED_MAIN_GALLERY_NAMES` 对齐 libCEED main `ceed-gallery-list.h`；`IdentityScalar`/`ScaleScalar` 别名；集成测与 `design_mapping` §5 / §8 同步。 |
| 2026-04-21 | QFunction：`QFunctionCategory` / `q_function_exterior`；Operator：`OperatorAssembleKind` 与 `LinearAssemble*` / FDM trait 占位；`design_mapping` §4.4–4.5、集成测与 §2.5–2.6、分级表 **D** 行更新。 |
| 2026-04-21 → 2026-04-21 | 36 条修订：CSR 装配、FDM 逆、`CeedMatrix` 句柄、稠密装配槽生命周期、复合算子行为等（详见正文修订历史条目）。 |
| 2026-05-01 | **全文重写**：基于当前代码快照全面更新。关键修订：(1) WGPU QFunction 从「仍在 CPU」升级为 **中高**，涵盖 17+ device QFunction + transpose pipeline；(2) WGPU Basis 从「中」升级为 **中高**，增补 `WgpuSimplexBasis`；(3) 引入 **WGPU hybrid operator** 行（混合路径数值一致性已测）；(4) Gallery 名称计数更新为 18 + 31；(5) 集成测试总量更新为 83（52 WGPU-gated）；(6) 附录 A 增补 `try_device_q_function_by_name` wasm 行为；(7) `CompositeOperatorBorrowed` 首次入表。 |
