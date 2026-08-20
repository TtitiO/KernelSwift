# Sinkhorn Operator Design

根据 `ascendc-operator-design` 规范，设计针对华为 Atlas A2 (Ascend 910B) 的纯 Vector 核心 (AIV) Sinkhorn 融合算子。

## 1. 算子需求分析与接口定义

*   **算子名称**: `sinkhorn`
*   **功能描述**: 对输入张量进行多次行列归一化迭代（双随机矩阵算法），将整个迭代过程融合在一个 Kernel 中，完全消除迭代过程中的 HBM（显存）读写开销。
*   **支持的数据类型**: `float32` (赛题指定)
*   **输入 Shape**: `[1, 1024, 4, 4]` (即 1024 个 `4x4` 矩阵)
*   **接口签名**:
    ```cpp
    at::Tensor sinkhorn_torch(const at::Tensor &x, int64_t repeat, double eps);
    ```
    同时注册 `TORCH_LIBRARY(sinkhorn_ops)` 和 `PYBIND11_MODULE(sinkhorn_ext)` 两个入口，
    Python 侧走 pybind 直调（见 §4 第 5 点）。

## 2. 计算逻辑设计

**实现路径选择**: **AscendC Kernel (纯 AIV 向量算子)**。
原因：此算子仅包含 element-wise 运算和沿维度的 reduction（求和、求最大值），不涉及高计算密度的矩阵乘法（无需 Cube 单元）。

**数学公式到 AscendC API 的映射**（最终实现，全向量化）:
整个计算全在 UB（片上缓存）中完成，循环 `repeat` 次。所有归约/广播均为全宽向量指令，
无标量 `GetValue/SetValue`（标量 UB 访问是 AIV 上最大的性能杀手）。

**关键原语**（数据布局 `x[16m + 4r + c]`，m: 矩阵, r: 行, c: 列）：

*   `GroupSum4` / `GroupMax4`（每 4 个连续元素归约为 1 个，输出布局 `s[4m + r]`）：
    两轮 `GatherMask`（内置奇偶拆分 pattern 1/2，每轮把相邻对拆分）+ `Add`/`Max`。
    第一轮 `e[j]=x[2j]`, `o[j]=x[2j+1]` → `p = e+o`；第二轮对 `p` 再拆分求和即得 4 元素组和。
*   广播展开 `bc[i] = s[i/4]`：预计算字节偏移表 `idx_exp` + `Gather`。
*   批量 4x4 转置：预计算置换偏移表 `idx_tr` + `Gather`。
    4x4 转置置换是对合（involution），正置/反置共用同一张表。
*   偏移表构建：只需 16 项标量写入（一个周期），之后 `Adds` 倍增扩展到全表。
    **注意增量不同**：`idx_exp` 每 16 项消费 4 个源元素（+16 字节/周期），
    `idx_tr` 每 16 项为一个完整矩阵（+64 字节/周期）。

**各阶段流程**：

1.  **Softmax (dim=-1) + eps**:
    `GroupMax4` → `Gather` 广播 → `Sub` → `Exp` → `GroupSum4` → `Reciprocal` + Newton 精化
    → `Gather` 广播 → `Mul` → `Adds(eps)`（softmax 分母不加 eps，与 baseline 一致）
2.  **Column-normalize (dim=-2)**:
    `Gather` 转置 → RowNormalize（转置域）→ `Gather` 转置回来
3.  **Row-normalize (dim=-1)**:
    `GroupSum4` → `Adds(eps)` → `Reciprocal` + Newton 精化 → `Gather` 广播 → `Mul`
4.  **执行顺序**（与 baseline 一致）: softmax → col → (row → col) × (repeat-1)

**精度说明**：
- AIV 的 `Reciprocal` 实测仅 ~10 bit 精度（相对误差 ~6e-4），19 级归一化级联后会逼近 1e-2 容差，
  因此补一步 Newton 迭代 `r = r*(2 - s*r)`，单级相对误差降至 ~4e-7。
- 用 `Reciprocal`+`Mul` 替代 `Div`：经 Newton 精化后精度足够，且避免 Div 的额外开销。

## 3. Tiling 策略 (两级 Tiling)

**算子特性分析**：
输入总数据量为 $1 \times 1024 \times 4 \times 4 \times 4 \text{ Bytes} = \mathbf{64 \text{ KB}}$。
而 910B 芯片单个 AIV 核心的 UB 容量为 **192 KB**。这意味着**全局数据甚至能塞进单个核心**！
为了最大化利用 40 个 AIV 核心的算力，我们必须进行空间切分（Block级 Tiling）。

### 3.1 Block 级 Tiling (多核切分 - 空间循环)
*   **总任务量**: 1024 个矩阵 (每矩阵 16 个 float32)。
*   **分配策略**: 均分策略，充分利用所有 AIV 子块。
*   **MIX 启动的实际核数**（重要，见 §4 第 4 点）：
    该工具链下 kernel 以 AIC+AIV 混合方式启动，`if ASCEND_IS_AIV` 守护后仅 AIV 执行；
    AIV 侧 `GetBlockIdx()` 是全局 AIV 下标，有效工作单元数 = `GetBlockNum() * GetSubBlockNum()`
    = 40 × 2 = **80 个 AIV 子块**。
*   **计算公式**:
    `matrices_per_core = ceil(1024 / 80) = 13`
    每核处理数据量：`13 * 16 = 208` 个 float32。尾核可能更少，需做边界判断。

### 3.2 UB 级 Tiling (核内切分 - 时间循环)
单核数据极小（$208 \times 4 = 832 \text{ Bytes}$），远远小于 UB 限制。
**策略：One-Shot 搬运，UB 内死循环。**
*   **切块 (Chunk)**: 不需要切块。
*   **对齐 padding**: 为喂满两轮 `GatherMask`（每轮消费 64 floats），每核工作区统一 pad 到
    **512 floats（32 个矩阵）**。padding 区为未初始化数据，但其计算结果只会留在 padding 组内
    （组边界按 4 元素对齐），不会污染有效区，无需清零。
*   **流程**: 
    1. 一次性 `DataCopy` 将 ≤13 个矩阵搬入 UB（`MTE2→V` 用 `SetFlag/WaitFlag` 同步）。
    2. 在 UB 内完成 softmax + 19 级行列归一化。
    3. `V→MTE3` 同步后，一次性 `DataCopy` 写回 GM（out-of-place 写到独立的 `out`）。

### 3.3 UB 分配表 (Buffer 规划)
数据类型 `float32`，每核工作区固定按 512 floats 规划（N=512, H=256, Q=128）：
| Buffer 名称 | 用途 | 大小 (floats) | 空间 (Bytes) |
| :--- | :--- | :--- | :--- |
| `x_` | 输入/输出工作区，原地计算 | 512 | 2048 |
| `t_` | 转置临时区（列归一化用） | 512 | 2048 |
| `bc_` | 广播展开临时区 | 512 | 2048 |
| `e_` / `o_` | GatherMask 奇偶拆分临时区（`e_` 兼作 Newton 临时） | 256 × 2 | 2048 |
| `s_` / `r_` | 归约结果 / 倒数（`r_` 兼作第二轮拆分临时） | 128 × 2 | 1024 |
| `idx_exp_` / `idx_tr_` | 广播展开 / 转置的字节偏移表（uint32） | 512 × 2 | 4096 |

*总 UB 使用量*：$\approx 13 \text{ KB}$，完美契合 192 KB 的限制。

## 4. 性能优化核心考量 (Performance Optimizations)

1.  **究极显存墙突破 (Ultimate Memory Wall Breakthrough)**：
    原版 PyTorch 实现中，10 次迭代会引发约 **40 次 HBM 读写**。
    本融合算子实现了真正的 Temporal 复用：**1 次读 HBM，1 次写 HBM**。理论带宽需求下降 97.5%。
2.  **批处理向量化 (Batched Vectorization)**（已实现）：
    全部计算为全宽向量指令，零标量 GetValue/SetValue：
    *   行/列的 4 元素归约：两轮 `GatherMask`（奇偶拆分 pattern 1/2）+ `Add`/`Max`；
    *   归约结果的广播展开：预计算字节偏移表 + `Gather`；
    *   列归一化：偏移表 `Gather` 做批量 4x4 转置（对合置换，正反向同表）→ 行归一化 → 转置回来；
    *   偏移表只需 16 项标量写入 + `Adds` 倍增扩展（注意：`idx_exp` 每 16 项增量为 16 字节，`idx_tr` 为 64 字节）。
3.  **精度补偿**：AIV 的 `Reciprocal` 仅 ~10 bit 精度，补一步 Newton 迭代 `r = r*(2 - s*r)`，相对误差降至 ~4e-7。
4.  **MIX 启动防护**：该工具链忽略 `core_type("AIV")` 属性，kernel 会被 AIC+AIV 同时执行。必须用 `if ASCEND_IS_AIV` 守护，且 AIV 侧 `GetBlockIdx()` 为全局 AIV 下标（每 AI 核 2 个子块），切分按 `GetBlockNum() * GetSubBlockNum()` 计算。
5.  **Host 侧固定开销优化**（64KB 小算子调用开销占主导）：
    *   out-of-place 写 `at::empty_like` 输出，避免 `clone` 的 D2D 拷贝；
    *   Python 侧通过 pybind11 直调入口（`PYBIND11_MODULE`）绕过 torch.ops 调度器。

## 5. 交付标准 (DoD)
- [x] 确定数据无需精度转换 (FP32 进，FP32 出)。
- [x] 确定 Tiling 结构仅需一维（按矩阵个数切分）。
- [x] UB Buffer 充足，无需 double-buffer 流水线，重点放在内层循环的计算密集化。

## 6. 实测结果 (auto_bench.py, seed 1/7/42)
- v0 (Torch baseline): ~1.3 ms；v1 (本算子): ~0.13–0.15 ms
- **Speedup: 8.7x – 10.2x**（accuracy PASS, atol/rtol=1e-2）
- 瓶颈已从 kernel 计算转移至每次调用的固定开销：synchronize 下限 ~26µs + host 入队
  （pybind + empty_like + launch）~21µs + kernel ~20–25µs。

## 7. 进一步优化实验记录（结论：kernel 侧空间已尽）
- **精简 PipeBarrier**：同 V pipe 内向量指令（含 `Gather`）依赖自动保序，仅 `GatherMask` 后必须保留
  barrier。全程 barrier 从 ~80 个减到 5 个，实测无显著变化（kernel 为发射/排空受限，非 barrier 受限）。
- **工作区 512→256 floats（失败，勿试）**：每核有效仅 208 floats，理论可减半向量工作量；但实测
  AIV 双子块（同一 AI 核的 sub-block 0/1）在 256 工作区下 `Gather` 会读到对方 UB 的数据
  （疑似子块对 UB 共享/寻址限制），core 0 正常、core ≥1 数据错乱。保持 512 不变。
- 剩余优化只能在评测方法学下限（per-call sync）之外寻找，kernel 侧不建议再投入。
