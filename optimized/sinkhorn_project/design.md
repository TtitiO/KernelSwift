# Sinkhorn Operator Design

根据 `ascendc-operator-design` 规范，设计针对华为 Atlas A2 (Ascend 910B) 的纯 Vector 核心 (AIV) Sinkhorn 融合算子。

## 1. 算子需求分析与接口定义

*   **算子名称**: `sinkhorn`
*   **功能描述**: 对输入张量进行多次行列归一化迭代（双随机矩阵算法），将整个迭代过程融合在一个 Kernel 中，完全消除迭代过程中的 HBM（显存）读写开销。
*   **支持的数据类型**: `float32` (赛题指定)
*   **输入 Shape**: `[1, 1024, 4, 4]` (即 1024 个 `4x4` 矩阵)
*   **接口签名**:
    ```cpp
    at::Tensor sinkhorn_npu(const at::Tensor &x, int64_t repeat, double eps);
    ```

## 2. 计算逻辑设计

**实现路径选择**: **AscendC Kernel (纯 AIV 向量算子)**。
原因：此算子仅包含 element-wise 运算和沿维度的 reduction（求和、求最大值），不涉及高计算密度的矩阵乘法（无需 Cube 单元）。

**数学公式到 AscendC API 的映射**:
整个计算全在 UB（片上缓存）中 In-place（原地）完成，循环 `repeat` 次。

1.  **Softmax (dim=-1) + eps**:
    *   `max(x)` -> `Max(max_buf, x, ...)`
    *   `x - max` -> `Sub(x, x, max_buf, ...)`
    *   `exp(x)` -> `Exp(x, x, ...)`
    *   `sum(exp)` -> `ReduceSum(sum_buf, x, ...)`
    *   `x / sum` -> `Reciprocal(recip_buf, sum_buf, ...)` + `Mul(x, x, recip_buf, ...)`
    *   `+ eps` -> `Adds(x, x, eps, ...)`
2.  **Column-normalize (dim=-2)**:
    *   `sum(x, dim=-2)` -> `ReduceSum(sum_buf, x, ...)` (沿列方向求和)
    *   `x / (sum + eps)` -> `Adds(sum_buf, sum_buf, eps, ...)` + `Reciprocal(recip_buf, sum_buf)` + `Mul(x, x, recip_buf)`
3.  **Loop (repeat - 1) 次**:
    *   Row-normalize (dim=-1) -> 类似上方列归一化，但 `ReduceSum` 沿行方向。
    *   Column-normalize (dim=-2) -> 同上。

## 3. Tiling 策略 (两级 Tiling)

**算子特性分析**：
输入总数据量为 $1 \times 1024 \times 4 \times 4 \times 4 \text{ Bytes} = \mathbf{64 \text{ KB}}$。
而 910B 芯片单个 AIV 核心的 UB 容量为 **192 KB**。这意味着**全局数据甚至能塞进单个核心**！
为了最大化利用 40 个 AIV 核心的算力，我们必须进行空间切分（Block级 Tiling）。

### 3.1 Block 级 Tiling (多核切分 - 空间循环)
*   **总任务量**: 1024 个矩阵 (每矩阵 16 个 float32)。
*   **分配策略**: 采用均分策略，充分利用所有 AIV。
*   **计算公式**: 
    假设 `core_num = 40`
    `matrices_per_core = ceil(1024 / 40) = 26`
    每核处理数据量：`26 * 16 = 416` 个 float32。
    尾核可能处理更少，需做边界判断。

### 3.2 UB 级 Tiling (核内切分 - 时间循环)
由于分配到单核的数据极小（$416 \times 4 = 1664 \text{ Bytes}$，不到 2KB），远远小于 UB 限制。
**策略：One-Shot 搬运，UB 内死循环。**
*   **切块 (Chunk)**: 不需要切块。
*   **流程**: 
    1. 一次性 `DataCopy` 将 26 个矩阵搬入 UB。
    2. 在 UB 内部写 `for (int i=0; i<repeat; i++)` 循环。
    3. 循环结束后，一次性 `DataCopy` 将结果写回 GM。

### 3.3 UB 分配表 (Buffer 规划)
数据类型 `float32` (dtypeSize = 4)
| Buffer 名称 | 用途 | 大小设计 (元素数) | 预估空间 (Bytes) |
| :--- | :--- | :--- | :--- |
| `x_local` | 存输入/输出数据，原地计算 | `matrices_per_core * 16` | $\approx 1664$ B |
| `tmp_max` | 存 Softmax 的行最大值 | `matrices_per_core * 4` | $\approx 416$ B |
| `tmp_sum` | 存 行/列 求和结果 | `matrices_per_core * 4` | $\approx 416$ B |
| `tmp_recip` | 存除数倒数用于乘法 | `matrices_per_core * 4` | $\approx 416$ B |
| `bcast_tmp` | 广播操作临时空间 (Brcb需要) | $256$ | $256$ B |

*总 UB 使用量*：$< 4 \text{ KB}$，完美契合 192 KB 的限制。

## 4. 性能优化核心考量 (Performance Optimizations)

1.  **究极显存墙突破 (Ultimate Memory Wall Breakthrough)**：
    原版 PyTorch 实现中，10 次迭代会引发约 **40 次 HBM 读写**。
    本融合算子实现了真正的 Temporal 复用：**1 次读 HBM，1 次写 HBM**。理论带宽需求下降 97.5%。
2.  **批处理向量化 (Batched Vectorization)**：
    由于每个矩阵只有 $4 \times 4$，填不满单条 256B (64 个 float32) 的向量指令。在设计 Kernel 时，不要一个矩阵一个矩阵地处理，而是将 `matrices_per_core` 个矩阵看作一个连续的内存块，使用 Batched Vector 指令一次性完成多个矩阵的加法/乘法。

## 5. 交付标准 (DoD)
- [x] 确定数据无需精度转换 (FP32 进，FP32 出)。
- [x] 确定 Tiling 结构仅需一维（按矩阵个数切分）。
- [x] UB Buffer 充足，无需 double-buffer 流水线，重点放在内层循环的计算密集化。
