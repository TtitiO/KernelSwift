# Indexer QK-Reduce Fused Megakernel Design

根据 `ascendc-operator-design` 规范，设计针对 Task 02 (Indexer) 的 `__mix__(1,2)` 混合融合算子。

## 1. 算子需求分析与接口定义

*   **算子名称**: `fused_indexer_qk_reduce`
*   **功能描述**: 融合大模型路由模块的核心打分阶段。
    *   完成 QK GEMM: `index_score = q @ kv^T`
    *   完成 激活、加权与归约: `out = sum_h(ReLU(index_score) * weights)`
*   **支持的数据类型**: 输入 `bfloat16`，输出 `float32` (后续给 TopK 排序用)
*   **输入 Shape**: 
    *   `q`: `[B, S, H, D]` -> 视作 `[8, 41600, 64]`
    *   `kv`: `[B, T, D]` -> `[8, 650, 64]`
    *   `weights`: `[B, S, H]` -> 视作 `[8, 41600]`
*   **输出 Shape**:
    *   `out`: `[B, S, T]` -> `[8, 2600, 650]`

## 2. 计算逻辑设计

**实现路径选择**: **AscendC Mixed Kernel `__mix__(1,2)`** (带 AIC/AIV 跨核同步)。

**数学公式到 AscendC API 的映射 (AIV端)**:
AIV 获取到一个 `[32, 656]` 的 FP32 Tile (代表 2 个完整的 S Token，每个 Token 包含 H=16 行)。
1.  **ReLU**: `Max(row_buf, row_buf, 0.0f, len)`
2.  **Multiply Weights**: 标量乘以向量 `Muls(row_buf, row_buf, weight_scalar, len)`
3.  **Reduce Sum (Head 维度)**: 将属于同一个 Token 的 16 行累加。
    `Add(out_buf, out_buf, row_buf, len)`

## 3. Tiling 策略 (多级切分)

**对齐约束 (Alignment)**：
Cube 矩阵乘法要求 N 维度是 16 的倍数。原始 T=650，必须 Padding 到 **N=656**。

### 3.1 Block 级 Tiling (多核切分 - AIC)
*   `M_total` = $2600 \times 16 = 41600$ (每 Batch)
*   `tile_M` = 64, `tile_N` = 656, `tile_K` = 64
*   每个 Batch 产生 $41600 / 64 = 650$ 个 Tile。
*   共 8 个 Batch，总 Tile 数 = 5200。
*   `blockDim = 20` (所有 20 个 AI Core 拉满)，每个 Core 处理 260 个 Tile。

### 3.2 Sub-Block 级 Tiling (核内分工 - AIV)
*   每个 AIC 输出一个 `[64, 656]` 的 `index_score` 中间块到 GM (由于 L2 Cache Hit，极速读写)。
*   **AIV0** 负责处理该块的前 32 行 (对应 2 个完整的 Token)。
*   **AIV1** 负责处理该块的后 32 行 (对应 2 个完整的 Token)。

### 3.3 UB 分配表 (Buffer 规划，针对单个 AIV 子块)
| Buffer 名称 | 用途 | 形状/大小 (元素数) | 预估空间 (Bytes) | 策略 |
| :--- | :--- | :--- | :--- | :--- |
| `in_scoresQ` | 接收 QK 结果 | `[32, 656]` FP32 | $83,968$ B | Double-buffer ($168$ KB) |
| `weights_buf`| 存对应的头权重 | `[32]` FP32 | $128$ B | Single |
| `out_reducedQ`| 存累加后的结果 | `[2, 656]` FP32 | $5,248$ B | Double-buffer ($10.5$ KB) |

*总 UB 使用量*：$\approx 178 \text{ KB} < 192 \text{ KB}$ (完美压线，将单核性能榨干到极致)。

## 4. 性能优化核心考量
1.  **消除巨型张量 HBM 流量**：原始 432MB 的中间张量 `[B,S,H,T]` 仅在片上 L2 Cache/UB 流转，直接输出 27MB 归约结果。
2.  **极致的指令对齐**：N=656 是 16 的倍数，意味着可以使用最高效的 256Byte 宽屏向量计算，没有尾部处理开销。
3.  **完美 AIV 负载均衡**：利用 `H=16` 的特性，AIV 切块刚好在 Token 边界，无需复杂的跨块(cross-tile)归约逻辑，全是干净的原地累加。
