# NNLM Optimization and Performance Profiling Report

This document records the optimization techniques, architectural invariants, and empirical profiling benchmarks for the feed-forward Neural Network Language Model (NNLM) Downpour SGD implementation in `packages/w2v/src/nnlm/`.

---

## 1. Executive Summary

Through phased algorithmic, cache, and SIMD optimizations, the NNLM Downpour training throughput on Table 3 dimensions ($D=640, H=640, N=8, B=250$) improved from **549.4 tokens/s** to **4,211.8 tokens/s** on an Intel Xeon Silver 4210 dual-socket workstation:

- **Single-thread throughput:** 60.0 tokens/s $\rightarrow$ **463.9 tokens/s (7.73x speedup)**
- **20-worker throughput:** 549.4 tokens/s $\rightarrow$ **4,211.8 tokens/s (7.67x speedup)**
- **Table 3 full execution (9 runs, 8.64B tokens):** Reduced from **182.0 days (~6 months)** to **21.7 ~ 23.8 days**.
- **Bit-level determinism:** 100% mathematical, gradient, and checkpoint resume compatibility preserved.

---

## 2. Optimization Stages

### 2.1 Step 1: Elimination of Dense `hidden_weights` Allocation
- **Problem:** Every target gradient previously allocated an independent $13.1\text{ MB}$ vector (`640 * 5120` float32) for $\nabla_W L$. For 250 targets, each batch allocated and deallocated over $3.28\text{ GB}$ of memory.
- **Solution:** Replaced dense allocation with dynamic outer product synthesis $\nabla_W L = dz \otimes \text{projection}$ using pre-existing `projection` and `hidden_bias` buffers.
- **Result:** Loop heap allocation reduced to zero. Single-thread throughput increased from 60.0 to 77.6 tokens/s; 20-thread throughput increased from 549.4 to 976.4 tokens/s.

### 2.2 Step 2: AVX2+FMA Batched GEMM Kernels
- **Problem:** Target-by-target matrix-vector operations underutilized SIMD execution units and memory bandwidth.
- **Solution:** Introduced batched execution (`execute_batch`) aggregating 250 targets per batch:
  - Forward GEMM: $Z = X W^T + b$
  - Backward GEMM 1: $dW = DZ^T X$
  - Backward GEMM 2: $DP = DZ \cdot W$
  - Leveraged `ReplicaScratch` pre-allocated buffers across iterations.
- **Result:** 20-thread throughput scaled to 1,730.2 tokens/s (3.15x over baseline).

### 2.3 Step 3: Zero-Copy Flat Buffer, In-Place Cache Refresh, and Batched HS
- **P1 (Flat Buffer Zero-Copy Gradient Push):** Replaced `BTreeMap<usize, Vec<Real>>` for dense hidden weights and bias with pre-allocated flat vectors (`hidden_rows`, `bias_row`). Gradient transmission to the Parameter Server uses `std::mem::take` zero-copy packet transfers.
- **P2 (In-Place Replica Cache Refresh):** Eliminated the $13.1\text{ MB}$ clone operation per mini-batch in `ReplicaCache`. Implemented `ReplicaCache::refresh(&mut self)` to update weights in-place via atomic loads into existing buffers.
- **P3 (Batched Hierarchical Softmax):** Sorted and grouped Huffman tree node visits (`NodeVisit`) across 250 targets, maximizing L1/L2 cache locality and reusing node weights.
- **Result:** 20-thread throughput reached 2,386.0 tokens/s.

### 2.4 Step 4: Sparse Flat Aggregation and 2D Block Tiling
- **Sparse Parameter Flat Accumulation:** Replaced `BTreeMap` in `ReplicaCache` with `HashMap` ($O(1)$ lookup, retained capacity on `clear()`). Aggregated output and input gradients by sorting visits and directly accumulating into flat vectors, eliminating thousands of tree node allocations per batch.
- **Blocked GEMM (Cache Tiling):**
  - **Forward GEMM ($Z = XW^T + b$):** Reordered outer loop over hidden dimension $j$. A single 20 KB row $W[j]$ resides in L1 D-cache (32 KB) across all 250 targets, reducing $W$ memory loads from 250 times to 1 time per batch.
  - **Backward GEMM 2 ($DP = DZ \cdot W$):** Similarly reordered loop over $j$ so $W[j]$ is reused in L1 cache across all targets.
  - **Backward GEMM 1 ($dW = DZ^T X$):** Applied $32 \times 32$ 2D block tiling over $H$ and $B$, fitting $X$ sub-blocks (655 KB) inside the core's 1 MB L2 cache and reducing $X$ memory passes from 640 to 20.
- **Result:** Single-thread throughput rose to 463.9 tokens/s.

---

## 3. Hardware Topology and Thread Placement

### 3.1 Dual-Socket NUMA Topology (`lscpu`)
- **Host CPU:** Dual Intel Xeon Silver 4210 @ 2.20GHz (20 physical cores, 40 logical threads)
- **Node 0:** Physical cores `0-9`, SMT/HT threads `20-29`
- **Node 1:** Physical cores `10-19`, SMT/HT threads `30-39`

### 3.2 Thread Affinity Findings
| Affinity Strategy | Cores | Characteristics | 20-Worker Throughput |
| :--- | :--- | :--- | :---: |
| **OS Default** | Arbitrary (`0-39`) | Uncontrolled core migration, cross-socket UPI traffic | 2,367.1 tokens/s |
| **Single NUMA (`0-9, 20-29`)** | Node 0 (10 physical + 10 HT) | Severe AVX2/FMA execution unit contention (-23%) | 1,827.2 tokens/s |
| **Physical Cores (`0-19`)** | Node 0 + Node 1 physical only | Full AVX2 FMA unit isolation per worker (**Optimal**) | **3,246.5 ~ 4,211.8 tokens/s** |

> **Key Takeaway:** Hyper-Threading (SMT) causes severe pipeline stalling on dense AVX2 FMA workloads. Pinning exclusively to physical cores (`taskset -c 0-19`) provides optimal throughput.

---

## 4. Parameter Server (PS) Scaling Analysis

To verify whether the Parameter Server was a bottleneck, throughput was benchmarked across shard counts on 20 workers pinned to cores `0-19`:

| PS Shards | Total Threads (20 workers + PS) | 26k Token Elapsed | Measured Throughput |
| :---: | :---: | :---: | :---: |
| **1 Shard** | **21** | **6.173s** | **4,211.8 tokens/s (Highest)** |
| 2 Shards | 22 | 7.821s | 3,324.5 tokens/s |
| 4 Shards | 24 | 8.907s | 2,919.1 tokens/s |
| 8 Shards | 28 | - | ~1,552.9 tokens/s |
| 16 Shards | 36 | - | ~1,826.9 tokens/s |

- **Conclusion:** The Parameter Server is **not** a bottleneck. A single PS thread easily processes gradient pushes from 20 workers.
- Increasing shard counts beyond 1 creates overthreading on the 20 physical cores, causing thread context switching overhead. Setting `parameter_server_shards=1` achieves the highest performance.

---

## 5. End-to-End Benchmark Progression

Table 3 configuration: $D=640, H=640, N=8, B=250$.

| Stage | Optimization Scope | 1 Thread Throughput | 20 Worker Throughput | Speedup vs Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | Initial scalar code with per-target 13.1 MB alloc | 60.0 tokens/s | 549.4 tokens/s | 1.00x |
| **Step 1** | Target dense alloc removal | 77.6 tokens/s | 976.4 tokens/s | 1.78x |
| **Step 2** | Batch GEMM with SIMD kernels | 542.0 tokens/s | 1,730.2 tokens/s | 3.15x |
| **Step 3** | Zero-copy flat buffers, cache refresh, batched HS | 396.3 tokens/s | 2,386.0 tokens/s | 4.34x |
| **Step 4** | Physical core affinity (`0-19`) | - | 3,246.5 tokens/s | 5.91x |
| **Step 5** | 2D Block Tiling + PS Shard 1 | **463.9 tokens/s** | **4,211.8 tokens/s** | **7.67x** |

---

## 6. Full Experiment Wall-Clock Forecasts

Full reproduction matrix covers 2 tables, 4 corpora, 3 seeds (18 total runs):
- **Table 3 (9 runs):** WMT, LM1B, UMBC $\times$ seeds 1, 7, 19 (8.64B total tokens)
- **Table 4 (9 runs):** FineWeb 20d, 50d, 100d $\times$ seeds 1, 7, 19 (162B total tokens)
- **Table 6 (0 new runs):** Reuses Table 4 100d runs.

### Forecast by Execution Mode

| Mode | Measured Throughput | Table 3 per Run (960M tokens) | **Table 3 Complete (9 Runs, 8.64B tokens)** |
| :--- | :---: | :---: | :---: |
| **Baseline** | 549.4 tokens/s | 20.2 days | **182.0 days (~6.1 months)** |
| **20 Workers (`0-19`, PS=1)** | **~4,200 tokens/s** | **2.65 days (63.5h)** | **23.8 days** |
| **10 Workers (Single socket `0-9`, PS=1)** | **~2,855 tokens/s** | **3.89 days (93.4h)** | **35.0 days** |
| **10 Workers (Distributed `0-19`, PS=1)** | **~4,600 tokens/s** | **2.41 days (58.0h)** | **21.7 days** |

---

## 7. Recommended Bound Invocation

To safely run production jobs under resource constraints without CPU/RAM overuse:

```bash
# Optimal execution: 10 workers distributed across physical cores 0-19
bound --cpu 0-19 --numa all --mem 128G --threads 10 run \
  uv run repro f2 w2v1 -e 02 -a wmt--nnlm-d640-h640-n8-w320m -s 1

# Or attach current tmux pane to the resource bound
bound pane --cpu 0-19 --numa all --mem 128G --threads 10
```
