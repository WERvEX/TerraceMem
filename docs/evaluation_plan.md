# TerraceMem 离线评测方案设计

本项目将「可复现量化评测」作为一等公民，提供两套独立 CLI 评测脚本，全部支持 `--dry-run`、断点续跑、本地缓存与 JSON + Markdown 自动报告生成。

---

## 1. Agentic RAG 消融评测（`evals/rag_ablation.py`）

### 1.1 数据集组合（兼顾权威公信力、中英文与工程边界）

| 数据集 | 抽样规模（MVP） | 重点验证能力 |
| :--- | :--- | :--- |
| **RGB (Retrieval-Augmented Generation Benchmark)** | 120 题（60 噪声鲁棒 + 60 **无答案拒答**） | 噪声抗干扰能力、**Negative Rejection（严格拒答率 / 误答率 FAR）** |
| **MultiHop-RAG** | 100 题（约 300–500 篇文档入库） | 查询拆解、BM25 + Dense 混合召回、Rerank 与 **2-hop 轻量关系边扩展** |
| **自建 `tech_docs_hardset`** | 30 题（Markdown/PDF 技术文档 + 版本变更 + 报错码 + 表格） | 结构化切分、父子上下文回溯、SQL 版本/租户硬过滤、精确词法匹配 |

### 1.2 7 组消融对照矩阵

| 组别 | 配置组合 | 验证假设 |
| :--- | :--- | :--- |
| `B0` | Naive Dense RAG（固定 512 字切分 + 纯向量 Top-5） | 行业最简基线 |
| `B1` | + 结构化树切分与标题面包屑上下文 | 结构感知切分的收益 |
| **`B_kw_only`** | **写时精炼（摘要/关键词）+ 纯 ParadeDB BM25（关闭向量）** | **量化检验“去向量、纯关键词+写时总结”路线在精确匹配 vs 同义改写/长尾细节题上的真实得失与入库成本** |
| `B2` | 混合召回（ParadeDB BM25 + pgvector Iterative HNSW + RRF） | 双路互补相比纯向量与纯关键词的净增益 |
| `B3` | `B2` + Cross-Encoder Rerank + 父子上下文回溯 | 精排与上下文补全的跃升 |
| `B4` | `B3` + 写时精炼增强（关键词并入 BM25 + 章节目录导航）+ 2-hop 关系边 | 写时增强与轻量多跳的边际贡献 |
| `B5` | **Full System**：`B4` + CRAG 门控、最多 2 次纠错重检、引用白名单校验与严格拒答 | 幻觉/误答率（FAR）压降幅度与延迟代价 |

### 1.3 指标口径（检索层与生成层严格解耦）

- **检索层（100% 确定性计算，零 Judge 成本）**：`Recall@5`、`Recall@10`、`MRR@10`、`NDCG@10`
- **可信生成层**：
  - `Citation Validity`：引用 ID 100% 属于检索证据白名单且有效的比例
  - `Faithfulness`：原子声明被对应引用证据块蕴含的比例
  - `Abstention F1` & `False Answer Rate (FAR)`：无答案题的正确拒答能力与幻觉作答率
- **系统成本层**：`P50 / P95 Latency (ms)`、`Avg Ingest Tokens`、`Avg Query Tokens`

---

## 2. 分层记忆对照评测（`evals/memory_benchmark.py`）

### 2.1 数据集：LongMemEval-S（ICLR 2025）
- 来源：`xiaowu0162/longmemeval-cleaned` (`longmemeval_s_cleaned.json`)
- MVP 抽样规模：**80–100 题分层抽样**，重点高比例纳入：
  - `knowledge-update`（事实冲突与覆盖更新 —— 直接检验双时态记忆核心价值）
  - `temporal-reasoning`（时间线与先后顺序推理）
  - `multi-session`（跨会话聚合推理）
  - `abstention`（虚假前提拒答）
  - `single-session-user` / `single-session-preference`（基础细节与偏好记忆）

### 2.2 4 组基线对照

| 基线 | 实现说明 | 观测重点 |
| :--- | :--- | :--- |
| `M0` | **Sliding Window**（仅保留最近 10 轮原始消息） | 下界基线，暴露长周期遗忘率 |
| `M1` | **Full Context**（全量历史会话直接塞入上下文） | 准确率参考上界 & Token 成本上限 |
| `M2` | **Flat Vector Memory**（扁平向量记忆，无分层与双时态冲突消解） | **核心对照靶子**：暴露旧事实未被失效导致的冲突污染 |
| `M3` | **TerraceMem Tiered Bi-temporal Memory**（L1 压缩 + L2 情景 + L3 双时态冲突消解 + 热度衰减） | 对比 `M2` 的知识更新准确率提升 & 对比 `M1` 的 Token 节省率 |

### 2.3 指标口径
- **分题型 Accuracy**：总分、`knowledge-update`、`temporal-reasoning`、`multi-session`、`abstention`
- **Token 效率**：相对 Full Context（`M1`）的 **Prompt Token Reduction %**
- **时延与安全性**：单次检索 `P95 Latency`、**False Invalidation Rate（冲突误伤率）**
