# TerraceMem Offline Evaluation Suite

本目录包含两套可复现离线评测脚本，均支持无外网确定性 `--dry-run` 基线与配置 API Key 后的 `--live` 在线评测：

1. **`evals/rag_ablation.py`**：基于 `evals/datasets/tech_docs_hardset.json`（30 题中英文技术文档、多版本覆盖、报错码、表格与 10 道无答案拒答题）或自定义数据集，执行 `B0`–`B5` 及 `B_kw_only` 共 7 组消融对照。
2. **`evals/memory_benchmark.py`**：执行 `M0`（Sliding Window）、`M1`（Full Context）、`M2`（Flat Vector）与 `M3`（TerraceMem 双时态分层记忆）对比评测，支持内置双时态基准或通过 `--longmemeval-path` 传入 LongMemEval-S (`longmemeval_s_cleaned.json`)。

---

## 快速复现命令

```bash
# 1) 7 组 RAG 消融（确定性离线 dry-run 基线，产物输出至 evals/reports/）
uv run python evals/rag_ablation.py --dry-run

# 2) 4 组记忆基线对比（确定性离线 dry-run）
uv run python evals/memory_benchmark.py --dry-run

# 3) 配置好 .env 中的 OpenAI 兼容 LLM/Embedding API 后运行在线评测：
uv run python evals/rag_ablation.py --live
uv run python evals/memory_benchmark.py --live --longmemeval-path evals/datasets/raw/longmemeval_s_cleaned.json
```

---

## 离线确定性基线结果（`dry_run_deterministic`，无外部 LLM/Reranker API）

> 说明：以下数值为本地 `--dry-run`（`DeterministicHashEmbeddingProvider` + `LexicalOverlapReranker` + 抽取式生成兜底）实测输出，用于验证管道与消融逻辑闭环；启用外部大模型与 Cross-Encoder Reranker 后可运行 `--live` 获取端到端生成表现。

### RAG 消融（`tech_docs_hardset`，30 题）

| Config | Recall@5 | MRR@10 | NDCG@10 | Citation Validity | Term Match | Abstention F1 | FAR | P95 (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0` | 1.000 | 0.887 | 0.914 | 1.000 | 0.600 | 0.000 | 1.000 | 16.0 |
| `B1` | 1.000 | 0.870 | 0.902 | 1.000 | 0.650 | 0.000 | 1.000 | 16.0 |
| `B_kw_only` | 1.000 | 1.000 | 1.000 | 1.000 | 0.850 | 0.000 | 1.000 | 8.2 |
| `B2` | 1.000 | 0.942 | 0.957 | 1.000 | 0.750 | 0.000 | 1.000 | 16.0 |
| `B3` | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 0.000 | 1.000 | 16.0 |
| `B4` | 1.000 | 1.000 | 1.000 | 1.000 | 0.650 | 0.000 | 1.000 | 16.0 |
| `B5` | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.947 | 0.100 | 16.0 |

### 分层双时态记忆对比（`builtin_bitemporal_suite`，8 组长短周期难例）

| Baseline | Overall Acc | Knowledge Update | Temporal | Multi-Session | Abstention | Token Reduction % | False Invalidation Rate | P95 (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `M0` | 0.125 | 0.000 | 0.000 | 0.000 | 0.000 | 7.4% | 0.000 | 0.0 |
| `M1` | 0.250 | 0.000 | 0.000 | 1.000 | 0.000 | 0.0% | 0.000 | 0.0 |
| `M2` | 0.250 | 0.000 | 0.000 | 1.000 | 0.000 | 23.8% | 1.000 | 233.3 |
| `M3` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 63.7% | 0.000 | 10.4 |

详细指标口径请参阅 [../docs/evaluation_plan.md](../docs/evaluation_plan.md)。
