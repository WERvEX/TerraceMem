# TerraceMem

> 面向 Agent（尤其是 Coding Agent 与多 Agent 编排平台）的 **可信检索与双时态分层记忆引擎**。  
> *Trustworthy Agentic Retrieval & Bi-temporal Tiered Memory Engine on Unified PostgreSQL.*

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](#快速开始)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PostgreSQL: ParadeDB + pgvector](https://img.shields.io/badge/PostgreSQL-ParadeDB%20%2B%20pgvector%200.8-336791.svg)](docs/architecture.md)

---

## 为什么做 TerraceMem？

**Terrace（河流阶地）** 是在地质时间中逐级沉淀、层层抬升形成的天然地层结构——旧的沉积层不会被销毁，而是保留为可追溯的历史脉络。这正是 TerraceMem 的设计哲学：

- **单一 PostgreSQL 事务底座（零多库割裂、零 Redis）**：在一个 ACID 事务底座内统一承载关系过滤、文档章节树、`pgvector 0.8` 迭代向量检索（`halfvec`）、ParadeDB 真 BM25（内置 `jieba` 中文分词 + Fast Fields 多租户下推）、2-hop 轻量关系图递归 CTE、三层记忆表、PG `SKIP LOCKED` + `LISTEN/NOTIFY` 异步 Outbox 队列与全链路 Trace；未启动 PostgreSQL 时在 `TERRACEMEM_STORAGE_BACKEND=auto` 下自动优雅降级为同语义内存引擎，开箱即试。
- **确定性边界包住 LLM 推理（可控、可审计、低幻觉）**：
  - 不绑死重型 Agent 框架，采用自研强类型异步显式状态机（Typed Async FSM）；
  - 混合召回（BM25 + Dense + RRF + 可选 2-hop）→ Cross-Encoder 精排与父子上下文回溯 → CRAG 两级证据门控与有限次纠错重检 → 强制引用白名单校验与严格拒答；
  - 支持可选的 **写时摘要/关键词精炼（`ENABLE_WRITE_TIME_ENRICHMENT`）** 与章节树导航工具，兼顾确定性关键词定位与语义兜底。
- **双时态三层记忆（Tiered Bi-temporal Memory）**：
  - **L1 工作记忆**：Token 分层预算 + 增量滚动压缩（`pinned_constraints` 绝对保底不丢硬约束）；
  - **L2 情景记忆**：带时间戳的会话事件与决策归档；
  - **L3 语义记忆**：显式分离 **有效时间轴（`valid_from` / `valid_to`）** 与 **事务时间轴（`recorded_at` / `invalidated_at` / `superseded_by`）**，配合四重确定性防误伤护栏，新事实推翻旧事实时软失效而非物理删除，支持历史 `as_of` 时间旅行审计。
- **Headless 引擎 + MCP 外脑 + 内置轻量 Inspector**：
  - 同时暴露 **FastAPI REST / SSE**、**MCP Server（Streamable HTTP `/mcp` + `stdio`）**，可直接挂载给 Claude Code、Codex 或自研多 Agent 编排平台；
  - 在 `/ui` 内置零构建负担的 4 面板开发者调试看板（文档树入库、RAG SSE Playground 与 Trace 瀑布流、双时态时间线、消融评测图表）。
- **可复现离线评测闭环**：内置 RAG 7 组消融评测（`evals/rag_ablation.py`）与 4 组记忆对照评测（`evals/memory_benchmark.py`），支持离线确定性 `--dry-run` 与在线 `--live` 一键复现。

---

## 系统架构概览

```mermaid
flowchart TB
    subgraph Clients["接入层"]
        A1["Claude Code / Codex\n(MCP stdio / Streamable HTTP)"]
        A2["多 Agent 编排平台 / 业务后端\n(REST / SSE)"]
        A3["开发者调试台\n(Embedded Inspector /ui)"]
    end

    subgraph Engine["TerraceMem Core (Python 3.12 + FastAPI + MCP SDK v2)"]
        RAG["Agentic RAG 显式状态机\nRouter -> Hybrid+RRF -> Rerank -> CRAG Gate -> CitationVerifier"]
        MEM["三层双时态记忆引擎\nL1 Compactor | L2 Episodic | L3 Conflict Guardrails"]
        ING["双轨结构化入库\nMarkdown/Code Tree + Docling + 可选写时精炼"]
        WRK["零 Redis 异步 Worker\nPG FOR UPDATE SKIP LOCKED + LISTEN/NOTIFY"]
    end

    subgraph PG["Unified PostgreSQL (ParadeDB + pgvector 0.8)"]
        T1["documents / doc_sections / doc_chunks\n(halfvec HNSW + Tantivy BM25 Jieba + Fast Fields)"]
        T2["entities / relation_edges\n(防环 2-hop 递归 CTE)"]
        T3["sessions / episodic_memories / semantic_memories\n(双时态有效期 + 冲突链)"]
        T4["async_jobs + execution_traces / trace_spans"]
    end

    A1 & A2 & A3 --> RAG & MEM & ING
    MEM & ING -.同事务入队.-> WRK
    RAG & MEM & ING & WRK --> T1 & T2 & T3 & T4
```

---

## 快速开始

### 1. 环境准备与依赖安装

```bash
# 1) 同步核心依赖与开发测试工具（推荐 uv）
uv sync --extra dev

# 2) 如需 PDF / DOCX 复杂版面解析（Docling）或离线评测依赖
uv sync --extra dev --extra docling --extra eval

# 3) 复制并配置环境变量（支持任意 OpenAI 兼容 API）
cp .env.example .env
```

### 2. 启动 PostgreSQL 统一底座（可选；未启动时 `auto` 模式会自动切换内存引擎）

```bash
docker compose up -d postgres
uv run alembic upgrade head
```

> 若本地或云端只有标准 PostgreSQL（无 `pg_search` 扩展），可在 `.env` 中设置 `TERRACEMEM_LEXICAL_BACKEND=native` 自动启用原生 `tsvector` + Python `jieba` 降级后端。

### 3. 启动服务 / Outbox Worker / 挂载 MCP

```bash
# 启动 FastAPI REST/SSE + Streamable HTTP MCP (/mcp) + Inspector (/ui)
uv run terracemem serve --reload

# 可选：运行独立的零 Redis PG Outbox 后台 Worker
uv run terracemem worker

# 本地 stdio 模式供 Claude Code / Codex 直接挂载
uv run terracemem mcp --stdio
```

访问内置调试台：`http://localhost:8000/ui`。

Claude Code / MCP 客户端配置示例（`stdio` 或 `Streamable HTTP /mcp` 二选一）：

```json
{
  "mcpServers": {
    "terracemem": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/TerraceMem", "terracemem", "mcp", "--stdio"]
    }
  }
}
```

### 4. 核心 API 端点一览

| 路径 | 方法 | 说明 |
| :--- | :--- | :--- |
| `/health` / `/api/v1/info` | `GET` | 服务健康状态、当前存储后端模式与预算/特性开关 |
| `/api/v1/documents` | `POST` / `GET` | 版本幂等结构化入库（可选写时精炼与构图）/ 文档列表 |
| `/api/v1/documents/{id}/outline` | `GET` | 查看层级章节目录、摘要与关键词 |
| `/api/v1/documents/{id}` | `DELETE` | 软删除文档及历史版本（零泄漏） |
| `/api/v1/search` | `POST` | 混合检索（`hybrid` / `dense_only` / `lexical_only` + 2-hop + 精排） |
| `/api/v1/ask` / `/api/v1/ask/stream` | `POST` | Agentic RAG 同步问答 & SSE 流式状态事件输出 |
| `/api/v1/traces` / `/api/v1/traces/{id}` | `GET` | 查询结构化执行 Trace 与各节点 Span 瀑布流 |
| `/api/v1/sessions` | `POST` | 创建或获取 L1 工作记忆会话及 `pinned_constraints` |
| `/api/v1/memory/commit` / `recall` | `POST` | 提交对话归档 L1/L2/L3（含四重防误伤护栏）与加权召回 |
| `/api/v1/memory/timeline` | `GET` | 双时态历史与 `as_of` 时间旅行审计 |
| `/api/v1/memory/{id}/pin` | `POST` | 锁定/解锁 L3 核心记忆防止隐式覆写 |
| `/mcp` | Streamable HTTP | 暴露 `kb_browse_outline`, `kb_search`, `kb_ask`, `memory_recall`, `memory_commit`, `memory_timeline` |
| `/ui` | `GET` | 内置 4 面板开发者调试看板 |

---

## 离线评测与测试验证

### 运行单测、静态检查与 PostgreSQL 集成测试

```bash
uv run ruff check .
uv run mypy src
uv run pytest

# 若已启动 PostgreSQL 底座，可指定 DSN 运行 PG 全链路集成测试：
TERRACEMEM_TEST_DATABASE_URL=postgresql://terracemem:terracemem@localhost:5432/terracemem uv run pytest tests/test_postgres_integration.py -vv
```

### 复现评测基线（`--dry-run` 确定性离线基线）

```bash
uv run python evals/rag_ablation.py --dry-run
uv run python evals/memory_benchmark.py --dry-run
```

**RAG 7 组消融基线（`tech_docs_hardset`，30 题，`dry_run_deterministic` 模式）**：

| Config | Recall@5 | MRR@10 | NDCG@10 | Citation Validity | Abstention F1 | FAR (越低越好) | P95 (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0` (Flat Dense) | 1.000 | 0.887 | 0.914 | 1.000 | 0.000 | 1.000 | 16.0 |
| `B1` (Tree Dense) | 1.000 | 0.870 | 0.902 | 1.000 | 0.000 | 1.000 | 16.0 |
| `B_kw_only` (Enrich+BM25) | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 8.2 |
| `B2` (Hybrid RRF) | 1.000 | 0.942 | 0.957 | 1.000 | 0.000 | 1.000 | 16.0 |
| `B3` (+Rerank+Expand) | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 16.0 |
| `B4` (+Enrich+Graph+Outline) | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 16.0 |
| `B5` (Full System + CRAG) | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.100 | 16.0 |

**分层双时态记忆对比（`builtin_bitemporal_suite`，`dry_run_deterministic` 模式）**：

| Baseline | Overall Acc | Knowledge Update | Temporal | Multi-Session | Abstention | Token Reduction % | False Invalidation Rate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `M0` (Sliding Window) | 0.125 | 0.000 | 0.000 | 0.000 | 0.000 | 7.4% | 0.000 |
| `M1` (Full Context) | 0.250 | 0.000 | 0.000 | 1.000 | 0.000 | 0.0% | 0.000 |
| `M2` (Flat Vector) | 0.250 | 0.000 | 0.000 | 1.000 | 0.000 | 23.8% | 1.000 |
| `M3` (TerraceMem Bi-temporal) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 63.7% | 0.000 |

> 完整在线大模型评测可使用 `--live` 参数并挂载 LongMemEval-S / RGB 数据集运行，详见 [evals/README.md](evals/README.md)。

---

## 文档导航

- 架构设计与技术选型权衡：[docs/architecture.md](docs/architecture.md)
- 离线评测方案与消融矩阵：[docs/evaluation_plan.md](docs/evaluation_plan.md)
- 14 天 MVP 开发排期与里程碑：[docs/roadmap.md](docs/roadmap.md)
- 评测套件说明：[evals/README.md](evals/README.md)

---

## 仓库结构

```text
TerraceMem/
├── docker-compose.yml          # 一键启动 ParadeDB (PG + pg_search + pgvector)
├── pyproject.toml              # uv 依赖与 ruff / mypy / pytest 配置
├── alembic/                    # PostgreSQL 统一底座迁移脚本
├── docs/                       # 架构、评测与路线图文档
├── src/terracemem/
│   ├── config.py               # pydantic-settings 强类型配置与特性开关
│   ├── runtime.py              # 统一运行时容器（PG 自动探测与零依赖内存降级）
│   ├── cli.py                  # terracemem serve / mcp / worker / info CLI
│   ├── domain/                 # Pydantic v2 核心领域模型
│   ├── db/                     # psycopg3 连接池、文档/图/记忆/Trace 仓储与 SQL 模板
│   ├── llm/                    # OpenAI 兼容 LLM / Embedding / Rerank Provider
│   ├── ingestion/              # Markdown/代码原生树解析、Docling 适配、写时精炼与入库管线
│   ├── retrieval/              # RRF 融合、轻量 2-hop 图扩展、精排与上下文回溯服务
│   ├── rag/                    # 显式异步状态机 (FSM)、预算熔断器、引用白名单校验与拒答
│   ├── memory/                 # L1 压缩、L2 情景、L3 双时态冲突消解与四重防误伤护栏
│   ├── workers/                # PG SKIP LOCKED + LISTEN/NOTIFY 异步 Outbox Worker
│   ├── observability/          # ExecutionTrace & Span 采集落盘
│   ├── api/                    # FastAPI REST / SSE 路由与 /ui Inspector 托管
│   ├── mcp_server/             # MCP SDK v2 工具集（Streamable HTTP + stdio）
│   └── ui/                     # 零构建 4 面板轻量开发者 Inspector 调试看板
├── evals/                      # 7 组 RAG 消融与 LongMemEval-S / 双时态记忆评测套件
└── tests/                      # 单元测试、端到端 API/MCP 测试与 PostgreSQL 集成测试
```

## License

Apache-2.0. See [LICENSE](LICENSE).
