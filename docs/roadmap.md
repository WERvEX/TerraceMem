# TerraceMem 14 天 MVP 开发排期与里程碑

遵循「先跑通确定性底座与端到端闭环，再叠加高级能力与量化消融评测」原则，严格控制 P0 / P1 边界。

---

## 里程碑总览

### Phase 1（Day 1–3）：统一底座、结构化入库与混合检索核心
- [x] Day 1：项目初始化（`uv` + `pyproject.toml` + `ruff`/`mypy`/`pytest` + `docker-compose.yml` + Alembic 初始全量 DDL + 核心配置/领域模型/确定性基础组件与单测）
- [x] Day 2：实现 PostgreSQL 仓储层（文档版本幂等入库、章节树与分块事务写入、ParadeDB BM25 + `pgvector 0.8` 迭代 HNSW 单 SQL 混合召回 + Native `tsvector` 降级适配器 + Docling 解析器适配）
- [x] Day 3：实现可选「写时摘要/关键词精炼」管线、Reranker Provider（支持 API 与降级透传）与父子/邻接上下文回溯；补齐多租户与版本软删零泄漏单测及 PostgreSQL 集成测试（`tests/test_postgres_integration.py`）

### Phase 2（Day 4–6）：Agentic RAG 显式状态机与双协议接口
- [x] Day 4：实现 Typed Async FSM（`IntakeAndFilter` → `QueryRouter` → 可选 `MemoryRecall` / `OutlineNavigate` → `HybridRetrieve` → `RerankAndExpand` → `SufficiencyGate` → `CorrectiveRewrite` → `GroundedGenerate` → `CitationVerify` / `StrictAbstain`）
- [x] Day 5：实现结构化 Trace 落盘（`execution_traces` / `trace_spans`）与 FastAPI REST + SSE 流式问答接口（`/api/v1/ask/stream`）
- [x] Day 6：完善 `MCPServer` (MCP SDK v2) 完整工具实现（`kb_browse_outline` / `kb_search` / `kb_ask` / `memory_recall` / `memory_commit` / `memory_timeline`），验证 Streamable HTTP (`/mcp`) 与工具调用

### Phase 3（Day 7–10）：三层双时态记忆、零 Redis Worker 与轻量图
- [x] Day 7：实现 L1 Working Memory（Token 预算配额裁剪、`pinned_constraints` 锁定、增量滚动 `session_compactions`）
- [x] Day 8：打通 PG `FOR UPDATE SKIP LOCKED` + `LISTEN/NOTIFY` Outbox Worker，实现 L2 情景记忆归档（事件摘要 + 决策清单 + 时间/语义混合召回）
- [x] Day 9：实现 L3 语义记忆原子事实抽取、双路冲突召回（`slot_key` + Top-K 向量）、关系判定（`NOOP/MERGE/CONTRADICT/UNRELATED`）+ 四重确定性防误伤护栏 + 单事务双时态软失效与 `as_of` 时间旅行查询
- [x] Day 10：实现轻量实体/关系边抽取与防环 2-hop 递归 CTE，接入 RAG 路由器；补齐冲突演化、防误伤、防环核心单测与集成测试

### Phase 4（Day 11–14）：内置 Inspector 看板、双评测基准与开源打磨
- [x] Day 11：交付 `/ui` 零构建单页 Inspector 调试台（Engine & Document Outline、RAG Playground & Trace 瀑布流、双时态记忆时间线、评测消融图表）
- [x] Day 12：实现并跑通 RAG 7 组消融评测脚本（`evals/rag_ablation.py` + `tech_docs_hardset`，含 `B_kw_only` 对照，完成 `--dry-run` 离线基线；真实在线 LLM 大规模 RGB/MultiHop-RAG 评测可通过 `--live` 参数执行）
- [x] Day 13：实现并跑通 4 组记忆对照评测脚本（`evals/memory_benchmark.py`，支持内置双时态套件与 LongMemEval-S `--longmemeval-path` 装载，完成 `--dry-run` 离线基线）
- [x] Day 14：更新 `README.md` 与 `evals/README.md`、完善复现脚本与 GitHub Actions CI（`.github/workflows/ci.yml`）

---

## P1 后续演进清单（MVP 后按需推进）
- MinerU 重度公式/扫描 PDF 解析器插件
- OpenTelemetry GenAI Semantic Conventions OTLP 导出（无缝对接 Langfuse / Phoenix）
- 后台定时 Sleep-time 记忆合并与冲突人工确认队列
- VectorChord (`vchord` RaBitQ) 千万级分块压测适配器与多租户表分区
