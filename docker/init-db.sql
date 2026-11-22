-- Enable pgvector (0.8+ with halfvec & iterative index scan) and ParadeDB pg_search (BM25 + Jieba)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
