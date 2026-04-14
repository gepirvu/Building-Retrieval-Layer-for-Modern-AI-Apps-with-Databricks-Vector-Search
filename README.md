# Building a Retrieval Layer for Modern AI Apps with Databricks Vector Search

A hands-on lab series that walks through building a production-ready retrieval layer for Retrieval-Augmented Generation (RAG) applications on the Databricks Lakehouse platform.

---

## What You'll Build

A full document retrieval pipeline — from raw PDFs in Unity Catalog Volumes to a queryable, re-ranked Vector Search index — the foundation of any serious RAG application.

---

## Lab Structure

```
01 - From Prompt Engineering to RAG
│   └── 1.1 - In Context Learning with AI Playground
│
02 - Preparing Data for RAG Solutions
│   └── 2.1 - Preparing Data for RAG
│
03 - Mosaic AI Vector Search
    └── 3.1 - Create Self-managed Vector Search Index
```

### Lab 1 — From Prompt Engineering to RAG
Introduces the AI Playground and demonstrates why a base LLM without retrieval produces generic or hallucinated answers on domain-specific questions. Sets the motivation for building a retrieval layer.

### Lab 2 — Preparing Data for RAG
- Reads PDF files from a Unity Catalog Volume using Spark `binaryfile` format
- Parses PDF bytes into text using PyPDF2
- Chunks text into 500-token segments (50-token overlap) using LlamaIndex `SentenceSplitter`
- Generates embeddings using the `databricks-gte-large-en` hosted endpoint via MLflow Deployments
- Stores chunks + vectors in a CDC-enabled Delta table

### Lab 3 — Mosaic AI Vector Search
- Provisions and monitors a Vector Search endpoint
- Creates a self-managed Delta Sync index backed by the embeddings table
- Performs semantic similarity search using an embedded query vector
- Re-ranks results using FlashRank (`rank-T5-flan`) for improved retrieval precision

---

## Stack

| Component | Technology |
|---|---|
| Platform | Databricks (DBR 17.3 ML) |
| Storage | Unity Catalog Volumes + Delta Lake |
| Embedding Model | `databricks-gte-large-en` (GTE-large, 1024 dims) |
| Vector Store | Databricks Mosaic AI Vector Search |
| Chunking | LlamaIndex `SentenceSplitter` |
| PDF Parsing | PyPDF2 |
| Re-Ranking | FlashRank (`rank-T5-flan`) |
| Governance | Unity Catalog (3-level namespace) |

---

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- A Vector Search endpoint provisioned
- A catalog/schema with a Volume containing PDF files
- The following libraries installed on your cluster:
  ```
  llama-index
  PyPDF2
  transformers
  flashrank
  databricks-vectorsearch
  databricks-sdk
  ```

---

## Key Concepts Covered

- **Chunking strategy** — token-based splitting with overlap to prevent information loss at boundaries
- **Self-managed vs. managed embeddings** — controlling the embedding step vs. delegating it to Databricks
- **Change Data Feed (CDC)** — required on the source Delta table for Vector Search sync
- **Two-stage retrieval** — bi-encoder (vector search) for speed, cross-encoder (re-ranker) for precision
- **Idempotent index creation** — create-or-sync pattern safe to re-run

---

## What Comes Next

This lab builds the retrieval layer. The next lab wires this pipeline into a full RAG chain — connecting retrieved context to an LLM to produce grounded, source-backed answers.

---

## Author

Built as part of a hands-on eLearning lab series on Databricks Generative AI and RAG architecture.
