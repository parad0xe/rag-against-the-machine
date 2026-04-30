_This project has been created as part of the 42 curriculum by nlallema._

# RAG against the machine

## Description

RAG against the machine is a project focused on building a fully functional Retrieval-Augmented Generation (RAG) system from scratch. Designed to answer technical questions about a specific codebase (the vLLM repository), the system ingests source code and documentation, indexes it for semantic and keyword-based retrieval, and leverages a local Large Language Model (Qwen/Qwen3-0.6B) to generate, source-grounded answers without hallucinations.

## Table of contents

- [Instructions](#instructions)
  - [Installation](#installation)
  - [Example usage](#example-usage)
- [Submissions](#submissions)
- [System architecture](#system-architecture)
- [Chunking strategy](#chunking-strategy)
- [Retrieval method](#retrieval-method)
- [Performance analysis](#performance-analysis)
- [Design decisions](#design-decisions)
- [Challenges faced](#challenges-faced)
- [Resources](#resources)

## Instructions

### Installation

To install the project dependencies using `uv` (as required), use the provided Makefile:

```bash
make install
```

### Example Usage

The system provides a Command-Line Interface (CLI) built with Python Fire. Here are clear examples of running the different components of the system:

```bash
# 1. Index the repository (creates BM25 and ChromaDB databases)
uv run python -m src index --extensions "md,py" --max_chunk_size 2000 --semantic

# 2. Check the manifest statistics
uv run python -m src manifest_stats --all

# 3. Search for a single query (retrieval only)
uv run python -m src search "How to configure OpenAI server?" --k 5

# 4. Answer a single query (retrieval + LLM generation)
uv run python -m src answer "How to configure OpenAI server?" --k 5 --details

# 5. Process a full dataset (search only)
uv run python -m src search_dataset --dataset_file_path data/datasets/UnansweredQuestions/dataset_docs_public.json --save_dir_path data/output/

# 6. Process and answer a full dataset
uv run python -m src answer_dataset --dataset_file_path data/datasets/AnsweredQuestions/dataset_code_public.json --save_dir_path data/output/

# 7. Evaluate search results against ground truth
uv run python -m src evaluate --dataset_file_path data/datasets/AnsweredQuestions/dataset_docs_public.json --predictions_file_path data/output/search_results.json --ks "1,3,5,10"
```

## Submissions

The expected files for the project are structured as follows:

```txt
.
├── src/
│   ├── application/                        # Ports and Services (Business logic)
│   ├── domain/                             # Pydantic Models and Exceptions
│   ├── infrastructure/                     # LLM Engines, Vector Stores, Loaders
│   ├── presentation/                       # CLI entrypoints (Fire)
│   ├── factories/                          # Dependency Injection
│   ├── config.py                           # Global settings
│   └── __main__.py                         # Application entrypoint
├── data/
│   ├── output/                             # Generation results
│   ├── processed/                          # Indexing results
│   └── datasets/
│       ├── AnsweredQuestions/
│       │   ├── dataset_code_public.json
│       │   └── dataset_docs_public.json
│       └── UnansweredQuestions/
│           ├── dataset_code_public.json
│           └── dataset_docs_public.json
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

## System architecture

The project strictly follows a **clean architecture** (ports and adapters) combined with **solid principles** to ensure modularity, testability, and SOC. It implements the 5 core systems required by the subject:

- **Knowledge base ingestion system**: processes the repository, applies language-aware chunking, and builds the search indices.
- **Retrieval system**: performs hybrid search (semantic + keyword) to return the top-k most relevant code snippets and docs.
- **Answer generation system**: uses the llm to stream natural language answers strictly grounded in the retrieved context.
- **Evaluation system**: calculates recall@k metrics by comparing retrieved sources against ground truth datasets.
- **Command-line interface**: provides a robust cli using python fire with progress bars and error handling.

The architecture is divided into 4 main layers:

| Layer | Name | Role |
|-------|----------------|-----------------------------------------------------------|
| 4 | presentation | cli interface (fire, rich), handling user inputs and outputs |
| 3 | application | core rag workflow, orchestration, services, and interfaces |
| 2 | domain | pydantic data models (`MinimalSource`, `RagDataset`), exceptions |
| 1 | infrastructure | adapters (huggingface, chromadb, bm25, readers) |

**The rag pipeline flow:**
1. **Translator**: translates user queries into english if necessary (`helsinki-nlp`).
2. **Expander (semantic only)**: extracts technical keywords from the query using the llm.
3. **Retriever**: queries the active index stores (bm25 only, or bm25 + chromadb if semantic is enabled) concurrently using `ThreadPoolExecutor`.
4. **Fusion**: merges results using reciprocal rank fusion (RRF).
5. **Reranking (semantic only)**: rescores the top results via a cross-encoder (`bge-reranker-base`).
6. **Generator**: streams the final answer using `qwen3-0.6b` based strictly on the retrieved context.

## Chunking strategy

Implementation of a language-aware chunking strategy using Langchain's `RecursiveCharacterTextSplitter`.

- **Logic**: the system automatically detects the file extension (Python, Markdown, C++, etc.) and applies appropriate code separators to maintain logical code blocks and syntax integrity.
- **Limit**: the default maximum chunk size is set to 2000 characters (configurable via CLI).
- **Overlap**: a soft overlap of 200 characters is applied to ensure no context is lost between consecutive chunks.

## Retrieval method

The system utilizes a **Hybrid retrieval** mechanism:

- **Keyword matching (BM25)**: excellent for finding specific variable names, exact function definitions, and code syntax within the repository.
- **Semantic search (ChromaDB)**: uses `all-MiniLM-L6-v2` embeddings to understand the conceptual meaning of a query, perfect for documentation and high-level architecture questions.
- **Reciprocal rank fusion (RRF)**: Merges the results from both stores based on their respective ranks and assigned weights (BM25: 0.6, Chroma: 0.4).
- **Cross-encoder re-ranking**: A secondary neural network (`BAAI/bge-reranker-base`) evaluates the relevance of the query against the top retrieved chunks to provide the final highly accurate top-k selection.

## Performance analysis

- **Indexing time**: optimized using a manifest file to track file hashes, avoiding re-indexing unchanged files. A full repository index takes well under the 5-minute threshold.
- **Recall@k**: evaluated using a 5% character overlap threshold. The hybrid approach allows the system to correspond to the mandatory thresholds (80% for docs, 50% for code).
- **Latency**: generative responses are streamed in real-time. Background tasks run concurrently to maintain a fast throughput. However, enabling the semantic mode loads several additional models (like the query expander and cross-encoder), which noticeably slows down the overall runtime.

## Design decisions

- **Strict data validation**: used `pydantic` across all domain models to ensure robust data validation, type safety, and consistent serialization.
- **Dependency injection pattern**: heavyweight resources (causal llms, translation engines, embedding models) are instantiated exactly once via a centralized `RetrieverFactory`. They are then injected into application services, mitigating memory leaks, and preventing vram saturation.
- **Interface segregation**: designed explicit protocols (ports) for each service. This decoupled the business logic from specific library implementations, allowing for effortless testing and future model swapping.

## Challenges faced

- **LLM hallucinations**: small models like Qwen 0.6B tend to hallucinate or ignore formatting rules. Solved by implementing strict system prompts, few-shot prompting for keyword extraction, and forcing the model to rely solely on the injected context.
- **Memory management**: loading a causal lm, a seq2seq translator, an embedding model, and a cross-encoder simultaneously requires careful VRAM management. Solved by utilizing `torch.inference_mode()`.
- **Overfitting in queries**: query expansion initially caused the model to bleed context from previous examples. Solved by using contrastive multi-shot prompting.

## Resources

### AI usage

The development of this project was supported by AI for:
- Assisting in the architectural refactoring.
- Generating boilerplate pydantic schemas.
- Drafting and formatting README.

### Links

- [Web: Understanding Okapi BM25 — Document Ranking algorithm](https://medium.com/@readwith_emma/understanding-okapi-bm25-document-ranking-algorithm-70d81adab001)
- [Web: RAG introduction](https://blog.stephane-robert.info/docs/developper/programmation/python/rag-introduction/)
- [Web: Implementing RAG in LangChain with Chroma](https://medium.com/@callumjmac/implementing-rag-in-langchain-with-chroma-a-step-by-step-guide-16fc21815339)
- [Web: HuggingFace Text Generation Strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)
- [Web: Indexing in langchain](https://www.geeksforgeeks.org/artificial-intelligence/indexing-in-langchain/)

- [Web: False positive flake E203 - PEP8](https://peps.python.org/pep-0008/#whitespace-in-expressions-and-statements)
