<div align="center">

# ð¡ï¸ TruthLens V.o.2
### An Explainable LMTA-Driven Retrieval-Augmented Generation (RAG) System for Automated Fact Verification

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-black?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-black?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-black?style=flat-square)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-black?style=flat-square)](LICENSE)
[![Deployment](https://img.shields.io/badge/Deployment-Vercel%20%7C%20Cloud-black?style=flat-square&logo=vercel)](https://vercel.com/)

**See the Truth Behind Every Claim â Evidence-Grounded, Explainable, and Real-Time.**

[Live Architecture](#-system-architecture) â¢ [Why TruthLens is Unique](#-why-truthlens-is-unique) â¢ [V.o.2 Updates](#-vo2-massive-architecture-upgrade) â¢ [API Reference](#-api-specification) â¢ [Deployment](#-vercel-deployment)

</div>

---

## ð Executive Summary

**TruthLens** is a specialized, production-grade fact-verification platform built on an explainable Retrieval-Augmented Generation (RAG) architecture. Unlike generic conversational chatbots that are prone to hallucinations, TruthLens acts as an impartial, evidence-bound verification engine.

The system ingests real-time global news feeds, historical ground-truth archives (ISOT Reuters & PolitiFact LIAR), and curated reference dossiers. When presented with a claim, news URL, or viral WhatsApp forward, TruthLens searches its vector database, queries live web sources, reranks retrieved evidence using Cross-Encoder neural models, and synthesizes a verifiable verdict with complete source attribution using zero-shot Natural Language Inference (NLI).

```text
                                  âââââââââââââââââââââââââââââââ
                                  â   4 EXPLAINABLE VERDICTS    â
                                  âââââââââââââââââââââââââââââââ¤
                                  â  â SUPPORTED               â
                                  â  â CONTRADICTED            â
                                  â  â ï¸ MISLEADING              â
                                  â  â INSUFFICIENT EVIDENCE   â
                                  âââââââââââââââââââââââââââââââ
```

---

## ð V.o.2 Massive Architecture Upgrade

Version 0.2 marks a complete paradigm shift for TruthLens. We have stripped out brittle heuristics and replaced them with a state-of-the-art **Language Model Task Agents (LMTA)** framework powered by deterministic, local neural networks. Every component is now backed by deep learning, ensuring semantic understanding without relying on costly external LLM APIs.

### 1. Multi-Agent Orchestration (LMTA)
- **Router Agent:** Evaluates the claim's complexity and dynamically routes searches to the Local Vault, Live Web (DuckDuckGo), or Wikipedia.
- **Researcher Agent:** Aggregates, cleans, and deduplicates evidence from all activated sources simultaneously.
- **Judge Agent:** Evaluates the final evidence, scores entailment probabilities, resolves conflicts, and casts the final verdict.

### 2. Cross-Encoder Re-Ranking Pipeline (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
Standard cosine similarity often retrieves documents that share keywords but lack contextual relevance. V.o.2 introduces a dedicated MS-MARCO Cross-Encoder Reranker.
- Reranks all retrieved dense chunks against the query.
- Implements a strict `logits > -2.0` relevance cutoff. If no evidence passes the threshold, the system definitively declares **INSUFFICIENT EVIDENCE** rather than hallucinating a false fact-check.

### 3. Zero-Shot NLI Evaluator (`cross-encoder/nli-deberta-v3-base`)
We entirely removed the need for an external cloud LLM for logical entailment!
- The system natively runs a highly accurate DeBERTa v3 NLI model.
- By framing retrieved chunks as the **Premise** and the user's claim as the **Hypothesis**, the model natively outputs whether the evidence **Entails (1)**, **Contradicts (0)**, or is **Neutral (2)** to the claim with absolute deterministic precision.
- Built-in resilience: Caches models locally and utilizes a 3-attempt exponential backoff retry loop with `HF_HUB_OFFLINE` support to bypass flaky network drops on Windows environments.

---

## ð Why TruthLens is Unique

| Feature / Capability | Generic LLMs (ChatGPT / Claude) | Standard Search Engines (Google) | Conventional Fact-Check Sites | **TruthLens V.o.2 RAG Engine** |
| :--- | :--- | :--- | :--- | :--- |
| **Hallucination Resistance** | â Prone to plausible fabrication | â Index contains SEO spam & unverified blogs | â High human accuracy | **ð¡ï¸ 100% Bound to Authoritative Knowledge Vault & Strict NLI** |
| **Explainable Evidence** | â Opaque narrative generation | â Gives links, no sentence-level stance | â ï¸ Manual long-form articles | **â Fine-grained Entailment vs Contradiction chunk audit** |
| **Cost & Privacy** | â Requires costly cloud API calls | â Tracks user searches | â Free to read | **â¡ 100% Free, Local, Offline-capable AI Inference** |
| **WhatsApp / Viral Forward Cleaner**| â Confused by forward boilerplate | â Fails on raw copy-pasted chatter | â No direct tool | **ð¬ Automated Regex Stripper for clickbait & forward headers** |
| **URL Article Claim Extractor** | â ï¸ Requires manual copy-pasting | â Inundated with page ads/noise | â Manual submission | **ð 1-Click Web Scraping & Instant Claim Verification** |

---

## ðï¸ System Architecture

TruthLens employs a modular pipeline composed of 5 distinct stages:

```mermaid
graph TD
    subgraph Ingestion["1. Knowledge Ingestion & Sync"]
        A1[Live RSS & Web Search: DuckDuckGo, Wikipedia] --> B1[Document Processor & Chunker]
        A2[Historical Datasets: ISOT True, LIAR CSV] --> B1
        A3[Curated Fact Dossiers & Custom PDFs] --> B1
        B1 --> C1[all-MiniLM-L6-v2 Embedder]
        C1 --> D1[(ChromaDB Vector Store)]
    end

    subgraph UserFlow["2. Multi-Mode Input & Extraction"]
        U1[Claim / Text] --> P1[Claim Normalizer]
        U2[Web Article URL] --> P2[URL Headline Extractor] --> P1
        U3[WhatsApp Viral Forward] --> P3[Viral Boilerplate Cleaner] --> P1
    end

    subgraph RAGCore["3. Retrieval & Cross-Encoder Reranking"]
        P1 --> Q1[Query Vectorization]
        Q1 --> R1[ChromaDB HNSW Cosine Search & Live Search]
        D1 -.-> R1
        R1 --> S1[Candidate Evidence Chunks]
        S1 --> S2[MS-MARCO MiniLM L-6 Cross-Encoder Reranker]
        S2 --> S3[Threshold Filtered Evidence Dossier]
    end

    subgraph Verification["4. NLI Engine & LMTA Judge"]
        S3 --> N1{DeBERTa v3 NLI Evaluator}
        N1 -->|Probability > 0.6| N3[Stance Classifier: Entailment vs Contradiction]
        N3 --> N4[Agent Orchestrator & Consensus Analyzer]
    end

    subgraph Synthesis["5. Explainable Synthesis & Delivery"]
        N4 --> V1[Verdict Generator: SUPPORTED / CONTRADICTED / MISLEADING / INSUFFICIENT]
        V1 --> V2[Confidence & Credibility Score Computer]
        V2 --> V3[Black & White Minimalist UI & WhatsApp Exporter]
    end
```

---

## ð ï¸ API Specification

### `POST /api/verify`
The core endpoint for the LMTA Engine.

**Request Body**
```json
{
  "claim": "The moon is made of green cheese.",
  "mode": "text"
}
```

**Response Payload**
```json
{
  "verdict": "INSUFFICIENT EVIDENCE",
  "confidence_score": 0,
  "summary": "No credible evidence was found to support or contradict this claim.",
  "evidence": {
    "supporting_chunks": [],
    "contradicting_chunks": [],
    "neutral_chunks": []
  },
  "llm_provider_used": "Local NLI + LMTA Agentic Pipeline"
}
```

---
<div align="center">
  <p><i>Truth is not a narrative. It is a verifiable state.</i></p>
</div>

---

## ? Installation & Quickstart

### 1. Clone the Repository
`ash
git clone https://github.com/Prashantj44/Truth-Lens.git
cd Truth-Lens
```

### 2. Install Dependencies
`ash
pip install -r requirements.txt
```

### 3. Launch TruthLens Engine (Local FastAPI Server)
`ash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Open in Browser
Open **http://127.0.0.1:8000** in your browser to access the complete application.

---

## ?? License
This project is licensed under the MIT License  see the [LICENSE](LICENSE) file for details.
