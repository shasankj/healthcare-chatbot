# 01. Architecture, Strategies, and Model Choices

How Care Bot is built, the alternatives that exist across the industry, and why each choice was made here.
Companion documents: [02 Tokenomics and Agent Testing](02-TOKENOMICS-AND-AGENT-TESTING.md),
[03 Interview Questions](03-INTERVIEW-QUESTIONS.md). The long-form record of problems and measurements is
[WORKBOOK.md](WORKBOOK.md).

> Numbers marked *(measured)* come from the repo's eval scripts. Numbers marked *(illustrative)* are my estimates for
> teaching. Model prices and model lineups change often: verify them on the vendor's pricing page before quoting.

---

## 1. The system in one page

Care Bot answers insurance and pharmacy questions from three sources and refuses everything else.

| Source | Holds | Access path |
|---|---|---|
| Medicare coverage PDFs (4) | Rules: prior auth, step therapy, appeals, deductibles | RAG: hybrid vector + keyword search |
| FDA drug labels (13) | Labeled side effects, warnings, uses | RAG with metadata filter on the drug |
| PostgreSQL tables | One patient's prescriptions, copays, plan | MCP server, 4 fixed read-only tools |

**Request path (one chat turn):**

```
Browser (React) --POST /chat, SSE stream, signed cookie--> FastAPI
  FastAPI: rate limit, Origin check, audit
    LangGraph:
      input_guard (code)  --blocked--> fixed reply
      gate (Haiku)        --refuse---> fixed reply
      retrieve (RAG)  || plan_tools (Haiku -> MCP)      (parallel)
      build_context (fence + injection screen)  --no evidence--> "I don't know"
      answer (Haiku, strict prompt)
      output_guard (code: citations, numbers, links, canary)
      finalize
PostgreSQL + pgvector (data, vectors, keyword index, audit log)
```

Design principle: **the model proposes, code disposes.** Everything that matters for safety (who you are, what you may
read, whether an answer is cited and grounded) is decided by deterministic code outside the model.

---

## 2. Architecture strategies that exist, and where Care Bot sits

### 2.1 How do you give an LLM knowledge it was not trained on?

| Strategy | What it is | Strengths | Weaknesses | Use when |
|---|---|---|---|---|
| **Prompt stuffing** | Paste all documents into the prompt | Trivial; no infra | Context limits, cost grows with corpus, "lost in the middle" quality loss | Corpus is small (a few pages) |
| **RAG** | Retrieve the relevant passages at question time, put only those in the prompt | Fresh data, citations, cheap, access control per document | Quality bounded by retrieval; more moving parts | Knowledge changes, you need sources. **Our choice for documents** |
| **Fine-tuning** | Train the model on your data | Style, format, domain vocabulary | Poor at facts, hard to update, cannot cite, cannot enforce per-user access | Behavior/style change, not knowledge |
| **Long-context + caching** | Put the whole corpus in a huge context and cache it | Simple; good for one fixed corpus | Per-call cost still scales, no per-user filtering | Small, stable, shared corpus |
| **Tool use / function calling** | Model asks code to fetch exact data | Exactness, live data, auth in code | Needs tool design and safety | Structured data. **Our choice for patient records** |
| **Text-to-SQL** | Model writes SQL against your DB | Flexible | Model can write unsafe or wrong queries | Analytics for trusted users. **Rejected here** |
| **Knowledge graph / GraphRAG** | Retrieve entities and relations | Multi-hop questions | Costly to build and maintain | Heavily relational corpora |

**Why we used RAG + tools together:** policy text is fuzzy and long (RAG fits); copays and tiers are exact numbers
that must never be "approximately retrieved" (tools fit).

### 2.2 How is the application orchestrated?

| Pattern | Description | Trade-off |
|---|---|---|
| **Single prompt** | One LLM call does everything | Cheapest, but guards cannot sit between steps |
| **Chain / pipeline** | Fixed sequence of calls | Predictable; no branching |
| **ReAct agent loop** | Model loops: think, call tool, observe, repeat | Flexible; unpredictable cost, latency, and behavior; hard to guarantee safety |
| **Workflow / state machine (LangGraph)** | Explicit graph with conditional edges and typed state | Predictable, testable, guards between nodes. **Our choice** |
| **Multi-agent** | Several specialized agents talk to each other | Powerful for open-ended work; expensive, hard to debug, more attack surface |
| **Router + specialists** | Cheap classifier routes to a handler | Good cost control; we use this idea in the gate |

**Why a state machine:** a healthcare assistant needs *guaranteed* ordering (guard before model, verify after model,
parallel fetch in the middle). A free-running agent loop cannot promise that. We also bounded tool calls
(`MAX_TOOL_CALLS = 3`), so cost per turn has a ceiling.

### 2.3 How is retrieval done?

| Strategy | Notes |
|---|---|
| **Vector (dense) search** | Embeds meaning; blurry on exact terms (acronyms, drug names) |
| **Keyword (sparse) search, BM25 / Postgres full-text** | Exact terms; blind to paraphrase |
| **Hybrid** | Run both and fuse. **Our choice.** |
| **Fusion method: RRF vs weighted sum** | RRF fuses by *rank*, so incompatible score scales do not matter. We use RRF (`score = Σ 1/(k + rank)`) |
| **Reranking (cross-encoder)** | Second-stage model rescoring the top 20-50. Not used: hit@5 was already 18/20 and 19/19 *(measured)*. The first upgrade if recall dropped |
| **Query rewriting / HyDE** | Rewrite or hallucinate an answer to embed. We rewrite in the gate (standalone question) |
| **Multi-query** | Split a compound question into focused sub-queries. **Used** (max 3, each re-screened) |
| **Metadata filtering** | Exact SQL filters (category, drug). **Used**; moved "right drug's label" from 6% to 100% *(measured)* |
| **Parent-document / small-to-big** | Search small chunks, return the surrounding larger one. Upgrade path |
| **Contextual retrieval** | Prepend a generated summary of the document to each chunk before embedding. Upgrade path |
| **Late interaction (ColBERT)** | Token-level matching; higher quality, higher storage |

### 2.4 Where do vectors live?

| Option | Pros | Cons | Pick when |
|---|---|---|---|
| **Postgres + pgvector** | One database for SQL, vectors, full-text, audit, backups, transactions | Not tuned for billions of vectors | You already run Postgres, < ~tens of millions of vectors. **Our choice** |
| Dedicated vector DB (Pinecone, Qdrant, Weaviate, Milvus) | Scale, managed ops, advanced filtering | A second system to secure, sync, and pay for | Very large scale or vector-heavy product |
| Search engine (OpenSearch/Elasticsearch) | Mature hybrid search, analyzers | Heavy to run | You already run it |
| In-memory (FAISS, Chroma) | Fast prototyping | No persistence/ACL story | Notebooks and prototypes |

**Index type:** HNSW (graph, strong recall/latency, no training) vs IVFFlat (clusters, needs training, cheaper to
build) vs exact scan (perfect recall, slow). We use **HNSW with cosine distance**, plus pgvector's **iterative scan**
because HNSW applies `WHERE` filters *after* finding neighbors and would otherwise return too few rows.

### 2.5 How is the tool layer exposed?

| Option | Notes |
|---|---|
| In-process Python functions | Simplest; no trust boundary |
| **MCP server (stdio child process)** | Standard protocol, separate process, one place for SQL, reusable by other clients. **Our choice** |
| MCP over HTTP / remote | Needed for multi-host; adds auth and network concerns |
| REST microservice | Familiar; you write the glue yourself |

A gateway in front of MCP overwrites the `member_id`, so **the model never sees or chooses whose data to read**.

### 2.6 How is the answer delivered to the browser?

| Option | Notes |
|---|---|
| Polling | Simple, wasteful, laggy |
| **SSE (Server-Sent Events)** | One-way server push over plain HTTP; auto-reconnect; proxy-friendly. **Our choice** |
| WebSockets | Two-way; more infra; unnecessary for one-way progress |
| Token streaming | Typewriter effect; but unverified text would reach the user, so **we stream progress events, not tokens** |

### 2.7 How is safety enforced?

| Layer type | Example | Reliability |
|---|---|---|
| Prompt instructions | "Never give medical advice" | Best effort. The model can be talked out of it |
| **Code guards before the model** | Input guard: normalization, emergency/injection/advice patterns | Deterministic; recall is partial (6 of 8 fresh attacks) *(measured)* |
| **Data fencing** | Retrieved text wrapped as untrusted, injection-like passages dropped | Defends against indirect injection |
| **Code guards after the model** | Output verifier: citation exists, `$` and `%` appear in sources, approved links only, canary | Deterministic; fail closed |
| LLM-as-judge guard | A second model reviews output | Flexible; adds cost, latency, and its own failure modes |
| Managed guardrail services | Vendor-provided classifiers | Fast to adopt; less control |
| Least privilege | Read-only DB role, fixed SQL, role allowlists, INSERT-only audit | Limits the blast radius when everything else fails |

**Defense in depth:** each layer assumes the previous one can fail.

### 2.8 Identity, sessions, deployment

| Topic | Choice | Alternatives |
|---|---|---|
| Session | Signed JWT (HS256) in HttpOnly SameSite=Lax cookie | `localStorage` token (readable by XSS), server-side session store, OAuth/OIDC (the real-world replacement) |
| CSRF | SameSite + Origin check + CORS allow-list | CSRF tokens |
| Conversation memory | Server-side, keyed `role:patient:conversation` | Client sends history (forgeable) |
| Deploy | One container behind Caddy, single origin | Kubernetes, serverless, PaaS |
| Scaling limit | One worker (memory and rate limits are in-process) | Postgres checkpointer + Redis for horizontal scale |
| Migrations | Numbered idempotent SQL in one transaction | Alembic, ORM auto-migrate |

---

## 3. LLM choices: what exists and why Haiku 4.5

### 3.1 The landscape (conceptual tiers)

| Tier | Typical models (verify current names) | Good at | Trade-off |
|---|---|---|---|
| **Small / fast / cheap** | Claude Haiku, GPT mini-class, Gemini Flash-class | Classification, extraction, grounded Q&A with good context, routing | Weaker on hard reasoning and long multi-step plans |
| **Mid / balanced** | Claude Sonnet, GPT flagship-class, Gemini Pro-class | Coding, analysis, agentic tool use | Several times the price of small |
| **Large / frontier** | Claude Opus-class, top reasoning models | Hard reasoning, ambiguous tasks | Highest cost and latency |
| **Reasoning / "thinking" models** | Extended-thinking variants | Math, planning, multi-step logic | Many hidden reasoning tokens; slow; costly |
| **Open-weights** | Llama, Mistral, Qwen, Gemma families | Data stays on your hardware; no per-token fee; fine-tunable | You run GPUs and safety; usually behind frontier on quality |
| **On-device / tiny** | Small quantized models | Offline, private | Limited capability |

### 3.2 Why Claude Haiku 4.5 here

1. **The model's job is narrow.** The gate classifies, the planner picks among 4 tools, the answerer *rewrites
   provided evidence with citations*. These are small-model tasks. Reasoning power is not the bottleneck: retrieval
   quality and guards are.
2. **Cost and latency.** Three calls per turn. At small-tier prices the whole turn costs a fraction of a cent *(illustrative,
   see doc 02)*; at frontier prices it would be many times more for no measurable gain on this task.
3. **The safety design reduces the need for a strong model.** Code verifies the output, so a weaker draft is rejected rather than
   trusted. You pay for a bigger model when you rely on it; here we do not.
4. **Structured output works well** for the gate schema, and `temperature=0` gives repeatable behavior.
5. **Provider fit.** One vendor SDK (`langchain-anthropic`), one key, one bill.

**What would change the choice:** if live eval showed failures on hard multi-hop questions, upgrade only the *answer*
node to a mid-tier model and keep the gate/planner on the small one. The graph already isolates each call, so this is
a one-line change in `make_llm`. This is the "model cascade" pattern (see section 5).

**Settings and why:** `temperature=0` (consistency for grounded answers), `max_tokens=700` (hard cost ceiling and
forces concise answers), `max_retries=2` (transient errors), `timeout=60`.

### 3.3 How to choose a model in general

1. Write the eval first (see doc 02). 2. Run the cheapest candidate. 3. Move up only on measured failures.
4. Compare on **cost per correct answer**, not price per token. 5. Re-run when vendors ship new models.

---

## 4. Embedding models: what exists and why bge-base-en-v1.5

### 4.1 What an embedding model does

It maps text to a vector so that texts with similar meaning are close (cosine distance). **The same model must embed
documents and queries**: vectors from different models live in different spaces. Changing the model means
re-embedding the whole corpus and rebuilding the index.

### 4.2 The landscape

| Category | Examples (verify current) | Dimensions | Notes |
|---|---|---|---|
| **Open, small** | `bge-small-en`, `all-MiniLM-L6-v2`, `e5-small` | 384 | Fast, tiny, lower quality; fine for prototypes |
| **Open, base** | **`bge-base-en-v1.5`**, `e5-base`, `gte-base` | 768 | Good quality/size balance. **Ours** |
| **Open, large** | `bge-large`, `e5-large`, `gte-large`, `nomic-embed` | 1024+ | Higher quality, ~3x compute and storage |
| **API, proprietary** | OpenAI `text-embedding-3-small/large`, Cohere Embed, Voyage AI, Google embeddings | 1024-3072 (some adjustable) | Top quality, zero ops, per-token fee, data leaves your network. Anthropic does not offer its own embeddings API and points to partners such as Voyage |
| **Multilingual** | `bge-m3`, `multilingual-e5`, Cohere multilingual | 768-1024 | Needed if users are not English-only. We reject non-English input, so not needed |
| **Domain-specific** | BioBERT/PubMedBERT-style, clinical embedders | 768 | Can help on medical vocabulary; check maintenance and licensing |
| **Sparse / learned sparse** | BM25 (classic), SPLADE | very high, sparse | The keyword half of hybrid search |
| **Late interaction** | ColBERT | per-token vectors | Best quality, heavy storage |

Useful tools for choosing: the public **MTEB leaderboard** (compare on *retrieval* tasks, and distrust scores from
models trained on the test sets), then **your own eval** because leaderboards rarely match your domain.

### 4.3 Why bge-base-en-v1.5, locally, 768 dimensions

| Reason | Detail |
|---|---|
| **Privacy** | Health-related text never leaves the process. No third party sees queries |
| **Cost** | Zero per-call cost. Ingestion of about 2,800 chunks and every query are free apart from CPU |
| **Quality** | Strong English retrieval quality for its size; hit@5 of 18/20 and 19/19 *(measured)* |
| **Operational simplicity** | `fastembed` runs ONNX on CPU, no GPU, no PyTorch. Model baked into the Docker image (about 210 MB) so startup never downloads |
| **Right-sized dimensions** | 768 is a middle path: noticeably better than 384 on harder queries, about half the storage of 1536 |
| **No vendor lock** | Anthropic has no embeddings API, and using an extra paid API would add a second vendor and send data out |

**Costs we accepted:** about 2x the storage and compute of the 384-d small model; the model lives in the app image and
uses memory; first request after startup is slower (model load, 8-10 s).

**Query vs passage prefix:** BGE models are trained so that queries get a short instruction prefix and passages get
none. `embed_query()` and `embed_passages()` exist separately for that reason. Forgetting this silently lowers quality.

### 4.4 Storage maths

`vector(768)` stored as float32 = 768 x 4 bytes = **3,072 bytes per vector**. About 2,800 chunks = roughly 8.6 MB
before the HNSW graph overhead. A million chunks would be about 3 GB of raw vectors, which is where you start
thinking about quantization (halfvec, binary) or a dedicated engine.

### 4.5 When to change the embedding model

* Retrieval eval drops, or new documents use new vocabulary.
* You need other languages.
* You need longer chunks than the model's max input (512 tokens for BGE-base).
* Re-embedding is cheap here (minutes of CPU for the whole corpus), so switching is a low-regret experiment: add a
  new column or table, ingest, run `eval_retrieval`, compare hit@k.

---

## 5. Cross-cutting strategies worth knowing (and where we used them)

| Strategy | Meaning | In Care Bot |
|---|---|---|
| **Model cascade / routing** | Cheap model first, escalate only if needed | Gate is a small classifier; refusals skip the answer model |
| **Short-circuit before the LLM** | Deterministic code answers/refuses first | Input guard; "no evidence" path; refusals cost $0 and take about 0.1 s |
| **Parallel fan-out** | Independent steps run concurrently | Retrieval and tool planning run in parallel |
| **Prompt caching** | Reuse a repeated prompt prefix at a discount | Not enabled. Our prompts are small; a candidate once prompts grow (doc 02) |
| **Idempotent ingestion** | Re-runs do nothing if the file hash is unchanged | SHA-256 per source; one transaction per document |
| **Dependency injection** | Pass in LLM, retriever, gateway, audit sink | Whole agent is testable with fakes at zero cost |
| **Fail closed** | On any doubt, send a safe canned reply | Output verifier failures become "I don't know" |
| **Observability** | Metrics, traces, audit | Audit log (hash only), admin dashboard, status events. No token metering yet (doc 02) |
| **Calibrated thresholds** | Choose cutoffs from data | Distance gate 0.47 from observed in-scope 0.18-0.44 vs off-topic 0.49-0.65 |

---

## 6. Honest limitations (say these in an interview before they ask)

* Regex input guard caught 6 of 8 fresh attacks: it is a speed bump, not a wall.
* The distance gate is a relevance heuristic, not a security control.
* Memory, rate limits, and counters are in process: one worker, no horizontal scale.
* Audit write fails open (acceptable for a demo; a regulated system should fail closed).
* Demo sign-in is simulated; real systems use OIDC with MFA.
* Evaluation sets are small (20, 19, 37 cases): good for regression, not statistical proof.
* No token or cost metering in the code yet (see doc 02 for how to add it).
