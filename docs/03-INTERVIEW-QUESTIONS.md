# 03. Interview Questions: Care Bot and the Concepts Around It

Questions an interviewer could ask about this project and the field around it, with answer outlines. Companion
documents: [01 Architecture](01-ARCHITECTURE-AND-STRATEGIES.md), [02 Tokenomics and Testing](02-TOKENOMICS-AND-AGENT-TESTING.md).
[WORKBOOK.md](WORKBOOK.md) Part 7 has a further question bank (about 40 questions) and STAR stories; this file adds
new questions and avoids repeating those answers.

**How to answer:** give the choice, the alternative, the reason, the trade-off, and the evidence. Say "I measured X" when
you did, and "I did not measure that" when you did not. Marked **[Honest]** are questions where admitting a limit is the
strong answer. Marked **[Design]** are open-ended; sketch before answering.

---

## A. Project overview and decisions

**A1. Explain the project in two minutes.**
Insurance and pharmacy assistant that answers only from cited sources. Three sources: Medicare PDFs and FDA labels via
RAG on Postgres/pgvector, patient records via an MCP server with four read-only tools. LangGraph routes each turn:
input guard (code), gate (small LLM), parallel retrieval and tool planning, fenced context, answer, output verifier
(code). Safety is enforced in code, so the model proposes and code disposes. Measured: hybrid retrieval, 37/37 live cases.

**A2. Why not just give the model all the documents?**
Context limits, cost scaling with corpus per call, quality loss in long contexts, and no per-document access control.
RAG sends only about 5 relevant chunks (about 1,250 tokens), gives page-level citations, and updates by re-ingesting.

**A3. Why RAG and tools instead of one approach?**
Fuzzy text (policy rules) suits similarity search. Exact facts (a copay of $5.00) must be exact, so a fixed SQL tool
returns them. Mixing them in one vector store would risk approximate answers to exact questions.

**A4. Why a state machine (LangGraph) rather than a ReAct agent?**
I need guaranteed ordering: guard before the model, verifier after it, parallel fetch in the middle. A loop where the
model decides the next step cannot promise that, and has unbounded cost and latency. The graph also lets me test each
branch with fake LLMs.

**A5. What is the single most important design decision?**
Enforcing safety outside the model. Prompts are requests; code is a guarantee. Identity comes from a signed cookie,
the tool schema has no `member_id`, and the verifier rejects uncited or ungrounded drafts.

**A6. What would you build differently if starting over?** [Honest]
Heading-aware chunking from day one, a held-out attack set before tuning guards, token/cost metering from the first
commit, and a Postgres checkpointer instead of in-memory state.

**A7. Why did you split into gate, planner, and answer calls?**
Narrow jobs are easier to prompt, test, and bound. The gate sees only the user's text (not retrieved data), so
injected document text cannot steer routing. Cost: extra latency of about 1 s per call.

**A8. Why Postgres for everything?**
Transactions, joins, vectors, full-text, and the audit log in one place with one set of backups and roles. The cost is
that it is not meant for billions of vectors; at our size (about 2,800 chunks) that is irrelevant.

---

## B. LLMs and model selection

**B1. How do you choose a model?**
Write an eval first, run the cheapest plausible model, move up only on measured failures, and compare cost per
*correct* answer, not price per token. Re-run when vendors ship new models.

**B2. Why Haiku-class here?**
The jobs are classification, tool selection, and rewriting provided evidence with citations. The code verifier
catches bad drafts, so I do not depend on frontier reasoning. Cheaper and faster, and the live eval passed.

**B3. When would you upgrade the model?**
If the eval shows failures on multi-step reasoning or poor answer faithfulness. I would upgrade only the answer node,
keeping gate and planner small (a cascade).

**B4. What does `temperature=0` do, and does it make output deterministic?**
It makes sampling pick the most likely token, which reduces variance. It is not a guarantee: we saw the same
borderline question classified differently across runs. So tests allow a set of acceptable outcomes.

**B5. What is a context window and why not fill it?**
The maximum tokens per call. Filling it raises cost and latency, and models use information in the middle of long
contexts less reliably. Send the minimum sufficient context.

**B6. Open-weights vs API models?**
Open: data stays with you, no per-token fee, fine-tunable, but you run GPUs and safety. API: best quality, no ops,
usage fees, data leaves your network (with contractual terms). For regulated data, hosting location and contracts
decide as much as quality.

**B7. What is structured output and why did it matter?**
Forcing the model to return a schema (our `Gate`). It removes parsing guesswork. Lesson learned: a strict schema made
the model omit fields it found irrelevant and caused user-visible errors, so only `intent` is required and the rest have
safe defaults (30/33 became 33/33).

**B8. Fine-tuning vs RAG?**
Fine-tuning changes behavior and style; it is bad at injecting facts, cannot cite, and cannot enforce per-user access.
RAG is for knowledge. They can be combined.

**B9. What are reasoning ("thinking") models and would they help?**
They generate hidden reasoning tokens before answering, which helps with math and planning and costs more and is slower.
Our tasks are not reasoning-bound, so no.

**B10. How do you handle model deprecation or an upgrade?**
Model id is configuration (`LLM_MODEL`). Run the retrieval and live evals on the new model, compare per-case results,
cost and latency, then switch. The eval suite is the safety net.

---

## C. Embeddings and vector search

**C1. What is an embedding?**
A vector representing text meaning so similar texts are close. Distance (cosine) ranks relevance.

**C2. Why `bge-base-en-v1.5`?**
Local (health text stays in process), free per call, strong English retrieval for its size, 768 dimensions as a middle
path, CPU-only ONNX runtime. Measured hit@5 of 18/20 and 19/19.

**C3. What if you used an API embedding model?**
Likely a quality bump and zero ops, but a per-token fee, a second vendor, and data leaving the network. Switching means
re-embedding the corpus and rebuilding the index, which here is minutes.

**C4. Why must queries and documents use the same model?**
Different models define different vector spaces. Distances across them are meaningless.

**C5. What is the BGE query prefix?**
BGE is trained so queries carry a short instruction and passages do not. Skipping it silently lowers quality, which is
why `embed_query` and `embed_passages` are separate functions.

**C6. How many bytes is one vector?**
768 x 4 bytes (float32) = 3,072 bytes. For one million chunks, about 3 GB raw before index overhead. That is where
quantization (halfvec, binary) or a dedicated engine appears.

**C7. Dimensions: bigger is better?**
Not automatically. More dimensions can capture more nuance but cost storage, index memory, and time. Choose by eval.

**C8. Cosine vs dot product vs Euclidean?**
For normalized vectors they give the same ranking. pgvector's `<=>` is cosine distance.

**C9. Explain HNSW in an interview-friendly way.**
A layered graph of nearby vectors. Search enters at a sparse top layer and greedily moves closer, descending to a dense
bottom layer. `ef_search` controls how many candidates are explored at query time: higher is more accurate and slower.

**C10. HNSW vs IVFFlat vs exact scan?**
Exact: perfect recall, slow at scale. IVFFlat: cluster-based, needs training data, cheaper build, recall depends on
probes. HNSW: best recall/latency, more memory, slower build. At 2,800 chunks exact would be fine; HNSW is the
realistic production habit.

**C11. What is the filtered-ANN problem?**
The index finds nearest neighbors first and applies `WHERE` after, so a selective filter can leave too few rows.
pgvector 0.8 iterative scan keeps scanning until enough rows pass.

**C12. Why hybrid search?**
Vectors blur exact terms (TrOOP, P&T, drug names); keywords miss paraphrases. Fuse both with RRF because ranks, unlike
scores, are comparable. Measured: hit@1 17/20 vs 16/20 (vector only), keyword OR was worse at 13/20.

**C13. What is RRF?**
`score(doc) = sum over lists of 1 / (k + rank)`. Documents that rank well in several lists win, with no score
normalization.

**C14. When would you add a reranker?**
When hit@N is good but precision at the top is poor. A cross-encoder rescoring the top 20-50 improves ordering at
a latency cost. Our hit@5 was 18/20 and 19/19, so no.

**C15. How do you pick chunk size?**
Start from a sensible default, inspect chunks, measure hit@k, adjust. Ours: about 1,000 characters (about 250 tokens),
sentence-aligned, per page, 200-character sentence overlap. Constraints: the embedding model's max input (512 tokens
for BGE-base), citation granularity, and answer context budget.

**C16. What are the failure modes of vector search?**
Exact numbers, rare tokens, negation ("not covered"), multi-hop questions, very short or very compound queries, and
out-of-domain queries that still return something "nearest".

**C17. What does "I don't know" depend on technically?**
A calibrated distance gate (0.47, from in-scope 0.18-0.44 vs off-topic 0.49-0.65), plus a code path that skips the
answer model when no evidence survives. It is a relevance heuristic, not a security control: 10 of 11 evasive attacks
passed it.

**C18. How would you handle a corpus of 50 million chunks?** [Design]
Quantize (halfvec/binary with rescoring), partition, consider a dedicated engine or sharded Postgres, add a reranker,
cache hot queries, monitor recall against an exact baseline sample.

---

## D. RAG quality and evaluation

**D1. Your RAG gives wrong answers. How do you debug?** [Design]
Separate retrieval failure from generation failure. Build a labeled set; check hit@k. If retrieval is wrong, inspect
chunks, query rewriting, filters, embedding choice. If retrieval is right but the answer is wrong, check the prompt,
context order and size, and faithfulness; tighten the verifier.

**D2. hit@1 vs hit@5: what does each tell you?**
hit@5 is whether the evidence reached the model; hit@1 is ranking quality. High hit@5 with low hit@1 points to a
reranker or fusion tuning.

**D3. What are MRR and nDCG?**
MRR averages 1/rank of the first correct result. nDCG weights relevance by position and handles graded relevance.

**D4. How do you measure hallucination?**
Faithfulness: is every claim supported by retrieved text? Cheap code checks (citation exists, numbers appear in
sources), then an LLM judge or human review on samples. Ours rejects drafts with invented `$`/`%`.

**D5. What is an LLM-as-judge and what are its risks?**
A model grades outputs against a rubric. Risks: bias (verbosity, position, self-preference), noise, and cost. Validate
it against human labels before trusting it, and prefer deterministic checks where possible.

**D6. Why did a metadata filter not change hit@k but you kept it?**
It is a guarantee, not a ranking tweak. Name-less queries returned the right drug's label 6% of the time without it,
100% with it. Without the category filter, label text leaked into 3 of 100 coverage passages.

**D7. How do you avoid overfitting to your eval set?**
Hold out cases you do not use while tuning; add new cases from real failures; refresh periodically. My attack set
became training data once I saw it, so I measured recall on 8 fresh attacks (6 of 8).

**D8. Multi-query retrieval: what and why?**
A compound question blurs into one embedding ("advantages and side effects"). The gate proposes up to 3 focused
sub-queries; each is re-screened (it is model output), searched in parallel, and fused by RRF.

**D9. How would you monitor quality in production?** [Design]
Outcome mix, block reasons, retrieval distance distribution, "no evidence" rate, user feedback, sampled human review,
a replay of the eval set nightly, cost and latency per node.

---

## E. Agents, MCP, tool use

**E1. What is MCP?**
Model Context Protocol: a standard way for AI applications to discover and call tools and resources on a server.
Benefits here: a separate process, one place for SQL, reuse by other clients, and a clean trust boundary.

**E2. Why a gateway between the agent and MCP?**
It overwrites `member_id` from the session, enforces role allowlists on every call, and caps calls. The model can
never name whose data to read.

**E3. Why no text-to-SQL tool?**
The model could write arbitrary or unsafe queries. Four fixed parameterized queries give a tiny attack surface and
predictable behavior. Trade-off: less flexibility.

**E4. What is a ReAct loop and its risks?**
Reason, act, observe, repeat. Flexible, but cost, latency, and behavior are unbounded unless you cap steps and validate
every tool call.

**E5. Single agent vs multi-agent?**
Multi-agent suits open-ended tasks with truly separate roles; it costs more tokens, is harder to debug, and widens the
attack surface. This task is a pipeline of narrow calls with deterministic glue.

**E6. How do you handle conversation memory safely?**
Server-side, keyed by `role:patient:conversation`, storing only screened input and verified output. A client-supplied
history could include forged assistant turns or another patient's data.

**E7. How do you bound agent cost?**
`max_tokens`, `MAX_TOOL_CALLS = 3`, a fixed graph, short-circuit refusals, rate limits, a concurrency cap, and (to add) a
daily spend breaker.

**E8. What is a reducer in LangGraph?**
The rule for merging a node's partial state update into existing state (append vs overwrite). Needed so parallel
branches can both write.

**E9. How do you make an agent testable?**
Inject the LLM, retriever, gateway, and audit sink; then fakes can force every branch for free. Keep the real-model
tests small and focused.

---

## F. Safety, security, privacy

**F1. Prompt injection: direct vs indirect.**
Direct: typed by the user. Indirect: hidden in data the model reads (a PDF, a DB row). Defense: separate data from
instructions (fence and escape), screen retrieved text, minimize capabilities, verify outputs, least privilege.

**F2. Why four layers?**
Each can fail alone: regex guard recall was 6 of 8 on fresh attacks; prompts can be talked around. Independent layers make
the system-level miss rate much lower than any layer's.

**F3. Why is the LLM gate "stricter-only"?**
So a manipulated classifier can add refusals but never grant access that code denied.

**F4. How is cross-patient access prevented?**
Identity from the signed cookie; no `member_id` in tool schemas; gateway injects it; role allowlists; fixed refusal for
questions about another person; tests assert the other patient's data never appears.

**F5. How does the bot discuss drugs without giving advice?**
Information (quote an official label for a catalog drug) is allowed; advice (personal wording, dosing, interactions
for me, "should I") is refused in code. A fixed notice is added by code, not the model.

**F6. What is fail-closed and where do you use it?**
When verification fails, return a safe canned reply rather than raw output. Used in the output guard. The audit write
is fail-open (a documented demo trade-off).

**F7. [Honest] Is it HIPAA compliant?**
No: synthetic data and simulated login. Production needs BAAs, real identity (OIDC, MFA), encryption at rest and in
transit, fail-closed audit, retention, access reviews, breach procedures, and a compliance review.

**F8. How do you avoid logging PHI?**
The audit log stores question length and an 8-character hash, never the text. Errors are redacted. Observability design
decides what is safe to keep.

**F9. What is a canary token?**
A random string in the system prompt. If it appears in output, the prompt leaked and the output is blocked.

**F10. Why strip unapproved links and images from output?**
They can exfiltrate data (a URL containing secrets that the browser fetches) or mislead users. Only approved links pass.

**F11. SQL injection defense stack?**
Bound parameters, strict input regexes, no free-form SQL, read-only role and session, statement timeout.

**F12. JWT pitfalls?**
`alg: none`, weak secrets, no expiry, storing in `localStorage`. We pin HS256, require `exp`, and use an HttpOnly cookie.
All forged-token cases are tested.

**F13. CSRF with cookies?**
SameSite=Lax, an Origin check on state-changing requests, and a CORS allow-list.

**F14. What is denial of wallet?**
Abuse that forces you to pay for model calls. Rate limits, input and output caps, and spend breakers.

**F15. [Design] Add write actions (refill request) safely.**
Separate write tools, explicit user confirmation, idempotency keys, role checks, audit before act, rate limits, no
model-chosen identifiers, and a human-approval step for risky actions.

---

## G. Tokens, cost, performance

**G1. How do you estimate cost of a turn?**
Sum over calls of `input x price_in + output x price_out`. Example in doc 02: about 3,100 input and 430 output tokens
= about $0.005 at $1/$5 per million. Retrieved chunks dominate input; output dominates cost per token.

**G2. Why are output tokens pricier and slower?**
They are generated sequentially, so cost and latency scale with them. Capping `max_tokens` and asking for concise
answers has a direct effect.

**G3. How do you cut LLM cost without hurting quality?**
Short-circuit in code, smaller models for narrow tasks, fewer or better chunks (reranker), trimmed history, prompt
caching for stable prefixes, response caching keyed by role and patient, batch API for evals. Verify each with the eval.

**G4. Prompt caching: how and when?**
A stable prefix (system prompt, tool schemas) is cached; reads cost a fraction of input price. It needs a minimum prefix
size and a stable prefix. Our prompts are small, so the saving is modest today.

**G5. What is cost per correct answer?**
Spend divided by the number of correct answers. It prevents choosing a cheap model that fails often.

**G6. Where is latency spent?**
Sequential LLM calls (gate, answer), output length, retrieval (small), the first-request model load (8-10 s). Parallelism
helps: retrieval and planning overlap. Refusals take about 0.1 s.

**G7. How would you add cost tracking?**
Collect `usage_metadata` through a LangChain callback per turn, multiply by a price table, and store numeric columns
in the audit table (no text). Dashboard daily cost, cost per turn, p95 latency, with alerts.

**G8. Capacity planning example.**
Budget $200/month at about $0.0042 blended per turn gives about 47,600 turns per month (assumptions in doc 02).

**G9. Is the embedding model free?**
No per-call fee, but it costs CPU, memory (about 210 MB model in the image), and slower first requests.

---

## H. Testing and evaluation of agents

**H1. How do you test a non-deterministic system?**
Assert behavior and invariants rather than exact text; allow sets of acceptable outcomes; check forbidden strings;
run borderline cases multiple times and use pass rates; pin temperature; version prompts and models.

**H2. Describe your test pyramid.**
Unit tests (guards, chunking), component tests with fake LLMs (graph branches), integration tests (real DB, MCP
subprocess, HTTP), retrieval evals (no LLM), live end-to-end evals (real model, cents), and production monitoring. Push
checks down to the cheap levels.

**H3. Why fake the LLM in tests?**
Speed, zero cost, determinism, and the ability to force rare branches. It tests my code, not the model.

**H4. What is a regression you caught with the live eval?**
After a prompt change the gate omitted a required field on off-topic questions, which raised a validation error and
dropped 33 cases to 30. I made only `intent` required, added tests, and re-ran: 33/33. Lesson: re-run evals after every
prompt edit.

**H5. What metrics matter for a safety-critical bot?**
Attack success rate (false answers on should-refuse cases) and over-refusal rate, tracked separately; guard recall and
false-positive rate; faithfulness. Accuracy alone hides these.

**H6. How do you test guardrails honestly?**
Held-out attacks, false-positive checks on real benign data (0 of 898 chunks flagged), and tracking block reasons
in production.

**H7. How do you test for data leaks?**
Cases whose must-not-contain list holds the other patient's data. Assert on data, not wording (an echoed name in a
refusal was harmless; leaked prescriptions are not).

**H8. What is trajectory testing?**
Asserting which tools were called and with what arguments, not only the final text. Catches right-answer-for-wrong-reason
cases.

**H9. How do you run evals in CI?** [Design]
Free tests on each push; retrieval eval on changes to chunking/embeddings; live eval nightly or before release with a
spend cap; store per-run JSON (commit, model, per-case result, tokens, cost); fail on regressions.

**H10. [Honest] How good is your eval?**
Small: 20, 19, and 37 cases. Good for regression and for finding holes, not a statistical guarantee. I would add a
held-out set, repeated runs, and an LLM-judge faithfulness check calibrated against human labels.

---

## I. Backend, frontend, deployment

**I1. Why SSE instead of WebSockets?**
Traffic is one-way server to client. SSE is plain HTTP, proxy-friendly, and auto-reconnects. Caddy needs
`flush_interval -1` or events get buffered.

**I2. Why stream progress and not tokens?**
Unverified text must never reach the user. Production could verify sentence by sentence.

**I3. Why one worker?**
Conversation memory and rate limits are in process memory. Scaling out needs a Postgres checkpointer and Redis.

**I4. Why one origin in production (Caddy)?**
No CORS preflights, cookies just work, automatic HTTPS. The Vite proxy mirrors it in development.

**I5. Why bake the embedding model into the Docker image?**
Startup never waits on a 210 MB download, and deployments are reproducible. Cost: a larger image.

**I6. What is a reducer pattern in the React chat state?**
A pure `(state, event) -> state` function handles stream events; testable and free of race-prone mutation.

**I7. Idempotent ingestion?**
SHA-256 per file; unchanged files are skipped; each document is stored in one transaction so a failed run never leaves
a half-ingested document.

**I8. How do you roll back a bad deploy?** [Design]
Tagged images, `docker compose up` with the previous tag, backward-compatible migrations (expand then contract),
health checks (`/healthz`, `/api/health`), and the eval suite as a pre-release gate.

---

## J. Scenario and system-design questions

**J1. [Design] 1M members, multiple insurers.**
Tenant-aware metadata filters in SQL, per-tenant document versioning, auth via OIDC and claims-based scopes, read
replicas, a Postgres checkpointer and Redis, queue and cap LLM concurrency, per-tenant rate limits and budgets,
observability, and a compliance review.

**J2. [Design] Documents update monthly. What changes?**
Scheduled ingestion keyed by hash, `effective_date` and `verified_on` metadata, answer citations showing document
version, a regression eval after each ingestion, and removal of superseded chunks.

**J3. [Design] Add support for Spanish.**
A multilingual embedding model (e.g., bge-m3 or multilingual-e5), re-embed the corpus, extend input-guard patterns
beyond English, translate fixed replies, add bilingual eval cases. Today non-English input is rejected by design.

**J4. [Design] Cut the answer latency in half.**
Merge gate and planner, a smaller model for them, cache stable prefixes, trim retrieved context, cap output tokens, and
overlap work. Measure each change with the eval, since merging calls hurts isolation.

**J5. [Design] A new attack bypasses the input guard. What do you do?**
Add it as a failing test, fix at the right layer (often the output verifier or privileges rather than another regex),
re-measure false positives on real data, deploy, and review block reasons in the audit log.

**J6. [Design] The bot says "I don't know" too often.**
Check the distance gate and retrieval hit@k for those questions, chunking, query rewriting and multi-query, whether the
verifier is too strict (derived math, percentages), and the over-refusal rate in the eval.

**J7. [Design] Replace pgvector with a dedicated vector DB.**
Only with a measured reason (scale, latency, features). Keep Postgres as the source of truth, sync vectors, preserve
metadata filters and access control, and re-run the retrieval eval. Cost: a second system to secure and keep consistent.

---

## K. Trick questions

* **"Temperature 0 means the model always answers the same?"** No (see B4).
* **"More chunks always improve answers?"** No: cost rises, noise rises, and the model may ignore the middle.
* **"The distance threshold is your safety layer, right?"** No (see C17).
* **"Fine-tune on the PDFs instead of RAG?"** Poor at facts, no citations, no per-user access control.
* **"Why not let the model call SQL directly?"** Unsafe and unpredictable (E3).
* **"Embeddings are encrypted, so the vector DB is safe?"** No. Embeddings can leak information about the text and can be
  partly inverted. Treat them as sensitive as the source.
* **"Cheaper model, so lower cost per correct answer?"** Not necessarily (G5).
* **"Your eval passed 37/37, so it is safe?"** No (H10). It shows no known regression on known cases.

---

## L. Numbers and phrases to have ready

0.47 distance gate (in-scope 0.18-0.44, off-topic 0.49-0.65) · 768 dimensions, 3,072 bytes per vector · HNSW + iterative
scan · RRF · about 1,000-char chunks, about 250 tokens · TOP_K 5 · hit@1 17/20 hybrid vs 16/20 vector, OR keyword 13/20 ·
labels hit@5 19/19, right-drug share 6% to 100% with filter · injection screen 0 of 898 false positives · fresh attack
recall 6 of 8 · live eval 37/37 · 221 backend + 15 frontend tests · refusals about 0.1 s and free · answers 2.5-6 s ·
illustrative turn about 3,100 in / 430 out tokens, about $0.005 · three Haiku calls per full turn · four guard layers ·
stricter-only gate · fail closed · identity from the cookie, never from the model.

---

## M. Questions to ask the interviewer

* How do you evaluate LLM features before and after release?
* How is spend monitored and capped per feature or tenant?
* What is your policy on sending sensitive data to model providers?
* How do you decide between a larger model and better retrieval?
* Who owns the golden dataset, and how does it get refreshed?
