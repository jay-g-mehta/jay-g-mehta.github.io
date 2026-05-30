---
title: "Building Agentic Systems with Memory, RAG, Knowledge Base, Vector DB, Knowledge Graph & Context Stores"
description: "Applied agentic AI — step-by-step guide to building AI agents with RAG, vector databases, knowledge graphs, and memory systems, from stateless LLM to full agentic architecture."
---

# Building Agentic Systems with Memory, RAG, Knowledge Base, Vector DB, Knowledge Graph & Context Stores

> This post explains how to build AI agents integrated with different types of memory and databases, touching when to use what and how.


> To explain this better, we'll walk through building a **Legal Research Agent** — an AI assistant that helps users navigate constitutional law, statutes, and case law. We start with a bare-bones agent and progressively add memory and knowledge layers as we hit real limitations. Each addition is motivated by a problem, not theory.

> **Why this use case helps understanding different agent's memory and database needs?** Constitutional law has deeply interconnected documents (amendments reference articles, cases cite other cases, statutes override prior statutes). This naturally demands every type of knowledge storage — vector databases, knowledge graphs, structured databases, and context stores. The architecture patterns transfer to any domain with complex, interrelated documentation.

---

## The Agent We're Building

A **Legal Research Agent** that can:

- Answer questions about constitutional provisions and statutes
- Find relevant case law for a legal question
- Explain how amendments relate to and override prior law
- Track which jurisdiction applies
- Remember the user's research thread across sessions

**What we're NOT building:** A lawyer. This agent retrieves and explains — it doesn't give legal advice. The architecture is the focus, not the legal domain.

---

## Stage 1: The Stateless Agent (Just an LLM)

We start with the simplest possible setup: a system prompt and an LLM.

```
User Query → [System Prompt + LLM] → Response
```

**What it can do:**
- Answer general questions about well-known constitutional concepts from its training data
- Explain basic legal terms
- Have a conversation within a single session

**What it can't do:**
- Reference specific constitutional text accurately (hallucination risk)
- Find relevant case law (not in training data, or outdated)
- Know which jurisdiction the user cares about
- Remember anything from prior sessions
- Follow complex chains: "Which amendment overrode this article, and which cases interpreted that amendment?"

**The limitations that motivate what comes next:**

| Problem | Root Cause | Solution (next sections) |
|---------|-----------|--------------------------|
| Hallucinated legal text | LLM's training data is limited on domain, compressed and lossy | Knowledge base + RAG |
| Can't find specific cases | Case law is vast and has a knowledge cutoff | Vector database + RAG |
| Can't trace legal relationships | Flat text doesn't capture structure | Knowledge graph |
| Forgets user's research thread | No persistence between sessions | Context store / memory |
| Doesn't know user's jurisdiction | No user state | Structured database |

---

## Stage 2: Adding Legal Knowledge — Knowledge Base + RAG

### The problem

The user asks: *"What does the 14th Amendment say about equal protection?"*

The stateless agent might paraphrase from training data — but it could hallucinate clauses, miss nuances, or cite outdated interpretations. We need the agent to reference **actual source text**.

### The solution: Retrieval-Augmented Generation (RAG)

Instead of relying on the LLM's memory, agentic system retrieve relevant documents at query time and inject them into the prompt to LLM as context.

```
User Query
    │
    ▼
EMBED the query (convert to vector)
    │
    ▼
SEARCH vector database (find similar document chunks)
    │
    ▼
RETRIEVE top-k relevant chunks
    │
    ▼
ASSEMBLE context (system prompt + retrieved chunks + user query)
    │
    ▼
LLM generates response grounded in retrieved text
```

### Knowledge Base

A **knowledge base** is the organized collection of domain information that your agent can search and retrieve from. It's not a single technology — it's the combination of your source documents, how they're processed (chunked, embedded), and where they're stored. Think of it as your agent's reference library: curated, indexed, and searchable. Without it, the agent only knows what the LLM memorized during training.

**A vector database is not the only retrieval layer.** While we focus on vector search (embeddings + similarity) here because it handles natural language queries well, knowledge bases can also use:
- **Full-text search** (Elasticsearch, OpenSearch) — keyword/BM25 matching, great when users search by specific terms like case names or statute numbers
- **Hybrid search** — combines vector similarity + keyword search for the best of both (many production systems use this)
- **Metadata filtering** — retrieve by structured attributes (date, jurisdiction, document type) without any semantic search
- **Graph databases** — store knowledge as entities and relationships (covered in Stage 4)

For legal text, where users often search by exact case names or statute numbers, **hybrid search (vector + keyword)** is often the strongest choice. We use vector search as the primary example because it's the most broadly applicable pattern.

### Building the knowledge base

**What goes in:**
- Constitutional text (articles, amendments, preamble)
- Federal and state statutes
- Case law opinions (Supreme Court, appellate courts)
- Legal commentary and analysis

**The ingestion pipeline:**

1. **Collect** source documents (public domain — constitutions and case law are freely available)
2. **Chunk** them into passages
3. **Embed** each chunk using an embedding model → dense vector (these vectors exist so that at query time, you can perform cosine similarity between the query vector and stored chunk vectors to find the most semantically relevant matches)
4. **Store** vectors + metadata in a vector database
5. **Index** for fast similarity search

**What is chunking?**

Chunking is splitting large documents into smaller passages that can be individually embedded and retrieved. You can't embed an entire 50-page court opinion as one vector — it would lose all specificity (the vector becomes an average of everything, matching nothing well). Additionally, most embedding models have a maximum input length (typically 512–8192 tokens) — text beyond that limit is simply truncated and lost. So chunking is both a quality decision and a technical constraint.

The chunk size is a tradeoff:
- Too small (a single sentence) → loses context, retrieval returns fragments that don't make sense alone
- Too large (entire documents) → embedding becomes too generic, retrieval loses precision
- Sweet spot for most use cases: 200–500 tokens per chunk, with overlap between adjacent chunks

**What is embedding (in this context)?**

Embedding is converting each text chunk into a dense vector (array of numbers) using an embedding model. This vector captures the semantic meaning of the chunk — so chunks about similar topics end up near each other in vector space. At query time, the user's question is also embedded, and you find chunks whose vectors are closest to the query vector. (See the [LLM Anatomy post](llm-anatomy-components-concepts) for a deeper explanation of embeddings and dimensions.)

**What is indexing?**

Indexing is organizing the stored vectors for fast retrieval. Without an index, finding the most similar vectors requires comparing your query against every single stored vector (brute-force) — too slow at scale. Index structures like HNSW (Hierarchical Navigable Small World) or IVF (Inverted File Index) create shortcuts that let you find approximate nearest neighbors in milliseconds, even across millions of vectors. The tradeoff: indexes are approximate — they trade a small amount of recall accuracy for massive speed gains.

### Chunking strategy matters

Legal text has structure — sections, clauses, subsections. Naive chunking (split every 500 tokens) breaks logical units apart.

Better approaches for legal text:
- **Section-aware chunking:** Split at article/section/clause boundaries
- **Hierarchical chunking:** Keep parent context (e.g., "Amendment 14, Section 1" stays together)
- **Overlapping chunks:** Include a few sentences of overlap so context isn't lost at boundaries
- **Metadata preservation:** Tag each chunk with its source (amendment number, case name, court, date)

### Why to use RAG with KnowledgeBase over stuffing everything in the prompt?

- Just like the US Constitution alone is ~8,000 words, and if we add case law and statutes — millions of pages; generally all domains have large documentation size. 
- Context windows are limited and expensive. You can't fit everything.
- **Context pollution:** Stuffing irrelevant text into the prompt actively degrades output quality. The LLM gets confused by noise and may anchor on the wrong passages. Less but relevant context beats more but unfocused context.
- RAG retrieves only what's relevant to *this specific query*.

### Why RAG over fine-tuning for this use case?

- Legal text changes (new cases, new interpretations). RAG uses live data; fine-tuning bakes in a snapshot.
- You need citations. RAG gives you the source chunk; fine-tuning gives you a paraphrase with no provenance.
- Updating is cheap: re-embed new documents. No retraining needed.

### Architecture at this stage:

```
┌──────────────────────────────────────────────┐
│              Legal Research Agent            │
│                                              │
│  User Query                                  │
│      │                                       │
│      ├──→ Embedding Model ──→ Query Vector   │
│      │                                       │
│      │    Vector Database                    │
│      │    ┌─────────────────────────┐        │
│      └──→ │ Constitutional text     │        │
│           │ Statutes                │        │
│           │ Case law opinions       │        │
│           │ Legal commentary        │        │
│           └─────────────────────────┘        │
│                     │                        │
│                     ▼                        │
│           Top-k relevant chunks              │
│                     │                        │
│                     ▼                        │
│           [System Prompt + Chunks + Query]   │
│                     │                        │
│                     ▼                        │
│                   LLM ──→ Response           │
│                                              │
└──────────────────────────────────────────────┘
```

### Practical takeaways for agentic architectures:

- **RAG is your first upgrade** for any agent that needs domain knowledge beyond the LLM's training data.
- **Chunking is an architectural decision**, not a preprocessing detail. Bad chunks = bad retrieval = bad answers.
- **Metadata is as important as the embedding.** Tagging chunks with source, date, and jurisdiction lets you filter before searching — critical for legal accuracy.
- **Embedding model choice matters.** General-purpose embeddings may not capture legal nuance. Test retrieval quality on your domain before committing.

### Troubleshooting when RAG doesn't perform well

RAG can fail silently — the agent returns confident-sounding answers grounded in the wrong context. The key principle: **test retrieval and generation independently**, not just the final answer. They're two separate systems that fail for different reasons.

| Evaluation Layer | What you're testing | How to test | LLM involved? |
|-----------------|--------------------|-----------|----|
| **Retrieval only** | Did the top-k chunks contain the relevant information? | Run queries against a labeled test set, measure recall/precision of returned chunks | No |
| **Generation only** | Given the correct chunks, does the LLM produce the right answer? | Manually provide known-good chunks, evaluate LLM output | Yes |
| **End-to-end** | Full pipeline: query → retrieval → generation → answer | Run test queries through the complete system, compare to expected answers | Yes |

If you only evaluate end-to-end, you can't tell whether a bad response was caused by retrieving the wrong chunks or by the LLM misinterpreting the right chunks.

### Symptom → Root Cause → Troubleshooting → Fix

**Retrieval problems (wrong chunks come back):**

| Symptom | Possible Root Cause | How to Reproduce / Troubleshoot | Approaches to Fix |
|---------|--------------------|---------------------------------|-------------------|
| Answer is vaguely related but misses the specific point | Chunks too large — vector is an average of too many ideas | Run retrieval in isolation; inspect returned chunks manually for specificity | Reduce chunk size; use section-aware chunking |
| Answer lacks context or feels fragmented | Chunks too small — each piece is meaningless alone | Inspect returned chunks; check if they make sense without surrounding text | Increase chunk size; add overlapping windows; include parent section as metadata |
| Results are semantically relevant but from the wrong category, source, or version | No metadata filtering — search returns any match regardless of source attributes | Query with a known filter (e.g., specific source or date range); check if results span multiple unintended categories | Add metadata (source, category, date, version); filter before or alongside vector search |
| Relevant document exists but isn't retrieved | Embedding model doesn't capture domain-specific distinctions | Embed a known query and its expected chunk; measure cosine similarity — if low, the model doesn't "see" them as related | Try a domain-tuned embedding model; test with multiple embedding models; add synonyms or summaries alongside raw text |
| User's phrasing doesn't match document language | Query-document vocabulary mismatch | Compare user query phrasing to actual document text; check if rephrasing the query improves retrieval | Add query expansion (rephrase before embedding); embed document summaries alongside raw text; use hybrid search (vector + keyword) |
| Retrieval is extremely slow or returns obviously wrong results despite correct embeddings | Index missing, misconfigured, or stale — search is falling back to brute-force or using an outdated index | Check vector DB logs for full-scan warnings; compare retrieval latency against expected (milliseconds vs. seconds); verify index was rebuilt after bulk inserts | Rebuild the index after large ingestion batches; verify index type matches your data size (HNSW for <10M vectors, IVF for larger); check that new vectors were added to the index, not just stored |

**Generation problems (right chunks, wrong answer):**

| Symptom | Possible Root Cause | How to Reproduce / Troubleshoot | Approaches to Fix |
|---------|--------------------|---------------------------------|-------------------|
| Answer contradicts the retrieved context | LLM ignores context and generates from training data | Provide the same chunks manually in a test prompt; check if LLM still ignores them | Strengthen system prompt: "Answer ONLY from provided context"; reduce model temperature; use a model with better instruction following |
| Answer mixes information from multiple unrelated chunks | Context pollution — too many chunks, some irrelevant | Reduce top-k to 1-2 and check if answer improves | Add a re-ranking step (score chunks for relevance before injecting); reduce top-k; filter by metadata |
| Answer is partially correct but misses key detail | Relevant information spans a chunk boundary | Check if the answer's missing detail is in an adjacent chunk that wasn't retrieved | Use overlapping chunks; retrieve adjacent chunks when confidence is low; increase chunk overlap |
| Answer is correct but lacks citation | Agent doesn't pass through source metadata | Check if chunk metadata (source, section, date) is included in the prompt template | Include metadata in the prompt alongside each chunk; instruct the LLM to cite sources |

**Systemic problems:**

| Symptom | Possible Root Cause | How to Reproduce / Troubleshoot | Approaches to Fix |
|---------|--------------------|---------------------------------|-------------------|
| Agent doesn't know about recently added or updated information | Knowledge base is stale — new documents not embedded | Ask about a known recent addition; check if it exists in the vector DB | Build an update pipeline with clear triggers (new document published → embed → index) |
| Retrieval quality degrades over time | Embedding model updated or document corpus grew without re-evaluation | Compare retrieval metrics (recall/precision) over time against a fixed test set | Schedule periodic evaluation; re-embed when embedding model changes; monitor retrieval scores |
| Can't tell if the system is improving or degrading | No evaluation framework | — | Create a test set of questions with known correct answers; measure retrieval recall + answer accuracy on every change |

---

## Stage 3: Adding Structure — Relational Data for Deterministic Lookups

### The problem

The user asks: *"What court decided Brown v. Board of Education, and what year?"*

This isn't a semantic search problem — it's a fact lookup. The answer is structured data: case name, court, year, outcome. Searching a vector database for this is overkill and unreliable (embeddings are fuzzy; facts need precision).

### The solution: Structured database via tool calls

Some knowledge is best stored in tables, not vectors:

| Data | Why structured DB | Example |
|------|------------------|---------|
| Case metadata | Exact lookups by name, date, court | `SELECT * FROM cases WHERE name = 'Brown v. Board'` |
| Jurisdiction hierarchy | Tree structure (federal → state → local) | Parent-child relationships |
| User profiles | Deterministic attributes | Preferred jurisdiction, research history |
| Statute status | Boolean/enum (active, repealed, amended) | `WHERE status = 'active'` |

### How the agent decides what to do

Before any lookup happens, the LLM must decide *how* to answer the query. This is the routing/reasoning step:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  AGENT                                                  │
│                                                         │
│  Passes query + available tool descriptions to LLM      │
│                                                         │
│      │                                                  │
│      ▼                                                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │  LLM (reasoning step)                           │    │
│  │                                                 │    │
│  │  "What kind of question is this?"               │    │
│  │                                                 │    │
│  │  Returns decision:                              │    │
│  │    • Which tool(s) to call                      │    │
│  │    • With what parameters                       │    │
│  └─────────────────────────────────────────────────┘    │
│      │                                                  │
│      ▼                                                  │
│  Agent receives decision, executes tool(s):             │
│                                                         │
│      ├──→ Vector DB (RAG)         ← semantic question   │
│      │                                                  │
│      ├──→ Structured DB (tool)    ← fact lookup         │
│      │                                                  │
│      └──→ Both (sequentially)     ← needs facts + text  │
│                                                         │
│      │                                                  │
│      ▼                                                  │
│  Agent passes tool results back to LLM                  │
│      │                                                  │
│      ▼                                                  │
│  LLM generates final response using retrieved data      │
│                                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Response to user
```

**How does the LLM know which tool to pick?**

The LLM receives a list of available tools with descriptions. It matches the query's intent to the right tool based on those descriptions. For example:

- Tool: `search_knowledge_base` — description: *"Semantic search over legal text. Use for open-ended questions about legal concepts, interpretations, or explanations."*
- Tool: `get_case_metadata` — description: *"Exact lookup of case facts. Use when the user asks for specific dates, courts, outcomes, or status."*

Clear, distinct tool descriptions are what make routing work. If descriptions are vague or overlapping, the LLM will pick the wrong tool.

**Examples of routing decisions:**

| User query | LLM reasoning | Tool(s) called |
|-----------|--------------|----------------|
| "What does the 14th Amendment say about equal protection?" | Open-ended, needs text explanation | `search_knowledge_base` (RAG) |
| "What year was Brown v. Board decided?" | Fact lookup — specific date | `get_case_metadata` (structured DB) |
| "What did the court say in Brown v. Board and why does it matter?" | Needs both: fact (which court) + explanation (significance) | `get_case_metadata` → then `search_knowledge_base` |

### How the agent uses structured data

The agent calls tools (functions) that query the database:

```
User: "Is the Voting Rights Act still active?"

Agent reasoning:
  → This is a status lookup, not a semantic question
  → Call tool: get_statute_status("Voting Rights Act")
  → DB returns: { status: "active", last_amended: "2006", notes: "Section 4(b) struck down in Shelby County v. Holder (2013)" }
  → Agent formats response with citation
```

### When to use structured DB vs. vector DB:

| Use structured DB when... | Use vector DB when... |
|--------------------------|----------------------|
| You need exact answers (dates, names, status) | You need semantically similar content |
| The query maps to a known schema | The query is natural language / open-ended |
| Precision matters more than recall | Recall matters more than precision |
| Data is tabular or relational | Data is unstructured text |

### Practical takeaways for agentic architectures:

- **Not everything is a RAG problem.** Agents need both fuzzy retrieval (vector) and precise lookup (structured). Conflating them degrades both.
- **Tool calls are how agents access structured data.** The LLM decides *when* to query the DB; the tool provides the interface.
- **Schema design matters for agents.** Design your tables around the questions your agent will ask, not just how humans would query them.

---

## Stage 4: Adding Relationships — Knowledge Graphs for Legal Reasoning

### The problem

The user asks: *"Which amendments modified the original voting provisions in the Constitution, and which Supreme Court cases interpreted those amendments?"*

This requires **multi-hop reasoning** across relationships:
1. Find original voting provisions (Article I, Article II)
2. Find amendments that modified them (15th, 19th, 24th, 26th)
3. Find cases that interpreted those amendments

RAG alone struggles here. A vector search for "voting amendments" might return relevant chunks, but it won't *trace the chain* of relationships. You'd need multiple retrieval rounds and hope the LLM connects the dots.

### The solution: Knowledge graph

A knowledge graph stores **entities** and **relationships** explicitly:

```
[14th Amendment] ──overrides──→ [Article I, Section 2]
[14th Amendment] ──interpreted_by──→ [Brown v. Board of Education]
[Brown v. Board] ──cites──→ [Plessy v. Ferguson]
[Brown v. Board] ──overrules──→ [Plessy v. Ferguson]
[15th Amendment] ──grants──→ [Voting Rights (race)]
[19th Amendment] ──grants──→ [Voting Rights (sex)]
[Shelby County v. Holder] ──strikes_down──→ [Voting Rights Act, Section 4(b)]
```

### Modelling legal knowledge as a graph

**Entities (nodes):**
- Constitutional articles and sections
- Amendments
- Statutes
- Court cases
- Legal concepts (e.g., "due process," "equal protection")
- Courts
- Jurisdictions

**Relationships (edges):**
- `amends`, `overrides`, `repeals`
- `interprets`, `cites`, `overrules`
- `grants`, `restricts`
- `decided_by` (case → court)
- `applies_to` (statute → jurisdiction)

### How the agent uses the graph

```
User: "What's the chain of authority for voting age?"
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  AGENT                                                       │
│                                                              │
│  Passes query + tool descriptions to LLM                     │
│      │                                                       │
│      ▼                                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  LLM (reasoning step 1)                                │  │
│  │                                                        │  │
│  │  Input: system prompt + tool descriptions +            │  │
│  │         user query                                     │  │
│  │                                                        │  │
│  │  "This asks about a chain of authority — I need to     │  │
│  │   traverse relationships. Use knowledge graph."        │  │
│  │                                                        │  │
│  │  Decision: call traverse_graph(entity="Voting Age",    │  │
│  │            relationships=["amended_by","interpreted_by"])││
│  └────────────────────────────────────────────────────────┘  │
│      │                                                       │
│      ▼                                                       │
│  Agent executes: Knowledge Graph tool                        │
│      │                                                       │
│      │  Returns:                                             │
│      │    Article I (original)                               │
│      │      ← amended_by ← 26th Amendment (1971)             │
│      │      ← interpreted_by ← Oregon v. Mitchell (1970)     │
│      │                                                       │
│      ▼                                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  LLM (reasoning step 2)                                │  │
│  │                                                        │  │
│  │  Input: system prompt + tool descriptions +            │  │
│  │         user query + step 1 reasoning +                │  │
│  │         step 1 tool call + graph results               │  │
│  │         (full accumulated context)                     │  │
│  │                                                        │  │
│  │  "I have the structure, but I need the full text of    │  │
│  │   the 26th Amendment and the case opinion for detail." │  │
│  │                                                        │  │
│  │  Decision: call search_knowledge_base("26th Amendment  │  │
│  │            full text") AND search_knowledge_base(      │  │
│  │            "Oregon v. Mitchell opinion")               │  │
│  └────────────────────────────────────────────────────────┘  │
│      │                                                       │
│      ▼                                                       │
│  Agent executes: Vector DB (RAG) — two lookups               │
│      │                                                       │
│      │  Returns: full text chunks for both                   │
│      │                                                       │
│      ▼                                                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  LLM (generation step)                                 │  │
│  │                                                        │  │
│  │  Input: system prompt + user query +                   │  │
│  │         all prior reasoning + all tool results         │  │
│  │         (graph structure + retrieved text)             │  │
│  │                                                        │  │
│  │  Generates: coherent narrative with citations          │  │
│  └────────────────────────────────────────────────────────┘  │
│      │                                                       │
│      ▼                                                       │
│  Response to user                                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**The loop pattern:** The agent doesn't execute everything in one shot. The LLM reasons, the agent executes a tool, results come back, the LLM reasons again and may call another tool. This **reason → act → observe → reason** loop continues until the LLM has enough information to generate a final answer. In this example, it took two loops: graph traversal first, then RAG for detail.

> **How does the LLM know to reason in multiple hops?** This is driven by the system prompt, tool descriptions, and few-shot examples — not magic. We cover this in depth in [LLM Reasoning in Agentic Systems](llm-reasoning-agentic-systems).

### Graph + RAG hybrid (GraphRAG)

The most powerful pattern combines both:
1. **Graph traversal** finds the relevant entities and their relationships (the structure)
2. **RAG retrieval** pulls the full text of those entities (the content)
3. **LLM** synthesizes both into a coherent answer

This is exactly what the flow above demonstrates — the graph tells the agent *what* to look for, and RAG provides the *content* of those things.

### Why graphs complement vector search

| Vector DB (RAG) | Knowledge Graph |
|----------------|-----------------|
| "Find text similar to my question" | "Trace the chain of relationships from A to B" |
| Fuzzy, semantic | Precise, structural |
| Good for: "Explain equal protection" | Good for: "What overrides what?" |
| Returns chunks of text | Returns entities and connections |
| Single-hop retrieval | Multi-hop traversal |

### How do you know you need a knowledge graph?

It's not magic — it's requirements analysis. Before building, examine your domain's relationships and expected query patterns:

**The signal:** Your users will ask questions that can't be answered from a single document chunk — they require tracing connections across multiple entities.

**How to discover this:**

- **Talk to domain experts.** Ask: "When you research this topic, what do you need to trace or follow?" If they describe chains, dependencies, or hierarchies — that's a graph.
- **Look at your documents.** If they're full of cross-references (citations, "see also," "as amended by," "supersedes") — the relationships between documents are as important as the documents themselves.
- **Prototype with RAG first.** Build the RAG-only version, test with real queries, and observe where it fails. If it consistently gives incomplete answers to questions that require connecting multiple pieces — that's your signal.
- **Categorize your expected queries.** If a significant portion require more than one retrieval step where each step depends on the previous result — flat retrieval won't cut it.

**Decision framework:**

| If your domain has... | You probably need... |
|----------------------|---------------------|
| Documents that reference other documents | Knowledge graph |
| Hierarchies (parent/child, jurisdiction nesting) | Knowledge graph |
| "What overrides/amends/cites what?" queries | Knowledge graph |
| Conditional paths ("what applies given X and Y?") | Knowledge graph |
| Mostly "explain this concept" queries | RAG is sufficient |
| Mostly fact lookups | Structured DB is sufficient |

### Practical takeaways for agentic architectures:

- **Use a knowledge graph when relationships are the answer**, not just the content. "What overrides what" and "what cites what" are graph problems.
- **Graphs and RAG are complementary, not competing.** Graph gives structure; RAG gives content. Use both.
- **Multi-hop reasoning is where graphs shine.** If your agent needs to traverse chains (A amended B, which was interpreted by C, which was overruled by D), a graph makes this a single query instead of multiple RAG rounds with uncertain results.
- **Graph construction is the hard part.** Extracting entities and relationships from legal text can be done manually (high quality, slow) or with LLMs (faster, needs validation). Plan for this investment.

---

## Stage 5: Adding Memory — Context Stores for Continuity

### The problem

The user has been researching voting rights for 30 minutes across multiple questions. They ask: *"Based on everything we've discussed, summarize the key amendments."*

The agent has no idea what "everything we've discussed" means — each request is independent. There's no memory of the research thread.

### The solution: Persistent memory across sessions

Agents need multiple types of memory, each stored differently:

### Types of agent memory:

**Short-term memory:**

| Memory Type | What it stores | Lifespan | Storage |
|-------------|---------------|----------|---------|
| **Working memory** | Current conversation turns, intermediate reasoning | Single session | In-memory / session store (Redis) |

**Long-term memory:**

| Memory Type | What it stores | Lifespan | Storage |
|-------------|---------------|----------|---------|
| **Episodic** | Past interactions, research threads, what happened and what worked | Permanent | Vector DB (searchable by similarity) |
| **Semantic** | Facts, knowledge, user preferences, domain rules | Until invalidated | Structured DB, document store, knowledge graph |
| **Procedural** | Multi-step workflows, routines the agent executes reliably | Permanent | System prompts, fine-tuned weights, or workflow definitions |

Short-term memory is the agent's scratchpad for the current session. Long-term memory is everything that persists across sessions — and it comes in three distinct types, each serving a different purpose.

Each type answers a different question the agent faces:
- **Working (short-term):** "What just happened in this conversation?"
- **Episodic:** "Have I encountered something like this before, and what happened?"
- **Semantic:** "What facts and knowledge apply here?"
- **Procedural:** "How do I execute this task?"

### Short-term memory: Conversation buffer

The simplest memory — keep the last N turns in the prompt:

```
System Prompt
+ Retrieved context (RAG)
+ Conversation history (last 10 turns)
+ Current user query
──→ LLM
```

**Challenge:** Context window fills up. A 30-turn legal research session with retrieved chunks can easily exceed token limits.

**Solutions:**
- **Sliding window:** Keep only the last N turns, drop older ones
- **Summarization:** Periodically summarize older turns into a compressed paragraph
- **Selective retention:** Keep turns that contain key decisions or facts, drop filler

### Semantic memory: User profile and domain knowledge

```json
{
  "user_id": "researcher_42",
  "jurisdiction": "US Federal",
  "research_topic": "14th Amendment equal protection",
  "preferred_detail_level": "detailed with citations",
  "past_sessions": ["session_001", "session_002"]
}
```

Semantic memory stores facts and knowledge that apply broadly — not tied to a specific interaction but to what the agent *knows*. This includes user preferences, domain rules, and accumulated knowledge.

Stored in a structured database or document store. Loaded at session start. Updated as the agent learns about the user or domain.

### Episodic memory: Past interactions as searchable knowledge

The most powerful pattern — embed past interactions and make them retrievable:

1. After each session, summarize the key Q&A pairs
2. Embed those summaries
3. Store in a vector database (separate from the legal knowledge base)
4. On new queries, search episodic memory: "Have we discussed something similar before?"

```
User: "Remind me what we found about the commerce clause last week"

Agent:
  → Search episodic memory for "commerce clause"
  → Retrieve: summary of session_002 where user explored commerce clause cases
  → Use as context for response
```

### Procedural memory: Learned workflows

Procedural memory is often overlooked but essential for agents that handle repetitive, multi-step tasks. It's the agent's learned skills — routines it can execute without reasoning through every step from scratch each time.

**In our legal agent:** When a user asks for a case summary, the agent doesn't need to reason about the format every time. It has a learned procedure: retrieve case metadata → pull the opinion text → extract holding → identify cited precedents → format as structured summary. This sequence becomes a cached routine.

**How procedural memory is implemented:**
- **System prompt instructions:** Explicit step-by-step workflows written into the prompt. The procedure lives inside the LLM's context — it reads the steps and follows them each time. Simplest and most common approach.
- **Fine-tuned behavior:** If the agent is fine-tuned on examples of correct multi-step execution, the procedure becomes part of its weights — baked into the model itself. The LLM executes the pattern without needing explicit instructions in the prompt.
- **Workflow definitions:** External workflow engines (state machines, DAGs, orchestration frameworks like LangGraph or Step Functions) that define the procedure outside the LLM entirely. The LLM doesn't decide the sequence — the orchestrator does. It calls the LLM at specific steps for reasoning or generation, but the overall flow is controlled by code. These execute at runtime: the orchestrator receives the user's request, determines which workflow applies, and steps through the defined sequence — invoking the LLM, tools, or databases at each node.
- **Distilled from experience:** Patterns that appear repeatedly in episodic memory can be extracted and codified into any of the above forms — promoted from "something the agent figured out" to "something the agent always does."

**Where does procedural memory live?**

| Implementation | Where it lives | Who controls the flow |
|---------------|---------------|----------------------|
| System prompt | In the LLM's context window | LLM follows instructions |
| Fine-tuned weights | Inside the model parameters | LLM executes implicitly |
| Workflow definitions | External code / orchestrator | Code controls flow, LLM is called at steps |
| Distilled patterns | Depends on where they're codified | Varies |

**Why it matters:** Without procedural memory, the agent re-reasons every routine task from scratch — slower, more expensive, and less consistent. With it, reasoning is freed up for novel situations while routine tasks execute reliably.

### Context assembly: What actually gets sent to the LLM

With multiple memory types and knowledge sources, the agent must **assemble context** carefully:

```
┌─────────────────────────────────────────────┐
│         CONTEXT ASSEMBLY                    │
│                                             │
│  Priority 1: System prompt (always)         │
│  Priority 2: User profile (jurisdiction,    │
│              preferences)                   │
│  Priority 3: Retrieved knowledge (RAG)      │
│  Priority 4: Graph relationships            │
│  Priority 5: Relevant episodic memory       │
│  Priority 6: Recent conversation turns      │
│                                             │
│  Total must fit within context window       │
│  → Rank by relevance to current query       │
│  → Truncate lowest priority if over budget  │
│                                             │
└─────────────────────────────────────────────┘
```

### Practical takeaways for agentic architectures:

- **Memory is not one thing.** Different types of memory serve different purposes and need different storage.
- **Context assembly is an architectural decision.** Your agent needs a strategy for what to include when the window is full — not just "stuff everything in."
- **Episodic memory makes agents feel intelligent.** Remembering past interactions is what separates a tool from an assistant.
- **Summarization is your compression algorithm.** When context overflows, summarize old turns rather than dropping them entirely.

---

## Stage 6: The Full Architecture

Here's what our Legal Research Agent looks like with all layers in place:

```
┌──────────────────────────────────────────────────────────────────┐
│                    LEGAL RESEARCH AGENT                          │
│                                                                  │
│  User Query                                                      │
│      │                                                           │
│      ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  CONTEXT ASSEMBLY                                       │     │
│  │                                                         │     │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐    │     │
│  │  │ Vector DB   │  │ Knowledge    │  │ Structured   │    │     │
│  │  │ (RAG)       │  │ Graph        │  │ DB           │    │     │
│  │  │             │  │              │  │              │    │     │
│  │  │ • Const.    │  │ • Amends     │  │ • Case meta  │    │     │
│  │  │   text      │  │ • Cites      │  │ • User prefs │    │     │
│  │  │ • Case law  │  │ • Overrides  │  │ • Statute    │    │     │
│  │  │ • Statutes  │  │ • Interprets │  │   status     │    │     │
│  │  │ • Commentary│  │ • Grants     │  │ • Juris.     │    │     │
│  │  └─────────────┘  └──────────────┘  └──────────────┘    │     │
│  │                                                         │     │
│  │  ┌─────────────┐  ┌──────────────┐                      │     │
│  │  │ Episodic    │  │ Session      │                      │     │
│  │  │ Memory      │  │ Memory       │                      │     │
│  │  │             │  │ (short-term) │                      │     │
│  │  │ • Past Q&A  │  │ • Current    │                      │     │
│  │  │ • Research  │  │   turns      │                      │     │
│  │  │   threads   │  │ • Working    │                      │     │
│  │  │             │  │   context    │                      │     │
│  │  └─────────────┘  └──────────────┘                      │     │
│  │                                                         │     │
│  │  Assembled Context (ranked, filtered, within budget)    │     │
│  └─────────────────────────────────────────────────────────┘     │
│      │                                                           │
│      ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  LLM                                                    │     │
│  │  (System Prompt + Assembled Context + Query)            │     │
│  └─────────────────────────────────────────────────────────┘     │
│      │                                                           │
│      ▼                                                           │
│  Response (with citations and source references)                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Decision framework: Which store for which need

| Need | Store | Why |
|------|-------|-----|
| "Find text about equal protection" | Vector DB (RAG) | Semantic similarity search over unstructured text |
| "What year was this case decided?" | Structured DB | Exact lookup, tabular data |
| "What did the 14th Amendment override?" | Knowledge Graph | Relationship traversal |
| "What did we discuss last session?" | Episodic Memory (Vector DB) | Semantic search over past interactions |
| "What jurisdiction does this user prefer?" | Structured DB | Deterministic user attribute |
| "Keep track of this conversation" | Session Store | Short-lived, in-memory |

### Common pitfalls:

- **Using RAG for everything.** Not all questions are semantic search problems. Exact lookups and relationship traversals need their own systems.
- **Ignoring metadata in retrieval.** Searching all legal text without filtering by jurisdiction or date returns noise. Always filter before (or alongside) semantic search.
- **Stale embeddings.** When source documents change (new cases, amended statutes), your embeddings must be re-generated. Build a refresh pipeline.
- **Context window stuffing.** More context ≠ better answers. Irrelevant context actively degrades LLM output. Be selective.
- **No citation trail.** In legal (and most professional) domains, the user needs to verify the source. Always pass through chunk metadata so the agent can cite its sources.

---

## Keeping It Fresh: Maintenance and Updates

Legal knowledge changes — new cases are decided, statutes are amended, interpretations evolve.

### Update strategies by store type:

| Store | Update trigger | Strategy |
|-------|---------------|----------|
| Vector DB | New document published | Embed and index incrementally; re-chunk if structure changed |
| Knowledge Graph | New case cites/overrules existing | Add new edges; mark overruled nodes |
| Structured DB | Case decided, statute amended | Update record; maintain audit trail |
| Episodic Memory | Each session ends | Summarize and embed session; no updates to past episodes |

### Monitoring retrieval quality:

- Track "retrieval relevance" — are the top-k chunks actually useful for the query?
- Log cases where the agent's answer contradicts retrieved text (hallucination despite context)
- Periodically test with known questions to catch drift

---

## Key Takeaways

1. **Start simple, add layers when you hit real limitations.** Don't build all five storage systems on day one.
2. **Each storage type solves a different problem.** Vector DB for semantic search, structured DB for exact lookups, knowledge graph for relationships, context stores for memory.
3. **Context assembly is the orchestration layer.** The agent's intelligence partly lives in *how* it decides what context to include.
4. **RAG is necessary but not sufficient.** Most production agents need RAG + at least one other storage type.
5. **The patterns transfer.** Swap "constitutional law" for any domain with complex, interrelated documentation — the architecture remains the same.

---

[← Back to home](/)
