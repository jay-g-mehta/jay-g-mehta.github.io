---
title: "LLM Anatomy & Concepts for Agentic System Builders"
description: "Applied AI deep dive into LLM internals — tokenization, embeddings, parameters, quantization, fine-tuning, and LoRA — with practical takeaways for building applied agentic AI systems."
---

# LLM Anatomy & Concepts for Agentic System Builders

> When you build AI agents or applications on top of LLMs, you're making decisions at every step: which model to use, how much context you can fit, how to retrieve relevant knowledge, and how to deploy affordably. These concepts are the ones that drive those decisions. Understanding them turns you from someone who *uses* LLMs into someone who *architects* with them.

**What goes on behind the scenes when you interact with an LLM:**

```
    Raw Text
        │
        ▼
    ╔═════════════════════════════════════╗
    ║         Tokenizer/Preprocessor      ║
    ║                                     ║
    ║   ▼                                 ║
    ║   TOKENIZATION (BPE)                ║
    ║   ▼                                 ║
    ║   Token IDs [5765, 2065, 374, ...]  ║
    ║                                     ║
    ╚═════════════════════════════════════╝
        │
        ▼
    ╔═════════════════════════════════════╗
    ║           LLM Model                 ║
    ║                                     ║
    ║   ▼                                 ║
    ║   EMBEDDING → Dense Vectors         ║
    ║   ▼                                 ║
    ║   TRANSFORMER LAYERS (attention,    ║
    ║   feed-forward, billions of params) ║
    ║   ▼                                 ║
    ║   Output Token Probabilities        ║
    ║                                     ║
    ╚═════════════════════════════════════╝
        │
        ▼
    Generated Text
```

---

## Anatomy of an LLM

An LLM is not a single monolithic thing — it's a pipeline of distinct components, each with a specific job:

| Component | What it does | Inside the model? |
|-----------|-------------|-------------------|
| **Tokenizer** | Splits raw text into subword tokens and maps them to integer IDs | No — deterministic algorithm, shipped alongside the model |
| **Embedding Layer** | Converts token IDs into dense vectors (high-dimensional representations) | Yes — first layer of the neural network |
| **Transformer Layers** | Process vectors through self-attention and feed-forward networks to build contextual understanding | Yes — the bulk of the model's parameters live here |
| **Output Head** | Projects final vectors into vocabulary-sized probabilities for next-token prediction | Yes — last layer of the neural network |

### Key distinctions:

- **Tokenizer vs. Model:** The tokenizer is a deterministic algorithm (trained once to learn merge rules, but has no neural network weights). It's bundled with the model because each model has its own vocabulary, but it's not part of the neural network.
- **Embedding vs. Transformer Layers:** The embedding layer is a simple lookup table (token ID → vector). The transformer layers are where reasoning, pattern matching, and knowledge retrieval happen.
- **Parameters live in:** the embedding layer, transformer layers, and output head. These are the billions of numbers that get trained.

### What "size" refers to:

When someone says "a 70B model," they mean 70 billion parameters spread across the embedding layer, transformer layers, and output head. The tokenizer adds negligible overhead.

---

## LLM Concepts in Modelling, Customizing, Optimizing, and at Inference

When you're building an AI agent or application, here's how these concepts show up at each stage:

**Modelling (creating the base model):**

1. **Pre-trained Data Size** — breadth of knowledge baked into the model, and its knowledge cutoff date
2. **Tokenization** — how text is split into units the model learns from; determines vocabulary and context limits
3. **Embeddings** — how tokens become vectors the model can process; learned during training
4. **Parameters** — what's actually inside that model making it smart
5. **Model Size** — how many parameters, which determines capability vs. cost vs. hardware requirements
6. **Features** — what the model learned to detect (you don't control this, but understanding it explains model behavior)

**Customizing & optimizing for deployment:**

7. **Fine-tuning & LoRA** — how you specialize a general model for your agent's specific domain or behavior
8. **Quantization** — how you deploy affordably without sacrificing too much quality

**At inference (every request):**

Tokenization and embeddings also run at inference time — the same flow (text → tokens → vectors → transformer layers → output) executes on every request your agent handles. The difference is that during training the model *learns* from this flow, while during inference it *applies* what it learned.

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL CREATION                           │
│                                                             │
│  Pre-trained Data (trillions of tokens)                     │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  PRE-TRAINING                                       │    │
│  │                                                     │    │
│  │  Text ──→ TOKENIZATION ──→ Token IDs                │    │
│  │                │                                    │    │
│  │                ▼                                    │    │
│  │           EMBEDDING ──→ Vectors                     │    │
│  │                │                                    │    │
│  │                ▼                                    │    │
│  │           TRANSFORMER LAYERS                        │    │
│  │           (learn parameters by predicting           │    │
│  │            next token, billions of iterations)      │    │
│  │                                                     │    │
│  └─────────────────────────────────────────────────────┘    │
│       │                                                     │
│       ▼                                                     │
│  Base Model (parameters/weights, e.g. 7B, 70B, 405B)        │
│       │                                                     │
│       ▼                                                     │
│  (Optional) FINE-TUNING / LoRA ──→ Specialized Model        │
│       │                                                     │
│       ▼                                                     │
│  (Optional) QUANTIZATION ──→ Compressed for deployment      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE (using the model)              │
│                                                             │
│  Raw Text (user prompt, system prompt, RAG context)         │
│       │                                                     │
│       ▼                                                     │
│  TOKENIZATION ──→ Token IDs [5765, 2065, 374, ...]          │
│       │                                                     │
│       ▼                                                     │
│  EMBEDDING ──→ Dense vectors (features learned in training) │
│       │                                                     │
│       ▼                                                     │
│  TRANSFORMER LAYERS (parameters detect features,            │
│       │              build understanding layer by layer)    │
│       │                                                     │
│       ├── + LoRA adapters (if kept separate, applied here   │
│       │      alongside transformer weights per layer)       │
│       │                                                     │
│       │   OR merged into weights (if pre-merged after       │
│       │      fine-tuning — no extra step visible here)      │
│       │                                                     │
│       ▼                                                     │
│  Output Token Probabilities ──→ Generated Text              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The top box happens once (or occasionally, when you fine-tune). The bottom box happens every time your agent processes a request. Understanding both helps you make informed decisions about which model to pick, whether to fine-tune, and how to deploy.

---

## Pre-trained Data Size

**Pre-trained data size** is the volume of text the model was trained on during pre-training. It's measured in tokens and determines the breadth of knowledge.

> **Note:** This is the *total* count of tokens processed during training, not unique tokens. The same document can appear multiple times across training epochs, so "15 trillion tokens" means the model read 15 trillion tokens in sequence as training examples — including repetitions. The actual unique token vocabulary is tiny by comparison (32K–128K entries). Think of it as "total tokens read" rather than "distinct tokens in the dataset."

### Scale reference:

| Model | Training Tokens | Approximate Raw Text |
|-------|----------------|---------------------|
| LLaMA-2 | 2 trillion | ~1.5 TB of text |
| LLaMA-3 | 15+ trillion | ~11 TB of text |
| GPT-4 (estimated) | 13+ trillion | ~10 TB of text |

> GPT-4's training data size is not officially disclosed by OpenAI. The figure above is a widely cited estimate.

### What's in the data:

- Web pages (Common Crawl, filtered)
- Books and academic papers
- Code (GitHub)
- Wikipedia, encyclopedias
- Conversations, forums
- Curated high-quality sources

### Scaling laws (Chinchilla) — how model size and training data relate:

The optimal balance between model size and training data follows a power law. For a fixed compute budget, there's an ideal ratio — you get the best performance by scaling parameters and training tokens roughly together, not by making one huge and the other small:
- A 7B model should see ~140B tokens (20x parameters)
- A 70B model should see ~1.4T tokens
- Modern models (LLaMA-3, Gemma) deliberately train far beyond Chinchilla-optimal — this makes the model slower to train but faster/cheaper at inference, since you get a smaller model with more knowledge baked in

### Why data size matters:

- **More data = broader knowledge** (more facts, languages, domains)
- **Data quality > data quantity** (filtered web >> raw web)
- **Diminishing returns** exist — doubling data doesn't double quality
- **Cutoff date** — the model only knows what was in its training data

### Practical takeaways for agentic architectures:

- **Knowledge cutoff is your agent's blind spot.** If your agent needs current information (prices, docs, APIs), you must supplement with RAG or tool use — the model's training data has a fixed end date.
- **Domain coverage:** A model trained on more code data will be better at code agents. Check what's in the training mix before choosing a model for a specialized agent.
- **Fine-tuning vs. RAG:** If the pre-training data doesn't cover your domain, you have two options: fine-tune (expensive, bakes knowledge into weights) or RAG (cheaper, retrieves knowledge at runtime). For most agents, RAG is the practical choice.
- **Chinchilla tradeoff for deployment:** Models trained beyond Chinchilla-optimal (like LLaMA-3) are smaller for their capability level — meaning cheaper to run your agent in production.

---

## Tokenization

**Tokenization** is the process of converting raw text into a sequence of integers (token IDs) that the model can process. The model doesn't see characters or words — it sees tokens.

**Raw text** is whatever input string gets sent to the model before any processing. It can originate from:

- User prompts typed into a chat interface
- System/developer prompts prepended by the application
- Retrieved context (from RAG, tool outputs, etc.)
- Code, documents, or any text the application concatenates together

It's simply the human-readable string that hasn't yet been split into tokens.

**When does the model "see" tokens?** At inference time, before any neural network computation begins:

1. The **tokenizer** (a deterministic preprocessor sitting *outside* the neural network) splits the raw text into subword pieces using BPE
2. Each piece **maps to an integer ID** via a fixed vocabulary lookup table (e.g., `"token"` → `5765`)
3. Those IDs **hit the embedding layer** — each ID indexes into a learned matrix to produce a dense vector (e.g., 4096 floats)
4. The **transformer layers process those vectors** — this is where the "thinking" happens (attention, feed-forward networks, billions of parameters)

From step 3 onward, the model only operates on high-dimensional vectors — it never processes raw characters or words directly.

```
Input:  "Tokenization is interesting!"
Tokens: ["Token", "ization", " is", " interesting", "!"]
IDs:    [5765, 2065, 374, 7185, 0]   (illustrative — actual IDs vary by tokenizer)
```

### How it works:

- **BPE (Byte-Pair Encoding):** The most common method. Starts with individual characters, then iteratively merges the most frequent pairs into single tokens.
- Common words become single tokens: "the" → 1 token
- Rare words get split: "tokenization" → "token" + "ization" (2 tokens)
- Vocabulary sizes: typically 32K–128K tokens

### Practical takeaways for agentic architectures:

- **Cost:** API pricing is per token
- **Context limits:** Windows are measured in tokens, not words (~1 token ≈ 0.75 words in English)
- **Non-English text:** Gets split into more tokens (higher cost, less context)
- **Code:** Special tokenizers handle code syntax better

---

## Embeddings

An **embedding** is a dense vector representation of a token (or sequence) in continuous high-dimensional space. It's how meaning gets encoded as numbers.

```
"king"  → [0.23, -0.45, 0.89, ..., 0.12]  (dimension: 4096)
"queen" → [0.25, -0.43, 0.87, ..., 0.14]  (similar vector!)
"car"   → [-0.67, 0.31, -0.22, ..., 0.88] (very different)
```

### Key properties:

- **Semantic similarity:** Words with similar meanings have vectors that are close together
- **Relationships are preserved:** king - man + woman ≈ queen
- **Learned during training:** The embedding layer is the first layer of the Transformer

### Vector dimensions:

Each embedding is a fixed-length array of floating-point numbers, each entry in this vector is a dimension. The length of that array is the **dimensionality** of the embedding.

- **What is a dimension?** A dimension is an axis — a direction in the vector space. Think of 3D space: you have height, width, and depth — three axes, three dimensions. A point in 3D space needs three numbers (one per axis) to describe its position. Now scale that up: a 4096-dimension vector is a point in a space with 4096 axes. 

    Before training, only the *number* of dimensions is decided (e.g., "this model will have 4096"). What each axis means is not defined by anyone — they start as blank slots.
    
     During training, the model learns what each axis should represent by adjusting weights to minimize prediction error. The meaning of each dimension emerges from the training process itself. No human labels them. The model fills each axis/dimension with whatever statistical pattern helps it predict text better. 
     
     After training, each axis has settled into capturing *something* — but it's not human-labeled. Dimension #742 doesn't explicitly mean "formality" or "topic." It's a direction that, combined with all other directions, helps the model separate different meanings. The number stored in each dimension is how far along that axis the word sits. Individually, one axis tells you almost nothing. But the combination of all 4096 positions places the word at a unique point where semantically similar words cluster together.
- **Why more dimensions?** Higher dimensionality = more capacity to distinguish between meanings. A 4096-dim vector can represent finer-grained differences than a 768-dim one.
- **Tradeoff:** More dimensions = more memory, more compute for similarity searches, and more data needed to train well.

**Common dimension sizes:**

| Model / Context | Dimensions | Notes |
|----------------|-----------|-------|
| BERT (base) | 768 | Older, smaller embedding models |
| OpenAI text-embedding-3-small | 1536 | Good balance of quality and cost |
| OpenAI text-embedding-3-large | 3072 | Higher quality, more expensive to store/search |
| LLaMA-3 8B (internal) | 4096 | Hidden dimension inside the LLM |
| LLaMA-3 70B (internal) | 8192 | Larger model = wider vectors |

**Practical impact:**
- A 1536-dim embedding for 1 million documents ≈ 6 GB of vector storage (float32)
- Similarity search (cosine, dot product) scales with dimensionality — higher dims = slower brute-force search
- Vector databases (Pinecone, pgvector, FAISS) use indexing to make this manageable

### Practical takeaways for agentic architectures:

- **RAG depends on embeddings.** When your agent retrieves context (documents, tool outputs, knowledge base entries), it uses embedding models to find semantically relevant chunks. Choosing the right embedding model and dimensionality directly affects retrieval quality.
- **Embedding model ≠ LLM.** Embedding models (like OpenAI's text-embedding-3, Cohere Embed, Amazon Titan Embeddings) are separate, smaller models purpose-built for search. You embed documents and queries, then find nearest neighbors by cosine similarity. The LLM's internal embedding layer is different — you don't access it directly.
- **Dimensionality affects your vector database costs.** Higher dimensions = better retrieval quality but more storage and slower search. For most agents, 1536 dimensions is a practical sweet spot.
- **Embedding choice impacts agent accuracy.** If your agent retrieves the wrong context because embeddings don't capture domain-specific meaning well, the LLM will generate wrong answers — garbage in, garbage out. Consider domain-specific or fine-tuned embedding models for specialized agents.

---

## Parameters / Weights

**Parameters** (or weights) are the learned numerical values inside the model. They are what the model "knows" — the compressed knowledge from training.

```
Neuron output = activation(weight₁·input₁ + weight₂·input₂ + ... + bias)
```

- Each connection between neurons has a weight
- Training = adjusting these weights to minimize prediction error
- Model size is counted in parameters: 7B, 70B, 405B

**Scale reference:**

| Model | Parameters | Approximate Size (FP16) |
|-------|-----------|------------------------|
| LLaMA-3 8B | 8 billion | ~16 GB |
| LLaMA-3 70B | 70 billion | ~140 GB |

More parameters = more capacity to store knowledge and patterns, but also more resources needed — both during training and when using the model:

- **During training:** More parameters require more GPU memory, more compute, and more training data to learn effectively. Training a 70B model costs orders of magnitude more than training an 8B model.
- **When using the model (inference):** Every parameter must be loaded into GPU RAM to run the model. The parameters *are* the knowledge — they don't get compressed or summarized after training. So a 70B model at FP16 needs ~140 GB of GPU memory just to load, every single time you use it. This is why quantization matters — it shrinks each parameter from 16 bits to 4 or 8 bits, cutting memory by 2–4× so large models can run on smaller hardware.

### Practical takeaways for agentic architectures:

- **More parameters ≠ always better for your agent.** A 70B model is smarter but slower and more expensive per request. If your agent makes dozens of LLM calls per task (tool use, planning, reflection), latency and cost compound fast.
- **Right-size your model to the task:** Use a small model (7-8B) for simple routing/classification steps, and a large model (70B+) only for complex reasoning steps. Many production agents mix model sizes across their pipeline.
- **Self-hosting vs. API:** If you self-host, parameter count directly determines your GPU bill. If you use an API, you pay per token — but the provider's model size still affects latency.

---

## LLM Model Size

**Model size** refers to the parameter count and directly determines memory requirements and capability:

### What determines model size:

- **Number of layers** (depth)
- **Hidden dimension** (width)
- **Number of attention heads**
- **Vocabulary size**

### Size vs. capability tradeoff:

```
Parameters    RAM (FP16)    Capability Level
─────────────────────────────────────────────────
  1-3B         2-6 GB      Simple tasks, classification
  7-8B        14-16 GB     General chat, code, reasoning
 13-14B       26-28 GB     Strong all-around performance
 30-34B       60-68 GB     Near frontier quality
 65-70B      130-140 GB    Frontier-class reasoning
 180B+       360+ GB       Largest open models
```

### Mixture of Experts (MoE):

Some models (Mixtral, GPT-4) use MoE — they have many parameters but only activate a subset per token. A router network decides which "expert" sub-networks handle each token:

- Mixtral 8x7B: 47B total parameters, but only ~13B active per inference
- Benefit: More knowledge stored (large total params), same inference cost as a smaller dense model
- Tradeoff: Requires more memory to load the full model, even though only a fraction runs per token

### Practical takeaways for agentic architectures:

- **Picking a model size:** Match capability to your agent's hardest task. If your agent just classifies intent and routes to tools, 7-8B is fine. If it needs multi-step reasoning or code generation, you need 30B+.
- **MoE models are attractive for agents** — you get large-model knowledge at smaller-model inference cost. Good for agents that need broad knowledge but run on a budget.
- **Latency matters for agents:** Larger models generate tokens slower. If your agent has multi-turn loops (think → act → observe → think), even small per-token latency differences compound into noticeable delays across dozens of calls.
- **Context window vs. model size:** A smaller model with a large context window may outperform a larger model with a small window for agents that need to process long tool outputs or conversation histories.

---

## Features

A **feature** is a pattern the model detects in the input to help it decide what to output next. Features are not defined by anyone — the model discovers them on its own during training.

### How the model uses features:

Features aren't a separate data structure the model looks up — they **are** the computation happening inside the transformer layers. The model builds features and uses them simultaneously, layer by layer:

1. **Input:** Token embeddings arrive as vectors (raw features — just position in space based on token identity)
2. **Each transformer layer transforms the vectors:**
   - **Attention** looks at all tokens and asks "which other tokens are relevant to this one?" — this detects relational features (e.g., "the word 'it' refers to 'the dog' earlier")
   - **Feed-forward network** transforms the vector further — this detects local features (e.g., "this pattern looks like a negation")
   - The output is a **new vector** that now encodes richer features than what came in
3. **After all layers:** The final vector encodes everything the model "understands" about the full input — all features combined. This gets projected into vocabulary probabilities to predict the next token.

Each layer reads features from the previous layer's output, builds higher-level features on top, and passes them forward. The model doesn't "detect a question and then use that fact" as separate steps. Instead, by some middle layer, the vector has been shaped in a way that *implicitly encodes* "this is a question" — and subsequent layers build on that shape to produce an appropriate answer.

**Features in training vs. inference:**

Features play a role in both phases, but differently:

- **During training (learning features):**
  - The model sees billions of text examples and adjusts its parameters (weights) to minimize prediction error
  - Through this process, the transformer layers learn to produce useful feature representations — patterns that help predict the next token accurately
  - This is when features are *discovered*: the model figures out that detecting things like "this is a question" or "this word is negated" helps it make better predictions
  - If a feature isn't useful for prediction, the model won't learn to detect it

- **During inference (using features):**
  - The same transformer layers run on your input, detecting the same patterns they learned during training
  - No new features are learned — the model applies what it already knows
  - The quality of the output depends entirely on what features the model learned to detect during training

Think of it like learning to drive vs. driving:
- **Training** = learning to recognize stop signs, lane markings, pedestrians (discovering features)
- **Inference** = actually driving and using those recognition skills in real time (using features)

The features exist *in* the parameters. Training shapes the parameters so they detect useful patterns. Inference runs input through those shaped parameters to produce predictions.

### Features emerge in layers:

- **Early layers (surface patterns):** The model notices basic things — "this token is punctuation," "these characters form a number," "this word starts with a capital letter."
- **Middle layers (higher-level patterns):** It combines surface patterns into richer signals — "this looks like a question," "this paragraph is about cooking," "this sentence has a negative tone."
- **Deep layers (complex understanding):** It assembles those into full comprehension — "the user is frustrated and asking for a refund," "this code has a bug in the loop logic," "this is sarcasm."

Each transformer layer builds increasingly abstract features on top of the previous layer's output.

### How features connect to embeddings and dimensions:

Features and dimensions are related but not the same thing:

- **Dimension** = a single axis (slot) in the vector. It holds one number. It's a structural concept — the container.
- **Feature** = a pattern the model has learned to detect (e.g., "is this a question?"). It's a semantic concept — the meaning.

**Why they're not one-to-one:**

- A single feature (like "negative tone") is encoded **across many dimensions** — no single number captures it alone.
- A single dimension contributes to **many features** simultaneously — dimension #500 might partially encode tone, topic, and formality all at once.

This is called a **distributed representation**. It's what makes embeddings powerful (and hard to interpret) — meaning is spread out, not neatly boxed into individual slots.

**Analogy:** Think of a painting. The dimensions are individual pixels. A feature is "there's a face in this image." No single pixel *is* the face — the face emerges from the combination of many pixels. And each pixel contributes to multiple things (face, background, lighting) at once.

### How is this different from traditional ML?

In older ML (like spam filters), a human would manually decide what features to look for:
- Count exclamation marks
- Check for ALL CAPS
- Look for the word "free"

You'd hand-engineer a list of signals and feed them to the model. In LLMs, nobody defines the features. The model discovers what patterns matter by itself during training. You can't easily inspect or name them — they emerge from the data.

### Practical takeaways for agentic architectures:

- **You don't control features directly** — but you influence them through prompt engineering. A well-structured prompt activates the right features (the model "recognizes" the pattern you're presenting).
- **Why some prompts work better:** When you add "Think step by step" or use few-shot examples, you're activating features the model learned during training — patterns associated with careful reasoning or structured output.
- **Model behavior debugging:** When your agent produces unexpected output, it's often because the input activated the wrong features. Rephrasing the prompt can shift which features dominate.
- **Fine-tuning sharpens features:** If you fine-tune a model on your domain, you're strengthening features relevant to your use case and weakening irrelevant ones — making the model more reliable for your agent's specific tasks.

---

## Fine-Tuning & LoRA

**Fine-tuning** is the process of taking a pre-trained model and training it further on a smaller, domain-specific dataset to specialize its behavior.

### Why fine-tune?

A pre-trained model is a generalist — it knows a lot about everything but isn't optimized for your specific task. Fine-tuning narrows its focus:

- Make it follow a specific output format consistently (JSON, structured responses)
- Teach it domain knowledge it didn't see in pre-training (your company's internal docs, proprietary APIs)
- Improve its tone or style (customer support voice, technical writing)
- Make a smaller model perform like a larger one on a narrow task

### Full fine-tuning vs. LoRA:

**Full fine-tuning** updates all parameters in the model:
- Requires enormous GPU memory (a 70B model needs hundreds of GBs)
- Produces the highest quality results
- Expensive and slow — often impractical for most teams

**LoRA (Low-Rank Adaptation)** is a fine-tuning technique that only trains a small set of adapter parameters, making it much cheaper in GPU memory and compute. The training dataset can be any size — LoRA's benefit is about compute efficiency, not data efficiency.

How it works:

In a transformer, the attention layers have large weight matrices (e.g., 4096 × 4096 = ~16 million numbers in one matrix). During full fine-tuning, you'd update all 16 million numbers. LoRA takes a different approach:

1. The original 4096 × 4096 matrix stays **completely frozen** — not a single number changes
2. Two new small matrices are **added alongside** it (e.g., A: 4096 × 16, B: 16 × 4096)
3. Only A and B get trained during the LoRA fine-tuning process
4. At inference time: `output = original_matrix × input + (A × B) × input`

Why does this work? When you fine-tune for a specific task, you don't need to change all 16 million numbers. The adjustment needed is much simpler — it can be captured with far fewer numbers. Think of it like: the original matrix is a detailed painting, but the edit you need is just "shift everything slightly warmer in tone" — that's a simple adjustment, not a full repaint. The two small matrices capture that simple adjustment.

The "16" in those dimensions is the **rank** — a knob you control:
- Higher rank (32, 64) = more capacity to learn complex adjustments, but more parameters to train
- Lower rank (4, 8) = cheaper and faster, but less expressive

**Deployment options after training:**

Once A and B are trained, you have two choices for how to use them:

- **Keep adapters separate:** `output = original_matrix × input + (A × B) × input`. Slightly more compute per request, but you can **swap adapters at runtime** — same base model, different LoRA adapters for different tasks, without reloading the full model.
- **Merge adapters into the base:** Multiply A × B to get a 4096 × 4096 matrix and permanently add it to the original weights: `merged = original + (A × B)`. Zero extra computation at runtime, but now it's a single specialized model — you can't swap behaviors anymore.

**Which parameters get updated?**

You (the practitioner) choose which layers to attach LoRA adapters to. Typically:
- LoRA is applied to the **attention layers** (the query, key, and value projection matrices) — these are the most impactful for changing model behavior
- You can also apply it to feed-forward layers, but attention layers give the most bang for the buck
- The **rank** (a hyperparameter you set, e.g., rank=16 or rank=64) controls how many new parameters each adapter has — higher rank = more capacity to learn, but more memory

The original model weights are completely frozen. Only the small adapter matrices receive gradient updates during training.

**How does training know when to stop?**

Same as any neural network training — you monitor a **validation loss**:
- You split your data into training set and validation set
- After each training epoch (pass through the data), you check how well the model performs on the validation set
- **Early stopping:** If validation loss stops improving (or starts getting worse), you stop training — continuing would mean the model is memorizing training data rather than learning generalizable patterns (overfitting)
- You can also set a fixed number of epochs or steps as a hard limit

There's no magic "done" signal — it's the practitioner's job to monitor metrics and decide when the adapters have learned enough.

**Analogy:** Imagine you have a textbook (the pre-trained model). Full fine-tuning takes a red pen to every page — the original content is the starting point, but every part is allowed to change. LoRA adds a small set of sticky notes to specific pages — the original pages stay untouched, and the sticky notes are cheap to create.

### QLoRA:

QLoRA = LoRA applied on top of a quantized model:
- Load the base model in 4-bit (saving memory)
- Train LoRA adapters in full precision on top
- This lets you fine-tune a 70B model on a single consumer GPU (24–48 GB)

### Practical takeaways for agentic architectures:

- **LoRA makes specialization affordable.** Instead of each team hosting their own full model or paying for expensive full fine-tuning, your organization hosts one base model and each team trains a small LoRA adapter tailored to their domain. Adapters are just a few hundred MBs each — cheap to store, fast to swap — while the base model (the expensive part) is shared infrastructure.
- **Swap adapters at runtime:** Since adapters are small and separate from the base model, you can load different ones on demand without reloading the entire model. One adapter for customer support tone, another for code generation, another for summarization — switch between them in milliseconds based on what your agent needs to do next.
- **When to fine-tune your agent:** If prompt engineering alone can't get consistent behavior (wrong format, wrong tone, missing domain knowledge), fine-tuning is the next step.
- **When NOT to fine-tune:** If your agent just needs access to current data or documents, RAG is simpler and doesn't require retraining.

**When to fine-tune vs. use RAG:**

| Approach | Best for | Tradeoff |
|----------|----------|----------|
| **RAG** | Factual recall, current information, large knowledge bases | No training needed, but adds latency and retrieval complexity |
| **Fine-tuning** | Behavior change, output format, tone, domain-specific reasoning | Requires training data and compute, but faster at inference |
| **Both** | Production agents that need specialized behavior AND current knowledge | Most robust but most complex to maintain |

---

## Quantization

**Quantization** reduces the numerical precision of model weights to make models smaller and faster, with minimal quality loss.

Remember: every parameter in a model is a number (a weight). By default, each weight is stored as a 16-bit floating-point number (FP16). A 70B model has 70 billion of these numbers, each taking 16 bits — that's ~140 GB just to load the model into memory. For many teams, that's too expensive or simply doesn't fit on available hardware.

Quantization solves this by asking: "Do we really need 16 bits of precision for each weight?" In most cases, the answer is no. You can round each weight to a less precise representation (8-bit, 4-bit, even 2-bit) and the model still produces nearly the same output. It's like reducing an image from 24-bit color to 8-bit color — you lose some subtle gradients, but the picture is still recognizable.

### When is quantization applied and by whom?

Quantization is typically applied **after training**, as a separate step before deployment:

- **Not during pre-training** — training needs full precision (FP16/FP32) because the model is making tiny adjustments to weights. Rounding during training would destroy the learning process.
- **Not during fine-tuning** — same reason (though QLoRA is a technique that fine-tunes *on top of* an already-quantized model — but the base quantization happened earlier).
- **After training, before deployment** — someone takes the full-precision trained model and runs a quantization algorithm on it to produce a smaller version. This is a one-time conversion step.
- **Sometimes at load time** — tools like bitsandbytes quantize on-the-fly as the model loads into memory, so you don't need a separate pre-quantized file.

**Who does it:**

- **The community** — for open models (LLaMA, Mistral), community members run quantization and publish the results on HuggingFace. You download a pre-quantized model ready to use.
- **You** — if you self-host, you can quantize a model yourself using tools like llama.cpp, AutoGPTQ, or AutoAWQ.
- **The API provider** — if you use OpenAI, Anthropic, or AWS Bedrock, they handle quantization (or don't) internally. You never see it or need to think about it.

```
Original (FP16):   1.234375 → stored in 16 bits
Quantized (INT4):  1.25     → stored in 4 bits (approximation)
```

### Why quantize:

- **Memory:** A 70B model at FP16 = 140 GB. At 4-bit = ~35 GB. Fits on consumer hardware.
- **Speed:** Less memory bandwidth needed = faster inference
- **Cost:** Smaller models = cheaper to run
- **Tradeoff:** Small quality degradation (often 1-3% on benchmarks)

### Quantization methods:

| Method | Approach |
|--------|----------|
| GPTQ | Post-training quantization using calibration data |
| AWQ | Activation-aware quantization (protects important weights) |
| GGUF | Format used by llama.cpp for CPU inference |
| bitsandbytes | Dynamic quantization during loading (NF4, INT8) |

### Practical takeaways for agentic architectures:

- **Quantization makes self-hosting viable.** Without it, running a 70B model requires ~140 GB of GPU RAM (multiple A100s). With 4-bit quantization, it fits on a single 48 GB GPU.
- **Quality vs. cost tradeoff:** For most agent tasks, 4-bit quantized models perform nearly as well as full precision. Test your specific use case — if your agent does complex reasoning, you may notice degradation; for routing and classification, it's usually fine.
- **Latency benefit:** Smaller model in memory = faster token generation = snappier agent responses, especially in multi-turn loops.
- **When NOT to quantize:** If you're using an API (OpenAI, Anthropic, Bedrock), the provider handles this. Quantization only matters when you self-host.

---

## Q-bits (Quantization Bits)

**Q-bits** refers to the number of bits used to represent each weight after quantization. Lower bits = smaller model = faster but potentially less accurate.

```
FP32 (32-bit): Full precision, baseline quality
FP16 (16-bit): Half precision, standard for training/inference
INT8 (8-bit):  Good quality, 2x smaller than FP16
INT4 (4-bit):  Acceptable quality, 4x smaller than FP16
INT2 (2-bit):  Experimental, significant quality loss
```

### Practical guide:

| Q-bits | Size Reduction | Quality Impact | Use Case |
|--------|---------------|----------------|----------|
| FP16 | 1x (baseline) | None | Cloud inference, training |
| INT8 | 2x smaller | Minimal (~1%) | Production serving |
| Q6_K | 2.7x smaller | Very small | Local with quality priority |
| Q4_K_M | 4x smaller | Small (~2-3%) | Local inference on laptop |
| Q2_K | 8x smaller | Noticeable | Experimentation only |

### Practical takeaways for agentic architectures:

- **Q4_K_M is the sweet spot for most self-hosted agents** — 4x smaller with minimal quality loss. Good enough for tool-calling, summarization, and general reasoning.
- **Use Q6_K or INT8 if your agent does complex multi-step reasoning** where small quality drops compound across steps.
- **Use Q2_K only for prototyping** — the quality loss is too high for production agents.
- **Match Q-bits to your hardware:** If you have a 24 GB GPU and want to run a 70B model, you need Q4 or lower. If you have 48 GB, you can afford Q6_K for better quality.

---

[← Back to home](/)
