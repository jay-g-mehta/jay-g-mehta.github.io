---
title: "Self-Attention & Multi-Head Attention Explained: Why Attention Is All You Need"
description: "Applied AI — ground-up walkthrough of self-attention in Transformers, Q, K, V matrices, scaled dot-product attention, and multi-head attention with step-by-step examples."
---

# Self-Attention & Multi-Head Attention Explained: Why Attention Is All You Need

> A ground-up walkthrough of how self-attention works in Transformers — from token embeddings to multi-head attention. Understanding this mechanism helps applied AI engineers make better decisions about prompt design, context management, retrieval strategies, and when to trust (or not trust) model outputs.

---

## The Problem Self-Attention Solves

Words don't mean anything in isolation. "Bank" means something different in "river bank" vs. "bank account." To understand (or generate) language, a model needs to know how each word relates to every other word in the sequence.

Before Transformers, models processed tokens sequentially (RNNs/LSTMs) — reading left to right, carrying a compressed "memory" forward. This created two problems:

1. **Long-range dependencies get lost.** By the time the model reaches token 500, information about token 3 has been compressed through hundreds of steps and is largely gone.
2. **Sequential processing is slow.** You can't parallelize — each step depends on the previous one.

**Self-attention solves both:** it lets every token directly attend to every other token in one parallel operation. Token 500 can look directly at token 3 with no information loss. The model learns *which* tokens are relevant to which — and how much to weight each relationship.

In short: self-attention lets each token look at every other token in the sequence and decide how much information to pull from each — producing a new representation that encodes not just the token itself, but its role in context.

---

## Self-Attention: The Core Mechanism

> We'll explain where Q, K, V come from shortly (see [How Q, K, V Are Computed](#how-q-k-v-are-computed-before-self-attention-begins)). For now, assume each token has been projected into three vectors: a Query (Q), a Key (K), and a Value (V).

Consider a sample string, assuming each word is a token:

```
"cat on mat"
```

Each token gets its own Q, K, and V vectors, computed using 3 weight matrices — W_Q, W_K, W_V — that are shared across all tokens.

The self-attention mechanism computes a scalar relevance score between two tokens:

```
score = (embedding_token1 · W_Q) · (embedding_token2 · W_K)ᵀ
```

This operation is repeated for all combinations of tokens, resulting in an n × n matrix (where n is the number of tokens):

```
              K_cat      K_on     K_mat
   Q_cat      0.380     0.330     0.379
   Q_on       0.329     0.278     0.328
   Q_mat      0.381     0.326     0.379
```

Then softmax is applied on this matrix:

```
              K_cat      K_on     K_mat      Sum
   Q_cat      0.339     0.323     0.339    = 1.000
   Q_on       0.339     0.322     0.339    = 1.000
   Q_mat      0.340     0.321     0.339    = 1.000
```

> **Note on scaling:** In practice, scores are divided by √d_k (e.g., √64 = 8) before softmax to prevent extreme values that would push softmax toward 0/1 and kill gradients. Scaling is omitted from this example for clarity — see [The Full Operation](#the-full-operation--step-by-step) below for the complete formula.

> Note: The "Sum" column is not part of the matrix — it highlights that softmax normalizes values within each row to sum to 1.0.

**How to read this:**
- Row for "cat": When updating "cat", pay 33.9% attention to itself, 32.3% to "on", and 33.9% to "mat"
- Row for "mat": When updating "mat", pay 34.0% to "cat", 32.1% to "on", 33.9% to itself

> **Why is attention nearly uniform here?** These scores come from random (untrained) weights. The model hasn't learned meaningful patterns yet, so every token looks roughly equally at every other token. After training, you'd see sharper distributions — e.g., "cat" paying 70% attention to "sat" and only 5% to "the" — because the model has learned which relationships matter.

---

## What V Does: Computing the Output

The final step uses the attention weights to blend value vectors:

```
output_cat = 0.339·V_cat + 0.323·V_on + 0.339·V_mat  → new vector for "cat"
output_on  = 0.339·V_cat + 0.322·V_on + 0.339·V_mat  → new vector for "on"
output_mat = 0.340·V_cat + 0.321·V_on + 0.339·V_mat  → new vector for "mat"
```

Each token gets a new representation that blends information from all tokens, weighted by the attention scores.

---

## The Full Operation — Step by Step

```
Step 1: Compute all Q, K, V for every token (upfront, in parallel)

Step 2: score = Q_cat · K_cat^T        → scalar, e.g. 4.2
         (repeat for all token pairs → n×n matrix of scores)

Step 3: scaled_scores = scores / √d     → prevents extreme values before softmax

Step 4: weights = softmax(scaled_scores) → e.g. [0.05, 0.42, 0.12, 0.03, 0.02, 0.36]

Step 5: output_cat = Σ (weight_i · V_i) → weighted sum of ALL V vectors
```

### The intuition behind each step

| Step | Operation | What it gives you |
|------|-----------|-------------------|
| 1. Compute Q, K, V | X · W_Q, X · W_K, X · W_V | Three different "views" of each token — one for asking, one for being found, one for carrying information |
| 2. Q · Kᵀ | Dot product of all query-key pairs | Relevance map — how related every token pair is (n×n raw scores). It's learned, asymmetric relevance. "mat"→"cat" might score high, while "cat"→"mat" might score differently |
| 3. Scale by √d | Divide scores by √d_k | Numerical stability — prevents scores from getting too large, which would make softmax produce near-0/1 extremes (killing gradients) |
| 4. Softmax (per row) | Normalize each row | Attention distribution — converts raw scores into weights (probabilities) that sum to 1 per token |
| 5. Multiply by V | Weights · V | Context-aware representation — each token's new vector, enriched with information from all relevant tokens. Steps 2–4 decided *how much* to take from each token. Step 5 actually *takes* it |

**What does the model gain from this?**

After one self-attention layer, each token's representation now contains information from the entire sequence, weighted by relevance.

---

## How Q, K, V Are Computed Before Self-Attention Begins

### 1. Tokenize the input

```
"cat on mat" → ["cat", "on", "mat"]
```

### 2. Look up embeddings (from a learned embedding table)

Each token maps to a fixed-size vector (e.g., 512 dimensions):

```
"cat" → [0.23, -0.91, 0.44, ..., 0.12]   (512 numbers)
"on"  → [0.67, 0.03, -0.55, ..., 0.88]
"mat" → [0.19, -0.84, 0.51, ..., 0.07]
```

These are learned during training (like a dictionary: word → vector).

### 3. Add positional encoding

Since the Transformer has no sense of order, we add position information:

```
embedding_cat = raw_embedding_cat + positional_encoding(position=0)
embedding_on  = raw_embedding_on  + positional_encoding(position=1)
embedding_mat = raw_embedding_mat + positional_encoding(position=2)
```

Now each vector encodes both meaning and position.

### 4. Project into Q, K, V

```
Q_cat = embedding_cat · W_Q    (512 → 64 dimensions)
K_cat = embedding_cat · W_K    (512 → 64 dimensions)
V_cat = embedding_cat · W_V    (512 → 64 dimensions)
```

W_Q, W_K, W_V are called **projection matrices**. They take a high-dimensional vector (512-dim embedding) and project it into a lower-dimensional subspace (64 dimensions). The reason for this dimension reduction is explained in the [Multi-Head Attention](#multi-head-attention) section below.

### Summary — the full pipeline before attention scores are computed

```
Input text
  → Tokenize
    → Embedding lookup (learned vectors)
      → Add positional encoding
        → Multiply by W_Q, W_K, W_V (one set per head)
          → Now you have Q, K, V for each token
            → Ready for Q·Kᵀ → softmax → ×V
```

---

## What is a "Head" in Self-Attention?

Everything described above is a **single head**. A "head" is one complete run of the self-attention mechanism with its own set of W_Q, W_K, W_V matrices.

```
One head = one Q·K·V computation over all tokens = one perspective on "what's relevant to what"
```

### Why is it called a "head"?

Think of it as one viewing angle. A single head can only learn one type of relationship between tokens:

- Head A might learn: "nouns attend to their verbs" (syntactic)
- Head B might learn: "adjectives attend to what they modify" (grammatical)
- Head C might learn: "nearby tokens attend to each other" (positional)

A single head can't do all of these simultaneously — it has only one W_Q and one W_K, so it encodes only one notion of "relevance." That's why multi-head attention exists.

---

## Multi-Head Attention

Multi-head = run multiple single-head attentions in parallel, each with its own W_Q, W_K, W_V, then concatenate the results.

### Why not just one head?

A single head learns one notion of relevance. But language has multiple simultaneous relationships:

- "cat" relates to "sat" (who did the action?)
- "cat" relates to "the" (which cat?)
- "sat" relates to "mat" (where?)

One head can only encode one of these patterns at a time. Multi-head lets the model capture all of them simultaneously.

### How it works (for 8 heads)

```
Input embedding: 512 dimensions per token

Split into 8 heads:
  Head 1: W_Q₁, W_K₁, W_V₁  →  512 → 64 dims  →  full attention  →  64-dim output
  Head 2: W_Q₂, W_K₂, W_V₂  →  512 → 64 dims  →  full attention  →  64-dim output
  Head 3: W_Q₃, W_K₃, W_V₃  →  512 → 64 dims  →  full attention  →  64-dim output
  ...
  Head 8: W_Q₈, W_K₈, W_V₈  →  512 → 64 dims  →  full attention  →  64-dim output

Concatenate: [64 + 64 + 64 + 64 + 64 + 64 + 64 + 64] = 512 dims

Final projection: 512 → 512 (one more learned matrix W_O)
```

### The formula from the paper

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., head₈) · W_O

where headᵢ = Attention(X·W_Qᵢ, X·W_Kᵢ, X·W_Vᵢ)
```

---

## Q, K, V Are Different in Each Head

Each head has its own W_Q, W_K, W_V matrices. The same input embedding gets projected differently:

```
Head 1:  Q_cat = embedding_cat · W_Q₁    →  64-dim vector (version A)
Head 2:  Q_cat = embedding_cat · W_Q₂    →  64-dim vector (version B)
Head 3:  Q_cat = embedding_cat · W_Q₃    →  64-dim vector (version C)
```

Same input embedding, different projection matrices, therefore different Q, K, V vectors. If Q, K, V were the same across heads, every head would compute identical attention patterns — making multiple heads useless. Different weight matrices let each head project the same input into different subspaces, capturing different relationships.

---

## Why Concatenate (Not Average)?

### The problem with averaging

If you average 8 head outputs, you destroy information:

```
Head 1 output for "cat": [0.9, -0.2, 0.1, 0.5]  ← learned "cat is a subject"
Head 2 output for "cat": [-0.8, 0.3, -0.1, -0.4] ← learned "cat is near 'the'"

Average: [0.05, 0.05, 0.0, 0.05]  ← meaningless — both signals cancelled out
```

### How concatenation preserves information

Line them up side by side:

```
Head 1 output: [0.9, -0.2, 0.1, 0.5]
Head 2 output: [-0.8, 0.3, -0.1, -0.4]

Concatenated: [0.9, -0.2, 0.1, 0.5, -0.8, 0.3, -0.1, -0.4]
               ↑--- Head 1's info ---↑  ↑--- Head 2's info ---↑
```

Nothing is lost. Both signals sit in their own "slots" within the 512-dim vector.

### What W_O then does

W_O is a 512×512 matrix that multiplies the concatenated vector. It can learn to:
- Selectively combine information across heads (e.g., "Take the subject-detection signal from Head 1 AND the proximity signal from Head 2 → combine into a richer feature")
- Keep them separate if that's more useful — by putting weight on one head's slot and zero on others for a given output dimension

**Analogy:**
- Average = Mix all paint colors together → get brown (lost all individual colors)
- Concatenate = Put all paint colors side by side on a palette → all still visible
- W_O = An artist who can pick from the full palette and decide how to blend for each stroke

---

## All Learned Matrices in One Multi-Head Attention Layer

| Matrix | Shape | Count |
|--------|-------|-------|
| W_Qᵢ | 512 × 64 | 8 (one per head) |
| W_Kᵢ | 512 × 64 | 8 |
| W_Vᵢ | 512 × 64 | 8 |
| W_O | 512 × 512 | 1 |

Total: 25 matrices per attention layer, all randomly initialized, all learned during training via backpropagation. Nobody hand-designs any of them.

---

## Summary

```
Multi-Head Attention =
  1. Project input into Q, K, V for each head (8 sets of W_Q, W_K, W_V)
  2. Run self-attention independently in each head (8 parallel n×n attention computations)
  3. Concatenate all head outputs (8 × 64 = 512)
  4. Apply W_O (final linear projection, 512 → 512)
  → Output: one 512-dim vector per token (same shape as input)
```

---

## Practical Takeaways for Applied AI Systems

Understanding self-attention internals isn't just academic — it directly informs how you design systems on top of LLMs:

- **Context window is finite attention budget.** Every token attends to every other token (O(n²)). Stuffing your prompt with irrelevant context doesn't just waste tokens — it dilutes the attention available for the tokens that matter. Be deliberate about what goes into the context.

- **Token order matters even though attention is "position-agnostic."** Positional encodings give the model a sense of order, but attention weights still decay with distance in practice. Put the most important information (instructions, key facts) where the model attends most reliably — typically the beginning and end of the context.

- **Chunking and retrieval quality directly affect reasoning.** Self-attention can only reason over what's in the sequence. If your RAG pipeline retrieves a poorly chunked passage where the answer spans two chunks, no amount of model capability will compensate — the relevant tokens simply aren't co-present for attention to connect them.

- **Multi-head attention explains why models handle multiple constraints simultaneously.** When you give an LLM a prompt with several requirements ("be concise, use formal tone, include code examples"), different heads can track different constraints in parallel. But there's a limit — too many simultaneous constraints exceed what the heads can cleanly separate.

- **Prompt engineering is attention engineering.** When you restructure a prompt and get better results, you're changing which tokens attend to which. Clear structure (headers, delimiters, explicit labels) helps attention heads form cleaner patterns. Ambiguous, run-on prompts create noisy attention maps.

- **Fine-tuning changes what the model attends to.** When you fine-tune on domain-specific data, you're updating W_Q, W_K, W_V matrices — literally redefining what the model considers "relevant" in a sequence. This is why fine-tuned models can outperform larger general models on narrow tasks with less prompting.

- **Long-context models aren't magic.** Extending context to 128K+ tokens doesn't mean the model attends equally well to all of it. Attention still spreads thinner as sequences grow. For long-document tasks, retrieval + focused context often outperforms dumping everything into a massive window.

---

[← Back to home](/)
