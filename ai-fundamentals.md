# AI Fundamentals

A quick reference to understand the key AI terms, how they connect, and how they're applied.

---

## 1. AI Disciplines — What Is It?

### Terminologies

**AI (Artificial Intelligence)** — The broad field of building systems that perform tasks requiring human-like intelligence: reasoning, learning, decision-making. The aim is for computers to think like human for problem solving.

**AI Model** — A program trained on large datasets to learn patterns and relationships, enabling it to make predictions, generate content, or make decisions on new data it has never seen before. The model is the trained artifact — the result of applying an algorithm to data. Examples: GPT-4, Claude, DALL-E, LLaMA.

**ML (Machine Learning)** — A subset of AI where systems learn patterns from data rather than being explicitly programmed.

**Neural Networks** — A ML technique inspired by the human brain, using layers of interconnected nodes to recognize patterns.

**Deep Learning** — Neural networks with many layers, enabling learning of complex representations from large datasets.

**NLP (Natural Language Processing)** — The discipline of enabling machines to understand, interpret, and generate human language.

**GenAI (Generative AI)** — AI that creates new content — text, images, audio, code — from learned patterns.

**LLM (Large Language Model)** — A deep learning model trained on massive text data that can both understand and generate language. Lives at the intersection of NLP and GenAI.

### How AI Disciplines Nest

```
┌───────────────────────────────────────────────┐
│  AI                                           │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │  Machine Learning                       │  │
│  │                                         │  │
│  │  ┌───────────────────────────────────┐  │  │
│  │  │  Neural Networks                  │  │  │
│  │  │                                   │  │  │
│  │  │  ┌─────────────────────────────┐  │  │  │
│  │  │  │  Deep Learning              │  │  │  │
│  │  │  └─────────────────────────────┘  │  │  │
│  │  └───────────────────────────────────┘  │  │
│  └─────────────────────────────────────────┘  │
└───────────────────────────────────────────────┘
```

### The NLP × GenAI Overlap

```
              Deep Learning
         ┌──────────────────────┐
         │                      │
         │   ┌──────┐ ┌──────┐  │
         │   │      │ │      │  │
         │   │ NLP  │ │GenAI │  │
         │   │      │ │      │  │
         │   │   ┌──┼─┼──┐   │  │
         │   │   │  LLM  │   │  │
         │   │   └──┼─┼──┘   │  │
         │   │      │ │      │  │
         │   └──────┘ └──────┘  │
         │                      │
         └──────────────────────┘
```

- **NLP** includes non-generative tasks (classification, sentiment analysis)
- **GenAI** includes non-language generation (images, audio, video)
- **LLM** = the overlap — understands language (NLP) AND generates content (GenAI)


## Quick Reference

| Term | Role | Relationship |
|------|------|-------------|
| AI | The whole field | Parent of all |
| ML | Learning from data | Subset of AI |
| Neural Networks | Pattern recognition | Subset of ML |
| Deep Learning | Multi-layer neural nets | Subset of Neural Networks |
| NLP | Language understanding | Discipline within DL |
| GenAI | Content creation | Capability within DL |
| LLM | Language model | Intersection of NLP + GenAI |

---

## 2. Core LLM Concepts

> The vocabulary you need before building with any LLM model or application interacting with LLM. Knowing these helps you prompt better, debug faster, and understand what you're paying for.

**Token** — The smallest unit of text an LLM processes. A word might be one token, or split into multiple. "chatbot" = 1 token, "unbelievable" = 3 tokens. An LLM takes in a sequence of tokens and predicts the next token — repeating this process one token at a time until the response is complete.

**Prompt** — The input you give to an AI model. Can be a question, instruction, or context. The quality of the prompt directly affects the quality of the output.

**Context Window** — The maximum amount of text (in tokens) a model can consider at once. Like the model's working memory. GPT-4: ~128K tokens. Claude: ~200K tokens.

**Inference** — The process where a trained model receives input and produces output. This is what happens every time you call an LLM API. You pay per inference (measured in tokens in/out).

**Hallucination** — When a model confidently generates incorrect or fabricated information. It doesn't "know" it's wrong — it's predicting likely text, not verifying facts.

**Parameters** — A parameter is a numerical value inside a model that controls how it transforms input into output. Think of them as millions of tiny dials — during training, each dial gets tuned until the model produces accurate results. These tuned values are what the model "knows." More parameters generally = more capable but more expensive to run. GPT-4 has hundreds of billions of parameters. When people say a model is "7B" or "70B," they're referring to the parameter count.

```
                        Putting It All Together

 "What is AI?"                                              "AI is a field of..."
      │                                                            ▲
      ▼                                                            │
 ┌──────────┐    ┌─────────────────────────────────────────┐   ┌──────────┐
 │Tokenizer │    │           Neural Network                │   │De-token  │
 │          │    │                                         │   │          │
 │ "What"   │    │  Layer 1        Layer 2       Layer 3   │   │  Output  │
 │ " is"    │    │                                         │   │  Tokens  │
 │ " AI"    │───>│  (O)──w──>(O)──w──>(O)                  │──>│  ──────> │
 │ "?"      │    │  (O)──w──>(O)──w──>(O)                  │   │  Text    │
 │          │    │  (O)──w──>(O)──w──>(O)                  │   │          │
 └──────────┘    │                                         │   └──────────┘
                 │  (O) = neuron (a math function)         │
  Input Tokens   │   w  = weight (a parameter)             │   Output Tokens
  (prompt)       │                                         │   (response)
                 └─────────────────────────────────────────┘

  Token:     Smallest unit of text the model reads/generates
  Neuron:    A math function that transforms inputs → output
  Weight:    A number on each connection between neurons (= a parameter)
  Parameter: All the weights collectively — what the model "learned"
  Input:     Your prompt, broken into tokens
  Output:    The model's response, generated one token at a time
```

---

## 3. LLM Optimization Techniques

> These are the terms you'll hear when people talk about optimizing LLM performance — whether by improving the model itself or improving how you use it.

**Training** — The process of feeding data to a model so it learns patterns. Requires massive compute and data. Done once (or rarely) by model providers.

**Fine-tuning** — Taking a pre-trained model and training it further on a smaller, domain-specific dataset. Makes a general model specialized (e.g., fine-tuning for medical or legal text).

**RAG (Retrieval-Augmented Generation)** — Instead of retraining, you retrieve relevant documents at query time and feed them to the model as context. Keeps responses current and grounded without retraining. Cost-effective alternative to fine-tuning.

**Prompt Engineering** — Crafting prompts to get better outputs. Techniques include few-shot examples, chain-of-thought reasoning, and system instructions.

---

## 4. LLM Applications

> These are the terms you'll hear when people talk about what gets built with LLMs — the products and systems end users interact with.

**Chatbot** — A conversational interface powered by an LLM. Responds to user queries in natural language. Stateless or with limited memory.

**AI Agent** — A system that can reason, plan, use tools, and take actions autonomously. Goes beyond Q&A — it can call APIs, search databases, execute code, and chain multiple steps to complete a goal.

**Copilot** — An AI assistant embedded in a workflow (IDE, email, docs) that suggests or completes work alongside the user. The human stays in control; the AI accelerates.

---

## 5. LLM Application Building Blocks — Tools & Protocols

> These are the terms you'll hear when people talk about how LLM applications are built — the tools, frameworks, and protocols that connect everything together.

**MCP (Model Context Protocol)** — A standard protocol for connecting AI models to external tools and data sources. Think of it as USB-C for AI — one standard interface to plug in any tool.

**Agentic Frameworks** — Libraries and platforms for building AI agents (e.g., LangChain, Amazon Bedrock Agents, CrewAI). Handle orchestration, memory, and tool calling.

**Orchestration** — Coordinating multiple AI components — models, tools, retrievers, memory — into a working system. The glue that turns individual capabilities into an application.

---

---

[← Back to home](/)
