---
title: "AI Agents — A Builder's Guide to Anatomy, Components, and Patterns"
description: "Applied agentic AI — understand what AI agents are, how they differ from chatbots, and the key architectural patterns including multi-agent systems, SOPs, and workflows."
---

# AI Agents — A Builder's Guide to Anatomy, Components, and Patterns

> In Applied AI, agents are the successor to chatbots. Instead of just answering questions, they reason, plan, and take action. This post covers the key terms you'll encounter in the agentic AI space.

---

## What is an AI Agent?

An AI Agent is a system that uses an LLM as its reasoning engine to autonomously plan and execute multi-step tasks. Unlike a chatbot that responds and waits, an agent:

- Breaks down a goal into steps
- Decides which tools to use
- Executes actions (API calls, code, searches)
- Observes results and adapts
- Repeats until the goal is complete

```
         ┌─────────────────────────────────┐
         │           AI Agent              │
         │                                 │
         │   Goal ──> Plan ──> Act ──┐     │
         │                           │     │
         │      Observe    <─────────┘     │
         │      │                          │
         │      ▼                          │
         │     Done? ──No──> Re-plan       │
         │       │                         │
         │      Yes                        │
         │       │                         │
         │       ▼                         │
         │     Result                      │
         └─────────────────────────────────┘
```

---

## Agentic AI

The broader paradigm of building AI systems that act autonomously rather than just respond. Agentic AI emphasizes:

- **Autonomy** — the system decides what to do next
- **Tool use** — interacting with external systems
- **Persistence** — maintaining state across steps
- **Goal-orientation** — working toward an outcome, not just answering a prompt

### AI Agent vs Agentic AI

**Agentic AI** is the design philosophy — building AI that acts autonomously. **AI Agent** is what you actually build following that philosophy.

Think of it like: "Agentic AI" is to "AI Agent" what "object-oriented programming" is to "a Java class." One is the approach, the other is the artifact.

Real-world examples: Amazon Q Developer Agent and GitHub Copilot Agent mode are AI Agents — they autonomously plan, edit files, run commands, and iterate. Grammarly or Copilot's inline autocomplete are AI-assisted but not agentic — they suggest, they don't act on their own.

---

## Anatomy of an Agent

Before diving into individual components, here's what an agent definition typically looks like. While frameworks differ in syntax, they all define an agent with the same core pieces:

```
Agent
├── Identity        — name, description, version
├── Model           — which LLM powers it (e.g., Claude, GPT-4)
├── Instructions    — system prompt that defines the agent's role/persona and behavior rules
├── Tools           — what it can call (APIs, MCP servers, functions)
├── Knowledge       — context files, skills, documents it can reference
└── Orchestration   — workflows, SOPs, routing logic
```

Think of this as the **agent spec** — a declarative blueprint that says *who* the agent is, *what* it can do, and *how* it should behave. The components below are what fill in each of these slots.

---

## Agent Components

An AI Agent is made up of modular pieces that define what it knows, how it behaves, and what it can do. These are the building blocks you'll assemble when creating an agent.

### Skill

A modular, reusable capability an agent can invoke. A skill encapsulates domain knowledge and instructions for a specific task. A skill is typically composed of:

- **Knowledge/context** — reference docs, examples, domain information
- **Instructions** — how to approach the task
- **SOPs** — step-by-step procedures (optional)
- **Tools** — specific tools the skill requires (optional)

Examples: "search code," "create a PR," "diagnose a build failure"

Think of skills as **what the agent knows how to do**.

### Agent SOP (Standard Operating Procedure)

A structured, step-by-step workflow written in markdown that guides an agent through a complex task. SOPs define:

- The sequence of steps
- Decision points and branching logic
- Required inputs and expected outputs
- Constraints and guardrails

Think of SOPs as **recipes the agent follows**.

### How Skills and SOPs Relate

Skills and SOPs are composable — they can reference each other:

- A **Skill** can contain SOPs (the skill packages knowledge + procedures together)
- An **SOP** can invoke Skills (a step in the procedure might load a specialized skill)

Think of it like: a Skill is a toolbox (contains knowledge, procedures, scripts). An SOP is a recipe (may pull tools from different toolboxes).

### Agent Script

An older term for Agent SOP. Same concept — a predefined procedure an agent executes. The community is converging on "SOP" as the standard term.

### Agent Workflow

A defined sequence of tasks an agent executes, often with dependencies between steps. Workflows can be:

- **Linear** — step 1 → step 2 → step 3
- **Branching** — if X then step A, else step B
- **Parallel** — steps 2 and 3 run simultaneously, step 4 waits for both

```
  ┌─────┐     ┌─────┐     ┌─────┐
  │Step1│────>│Step2│────>│Step4│
  └─────┘  │  └─────┘  ▲  └─────┘
           │           │
           │  ┌─────┐  │
           └─>│Step3│──┘
              └─────┘
         (2 and 3 run in parallel)
```

### SOP vs Workflow

An SOP tells the agent *how to do one thing*. A workflow tells the agent *what things to do and in what order* — and may invoke multiple SOPs along the way.

Think of it like: an SOP is a recipe for one dish. A workflow is a meal plan that sequences multiple recipes.

---

## Multi-Agent Patterns

### Sub-Agent

A child agent spawned by a parent agent to handle a specific subtask. The parent delegates, the sub-agent executes, and returns results to the parent.

```
  ┌──────────────┐
  │ Parent Agent │
  │              │
  │  "Research X"│──────> ┌────────────┐
  │              │        │ Sub-Agent  │
  │  (waits)     │<────── │(researches)│
  │              │        └────────────┘
  │  "Now build" │──────> ┌────────────┐
  │              │        │ Sub-Agent  │
  │              │<────── │ (builds)   │
  └──────────────┘        └────────────┘
```

### Multi-Agent

A system where multiple agents collaborate, each with a specialized role. They can work:

- **In sequence** — one agent's output feeds the next (pipeline)
- **In parallel** — multiple agents work simultaneously on different parts
- **Collaboratively** — agents communicate and coordinate with each other

```
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │Researcher│───>│Implementer│───>│ Reviewer │
  └──────────┘    └──────────┘    └──────────┘
       (pipeline: research → implement → review)
```

---

## Summary

| Term | What it is |
|------|-----------|
| AI Agent | LLM + reasoning + tools + autonomy |
| Agentic AI | The paradigm of autonomous AI systems |
| Skill | A reusable capability an agent can invoke |
| SOP | A step-by-step procedure guiding an agent |
| Agent Script | Older term for SOP |
| Workflow | A defined sequence of tasks with dependencies |
| Sub-Agent | A child agent handling a delegated subtask |
| Multi-Agent | Multiple specialized agents collaborating |

---

[← Back to home](/)
