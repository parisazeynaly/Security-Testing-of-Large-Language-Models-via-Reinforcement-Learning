# CausalRLBreaker

### Causally Guided Reinforcement Learning for LLM Security Testing

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Research Project](https://img.shields.io/badge/Project-Research%20Code-blue)](https://github.com/parisazeynaly/Security-Testing-of-Large-Language-Models-via-Reinforcement-Learning)

Research implementation accompanying the M.Sc. thesis:

> **Security Testing of Large Language Models via Causal Reinforcement Learning**

University of Naples Federico II

---

## Overview

CausalRLBreaker is a research framework for automated black-box security
testing of large language models using reinforcement learning and structural
causal information.

The framework extends RL-based adaptive prompt search with an interpretable
six-factor representation derived from red-teaming trajectories. Structural
relationships among these factors are estimated using Fast Causal Inference
(FCI) and represented through a Partial Ancestral Graph (PAG).

A Structural Causal Model (SCM) is then used to provide model-based structural
guidance during reinforcement learning through intermediate feedback and
causal-graph-guided action selection.

The framework combines:

- PPO-based adaptive prompt search
- COAT-based factor discovery and annotation
- Fast Causal Inference (FCI)
- Partial Ancestral Graph (PAG) representation
- Structural Causal Model (SCM)-informed feedback
- Causal-graph-guided prompt mutation
- Automated LLM security evaluation

The causal components are used as **model-based structural guidance**.
SCM-derived intervention scores should not be interpreted as experimentally
identified causal effects.

---

## Research Question

> **Can structural causal information improve the effectiveness and resource
> efficiency of reinforcement-learning-based black-box security testing of
> large language models?**

---

## Method

### 1. Adaptive Prompt Search

CausalRLBreaker formulates automated red-teaming as a sequential decision
problem.

A PPO agent selects prompt-transformation actions and receives feedback from
the target LLM and automated security evaluator.

---

### 2. Structural Factor Discovery

Interaction trajectories are transformed into an interpretable structural
representation using a COAT-based factor-discovery procedure.

FCI is applied to the resulting observations to identify conditional
dependence structure and construct a Partial Ancestral Graph.

Six factors are retained:

1. Instructional style
2. Responsibility externalization
3. Obfuscation techniques
4. Hypothetical framing
5. Imperative tone
6. Malicious intent

---

### 3. Runtime State Representation

At runtime, each adversarial prompt is mapped to a six-dimensional state:

\[
s_t \in [0,1]^6.
\]

A lightweight deterministic lexical factor extractor estimates the activation
of each factor and maps the resulting evidence to bounded continuous values.

This provides the RL agent with an interpretable factor-level representation
of prompt transformations.

---

### 4. SCM-Informed Intermediate Feedback

The structural representation is incorporated into a Structural Causal Model.

The SCM provides model-based predictions that are used as intermediate
feedback during adaptive prompt search, supplementing the sparse final
security outcome.

---

### 5. Causal-Graph-Guided Action Selection

Candidate prompt transformations are evaluated using SCM-internal simulated
interventions on their associated factors.

For a factor \(f\), the intervention score is based on

\[
\Delta_f =
SCM(s_t \mid do(f=1)) - SCM(s_t).
\]

Scores are aggregated across the factors associated with each mutation.

The resulting structural scores guide action selection while retaining
stochastic use of the PPO proposal.

This guidance operates at the environment/action-selection level and does
**not** modify or mask PPO logits.

---

## High-Level Pipeline

```text
              Initial Adversarial Prompt
                         │
                         ▼
                 PPO Action Proposal
                         │
                         ▼
               Prompt Transformation
                         │
                         ▼
               Six-Factor Extraction
                         │
                         ▼
                Structural Guidance
                 ┌───────┴───────┐
                 ▼               ▼
             FCI / PAG           SCM
                                 │
                       ┌─────────┴─────────┐
                       ▼                   ▼
               Intermediate          Intervention
                  Feedback              Scores
                       └─────────┬─────────┘
                                 ▼
                         Action Selection
                                 │
                                 ▼
                            Target LLM
                                 │
                                 ▼
                       Security Evaluation
                                 │
                                 ▼
                              Reward
```

---

## Key Results

Under the reported experimental configuration:

| Metric | Result |
|---|---:|
| **CausalRLBreaker ASR** | **61.54% ± 1.67%** |
| Archived RLBreaker ASR | 19.33% ± 3.21% |
| DAN baseline ASR | 18.27% ± 2.88% |
| API-call reduction vs. RLBreaker | **41.82%** |
| Token reduction vs. RLBreaker | **61.44%** |
| Training-time reduction vs. RLBreaker | **64.93%** |

### Evaluation Note

The archived RLBreaker evaluation and the primary CausalRLBreaker evaluation
are not fully prompt-aligned.

The headline ASR comparison between these experiments is therefore treated as
a **descriptive comparison rather than a paired statistical comparison**.

---

## Causal Representation Analysis

A representation-level ablation evaluates how much predictive information is
contributed by each of the six discovered factors.

The full representation achieved a mean ROC-AUC of approximately **0.736**
across repeated cross-validation partitions.

The largest performance reduction after factor removal was observed for
**hypothetical framing**, followed by **obfuscation techniques**.

These results characterize the predictive contribution of the representation
and should not be interpreted as independent causal-effect estimates.

---

## Experimental Configuration

The primary experiments use:

- **RL algorithm:** PPO
- **Training budget:** 10,240 environment steps
- **Dataset:** AdvBench
- **Split:** 80/20 with `random_state=42`
- **Target model:** Llama-3.3-70B-Versatile
- **Target temperature:** 0.7
- **Target generation limit:** 150 tokens
- **Mutator model:** Llama-3.1-8B-Instant
- **Mutator temperature:** 0.7
- **Mutator generation limit:** 200 tokens
- **Judge model:** Llama-3.3-70B-Versatile
- **Judge temperature:** 0
- **Judge generation limit:** 100 tokens
- **FCI independence test:** Fisher's Z
- **FCI significance level:** 0.05

---

## Repository Structure

```text
Security-Testing-of-Large-Language-Models-via-Reinforcement-Learning/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .env.example
│
├── src/                    # Refactored implementation modules
│   ├── causal/             # SCM and structural guidance
│   ├── LLM/                # LLM clients and evaluation
│   └── mutation.py         # Prompt mutations and guided action selection
├── RLBreaker.ipynb         # Baseline research notebook
├── Causal RLbreaker/       # Historical experiment artifacts
├── causal_rl_shaping/      # Historical shaping experiments
├── empirical_evaluation/   # Evaluation and analysis artifacts
├── data/
└── docs/
```

The repository retains historical experiment directories to preserve the
research record. The `src/` tree is the cleaner, modularized implementation
for code inspection and continued reproducibility work. Historical artifacts
should not be assumed to represent the final manuscript configuration.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/parisazeynaly/Security-Testing-of-Large-Language-Models-via-Reinforcement-Learning.git

cd Security-Testing-of-Large-Language-Models-via-Reinforcement-Learning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment configuration:

```bash
cp .env.example .env
```

Add your own authorized API credentials to `.env`.

**Never commit API keys or credentials to the repository.**

> **Configuration note:** the current Python implementation defines model-role
> configuration in `src/LLM/clients.PY`. The `.env.example` file is kept
> intentionally limited to credentials so that it does not imply runtime
> configuration that the code does not actually consume.

---

## Reproducibility

The experiments combine stochastic reinforcement learning with external LLM
API calls.

Exact numerical reproduction may therefore depend on:

- random seeds,
- API/model availability,
- provider-side model versions,
- generation configuration, and
- external service behavior.

The repository preserves the experimental implementation and configuration
artifacts needed to inspect the reported pipeline. Some historical files
contain exploratory configurations; the manuscript should be treated as the
source of truth for reported experimental settings. Exact reproduction also
depends on external model/API availability and provider-side model versions.

---

## Citation

If you use this repository in academic work, please cite the accompanying
CausalRLBreaker manuscript.

Machine-readable citation metadata is available in:

[`CITATION.cff`](CITATION.cff)

The public preprint link will be added after release.

---

## Responsible Use

This repository contains methods for adversarial prompting and automated LLM
security testing.

It is intended exclusively for:

- authorized security evaluation,
- controlled red-teaming research,
- AI safety research, and
- defensive analysis of language-model vulnerabilities.

Do not use this repository to facilitate harmful activity or unauthorized
attacks against deployed systems.

---

## License

Released under the [Apache License 2.0](LICENSE).
