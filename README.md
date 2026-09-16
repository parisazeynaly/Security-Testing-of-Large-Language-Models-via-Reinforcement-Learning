# CausalRLBreaker

**Causally Guided Reinforcement Learning for LLM Security Testing**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Research Project](https://img.shields.io/badge/Project-Research%20Code-blue)](https://github.com/parisazeynaly/Security-Testing-of-Large-Language-Models-via-Reinforcement-Learning)

Research implementation accompanying the M.Sc. thesis **Security Testing of Large Language Models via Causal Reinforcement Learning** at the University of Naples Federico II.

> **Research-use notice:** This repository studies automated red-teaming and security evaluation of LLMs in controlled settings. It is intended for defensive and authorized research.

## Overview

CausalRLBreaker studies whether structural information discovered from red-teaming trajectories can guide reinforcement-learning-based prompt search more effectively and efficiently. The framework combines PPO-based adaptive search with an interpretable six-factor prompt representation, Fast Causal Inference (FCI), a Partial Ancestral Graph (PAG), Structural Causal Model (SCM)-informed intermediate feedback, and causal-graph-guided action selection.

The causal components are used as **model-based structural guidance** for search. They should not be interpreted as experimentally identified causal effects of prompt characteristics.

## Research Question

**Can structural causal information improve the effectiveness and resource efficiency of reinforcement-learning-based black-box security testing of large language models?**

## Method

The experimental pipeline is organized around four stages:

1. **Adaptive prompt search.** PPO selects prompt-transformation actions in a black-box red-teaming environment.
2. **Interpretable factor representation.** Interaction trajectories are transformed into six prompt-level factors through a COAT-based factor-discovery procedure and FCI structural discovery.
3. **SCM-informed guidance.** The learned structural representation supplies intermediate feedback and model-based intervention scores for candidate transformations.
4. **Security evaluation.** Generated prompts are evaluated against the target LLM using an automated judge and attack-success metrics.

### Six-factor representation

The state representation contains:

- instructional style
- responsibility externalization
- obfuscation techniques
- hypothetical framing
- imperative tone
- malicious intent

At runtime, these dimensions are estimated with a lightweight deterministic lexical factor extractor and mapped to bounded activations in `[0,1]`.

### Causal-graph-guided action selection

For candidate mutation actions, the framework evaluates SCM-internal simulated interventions on the factors associated with each action. The resulting scores guide action selection while retaining stochastic use of the PPO proposal. This mechanism operates at the environment/action-selection level; it does **not** mask PPO logits.

## Key Results

Under the reported experimental configuration, CausalRLBreaker achieved:

| Metric | Result |
|---|---:|
| CausalRLBreaker ASR | **61.54% ± 1.67%** |
| Archived RLBreaker ASR | 19.33% ± 3.21% |
| DAN baseline ASR | 18.27% ± 2.88% |
| API-call reduction vs. RLBreaker | **41.82%** |
| Token reduction vs. RLBreaker | **61.44%** |
| Training-time reduction vs. RLBreaker | **64.93%** |

**Important comparison note.** The archived RLBreaker evaluation and the primary CausalRLBreaker evaluation are not fully prompt-aligned. Their headline ASR comparison is therefore reported descriptively rather than as a paired statistical comparison.

A separate representation-level ablation evaluates the predictive contribution of the six-factor representation across repeated cross-validation partitions. These ablations measure **predictive information**, not independently identified causal effects.

## Repository Contents

The repository currently contains the thesis implementation, RLBreaker baseline material, causal-RL experiments, empirical evaluation artifacts, data, documentation, environment configuration, and citation metadata. The codebase is being consolidated into a cleaner reproducibility-oriented layout; some historical experiment directories are retained to preserve the research record.

Key top-level resources include:

- `RLBreaker.ipynb` — RLBreaker baseline notebook
- `Causal RLbreaker/` — causal-guided experimental implementation and artifacts
- `causal_rl_shaping/` — causal reward-shaping components
- `empirical_evaluation/` — empirical evaluation material
- `data/` — project data resources
- `docs/` — supplementary documentation
- `.env.example` — environment-variable template
- `CITATION.cff` — citation metadata

## Installation

Clone the repository and install the project dependencies:

```bash
git clone https://github.com/parisazeynaly/Security-Testing-of-Large-Language-Models-via-Reinforcement-Learning.git
cd Security-Testing-of-Large-Language-Models-via-Reinforcement-Learning
pip install -r requirements.txt
```

Copy the environment template and provide your own authorized API credentials where required:

```bash
cp .env.example .env
```

Never commit API keys or other credentials to the repository.

## Reproducibility

The experiments involve stochastic RL optimization and external LLM API calls. Exact reproduction therefore depends on model availability, API configuration, random seeds, and provider-side model behavior. The repository preserves experimental code and configuration artifacts so that the reported pipeline can be inspected and reproduced as closely as the external dependencies permit.

The primary experiments use a fixed AdvBench train/test split with `random_state=42`. Evaluation results should be interpreted under the exact model and prompt configuration documented by the corresponding experiment.

## Citation

If you use this repository in academic work, please cite the accompanying manuscript/thesis. Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff). The manuscript/preprint link will be added once the public version is available.

## Responsible Use

This project concerns adversarial prompting and LLM security testing. Use the code only for authorized research, controlled evaluation, and defensive security analysis. Do not use it to facilitate harmful activity or unauthorized attacks on deployed systems.

## License

This repository is released under the [Apache License 2.0](LICENSE).
