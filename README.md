# LLM Summarization Evaluation Framework

> A model-agnostic, research-grade pipeline for evaluating Large Language Model summarization quality across factual accuracy, hallucination, recall, stability, latency, and token efficiency.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Pipeline Architecture](#pipeline-architecture)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Metrics Reference](#metrics-reference)
- [Extending the Framework](#extending-the-framework)
- [Experimental Results](#experimental-results)
- [Experiment Designs](#experiment-designs)
- [Roadmap](#roadmap)
- [License](#license)

---

## Problem Statement

Large Language Models are widely deployed for summarization, yet most evaluations rely on surface-level metrics like ROUGE or BERTScore that correlate poorly with factual faithfulness. A model can achieve a high ROUGE score while hallucinating critical facts, omitting key entities, or producing output that is inconsistent across runs.

This framework addresses the need for a **structured, reproducible, factuality-first evaluation pipeline** that works across any LLM provider and produces interpretable, structured output rather than a single aggregate score.

---

## Approach

The pipeline decomposes summarization quality into measurable, independently-verifiable dimensions:

1. **Factuality** — Are the claims the model makes supported by the source?
2. **Hallucination** — How often does the model assert things contradicted by or absent from the source?
3. **Recall** — How much of the source's information did the summary preserve?
4. **Stability** — Does the model produce consistent summaries across multiple runs?
5. **Efficiency** — What is the latency and token cost of generation?

Each dimension is computed via a dedicated module, making it easy to swap, tune, or extend any part of the pipeline independently.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT: Source Text                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
             ┌──────────────────────────────────┐
             │         Summarizer Module         │
             │   (any BaseLLM adapter)           │
             │   → summary text                  │
             │   → token usage, latency          │
             └────────────────┬─────────────────┘
                              │
              ┌───────────────┴──────────────┐
              │                              │
              ▼                              ▼
 ┌────────────────────────┐    ┌─────────────────────────────┐
 │   Claim Extractor      │    │  Reference Fact Extractor   │
 │  (LLM prompt → JSON)  │    │  (LLM prompt → JSON)        │
 │  → [claim₁, claim₂…] │    │  → [fact₁, fact₂, fact₃…]  │
 └──────────┬─────────────┘    └──────────────┬──────────────┘
            │                                 │
            ▼                                 │
 ┌──────────────────────────┐                 │
 │     NLI Verifier         │                 │
 │  (cross-encoder or LLM)  │◄────────────────┘
 │                          │  (verifies recall direction too)
 │  claim ← source text     │
 │  → SUPPORTED / REFUTED   │
 │  → UNVERIFIABLE          │
 └──────────┬───────────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                      Metrics Module                          │
 │                                                              │
 │  FactScore    = supported / total_claims                     │
 │  Halluc. rate = (refuted + unverifiable) / total_claims      │
 │  Recall       = reference_facts_covered / total_ref_facts    │
 │  Omission     = 1 - Recall                                   │
 │  Stability    = mean pairwise Jaccard (across N runs)        │
 │  Latency      = wall-clock seconds for generation            │
 │  Tokens       = prompt + completion token counts             │
 └──────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │     EvaluationResult (JSON)   │
            │   model, summary, metrics,    │
            │   claims, verifications,      │
            │   hallucinated, omitted…      │
            └───────────────────────────────┘
```

---

## Repository Structure

```
llm-eval-framework/
│
├── src/                          # All importable source code
│   ├── llms/                     # LLM abstraction layer
│   │   ├── base.py               # BaseLLM + LLMResponse dataclass
│   │   ├── ollama_adapter.py     # Ollama (local)
│   │   ├── openai_adapter.py     # OpenAI GPT
│   │   ├── anthropic_adapter.py  # Anthropic Claude
│   │   ├── hf_adapter.py         # HuggingFace (local transformers)
│   │   └── __init__.py           # get_llm() factory
│   │
│   ├── modules/                  # Pipeline processing modules
│   │   ├── summarizer.py         # Summary generation
│   │   ├── claim_extractor.py    # Atomic claim decomposition
│   │   ├── fact_extractor.py     # Reference fact extraction
│   │   └── nli_verifier.py       # NLI-based claim verification
│   │
│   ├── metrics/
│   │   └── metrics.py            # All metric computation functions
│   │
│   ├── pipeline/
│   │   └── orchestrator.py       # EvaluationPipeline (main engine)
│   │
│   ├── benchmark/
│   │   └── runner.py             # BenchmarkRunner (multi-model)
│   │
│   └── utils/
│       ├── config.py             # YAML/JSON config loader
│       └── logging_setup.py      # Structured logging
│
├── configs/
│   ├── default.yaml              # Single-model config
│   └── benchmark.yaml            # Multi-model benchmark config
│
├── prompts/
│   ├── summarize.txt             # Summarization prompt template
│   ├── extract_claims.txt        # Claim extraction prompt
│   └── extract_facts.txt         # Reference fact extraction prompt
│
├── data/
│   └── samples/                  # Sample input documents
│       ├── jwst_article.txt
│       └── fed_rate_article.txt
│
├── outputs/                      # All evaluation results (gitignored)
│
├── experiments/
│   ├── model_comparison.py       # Multi-model, multi-doc benchmark
│   ├── prompt_ablation.py        # Prompt variant sensitivity analysis
│   ├── failure_analysis.py       # Structured hallucination/omission audit
│   └── prompt_variants/          # Alternative prompt files for ablation
│
├── tests/
│   ├── unit/                     # Fast, no-network tests
│   │   ├── test_metrics.py
│   │   ├── test_llm_adapters.py
│   │   └── test_extractors.py
│   └── integration/
│       └── test_pipeline.py      # Full pipeline with mock LLM
│
├── run_pipeline.py               # CLI entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

**Why this structure?**

- `src/llms/` is isolated so the adapter layer can be tested or swapped without touching the pipeline.
- `src/modules/` contains pure processing logic — each module is independently testable.
- `src/pipeline/` owns only orchestration — it composes modules but adds no business logic itself.
- `configs/` and `prompts/` are separated from code so non-engineers can tune behaviour without touching Python.
- `experiments/` are standalone scripts, not part of the library, making them reproducible without import complexity.

---

## Features

- **Model-agnostic** — plug in Ollama, OpenAI, Claude, or any HuggingFace model with one config change
- **Two NLI backends** — fast local cross-encoder (BART/DeBERTa) or LLM-based verification
- **Atomic claim decomposition** — summaries are broken into individually-verifiable facts before scoring
- **Recall evaluation** — measures what the source contains that the summary missed, not just what the summary asserts
- **Stability measurement** — quantifies output variance across multiple runs via pairwise Jaccard similarity
- **Structured JSON output** — every run produces a fully serialisable result with claims, verdicts, and metrics
- **Config-driven** — all model settings, prompt paths, and eval parameters live in YAML files
- **Clean CLI** — `run`, `benchmark`, and `report` subcommands
- **Reproducible experiments** — experiment scripts are self-contained with explicit configs and data paths
- **17 unit + integration tests** — covering metrics, parsers, config loading, and the full pipeline

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/4mritz/llm-eval-framework.git
cd llm-eval-framework

# 2. Install dependencies
pip install -r requirements.txt

# 3. Evaluate a single model (requires Ollama running locally)
python run_pipeline.py run \
  --provider ollama \
  --model llama3.1:8b-instruct-q4_K_M \
  --input data/samples/jwst_article.txt

# 4. Benchmark multiple models
python run_pipeline.py benchmark \
  --config configs/benchmark.yaml \
  --input data/samples/jwst_article.txt

# 5. View an existing result
python run_pipeline.py report --result outputs/llama3.1-8b_result.json
```

---

## Installation

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) (for local models) **or** API keys for OpenAI / Anthropic

### Install core dependencies

```bash
pip install -r requirements.txt
```

### Install optional provider packages

```bash
# OpenAI
pip install openai

# Anthropic Claude
pip install anthropic

# HuggingFace local models
pip install transformers torch accelerate
```

### Set API keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Install Ollama models

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull mistral:7b-instruct
```

---

## Usage

### Evaluate a single model

```bash
python run_pipeline.py run \
  --provider ollama \
  --model llama3.1:8b-instruct-q4_K_M \
  --input data/samples/jwst_article.txt \
  --output outputs/
```

### Evaluate with OpenAI

```bash
python run_pipeline.py run \
  --provider openai \
  --model gpt-4o-mini \
  --input data/samples/jwst_article.txt
```

### Run a multi-model benchmark

```bash
python run_pipeline.py benchmark \
  --config configs/benchmark.yaml \
  --input data/samples/jwst_article.txt \
  --output outputs/my_benchmark/
```

### Print a formatted report from a saved result

```bash
python run_pipeline.py report --result outputs/my_benchmark/report.json
```

### Run tests

```bash
pytest tests/ -v
```

### Use the Python API directly

```python
from src.llms import get_llm
from src.modules.nli_verifier import TransformersNLIVerifier
from src.pipeline.orchestrator import EvaluationPipeline

llm = get_llm("ollama", model_name="llama3.1:8b-instruct-q4_K_M")
verifier = TransformersNLIVerifier()

pipeline = EvaluationPipeline(
    summarizer_llm=llm,
    extractor_llm=llm,
    verifier=verifier,
)

with open("data/samples/jwst_article.txt") as f:
    source = f.read()

result = pipeline.run(source)
print(result.to_json())
```

---

## Configuration

### `configs/default.yaml` — single model evaluation

```yaml
llm:
  provider: ollama
  model_name: "llama3.1:8b-instruct-q4_K_M"
  temperature: 0.3
  max_tokens: 512

extractor:
  provider: ollama
  model_name: "llama3.1:8b-instruct-q4_K_M"
  temperature: 0.1
  max_tokens: 1024

verifier:
  backend: transformers           # "transformers" | "llm"
  model_name: "facebook/bart-large-mnli"
  device: cpu

evaluation:
  n_runs: 1
  target_words: 150
```

### `configs/benchmark.yaml` — multi-model comparison

```yaml
models:
  - provider: ollama
    model_name: "llama3.1:8b-instruct-q4_K_M"
  - provider: openai
    model_name: "gpt-4o-mini"
  - provider: anthropic
    model_name: "claude-3-5-haiku-20241022"

evaluation:
  n_runs: 3    # run each model 3× to measure stability
```

---

## Metrics Reference

| Metric | Formula | Interpretation |
|---|---|---|
| **FactScore** | `supported_claims / total_claims` | Higher is better. 1.0 = fully faithful. |
| **Hallucination Rate** | `(refuted + unverifiable) / total_claims` | Lower is better. Includes uncertain claims. |
| **Refuted Rate** | `refuted / total_claims` | Strict hallucination: claims the source contradicts. |
| **Unverifiable Rate** | `unverifiable / total_claims` | Claims neither confirmed nor denied by source. |
| **Recall** | `recalled_facts / total_reference_facts` | Higher is better. Fraction of source facts preserved. |
| **Omission Rate** | `1 - recall` | Lower is better. Facts the summary missed entirely. |
| **Stability (Jaccard)** | `mean pairwise token-Jaccard across N runs` | Higher is better. Measures output consistency. |
| **Latency** | Wall-clock seconds for generation | Lower is better. |
| **Total Tokens** | `prompt_tokens + completion_tokens` | Efficiency indicator. |

---

## Example Output

```json
{
  "model_name": "llama3.1:8b-instruct-q4_K_M",
  "summary": "The James Webb Space Telescope, launched December 25 2021 at a cost of $10 billion, operates from the L2 Lagrange point 1.5 million km from Earth...",
  "claims": [
    "JWST launched on December 25, 2021.",
    "The telescope cost approximately $10 billion.",
    "JWST orbits at the L2 Lagrange point."
  ],
  "supported_claims": [
    "JWST launched on December 25, 2021.",
    "The telescope cost approximately $10 billion.",
    "JWST orbits at the L2 Lagrange point."
  ],
  "refuted_claims": [],
  "unverifiable_claims": [],
  "metrics": {
    "fact_score": 1.0,
    "hallucination_rate": 0.0,
    "refuted_rate": 0.0,
    "unverifiable_rate": 0.0,
    "recall": 0.75,
    "omission_rate": 0.25,
    "latency_seconds": 4.21,
    "prompt_tokens": 312,
    "completion_tokens": 187,
    "total_tokens": 499,
    "n_claims": 3,
    "n_supported": 3,
    "n_reference_facts": 8,
    "n_recalled_facts": 6
  }
}
```

### Benchmark Comparison Table (example output)

```
================================================================================
BENCHMARK COMPARISON
================================================================================
Model                               FactScore  Halluc%   Recall   Latency   Tokens
--------------------------------------------------------------------------------
claude-3-5-haiku-20241022               0.923    0.077    0.875     1.84s     642
gpt-4o-mini                             0.889    0.111    0.812     2.11s     589
llama3.1:8b-instruct-q4_K_M             0.800    0.200    0.750     4.21s     499
mistral:7b-instruct                     0.762    0.238    0.687     5.03s     521
================================================================================
```

---

## Extending the Framework

### Add a new LLM provider

1. Create `src/llms/myprovider_adapter.py`
2. Subclass `BaseLLM` and implement `generate()` returning an `LLMResponse`
3. Register it in `src/llms/__init__.py`:

```python
from .myprovider_adapter import MyProviderLLM
_REGISTRY["myprovider"] = MyProviderLLM
```

4. Use it immediately:

```bash
python run_pipeline.py run --provider myprovider --model my-model-v1 --input data/samples/jwst_article.txt
```

### Swap the NLI backend

Replace `TransformersNLIVerifier` with `LLMNLIVerifier` in your config:

```yaml
verifier:
  backend: llm
  provider: ollama
  model_name: "llama3.1:8b-instruct-q4_K_M"
```

Or pass it directly in Python:

```python
from src.modules.nli_verifier import LLMNLIVerifier
verifier = LLMNLIVerifier(llm=get_llm("openai", model_name="gpt-4o-mini"))
```

### Use a different NLI model

```yaml
verifier:
  backend: transformers
  model_name: "cross-encoder/nli-deberta-v3-large"  # more accurate, slower
  device: cuda
```

### Add a custom prompt

Edit or replace any file in `prompts/`. The templates use Python `.format()` syntax with named variables (`{text}`, `{target_words}`).

### Run on a dataset (CNN/DailyMail, XSum, etc.)

```python
from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:100]")
for item in dataset:
    result = pipeline.run(item["article"])
    result.save(f"outputs/cnn/{item['id']}.json")
```

---

## Experimental Results

> Results below are illustrative. Run `experiments/model_comparison.py` with your own models to populate this section.

### Model Comparison — JWST Article

| Model | FactScore ↑ | Halluc% ↓ | Recall ↑ | Stability ↑ | Latency ↓ | Tokens |
|---|---|---|---|---|---|---|
| claude-3-5-haiku | **0.923** | **0.077** | **0.875** | 0.84 | 1.84s | 642 |
| gpt-4o-mini | 0.889 | 0.111 | 0.812 | **0.91** | 2.11s | 589 |
| llama3.1:8b q4 | 0.800 | 0.200 | 0.750 | 0.76 | 4.21s | **499** |
| mistral:7b | 0.762 | 0.238 | 0.687 | 0.71 | 5.03s | 521 |

### Key Observations

- API models (Claude, GPT) show substantially lower hallucination rates than local 7-8B models on factually dense articles.
- Stability scores reveal that local models produce more variable output across runs — important for production reliability assessments.
- Token efficiency favours local models when cost is factored in: Llama achieves 0.80 FactScore at zero marginal API cost.
- Recall gaps (0.75 vs 0.875) indicate local models are more likely to omit secondary facts, not just hallucinate primary ones.

---

## Experiment Designs

### 1. Model Comparison

Evaluate N models on M documents, aggregate per-model averages, rank by FactScore.

```bash
python experiments/model_comparison.py --config configs/benchmark.yaml
```

### 2. Prompt Ablation

Test how summarization prompt wording affects factuality and recall.
Place prompt variants in `experiments/prompt_variants/` and run:

```bash
python experiments/prompt_ablation.py \
  --input data/samples/jwst_article.txt \
  --provider ollama \
  --model llama3.1:8b-instruct-q4_K_M
```

Dimensions to vary: instruction strictness, length target, style constraint, chain-of-thought priming.

### 3. Failure Analysis

Load benchmark outputs and produce a structured audit of hallucinated claims and omitted facts per model:

```bash
python experiments/failure_analysis.py \
  --results-dir outputs/experiments/model_comparison/benchmark_jwst_article_<timestamp>/
```

Produces a Markdown report categorising failures by model and claim type.

---

## Roadmap

- [ ] ROUGE / BERTScore integration for surface-level baseline comparison
- [ ] Cost tracking with per-model USD estimates in benchmark reports
- [ ] Caching layer for LLM responses (hash-keyed, SQLite-backed)
- [ ] Dataset connectors: CNN/DailyMail, XSum, PubMed via HuggingFace `datasets`
- [ ] HTML/Markdown benchmark report generation with charts
- [ ] Evaluator bias mitigation: use a separate model family for extraction and verification
- [ ] Streaming support for real-time token counting
- [ ] FastAPI wrapper for serving the pipeline as an evaluation endpoint

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built for reproducible, factuality-first LLM evaluation. Contributions welcome.*
