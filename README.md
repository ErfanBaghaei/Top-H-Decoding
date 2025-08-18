# Top-H Decoding

**Top-H** is a **training-free** decoding method that balances **creativity** and **coherence** in open-ended text generation by **constraining entropy** at each step. It solves an **entropy-constrained mass maximization** problem with an efficient greedy procedure, yielding robust, high-temperature generations that remain coherent.

> 📄 This repository accompanies our paper:  
> **Top-H Decoding: Adapting the Creativity and Coherence with Bounded Entropy in Text Generation**

---

## 🧭 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Results Summary](#-results-summary)
- [Installation](#-installation)
- [Reproducing Paper Results](#-reproducing-paper-results)
- [Method Details](#-method-details)
- [Tuning & Tips](#-tuning--tips)
- [Supported Models](#-supported-models)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🚀 Overview

Classic truncated sampling (temperature, top-k, top-p, **min-p**) trades off diversity vs. coherence but often ignores the **shape** of the next-token distribution. **Top-H** makes this trade-off explicit by **upper-bounding the entropy** of the truncated distribution relative to the original model distribution — exploring more when the model is unsure, and tightening when it is confident.

At a glance:
- Formulates **Entropy-Constrained Minimum Divergence (ECMD)** and proves equivalence to **Entropy-Constrained Mass Maximization (ECMM)** (NP-hard).
- Introduces a **greedy** approximation (**Top-H**) with a simple **termination guarantee** controlled by an entropy scale **α**.
- Delivers strong empirical gains over **min-p** and **top-p**, especially at **higher temperatures**.

---

## 🧠 Key Features

- 🛠 **Training-free & model-agnostic** — drop-in decoding; no fine-tuning.
- 🎛 **Entropy-aware truncation** — caps randomness via **H(q) ≤ α·H(p)**, recalculated **every step**.
- 🧮 **Theory-backed** — ECMD ⇔ ECMM (NP-hard); practical greedy rule with early-stop criterion.
- 🔥 **Robust at high temperature** — maintains coherence where min-p/top-p degrade.
- 🧪 **Wide evaluation** — creative writing (e.g., Alpaca-Eval, MT-Bench) and QA (GPQA, GSM8K).

---

## 📊 Results Summary

- On creative writing benchmarks, **Top-H** outperforms SoTA alternatives by **up to 25.63%**, while preserving consistency.
- On reasoning datasets (**GSM8K**, **GPQA**), Top-H remains robust at elevated temperatures.

### Example (from paper)

| Benchmark / Model / T           | min-p | top-p | **Top-H** |
|---------------------------------|------:|------:|----------:|
| **GSM8K** — LLaMA-3.1-8B — T=2 | 13.72 |  2.65 | **39.35** |
| **GPQA** — Phi-3-Mini — T=2     | 23.44 | 18.53 | **30.80** |

> See the paper for full tables, settings, and ablations.

---

## 📦 Installation

```bash
git clone https://github.com/your-org/top-h-decoding.git
cd top-h-decoding
pip install -r requirements.txt
```

---

## 🔬 Reproducing Paper Results

Install dependencies, then run the evaluation script(s):

```
pip install -r requirements.txt
bash alpaca_evaluate.sh
```

