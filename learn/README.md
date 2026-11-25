# Post-Training Liquid Foundation Models: a course

This course teaches you how to post-train Liquid AI's LFM models: LFM2, LFM2.5,
LFM2-VL, the LFM2 MoE models and LFM3. You learn it with the tool Liquid
actually ships, this repo (`liquid-finetune`). Each lesson has three parts:

1. **Concepts**: the ideas, explained from first principles.
2. **Lab**: a script that runs **on a CPU laptop in under a minute**. It uses
   tiny randomly initialized LFM models, so you see the real code paths
   without needing a GPU.
3. **In the real repo**: the exact files and YAML knobs where the concept
   lives, and how to run it at full scale on a GPU.

> Post-training is everything that happens **after** pre-training to turn a
> base model, which only predicts the next token of web text, into a useful
> product. That means instruction following, chat, tool use, reasoning,
> preferences, safety, a specific domain, and finally a small, fast,
> quantized file that runs on a device.

---

## Setup for the labs (CPU is fine)

Pick one option:

```bash
# Option A: the repo's own environment (Python 3.12, works on Linux and macOS)
uv sync
uv run python learn/01_architecture/lab_inspect_lfm2.py
```

```bash
# Option B: a light venv just for the labs
python3 -m venv .venv-learn
.venv-learn/bin/pip install -r learn/requirements.txt
.venv-learn/bin/python learn/01_architecture/lab_inspect_lfm2.py
```

For the **GPU projects** (real LFM checkpoints from Hugging Face), the easiest
route without your own GPU is the repo's **Modal** backend. You run one command
from your laptop and it rents an H100 by the minute (see Module 4). Any single
24 GB+ GPU (Colab, RunPod, Lambda) also works. On exactly one visible GPU the
repo uses the plain Hugging Face Trainer, with no Ray or cluster needed.

---

## Syllabus

Status key: ✅ written (lesson and lab ready) · 🔜 next up · 📋 planned

| # | Module | You will be able to… | Status |
|---|--------|----------------------|--------|
| 1 | [**LFM architecture**](01_architecture/LESSON.md) | Explain the hybrid short-conv + GQA design, why it's fast on-device, and which weights LoRA touches | ✅ |
| 2 | [**Chat templates & loss masking**](02_chat_templates_and_masking/LESSON.md) | Read the LFM2/LFM2.5 templates, predict exactly which tokens get gradient, and avoid the classic silent bugs | ✅ |
| 3 | [**SFT & LoRA from the inside**](03_sft_and_lora/LESSON.md) | Do pretrain → SFT → merge by hand, and explain the LoRA math, `r`, `alpha`, and merging | ✅ |
| 4 | **Your first real run with `liquid-finetune`** | Write a YAML config (`extends`), validate data, launch LFM2-350M/1.2B SFT locally or on Modal, read loss curves, resume and export | 🔜 |
| 5 | **Data is the model** | Build and clean SFT data: formats, dedup, decontamination, mixing, synthetic data, tool-calling and `<think>` data | 📋 |
| 6 | **Preference optimization: DPO & KTO** | Derive the DPO loss and explain `beta`, the reference model, and why the LR is 1e-6; when to use KTO instead | 📋 |
| 7 | **Reinforcement learning: GRPO** | Compute group-relative advantages, write reward functions and recipes, use an LLM judge, set up vLLM rollouts, and spot reward hacking | 📋 |
| 8 | **Evaluation** | Run in-training, async and standalone evals; avoid contamination; build a task-specific eval before training | 📋 |
| 9 | **Vision-language models (LFM2-VL / LFM3-VL)** | Explain image tokens and tiling, freeze the vision tower, use per-component LRs, and run VLM SFT/DPO/GRPO | 📋 |
| 10 | **Mixture-of-Experts (LFM2-8B-A1B, 24B-A2B)** | Explain routers, active vs. total params, expert parallelism, and MoE-specific LoRA | 📋 |
| 11 | **Shipping it: merge, quantize, QAT, GGUF** | Merge adapters, export GGUF (Q4_K_M…), apply quantization-aware training, and run on llama.cpp / vLLM | 📋 |
| 12 | **Retrieval models (embedding & ColBERT)** | Fine-tune LFM embedding/ColBERT models with in-batch negatives | 📋 (optional) |
| 13 | **Capstone** | Ship a specialized on-device assistant end to end: SFT → DPO → GRPO → QAT → GGUF → eval | 📋 |

### How the stages fit together

```
            ┌───────────── what you change ─────────────┐
 pre-trained│  SFT            DPO / KTO        GRPO (RL) │  QAT + GGUF    on-device
 base model ├──────────► ────────────────► ──────────────┼──────────────► app
  (LFM2…)   │ "how to talk"  "what's better"  "what works"│  "make it small"
            └────────────────────────────────────────────┘
 data:        demonstrations  pairs (chosen/   prompts + a
              (messages)      rejected)         reward function
 repo key:    training_type:  dpo | kto         grpo | vlm_grpo
              sft | vlm_sft   vlm_dpo
```

## How to study with me

- Do one module at a time: read `LESSON.md`, run the lab, do the exercises,
  then answer the **checkpoint questions** at the bottom. Send me your answers
  and I'll grade them and fill the gaps.
- Change the labs. Every exercise asks you to modify one thing and predict the
  outcome **before** you run it. That habit makes good ML engineers.
- When you're ready for the next module, ask me and I'll write it.
