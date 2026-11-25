# Module 3: SFT and LoRA from the inside

**Goal:** understand supervised fine-tuning and LoRA well enough that the
repo's YAML knobs (`learning_rate`, `r`, `lora_alpha`, `target_modules`,
`use_peft`) feel obvious.

**Lab:** `python learn/03_sft_and_lora/lab_sft_lora.py` (CPU, about 40 seconds)

---

## 3.1 What SFT is

**Supervised fine-tuning** is next-token prediction (the same loss as
pre-training) on **demonstrations** of the behaviour you want, usually chat
transcripts, with the loss masked to the assistant's tokens (Module 2):

```
L(θ) = − Σ_{t ∈ assistant tokens}  log p_θ(token_t | tokens_<t)
```

That's it. No reward, no comparison: "here is a good answer, make it more
likely."

### What SFT teaches, and what it doesn't

The lab shows the most important intuition in post-training:

- **Stage A** pre-trains a toy LFM2 on plain sentences. It "knows" capitals,
  but given a chat prompt it outputs garbage. It has never seen
  `<|im_start|>assistant`.
- **Stage B** runs SFT with LoRA on 9 countries in chat format.
- On **3 held-out countries** that were never in the SFT data, the SFT model
  still answers correctly: `The capital of Ghana is Accra.<|im_end|>`

**SFT mainly teaches format, style and behaviour. Knowledge comes from
pre-training.** That's why a few thousand good examples can turn a base model
into an assistant, and why SFT on facts the base model doesn't know tends to
teach **hallucination**: it learns to *sound* confident about things it can't
look up.

> 🔬 **Exercise 1 (the most important one in this module):** change Stage A
> to `steps=300, lr=3e-3`, which gives a weaker base model, and re-run. The
> held-out countries fail and the model answers *training-set* capitals
> instead. SFT memorized instead of generalizing, because the base never
> learned the underlying skill (copying the country from the question and
> recalling the fact). **SFT can't add what the base model lacks.** This is
> why model choice (350M vs 1.2B vs 2.6B) matters more than most
> hyperparameters.

## 3.2 Full fine-tuning vs. LoRA

**Full fine-tuning** updates every weight. AdamW keeps two extra fp32 states
per parameter, so memory ≈ weights + gradients + 2 × optimizer states. That's
roughly **16 bytes per parameter** in mixed precision, plus activations. For
LFM2-1.2B that's about 19 GB before activations. That's fine on one H100, but
it's why the repo defaults to **DeepSpeed ZeRO-2**, which shards optimizer
state and gradients across GPUs (`DEEPSPEED_ZERO2_WITH_OPTIMIZER_CONFIG` in
`DEFAULT_SFT`).

**LoRA** (Low-Rank Adaptation) freezes the original weight `W` (d_out × d_in)
and learns a low-rank update:

```
h = W x + (alpha / r) · B (A x)        A: r × d_in,  B: d_out × r,  r ≪ d
```

- `B` starts at **zero**, so at step 0 the model is exactly the base model.
  The lab verifies this with `torch.allclose`.
- Trainable params per layer: `r · (d_in + d_out)` instead of `d_in · d_out`.
  In the lab, `conv.in_proj` is 288×96 = 27,648 frozen vs. 8×96 + 288×8 =
  3,072 trainable.
- `alpha / r` is a **scale**. The repo uses `alpha = 2r` (scale 2.0) in both
  `DEFAULT_LORA` (r=8, α=16) and `HIGH_R_LORA` (r=16, α=32). If you double `r`
  but keep `alpha` fixed, the effective step size changes. Keep the ratio
  fixed when you sweep `r`.
- **Merging:** `W' = W + (alpha/r)·B·A`. After `merge_and_unload()` you have a
  plain `Lfm2ForCausalLM` with zero inference overhead. The lab checks the
  logits match to about 1e-6. The repo does this automatically at the end of
  a LoRA run (`merge_and_save_peft_model` in `training/sft.py`).

### When to use which

| Choose **LoRA** when… | Choose **full FT** when… |
|---|---|
| GPU memory is tight | You have the memory (small LFMs often fit) |
| You want to keep the base's general skills (less "forgetting") | You're making large changes (new language, big domain shift, long training) |
| You want many small adapters per customer or task | You want maximum quality on one task |
| Quick iteration | You plan QAT or other whole-model processing |

LoRA's frozen base gives it a mild built-in regularizer: it "learns less and
forgets less." That's often exactly what you want for a narrow product
feature on top of a strong instruct model.

## 3.3 Hyperparameters that matter (and the repo's defaults)

From `src/liquid_finetune/training/default_configs/sft_configs.py::DEFAULT_SFT`:

| Knob | Default | Intuition |
|---|---|---|
| `learning_rate` | `5e-5` (example configs use `2e-5`) | Too high: loss spikes and the model forgets. Too low: nothing changes. LoRA usually tolerates **higher** LRs than full FT (roughly 1e-4 to 2e-4 is common) because B·A starts at zero. Sweep it. |
| `num_train_epochs` | `3` | More epochs on small data means memorization. Watch eval loss rise while train loss keeps falling. |
| `warmup_ratio` | `0.2` | Ramps the LR up so the first noisy steps don't wreck the weights. |
| `lr_scheduler_type` | `linear` | Decay to 0 by the end. |
| `max_length` | `2048` | Longer rows are **truncated**. Check your data's length distribution. |
| `per_device_train_batch_size` | `16` | Effective batch = per_device × GPUs × `gradient_accumulation_steps`. |
| `group_by_length` | `True` | Batches similar lengths together, so less padding and faster runs. |
| `packing` / `padding_free` | `False` | Concatenates short rows into full sequences for speed. |
| `max_grad_norm` | 1.0 (HF default) | Gradient clipping for stability. The lab uses it too. |

LoRA knobs (`peft_config`): `r` (capacity), `lora_alpha` (scale),
`lora_dropout` (0.1 default, a regularizer), and `target_modules` (Module 1:
use GLU + attention + conv for LFM).

## 3.4 Mapping the lab onto the repo

| Lab code | Repo equivalent |
|---|---|
| `render(...)` + `encode(...)` + masks | `data_loading/tokenize_data.py::tokenize_sft` |
| `batchify` (pad, −100 on padding and non-assistant tokens) | `training/sft.py::LFMDataCollatorForLanguageModeling` |
| `LoraConfig(...)` + `get_peft_model` | `peft_config: {extends: DEFAULT_LORA, use_peft: true}` → `training/peft/peft.py::apply_peft_to_model` |
| `train(...)` loop with AdamW and clipping | Hugging Face `Trainer` (`LFMSFTTrainer`) with DeepSpeed/Ray |
| `merge_and_unload()` | `merge_and_save_peft_model` at the end of the run |

The equivalent real config is `job_configs/sft_with_lora_example.yaml`:

```yaml
model_name: "LFM2-1.2B"
training_type: "sft"
dataset: { path: "HuggingFaceTB/smoltalk", type: "sft", limit: 1000, test_size: 0.2, subset: "all" }
training_config:
  extends: "DEFAULT_SFT"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 2e-5
  assistant_only_loss: true      # ← add this for chat data (Module 2!)
peft_config:
  extends: "DEFAULT_LORA"
  use_peft: true
```

You'll run this for real in Module 4.

## 3.5 Exercises

1. **Weak base → no generalization** (see 3.1). Then try to *rescue* it by
   training SFT longer. Does it help? Why not?
2. **No masking:** in `batchify`, supervise *all* tokens for the SFT stage.
   Compare held-out accuracy and what the model generates.
3. **Rank sweep:** try `r=1, 2, 8, 32` with `lora_alpha = 2r`. Print
   trainable params and held-out accuracy. What's the smallest rank that
   still works on this toy task?
4. **LR sweep:** run SFT with `lr ∈ {3e-4, 3e-3, 3e-2}`. Which diverges?
5. **Full FT instead of LoRA:** skip `get_peft_model` (train all params) with
   `lr=3e-4`. Then re-check the base model's *plain text* completion
   (`'The capital of Ghana is'`). Did full FT damage it more than LoRA did?
   That damage is called *catastrophic forgetting*.
6. **Attention-only LoRA:** set `target_modules=["self_attn.q_proj",
   "self_attn.v_proj"]`. Compare trainable params and results. This ties back
   to Module 1.

## ✅ Checkpoint questions

1. Why is `B` initialized to zero and not `A`? What would go wrong if both were zero?
2. LFM2-1.2B, full FT, AdamW, bf16 mixed precision: estimate the GPU memory
   needed for weights, gradients and optimizer states. Why does ZeRO-2 help on 4 GPUs?
3. You have 500 examples of a customer's support tone and want to keep the
   model's general skills. LoRA or full FT? Which `r` would you start with?
4. Your SFT train loss keeps dropping but eval loss starts rising after epoch
   1. What's happening, and which two knobs do you change first?
5. In your own words: why can't SFT reliably teach a model new facts?
