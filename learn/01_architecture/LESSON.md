# Module 1: What is an LFM, and why does it matter for fine-tuning?

**Goal:** know the LFM model you're training well enough to reason about memory,
speed, LoRA targets, and failure modes.

**Lab:** `python learn/01_architecture/lab_inspect_lfm2.py` (CPU, about 5 seconds)

---

## 1.1 The family at a glance

Liquid Foundation Models are small models built for **on-device** use
(phones, laptops, cars, embedded hardware). The names you'll see in this repo:

| Family | Examples (as named in the repo / HF `LiquidAI/…`) | Notes |
|---|---|---|
| **LFM2** dense text | `LFM2-350M`, `LFM2-700M`, `LFM2-1.2B`, `LFM2-2.6B` | Hybrid conv + attention |
| **LFM2 MoE** | `LFM2-8B-A1B`, `LFM2-24B-A2B` | "8B total, ~1B active per token" |
| **LFM2.5** | `LFM2.5-1.2B-…` (Base / Instruct / Thinking variants) | Newer recipe, new chat template (`<think>` support) |
| **LFM2-VL / LFM2.5-VL** | vision-language | Vision encoder + projector + LFM2 language model |
| **LFM3 / LFM3-VL** | newest | Adds sliding-window attention; conv kernel 7 |

The repo decides "which family is this?" by name in
`src/liquid_finetune/checkpointing/model_info.py` (`get_model_family`,
`is_moe_model_from_name`). That choice changes the chat template and the
tool-calling format. You'll see why in Module 2.

## 1.2 The core idea: mostly convolutions, a little attention

A standard transformer stacks N identical blocks of `attention → MLP`.
**LFM2 replaces most of the attention operators with a gated short
convolution**:

```
every block:   x ─► RMSNorm ─► OPERATOR ─►(+x)─► RMSNorm ─► SwiGLU MLP ─►(+)─►
                                 │
                 either  Lfm2ShortConv   (most layers)
                 or      Lfm2Attention   (a few layers, GQA)
```

Liquid's LFM2 release describes the 350M/700M/1.2B models as **16 blocks: 10
short-conv + 6 GQA attention**. Check this yourself in Exercise 4.

### The short-conv operator (read the real code)

From `transformers/models/lfm2/modeling_lfm2.py` (`Lfm2ShortConv.forward`):

```python
BCx = self.in_proj(x)            # Linear: hidden -> 3*hidden
B, C, x = BCx.chunk(3)           # split into two gates and a value
h = B * x                        # input gate (elementwise)
h = causal_conv1d(h, kernel=3)   # depthwise conv: each channel mixes the last 3 tokens
y = C * h                        # output gate (elementwise)
y = self.out_proj(y)             # Linear: hidden -> hidden
```

Three things to take away:

1. **It's input-dependent ("double-gated").** B and C are computed from the
   input, so the operator adapts per token, unlike a fixed CNN filter.
2. **It only looks back `conv_L_cache` tokens (3 in LFM2, 7 in LFM3).** The
   few attention layers handle long-range information.
3. **Its cache does not grow.** At inference, a conv layer only remembers its
   last `kernel−1` inputs. An attention layer must keep K and V for every past
   token. Lab section 5 prints this. It's why LFM2 decodes fast and uses little
   memory on a phone.

### The attention operator

Ordinary causal attention with:
- **GQA** (grouped-query attention): fewer K/V heads than Q heads, which
  shrinks the KV cache.
- **QK-norm**: RMSNorm on q and k per head (`q_layernorm`, `k_layernorm`), for
  training stability.
- **RoPE** positions.
- LFM2 names the output projection `out_proj`. LFM3 names it `o_proj`.

### The rest

- **SwiGLU MLP**: `w2(silu(w1 x) * w3 x)`. This is where **most parameters
  live** (about 70%+ in the lab).
- **Tied embeddings**: `lm_head.weight` *is* `embed_tokens.weight`. The repo
  enforces this (`_enforce_lfm_tied_word_embeddings` in
  `checkpointing/model_loading.py`). With a 65,536-token vocab, untying
  would add a big matrix to a small model.

## 1.3 Why this matters for post-training

| Architecture fact | Consequence for you |
|---|---|
| Most params are in the MLP and conv projections | LoRA must target `feed_forward.w1/w2/w3` and `conv.in_proj/out_proj`, not just attention. Attention-only LoRA (an old Llama habit) would miss most of the model. |
| The repo's `DEFAULT_LORA` targets GLU + MHA + CONV modules | See `src/liquid_finetune/training/peft/peft_configs.py`. The list includes both `out_proj` and `o_proj` so one config works for LFM2 and LFM3; PEFT ignores names that aren't present. |
| The depthwise `conv.conv` is an `nn.Conv1d`, not a Linear | LoRA never touches it. It is tiny, and only full fine-tuning updates it. |
| `lm_head` is tied to the embedding | Not a LoRA target. If you add new special tokens you must resize embeddings, and the head comes along. |
| Small models (350M–2.6B) | Full fine-tuning is often affordable. LoRA is about memory and keeping the base intact, not about whether training is possible at all. |
| Conv layers have state | Fast kernels matter. Install `causal-conv1d` on GPU. The lab warns that it falls back to a slow PyTorch path. |

## 1.4 Lab walkthrough

Run the lab and connect each printed section to the text above:

1. Layer layout: conv vs. attention per layer.
2. Module tree: find `in_proj` (out_features = 3 × hidden) and the
   depthwise `Conv1d(groups=hidden)`.
3. Parameter budget: notice how small attention is.
4. LoRA targets: the repo's lists wrap 46 Linear layers and skip only
   `lm_head`.
5. Cache growth: the conv cache is constant and the KV cache is linear in T.
6. It's still a causal LM: random init gives loss ≈ ln(vocab), meaning
   uniform guessing.

## 1.5 Exercises (predict first, then run)

1. Change `conv_L_cache=3` to `7` (the LFM3 value). Which parameter count
   changes, and by how much? *(Hint: depthwise conv weight shape is
   `[hidden, 1, kernel]`.)*
2. Make every layer `"conv"`. What happens to the cache table? What
   capability would such a model lose? *(Think: retrieving a fact from 2,000
   tokens ago.)*
3. Set `num_key_value_heads=4` (equal to the number of query heads, so plain
   MHA instead of GQA). How does the attention parameter count and the KV
   cache change?
4. **On a machine with HF access:** run
   `AutoConfig.from_pretrained("LiquidAI/LFM2-1.2B")` and print
   `layer_types` (or `full_attn_idxs`), `hidden_size`, `conv_L_cache`, and
   `vocab_size`. Does it match "10 conv + 6 attention"?
5. Build a `Lfm3Config()` with defaults and print its `layer_types`. What are
   the three operator kinds?

## ✅ Checkpoint questions (send me your answers)

1. In one sentence each: what do the gates B and C do in the short-conv block?
2. Why does LFM2 need *any* attention layers at all?
3. A friend fine-tunes LFM2 with LoRA on `q_proj, v_proj` only and sees weak
   results. Using the lab's parameter breakdown, explain why.
4. Why is `lm_head` not in the LoRA target list?
