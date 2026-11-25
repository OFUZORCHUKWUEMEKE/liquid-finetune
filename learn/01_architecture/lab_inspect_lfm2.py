"""Lab 1 - Look inside an LFM2 model (CPU only, no downloads).

We build a *tiny, randomly initialised* LFM2 from a config so you can see the
real Hugging Face module tree that liquid-finetune trains, without a GPU.

Run:  python learn/01_architecture/lab_inspect_lfm2.py
"""

from collections import Counter

import torch
from transformers import Lfm2Config, Lfm2ForCausalLM

# Same LoRA target lists the repo uses (src/liquid_finetune/training/peft/peft_configs.py)
GLU_MODULES = ["feed_forward.w1", "feed_forward.w2", "feed_forward.w3"]
MHA_MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.out_proj", "self_attn.o_proj", "self_attn.gate_proj"]
CONV_MODULES = ["conv.in_proj", "conv.out_proj"]

torch.manual_seed(0)

# A miniature version of the LFM2 layout: mostly short-conv blocks, a few
# grouped-query-attention (GQA) blocks.  Real LFM2-350M/700M/1.2B use 16 layers
# (10 conv + 6 attention); here we use 8 layers (5 conv + 3 attention).
layer_types = ["conv", "conv", "full_attention", "conv", "conv",
               "full_attention", "conv", "full_attention"]
config = Lfm2Config(
    vocab_size=512,
    hidden_size=64,
    intermediate_size=256,
    num_hidden_layers=len(layer_types),
    layer_types=layer_types,
    num_attention_heads=4,
    num_key_value_heads=2,   # GQA: 2 KV heads shared by 4 query heads
    conv_L_cache=3,          # short conv kernel: each conv layer sees 3 tokens
    max_position_embeddings=1024,
)
model = Lfm2ForCausalLM(config).eval()

print("=" * 70)
print("1) Layer layout")
print("=" * 70)
for i, layer in enumerate(model.model.layers):
    kind = "ATTN (GQA)" if layer.is_attention_layer else "SHORT-CONV"
    print(f"  layer {i}: {kind}")

print("\n" + "=" * 70)
print("2) One conv block and one attention block, as PyTorch sees them")
print("=" * 70)
print(model.model.layers[0])
print(model.model.layers[2])

print("\n" + "=" * 70)
print("3) Where do the parameters live?")
print("=" * 70)
buckets = Counter()
for name, p in model.named_parameters():
    if "embed_tokens" in name:
        buckets["embedding (tied with lm_head)"] += p.numel()
    elif ".conv." in name:
        buckets["short-conv operator"] += p.numel()
    elif ".self_attn." in name:
        buckets["attention operator"] += p.numel()
    elif ".feed_forward." in name:
        buckets["SwiGLU feed-forward"] += p.numel()
    else:
        buckets["norms"] += p.numel()
total = sum(buckets.values())
for k, v in buckets.most_common():
    print(f"  {k:32s} {v:>9,d}  ({100 * v / total:4.1f}%)")
print(f"  {'TOTAL':32s} {total:>9,d}")
print("  tie_word_embeddings =", config.tie_word_embeddings,
      "-> lm_head.weight is embed_tokens.weight:",
      model.lm_head.weight is model.model.embed_tokens.weight)

print("\n" + "=" * 70)
print("4) Which linear layers does the repo's DEFAULT_LORA hit?")
print("=" * 70)
targets = GLU_MODULES + MHA_MODULES + CONV_MODULES
hit, missed = [], []
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        (hit if any(name.endswith(t) for t in targets) else missed).append(name)
print(f"  LoRA would wrap {len(hit)} Linear layers, e.g.:")
for n in hit[:6]:
    print("    +", n)
print("  Linear layers NOT targeted:", missed or "none")
print("  Note: the depthwise nn.Conv1d (conv.conv) is never LoRA-targeted;")
print("  it is tiny (hidden_size * kernel) and is trained only in full fine-tuning.")

print("\n" + "=" * 70)
print("5) Why LFM2 is cheap at inference: cache size vs. sequence length")
print("=" * 70)
n_conv = layer_types.count("conv")
n_attn = len(layer_types) - n_conv
head_dim = config.hidden_size // config.num_attention_heads
for T in (128, 1024, 8192):
    # attention layers keep K and V for every past token
    kv = n_attn * 2 * config.num_key_value_heads * head_dim * T
    # conv layers keep only the last (kernel-1) inputs, independent of T
    conv = n_conv * config.hidden_size * (config.conv_L_cache - 1)
    pure_transformer = len(layer_types) * 2 * config.num_key_value_heads * head_dim * T
    print(f"  T={T:5d}: LFM2 cache = {kv + conv:>9,d} floats "
          f"(conv part {conv:,d} is constant) | all-attention model = {pure_transformer:>9,d}")

print("\n" + "=" * 70)
print("6) It's a normal causal LM: logits for next token")
print("=" * 70)
ids = torch.randint(0, config.vocab_size, (1, 10))
with torch.no_grad():
    out = model(input_ids=ids, labels=ids)
print("  logits shape:", tuple(out.logits.shape), "(batch, seq, vocab)")
print(f"  loss on random init: {out.loss.item():.3f}  (ln(vocab)={torch.log(torch.tensor(512.)).item():.3f})")
