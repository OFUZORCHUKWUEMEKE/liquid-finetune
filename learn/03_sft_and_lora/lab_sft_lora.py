"""Lab 3 - The whole post-training story on a toy LFM2, on CPU, in ~1 minute.

  Stage A  "pre-training": full-parameter next-token training on plain facts
           -> a BASE model that knows facts but can't chat.
  Stage B  "SFT with LoRA": chat-formatted data, assistant-only loss, only
           LoRA adapters train (same target modules as the repo's DEFAULT_LORA).
  Stage C  merge the adapter into the base weights (what
           merge_and_save_peft_model does at the end of a repo run).

Key lesson: SFT mostly teaches FORMAT/BEHAVIOUR. Facts come from pre-training,
so the SFT model can answer countries it never saw in SFT data.

Run:  python learn/03_sft_and_lora/lab_sft_lora.py
"""

import math
import pathlib
import re

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Lfm2Config, Lfm2ForCausalLM
from transformers.utils.chat_template_utils import render_jinja_template

torch.manual_seed(0)
ROOT = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE = (ROOT / "job_configs/chat_templates/lfm2_5_chat_template.jinja").read_text()
BOS, EOS = "<|startoftext|>", "<|im_end|>"

FACTS = {
    "France": "Paris", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin",
    "Japan": "Tokyo", "Egypt": "Cairo", "Kenya": "Nairobi", "Peru": "Lima",
    "Nigeria": "Abuja", "Canada": "Ottawa", "Chile": "Santiago", "Ghana": "Accra",
}
SFT_COUNTRIES = list(FACTS)[:9]      # seen during SFT
HELD_OUT = list(FACTS)[9:]           # never seen in chat format


# ---------------------------------------------------------------- tokenizer
TOKEN_RE = re.compile(r"<\|[a-z_]+\|>| ?\w+|[^\w\s]|\s+")


def render(messages, add_generation_prompt=False):
    texts, spans = render_jinja_template(
        conversations=[messages], chat_template=TEMPLATE, bos_token=BOS,
        eos_token=EOS, return_assistant_tokens_mask=True,
        add_generation_prompt=add_generation_prompt)
    return texts[0], spans[0]


def chat(country, answer=None):
    msgs = [{"role": "user", "content": f"What is the capital of {country}?"}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": f"The capital of {country} is {answer}."})
    return msgs


# Like web text, the "pre-training corpus" states each fact a few ways,
# including as a question + answer (base models have seen lots of Q&A text).
pretrain_texts = []
for c, city in FACTS.items():
    pretrain_texts += [
        f"{BOS}The capital of {c} is {city}.",
        f"{BOS}{city} is the capital of {c}.",
        f"{BOS}Q: What is the capital of {c}? A: The capital of {c} is {city}.",
    ]
all_text = "".join(pretrain_texts) + "".join(
    render(chat(c, FACTS[c]))[0] for c in FACTS) + render(chat("X"), True)[0]
vocab = sorted({m.group() for m in TOKEN_RE.finditer(all_text)})
stoi = {t: i for i, t in enumerate(vocab)}


def encode(text, spans=None):
    ids, mask = [], []
    for m in TOKEN_RE.finditer(text):
        ids.append(stoi[m.group()])
        mask.append(1 if spans is None else int(any(s <= m.start() and m.end() <= e for s, e in spans)))
    return ids, mask


def batchify(rows):
    """Right-pad; labels=-100 on padding and on unsupervised tokens."""
    L = max(len(ids) for ids, _ in rows)
    inp = torch.zeros(len(rows), L, dtype=torch.long)
    lab = torch.full((len(rows), L), -100)
    att = torch.zeros(len(rows), L, dtype=torch.long)
    for i, (ids, mask) in enumerate(rows):
        inp[i, :len(ids)] = torch.tensor(ids)
        att[i, :len(ids)] = 1
        lab[i, :len(ids)] = torch.tensor([t if m else -100 for t, m in zip(ids, mask)])
    return inp, att, lab


@torch.no_grad()
def greedy(model, prompt_text, max_new=12):
    ids, _ = encode(prompt_text)
    ids = torch.tensor([ids])
    for _ in range(max_new):
        nxt = model(input_ids=ids).logits[0, -1].argmax().item()
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        if vocab[nxt] == EOS:
            break
    return "".join(vocab[i] for i in ids[0, -(ids.shape[1] - len(encode(prompt_text)[0])):])


def train(model, rows, steps, lr, label):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    inp, att, lab = batchify(rows)
    for step in range(1, steps + 1):
        loss = model(input_ids=inp, attention_mask=att, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)  # repo default max_grad_norm=1.0
        opt.step()
        opt.zero_grad()
        if step == 1 or step % (steps // 5) == 0:
            print(f"  [{label}] step {step:4d}  loss {loss.item():.4f}")


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ---------------------------------------------------------------- model
config = Lfm2Config(
    vocab_size=len(vocab), hidden_size=96, intermediate_size=384,
    num_hidden_layers=6, layer_types=["conv", "conv", "full_attention"] * 2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
)
model = Lfm2ForCausalLM(config)
print(f"vocab={len(vocab)} tokens, model={sum(p.numel() for p in model.parameters()):,} params")

# ================================================================ Stage A
banner("Stage A: pre-training (full fine-tune, loss on every token)")
train(model, [encode(t) for t in pretrain_texts], steps=800, lr=2e-3, label="pretrain")
model.eval()
print("\nBase model completes text fine:")
print("  'The capital of Ghana is' ->", repr(greedy(model, f"{BOS}The capital of Ghana is", 3)))
print("  'Q: What is the capital of Ghana? A:' ->",
      repr(greedy(model, f"{BOS}Q: What is the capital of Ghana? A:", 7)))
print("But it cannot chat (it has never seen <|im_start|>assistant):")
print("  ->", repr(greedy(model, render(chat("Ghana"), True)[0])))

# ================================================================ LoRA math
banner("LoRA in one line:  h = W x + (alpha / r) * B(A(x)),  B initialised to 0")
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.0,
    # identical to the repo's DEFAULT_LORA target list
    target_modules=["feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.out_proj", "conv.in_proj", "conv.out_proj"],
)
probe = torch.tensor([encode(render(chat("Peru"), True)[0])[0]])
with torch.no_grad():
    before = model(input_ids=probe).logits
model = get_peft_model(model, lora_cfg)
with torch.no_grad():
    after = model(input_ids=probe).logits
print("Output identical right after wrapping (because B=0):", torch.allclose(before, after))
model.print_trainable_parameters()
w = model.base_model.model.model.layers[0].conv.in_proj
print(f"e.g. conv.in_proj: W {tuple(w.base_layer.weight.shape)} frozen, "
      f"A {tuple(w.lora_A['default'].weight.shape)} + B {tuple(w.lora_B['default'].weight.shape)} trainable,"
      f" scale alpha/r = {w.scaling['default']}")

# ================================================================ Stage B
banner("Stage B: SFT with LoRA, assistant_only_loss=True")
sft_rows = [encode(*render(chat(c, FACTS[c]))) for c in SFT_COUNTRIES]
n_sup = sum(sum(m) for _, m in sft_rows)
n_all = sum(len(m) for _, m in sft_rows)
print(f"supervised tokens: {n_sup}/{n_all} ({100 * n_sup / n_all:.0f}%) - only the assistant reply")
model.train()
train(model, sft_rows, steps=150, lr=3e-3, label="sft-lora")
model.eval()

print("\nSFT countries (seen in chat format):")
for c in SFT_COUNTRIES[:3]:
    print(f"  {c:8s} -> {greedy(model, render(chat(c), True)[0])!r}")
print("HELD-OUT countries (only seen in pre-training text!):")
hits = 0
for c in HELD_OUT:
    out = greedy(model, render(chat(c), True)[0])
    hits += FACTS[c] in out
    print(f"  {c:8s} -> {out!r}")
print(f"held-out correct: {hits}/{len(HELD_OUT)}  "
      "<- format learned in SFT + facts from pre-training")

# ================================================================ Stage C
banner("Stage C: merge adapter -> plain Lfm2ForCausalLM (what gets exported)")
with torch.no_grad():
    lora_logits = model(input_ids=probe).logits
merged = model.merge_and_unload()
with torch.no_grad():
    merged_logits = merged(input_ids=probe).logits
print("type after merge:", type(merged).__name__)
print("max |logit diff| LoRA vs merged:", f"{(lora_logits - merged_logits).abs().max().item():.2e}")
print("W_merged = W + (alpha/r) * B @ A  -> zero extra inference cost.")
