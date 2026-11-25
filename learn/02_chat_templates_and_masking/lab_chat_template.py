"""Lab 2 - Chat templates and loss masks, using the repo's own LFM templates.

No model download needed: we render the pinned Jinja templates from
job_configs/chat_templates/ exactly the way `tokenizer.apply_chat_template`
does, then turn the assistant *character spans* into *token labels*.

Run:  python learn/02_chat_templates_and_masking/lab_chat_template.py
"""

import pathlib
import re

from transformers.utils.chat_template_utils import render_jinja_template

ROOT = pathlib.Path(__file__).resolve().parents[2]
LFM25 = (ROOT / "job_configs/chat_templates/lfm2_5_chat_template.jinja").read_text()
LFM2 = (ROOT / "job_configs/chat_templates/lfm2_chat_template.jinja").read_text()
SPECIAL = dict(bos_token="<|startoftext|>", eos_token="<|im_end|>")


def render(template, messages, **kw):
    texts, spans = render_jinja_template(
        conversations=[messages], chat_template=template,
        return_assistant_tokens_mask=True, **SPECIAL, **kw,
    )
    return texts[0], spans[0]


def banner(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


conversation = [
    {"role": "system", "content": "You are a concise geography tutor."},
    {"role": "user", "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And Italy?"},
    {"role": "assistant", "content": "Rome."},
]

# ---------------------------------------------------------------------------
banner("1) What the model actually sees (LFM2.5 template, ChatML-style)")
text, spans = render(LFM25, conversation)
print(text)
print("Assistant character spans marked by {% generation %}:", spans)
for s, e in spans:
    print(f"   supervised -> {text[s:e]!r}")
print("Notice: the '<|im_start|>assistant\\n' header is NOT supervised, but the")
print("closing '<|im_end|>' IS - that's how the model learns when to stop.")

# ---------------------------------------------------------------------------
banner("2) Inference prompt: add_generation_prompt=True")
prompt, _ = render(LFM25, conversation[:4], add_generation_prompt=True)
print(prompt[-60:])
print("The prompt ends with an open assistant header; generation continues from there.")
print("Train/inference mismatch here (different template) is a classic silent bug.")

# ---------------------------------------------------------------------------
banner("3) Characters -> tokens -> labels (what tokenize_sft + the collator do)")

# A toy tokenizer: special tokens are atomic, everything else splits on words.
# (Leading spaces stick to words, like BPE.) Real LFM tokenizers are BPE (65k vocab) but the masking logic is identical:
# a token is supervised iff its characters fall inside an assistant span.
TOKEN_RE = re.compile(r"<\|[a-z_]+\|>| ?\w+|[^\w\s]|\s+")


def toy_tokenize(s):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(s)]


def masks_from_spans(tokens, spans):
    return [int(any(s <= a and b <= e for s, e in spans)) for _, a, b in tokens]


def final_span_only(mask):
    """Same idea as tokenize_data._final_assistant_span_mask."""
    out = [0] * len(mask)
    i = len(mask) - 1
    while i >= 0 and not mask[i]:
        i -= 1
    while i >= 0 and mask[i]:
        out[i] = 1
        i -= 1
    return out


tokens = toy_tokenize(text)
assistant_mask = masks_from_spans(tokens, spans)
completion_mask = final_span_only(assistant_mask)

print(f"{'token':28s} full  assistant_only  completion_only")
for (tok, _, _), a, c in zip(tokens, assistant_mask, completion_mask):
    shown = repr(tok)[:26]
    print(f"{shown:28s}  1          {a}              {c}")

n = len(tokens)
print(f"\nSupervised tokens: full={n}, assistant_only_loss={sum(assistant_mask)}, "
      f"completion_only_loss={sum(completion_mask)}")
print("labels = input_ids where mask==1 else -100  (PyTorch CE ignores -100)")
print("\nREPO DEFAULT for text SFT: assistant_only_loss=False -> the model also")
print("learns to *write the user/system turns*. VLM SFT defaults to True.")

# ---------------------------------------------------------------------------
banner("4) Reasoning traces: LFM2.5 drops <think> from PAST assistant turns")
thinking_convo = [
    {"role": "user", "content": "2+2?"},
    {"role": "assistant", "content": "<think>trivial sum</think>4"},
    {"role": "user", "content": "3+3?"},
    {"role": "assistant", "content": "<think>add three twice</think>6"},
]
text, spans = render(LFM25, thinking_convo)
print(text)
print("Turn 1 lost its <think>...</think>; only the LAST turn keeps it.")
print("So if you SFT multi-turn reasoning data, only the final turn teaches thinking.")
text_keep, _ = render(LFM25, thinking_convo, keep_past_thinking=True)
print("With keep_past_thinking=True the first turn is:",
      repr(text_keep.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]))

# ---------------------------------------------------------------------------
banner("5) Tool calling: LFM2 vs LFM2.5 render tools differently")
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                   "required": ["location"]}}}]
tool_convo = [
    {"role": "user", "content": "Weather in Boston?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather",
                                          "arguments": {"location": "Boston"}}}]},
    {"role": "tool", "content": '{"temp_f": 72, "sky": "sunny"}'},
    {"role": "assistant", "content": "72F and sunny."},
]
t25, s25 = render(LFM25, tool_convo, tools=tools)
print("--- LFM2.5 ---")
print(re.sub(r"Today's date: \d{4}-\d{2}-\d{2}", "Today's date: <today>", t25))
for s, e in s25:
    print("   supervised ->", repr(t25[s:e]))

t2, _ = render(LFM2, [{"role": "system", "content": ""}] + tool_convo[:1], tools=tools)
print("--- LFM2 system prompt (note <|tool_list_start|> markers) ---")
print(t2.split("<|im_end|>")[0])
print("\nLFM2's template also wraps role:tool content in <|tool_response_start|>...;"
      "\nthat's why the repo tells you NOT to add those markers yourself.")
