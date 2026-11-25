# Module 2: Chat templates, tokenization and loss masking

**Goal:** know *exactly* which characters the model sees and which tokens it is
trained on. Most "my fine-tune is weird" bugs live here.

**Lab:** `python learn/02_chat_templates_and_masking/lab_chat_template.py` (CPU, instant)

---

## 2.1 A chat is just a string

A model never sees `{"role": "user", ...}`. A **chat template** (a Jinja
program shipped with the tokenizer) flattens the conversation into one string
with special tokens. LFM uses a ChatML-style format:

```
<|startoftext|><|im_start|>system
You are a concise geography tutor.<|im_end|>
<|im_start|>user
Capital of France?<|im_end|>
<|im_start|>assistant
Paris.<|im_end|>
```

The repo pins its own copies of the templates in
`job_configs/chat_templates/`:

- `lfm2_chat_template.jinja`: LFM2.
- `lfm2_5_chat_template.jinja`: LFM2.5 (and LFM2-24B). Selected automatically
  for non-local LiquidAI LFM2.5 models (`_default_chat_template_path` in
  `checkpointing/model_loading.py`). You can override it with
  `chat_template_path` in your YAML.

**Rule #1: train with the same template you'll serve with.** If inference
(llama.cpp, vLLM, your app) renders the prompt differently, even by one
newline, your fine-tune underperforms and nothing errors.

## 2.2 Training vs. inference rendering

- **Training:** render the *whole* conversation, including the assistant
  replies.
- **Inference:** render up to the last user turn, plus
  `add_generation_prompt=True`, which appends an open
  `<|im_start|>assistant\n`. The model writes from there and should stop by
  emitting `<|im_end|>`.

Corollary: the model must **learn to emit `<|im_end|>`**, so that token must be
supervised. If it isn't, your model rambles forever.

## 2.3 Loss masking: which tokens get gradient?

The training loss is cross-entropy on next-token predictions. You choose
**which positions count** by setting `labels = -100` on the ones to ignore
(PyTorch's `ignore_index`).

The LFM templates wrap assistant output in `{% generation %} …
{% endgeneration %}`. When you call
`apply_chat_template(..., return_assistant_tokens_mask=True)`, Transformers
returns a mask that is 1 exactly on those tokens. The lab shows that the span
is the assistant **content plus `<|im_end|>\n`**. The `<|im_start|>assistant\n`
header is *not* supervised, because it's always given in the prompt.

The three modes you can pick in `training_config` (text SFT):

| Setting | Trains on | When to use |
|---|---|---|
| *(default)* both `False` | **every token**, including system and user turns | Plain-text/continued pre-training. Rarely what you want for chat. |
| `assistant_only_loss: true` | every assistant turn | **Standard choice for chat SFT.** |
| `completion_only_loss: true` | only the **final** assistant turn | Multi-turn data where earlier assistant turns are context, not targets (for example, synthetic histories of mixed quality). |

> ⚠️ **Repo default:** for **text** SFT, `assistant_only_loss` defaults to
> `False` (`data_loading/ray_data_utils.py`). **VLM** SFT defaults to `True`
> (`DEFAULT_VLM_SFT`). For chat data, set `assistant_only_loss: true`
> explicitly.

How it flows through the repo:

1. `data_loading/tokenize_data.py::tokenize_sft` calls
   `apply_chat_template(..., return_assistant_tokens_mask=True)` and stores
   `assistant_masks` or `completion_mask` (`_final_assistant_span_mask` keeps
   only the last contiguous span).
2. It **raises** if a row's mask is all zeros and it wasn't truncated. That
   catches a template without `{% generation %}` markers, which would
   otherwise train on *zero* tokens while the loss curve still looks fine.
3. `training/sft.py::LFMDataCollatorForLanguageModeling` turns masks into
   `labels` (−100 where the mask is 0).

## 2.4 Reasoning (`<think>`) data in LFM2.5

The LFM2.5 template **strips `<think>…</think>` from every assistant turn
except the last** (unless `keep_past_thinking=True`). That mirrors inference:
past reasoning isn't fed back into the context. Consequences:

- In multi-turn reasoning data, **only the final turn teaches thinking**.
- If you want each turn's reasoning to be trained, split the conversation into
  several rows, one per assistant turn.
- You can also pass reasoning in a separate `thinking` field on the message.
  The template inserts it as `<think>…</think>` before the content.

## 2.5 Tool calling

LFM writes tool calls as Python-like calls between special tokens:

```
<|tool_call_start|>[get_weather(location="Boston")]<|tool_call_end|>
```

- Tool **definitions** go in the system prompt: pass `tools=[...]`, or
  write `List of tools: [...]` yourself.
- **LFM2** wraps the list in `<|tool_list_start|>…<|tool_list_end|>` and wraps
  tool results in `<|tool_response_start|>…<|tool_response_end|>`.
- **LFM2.5** does neither and adds `Today's date: …` to the system prompt.
- Don't put the response markers in your data yourself; the template adds them.
- OpenAI-style `tool_calls` fields are converted for you
  (`data_loading/validate_tool_format.py`). Foreign formats such as `<tool_call>`
  XML are rejected.
- With `assistant_only_loss`, the **tool-call turn is supervised** (the lab
  shows it). That's how the model learns to call tools. Tool *results* are
  not supervised, since the model doesn't produce them.

## 2.6 Common bugs checklist

- [ ] Inference uses a different template or system prompt than training.
- [ ] `assistant_only_loss` not set for chat data, so the model learns to
      imitate users.
- [ ] `<|im_end|>` not supervised (custom template), so the model never stops.
- [ ] Truncation (`max_length`, default 2048) cuts off the assistant answer,
      which leaves a row with no supervised tokens. Consider
      `drop_overlength: true` for long-context data.
- [ ] LFM2 tool markers in LFM2.5 data or vice versa (the repo warns and
      auto-strips one case).
- [ ] BOS duplicated: you added `<|startoftext|>` to raw text *and* the
      template adds it.

## 2.7 Exercises

1. In the lab, make the first assistant turn 3× longer. How do the three
   mask counts change? Which mode is least affected?
2. Delete `{%- generation -%}` and `{%- endgeneration -%}` from a *copy* of
   the template and render it. What does the span list become, and which
   repo check would catch this?
3. Render the tool conversation with `lfm2_chat_template.jinja`. Find the
   `<|tool_response_start|>` the template added.
4. Write a 3-turn reasoning conversation. Under `completion_only_loss`, how
   many `<think>` blocks are supervised? Under `assistant_only_loss`?
5. **With a real tokenizer (needs HF access):**
   `AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")`. Call
   `apply_chat_template(conv, tokenize=True, return_dict=True,
   return_assistant_tokens_mask=True)` and decode only the masked ids.

## ✅ Checkpoint questions

1. Why must `<|im_end|>` be inside the supervised span?
2. You SFT a customer-support bot with the repo defaults and it starts
   generating fake "User:" turns. What is the most likely cause, and which
   single YAML line fixes it?
3. When would you choose `completion_only_loss` over `assistant_only_loss`?
4. Your LFM2.5 multi-turn reasoning model "forgets how to think" after turn
   one. Using section 2.4, explain why and propose a data fix.
