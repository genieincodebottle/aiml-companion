# tool_common.py: the shared code of the lltToolUse lab (data, labels, parsers, checker, constrained decoding,
# evaluation). The lab page shows these functions in its steps; the notebook writes this file in its third cell.
# No backslashes anywhere in this file: the page keeps its code in JS template literals. NL is the newline.
import json, random, re

NL = chr(10)

# ---------- Step 2. Hermes rows to structured messages ----------
def hermes_to_chat(row):
    # The Hermes text writes calls as JSON inside <tool_call> tags. Turn them into structured tool_calls,
    # so the model's own chat template renders them in the model's own format.
    tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    messages = []
    for turn in row["conversations"]:
        role, text = turn["from"], turn["value"]
        if role == "system":
            continue                                   # the template writes its own tool prompt from the tools list
        if role == "human":
            messages.append({"role": "user", "content": text})
        elif role == "tool":
            body = text.replace("<tool_response>", "").replace("</tool_response>", "").strip()
            messages.append({"role": "tool", "content": body})
        elif role == "gpt":
            calls = []
            for block in re.findall("<tool_call>(.*?)</tool_call>", text, re.S):
                call = json.loads(block)               # raises on a malformed call; the caller drops the row
                calls.append({"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}})
            content = re.sub("<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
            msg = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
    return {"messages": messages, "tools": tools or []}


def parses(row):
    try:
        hermes_to_chat(row)
        return True
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def select_rows(rows, n, render, max_tokens):
    # Keep rows in file order whose calls parse and whose rendered length fits. Returns kept rows and drop counts.
    kept, dropped = [], {"bad_json": 0, "too_long": 0, "no_tools": 0}
    for row in rows:
        try:
            chat = hermes_to_chat(row)
        except (json.JSONDecodeError, KeyError, TypeError):
            dropped["bad_json"] += 1
            continue
        if not chat["tools"]:
            dropped["no_tools"] += 1
            continue
        ids = render(chat)
        if len(ids) > max_tokens:
            dropped["too_long"] += 1
            continue
        kept.append(chat)
        if len(kept) == n:
            break
    return kept, dropped


# ---------- Step 3. The label mask ----------
def build_labels(ids, header, im_end, newline):
    # Loss on each assistant turn: every token after the header up to and including <|im_end|> and the newline
    # after it, which is the span TRL's training template marks with {% generation %}. Everything else is -100:
    # the tool prompt, the user turns and every tool response.
    labels = [-100] * len(ids)
    h = len(header)
    t = 0
    while t < len(ids):
        if ids[t:t + h] == header:
            t += h
            while t < len(ids):
                labels[t] = ids[t]
                t += 1
                if ids[t - 1] == im_end:
                    if t < len(ids) and ids[t] == newline:
                        labels[t] = ids[t]
                        t += 1
                    break
        else:
            t += 1
    return labels


# ---------- Step 4. BFCL items with JSON Schema tools ----------
BFCL_TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string", "list": "array", "bool": "boolean"}

def to_json_schema(p):
    # BFCL writes types in its own dialect (dict, float, tuple). Map them to JSON Schema, recursively.
    p = dict(p)
    if "type" in p:
        p["type"] = BFCL_TYPES.get(p["type"], p["type"])
    if "properties" in p:
        p["properties"] = {k: to_json_schema(v) for k, v in p["properties"].items()}
    if "items" in p and isinstance(p["items"], dict):
        p["items"] = to_json_schema(p["items"])
    return p

def bfcl_tools(functions):
    return [{"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                              "parameters": to_json_schema(f["parameters"])}} for f in functions]

def load_bfcl(data_dir, plan):
    # plan: {category: stride}. Takes every stride-th item of BFCL_v4_<category>.json, in file order.
    items = []
    for cat, stride in plan.items():
        qs = [json.loads(l) for l in open(data_dir + "/BFCL_v4_" + cat + ".json", encoding="utf-8") if l.strip()]
        gt = {}
        try:
            for l in open(data_dir + "/possible_answer/BFCL_v4_" + cat + ".json", encoding="utf-8"):
                if l.strip():
                    a = json.loads(l)
                    gt[a["id"]] = a["ground_truth"]
        except FileNotFoundError:
            pass                                       # irrelevance has no answers: the right output is no call
        for q in qs[::stride]:
            items.append({"id": q["id"], "category": cat, "messages": q["question"][0], "functions": q["function"],
                          "tools": bfcl_tools(q["function"]), "ground_truth": gt.get(q["id"])})
    return items


# ---------- Step 5. Parse the model's output, one parser per family ----------
def coerce(raw, typ):
    # Type an XML parameter value with the tool's schema, as transformers' parse_response does.
    try:
        if typ == "integer":
            return int(raw)
        if typ == "number":
            return float(raw) if ("." in raw or "e" in raw.lower()) else int(raw)
        if typ == "boolean" and raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        if typ in ("array", "object"):
            return json.loads(raw)
    except ValueError:
        pass
    return raw

def parse_xml_calls(text, tools):
    # Qwen3.5 format: <tool_call> <function=NAME> <parameter=KEY> VALUE </parameter> ... </function> </tool_call>
    # Returns a list of {name: arguments}; [] when there is no call; None when a call is malformed.
    if text.count("<tool_call>") != text.count("</tool_call>"):
        return None
    props = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in tools}
    calls = []
    for block in re.findall("<tool_call>(.*?)</tool_call>", text, re.S):
        found = re.search("<function=([^>]+)>", block)
        if found is None or block.count("<parameter=") != block.count("</parameter>"):
            return None
        name = found.group(1).strip()
        args = {}
        for key, raw in re.findall("<parameter=([^>]+)>(.*?)</parameter>", block, re.S):
            schema = props.get(name, {}).get(key.strip(), {})
            args[key.strip()] = coerce(raw.strip(), schema.get("type"))
        calls.append({name: args})
    return calls

def parse_json_calls(text, tools=None):
    # Qwen3 (Hermes) format: <tool_call> {"name": ..., "arguments": {...}} </tool_call>
    if text.count("<tool_call>") != text.count("</tool_call>"):
        return None
    calls = []
    for block in re.findall("<tool_call>(.*?)</tool_call>", text, re.S):
        try:
            call = json.loads(block)
        except json.JSONDecodeError:
            return None
        if not isinstance(call, dict) or "name" not in call or not isinstance(call.get("arguments", {}), dict):
            return None
        calls.append({call["name"]: call.get("arguments", {})})
    return calls


# ---------- Step 6. A BFCL-style AST check ----------
def norm(s):
    # BFCL compares strings without spaces and the punctuation , . / - _ * ^, ignoring case
    for ch in " ,./-_*^":
        s = s.replace(ch, "")
    return s.lower().replace("'", '"')

def value_ok(value, allowed):
    for a in allowed:
        if a is None and value is None:
            return True
        if isinstance(a, str) and isinstance(value, str) and norm(a) == norm(value):
            return True
        if isinstance(a, bool) or isinstance(value, bool):
            if a is value:
                return True
            continue
        if isinstance(a, (int, float)) and isinstance(value, (int, float)) and a == value:
            return True
        if isinstance(a, list) and isinstance(value, list) and len(a) == len(value) and all(value_ok(v, [x]) for v, x in zip(value, a)):
            return True
        if isinstance(a, dict) and isinstance(value, dict) and set(value) <= set(a) and all(value_ok(value[k], a[k]) for k in value) and all("" in a[k] for k in a if k not in value):
            return True
    return False

def call_ok(call, expected, schema):
    # call {name: args}; expected {name: {param: [allowed values, "" if it may be left out]}}
    name, allowed = next(iter(expected.items()))
    if name not in call:
        return False
    args = call[name]
    if any(p not in args for p in schema.get("required", [])):
        return False
    if any(p not in allowed or p not in schema.get("properties", {}) for p in args):
        return False
    if any(p not in args and "" not in allowed[p] for p in allowed):
        return False
    return all(value_ok(v, allowed[p]) for p, v in args.items())

def check(item, calls):
    # True when the parsed calls match the ground truth. calls is None when the reply had a malformed call.
    # As in BFCL, an irrelevance item is right when no well-formed call comes out.
    if item["category"].endswith("irrelevance"):
        return not calls
    if not calls:
        return False
    expected = item["ground_truth"]
    if len(calls) != len(expected):
        return False
    schemas = {f["name"]: f["parameters"] for f in item["functions"]}
    unused = list(range(len(calls)))
    for exp in expected:                                   # any order: give each expected call its own match
        hit = next((i for i in unused if call_ok(calls[i], exp, schemas.get(next(iter(exp)), {}))), None)
        if hit is None:
            return False
        unused.remove(hit)
    return True


# ---------- Step 8. Constrained decoding: force a well-formed call with offered names ----------
TEXT_INDEX = {}

class CallsOnly:
    # A logits processor for the Qwen3.5 format. Each reply must be one or more tool calls. Function and parameter
    # names come from the offered tools; parameter values stay free text. This is tool_choice "required" plus a grammar.
    def __init__(self, tok, prompt_len, tools_per_row):
        self.tok, self.prompt_len, self.rows = tok, prompt_len, tools_per_row
        self.end_id = tok.convert_tokens_to_ids("<|im_end|>")
        if id(tok) not in TEXT_INDEX:                    # token text -> ids, built once per tokenizer
            index = {}
            for i in range(len(tok)):
                s = tok.decode([i])
                if s and s.isascii():
                    index.setdefault(s, []).append(i)
            TEXT_INDEX[id(tok)] = index
        self.by_text = TEXT_INDEX[id(tok)]

    def options(self, text, tools):
        # The strings that may come next. None means free text (inside a parameter value).
        names = {t["function"]["name"]: list(t["function"]["parameters"].get("properties", {})) for t in tools}
        pos = 0
        start = "<tool_call>" + NL + "<function="
        while True:
            rest = text[pos:]
            if not rest.startswith(start):
                return [start[len(rest):]] if start.startswith(rest) else []
            pos += len(start)
            rest = text[pos:]
            heads = [n + ">" + NL for n in names]
            got = [h for h in heads if rest.startswith(h)]
            if not got:
                return [h[len(rest):] for h in heads if h.startswith(rest)]
            name = got[0][:-2]
            pos += len(got[0])
            used = set()
            while True:
                rest = text[pos:]
                opts = ["<parameter=" + p + ">" + NL for p in names[name] if p not in used] + ["</function>" + NL + "</tool_call>"]
                got = [o for o in opts if rest.startswith(o)]
                if not got:
                    return [o[len(rest):] for o in opts if o.startswith(rest)]
                pos += len(got[0])
                if got[0].startswith("</function>"):
                    break
                used.add(got[0][len("<parameter="):-2])
                close = text.find("</parameter>", pos)
                if close < 0:
                    return None                          # inside a value: free text
                pos = close + len("</parameter>")
                rest = text[pos:]
                if not rest.startswith(NL):
                    return [NL] if rest == "" else []
                pos += 1
            rest = text[pos:]
            if rest == "":
                return [NL + start, "<|im_end|>"]
            if not rest.startswith(NL):
                return []
            pos += 1

    def __call__(self, input_ids, scores):
        for r in range(input_ids.shape[0]):
            text = self.tok.decode(input_ids[r, self.prompt_len:], skip_special_tokens=False)
            if "<|im_end|>" in text:
                continue
            opts = self.options(text, self.rows[r])
            if opts is None:
                continue
            allowed = set()
            for o in opts:
                if o == "<|im_end|>":
                    allowed.add(self.end_id)
                    continue
                for k in range(1, len(o) + 1):
                    allowed.update(self.by_text.get(o[:k], []))
            if not allowed:
                allowed.add(self.end_id)                  # a dead end: stop, and the parser reports the bad call
            keep = scores[r, list(allowed)].clone()
            scores[r] = float("-inf")
            scores[r, list(allowed)] = keep
        return scores


# ---------- Step 7. Generate replies for many items ----------
def generate(model, tok, items, batch_size=32, max_new_tokens=384, constrained=False, device="cuda"):
    import torch
    from transformers import LogitsProcessorList
    tok.padding_side = "left"
    stops = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    prompts = [tok.apply_chat_template(it["messages"], tools=it["tools"], add_generation_prompt=True, tokenize=False) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))
    out = [None] * len(items)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        procs = LogitsProcessorList()
        if constrained:
            procs.append(CallsOnly(tok, enc["input_ids"].shape[1], [items[i]["tools"] for i in idx]))
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=stops,
                                 pad_token_id=tok.pad_token_id, logits_processor=procs)
        for j, i in enumerate(idx):
            text = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=False)
            out[i] = text.split("<|im_end|>")[0].replace(tok.pad_token, "")
    return out


# ---------- Scoring ----------
def score(items, outputs, parse):
    rows = []
    for it, text in zip(items, outputs):
        calls = parse(text, it["tools"])
        rows.append({"id": it["id"], "category": it["category"], "parsed": calls is not None,
                     "calls": calls, "correct": check(it, calls)})
    return rows

def summary(rows):
    cats = {}
    for r in rows:
        c = cats.setdefault(r["category"], [0, 0, 0])
        c[0] += 1
        c[1] += int(r["correct"])
        c[2] += int(r["parsed"])
    total = len(rows)
    return {"accuracy": round(sum(r["correct"] for r in rows) / total, 4),
            "parseRate": round(sum(r["parsed"] for r in rows) / total, 4),
            "byCategory": {k: {"n": v[0], "accuracy": round(v[1] / v[0], 4), "parseRate": round(v[2] / v[0], 4)} for k, v in cats.items()}}

def paired_bootstrap(a, b, n=10000, seed=0):
    # a, b: lists of 0/1 per item for two systems on the same items. Returns the mean difference b - a and a 95% CI.
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b)]
    means = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return round(sum(d) / len(d), 4), round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)


def official_check():
    # BFCL's own AST checker from the bfcl-eval package (pip install --no-deps bfcl-eval==2026.3.23).
    # Its model_config module imports every API client, so stand in a one-entry table for it.
    import sys, types
    stub = types.ModuleType("bfcl_eval.constants.model_config")
    stub.MODEL_CONFIG_MAPPING = {"lab": types.SimpleNamespace(underscore_to_dot=False)}
    sys.modules["bfcl_eval.constants.model_config"] = stub
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.constants.enums import Language

    def run(item, calls):
        if item["category"].endswith("irrelevance"):
            return not calls
        if not calls:
            return False
        return ast_checker(item["functions"], calls, item["ground_truth"], Language.PYTHON, item["category"], "lab")["valid"]
    return run


# ---------- Step 6. Render a ground-truth answer, to test the parser and the checker ----------
def first_value(v):
    # BFCL nests allowed-value lists inside dict answers: {"width": [20]} means width must be 20.
    if isinstance(v, dict):
        return {k: first_value([x for x in vals if x != ""][0]) for k, vals in v.items() if [x for x in vals if x != ""]}
    if isinstance(v, list):
        return [first_value(x) for x in v]
    return v

def gt_reply(tok, item):
    # The reply the template writes for the first allowed answer. Irrelevance items get an empty reply.
    if item["ground_truth"] is None:
        return ""
    calls = []
    for exp in item["ground_truth"]:
        name, allowed = next(iter(exp.items()))
        args = {p: first_value([x for x in vals if x != ""][0]) for p, vals in allowed.items() if [x for x in vals if x != ""]}
        calls.append({"type": "function", "function": {"name": name, "arguments": args}})
    prompt = tok.apply_chat_template(item["messages"], tools=item["tools"], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(item["messages"] + [{"role": "assistant", "content": "", "tool_calls": calls}],
                                   tools=item["tools"], tokenize=False)
    return full[len(prompt):].split("<|im_end|>")[0]


# ---------- Step 8. LoRA on a frozen fp16 base with TRL ----------
def train_lora(name, tok, rows, seed=0, rank=16, lr=2e-4, out="lora-out", special=None):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda")   # frozen base, fp16
    model.enable_input_require_grads()
    if special:                                          # step 10: give untrained special tokens distinct rows
        E = model.get_input_embeddings().weight
        with torch.no_grad():
            for s in special:                            # the mean of the rows of its spelling, "<" + "tool_call>"
                pieces = tok.encode(s[0], add_special_tokens=False) + tok.encode(s[1:], add_special_tokens=False)
                E[tok.convert_tokens_to_ids(s)] = E[pieces].mean(0)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    args = SFTConfig(output_dir=out, report_to="none", save_strategy="no", seed=seed, data_seed=seed,
                     fp16=True, bf16=False,              # a T4 has no bf16 tensor cores; SFTConfig defaults to bf16
                     learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=10, num_train_epochs=1,
                     per_device_train_batch_size=2, gradient_accumulation_steps=2, max_length=1536,
                     gradient_checkpointing=True, logging_steps=5)
    trainer = SFTTrainer(model=model, args=args, processing_class=tok, train_dataset=Dataset.from_list(rows))
    for p in trainer.model.parameters():                 # GradScaler needs fp32 adapters; re-cast any that are not
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    trainer.train()
    return trainer


# ---------- Step 11. Conversations whose tools cannot answer ----------
NO_TOOL_REPLY = "None of the functions I have can do this, so I will not call one. I can answer in plain text if that helps."

def make_negatives(rows, start, count, shift):
    # Pair each request with the tools of another conversation that share no function name with its own tools.
    # The target reply is plain text with no call: what BFCL's irrelevance items ask for.
    out = []
    for k in range(start, start + count):
        a, b = rows[k], rows[(k - shift) % len(rows)]
        own = {t["function"]["name"] for t in a["tools"]}
        other = {t["function"]["name"] for t in b["tools"]}
        if own & other:
            continue
        out.append({"messages": [a["messages"][0], {"role": "assistant", "content": NO_TOOL_REPLY}], "tools": b["tools"]})
    return out
