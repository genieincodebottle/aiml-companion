"""CPU measurements of the lltToolUse page (T1). Each sub-command writes runs/lltToolUse/<runId>.json.

Usage (from this folder, with the isolated venv):
  python -I measure.py <cmd> --bfcl <dir holding bfcl_eval> --hermes <dir holding the Hermes json files>
cmd: template | data | roundtrip | maskloss | rescore <run.json>
"""
import argparse, json, os, platform, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "..", "runs", "lltToolUse")
sys.path.insert(0, HERE)
import tool_common as C

PLAN = {"simple_python": 4, "multiple": 4, "parallel": 4, "parallel_multiple": 4, "irrelevance": 4}
MAIN = "Qwen/Qwen3.5-2B-Base"
REF = "Qwen/Qwen3-1.7B-Base"


def env():
    import torch, transformers, trl, tokenizers, numpy
    out = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "trl": trl.__version__, "tokenizers": tokenizers.__version__, "numpy": numpy.__version__}
    try:
        import peft, datasets
        out.update(peft=peft.__version__, datasets=datasets.__version__)
    except ImportError:
        pass
    return out


def save(run_id, body):
    os.makedirs(RUNS, exist_ok=True)
    body = {"runId": run_id, "module": "lltToolUse", "tier": "T1", "hardware": {"cpu": platform.processor(), "threads": 4},
            "env": env(), "date": time.strftime("%Y-%m-%d"), **body}
    json.dump(body, open(os.path.join(RUNS, run_id + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", run_id)


def tokenizers():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MAIN), AutoTokenizer.from_pretrained(REF)


def cmd_template(a):
    from trl.chat_template_utils import get_training_chat_template
    t35, t3 = tokenizers()
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}},
                             "required": ["city"]}}}]
    chat = [{"role": "user", "content": "Weather in Pune for 3 days?"},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Pune", "days": 3, "metric": True}}}]}]
    res = {}
    for name, tok in [("qwen3.5", t35), ("qwen3", t3)]:
        full = tok.apply_chat_template(chat, tools=tools, tokenize=False)
        prompt = tok.apply_chat_template(chat[:1], tools=tools, tokenize=False, add_generation_prompt=True)
        no_tools = tok.apply_chat_template(chat[:1], tokenize=False, add_generation_prompt=True)
        try:
            get_training_chat_template(tok)
            trl_tpl = "available"
        except ValueError as e:
            trl_tpl = "ValueError: " + str(e)[:120]
        res[name] = {"eos": tok.eos_token, "pad": tok.pad_token, "vocab": len(tok), "reply": full[len(prompt):] if full.startswith(prompt) else full[-260:],
                     "promptEndsWith": prompt[-40:], "toolPromptTokens": len(tok(prompt)["input_ids"]) - len(tok(no_tools)["input_ids"]),
                     "trlTrainingTemplate": trl_tpl}
        print(name, json.dumps(res[name], indent=1))
    save("lltToolUse-t1-template", {"label": "template", "results": res})


def load_hermes(path):
    return json.load(open(path, encoding="utf-8"))


def cmd_data(a):
    from trl.chat_template_utils import get_training_chat_template
    t35, t3 = tokenizers()
    train_tpl = get_training_chat_template(t35)
    single = load_hermes(os.path.join(a.hermes, "func-calling-singleturn.json"))
    glaive = load_hermes(os.path.join(a.hermes, "glaive-function-calling-5k.json"))
    render = lambda c: t35.apply_chat_template(c["messages"], tools=c["tools"], chat_template=train_tpl)["input_ids"]
    out = {}
    sets = {}
    for name, rows in [("singleturn", single), ("glaive", glaive)]:
        kept, dropped = C.select_rows(rows, 600, render, 1536)
        read = None
        n_seen = 0
        sets[name] = kept
        lens, n_calls, n_turns_with_call, n_tool_msgs = [], 0, 0, 0
        for c in kept:
            lens.append(len(render(c)))
            for m in c["messages"]:
                if m.get("tool_calls"):
                    n_turns_with_call += 1
                    n_calls += len(m["tool_calls"])
                n_tool_msgs += int(m["role"] == "tool")
        out[name] = {"rowsInFile": len(rows), "kept": len(kept), "dropped": dropped, "rowsRead": len(kept) + sum(dropped.values()),
                     "meanTokens": round(sum(lens) / len(lens), 1), "maxTokens": max(lens), "calls": n_calls,
                     "assistantTurnsWithCalls": n_turns_with_call, "toolResponses": n_tool_msgs}
        print(name, out[name])
    # function-name overlap with the BFCL subset (a contamination check by name only)
    sys.path.insert(0, a.bfcl)
    items = C.load_bfcl(os.path.join(a.bfcl, "bfcl_eval", "data"), PLAN)
    train_names = {t["function"]["name"] for s in sets.values() for c in s for t in c["tools"]}
    bfcl_names = {f["name"] for it in items for f in it["functions"]}
    out["bfclSubset"] = {"items": len(items), "byCategory": {k: sum(it["category"] == k for it in items) for k in PLAN},
                         "functionNames": len(bfcl_names), "namesAlsoInTraining": sorted(bfcl_names & train_names)}
    print(out["bfclSubset"])
    save("lltToolUse-t1-data", {"label": "data", "results": out})


def render_reply(tok, item, calls_from_gt):
    chat = list(item["messages"]) + [{"role": "assistant", "content": "", "tool_calls": calls_from_gt}]
    full = tok.apply_chat_template(chat, tools=item["tools"], tokenize=False)
    prompt = tok.apply_chat_template(item["messages"], tools=item["tools"], tokenize=False, add_generation_prompt=True)
    assert full.startswith(prompt), "template is not prefix-preserving here"
    return full[len(prompt):].split("<|im_end|>")[0]


def first_value(v):
    # BFCL nests allowed-value lists inside dict answers: {"width": [20]} means width must be 20.
    if isinstance(v, dict):
        return {k: first_value([x for x in vals if x != ""][0]) for k, vals in v.items() if [x for x in vals if x != ""]}
    if isinstance(v, list):
        return [first_value(x) for x in v]
    return v


def gt_calls(item):
    calls = []
    for exp in item["ground_truth"]:
        name, allowed = next(iter(exp.items()))
        args = {}
        for p, vals in allowed.items():
            v = [x for x in vals if x != ""]
            if v:
                args[p] = first_value(v[0])
        calls.append({"type": "function", "function": {"name": name, "arguments": args}})
    return calls


def cmd_roundtrip(a):
    # Render each ground-truth answer in each family's format, parse it back and score it with our check and with
    # BFCL's checker. A correct pipeline scores every item right (irrelevance items get an empty reply).
    sys.path.insert(0, a.bfcl)
    official = C.official_check()
    items = C.load_bfcl(os.path.join(a.bfcl, "bfcl_eval", "data"), PLAN)
    t35, t3 = tokenizers()
    res = {}
    examples = {}
    for name, tok, parse in [("qwen3.5", t35, C.parse_xml_calls), ("qwen3", t3, C.parse_json_calls)]:
        own = off = agree = 0
        fails = []
        for it in items:
            text = "" if it["ground_truth"] is None else render_reply(tok, it, gt_calls(it))
            calls = parse(text, it["tools"])
            o, f = C.check(it, calls), official(it, calls)
            own += o; off += f; agree += (o == f)
            if not (o and f) or o != f:
                fails.append({"id": it["id"], "text": text[:300], "calls": calls, "own": o, "official": f})
            if it["id"] in ("simple_python_0", "parallel_0") and it["id"] not in examples.get(name, {}):
                examples.setdefault(name, {})[it["id"]] = text
        res[name] = {"items": len(items), "ownCorrect": own, "officialCorrect": off, "agree": agree, "fails": fails[:10]}
        print(name, {k: v for k, v in res[name].items() if k != "fails"}, len(fails))
        for f in fails[:4]:
            print("  FAIL", f)
    # the wrong parser on the right text: Qwen3.5 replies read with the Hermes JSON parser
    wrong = 0
    for it in items:
        if it["ground_truth"] is None:
            continue
        text = render_reply(t35, it, gt_calls(it))
        wrong += int(C.parse_json_calls(text, it["tools"]) is None)
    res["xmlReadWithJsonParser"] = {"relevantItems": sum(it["ground_truth"] is not None for it in items), "unparsed": wrong}
    print(res["xmlReadWithJsonParser"])
    save("lltToolUse-t1-roundtrip", {"label": "roundtrip", "results": res, "examples": examples})


def cmd_maskloss(a):
    # Labels and loss against TRL on CPU, with Qwen3.5-0.8B-Base (same tokenizer and template as the 2B).
    import torch
    from datasets import Dataset, Features, Json, List
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from trl.chat_template_utils import get_training_chat_template
    torch.set_num_threads(4)
    small = "Qwen/Qwen3.5-0.8B-Base"
    tok = AutoTokenizer.from_pretrained(small)
    train_tpl = get_training_chat_template(tok)
    render = lambda c: list(tok.apply_chat_template(c["messages"], tools=c["tools"], chat_template=train_tpl)["input_ids"])
    single = load_hermes(os.path.join(a.hermes, "func-calling-singleturn.json"))
    glaive = load_hermes(os.path.join(a.hermes, "glaive-function-calling-5k.json"))
    s4, _ = C.select_rows(single, 4, render, 1536)
    g4, _ = C.select_rows(glaive, 4, render, 1536)
    convs = s4 + g4
    res = {}
    # full-file count of rows whose calls are not valid JSON
    bad = 0
    for row in single:
        try:
            C.hermes_to_chat(row)
        except (json.JSONDecodeError, KeyError, TypeError):
            bad += 1
    res["singleturnBadJsonRows"] = bad
    # 1. the Arrow trap: Dataset.from_list merges argument keys across rows and fills the gaps with None
    plain = Dataset.from_list([{"messages": c["messages"], "tools": json.dumps(c["tools"])} for c in convs])
    typed = Dataset.from_list([{"messages": c["messages"], "tools": json.dumps(c["tools"])} for c in convs],
                              features=Features({"messages": List(Json()), "tools": Json()}))
    def tokens_of(ds):
        out = []
        for r in ds:
            tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
            out.append(len(tok.apply_chat_template(r["messages"], tools=tools, chat_template=train_tpl)["input_ids"]))
        return out
    nones = sum(str(m.get("tool_calls")).count("None") for r in plain for m in r["messages"] if m.get("tool_calls"))
    res["arrow"] = {"ourTokens": [len(render(c)) for c in convs], "fromListTokens": tokens_of(plain), "jsonFeatureTokens": tokens_of(typed),
                    "noneArgumentsAdded": nones, "argumentKeysInFirstCall": len(plain[0]["messages"][1]["tool_calls"][0]["function"]["arguments"])}
    print(res["arrow"])
    # 2. labels: ours vs TRL assistant_only_loss with its training template
    labels_ours = []
    header = tok.encode("<|im_start|>assistant" + C.NL, add_special_tokens=False)
    im_end, nl = tok.convert_tokens_to_ids("<|im_end|>"), tok.encode(C.NL, add_special_tokens=False)[0]
    res["header"] = header
    model = AutoModelForCausalLM.from_pretrained(small, dtype=torch.float32).eval()
    args = SFTConfig(output_dir=os.path.join(HERE, "tmp-sft-check"), report_to="none", use_cpu=True, bf16=False,
                     assistant_only_loss=True, max_length=2048, per_device_train_batch_size=4)
    t0 = time.time()
    trainer = SFTTrainer(model=model, args=args, processing_class=tok, train_dataset=typed)
    batch = next(iter(trainer.get_train_dataloader()))
    res["trlInitSec"] = round(time.time() - t0, 1)
    eq = []
    ours = torch.full_like(batch["input_ids"], -100)
    for r in range(batch["input_ids"].shape[0]):
        n = int(batch["attention_mask"][r].sum())
        ids = batch["input_ids"][r, :n].tolist()
        ours[r, :n] = torch.tensor(C.build_labels(ids, header, im_end, nl))
    res["labelsEqual"] = bool(torch.equal(ours, batch["labels"]))
    res["batchShape"] = list(batch["input_ids"].shape)
    keep_all = (ours != -100)
    res["lossShare"] = round(float(keep_all.sum() / batch["attention_mask"].sum()), 4)
    # tool responses carry no loss
    with torch.no_grad():
        num = 0.0
        cnt = 0
        for r in range(batch["input_ids"].shape[0]):
            x = batch["input_ids"][r:r + 1]
            logits = model(input_ids=x, attention_mask=batch["attention_mask"][r:r + 1], use_cache=False).logits[0].double()
            y = ours[r, 1:]
            keep = y != -100
            num += torch.nn.functional.cross_entropy(logits[:-1][keep], y[keep], reduction="sum").item()
            cnt += int(keep.sum())
        own = num / cnt
        lib = trainer.compute_loss(trainer.model, dict(batch)).item()
    res.update(ownLoss=own, trlLoss=lib, absDiff=abs(own - lib), labelledTokens=cnt)
    print({k: v for k, v in res.items() if k != "arrow"})
    # the whole 1,200-row training set: share of tokens with loss and the share inside tool responses
    t35, _ = tokenizers()
    save("lltToolUse-t1-maskloss", {"label": "mask and loss vs TRL", "model": small, "results": res})


def cmd_gencheck(a):
    # Code-path check of generate() and the constrained decoder on CPU with Qwen3.5-0.8B-Base, 6 BFCL items.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    sys.path.insert(0, a.bfcl)
    items = C.load_bfcl(os.path.join(a.bfcl, "bfcl_eval", "data"), PLAN)
    pick = [items[0], items[1], items[100], items[150], items[250], items[251]]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base", dtype=torch.float32).eval()
    t0 = time.time()
    plain = C.generate(model, tok, pick, batch_size=6, max_new_tokens=160, device="cpu")
    t1 = time.time()
    single = [C.generate(model, tok, [it], batch_size=1, max_new_tokens=160, device="cpu")[0] for it in pick[:2]]
    t2 = time.time()
    cons = C.generate(model, tok, pick, batch_size=6, max_new_tokens=160, constrained=True, device="cpu")
    t3 = time.time()
    res = {"plain": plain, "singleEqualsBatched": [x == y for x, y in zip(single, plain[:2])], "single": single, "constrained": cons,
           "plainParsed": [C.parse_xml_calls(t, it["tools"]) for t, it in zip(plain, pick)],
           "constrainedParsed": [C.parse_xml_calls(t, it["tools"]) for t, it in zip(cons, pick)],
           "sec": {"plain": round(t1 - t0, 1), "single2": round(t2 - t1, 1), "constrained": round(t3 - t2, 1)}}
    for k in ("singleEqualsBatched", "plainParsed", "constrainedParsed", "sec"):
        print(k, res[k])
    for t in cons:
        print("----", t[:400])
    json.dump(res, open(os.path.join(HERE, "tmp-gencheck.json"), "w"), indent=1)


def cmd_specialtokens(a):
    # Embedding rows of the chat and tool special tokens in both bases. Identical small rows mean the base never
    # trained them, so a model that keeps embeddings frozen cannot learn to write them.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    SPECIAL = ["<tool_call>", "</tool_call>", "<|im_end|>", "<think>", "</think>"]
    res = {}
    for name in (REF, MAIN):
        tok = AutoTokenizer.from_pretrained(name)
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        E = m.get_input_embeddings().weight.detach()
        norms = E.norm(dim=1)
        ids = [tok.convert_tokens_to_ids(s) for s in SPECIAL]
        rows = E[ids]
        res[name] = {"tied": m.get_output_embeddings().weight is m.get_input_embeddings().weight, "rows": E.shape[0],
                     "medianNorm": round(norms.median().item(), 4), "special": {s: {"id": i, "norm": round(norms[i].item(), 4)} for s, i in zip(SPECIAL, ids)},
                     "allSpecialRowsIdentical": bool(all(torch.equal(rows[0], r) for r in rows[1:])),
                     "oddToken": {"text": chr(0x2B7E6), "ids": tok.encode(chr(0x2B7E6), add_special_tokens=False)}}
        res[name]["oddToken"]["norms"] = [round(norms[i].item(), 4) for i in res[name]["oddToken"]["ids"]]
        print(name, json.dumps(res[name]))
        del m
    save("lltToolUse-t1-specialtokens", {"label": "special-token embedding rows", "results": res})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--bfcl", default="")
    ap.add_argument("--hermes", default="")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)
