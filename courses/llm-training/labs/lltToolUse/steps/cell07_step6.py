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
    return all(C.value_ok(v, allowed[p]) for p, v in args.items())

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

official = C.official_check()                      # BFCL's ast_checker from the bfcl-eval package
for name, t, parse in [("Qwen3.5 XML", tok, parse_xml_calls), ("Qwen3 JSON", ref_tok, C.parse_json_calls)]:
    ours = theirs = agree = 0
    for it in items:
        calls = parse(C.gt_reply(t, it), it["tools"])
        a, b = check(it, calls), official(it, calls)
        ours, theirs, agree = ours + a, theirs + b, agree + (a == b)
    print(name, "| right answers scored correct, ours", ours, "BFCL", theirs, "of", len(items), "| agree", agree)
