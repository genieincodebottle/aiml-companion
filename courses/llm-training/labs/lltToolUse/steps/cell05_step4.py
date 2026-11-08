import bfcl_eval                                     # pip install --no-deps bfcl-eval==2026.3.23, in the install cell
BFCL = os.path.join(os.path.dirname(bfcl_eval.__file__), "data")
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

PLAN = {"simple_python": 4, "multiple": 4, "parallel": 4, "parallel_multiple": 4, "irrelevance": 4}   # every 4th item
items = C.load_bfcl(BFCL, PLAN)
print(len(items), "items", {k: sum(it["category"] == k for it in items) for k in PLAN})
it = items[0]
print(it["messages"][0]["content"])
print(it["functions"][0]["parameters"]["properties"]["base"], "->", it["tools"][0]["function"]["parameters"]["type"])
print("answer:", it["ground_truth"])
