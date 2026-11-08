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

par = next(it for it in items if it["category"] == "parallel")
reply = C.gt_reply(tok, par)                       # the right answer to the first parallel item, in Qwen3.5 format
print(reply)
print("XML parser :", parse_xml_calls(reply, par["tools"]))
print("JSON parser:", C.parse_json_calls(reply, par["tools"]))
print("Qwen3 reply:", C.gt_reply(ref_tok, par)[:120])
