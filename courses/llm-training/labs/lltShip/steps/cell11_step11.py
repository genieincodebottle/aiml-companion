import urllib.request

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())

def serve(gguf_file, port, extra=()):
    # llama-server on the CPU, OpenAI-compatible, using the chat template stored in the GGUF file (--jinja)
    p = subprocess.Popen([BIN + "/llama-server", "-m", gguf_file, "--port", str(port), "-c", "2048", "--jinja",
                          "-t", str(os.cpu_count()), *extra], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            if json.loads(urllib.request.urlopen("http://127.0.0.1:" + str(port) + "/health").read())["status"] == "ok":
                return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise RuntimeError("llama-server did not start")

def smoke(port, n=20):
    # The post-deployment smoke eval: same prompts as the panel, through the endpoint a user would call
    url = "http://127.0.0.1:" + str(port)
    stops, leaks, same_ids, outs = 0, 0, 0, []
    t = time.time()
    for q in prompts[:n]:
        r = post(url + "/v1/chat/completions", {"messages": [{"role": "user", "content": q}], "max_tokens": 256, "temperature": 0})
        text = r["choices"][0]["message"]["content"]
        stops += r["choices"][0]["finish_reason"] == "stop"
        leaks += "### Instruction" in text or "<|im_start|>" in text
        same_ids += post(url + "/tokenize", {"content": TEMPLATE.format(q)})["tokens"] == tok(TEMPLATE.format(q)).input_ids
        outs.append(text)
    return {"stopRate": stops / n, "leakRate": leaks / n, "sameTokenIds": same_ids / n, "sec": round(time.time() - t, 1), "first": outs[0]}

server = serve("merged-Q4_K_M.gguf", 8090)
runs["serve ours"] = smoke(8090)
server.kill()
print("our template    ", {k: v for k, v in runs["serve ours"].items() if k != "first"})
print("   first reply:", runs["serve ours"]["first"][:300])

# The trap: serve the same file with ChatML, the usual Qwen chat format, whose control tokens this base model never trained
server = serve("merged-Q4_K_M.gguf", 8091, ["--chat-template", "chatml"])
runs["serve chatml"] = smoke(8091)
server.kill()
print("ChatML template ", {k: v for k, v in runs["serve chatml"].items() if k != "first"})
print("   first reply:", runs["serve chatml"]["first"][:300])

# For Ollama on your own machine: the same GGUF file, with the template written out in Ollama's own format
MODELFILE = '''FROM ./merged-Q4_K_M.gguf
TEMPLATE """### Instruction:
{{ .Prompt }}

### Response:
"""
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0
'''
open("Modelfile", "w").write(MODELFILE)
json.dump(prompts[:20], open("prompts.json", "w"))         # the same 20 prompts, for the Ollama check
