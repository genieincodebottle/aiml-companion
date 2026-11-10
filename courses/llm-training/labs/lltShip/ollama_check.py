"""Recorded Ollama check for step 11 (CPU only, no GPU lock needed). Run inside a work dir that holds merged-Q4_K_M.gguf,
Modelfile and prompts.json (20 held-out prompts), with an `ollama` binary on PATH:

    python ollama_check.py <path to ollama binary>

It serves Ollama on 127.0.0.1:11500 with the GPU hidden, creates two models from the same GGUF file, one with the
Modelfile's TEMPLATE and one with no TEMPLATE line, sends the prompts to each through /api/chat and writes
ollama_check.json: stop rate (done_reason "stop"), replies that run into a new "### Instruction", first reply, seconds.
"""
import json, os, subprocess, sys, time, urllib.request

OLLAMA = sys.argv[1]
env = dict(os.environ, OLLAMA_HOST="127.0.0.1:11500", CUDA_VISIBLE_DEVICES="", OLLAMA_MODELS=os.path.abspath("ollama-models"))
version = subprocess.run([OLLAMA, "--version"], capture_output=True, text=True, env=env).stdout.strip()
server = subprocess.Popen([OLLAMA, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
url = "http://127.0.0.1:11500"
for _ in range(60):
    try:
        urllib.request.urlopen(url + "/api/version")
        break
    except Exception:
        time.sleep(1)

modelfile = open("Modelfile").read()
open("Modelfile.notemplate", "w").write("\n".join(l for l in modelfile.splitlines()
                                                  if not l.startswith(("TEMPLATE", "{{", "###", '"""')) and l.strip()) + "\n")
prompts = json.load(open("prompts.json"))
out = {"ollama": version, "gguf": "merged-Q4_K_M.gguf", "prompts": len(prompts)}
for name, mf in (("ship", "Modelfile"), ("ship-notemplate", "Modelfile.notemplate")):
    subprocess.run([OLLAMA, "create", name, "-f", mf], env=env, check=True, capture_output=True)
    shown = subprocess.run([OLLAMA, "show", name, "--template"], env=env, capture_output=True, text=True).stdout
    stops, leaks, first, t = 0, 0, None, time.time()
    for q in prompts:
        body = {"model": name, "messages": [{"role": "user", "content": q}], "stream": False,
                "options": {"temperature": 0, "num_predict": 256}}
        req = urllib.request.Request(url + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        text = r["message"]["content"]
        stops += r.get("done_reason") == "stop"
        leaks += "### Instruction" in text or "<|im_start|>" in text
        first = first if first is not None else text
    out[name] = {"stopRate": stops / len(prompts), "leakRate": leaks / len(prompts), "sec": round(time.time() - t, 1),
                 "template": shown[:300], "first": first[:400]}
    print(name, {k: v for k, v in out[name].items() if k not in ("first", "template")}, flush=True)
server.kill()
json.dump(out, open("ollama_check.json", "w"), indent=1)
print("WROTE ollama_check.json", version)
