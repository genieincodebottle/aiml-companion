"""One generate() for every stage: vLLM on T3, Hugging Face generate on T2. Same inputs, same outputs.

inputs:  a list of plain prompt strings (our models, already in TEMPLATE) or a list of chat message lists
         (the official instruct model and the judges, rendered with their own chat template).
returns: one list per input of n dicts {text, tokens, finished}; finished is False when the reply hit max_new.
"""
import gc
import os


def generate(model_name, inputs, *, backend, n=1, temperature=0.0, top_p=1.0, max_new=1024, seed=0, chat_kwargs=None,
             max_model_len=4096, dtype="bfloat16", quant=None, gpu_util=0.85):
    if backend == "vllm":
        return _vllm(model_name, inputs, n, temperature, top_p, max_new, seed, chat_kwargs, max_model_len, dtype, gpu_util)
    return _hf(model_name, inputs, n, temperature, top_p, max_new, seed, chat_kwargs, dtype, quant)


def _vllm(model_name, inputs, n, temperature, top_p, max_new, seed, chat_kwargs, max_model_len, dtype, gpu_util, llm=None):
    from vllm import LLM, SamplingParams
    own = llm is None
    if own:
        llm = vllm_load(model_name, max_model_len, dtype, gpu_util, seed)
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_new, seed=seed)
    if inputs and isinstance(inputs[0], list):
        outs = llm.chat(inputs, sp, use_tqdm=True, chat_template_kwargs=chat_kwargs or {})
    else:
        outs = llm.generate(inputs, sp, use_tqdm=True)
    res = [[{"text": o.text, "tokens": len(o.token_ids), "finished": o.finish_reason == "stop"} for o in out.outputs]
           for out in outs]
    if own:
        del llm
        gc.collect()
        _free_vllm()
    return res


def vllm_load(model_name, max_model_len, dtype, gpu_util, seed):
    from vllm import LLM
    return LLM(model=model_name, dtype=dtype, max_model_len=max_model_len, gpu_memory_utilization=gpu_util, seed=seed,
               enable_prefix_caching=True)


def _free_vllm():
    try:
        import torch
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _hf(model_name, inputs, n, temperature, top_p, max_new, seed, chat_kwargs, dtype, quant):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": getattr(torch, dtype)}
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=getattr(torch, dtype))
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0} if torch.cuda.is_available() else None, **kw).eval()
    if inputs and isinstance(inputs[0], list):
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **(chat_kwargs or {})) for m in inputs]
    else:
        texts = list(inputs)
    stop = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            stop.add(i)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))      # similar lengths together: less padding
    res = [None] * len(texts)
    bs = int(os.environ.get("GEN_BATCH", "16"))
    torch.manual_seed(seed)
    def run_batch(idx):
        """Generate for the prompts in idx. The longest prompts come last (sorted by length) and a batch that fit
        for the short ones can run out of GPU memory on a 15 GB T4: split that batch in two and go on."""
        oom = False
        try:
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=temperature > 0,
                                     temperature=temperature if temperature > 0 else None, top_p=top_p if temperature > 0 else None,
                                     num_return_sequences=n, eos_token_id=sorted(stop), pad_token_id=tok.pad_token_id)
        except torch.OutOfMemoryError:
            if len(idx) == 1:
                raise
            oom = True
        if oom:                      # split outside the except block, so the failed batch's memory is released first
            enc = out = None
            gc.collect()
            torch.cuda.empty_cache()
            print("out of memory on a batch of", len(idx), "prompts: splitting it", flush=True)
            half = len(idx) // 2
            run_batch(idx[:half])
            run_batch(idx[half:])
            return
        gen = out[:, enc.input_ids.shape[1]:].tolist()
        for j, i in enumerate(idx):
            res[i] = []
            for s in gen[j * n:(j + 1) * n]:
                cut = next((p for p, t in enumerate(s) if t in stop), None)
                ids = s if cut is None else s[:cut]
                res[i].append({"text": tok.decode(ids, skip_special_tokens=True), "tokens": len(ids), "finished": cut is not None})

    for k in range(0, len(order), bs):
        run_batch(order[k:k + bs])
        print("generated", min(k + bs, len(order)), "/", len(order), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def run_job(job, inputs, out_path):
    """Run one generate() in a fresh process (vLLM frees GPU memory reliably only when its process exits).
    Writes inputs and settings next to out_path, then reads back the results. Skips the job if out_path exists."""
    import json
    import subprocess
    import sys
    from .common import done, read_json, write_json
    if done(out_path):
        return read_json(out_path)
    spec = out_path + ".job.json"
    write_json(spec, {"job": job, "inputs": inputs, "out": out_path})
    subprocess.run([sys.executable, "-m", "capstone.gen", spec], check=True)
    return read_json(out_path)


def run_multi(model_args, groups):
    """Several generate() calls on ONE loaded model (vLLM loads and compiles once, about a minute, instead of once per
    call). model_args: model, backend, seed, max_model_len, dtype, quant. groups: [{"inputs", "params", "out"}], where
    params holds n, temperature, top_p, max_new, chat_kwargs. Groups whose out file exists are skipped."""
    import subprocess
    import sys
    from .common import done, read_json, write_json
    todo = [g for g in groups if not done(g["out"])]
    if todo:
        spec = todo[0]["out"] + ".multi.json"
        write_json(spec, {"model_args": model_args, "groups": todo})
        subprocess.run([sys.executable, "-m", "capstone.gen", spec], check=True)
    return [read_json(g["out"]) for g in groups]


def _main_multi(spec):
    import time
    from .common import write_json
    ma = dict(spec["model_args"])
    model = ma.pop("model")
    t0 = time.time()
    llm = None
    if ma["backend"] == "vllm":
        llm = vllm_load(model, ma.get("max_model_len", 4096), ma.get("dtype", "bfloat16"), ma.get("gpu_util", 0.85), ma.get("seed", 0))
    load_sec = round(time.time() - t0, 1)
    for g in spec["groups"]:
        t1 = time.time()
        pr = dict(n=1, temperature=0.0, top_p=1.0, max_new=1024, chat_kwargs=None)
        pr.update(g["params"])
        if llm is not None:
            res = _vllm(model, g["inputs"], pr["n"], pr["temperature"], pr["top_p"], pr["max_new"], ma.get("seed", 0),
                        pr["chat_kwargs"], None, None, None, llm=llm)
        else:
            res = _hf(model, g["inputs"], pr["n"], pr["temperature"], pr["top_p"], pr["max_new"], ma.get("seed", 0),
                      pr["chat_kwargs"], ma.get("dtype", "float16"), ma.get("quant"))
        sec = round(time.time() - t1, 1)
        write_json(g["out"], {"results": res, "wallClockSec": sec, "loadSec": load_sec,
                              "tokens": sum(o["tokens"] for r in res for o in r)})
        print("group done in", sec, "s:", g["out"], flush=True)


if __name__ == "__main__":
    import json
    import sys
    import time
    from .common import peak_gb, read_json, write_json
    spec = read_json(sys.argv[1])
    if "groups" in spec:
        _main_multi(spec)
        sys.exit(0)
    t0 = time.time()
    res = generate(spec["job"].pop("model"), spec["inputs"], **spec["job"])
    sec = round(time.time() - t0, 1)
    write_json(spec["out"], {"results": res, "wallClockSec": sec, "peakMemGB": peak_gb(),
                             "tokens": sum(o["tokens"] for r in res for o in r)})
    print("job done in", sec, "s")
