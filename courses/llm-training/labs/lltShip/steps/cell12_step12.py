RUN_VLLM = False      # True builds a separate vLLM environment (it pins its own torch) and serves on this GPU; recorded on an A100.
                      # With FROM_OUR_CHECKPOINT = True and this False, step 12 prints our recorded vLLM numbers

# 1. Take the SFT fine-tune apart again: its task vector has rank 16 per layer, so an SVD recovers a LoRA adapter
from safetensors.torch import save_file
lora, worst = {}, 0.0
for k in W["base"]:
    if k.endswith("proj.weight") and "embed" not in k:
        d = tau("sft", k)
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        B, A = U[:, :16] * S[:16].sqrt(), S[:16].sqrt()[:, None] * Vh[:16]
        worst = max(worst, float((B @ A - d).norm() / d.norm()))
        name = "base_model.model." + k[:-len(".weight")]
        lora[name + ".lora_A.weight"], lora[name + ".lora_B.weight"] = A.contiguous(), B.contiguous()
os.makedirs("sft-lora", exist_ok=True)
save_file(lora, "sft-lora/adapter_model.safetensors")
json.dump({"base_model_name_or_path": BASE, "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16, "lora_alpha": 16,
           "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
          open("sft-lora/adapter_config.json", "w"))
print(len(lora) // 2, "layers | largest relative error of the rank-16 rebuild", round(worst, 4))

# 2. One merged model, or one base model with several adapters swapped per request
VLLM = '''import json, sys, time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
mode, prompts = sys.argv[1], json.load(open("vllm_prompts.json"))
sp = SamplingParams(temperature=0, max_tokens=256)
if mode == "multi-lora":
    llm = LLM(model=sys.argv[2], dtype="float16", max_model_len=1024, gpu_memory_utilization=0.5, enable_lora=True, max_lora_rank=16, max_loras=2)
    reqs = [LoRARequest("sft", 1, "sft-lora"), LoRARequest("math", 2, sys.argv[3])]
    t = time.time()
    outs = llm.generate(prompts + prompts, sp, lora_request=[reqs[0]] * len(prompts) + [reqs[1]] * len(prompts))
else:
    llm = LLM(model=sys.argv[2], dtype="float16", max_model_len=1024, gpu_memory_utilization=0.5)
    t = time.time()
    outs = llm.generate(prompts, sp)
sec = time.time() - t
toks = [list(o.outputs[0].token_ids) for o in outs]
print("RESULT", json.dumps({"sec": round(sec, 2), "tokens": sum(map(len, toks)), "requests": len(toks),
                            "stopRate": sum(o.outputs[0].finish_reason == "stop" for o in outs) / len(outs), "ids": toks}))
'''
if RUN_VLLM:
    side_env("vllm-env", ["vllm==0.33.0"], share=False)      # vLLM 0.33.0 brings its own torch 2.14.0
    open("vllm_job.py", "w").write(VLLM)
    json.dump([TEMPLATE.format(q) for q in prompts], open("vllm_prompts.json", "w"))
    math_lora = snapshot_download(HUB, allow_patterns=["lltDistillation/onpolicy-reverse-s0/*"]) + "/lltDistillation/onpolicy-reverse-s0"
    model.to("cpu")                                        # give the GPU to vLLM
    torch.cuda.empty_cache()
    res = {}
    for mode, args in [("merged", ["merged"]), ("sft-full", [DIRS["sft"]]), ("multi-lora", [DIRS["base"], math_lora])]:
        # The FlashInfer sampler compiles a CUDA kernel on first use and needs ninja and nvcc on PATH; vLLM's PyTorch sampler needs neither
        sec, out = sh(["vllm-env/bin/python", "vllm_job.py", mode, *args], env=dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0"))
        res[mode] = json.loads(out.split("RESULT ")[1].splitlines()[0])
        r = res[mode]
        print(mode.ljust(10), r["requests"], "requests |", r["tokens"], "tokens in", r["sec"], "s =", round(r["tokens"] / r["sec"]),
              "tokens/s | stop rate", round(r["stopRate"], 2), "| process", sec, "s")
    n = len(prompts)
    same = sum(a == b for a, b in zip(res["sft-full"]["ids"], res["multi-lora"]["ids"][:n]))
    print("SFT model vs base + rebuilt SFT adapter: identical greedy replies", same, "of", n)
    runs["vllm"] = {k: {kk: vv for kk, vv in v.items() if kk != "ids"} for k, v in res.items()}
    runs["vllm"]["identical"] = same
    model.to(DEV)
elif FROM_OUR_CHECKPOINT:
    runs["vllm"] = REC["vllm"]
    for mode in ("merged", "sft-full", "multi-lora"):
        r = runs["vllm"][mode]
        print(mode.ljust(10), r["requests"], "requests |", r["tokens"], "tokens in", r["sec"], "s =", round(r["tokens"] / r["sec"]),
              "tokens/s | stop rate", round(r["stopRate"], 2), "| recorded on our A100")
    print("SFT model vs base + rebuilt SFT adapter: identical greedy replies", runs["vllm"]["identical"], "of", runs["vllm"]["merged"]["requests"], "| recorded")
