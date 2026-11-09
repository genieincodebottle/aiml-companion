USE_OUR_ADAPTER = False    # True loads our trained adapter from the Hub instead of training this arm on your T4
name = "On-policy, reverse KL on student samples"

if USE_OUR_ADAPTER:
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    folder = snapshot_download("aimlcompanion/llm-training-lab", allow_patterns=["lltDistillation/onpolicy-reverse-s0/*"])
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float16).to(GPU),
                                      os.path.join(folder, "lltDistillation", "onpolicy-reverse-s0"))
    sync(model, student)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    evaluate(student, name)
else:
    run_arm(name, "reverse", "student")
    r = R[name]
    print("sampling", r["sampleSec"], "s of", r["trainSec"], "s training | loss at step 1", r["loss"][0], "| at step", len(r["loss"]), r["loss"][-1])