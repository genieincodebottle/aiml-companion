"""Generate cells_qlora.py from cells_finetune.py: keep setup, data, the run() helper and the QLoRA cell.

Fixes applied to run() for QLoRA on a T4 (measured traps):
- build the PEFT model ourselves and keep adapters in fp32 (run 2: bf16 adapters broke fp16 GradScaler);
- after SFTTrainer init, record any trainable parameter that is not fp32 and cast it (run 3: still bf16 grads).
"""
import runpy

cells = runpy.run_path('cells_finetune.py')['CELLS']
keep = [c for c in cells if not (c[0] == 'code' and ('# 1. Full SFT' in c[1] or '# 2. LoRA on Qwen3' in c[1]))]
assert len(keep) == len(cells) - 2

PEFT_OLD = "        tr = SFTTrainer(model=model, args=args, train_dataset=data, processing_class=tok, peft_config=peft_cfg, callbacks=[timer])"
PEFT_NEW = '''        if kbit and peft_cfg is not None:
            # QLoRA on a T4: build the PEFT model here and keep the adapters in fp32.
            from peft import get_peft_model
            model = get_peft_model(model, peft_cfg)
            for p in model.parameters():
                if p.requires_grad and p.dtype != torch.float32:
                    p.data = p.data.float()
            out["adapterDtypes"] = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
            peft_cfg = None
        tr = SFTTrainer(model=model, args=args, train_dataset=data, processing_class=tok, peft_config=peft_cfg, callbacks=[timer])'''

TRAIN_OLD = "        res = tr.train()"
TRAIN_NEW = '''        # List trainable params that are not fp32 AFTER SFTTrainer init (something re-casts them), then cast.
        bad = [(n, str(q.dtype)) for n, q in tr.model.named_parameters() if q.requires_grad and q.dtype != torch.float32]
        out["nonFp32TrainableAfterInit"] = {"count": len(bad), "examples": bad[:6]}
        for n, q in tr.model.named_parameters():
            if q.requires_grad and q.dtype != torch.float32:
                q.data = q.data.float()
        print("non-fp32 trainable after init:", len(bad), bad[:6])
        res = tr.train()'''

fixed = []
for kind, s in keep:
    if kind == 'code' and 'def run(name, model_id' in s:
        assert s.count(PEFT_OLD) == 1 and s.count(TRAIN_OLD) == 1
        s = s.replace(PEFT_OLD, PEFT_NEW).replace(TRAIN_OLD, TRAIN_NEW)
    if kind == 'code' and 'llt_finetune_results.json' in s:
        s = s.replace('llt_finetune_results.json', 'llt_qlora_results.json')
    if kind == 'md' and s.startswith('# LLM Training Lab: fine-tuning test runs'):
        s = ('# LLM Training Lab: QLoRA test on a free T4\n\nQLoRA on Qwen3-8B base: NF4 + double quantization, '
             'fp16 compute, LoRA adapters kept in fp32. Writes `llt_qlora_results.json`.')
    fixed.append((kind, s))

with open('cells_qlora.py', 'w', encoding='utf8') as f:
    f.write('CELLS = ' + repr(fixed) + '\n')
compile(open('cells_qlora.py', encoding='utf8').read(), 'cells_qlora.py', 'exec')
print('cells_qlora.py written and compiles:', len(fixed), 'cells')
