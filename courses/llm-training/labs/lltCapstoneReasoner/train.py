"""Resumable GRPO on Countdown, using the lltRL lab's GRPO code (rl_core).

The update inside each step is the lltRL lab's train_grpo update, line for line in order: sample G replies per
prompt, score them, compute group advantages, take old and reference log-probs, then split the batch into
minibatches with one optimizer step each. What this file adds: Countdown data, checkpoints you can resume
from after a dropped session or a preempted VM, a held-out greedy evaluation every few steps, and extra
logging (reply length of correct and of wrong replies, answer-tag rate).

  python train.py --out runs/agg-grpo_scale-1_s0 --agg grpo --scale 1 --seed 0
  (run the same command again after an interruption: it continues from the last checkpoint)

Ablation factors (the capstone's 2 x 2):
  --agg   grpo     average each reply's token losses, then average over replies (the original GRPO)
          dr_grpo  sum every token loss, divide by a constant (replies x max length) (Dr. GRPO)
          dapo     sum every token loss, divide by the number of tokens in the batch (TRL's default)
  --scale 1        divide each group's advantages by the group's reward std (the original GRPO)
          0        do not (Dr. GRPO)
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import countdown as cd   # noqa: E402
import rl_core           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
ap.add_argument("--agg", default="grpo", choices=["grpo", "dr_grpo", "dapo"])
ap.add_argument("--scale", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--prompts", type=int, default=8)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--beta", type=float, default=0.0)
ap.add_argument("--eps", type=float, default=0.2)
ap.add_argument("--eps_high", type=float, default=None)
ap.add_argument("--minibatches", type=int, default=2)
ap.add_argument("--micro", type=int, default=8)
ap.add_argument("--max_new", type=int, default=256)
ap.add_argument("--gen_batch", type=int, default=64)
ap.add_argument("--eval_every", type=int, default=25)
ap.add_argument("--n_test", type=int, default=200)
ap.add_argument("--ckpt_every", type=int, default=5)
ap.add_argument("--save_at", default="50", help="comma-separated steps to keep a copy of the adapter, e.g. 50,100")
ap.add_argument("--data", default=str(HERE / "data"))
ap.add_argument("--prompt", default="oneshot", choices=["zero", "oneshot"])
ap.add_argument("--reward", default="format", choices=["binary", "format"],
                help="binary: 1 correct, else 0. format: also 0.1 for a well-formed but wrong equation (TinyZero's format score)")
ap.add_argument("--train_file", default="train3.jsonl")
ap.add_argument("--max_minutes", type=float, default=0, help="stop cleanly after this many minutes (0 = no limit); resume later")
args = ap.parse_args()
cd.PROMPT_STYLE = args.prompt

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
CKPT, CKPT_OLD, CKPT_NEW = OUT / "ckpt", OUT / "ckpt_old", OUT / "ckpt_new"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = {k: v for k, v in vars(args).items() if k not in ("out", "data")}
json.dump(cfg, open(OUT / "config.json", "w"), indent=1)

qtok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
ns = rl_core.load(args.model, args.max_new, cd.STOPS, qtok, DEV)
json.dump(dict(hashes=ns["_hashes"], steps_dir=str(rl_core.STEPS)), open(OUT / "lltrl_functions.json", "w"), indent=1)
(OUT / "lltrl_functions_used.py").write_text("\n\n".join(ns["_sources"].values()), encoding="utf-8")
group_advantages, grpo_loss, completion_logps = ns["group_advantages"], ns["grpo_loss"], ns["completion_logps"]
load_policy, sample, kl_k3, autocast = ns["load_policy"], ns["sample"], ns["kl_k3"], ns["autocast"]

read = lambda f: [json.loads(line) for line in open(Path(args.data) / f, encoding="utf-8")]   # noqa: E731
train = [dict(q=cd.problem_text(p), gold=cd.problem_text(p), n=len(p["nums"])) for p in read(args.train_file)]
test = [dict(q=cd.problem_text(p), gold=cd.problem_text(p), n=len(p["nums"])) for p in read("test.jsonl")][:args.n_test]
assert len(train) >= args.steps * args.prompts, "not enough training problems for this many steps"


def evaluate(model):
    rows = sample(model, [cd.prompt(x["q"]) for x in test], temperature=0, batch=max(args.gen_batch, 64))
    replies = [qtok.decode(r) for r in rows]
    ok = [cd.correct(r, x["gold"]) for r, x in zip(replies, test)]
    by_n = {str(n): round(sum(o for o, x in zip(ok, test) if x["n"] == n) / max(1, sum(x["n"] == n for x in test)), 4) for n in (3, 4)}
    return dict(greedy=round(sum(ok) / len(ok), 4), byNumbers=by_n, meanTokens=round(sum(len(r) for r in rows) / len(rows), 1),
                answerTag=round(sum(cd.extract_answer(r) is not None for r in replies) / len(replies), 4))


def save_ckpt(model, opt, scaler, step):
    if CKPT_NEW.exists():
        shutil.rmtree(CKPT_NEW)
    CKPT_NEW.mkdir()
    save_file({k: v.contiguous() for k, v in get_peft_model_state_dict(model).items()}, str(CKPT_NEW / "adapter.safetensors"))
    torch.save(dict(opt=opt.state_dict(), scaler=scaler.state_dict(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if DEV.type == "cuda" else None, py_rng=random.getstate()),
               CKPT_NEW / "state.pt")
    json.dump(dict(step=step, time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(CKPT_NEW / "step.json", "w"))
    if CKPT.exists():
        CKPT.rename(CKPT_OLD)
    CKPT_NEW.rename(CKPT)
    if CKPT_OLD.exists():
        shutil.rmtree(CKPT_OLD)


def load_ckpt(model, opt, scaler):
    d = CKPT if (CKPT / "step.json").exists() else CKPT_OLD if (CKPT_OLD / "step.json").exists() else None
    if d is None:
        return 0
    set_peft_model_state_dict(model, {k: v.to(DEV) for k, v in load_file(str(d / "adapter.safetensors")).items()})
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    st = torch.load(d / "state.pt", weights_only=False)
    opt.load_state_dict(st["opt"])
    scaler.load_state_dict(st["scaler"])
    torch.set_rng_state(st["torch_rng"])
    if st["cuda_rng"] is not None and DEV.type == "cuda":
        torch.cuda.set_rng_state_all(st["cuda_rng"])
    random.setstate(st["py_rng"])
    return json.load(open(d / "step.json"))["step"]


def main():
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = load_policy()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=2048, enabled=DEV.type == "cuda")
    start = load_ckpt(model, opt, scaler)
    mfile, efile = OUT / "metrics.jsonl", OUT / "evals.jsonl"
    for f in (mfile, efile, OUT / "groups.jsonl"):                 # drop lines written after the checkpoint we resume from
        if f.exists():
            keep = [line for line in open(f) if json.loads(line)["step"] <= start]
            open(f, "w").writelines(keep)
    print("start at step", start, "| trainable", sum(p.numel() for p in params), "| lltRL functions", ns["_hashes"], flush=True)
    if start == 0:
        e = dict(step=0, **evaluate(model))
        open(efile, "a").write(json.dumps(e) + "\n")
        print("eval", e, flush=True)
    save_at = {int(s) for s in args.save_at.split(",") if s}
    n, G, t_start = args.prompts * args.G, args.G, time.time()
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(start, args.steps):
        ts = time.time()
        batch = train[step * args.prompts:(step + 1) * args.prompts]
        texts = [cd.prompt(x["q"]) for x in batch for _ in range(G)]
        # 1. Sample G replies per prompt and score them
        rows = sample(model, texts, batch=args.gen_batch)
        gen_sec = time.time() - ts
        golds = [x["gold"] for x in batch for _ in range(G)]
        replies = [qtok.decode(r) for r in rows]
        score = cd.correct if args.reward == "binary" else cd.reward_with_format
        rewards = torch.tensor([score(r, g) for r, g in zip(replies, golds)])
        solved = torch.tensor([cd.correct(r, g) for r, g in zip(replies, golds)])
        adv = group_advantages(rewards, G, scale=bool(args.scale)).to(DEV)
        # 2. One padded batch: prompt left-padded, reply right-padded, mask from each reply's length
        penc = qtok(texts, return_tensors="pt", padding=True).to(DEV)
        L = max(len(r) for r in rows)
        comp = torch.full((n, L), qtok.pad_token_id, device=DEV)
        cmask = torch.zeros((n, L), dtype=torch.long, device=DEV)
        for i, r in enumerate(rows):
            comp[i, :len(r)] = torch.tensor(r, device=DEV)
            cmask[i, :len(r)] = 1
        # 3. Log-probs of the sampling policy and of the reference (the base model: LoRA switched off)
        mi = args.micro
        with torch.no_grad(), autocast():
            old_lp = torch.cat([completion_logps(model, penc.input_ids[i:i + mi], penc.attention_mask[i:i + mi], comp[i:i + mi], cmask[i:i + mi])
                                for i in range(0, n, mi)])
            with model.disable_adapter():
                ref_lp = torch.cat([completion_logps(model, penc.input_ids[i:i + mi], penc.attention_mask[i:i + mi], comp[i:i + mi], cmask[i:i + mi])
                                    for i in range(0, n, mi)])
        # 4. Updates: the batch is split into minibatches, one optimizer step each
        model.train()
        clip, ent = [], []
        for mb in torch.randperm(n, device=DEV).chunk(args.minibatches):
            m = cmask[mb]
            denom = {"grpo": len(mb), "dr_grpo": len(mb) * args.max_new}.get(args.agg, m.sum().item())
            opt.zero_grad(set_to_none=True)
            for j in range(0, len(mb), mi):
                idx = mb[j:j + mi]
                with autocast():
                    new_lp = completion_logps(model, penc.input_ids[idx], penc.attention_mask[idx], comp[idx], cmask[idx])
                    loss = grpo_loss(new_lp, old_lp[idx], adv[idx], cmask[idx], args.eps, args.eps_high, args.beta, ref_lp[idx],
                                     args.agg, args.max_new, denom)
                scaler.scale(loss).backward()
                with torch.no_grad():
                    ratio = torch.exp(new_lp - old_lp[idx])
                    a = adv[idx][:, None]
                    hi = args.eps_high or args.eps
                    clipped = ((ratio > 1 + hi) & (a > 0)) | ((ratio < 1 - args.eps) & (a < 0))
                    clip.append(((clipped * cmask[idx]).sum() / cmask[idx].sum()).item())
                    ent.append((-(new_lp * cmask[idx]).sum() / cmask[idx].sum()).item())
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            scaler.step(opt)
            scaler.update()
        lens = [len(r) for r in rows]
        okl = [ln for ln, r in zip(lens, solved.tolist()) if r > 0]
        bad = [ln for ln, r in zip(lens, solved.tolist()) if r == 0]
        rec = dict(step=step + 1, reward=round(rewards.mean().item(), 4), solved=round(solved.mean().item(), 4),
                   zeroStd=round((rewards.view(-1, G).std(1) == 0).float().mean().item(), 3),
                   meanLen=round(sum(lens) / n, 1), lenCorrect=round(sum(okl) / len(okl), 1) if okl else None,
                   lenWrong=round(sum(bad) / len(bad), 1) if bad else None,
                   truncated=round(sum(1 for ln in lens if ln >= args.max_new) / n, 3),
                   answerTag=round(sum(cd.extract_answer(r) is not None for r in replies) / n, 3),
                   kl=round(((kl_k3(old_lp, ref_lp) * cmask).sum() / cmask.sum()).item(), 5), clipFrac=round(sum(clip) / len(clip), 4),
                   entropy=round(sum(ent) / len(ent), 4), gradNorm=round(gnorm, 3), genSec=round(gen_sec, 1), sec=round(time.time() - ts, 1))
        open(mfile, "a").write(json.dumps(rec) + "\n")
        if step == start or (step + 1) % 25 == 0:
            g = dict(step=step + 1, problems=[x["q"] for x in batch[:2]], replies=replies[:2 * G], rewards=rewards[:2 * G].tolist(),
                     tokens=lens[:2 * G], advantages=[round(v, 4) for v in adv[:2 * G].tolist()])
            open(OUT / "groups.jsonl", "a").write(json.dumps(g) + "\n")
        if (step + 1) % 10 == 0 or step == start:
            print(rec, flush=True)
        if args.eval_every and (step + 1) % args.eval_every == 0:
            e = dict(step=step + 1, **evaluate(model))
            open(efile, "a").write(json.dumps(e) + "\n")
            print("eval", e, flush=True)
        if step + 1 in save_at:
            model.save_pretrained(str(OUT / ("adapter-step" + str(step + 1))))
        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            save_ckpt(model, opt, scaler, step + 1)
        if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes and step + 1 < args.steps:
            save_ckpt(model, opt, scaler, step + 1)
            print("time limit reached at step", step + 1, "; run the same command again to continue", flush=True)
            return
    model.save_pretrained(str(OUT / "adapter-final"))
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEV.type == "cuda" else None
    json.dump(dict(done=True, steps=args.steps, peakGB=peak, gpu=torch.cuda.get_device_name(0) if DEV.type == "cuda" else None,
                   sessionSec=round(time.time() - t_start, 1)), open(OUT / "done.json", "w"))
    print("DONE", args.steps, "peak GB", peak, flush=True)


if __name__ == "__main__":
    main()
