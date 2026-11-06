import gc, time
from transformers import TrainerCallback

RUN_EXPERIMENT = False      # True retrains on your T4; our runs are in the tables of steps 9 to 11

class EpochScore(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, **kw):
        loss = C.eval_loss(model, tok360, held, GPU, batch_size=1, dtype=torch.float16)["loss"]
        print("epoch", round(state.epoch), "| step", state.global_step, "| held-out loss", round(loss, 4))

def train_and_score(trainer, name, each_epoch=False):
    if each_epoch:
        trainer.add_callback(EpochScore())
    t0 = time.time()
    trainer.train()
    sec = time.time() - t0
    trainer.model.eval()
    loss = C.eval_loss(trainer.model, tok360, held, GPU, batch_size=1, dtype=torch.float16)["loss"]
    g = C.eval_generate(trainer.model, tok360, gen_set, GPU, dtype=torch.float16)
    logs = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(name, "| steps", trainer.state.global_step, "| seconds", round(sec), "| train loss", round(logs[0], 4), "->",
          round(logs[-1], 4), "| held-out loss", round(loss, 4), "| stops", round(g["stopRate"] * 60), "of 60")
    return {"name": name, "steps": trainer.state.global_step, "seconds": round(sec, 1), "trainLoss": [logs[0], logs[-1]],
            "heldOut": loss, "stopRate": g["stopRate"], "meanNewTokens": g["meanNewTokens"]}

def fresh_trainer(**changes):
    gc.collect()
    torch.cuda.empty_cache()                       # the previous run's weights and optimizer state go first
    return make_trainer(fresh_model(), **changes)

if RUN_EXPERIMENT:
    del trainer, model360                          # free step 8's model before the next one loads
    train_and_score(fresh_trainer(assistant_only_loss=False), "every token, seed 0")
