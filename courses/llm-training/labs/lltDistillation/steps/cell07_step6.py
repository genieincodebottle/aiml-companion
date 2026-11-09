run_arm("Logit KD, forward KL on teacher answers", "forward", "teacher")
for name in ["SFT on teacher answers", "Logit KD, forward KL on teacher answers"]:
    r = R[name]
    print(name, "| accuracy", r["acc"], "| loss at step 1", r["loss"][0], "| at step", len(r["loss"]), r["loss"][-1],
          "| training s", r["trainSec"], "| peak GB", r["peakGB"])