RUN_EXPERIMENT = False    # True trains the other three arms of the grid and a second seed of the first three on your T4

grid = [("SFT on reference answers", "sft", "reference", 0),
        ("Logit KD, reverse KL on teacher answers", "reverse", "teacher", 0),
        ("On-policy, forward KL on student samples", "forward", "student", 0),
        ("SFT on teacher answers, seed 1", "sft", "teacher", 1),
        ("Logit KD, forward KL on teacher answers, seed 1", "forward", "teacher", 1),
        ("On-policy, reverse KL on student samples, seed 1", "reverse", "student", 1)]
if RUN_EXPERIMENT:
    for name, kind, source, seed in grid:
        run_arm(name, kind, source, seed)

print("arm | accuracy | reverse KL on own answers | forward KL on teacher answers | student entropy on own answers")
for name, r in R.items():
    if name != "teacher":
        print(name, "|", r["acc"], "|", r["onOwnAnswers"]["reverseKL"], "|", r["onTeacherAnswers"]["forwardKL"], "|",
              r["onOwnAnswers"]["studentEntropy"])