# Rubric: Build an Instruction-Following Assistant (lltCapstoneAssistant)

100 points over five criteria. The weights are the course plan's. Score each criterion yourself from 1 to 5 before you submit, then spend your last session on the lowest one.

| Criterion | Weight | Full marks look like |
|---|---|---|
| Data quality and documentation | 20 | A data card that a stranger could use to rebuild your SFT set and your preference pairs |
| Training correctness | 20 | SFT and DPO that provably train what you think they train, with curves that show it |
| Evaluation rigour | 25 | Every comparison has an interval, the judge is checked against human labels, length is controlled |
| Behaviour and safety analysis | 15 | Refusal and over-refusal measured before and after DPO, with the failure cases read and named |
| Report and release | 20 | A model card with your own limitations, a working Hub release and a report a reviewer can check |

## 1. Data quality and documentation (20)

- **5 points.** The data card lists every source with its licence and the share of the mix it takes, and says why each excluded source was excluded.
- **5 points.** Every filter is listed with the number of rows it removed (single-turn, language, length, identity, duplicates).
- **5 points.** Decontamination is run against every evaluation set you report, the method is stated (for example 13-word overlap) and the count removed is given.
- **5 points.** The preference pairs are described: prompt source, replies per prompt, labeller, the rule for chosen and rejected, the minimum score gap, how many prompts gave no pair and why, and how often the chosen reply was the longer one.

Weak version: "we used a public SFT mixture". It gives no counts, no licences and no decontamination.

## 2. Training correctness (20)

- **5 points.** The label mask is checked: prompt tokens carry -100, every reply ends with a labelled end-of-text token, and a test shows it.
- **5 points.** Packing is checked: a packed row gives the same loss as its conversations run one by one (the CPU test in `tests/test_cpu.py` does this).
- **5 points.** SFT reports held-out loss before and after, and the share of replies that end on their own.
- **5 points.** DPO reports held-out reward accuracy and margins before and after, and the chosen and rejected log-probabilities, so likelihood displacement is visible if it happened.

Weak version: a training loss that goes down, with nothing held out.

## 3. Evaluation rigour (25)

- **5 points.** IFEval prompt-level strict accuracy for base, SFT, DPO and the official instruct model, each with a 95% interval.
- **5 points.** Paired differences (DPO minus SFT, DPO minus official) with a paired bootstrap interval, and a claim only where the interval excludes zero.
- **5 points.** The judge is calibrated on human-labelled pairs: agreement with an interval, the share of pairs with the same verdict in both orders, and the share of verdicts that pick the first position.
- **5 points.** Win rates are judged in both orders and reported raw and length-controlled, with the mean reply length of each model.
- **5 points.** A small sample of your own model's comparisons is labelled by you before you read the judge's verdicts, and the agreement is reported with its interval, however low it is.

Weak version: one win rate from one order, no interval, and a judge nobody checked.

## 4. Behaviour and safety analysis (15)

- **5 points.** Refusal on unsafe prompts and over-refusal on safe prompts (XSTest) for SFT, DPO and the official model, with intervals, and a note that string matching gives lower bounds.
- **5 points.** At least five failure cases read and named (looping, ignoring a constraint, wrong refusal, length padding), each with the prompt id.
- **5 points.** One finding about what DPO changed in behaviour, stated with evidence, including when it made something worse.

Scope: use established benchmark prompts only. Never generate, store or publish harmful text. Keep only refusal labels for the unsafe prompts.

## 5. Report and release (20)

- **5 points.** The model card states the base model, the prompt format, the data, the training, every evaluation number with its interval, and limitations written from your own results.
- **5 points.** The release loads and generates from the Hub with a few lines of code, and the chat template produces the training format exactly.
- **5 points.** The report names the tier you used, the GPU-hours per stage, and which stages, if any, started from our checkpoints.
- **5 points.** Every number in the report traces to a file in your work folder.

## Submission checklist

- [ ] `tests/test_cpu.py` passes on your machine
- [ ] `data/card.json` and your written data card
- [ ] `records/` for every stage you ran, and the logs
- [ ] IFEval, XSTest and judge tables with intervals
- [ ] Your hand-labelled sheet and the agreement it gave
- [ ] Model card on the Hub, and the load-and-generate check
- [ ] The report, with GPU-hours per stage and the self-score per criterion
- [ ] The GPU is released. The cloud VM is stopped, or the notebook session has ended
