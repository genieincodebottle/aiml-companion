// lltDistillation - Distillation for LLMs, off-policy, on-policy and logits (LLM Training Lab, module 20, premium).
//
// Lab page (type 'lab-lesson'), rendered by src/components/module/LabLesson.jsx.
// Blueprint: .claude/docs/LLM_TRAINING_PROGRAM_PLAN.md section 10 entry 20, 7.8.3 (open question: on-policy vs SFT on teacher
// outputs per training FLOP), section 12 (Hinton 2015, Kim and Rush 2016, Agarwal 2024, Gu 2024, Lu 2025). Paired with
// llm/llmDistillation (recap card). Class A: a narrated overview is built later from llt6/lltDistillation.OVERVIEW_SPEC.md,
// and a Try it widget from llt6/lltDistillation.TRYIT_SPEC.md (data llt6/lltDistillation.tryit_data.json). This draft sets
// neither overview nor lesson.tryIt.
//
// NUMBERS ARE A CONTRACT. Every figure traces to aiml-companion/courses/llm-training/runs/lltDistillation/ or an exact
// derivation. Code blocks are the files labs/lltDistillation/steps/cellNN_stepK.py (export_page_code.py builds them in),
// so the page, the notebook and the recorded runs share one source.
//   T1, laptop CPU (python 3.12.3, torch 2.15.0+cpu, trl 1.17.1, numpy 2.5.4):
//   lltDistillation-t1-cells      page cells 1 and 2: Gaussian fits (forward mean -0.4 sd 2.05, valley 0.1873; reverse
//                                 mean -2.0 sd 0.6, 0.51, other mode 0.915); own divergence == TRL kernel (gaps 2e-6)
//   T2, Kaggle T4 (python 3.13, torch 2.11.0+cu130, transformers 5.22.1, trl 1.17.1, peft 0.21.2, bitsandbytes 0.50.2),
//   kernel llt-distill-main v1 (234 min, 2026-11-08), the notebook with RUN_EXPERIMENT = True and the teacher writing its data:
//   lltDistillation-T2-baselines       teacher 0.756, student 0.522 on 500 GSM8K test questions; KL of the untrained student
//   lltDistillation-T2-teacher-data    2,048 answers, 197,439 tokens in 2,074 s (95 tokens/s), 0.817 correct
//   lltDistillation-T2-<arm>           nine trained arms (64 steps x 32 questions, LoRA r16), accuracy, KL, entropy, seconds
//   lltDistillation-T2-trl-distill     DistillationTrainer, 3 steps, 76.7 s, trainable params fp32
//   lltDistillation-T2-smoke-stopbug   kernel llt-distill-smoke v1: the broken stop string, rescored on the same replies
// Not yet run: the notebook with its learner defaults (USE_OUR_TEACHER_DATA = True, USE_OUR_ADAPTER = True); job packaged in
// courses/llm-training/gcp/llt-distill-notebook/, run on a Google Cloud T4 (Python 3.12.16, torch 2.11.0+cu130), 50.0 min:
//   lltDistillation-T2-notebook        learner defaults + USE_OUR_ADAPTER = True; adapter 0.532; same-seed SFT 0.490, logit KD 0.520
// Code strings: no backslash, no backtick, no dollar-brace and no non-ASCII character.

// @@CODE@@

export const lltDistillation = {
    type: 'lab-lesson',
    title: 'Distil a 4B Teacher Into a 0.6B Student, Three Ways',
    subtitle: 'Fit forward and reverse KL by hand and match TRL\'s distillation kernel. Then distil Qwen3-4B-Base into Qwen3-0.6B-Base on GSM8K on a T4, with SFT on teacher answers, logit KD and on-policy distillation, and measure what each one changed.',
    prerequisites: ['llmDistillation', 'lltLoRA'],
    objectives: [
        'Show with a two-mode example why forward KL spreads a student across the teacher\'s modes and reverse KL picks one',
        'Write the forward KL, reverse KL and generalized JSD over a full vocabulary, and check them against the kernel inside TRL\'s DistillationTrainer',
        'Distil Qwen3-4B-Base into Qwen3-0.6B-Base three ways on a T4, from teacher answers, from teacher logits and from the student\'s own samples',
        'Measure each student with accuracy, paired intervals and two seeds, plus its KL to the teacher and its entropy',
        'Tell a training-seed difference from a method difference before you report a winner',
    ],
    tldr: 'You fit one Gaussian to a two-mode teacher both ways, write the three divergences and match TRL\'s kernel within 3.1e-6. Then you distil Qwen3-4B-Base, 75.6% on 500 GSM8K questions, into Qwen3-0.6B-Base, 52.2%, with 64 LoRA steps of 32 questions per method. No accuracy difference from the untrained student excluded zero, and two seeds of SFT differed by 4.8 points, close to the largest gap between any two methods. What did move is the divergence each KL method trains on, and the student\'s entropy went up under forward KL and down under reverse KL, in the direction the two-mode example suggests.',
    recap: {
        trackId: 'llm',
        moduleId: 'llmDistillation',
        moduleTitle: 'Knowledge Distillation',
        points: [
            'Distillation trains a small student to copy a large teacher, from the teacher\'s full next-token distribution or from text the teacher wrote.',
            'Matching distributions needs a shared vocabulary. Training on the teacher\'s text does not.',
            'The gap between two distributions is measured with KL divergence, and the student is trained to shrink it.',
        ],
    },
    regime: {
        badge: 'DISTILLATION',
        startsFrom: 'Qwen3-0.6B-Base, frozen in fp16, with new LoRA adapters; teacher Qwen3-4B-Base in 4-bit NF4, frozen',
        changes: 'LoRA adapters only, rank 16 on every linear layer, 10,092,544 parameters',
        objective: 'Cross-entropy on teacher answers, forward or reverse KL to the teacher\'s full next-token distribution on teacher answers, or on the student\'s own samples',
    },
    lab: {
        defaultTier: 'T2',
        quickPath: {
            minutes: 44,
            reaches: 'In our Kaggle run, steps 1 to 3 took 16.7 min and steps 5 and 6 took 27 min on one T4, after the install and the first model download. Step 4 downloads the teacher\'s answers. You reach the teacher at 75.6%, the student at 52.2% and two distilled students with their KL to the teacher.',
        },
        noGpuPath: 'Steps 1 and 2 run on a laptop CPU, and the challenge runs in this page. For steps 3 to 9, read our recorded T4 runs. The page quotes their numbers, and every run file is in https://github.com/genieincodebottle/aiml-companion/tree/main/courses/llm-training/runs/lltDistillation',
        tiers: [
            {
                tier: 'T2',
                model: 'Teacher Qwen3-4B-Base in NF4, student Qwen3-0.6B-Base with LoRA rank 16, GSM8K, one T4 in fp16 autocast (measured on Kaggle)',
                peakMemGB: 9.66,
                wallClock: 'In our Kaggle run, steps 1 to 3 took 16.7 min, the teacher\'s 2,048 answers 34.6 min, step 5 10.8 min, step 6 16.2 min and step 7 31.6 min. With the defaults, which download the teacher\'s answers, and our on-policy adapter loaded in step 7, the whole notebook took 50.0 min. That run was on a Google Cloud T4 with the same torch build (lltDistillation-T2-notebook). The highest peak was 9.66 GB, in the on-policy forward-KL run of step 8. All these times leave out the install cell, and the Kaggle times also leave out the first download of about 9 GB of models',
                costUSD: 0,
                givesUp: 'Nothing on this page. The teacher\'s training answers and the on-policy adapter come from our run, so the defaults skip 34.6 min of teacher generation. Set USE_OUR_ADAPTER to skip the 25 min of on-policy training too.',
            },
            {
                tier: 'T1',
                model: 'Steps 1 and 2 on a laptop CPU, the two-mode fit in NumPy and the divergence check against TRL',
                peakMemGB: 0.6,
                wallClock: '0.4 min for steps 1 and 2 on our laptop CPU at 4 threads, with the packages already installed. Peak RAM 0.6 GB (lltDistillation-t1-timing-v2026-11)',
                costUSD: 0,
                givesUp: 'Both models and every training run, steps 3 to 9. Our recorded runs stand in for them.',
            },
        ],
        notebooks: {
            colab: 'https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/courses/llm-training/labs/lltDistillation/lltDistillation.ipynb',
        },
        env: 'T2, Kaggle image 2026-11-08, python 3.13 | torch 2.11.0+cu130 | transformers 5.22.1 | trl 1.17.1 | peft 0.21.2 | bitsandbytes 0.50.2 | accelerate 1.15.0 | datasets 5.1.0 | fp16 autocast. T1, python 3.12.3 | torch 2.15.0+cpu | trl 1.17.1 | numpy 2.5.4 | Pyodide 0.27 for the challenge. Recorded on Kaggle; Colab not run',
        checkedOn: '2026-11-08',
    },
    lesson: {
        build: [
            {
                heading: 'Fit one Gaussian to two modes, both ways',
                kind: 'step',
                blocks: [
                    {
                        text: 'Every code block on this page is a cell of the lab notebook, opened from the Lab card. Run the notebook\'s runtime check and pinned install first, then the cells in order. Steps 1 and 2 also run on a laptop CPU.',
                    },
                    {
                        text: 'A distilled student can never match its teacher exactly, so the loss has to decide which mismatch costs more. KL divergence, the expected log ratio of two distributions, is not symmetric. Forward KL, KL(teacher || student), averages over where the teacher puts mass. Reverse KL, KL(student || teacher), averages over where the student puts mass.',
                    },
                    {
                        formula: [
                            { label: 'Forward KL, averaged under the teacher p', tex: '\\mathrm{KL}(p \\,\\|\\, q) = \\sum_x p(x) \\log \\frac{p(x)}{q(x)}' },
                            { label: 'Reverse KL, averaged under the student q', tex: '\\mathrm{KL}(q \\,\\|\\, p) = \\sum_x q(x) \\log \\frac{q(x)}{p(x)}' },
                        ],
                    },
                    {
                        text: 'Make the mismatch visible. The teacher has two modes, 60% of its mass at -2 and 40% at +2, each with a standard deviation of 0.6. The student is one Gaussian. Search a grid of means and widths for the student that minimises each KL.',
                    },
                    { code: CELL1_CODE, title: 'Code cell 1, the best single Gaussian under forward and reverse KL' },
                    {
                        text: 'Forward KL charges the student wherever the teacher has mass the student misses, so the student widens until it covers both modes. It puts 18.7% of its mass in the valley between them, where the teacher has 0.6%. Reverse KL charges the student wherever it puts mass the teacher lacks, so it sits on one mode and ignores the other. In the Try it widget, switch to reverse KL, start the student on the lighter mode and see whether it leaves.',
                    },
                    {
                        depth: 'deep',
                        note: 'derivation',
                        text: 'Both results have closed forms. Forward KL to a Gaussian is minimised by matching the teacher\'s mean and variance. Here the mean is 0.6 x -2 + 0.4 x 2 = -0.4 and the variance is 0.36 + 4 - 0.16 = 4.2, a standard deviation of 2.05. A student sitting exactly on one well-separated mode has reverse KL equal to minus the log of that mode\'s weight, -ln 0.6 = 0.511 and -ln 0.4 = 0.916. The grid of code cell 1 prints them as 0.51 and 0.915. So the heavier mode wins, but the lighter one is still a local minimum. A student started there and moved only downhill stays there. For language models the same split reads token by token. Cross-entropy on text sampled from the teacher equals forward KL plus the teacher\'s entropy, in expectation. Greedy teacher text is not a sample, it is the teacher\'s top token at every position, so step 5 trains toward the mode.',
                    },
                ],
                expect: 'The forward-KL fit is mean -0.4 and standard deviation 2.05, with 0.1873 of its mass in the valley |x| < 0.5, against 0.006 for the teacher. The reverse-KL fit is mean -2.0 and standard deviation 0.6, with 0.006 in the valley and a reverse KL of 0.51. On the other mode, mean 2.0, the reverse KL is 0.915.',
                solution: 'Compute KL with a small floor inside the log, `np.log(a + 1e-300)`, because the Gaussians underflow to 0 far from their mean. The grid covers means from -4 to 4 and widths from 0.2 to 4 in steps of 0.05.',
            },
            {
                heading: 'Write the divergence and match TRL\'s kernel',
                kind: 'step',
                blocks: [
                    {
                        text: 'A language model gives a distribution over its whole vocabulary at every position. Distillation compares the student\'s and the teacher\'s distributions at each token of an answer and averages over the tokens. Write one function for three divergences. Beta 0 gives forward KL, beta 1 gives reverse KL, and values in between give the generalized Jensen-Shannon divergence, JSD, of the GKD paper. The JSD formula alone is 0 at both ends, so TRL and this function define the two endpoints as the KLs themselves.',
                    },
                    {
                        formula: [
                            { label: 'Generalized JSD, with teacher p, student q and mixture m', tex: '\\mathrm{JSD}_\\beta = \\beta\\, \\mathrm{KL}(p \\,\\|\\, m) + (1 - \\beta)\\, \\mathrm{KL}(q \\,\\|\\, m), \\quad m = \\beta p + (1 - \\beta) q' },
                        ],
                    },
                    {
                        text: 'TRL\'s `DistillationTrainer` computes the same quantity without ever building the full logits. It takes each model\'s last hidden states and output layer and works through the 151,936-row vocabulary in tiles. Compare its kernel with your function on random hidden states, values and gradients, for beta 0, 0.5 and 1.',
                    },
                    { code: CELL2_CODE, title: 'Code cell 2, three divergences against TRL\'s ChunkedDivergenceFunction' },
                    {
                        text: 'The student and the teacher here have different hidden sizes, 64 and 96, and only the vocabulary is shared. That is the one structural requirement of token-level distillation. TRL\'s trainer raises an error when the two vocabulary sizes differ.',
                    },
                ],
                expect: 'For beta 0, 0.5 and 1 your values and TRL\'s agree to five decimals, 0.851494, 0.176187 and 0.853843 on our laptop, where TRL printed 0.853842 for beta 1. The largest value gap is 3.1e-06 or less, and the largest gradient gap 1.2e-07 or less. On the T4 the gaps were 2.4e-06 and 4.1e-08.',
                solution: '`ChunkedDivergenceFunction` lives in `trl.kernels.chunked_divergence`, an internal module, so its import path can change between TRL releases. On a CPU it runs its PyTorch tiles, and on a GPU its Triton kernels. Pass `None` for the two bias arguments, since Qwen3 output layers have no bias.',
            },
            {
                heading: 'Load the teacher and the student, and measure both on GSM8K',
                kind: 'step',
                blocks: [
                    {
                        text: 'Move to a T4. The teacher is Qwen3-4B-Base and the student Qwen3-0.6B-Base, both put on the Hugging Face Hub by Qwen on 2025-04-28 under Apache-2.0. They share one tokenizer and a vocabulary of 151,936 rows. Load the teacher in 4-bit NF4, which keeps its weights at 2.65 GB on the GPU.',
                    },
                    {
                        text: 'Qwen3.5-4B-Base and Qwen3.5-0.8B-Base are newer, put on the Hub on 2026-02-27 and 28. Their vocabulary has 248,320 rows. In our probe, an fp16 Qwen3.5-4B teacher took 8.41 GB for its weights alone, and logit KD with it ran out of memory on a T4 at 15.1 GB (run llt-c-probe, kd-q35-t4b-fp16-s0.8b-lora). In NF4 the same pair fit at 11.06 GB. This lab uses the Qwen3 pair, whose logits are 0.6 times the size per token.',
                    },
                    {
                        text: 'The task is GSM8K, grade-school maths word problems with a single number as the answer, MIT-licensed. Use the first 500 test questions. Prompt each model with `Question:`, the question, a new line and `Answer:`, decode greedily for up to 256 tokens, and read the number after "The answer is".',
                    },
                    { code: CELL3_CODE, title: 'Code cell 3, load the pair, check the shared vocabulary, prepare GSM8K' },
                    {
                        text: 'A base model does not stop after its answer. It writes a new question and answers that too. Stop generation at a new line followed by `Question` or `[Question]`, cut the reply there, and score only what came before. Then score each model\'s answers with the teacher. For every answer token, compute the forward KL, the reverse KL and the student\'s entropy over the whole vocabulary, on the student\'s own answers and on the teacher\'s answers.',
                    },
                    { code: CELL4_CODE, title: 'Code cell 4, accuracy plus KL to the teacher on two sets of answers' },
                    {
                        table: {
                            caption: 'Before any training, first 500 GSM8K test questions, greedy decoding (lltDistillation-T2-baselines)',
                            head: ['', 'Teacher, Qwen3-4B-Base NF4', 'Student, Qwen3-0.6B-Base'],
                            rows: [
                                ['Accuracy', '0.756', '0.522'],
                                ['Replies that stopped at the stop string', '0.87', '0.86'],
                                ['Mean answer tokens', '95.5', '103.0'],
                                ['Generation time, 500 questions', '497 s, batch 64', '191 s, batch 100'],
                                ['Reverse KL on the student\'s own answers, nats per token', '', '0.1966'],
                                ['Forward KL on the teacher\'s answers, nats per token', '', '0.1951'],
                                ['Student entropy on its own answers, nats per token', '', '0.4479'],
                            ],
                        },
                    },
                    {
                        text: 'The teacher is 23.4 points ahead, with a paired 95% interval of 18.8 to 28.2 points. It answers 145 questions the student misses, and the student answers 28 the teacher misses. On the first test question the teacher reaches 18 and the student 55, because the student ignores the eggs that are eaten or baked and then adds its own subtotals.',
                    },
                ],
                expect: 'The teacher takes 2.65 GB. Both vocabulary sizes are 151,936, the tokenizer has 151,669 entries and the token ids are the same. The longest prompt is 198 tokens. The teacher scores 0.756 and the student 0.522, with the KL numbers in the table.',
                solution: 'Pass `device_map={"": 0}` when you load the NF4 teacher, and do not call `.to()` on it afterwards. `logits_at` projects only the positions that predict an answer token, so the fp32 logits stay at tokens x 151,936 instead of rows x length x 151,936.',
                troubleshoot: [
                    {
                        error: 'Accuracy near 0.2 for the teacher, and almost every reply is 256 tokens long',
                        cause: 'The stop string misses how the model starts its next question. Both models continue with a `[Question]` block, so replies run to the token cap and the last number belongs to an invented question.',
                        fix: 'Stop at both forms, as in this cell, and read the number after "The answer is". Check a few full replies before you trust a score.',
                    },
                ],
            },
            {
                heading: 'Let the teacher write the training answers',
                kind: 'step',
                blocks: [
                    {
                        text: 'Off-policy distillation trains on text the teacher wrote once, before training starts. Kim and Rush 2016 called this sequence-level knowledge distillation. Code cell 3, in step 3, already shuffled the GSM8K training split with seed 0 and took 2,048 questions. Let the teacher answer each one greedily.',
                    },
                    {
                        text: 'Teacher generation was the slowest part of our run. The NF4 teacher wrote 197,439 tokens in 2,074 s at batch 64, 95 tokens per second, so 34.6 min (lltDistillation-T2-teacher-data). The default downloads those answers instead. Keep the human reference solutions too, ending with the same stop string, for a comparison in step 8.',
                    },
                    { code: CELL5_CODE, title: 'Code cell 5, the teacher\'s 2,048 training answers' },
                    {
                        text: 'The teacher is wrong on 18.3% of these questions, and the answers are kept as written. Filtering to correct answers is a common next step, and this lab leaves it out so that every method sees the same 2,048 answers.',
                    },
                ],
                expect: 'The cell prints 2,048 answers, 0.817 correct, 0.893 ending at the stop string and 96.4 tokens on average. The reference solutions average 125.7 tokens. With `USE_OUR_TEACHER_DATA = False` it also prints the token count, the seconds and the tokens per second.',
                solution: 'The download checks that its 2,048 questions equal yours in order. If the assertion fails, keep `seed=0` and `N_TRAIN = 2048` in code cell 3, in step 3. The file holds token ids, not text, so the student trains on exactly the tokens the teacher produced.',
            },
            {
                heading: 'Off-policy, SFT on the teacher\'s answers',
                kind: 'step',
                blocks: [
                    {
                        text: 'Train the student on the teacher\'s answers with plain cross-entropy, as in lltSFT. Freeze the student in fp16 and add LoRA adapters of rank 16 to every linear layer, held in fp32. Every method on this page then uses the same 64 steps of 32 questions, learning rate 2e-4, 3 warmup steps and cosine decay.',
                    },
                    {
                        text: 'Write one training function for all methods. `source` says who wrote the answers, the reference, the teacher or the student itself. `kind` picks the loss, cross-entropy, forward KL or reverse KL. `sync` writes base plus LoRA weights into the plain fp16 student from step 3, which then samples and evaluates without adapter layers.',
                    },
                    { code: CELL6_CODE, title: 'Code cell 6, one training function, then SFT on the teacher\'s answers' },
                ],
                expect: 'The loss goes 0.3922 at step 1, 0.2798 at step 17 and 0.2914 at step 64, in 248 s, peaking at 7.03 GB. The student scores 0.48, stops 0.932 of its replies and writes 99.2 tokens on average. Its reverse KL on its own answers is 0.1739 and its entropy 0.2784.',
                solution: 'Run step 3 first, since the loop uses `pack`, `logits_at`, `generate` and `evaluate`. The loss is summed over answer tokens and divided by the tokens in the whole batch of 32, so micro-batches of 4 add up to one step. Expect a few questions of difference even with the same seed, since fp16 kernels on a GPU are not bit-exact between runs. Our rerun of this cell on another T4 with the same torch build scored 0.490 against 0.480, with 5 of 500 answers changed. Logit KD scored 0.520 against 0.518, with 7 changed (lltDistillation-T2-notebook).',
                troubleshoot: [
                    {
                        error: 'torch.OutOfMemoryError: CUDA out of memory',
                        cause: 'The fp32 logits of 4 answers over 151,936 rows, for the student and the teacher, plus a previous model still on the GPU.',
                        fix: 'Lower `MICRO` to 2. `run_arm` deletes each trained model before the next one starts. If you trained by hand, `del` the model and call `torch.cuda.empty_cache()`.',
                    },
                    {
                        error: 'ValueError: Attempting to unscale FP16 gradients.',
                        cause: 'The LoRA weights were created in fp16 with the base, and GradScaler cannot unscale fp16 gradients.',
                        fix: 'Cast every trainable parameter to fp32 after `get_peft_model`, as `new_student` does.',
                    },
                ],
            },
            {
                heading: 'Logit KD, forward KL over the whole vocabulary',
                kind: 'step',
                blocks: [
                    {
                        text: 'Cross-entropy on a teacher answer uses one token per position, the one the teacher wrote. Logit KD uses the teacher\'s whole next-token distribution at that position. Train on the same answers with forward KL in place of cross-entropy, which adds one teacher forward pass per micro-batch.',
                    },
                    { code: CELL7_CODE, title: 'Code cell 7, logit KD with forward KL on the teacher\'s answers' },
                    {
                        table: {
                            caption: 'Two off-policy methods on the same 2,048 teacher answers, seeds 0 and 1 (lltDistillation-T2-sft-on-teacher-answers, -logit-kd-forward-kl-on-teacher-answers, and their -seed-1 runs)',
                            head: ['', 'SFT on teacher answers', 'Logit KD, forward KL'],
                            rows: [
                                ['Accuracy, seed 0 and seed 1', '0.480, 0.528', '0.518, 0.528'],
                                ['Forward KL on the teacher\'s answers', '0.2237, 0.2212', '0.1572, 0.1579'],
                                ['Student entropy on its own answers', '0.2784, 0.2886', '0.4966, 0.5187'],
                                ['Training time', '248 s, 246 s', '573 s, 576 s'],
                                ['Peak GPU memory, seed 0', '7.03 GB', '8.32 GB'],
                            ],
                        },
                    },
                    {
                        text: 'Before training, the forward KL on the teacher\'s answers was 0.1951 and the entropy 0.4479. Logit KD lowered that KL to 0.157 in both seeds, the lowest of any method on this page. SFT on the same answers raised it to 0.22 and cut the entropy by more than a third. The teacher\'s answers were greedy, so each target is the teacher\'s top token only, and the student learns to be as sure as that.',
                    },
                    {
                        text: 'The loss values of the two methods are not comparable. Cross-entropy includes the entropy of the target, and the KL does not, so a KL loss sits lower from step 1.',
                    },
                ],
                expect: 'The loss goes 0.2631 at step 1, 0.1723 at step 17 and 0.1893 at step 64, in 573 s at 8.32 GB. The student scores 0.518, with a forward KL on the teacher\'s answers of 0.1572 and an entropy of 0.4966. The comparison lines print 0.48 for SFT and 0.518 for logit KD.',
                solution: 'Each micro-batch runs the teacher once under `torch.no_grad()`, so a step takes about 2.3 times as long as SFT, 573 s against 248 s for 64 steps. The teacher\'s logits never need storing, since the teacher is on the same GPU.',
            },
            {
                heading: 'On-policy, the student samples and the teacher grades every token',
                kind: 'step',
                blocks: [
                    {
                        text: 'In steps 5 and 6 the student learns on prefixes the teacher wrote. At test time it continues its own prefixes, including its own mistakes, which it never trained on. On-policy distillation removes that mismatch. At every step the student samples its own answer to each question, at temperature 1, and is trained to lower the reverse KL to the teacher at every token of that answer.',
                    },
                    {
                        text: 'Reverse KL on the student\'s own samples only asks the teacher to grade tokens the student produces. The student is not pushed to spread mass over everything the teacher might say. Agarwal et al. 2024 call training on the student\'s own samples generalized knowledge distillation, GKD, and leave the divergence as a setting. Gu et al. 2024 trained MiniLLM with reverse KL on student samples and a policy-gradient estimator.',
                    },
                    { code: CELL8_CODE, title: 'Code cell 8, on-policy distillation with reverse KL' },
                    {
                        text: 'TRL\'s stable `DistillationTrainer` runs this loop for you. It samples from the student and applies the step 2 divergence, with beta 1.0 as the default. Run 3 steps of 8 questions to see the call. Pass dtypes in its config as strings, since `torch.float32` raises a TypeError there.',
                    },
                    { code: CELL9_CODE, title: 'Code cell 9, the same objective in TRL DistillationTrainer, 3 steps' },
                    {
                        depth: 'deep',
                        note: 'runNote',
                        text: 'Sampling dominates the step. Our on-policy run took 1,496 s, 894.5 s of it sampling 64 x 32 answers, so 23.4 s per step against 3.9 s for SFT. We sample from a plain fp16 copy of the student without adapter layers, refreshed with `sync` before each step, at a cost of 1.2 GB. The forward-KL variant on student samples ran longer, 1,763 s. Its batches grew to 4,400 to 5,300 answer tokens after step 1, against 3,100 to 3,900 for reverse KL. Off-policy methods instead pay once for the teacher\'s answers, 2,074 s here, shared by every method that uses them (lltDistillation-T2-on-policy-reverse-kl-on-student-samples, -teacher-data).',
                    },
                ],
                expect: 'The loss goes 0.357 at step 1, 0.2727 at step 17 and 0.2409 at step 64, after 1,496 s of which 894.5 s were sampling. The student scores 0.532, and its reverse KL on its own answers is 0.1503, the lowest of any method. TRL prints float32 trainable weights and three losses, 0.5307, 0.46 and 0.423, in 76.7 s.',
                solution: 'Set `USE_OUR_ADAPTER = True` to load our adapter from the Hub and skip the 25 min of training. In our rerun that path scored 0.532, with no question scored differently from the trained run. The TRL losses are not comparable with yours, since TRL samples other questions, up to 256 tokens each with no stop string.',
                troubleshoot: [
                    {
                        error: 'TypeError: attribute name must be string, not \'torch.dtype\'',
                        cause: '`model_init_kwargs` or `teacher_model_init_kwargs` in DistillationConfig got `dtype=torch.float32`.',
                        fix: 'Pass the string, `{"dtype": "float32"}`.',
                    },
                    {
                        error: 'ValueError: The student model has vocab_size ... but the teacher model has vocab_size ...',
                        cause: 'DistillationTrainer compares full next-token distributions, so the two vocabularies must be the same size.',
                        fix: 'Use a teacher from the same tokenizer family. For different tokenizers TRL has GOLDTrainer, which is experimental and may change in any release.',
                    },
                ],
                checkpoint: { runId: 'lltDistillation-T2-on-policy-reverse-kl-on-student-samples', step: 64 },
            },
            {
                heading: 'Separate the source of the answers from the KL direction',
                kind: 'experiment',
                predict: 'The two-mode fit of step 1 says forward KL spreads a student and reverse KL narrows it. Train both directions on teacher answers and on student samples. Which way do you expect the student\'s entropy to move in each?',
                blocks: [
                    {
                        text: 'Steps 6 and 7 changed two things at once, who wrote the answers and which KL is used. Fill in the other two cells of that grid, reverse KL on teacher answers and forward KL on student samples. Add SFT on the human reference solutions, and a second seed of the three methods from steps 5 to 7.',
                    },
                    { code: CELL10_CODE, title: 'Code cell 10, the rest of the grid and a second seed' },
                    {
                        table: {
                            caption: 'Every method, 64 steps of 32 questions, the same 500 test questions; KL and entropy in nats per token (lltDistillation-T2-baselines and the nine method runs)',
                            head: ['Method', 'Accuracy', 'Reverse KL, own answers', 'Forward KL, teacher answers', 'Entropy, own answers', 'Training s'],
                            rows: [
                                ['Before training', '0.522', '0.1966', '0.1951', '0.4479', ''],
                                ['SFT on reference answers', '0.486', '0.4196', '0.4812', '0.4112', '274'],
                                ['SFT on teacher answers, 2 seeds', '0.480, 0.528', '0.1739, 0.1735', '0.2237, 0.2212', '0.2784, 0.2886', '248, 246'],
                                ['Logit KD, forward, teacher answers, 2 seeds', '0.518, 0.528', '0.1709, 0.1786', '0.1572, 0.1579', '0.4966, 0.5187', '573, 576'],
                                ['Logit KD, reverse, teacher answers', '0.534', '0.1711', '0.2815', '0.2460', '575'],
                                ['On-policy, forward, student samples', '0.502', '0.2042', '0.1725', '0.6297', '1,763'],
                                ['On-policy, reverse, student samples, 2 seeds', '0.532, 0.514', '0.1503, 0.1548', '0.2025, 0.2054', '0.3623, 0.3493', '1,496, 1,534'],
                            ],
                        },
                    },
                    {
                        text: 'The direction set the entropy. Both forward-KL methods raised it, to 0.50 to 0.52 on teacher answers and 0.63 on student samples. Both reverse-KL methods lowered it, to 0.25 and 0.35 to 0.36. That is the direction of the two-mode picture of step 1, read token by token: forward KL keeps probability on every token the teacher considers, and reverse KL concentrates it. SFT on the teacher\'s greedy answers lowered the entropy too, as step 6 showed.',
                    },
                    {
                        text: 'Each of the four KL methods gave the lowest value of any method for the divergence it trains on, measured on the kind of answers it trains on. On-policy reverse KL gave the lowest reverse KL on the student\'s own answers, 0.150 and 0.155. Logit KD with forward KL gave the lowest forward KL on the teacher\'s answers, 0.157 and 0.158. The other two cells are in their run files, 0.182 and 0.153. SFT on teacher answers lowered the reverse KL on its own answers less, to 0.174, and raised the forward KL on the teacher\'s answers, to 0.224 and 0.221. On the human solutions it moved the student away from the teacher on both measures, 0.42 and 0.48.',
                    },
                    {
                        text: 'These KL and entropy values are means over about 50,000 answer tokens per method, without intervals. Where we ran two seeds they agree within 0.022 for entropy and within 0.008 for the KLs. The gaps between methods are 0.14 to 0.38 for entropy but only 0.015 to 0.02 for the lowest-KL claims, so read those orderings as likely, not settled. Three of the cells have one seed.',
                    },
                ],
                expect: 'With `RUN_EXPERIMENT = True` the cell trains six more students, 123 min on a T4 in our run, and prints the table above. With the default `False` it prints the table for the students you trained in steps 5 to 7.',
                solution: 'The six runs go one after another, and `run_arm` frees the GPU between them. Our highest peak was 9.66 GB, in the on-policy forward-KL run. Each run saves its adapter under `adapters/`, so a dropped session loses only the run in progress.',
            },
            {
                heading: 'Compare the methods with paired intervals and two seeds',
                kind: 'step',
                blocks: [
                    {
                        text: 'Every method was scored on the same 500 questions, so compare them question by question. The paired bootstrap resamples the 500 questions 10,000 times and recomputes the difference in accuracy each time. It shows how much the choice of test questions moves the difference, and nothing about the choice of training seed.',
                    },
                    { code: CELL11_CODE, title: 'Code cell 11, accuracy differences with 95% paired intervals' },
                    {
                        table: {
                            caption: 'Accuracy minus the untrained student\'s 0.522, with a 95% paired bootstrap interval over the 500 questions. Cell 11 prints the seed columns; the mean-of-seeds column averages the two correctness lists per question and uses the same bootstrap (lltDistillation-T2 method runs)',
                            head: ['Method', 'Seed 0', 'Seed 1', 'Mean of 2 seeds'],
                            rows: [
                                ['SFT on teacher answers', '-4.2 (-8.6 to 0.0)', '+0.6 (-3.8 to +4.8)', '-1.8 (-5.9 to +2.1)'],
                                ['Logit KD, forward KL', '-0.4 (-4.6 to +3.8)', '+0.6 (-3.8 to +4.8)', '+0.1 (-3.8 to +4.0)'],
                                ['On-policy, reverse KL', '+1.0 (-3.4 to +5.4)', '-0.8 (-5.2 to +3.6)', '+0.1 (-4.0 to +4.2)'],
                                ['Logit KD, reverse KL', '+1.2 (-3.2 to +5.6)', '', ''],
                                ['On-policy, forward KL', '-2.0 (-6.2 to +2.2)', '', ''],
                                ['SFT on reference answers', '-3.6 (-8.0 to +0.8)', '', ''],
                            ],
                        },
                    },
                    {
                        text: 'No interval in the table excludes zero, and SFT seed 0\'s ends exactly at 0.0. The seed gaps are 4.8 points for SFT on teacher answers, 1.0 for logit KD and 1.8 for on-policy reverse KL. For SFT, seed 1 minus seed 0 is +4.8, with a paired interval of 1.4 to 8.4, which excludes zero. Those two students answer different questions correctly. The 4.8-point gap is close to the largest gap between single runs of different methods, 5.4 points.',
                    },
                    {
                        text: 'That gap decides what you may claim. Against seed 0 of SFT, on-policy reverse KL is 5.2 points ahead, interval 1.4 to 9.2. Against seed 1 it is 0.4 ahead, interval -3.4 to +4.0. In our rerun of the default notebook, against a fresh SFT run with the same seed, it read +4.2, interval 0.4 to 8.0. With two seeds per method we cannot say how large a typical seed gap is. At this budget, 2,048 questions and 64 LoRA steps, we cannot rank the methods by accuracy.',
                    },
                    {
                        text: 'We did not test why accuracy stayed flat. The untrained student already writes the answer format, the teacher\'s training answers are 18.3% wrong, and 64 steps of rank-16 LoRA is a small update. Any of these could matter.',
                    },
                ],
                expect: 'The cell prints the teacher at 0.756 and the untrained student at 0.522. Then each method gets one line, with its accuracy, its difference from the untrained student and from SFT on teacher answers, and its training seconds. The intervals match the table.',
                solution: 'The bootstrap uses `np.random.default_rng(0)`, so the intervals match ours exactly for the same correctness lists. Run two seeds of any method you want to rank, and compare the seed gap with the gap between methods.',
            },
        ],
        keyPoints: [
            'Forward KL averages under the teacher and spreads the student over every mode. Reverse KL averages under the student and lets it settle on one. Our students\' entropy moved in that direction in all four KL methods.',
            'Off-policy distillation trains on text the teacher wrote. On-policy distillation samples from the student at every step and asks the teacher to grade each token, so it trains on the prefixes the student meets at test time.',
            'Token-level distillation needs a shared vocabulary, and its memory grows with the vocabulary. Project only the answer positions, and use an NF4 teacher on a T4.',
            'Each KL method gave the lowest value of the divergence it trains on. SFT on teacher answers lowered the reverse KL on its own answers less and raised the forward KL on the teacher\'s answers. No GSM8K accuracy difference from the untrained student excluded zero at 2,048 questions and 64 LoRA steps.',
            'Two seeds of SFT differed by 4.8 points, about as much as the largest gap between any two single runs, 5.4. The other two methods\' seeds differed by 1.0 and 1.8. Run several seeds before you name a winner.',
        ],
    },
    results: {
        runs: [
            'lltDistillation-t1-cells',
            'lltDistillation-T2-baselines', 'lltDistillation-T2-teacher-data',
            'lltDistillation-T2-sft-on-teacher-answers', 'lltDistillation-T2-sft-on-teacher-answers-seed-1',
            'lltDistillation-T2-logit-kd-forward-kl-on-teacher-answers', 'lltDistillation-T2-logit-kd-forward-kl-on-teacher-answers-seed-1',
            'lltDistillation-T2-on-policy-reverse-kl-on-student-samples', 'lltDistillation-T2-on-policy-reverse-kl-on-student-samples-seed-1',
            'lltDistillation-T2-sft-on-reference-answers', 'lltDistillation-T2-logit-kd-reverse-kl-on-teacher-answers',
            'lltDistillation-T2-on-policy-forward-kl-on-student-samples',
            'lltDistillation-T2-trl-distill', 'lltDistillation-T2-smoke-stopbug',
            'lltDistillation-T2-notebook', 'lltDistillation-t1-timing-v2026-11',
        ],
        broken: [
            {
                runId: 'lltDistillation-T2-smoke-stopbug',
                symptom: 'In a 32-question check, the teacher scores 0.219 on GSM8K and the untrained student 0.125. Only 9.4% of the teacher\'s replies end at the stop string, and they average 241.7 tokens.',
                cause: 'The only stop string was a new line plus `Question:`. Both base models start their next question with `[Question]`, so replies ran to the 256-token cap, and the last number came from an invented question. Rescored with the shipped stops on the same replies, the teacher is at 0.8125.',
                fix: 'Stop at every form the model uses to start a new question, cut the reply there, and read the number after "The answer is". Read several full replies before you trust any score.',
                distractors: [
                    'The NF4 teacher lost its maths ability to 4-bit rounding, so it needs fp16 weights.',
                    'Greedy decoding repeats itself on maths problems, so the teacher needs sampling at temperature 1.',
                ],
            },
            {
                runId: 'lltDistillation-T2-sft-on-teacher-answers',
                symptom: 'On-policy distillation scores 0.532 and SFT on teacher answers 0.480 on the same 500 questions. The paired interval for the 5.2-point gap is 1.4 to 9.2, so it looks like a clear win.',
                cause: 'The interval resamples test questions, not training seeds. A second SFT seed scored 0.528, and against it on-policy is 0.4 ahead with an interval of -3.4 to +4.0. For SFT, seed 1 minus seed 0 is +4.8 points, interval 1.4 to 8.4.',
                fix: 'Train at least two seeds of each method you compare, report the seed gap next to the method gap, and call a gap a tie unless it exceeds the spread you measured across seeds.',
                distractors: [
                    'The SFT student saw fewer questions than the on-policy student, so it was undertrained.',
                    'The paired bootstrap is biased upward on 500 questions, so the interval is too narrow.',
                ],
            },
            {
                runId: 'lltDistillation-T2-sft-on-teacher-answers-seed-1',
                symptom: 'After SFT on the teacher\'s answers, the student\'s entropy on its own answers fell from 0.448 to 0.289 nats per token, and its forward KL on the teacher\'s answers rose from 0.195 to 0.221.',
                cause: 'The teacher wrote its answers greedily, so every target token is the teacher\'s top token. Cross-entropy on them pushes the student toward the teacher\'s mode and away from the rest of its distribution.',
                fix: 'Use the teacher\'s full distribution on the same answers. Logit KD with forward KL kept the entropy at 0.50 to 0.52 and lowered that forward KL to 0.157. Sampling the teacher\'s answers at temperature 1 is another option, untested here.',
                distractors: [
                    'The NF4 teacher\'s answers carry random tokens from 4-bit rounding, so the student learned noise.',
                    'The LoRA weights were held in fp16, so the small updates rounded away.',
                ],
            },
        ],
    },
    challenges: [
        {
            prompt: 'Write `log_softmax`, `divergence` and `token_reward` in NumPy. `divergence` is the step 2 function for one row of logits per position. `token_reward` is the signal Thinking Machines\' on-policy distillation uses per sampled token, the teacher\'s log-probability minus the student\'s. Run the cell, and it prints All checks passed when all three are right. The checks include logits of 1,000, symmetry of the JSD at beta 0.5, and a Monte Carlo test that the mean reward over student samples is minus the reverse KL.',
            starter: CHALLENGE_STARTER,
            hints: [
                'Subtract the row maximum before `np.exp`, then subtract the log of the summed exponentials.',
                'For 0 < beta < 1, compute log m with `np.logaddexp(np.log(beta) + log_pt, np.log(1 - beta) + log_ps)`, so nothing is exponentiated twice.',
                'The expected value of log p_t(x) - log p_s(x) for x drawn from the student is the sum of p_s (log p_t - log p_s), which is minus KL(student || teacher).',
            ],
            solution: CHALLENGE_SOLUTION,
        },
    ],
    papers: [
        {
            cite: 'Hinton, Vinyals and Dean 2015, Distilling the Knowledge in a Neural Network',
            url: 'https://arxiv.org/abs/1503.02531',
            problem: 'Ensembles and large models predict well but are too slow to deploy to many users.',
            contribution: 'Train a small model on the soft targets of a large one, its output probabilities softened with a temperature, so the small model also learns how the large one ranks the wrong answers. Results on MNIST and a commercial speech system.',
            reproduce: 'Step 6 trains the student on the teacher\'s full next-token distribution at temperature 1, with forward KL, which differs from cross-entropy on soft targets only by the teacher\'s entropy, a constant.',
            ourResult: 'Logit KD lowered the forward KL on the teacher\'s answers from 0.195 to 0.157 in both seeds and kept the student\'s entropy at 0.50 to 0.52. GSM8K accuracy was 0.518 and 0.528 against 0.522 before training, inside the interval (lltDistillation-T2-logit-kd-forward-kl-on-teacher-answers and -seed-1).',
        },
        {
            cite: 'Kim and Rush 2016, Sequence-Level Knowledge Distillation',
            url: 'https://arxiv.org/abs/1606.07947',
            problem: 'Word-level distillation matches the teacher one token at a time, but a translation system is used one sequence at a time.',
            contribution: 'Train the student on the teacher\'s own beam-search outputs. On machine translation the best student ran 10 times faster than its teacher with little loss, and sequence-level training seemed to remove the need for beam search.',
            reproduce: 'Steps 4 and 5 train the student on the teacher\'s greedy answers to 2,048 GSM8K questions.',
            ourResult: 'At our budget, SFT on the teacher\'s answers scored 0.480 and 0.528 on two seeds, against 0.522 before training. It also cut the student\'s entropy from 0.448 to 0.28 and 0.29, since every target was the teacher\'s top token (lltDistillation-T2-sft-on-teacher-answers and -seed-1).',
        },
        {
            cite: 'Agarwal et al. 2024, On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (ICLR 2024)',
            url: 'https://arxiv.org/abs/2306.13649',
            problem: 'A student trained on fixed teacher outputs meets different sequences at inference, its own, and its errors are never corrected.',
            contribution: 'Generalized knowledge distillation, GKD, trains the student on its own sampled sequences with feedback from the teacher, and lets the divergence be chosen, forward KL, reverse KL or a generalized JSD. Evaluated on summarisation, translation and arithmetic reasoning.',
            reproduce: 'Step 2 writes the generalized JSD and matches TRL\'s kernel. Steps 7 and 8 run the on-policy and off-policy cells of the GKD grid with both KL directions.',
            ourResult: 'On student samples, reverse KL gave the lowest reverse KL on the student\'s own answers, 0.150 and 0.155, against 0.197 before training. Accuracy was 0.532 and 0.514 against 0.522, inside the interval. Forward KL on student samples raised the entropy to 0.63 (lltDistillation-T2-on-policy runs).',
        },
        {
            cite: 'Gu, Dong, Wei and Huang 2024, MiniLLM: On-Policy Distillation of Large Language Models (ICLR 2024)',
            url: 'https://arxiv.org/abs/2306.08543',
            problem: 'Forward KL makes a small generative student put probability on regions where the large teacher has little.',
            contribution: 'Replace forward KL with reverse KL, which keeps the student from overestimating the teacher\'s low-probability regions, and optimise it on the student\'s own samples with a policy-gradient method. Tested from 120M to 13B parameters on instruction following.',
            reproduce: 'Step 1 shows the low-probability valley in one dimension. Step 7 trains with reverse KL on student samples, using the full-vocabulary gradient instead of a policy-gradient estimator.',
            ourResult: 'Reverse-KL students had the lowest entropy of the KL methods, 0.25 on teacher answers and 0.35 to 0.36 on their own samples, against 0.50 to 0.63 for forward KL. We measured GSM8K accuracy, not instruction-following quality, and no accuracy difference from the untrained student excluded zero (lltDistillation-T2 method runs).',
        },
        {
            cite: 'Lu and Thinking Machines Lab 2025, On-Policy Distillation (blog post, 2025-10-27)',
            url: 'https://thinkingmachines.ai/blog/on-policy-distillation/',
            problem: 'Reinforcement learning gives a sparse reward per answer and off-policy distillation trains on states the student never reaches.',
            contribution: 'The student samples, a stronger teacher grades each token, and the per-token advantage is the negative reverse KL with a discount factor of zero. Starting from Qwen3-8B-Base with LoRA rank 128, the post reports learning the RL-trained policy in about 7-10x fewer gradient steps than RL, a compute efficiency of 50-100x.',
            reproduce: 'Step 7 trains on the student\'s samples with per-token reverse KL. The challenge writes the per-token reward and checks that its mean is minus the reverse KL.',
            ourResult: 'We did not run RL, so this page does not test the 7-10x claim. Our on-policy step took 23.4 s against 3.9 s for SFT, and needed no teacher generation, which cost 2,074 s for the off-policy methods (lltDistillation-T2-on-policy-reverse-kl-on-student-samples, -teacher-data).',
        },
    ],
    extension: {
        task: 'Find the budget at which the methods separate. Train SFT on teacher answers and on-policy reverse KL at 64, 128 and 256 steps of 32 questions, each with two seeds. Reuse the seed-0 shuffle of code cell 3, in step 3, and repeat the 2,048 questions after the first epoch. Score every student on the same 500 test questions.',
        deliverable: 'A table with one row per method and budget. Give the accuracy of each seed, the mean, the paired interval against the untrained student and the training seconds, including the teacher\'s 2,074 s for SFT. Add two sentences on whether the methods separate beyond the seed spread, and at what cost.',
        rubric: 'Both methods use the same questions, learning rate and schedule at each budget, and every student is scored on the same 500 questions. A gap is called a tie unless it exceeds the spread measured across seeds, and the answer says how many seeds that spread comes from. The training seconds include sampling for on-policy and the teacher\'s generation for SFT.',
    },
    openQuestion: {
        question: 'Per training FLOP, does on-policy distillation beat SFT on teacher outputs for a 0.6B student on GSM8K?',
        known: 'Agarwal et al. 2024 report that GKD, trained on the student\'s own samples, is effective for distillation on summarisation, translation and arithmetic reasoning. The Qwen3 Technical Report (Yang et al. 2025) distils Qwen3-8B off-policy first. From there, on-policy distillation reached 74.4 on AIME\'24 with 1,800 GPU hours, against 67.6 and 17,920 GPU hours for reinforcement learning. Lu and Thinking Machines Lab 2025 report 7-10x fewer gradient steps than RL. None of these compares against SFT on teacher outputs at equal compute for a student this small.',
        knownAsOf: '2026-11',
        protocol: 'Train Qwen3-0.6B-Base with LoRA from Qwen3-4B-Base by SFT on teacher answers and by on-policy reverse KL at three budgets, counting T4 seconds including teacher generation and student sampling. Use 2 seeds each and the first 500 GSM8K test questions, and report accuracy with paired intervals and the seed spread.',
        tier: 'T2',
        ourResult: 'One budget measured, two seeds each. At 64 steps of 32 questions, SFT on teacher answers scored 0.480 and 0.528 in 248 s and 246 s, plus 2,074 s of teacher generation. On-policy reverse KL scored 0.532 and 0.514 in 1,496 s and 1,534 s. The untrained student scored 0.522, and no difference from it excluded zero (lltDistillation-T2 method runs, -teacher-data).',
    },
    quiz: [
        {
            q: 'A teacher has two well-separated modes. You fit a single Gaussian student by minimising forward KL, KL(teacher || student). Where does the student end up?',
            o: [
                'On the heavier mode, narrow, ignoring the lighter one',
                'Between the modes, wide enough to cover both of them',
                'On the lighter mode, since it is the harder one to fit',
                'At the heavier mode\'s mean, but as wide as both modes together',
            ],
            a: 1,
            e: 'Forward KL is averaged under the teacher, so it charges heavily wherever the teacher has mass the student misses. The best Gaussian matches the teacher\'s mean and variance, a mean of -0.4 and a standard deviation of 2.05 in step 1. It puts 18.7% of its mass in the valley where the teacher has 0.6%.',
        },
        {
            q: 'In step 1 the reverse-KL student sits on the mode holding 60% of the teacher\'s mass. Its reverse KL is 0.51. Where does that number come from?',
            o: [
                'Minus the log of 0.6, the weight of that mode',
                'Log 2, the cost of ignoring one of two modes',
                'The 0.4 of mass left on the other mode, plus a width term',
                'The teacher\'s entropy minus the student\'s entropy',
            ],
            a: 0,
            e: 'A student that matches one well-separated mode exactly has reverse KL = -ln(weight of that mode). -ln 0.6 = 0.511, and on the 40% mode -ln 0.4 = 0.916, which is why the other mode is a worse but still local minimum.',
        },
        {
            q: 'Why can you distil Qwen3-4B-Base into Qwen3-0.6B-Base with logit KD, but not into a SmolLM2 model?',
            o: [
                'SmolLM2 has no chat template, so its prompts differ',
                'SmolLM2 is too small to hold the teacher\'s knowledge',
                'The two Qwen3 models share one vocabulary, and SmolLM2 does not',
                'Logit KD needs the student and teacher to have the same hidden size',
            ],
            a: 2,
            e: 'Logit KD compares two distributions over the same tokens, so the vocabularies must match; Qwen3 has 151,936 rows and SmolLM2 49,152. Hidden sizes may differ, as step 2 shows with 64 and 96. Training on the teacher\'s text works across tokenizers.',
        },
        {
            q: 'You train the student with cross-entropy on the teacher\'s greedy answers. What happened to the student\'s entropy on its own answers in our run?',
            o: [
                'It rose from 0.45 to about 0.63, like forward KL on student samples',
                'It stayed at 0.45, since cross-entropy has no direction',
                'It rose to about 0.50, the same as logit KD on those answers',
                'It fell from 0.45 to about 0.28, toward the teacher\'s top tokens',
            ],
            a: 3,
            e: 'A greedy answer holds only the teacher\'s most likely token at each position, so cross-entropy on it trains the student toward the mode. Its entropy fell to 0.278 and 0.289 on two seeds, while logit KD on the same answers kept it at 0.50 to 0.52.',
        },
        {
            q: 'What does on-policy distillation change compared with logit KD on the teacher\'s answers?',
            o: [
                'The student trains on answers it sampled itself, and the teacher grades every token of those answers',
                'The teacher writes a fresh answer to every question at every step, so the training data never repeats',
                'The student copies the teacher\'s hidden states at every layer as well as its next-token outputs',
                'The loss uses only the teacher\'s top token at each position, so it needs no shared vocabulary',
            ],
            a: 0,
            e: 'The prefixes come from the student, so it learns on the states it reaches at test time. That is why sampling, 894.5 s of the 1,496 s in our run, dominates the step.',
        },
        {
            q: 'On-policy reverse KL scores 0.532 and SFT on teacher answers 0.480 on the same 500 questions, a paired interval of 1.4 to 9.2 points. A second SFT seed scores 0.528. What can you claim?',
            o: [
                'On-policy wins, since the interval excludes zero',
                'SFT wins, since its second seed is closer to the teacher',
                'Nothing yet, since a second SFT seed closes most of the gap',
                'Both win, since both beat the untrained student\'s 0.522',
            ],
            a: 2,
            e: 'The paired interval resamples test questions, not training seeds. The two SFT seeds differ by 4.8 points, and against the second one on-policy is only 0.4 ahead, interval -3.4 to +4.0. SFT seed 0 at 0.480 is below the untrained 0.522, so the option saying both beat 0.522 is false.',
        },
        {
            q: 'Which method lowered the reverse KL on the student\'s own answers the most in our grid?',
            o: [
                'Logit KD with forward KL on the teacher\'s greedy answers',
                'On-policy distillation with reverse KL on student samples',
                'SFT on the human-written GSM8K reference answers',
                'On-policy distillation with forward KL on student samples',
            ],
            a: 1,
            e: 'It trains that exact quantity, on those exact answers: 0.150 and 0.155 against 0.197 before training. SFT on the human solutions raised it to 0.420, since those answers are written in a style the teacher would not produce.',
        },
        {
            q: 'In our notebook the NF4 teacher stays loaded for every method. What adds GPU memory in logit KD over SFT on the same teacher answers?',
            o: [
                'The teacher\'s weights, which SFT on the same answers does not need loaded',
                'A second set of LoRA adapters and AdamW states for the teacher model',
                'The KV cache that the prompts of each micro-batch build during training',
                'The teacher\'s forward pass and its full-vocabulary logits at each answer token',
            ],
            a: 3,
            e: 'SFT peaked at 7.03 GB and logit KD at 8.32 GB on the same answers, with the 2.65 GB NF4 teacher resident in both. The 1.29 GB difference comes from the only extra work, the teacher\'s forward pass and its fp32 logits, about 0.6 MB per answer token over 151,936 rows.',
        },
    ],
    interviewQA: [
        {
            id: 'iv-llt-kd-1',
            difficulty: 'medium',
            tags: ['distillation', 'KL divergence'],
            question: 'Forward or reverse KL for distilling a language model? What does each do to the student?',
            answer: `Forward KL, KL(teacher || student), is averaged under the teacher. Reverse KL, KL(student || teacher), is averaged under the student.

    - **Forward KL** charges the student for missing anything the teacher considers. A student that cannot match the teacher spreads out, mode-covering, and puts mass where the teacher has little. Fitting one Gaussian to a two-mode teacher, it sat between the modes with 18.7% of its mass in the valley.
    - **Reverse KL** charges the student for putting mass where the teacher has little. A limited student settles on one mode, mode-seeking. On the 60% mode its reverse KL is -ln 0.6 = 0.51.
    - **In practice.** Cross-entropy on text sampled from the teacher is forward KL in expectation; on greedy teacher text it trains toward the mode. Reverse KL is the choice in MiniLLM and in Thinking Machines' on-policy distillation, because it only asks the teacher about tokens the student produces. GKD leaves the divergence as a setting.

    In our GSM8K runs the entropy moved with the direction in all four KL methods: forward-KL students rose to 0.50 to 0.63 nats per token from 0.45, reverse-KL students fell to 0.25 to 0.36.`,
            keyPoints: [
                'Forward KL is mode-covering, reverse KL mode-seeking',
                'Cross-entropy on sampled teacher text is forward KL in expectation',
                'MiniLLM and Thinking Machines use reverse KL on student samples',
                'Measure the student\'s entropy to see which way it moved',
            ],
        },
        {
            id: 'iv-llt-kd-2',
            difficulty: 'medium',
            tags: ['distillation', 'on-policy', 'exposure bias'],
            question: 'What is on-policy distillation, and when is it worth its cost?',
            answer: `The student samples its own answer, the teacher scores every token of it, and the student is trained to lower the per-token divergence, usually reverse KL, on those samples.

    - **Why.** Off-policy methods train on prefixes the teacher wrote. At test time the student continues its own prefixes, including its own mistakes, which it never trained on. On-policy training happens on exactly those states.
    - **Cost.** Sampling every step. Ours spent 894.5 s of 1,496 s sampling, 23.4 s per step of 32 questions against 3.9 s for SFT. But it needs no teacher-written dataset, which cost us 2,074 s for 2,048 answers.
    - **Evidence.** Thinking Machines (2025-10-27) report learning an RL-trained policy in 7-10x fewer gradient steps than RL. The Qwen3 Technical Report reports 74.4 on AIME'24 with 1,800 GPU hours against 67.6 and 17,920 for RL.
    - **Our scale.** With 2,048 GSM8K questions and 64 LoRA steps, it gave the lowest reverse KL on the student's own answers, 0.150. Accuracy was 0.532 and 0.514 against 0.522 before training, inside the interval.

    It is worth it when the student already produces sensible answers and a teacher can grade them, typically after an off-policy stage.`,
            keyPoints: [
                'Student samples, teacher grades each token',
                'Trains on the states the student reaches at test time',
                'Pays for sampling every step, not for a teacher dataset',
                'Reported gains are against RL; compare at equal compute and seeds',
            ],
        },
        {
            id: 'iv-llt-kd-3',
            difficulty: 'hard',
            tags: ['distillation', 'evaluation', 'statistics'],
            question: 'Your on-policy distilled model beats SFT on teacher outputs by 5 points on 500 GSM8K questions, and the paired interval excludes zero. Do you ship the claim?',
            answer: `Not yet. A paired bootstrap over test questions measures how much the choice of questions moves the gap. It says nothing about the training seed.

    - **What we saw.** On-policy reverse KL scored 0.532, SFT seed 0 scored 0.480, a gap of 5.2 points with an interval of 1.4 to 9.2.
    - **The second seed.** SFT seed 1 scored 0.528. Against it the gap was 0.4, interval -3.4 to +4.0. The two SFT seeds differed by 4.8 points, seed 1 minus seed 0, with their own paired interval of 1.4 to 8.4. The other two methods' seeds differed by 1.0 and 1.8.
    - **What to do.** Train at least two seeds per method, report every seed gap next to the method gap, and average seeds before comparing. Our two-seed means were 0.504 for SFT and 0.523 for on-policy, both within 2 points of the untrained 0.522.
    - **Also report** what each method optimises. The KL each method trains on moved, the accuracy did not.

    Call a gap a tie unless it exceeds the spread you measured across seeds, whatever the item-level interval says, and say how many seeds that spread comes from.`,
            keyPoints: [
                'Item-level intervals ignore training-seed variance',
                'Run two or more seeds per method',
                'Compare the method gap with the seed gap',
                'Report the optimised divergence alongside accuracy',
            ],
        },
        {
            id: 'iv-llt-kd-4',
            difficulty: 'medium',
            tags: ['distillation', 'implementation', 'memory'],
            question: 'You distil a 4B model into a 0.6B model with logit KD on one 15 GB GPU. What do you watch for?',
            answer: `Five things, all of which we hit or checked on a T4.

    - **Shared vocabulary.** Token-level KD compares distributions over the same tokens. Qwen3-4B and Qwen3-0.6B share 151,936 rows. TRL's DistillationTrainer raises an error otherwise.
    - **Logit memory.** Two fp32 distributions per answer token over 151,936 rows. Compute logits only at answer positions, use micro-batches of 4, and load the teacher in NF4 (2.65 GB). Our peaks were 7.0 to 9.7 GB, and logit KD used 1.29 GB more than SFT on the same answers. In our probe an fp16 Qwen3.5-4B teacher, 8.41 GB of weights and 248,320 rows, ran out of memory.
    - **Precision.** fp16 autocast with a GradScaler, an fp16 frozen base and fp32 LoRA weights. Check the trainable dtypes after building a TRL trainer.
    - **Stopping and scoring.** Base models keep writing new questions. A stop string that missed \`[Question]\` put our teacher at 0.219 instead of 0.81.
    - **The loss itself.** Check your divergence against the library. Ours matched TRL's kernel within 3.1e-6 in value and 1.2e-7 in gradient for beta 0, 0.5 and 1.

    Then measure the teacher's generation speed before you plan off-policy data: ours wrote 95 tokens per second at batch 64 in NF4.`,
            keyPoints: [
                'Same vocabulary, or switch to training on teacher text',
                'Project only answer positions; NF4 teacher on small GPUs',
                'fp16 with GradScaler and fp32 adapters on a T4',
                'Read full replies before trusting a score',
            ],
        },
    ],
};
