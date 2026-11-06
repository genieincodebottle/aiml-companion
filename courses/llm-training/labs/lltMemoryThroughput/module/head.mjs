// lltMemoryThroughput - Make Training Fit in Memory, Run Faster and Use Two GPUs (LLM Training Lab, module 8, premium).
//
// Lab page (type 'lab-lesson'), rendered by src/components/module/LabLesson.jsx.
// Blueprint: .claude/docs/LLM_TRAINING_PROGRAM_PLAN.md section 9 entry 8, sections 5.2-5.4, 7.8.3.
// Built by aiml-companion/courses/llm-training/labs/lltMemoryThroughput/module/build_module.py.
// The CODE strings below are the step files in labs/lltMemoryThroughput/steps/ verbatim; edit those, then rebuild.
//
// NUMBERS ARE A CONTRACT. Every figure traces to a Run Library file in
// aiml-companion/courses/llm-training/runs/lltMemoryThroughput/ or to an exact derivation.
//   Part 1, one Kaggle T4 (kernel llt-memthru-p1 v2, 2026-11-05, python 3.13.16, torch 2.11.0+cu130)
//     -T2-memory-grid, -T2-memory-cuts, -T2-precision, -T2-broken-no-synchronize,
//     -T2-broken-reserved-vs-allocated, -T2-broken-fp16-half-adamw, -T2-broken-leftover-loss (v1 vs v2)
//   Part 2, Kaggle 2x T4 (kernel llt-memthru-p2 v1)
//     -T2x2-allreduce, -T2x2-ddp, -T2x2-broken-same-seed, -T2x2-fsdp2, -T2x2-dcp, -T2x2-broken-rank0-save
//   Part 3, Kaggle 2x T4 (kernel llt-memthru-p3 v1): -T2-lr-<size>-<variant>-<lr>, 30 runs, and -T2x2-dcp-repeat
//   Also cited: llt-gpu-smoke (22.1 TFLOPS fp16 matmul on the same Kaggle T4).
// Derived: 16N = 1.991 GB and 14N = 1.742 GB for N = 124,439,808; 12V = 603,084 B; 38dL = 350,208 B;
//   6N + 12Lds = 746.6M + 113.2M FLOPs per token at s = 1,024; 16N = 21.02 GB for N = 1,313,626,112.
// Code strings: no backslash and no dollar-brace anywhere (they sit in template literals).

/*@@CODE@@*/
export const lltMemoryThroughput = {
    type: 'lab-lesson',
    title: 'Make Training Fit in Memory, Run Faster and Use Two GPUs',
    subtitle: 'Predict peak training memory within 2% on a T4 and cut it by 71%. Then train on two T4s with DDP and FSDP2, and measure how far QK-norm pushes the learning rate.',
    prerequisites: ['dlTraining', 'dlGpuFundamentals', 'dlDistributed', 'collectiveCommsNetworking'],
    objectives: [
        'Derive the peak memory of a GPT-2 training step from the tensors autograd keeps, and check it against `torch.cuda.max_memory_allocated` on a T4.',
        'Time a step correctly, compute MFU, and measure what checkpointing, chunked cross-entropy, bf16 and `torch.compile` cost or save on a T4.',
        'Measure the all-reduce between two T4s and the speedup DDP gets from them, with and without `no_sync`.',
        'Train a 1.3B-parameter model that one T4 can\'t hold with FSDP2, and save and reload it with a sharded checkpoint.',
        'Reproduce a learning-rate blow-up in a 10M-parameter model, and measure how QK-norm and z-loss change the range of learning rates that train.',
    ],
    tldr: 'Training GPT-2 small needs 16 bytes per parameter for its weights, gradients and Adam state, plus memory for activations and logits that grows with batch x sequence. You predict the peak within 2% on a T4 and cut it by 71%. Then you train on two T4s and measure how far QK-norm and z-loss push the learning rate.',
    recap: {
        trackId: 'deepLearning',
        moduleId: 'dlTraining',
        moduleTitle: 'Training Deep Networks',
        points: [
            'Mixed precision runs matrix multiplies in fp16 or bf16 and keeps the weights in fp32.',
            'A gradient scaler multiplies the loss so small fp16 gradients don\'t round to zero.',
            'Gradient accumulation adds several micro-batches\' gradients before one optimizer step.',
        ],
    },
    regime: {
        badge: 'FROM SCRATCH',
        startsFrom: 'random weights, GPT-2 small and a 1.3B-parameter GPT for parts 1 and 2, a 10M or 27M TinyStories GPT for part 3',
        changes: 'all weights; parts 1 and 2 train a few steps on random tokens to measure memory and speed, part 3 trains 600 steps on TinyStories',
        objective: 'parts 1 and 2 use random tokens as their own targets, so their loss only checks the code. Part 3 minimizes next-token loss on TinyStories, plus z-loss in one variant',
    },
    lab: {
        defaultTier: 'T2',
        quickPath: { minutes: '@@QUICK_MIN@@', reaches: '@@QUICK_REACHES@@' },
        noGpuPath: 'This lab needs a GPU for its notebook. Without one, run the predictor in step 2 and both challenges in this page, and read every other result from our recorded Kaggle runs. The run files, with configs, seeds and versions, are at https://github.com/genieincodebottle/aiml-companion/tree/main/courses/llm-training/runs/lltMemoryThroughput.',
        tiers: [
            { tier: 'T2', model: 'Kaggle, GPU T4 x2. GPT-2 small on one T4, DDP and a 1.3B FSDP2 model on two, a 10M TinyStories sweep', peakMemGB: 13.32, wallClock: '@@T2_WALL@@', costUSD: 0, givesUp: 'Nothing. bf16 runs but is emulated on a T4, so its speed is not what an Ampere GPU gives.' },
            { tier: 'T0', model: 'The predictor in step 2 and both challenges, in this page', costUSD: 0, givesUp: 'Every measurement. Read them from our runs.' },
        ],
        notebooks: {
            colab: 'https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/courses/llm-training/labs/lltMemoryThroughput/lltMemoryThroughput.ipynb',
            kaggle: 'https://www.kaggle.com/kernels/welcome?src=https://github.com/genieincodebottle/aiml-companion/blob/main/courses/llm-training/labs/lltMemoryThroughput/lltMemoryThroughput.ipynb',
        },
        env: 'Kaggle 2x Tesla T4 15.6 GB | python 3.13.16 | torch 2.11.0+cu130, the APIs used also checked in the PyTorch 2.15.0 source | datasets 4.8.5 and tokenizers 0.23.3 for part 3; the notebook pins datasets 5.1.0, which writes the same token files on our CPU check | browser cells on Pyodide 0.27',
        checkedOn: '2026-11-05',
    },
    lesson: {
        build: [
/*@@body_part1.mjs@@*/
/*@@body_part2.mjs@@*/
/*@@body_part3.mjs@@*/
        ],
