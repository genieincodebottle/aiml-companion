        {
            q: 'You train GPT-2 small, 124,439,808 parameters, under fp16 autocast with AdamW. How much memory do its weights, gradients and Adam state take before any activation?',
            o: ['About 0.5 GB, fp16 weights and gradients only', 'About 1.5 GB, at 12 bytes per parameter', 'About 2.0 GB, at 16 bytes per parameter', 'About 1.0 GB, fp32 weights and gradients only'],
            a: 2,
            e: 'Autocast keeps fp32 weights, so each parameter holds a 4-byte weight, a 4-byte gradient and two 4-byte Adam moments. 16 x 124.4M = 1.99 GB, and step 1 measured 2.02 GB. The 12-byte count leaves out the fp32 weights.',
        },
        {
            q: 'GPT-2 small peaks at 9.48 GB on a T4 at batch 8, sequence 1,024. You double the batch to 16. What happens on the 15.6 GB card?',
            o: ['It fits, at about 11.4 GB', 'It runs out of memory, needing about 17.4 GB', 'It fits, at about 13.5 GB', 'It runs out of memory, needing about 19.0 GB'],
            a: 1,
            e: 'Only the per-token part doubles. 1.74 GB of weights, Adam state and fp16 copies plus 16,384 tokens x 958 KB gives 17.44 GB, and the run ran out of memory. Doubling the whole 9.48 GB would overcount the fixed part.',
        },
        {
            q: 'At batch 8, sequence 1,024, which single switch cuts GPT-2 small\'s peak memory more on a T4?',
            o: ['Checkpointing, since the 12 blocks hold most activations', 'Chunked cross-entropy, since the logits hold more', 'Both save the same, as each halves the activations', 'Checkpointing, but only once torch.compile is on'],
            a: 1,
            e: 'With a 50,257-token vocabulary the loss holds 12V = 603 KB per token at its peak, against 350 KB for all 12 blocks. Chunking saved 4.13 GB and checkpointing 2.60 GB.',
        },
        {
            q: 'A notebook picks bf16 because `torch.cuda.is_bf16_supported()` returns True on a Kaggle T4. Predict its training speed against fp16 autocast.',
            o: ['About 1.0x, the same speed as fp16', 'About 1.4x faster, as bf16 needs no loss scaling', 'It stops with an unsupported-dtype error', 'About 5x slower than fp16, slower even than fp32'],
            a: 3,
            e: 'The T4 has fp16 tensor cores and no bf16 ones. bf16 ran at 2,546 tokens/s against 13,031 for fp16, slower even than fp32. Choose the dtype by compute capability 8.0 or higher.',
        },
        {
            q: 'Without `torch.cuda.synchronize()`, GPT-2 small\'s forward pass times at 9.2 ms on a T4 and its backward pass at 13.6 ms. What do the synchronized timings show?',
            o: ['Forward about 204 ms, backward about 318 ms', 'Forward about 9 ms, backward about 300 ms', 'Both within 10% of the unsynchronized values', 'Forward about 2 ms, backward about 14 ms'],
            a: 0,
            e: 'CUDA calls return once the work is queued. Unsynchronized timings measure the queueing, so both passes look tiny. Synchronized, the forward pass took 204.3 ms and the backward pass 317.9 ms.',
        },
        {
            q: 'DDP on two T4s runs without errors, and both ranks log a loss of 9.638648 at the same step. What does that tell you?',
            o: ['DDP averaged the losses before printing them', 'The all-reduce failed and each rank trained alone', 'Both ranks drew the same batches every step', 'The model has converged on both GPUs'],
            a: 2,
            e: 'DDP all-reduces gradients, not the logged loss, so each rank prints the loss of its own batch. Identical losses to 6 decimals mean identical batches. Seeding the data with 1234 + rank gave 9.481456 and 9.501394.',
        },
        {
            q: 'A 1.31B model trains with FSDP2 on two GPUs. Rank 0 alone calls `torch.save(model.state_dict())`. What does the file hold?',
            o: ['The full model, gathered by FSDP for the save', 'Rank 0\'s shard of each weight, about half', 'The full model and its optimizer state', 'Nothing, since FSDP blocks the call with an error'],
            a: 1,
            e: 'Each tensor in a sharded state dict is that rank\'s shard. Our file was 2.63 GB and held 708,278,272 of 1,416,552,448 values, and loading it into a full model failed with a size mismatch.',
        },
@@QUIZ_P3@@
