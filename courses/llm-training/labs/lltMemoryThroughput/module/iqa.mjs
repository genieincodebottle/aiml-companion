        {
            id: 'iv-lltmemorythroughput-1',
            difficulty: 'medium',
            level: 'Mid',
            tags: ['memory', 'fine-tuning', 'sharding'],
            question: 'How much GPU memory does full fine-tuning of an 8B-parameter model with AdamW need, and how would you fit it?',
            answer: `Start with the training state, then the activations, then the logits.

**Training state**

Under PyTorch mixed precision each parameter holds an fp32 weight, an fp32 gradient and Adam's two fp32 moments, 16 bytes. For 8B parameters that is 128 GB before a single activation, more than one 80 GB GPU.

**Sharding**

ZeRO stage 3, which is what FSDP does, divides those 16 bytes by the number of GPUs. On 8 GPUs that is 16 GB each. In our lab, 16N/2 predicted 10.51 GB per T4 for a 1.31B model, and the run peaked at 13.16 GB once activations and the gathered blocks were added.

**Activations and logits**

Checkpoint every block so it keeps only its input. Then look at the logits, which are easy to miss. In the setup we measured, the loss holds 12 bytes per vocabulary entry per token at its peak. With a 128,000-token vocabulary that is 1.5 MB per token, about 50 GB for 32,768 tokens. Chunked cross-entropy divides it by the number of chunks.

**Then check it**

Predict the peak, run three steps, and compare with torch.cuda.max_memory_allocated after resetting the peak. Our GPT-2 predictor landed within 1.6% across 17 sizes, so a large gap points at a tensor the count missed.`,
            keyPoints: [
                '16 bytes per parameter under AMP with Adam, so 128 GB for 8B parameters',
                'FSDP or ZeRO-3 divides that by the number of GPUs, 16 GB each on 8',
                'Logits at 12 bytes per vocabulary entry per token can rival the weights at large vocabularies',
                'Predict, then check against max_memory_allocated, not reserved memory',
            ],
        },
        {
            id: 'iv-lltmemorythroughput-2',
            difficulty: 'medium',
            level: 'Mid',
            tags: ['DDP', 'communication', 'profiling'],
            question: 'Your DDP run on two GPUs gets 1.82 times the throughput of one GPU. Where does the rest go, and how do you find out?',
            answer: `Measure the link first, then compare the step time with and without communication.

**The link**

On Kaggle's two T4s, nvidia-smi topo -m shows a PCIe host bridge and no NVLink. An all-reduce benchmark reached 4.06 GB/s of bus bandwidth for large messages.

**What one step costs**

GPT-2 small's fp32 gradients are 497.8 MB. A ring all-reduce on 2 GPUs sends that once per GPU, about 123 ms at 4.06 GB/s. One GPU took 2.23 s for its 32,768 tokens per optimizer step, and the two-GPU step took 2.45 s. So 0.22 s was added, of which the all-reduce explains about 0.12 s if none of it overlapped.

**Levers**

- no_sync on all but the last micro-batch. Syncing every micro-batch cost 12% of throughput in our run.
- More accumulation per optimizer step, so one all-reduce covers more tokens.
- A faster link. NVLink-connected GPUs move far more than 4 GB/s.

**What I would not claim**

The other 0.1 s is not explained by the all-reduce alone. To find it, run torch.profiler on both ranks and look for one rank waiting on the other, or for gradient bucket copies.`,
            keyPoints: [
                'Measure bus bandwidth with an all-reduce benchmark before reasoning about scaling',
                'A 2-GPU ring all-reduce sends the full gradient size once per GPU',
                'no_sync during accumulation saved 12% of throughput in our run',
                'Profile both ranks for the part the all-reduce does not explain',
            ],
        },
        {
            id: 'iv-lltmemorythroughput-3',
            difficulty: 'hard',
            level: 'Senior',
            tags: ['FSDP', 'ZeRO', 'trade-offs'],
            question: 'When do you choose FSDP over DDP, and what does it cost you?',
            answer: `Choose FSDP when the training state does not fit on one GPU. Otherwise DDP is simpler and faster.

**The memory side**

DDP keeps all 16 bytes per parameter on every GPU. FSDP shards weights, gradients and optimizer state, so each GPU keeps 16N/n plus working memory. Our 1.31B model needed 21.02 GB of training state. One T4 ran out of memory at 15.33 GB, and two T4s trained it at a 13.16 GB peak each.

**The communication side**

DDP moves the gradients once per step. FSDP all-gathers each block's weights before it runs, again in the backward pass, and reduce-scatters the gradients. The ZeRO paper puts stage 3 at 1.5 times DDP's volume, and activation checkpointing adds another gather for the recompute.

**What it cost here**

Over PCIe at about 4 GB/s, the 1.31B model ran at 2,033 tokens/s on two T4s, 13.3% MFU. GPT-2 small under DDP on the same pair ran at 26,768 tokens/s.

**Middle options**

ZeRO stage 1 shards only the optimizer state and stage 2 also the gradients, both at DDP's communication volume. If the model fits once the optimizer state is sharded, start there.`,
            keyPoints: [
                'FSDP when 16N does not fit on one GPU, DDP otherwise',
                'Stage 3 moves 1.5 times the data of DDP, more with recomputation',
                'Our 1.31B model trained on two T4s at 13.16 GB each and 13.3% MFU',
                'Stages 1 and 2 save memory at DDP\'s communication volume',
            ],
        },
@@IQA_P3@@
