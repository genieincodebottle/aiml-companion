FULL_LR = 1e-5        # best held-out loss of 3e-6, 1e-5, 3e-5 and 1e-4 in our sweep
LORA_LR = 3e-4        # best held-out loss of 3e-5, 1e-4, 3e-4, 1e-3 and 3e-3, r16 on all linear layers
FULL_BUDGET = 1024    # tokens per micro-batch (rows x longest row) for full fine-tuning
LORA_BUDGET = 2048    # LoRA keeps no gradients or optimizer state for the base, so it fits twice the tokens
