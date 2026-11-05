# hf_wrapper.py - wrap the course GPT as a Hugging Face PreTrainedModel so lm-evaluation-harness can score it.
# This is the reference solution for the capstone's "write the HF-compatible wrapper" step.
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from gpt import GPT


class LLTConfig(PretrainedConfig):
    model_type = "llt-gpt"

    def __init__(self, V=50257, d=768, L=12, H=12, ctx=1024, **kwargs):
        self.V, self.d, self.L, self.H, self.ctx = V, d, L, H, ctx
        self.vocab_size = V
        self.max_position_embeddings = ctx      # lm-eval reads this to know the context length
        super().__init__(**kwargs)


class LLTForCausalLM(PreTrainedModel):
    config_class = LLTConfig

    def __init__(self, config):
        super().__init__(config)
        self.gpt = GPT(config.V, config.d, config.L, config.H, config.ctx)
        self.post_init()

    def _init_weights(self, module):
        pass                                     # weights come from our checkpoint, never from HF init

    def tie_weights(self, *args, **kwargs):
        pass                                     # the GPT already shares one tensor between embedding and head

    def get_input_embeddings(self):
        return self.gpt.emb

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Right padding (what lm-eval uses for causal models) cannot change earlier positions, so the mask is unused.
        logits = self.gpt(input_ids)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().flatten(0, 1), labels[:, 1:].flatten())
        return CausalLMOutput(loss=loss, logits=logits)


def load(weights_path, device="cuda", dtype=torch.float16):
    """Load a weights_step_XXXXXX.pt file written by train.py."""
    blob = torch.load(weights_path, map_location="cpu", weights_only=True)   # tensors and plain types only
    model = LLTForCausalLM(LLTConfig(**blob["config"]))
    model.gpt.load_state_dict(blob["state_dict"], strict=True)
    return model.to(device=device, dtype=dtype).eval(), blob
