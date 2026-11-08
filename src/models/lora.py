from peft import LoraConfig, get_peft_model  # LoRA implementation


DEFAULT_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]  # attention layers only


def add_lora(unet, rank=16, alpha=16, target_modules=None):
    if target_modules is None:
        target_modules = DEFAULT_TARGETS              # use defaults if not specified

    config = LoraConfig(
        r=rank,                                       # adapter matrix size (ablated A05)
        lora_alpha=alpha,                             # scales LoRA output strength
        target_modules=target_modules,                # which layers get adapters
        lora_dropout=0.0,                             # no dropout for now
        bias="none",                                  # don't train bias terms
    )

    unet = get_peft_model(unet, config)               # wraps unet with LoRA adapters
    unet.print_trainable_parameters()                 # shows how few params we train

    return unet
