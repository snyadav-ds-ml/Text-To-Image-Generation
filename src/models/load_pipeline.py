import torch                                                    # tensor dtype
from diffusers import StableDiffusionPipeline                  # SD v1.5 pipeline
from diffusers import DPMSolverMultistepScheduler              # default sampler


def load_pipeline(model_id, lora_path=None, device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,                             # halves GPU memory
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config                                  # swap in DPM-Solver++
    )

    if lora_path:
        pipe.unet.load_attn_procs(lora_path)                   # attach LoRA weights

    pipe = pipe.to(device)                                     # move to GPU
    pipe.enable_attention_slicing()                            # saves GPU memory

    return pipe
