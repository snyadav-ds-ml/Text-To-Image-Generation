import torch
from diffusers import (
    AutoPipelineForText2Image,     # auto-detects SD v1.5 vs SDXL vs others
    DPMSolverMultistepScheduler,   # dpm_solver_pp
    DDIMScheduler,                 # ddim
    EulerDiscreteScheduler,        # euler
    PNDMScheduler,                 # pndm
)

SCHEDULERS = {
    "dpm_solver_pp": DPMSolverMultistepScheduler,
    "ddim":          DDIMScheduler,
    "euler":         EulerDiscreteScheduler,
    "pndm":          PNDMScheduler,
}


def load_pipeline(model_id, lora_path=None, device="cuda", scheduler="dpm_solver_pp"):
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,                                # works for SD v1.5, SDXL, and others
    )

    # swap scheduler based on config (supports A04 ablation)
    scheduler_class = SCHEDULERS.get(scheduler, DPMSolverMultistepScheduler)
    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)

    if lora_path:
        pipe.load_lora_weights(lora_path)                      # load PEFT-saved LoRA weights

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()                            # saves GPU memory

    return pipe
