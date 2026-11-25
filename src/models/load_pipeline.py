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
    # Use float16 on GPU (fast), float32 on CPU (safe)
    dtype = torch.float32 if device == "cpu" else torch.float16

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,                                # works for SD v1.5, SDXL, and others
    )

    # swap scheduler based on config (supports A04 ablation)
    scheduler_class = SCHEDULERS.get(scheduler, DPMSolverMultistepScheduler)
    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)

    if lora_path:
        import os
        # detect format by checking which weight file exists in the checkpoint folder
        peft_file = os.path.join(lora_path, "adapter_model.safetensors")
        diff_file = os.path.join(lora_path, "pytorch_lora_weights.safetensors")

        if os.path.exists(peft_file):
            # PEFT format (saved with unet.save_pretrained before fix)
            from peft import PeftModel
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        elif os.path.exists(diff_file):
            # diffusers format (saved with unet.save_attn_procs after fix)
            pipe.load_lora_weights(lora_path)
        else:
            raise FileNotFoundError(f"No LoRA weights found in {lora_path}")

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()                            # saves GPU memory

    return pipe
