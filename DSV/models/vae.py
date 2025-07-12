import datetime

import apex
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from torch.profiler import ProfilerActivity, profile, record_function

device = "cuda"


# vae = AutoencoderKLTemporalDecoder.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",subfolder="vae", torch_dtype=torch.bfloat16).to(device)


vae = (
    AutoencoderKL.from_pretrained("/data/hf_model/sd-vae-ft-ema")
    .to(device)
    .to(torch.bfloat16)
)
vae.eval()


input_ = torch.randn(16, 3, 768, 768, device=device, dtype=torch.bfloat16)

print(f"pytorch reserved memory: {torch.cuda.memory_reserved(device)/1024/1024} MB")
print(f"pytorch allocated memory: {torch.cuda.memory_allocated(device)/1024/1024} MB")

with torch.no_grad():
    output_vae = vae.encode(input_).latent_dist.sample().mul_(0.18215)

print(output_vae.shape)

exit()

activities = [
    ProfilerActivity.CPU,
    ProfilerActivity.CUDA,
]

log_dir = f"./log/vae_profile"


with profile(
    activities=activities,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for i in range(100):
        with record_function("vae_process"):
            with torch.no_grad():
                with record_function("encoding"):
                    latent = vae.encode(input).latent_dist.sample()
                prof.step()
