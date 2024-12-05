from pathlib import Path
import sys, os


import torch
import torch.nn as nn
import torchvision.transforms as T
from diffusion.utils.misc import read_config
from diffusion.model.nets import PixArtMS
from transformers import T5EncoderModel, T5Tokenizer

from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import InterpolationMode
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
from diffusion import IDDPM, DPMS
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
import os
from diffusion.data.transforms import get_transform
from torchvision.datasets.folder import default_loader
import json
import sys



def read_json(path):
    with open(path,'r') as f:
        x = json.load(f)
    return x


def edit_image(pixart_model, src_image, instruction, config, device,
              text_guidance, img_guidance, seed=0, transform=None, tokenizer=None, text_encoder=None, vae=None, null_y=None):

    set_random_seed(seed)
    latent_size = config.image_size//8
    z = torch.randn(1, 4, latent_size, latent_size, device=device)
    src_image = transform(src_image)
    
    txt_tokens = tokenizer(
        instruction, max_length=config.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).to("cuda")
    caption_embs = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]

    latent_src = vae.encode(src_image.to(device, dtype=vae.dtype).unsqueeze(0)).latent_dist.mode()
    
    # latent_src = vae.encode(src_image.to(device, dtype=vae.dtype).unsqueeze(0)).latent_dist.sample() * config.scale_factor

    uncond_latent_src = torch.zeros_like(latent_src)
    ar = 1.
    hw = torch.tensor([[512, 512]], dtype=torch.float, device=device).repeat(1, 1)
    ar = torch.tensor([[ar]], device=device).repeat(1, 1)
    model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=txt_tokens.attention_mask, 
                                z_src=torch.cat([latent_src, latent_src, uncond_latent_src]))


    dpm_solver = DPMS(pixart_model.forward_with_dpmsolver,
                      condition=caption_embs,
                      uncondition=null_y,
                      cfg_scale=text_guidance,
                      model_kwargs=model_kwargs,
                      img_cfg_scale=img_guidance)

    denoised = dpm_solver.sample(
            z,
            steps=14,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
    latent = denoised.to(torch.float16)
    samples = vae.decode(latent.detach() / vae.config.scaling_factor).sample
    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    image = Image.fromarray(samples)
    return image


def main(image_path, edit_instruction):
    ROOT_DIR = "./ckpt"
    CKPT_PATH = os.path.join(ROOT_DIR, 'epoch_40_step_90041.pth')
    CONFIG_PATH = os.path.join('./configs/pixart_sigma_config/editing_at_512.py', "config.py")
    PIPELINE_LOAD_PATH = "output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers"

    SD = torch.load(CKPT_PATH, 'cpu')
    CONFIG = read_config(CONFIG_PATH)
    kwargs = {'config': CONFIG}
    kwargs['config']['input_size'] = 64

    CONFIG.seed = init_random_seed(CONFIG.get('seed', None))
    set_random_seed(CONFIG.seed)

    pixart_model = PixArtMS(depth=28, hidden_size=1152, patch_size=2, num_heads=16, in_channels=8, **kwargs)

    pixart_model.load_state_dict(SD['state_dict'], strict=False)

    device = "cuda"
    pixart_model = pixart_model.to(device).eval()

    image_size = CONFIG.image_size
    vae = AutoencoderKL.from_pretrained(CONFIG.vae_pretrained).to(device).to(torch.float16)
    tokenizer = T5Tokenizer.from_pretrained(PIPELINE_LOAD_PATH, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
                PIPELINE_LOAD_PATH, subfolder="text_encoder", torch_dtype=torch.float16).to(device)

    transform = get_transform('default_train', image_size)
    null_y = torch.load(f'./output/pretrained_models/null_embed_diffusers_{CONFIG.model_max_length}token.pth')
    null_y = null_y['uncond_prompt_embeds'].to(device)


    seed = 0


    ip_image = Image.open(image_path).convert('RGB').resize((512,512))
    
    edited_image = edit_image(pixart_model, ip_image, edit_instruction, CONFIG, device, text_guidance=4.5, img_guidance=1.5, transform=transform, tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, null_y=null_y)

    edited_image.save(f"./output/{edit_instruction.replace(' ','_')}.png")
    print(f"Edited Image saved at ./output/{edit_instruction.replace(' ','_')}.png")

if __name__ == "__main__":
    image_path, edit_instruction = sys.argv[1], sys.argv[2]
    main(image_path, edit_instruction)