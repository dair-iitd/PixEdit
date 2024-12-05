_base_ = ['../PixArt_xl2_internal.py']
data_root = ""
image_list_json = ["pixart_aurora.json"]

data = dict(
    type='InternalDataSigma_Editing', root=data_root, image_list_json=image_list_json, transform='default_train',
    load_vae_feat=False, load_t5_feat=False,
)
image_size = 256

# model setting
model = 'PixArtMS_XL_2'
mixed_precision = 'fp16'  # ['fp16', 'fp32', 'bf16']
fp32_attention = True
load_from = "output/pretrained_models/PixArt-Sigma-XL-2-256x256.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
resume_from = None
vae_pretrained = "output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
aspect_ratio_type = 'ASPECT_RATIO_512'
multi_scale = False  # if use multiscale dataset model training
pe_interpolation = 0.5

# training setting
num_workers = 10
train_batch_size = 8  # 48 as default
num_epochs = 3  # 3
gradient_accumulation_steps = 4
grad_checkpointing = False
gradient_clip = 0.01
optimizer = dict(type='CAMEWrapper', lr=2e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 1000
visualize = True
log_interval = 20
save_model_epochs = 5
save_model_steps = 2500
work_dir = 'output/debug'

# pixart-sigma
scale_factor = 0.13025
real_prompt_ratio = 1.0
model_max_length = 300
class_dropout_prob = 0.05


report_to="wandb"
tracker_project_name="pixart-editing"