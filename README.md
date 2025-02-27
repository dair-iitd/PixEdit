# Installation
```python
conda create -n pixedit python==3.9.0
conda activate pixedit
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

#Install git, if not available
conda install anaconda::git

git clone https://github.com/dair-iitd/PixEdit
cd PixEdit
pip install -r requirements.txt

pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118

#Install git-lfs, if not available
conda install anaconda::git-lfs

# SDXL-VAE, T5 checkpoints
git lfs install
git clone https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers

```

# Dataset Prepration
## Seed-Edit
```python
git lfs install

#We use just the real image editing pairs 
git clone https://huggingface.co/datasets/AILab-CVC/SEED-Data-Edit-Part2-3
cd SEED-Data-Edit-Part2-3/multi_turn_editing/images

cat multi_turn.tar.gz.part-* > multi_turn.tar.gz

#unzip the images
tar -xvf multi_turn.tar.gz
```

## AURORA
Follow [this](https://github.com/McGill-NLP/AURORA?tab=readme-ov-file#training-data-aurora) for setting up AURORA training data


We require all the dataset to be in the format required by Pixart-$`\Sigma`$. Example can be found [here](https://github.com/PixArt-alpha/PixArt-sigma/blob/master/asset/docs/convert_image2json.md). We provide the necessary `.json` files for both Seed-edit and Aurora datasets [here](https://csciitd-my.sharepoint.com/:f:/g/personal/aiz228170_iitd_ac_in/EtvsDFGW0kFFibI20yeckw8BpAwePzQ3bwiQuTeMPIjxNg?e=WsuMLU)


You can additionally use the following command to convert any dataset of your choice in the required format.

`python tools/convert_data_pixedit.py [params] images_path output_path`




# Training 
We performed all training on a 8xA100 server. Set `--nproc_per_node` according to your configuration.\
Download the PixArt-Sigma Checkpoint from [here](https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-512-MS.pth).
```python
python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port=12345 train_scripts/train.py \
    configs/pixart_sigma_config/editing_at_512.py \
    --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
    --work-dir output/run1 --report_to wandb --tracker_project_name PixEdit
```

# Inference
Download the v1 trained checkpoint [PixEdit-v1.pth](https://anon-cvpr.s3.eu-north-1.amazonaws.com/epoch_40_step_90041.pth) or [🤗](https://huggingface.co/aggr8/PixEdit-v1), place it in `ckpt` folder.

```python
python edit_image.py <image_path> <edit_instruction>
```


# Release Checklist
- [x] Release Training and Inference Code.
- [x] Release PixEdit-v1.
- [ ] PixEdit-v2

# Acknowledgements
- Thanks to [Pixart-$`\Sigma`$](https://github.com/PixArt-alpha/PixArt-sigma) for their wonderful codebase!

# Citation
If you find this repository useful, please consider giving a star ⭐ and citation.
```text
@misc{goswami2024grapegenerateplaneditframeworkcompositional,
      title={GraPE: A Generate-Plan-Edit Framework for Compositional T2I Synthesis}, 
      author={Ashish Goswami and Satyam Kumar Modi and Santhosh Rishi Deshineni and Harman Singh and Prathosh A. P and Parag Singla},
      year={2024},
      eprint={2412.06089},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.06089}, 
}
```