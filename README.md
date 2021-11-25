# DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cvb5xZY7NU6tgkpuYwiLIAOr9KPYJkv_)

<p align="center">

  <img src="https://github.com/submission10095/DiffusionCLIP_temp/blob/master/imgs/main1.png" />

  <img src="https://github.com/submission10095/DiffusionCLIP_temp/blob/master/imgs/main2.png" />

</p> 

[comment]: <> (![]&#40;imgs/main1.png&#41;)

[comment]: <> (![]&#40;imgs/main2.png&#41;)

> **DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation**<br>
> Annonymous
>
>**Abstract**: <br>
> Recently, GAN inversion methods combined with Contrastive Language-Image Pretraining (CLIP) enables zero-shot image manipulation guided by text prompts. 
> However, their applications to diverse real images are still difficult due to the limited GAN inversion capability. 
> Specifically, these approaches often have difficulties in reconstructing images with novel poses, views, and highly variable contents compared to the training data, altering object identity, or producing unwanted image artifacts. 
> To mitigate these problems and enable faithful manipulation of real images, we propose a novel method, dubbed DiffusionCLIP, that performs text-driven image manipulation using diffusion models. 
> Based on full inversion capability and high-quality image generation power of recent diffusion models, our method performs zero-shot image manipulation successfully even between unseen domains. 
> Furthermore, we propose a novel noise combination method that allows straightforward multi-attribute manipulation. 
> Extensive experiments and human evaluation confirmed robust and superior manipulation performance of our methods compared to the existing baselines.

## Description

This repo includes the official PyTorch implementation of DiffusionCLIP, a CLIP-based text-guided image manipulation method for Diffusion models.
DiffuionCLIP leverages the sampling and inversion processes based on [DDIM](https://arxiv.org/abs/2010.02502) sampling and its reversal,
which not only accelerate the manipulation but also enable nearly **perfect inversion**.  
DiffusonCLIP can perform following tasks. 

* Manipulation of Images in Trained Domain & to Unseen Domain
  * Our method can even manipulate __ImageNet-512 images__ successfully, which haven't rarely tried by GAN inversion methods due to the limited inversion performance which is resulted from the diversity of ImageNet images.
* Image Translation from Unseen Domain into Another Unseen Domain
* Generation of Images in Unseen Domain from Strokes
* Multiple attribute changes

With existing works, users often require **the combination of multiple models, tricky task-specific
loss designs or dataset preparation with large manual effort**. On the other hand, our method is **free
of such effort** and enables applications in a natural way with the original pretrained and fine-tuned
models by DiffusionCLIP. 

The training process is illustreted in the following figure:
 
![](imgs/method1.png)

We also propose two fine-tuning scheme. Quick original fine-tuning and GPU-efficient fine-tuning. For more details, please refer to Sec. B.1 in Supplementary Material.
![](imgs/method2.png)


## Getting Started

### Installation
We recommend running our code using:

- NVIDIA GPU + CUDA, CuDNN
- Python 3, Anaconda

To install our implementation, clone our repository and run following commands to install necessary packages:
  ```shell script
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```
### Resources
- For the original fine-tuning, VRAM of 24 GB+ for 256x256 images are required.  
- For the GPU-efficient fine-tuning, VRAM of 12 GB+ for 256x256 images and 24 GB+ for 512x512 images are required.   
- For the inference, VRAM of 6 GB+ for 256x256 images and 9 GB+ for 512x512 images are required.  

### Pretrained Models for DiffusionCLIP Fine-tuning

To manipulate soure images into images in CLIP-guided domain, the **pretrained Diffuson models** are required.

| Image Type to Edit |Size| Pretrained Model | Dataset | Reference Repo. 
|---|---|---|---|---
| Human face |256×256| Diffusion (Auto), [IR-SE50](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view) | [CelebA-HQ](https://arxiv.org/abs/1710.10196) | [SDEdit](https://github.com/ermongroup/SDEdit), [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) 
| Church |256×256| Diffusion (Auto) | [LSUN-Bedroom](https://www.yf.io/p/lsun) | [SDEdit](https://github.com/ermongroup/SDEdit) 
| Bedroom |256×256| Diffusion (Auto) | [LSUN-Church](https://www.yf.io/p/lsun) | [SDEdit](https://github.com/ermongroup/SDEdit) 
| Dog face |256×256| [Diffusion](https://drive.google.com/file/d/14OG_o3aa8Hxmfu36IIRyOgRwEP6ngLdo/view) | [AFHQ-Dog](https://arxiv.org/abs/1912.01865) | [ILVR](https://github.com/jychoi118/ilvr_adm)
| ImageNet |512×512| [Diffusion](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt) | [ImageNet](https://image-net.org/index.php) | [Guided Diffusion](https://github.com/openai/guided-diffusion)
- The pretrained Diffuson models on 256x256 images in [CelebA-HQ](https://arxiv.org/abs/1710.10196), [LSUN-Church](https://www.yf.io/p/lsun), and [LSUN-Bedroom](https://www.yf.io/p/lsun) are automatically downloaded in the code.
- In contrast, you need to download the models pretrained on [AFHQ-Dog-256](https://arxiv.org/abs/1912.01865) or [ImageNet-512](https://image-net.org/index.php) in the table and put it in `./pretrained` directory. 
- In addtion, to use ID loss for preserving Human face identity, you are required to download the pretrained [IR-SE50](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view) model from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)  and put it in `./pretrained` directory.


### Datasets 
To precompute latents and fine-tune the Diffusion models, you need about 30+ images in the source domain. You can use both **sampled images** from the pretrained models or **real source images** from the pretraining dataset. 
If you want to use [CelebA-HQ](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ), [LSUN-Church](https://www.yf.io/p/lsun), [LSUN-Bedroom](https://www.yf.io/p/lsun) and [AFHQ-Dog](https://github.com/clovaai/stargan-v2) and [ImageNet](https://image-net.org/index.php)  datastet directly, you can download them and fill their paths in `./configs/paths_config.py`. 

### Colab Notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cvb5xZY7NU6tgkpuYwiLIAOr9KPYJkv_)
We provide a colab notebook for you to play with DiffusionCLIP! Due to 12GB of the VRAM limit in Colab, we only provide the codes of inference & applications with the fine-tuned DiffusionCLIP models, not fine-tuning code. 
We provide a wide range of types of edits, but you can also upload your fine-tuned models following below instructions on Colab and test them.



## DiffusionCLIP Fine-tuning 


To fine-tune the pretrained Diffusion model guided by CLIP, run the following commands:

```
python main.py --clip_finetune          \
               --config celeba.yml      \
               --exp ./runs/test        \
               --edit_attr neanderthal  \
               --do_train 1             \
               --do_test 1              \
               --n_train_img 50         \
               --n_test_img 10          \
               --n_iter 5               \
               --t_0 500                \
               --n_inv_step 40          \
               --n_train_step 6         \
               --n_test_step 40         \
               --lr_clip_finetune 8e-6  \
               --id_loss_w 0            \
               --l1_loss_w 1            
```
- You can use `--clip_finetune_eff` instead of `--clip_finetune` to save GPU memory.
- `config`: `celeba.yml` for human face, `bedroom.yml` for bedroom, `church.yml` for church, `afhq.yml` for dog face and , `imagenet.yml` for images from ImageNet.
- `exp`: Experiment name.
- `edit_attr`: Attribute to edit, you can use `./utils/text_dic.py` to predefined source-target text pairs or define new pair. 
  - Instead, you can use `--src_txts` and `--trg_txts`. 
- `do_train`, `do_test`: If you finish training quickly withouth checking the outputs in the middle of training, you can set `do_test` as 1.
- `n_train_img`, `n_test_img`: # of images in the trained domain for training and test.        
- `n_iter`: # of iterations of a generative process with `n_train_img` images.
- `t_0`: Return step in [0, 1000), high `t_0` enable severe change but may lose more identity or semantics in the original image.  
- `n_inv_step`, `n_train_step`, `n_test_step`: # of steps during the generative pross for the inversion, training and test respectively. They are in `[0, t_0]`. We usually use 40, 6 and 40  for `n_inv_step`, `n_train_step` and `n_test_step` respectively. 
   - We found that the manipulation quality is better when `n_***_step` does not divide `t_0`. So we usally use 301, 401, 500 or 601 for `t_0`.
- `lr_clip_finetune`: Initial learning rate for CLIP-guided fine-tuning.
- `id_loss_w`, `l1_loss` : Weights of ID loss and L1 loss when CLIP loss weight is 3.



## Novel Applications

The fine-tuned models through DiffusionCLIP can be leveraged to perform the several novel applications. 

### Manipulation of Images in Trained Domain & to Unseen Domain
![](imgs/app_1_manipulation.png)

You can edit one image into the CLIP-guided domain by running the following command:
``` 
python main.py --edit_one_image            \
               --config celeba.yml         \
               --exp ./runs/test           \
               --t_0 500                   \
               --n_inv_step 40             \
               --n_test_step 40            \
               --n_iter 1                  \
               --img_path imgs/celeb1.png  \
               --model_path  checkpoint/neanderthal.pth
```
- `img_path`: Path of an image to edit
- `model_path`: Finetuned model path to use

You can edit multiple images from the dataset into the CLIP-guided domain by running the following command:
```
python main.py --edit_images_from_dataset  \
               --config celeba.yml         \
               --exp ./runs/test           \
               --n_test_img 50             \
               --t_0 500                   \
               --n_inv_step 40             \
               --n_test_step 40            \
               --model_path checkpoint/neanderthal.pth
```
 

### Image Translation from Unseen Domain into Another Unseen Domain
![](imgs/app_2_unseen2unseen.png)


###  Generation of Images in Unseen Domain from Strokes
![](imgs/app_3_stroke2unseen.png)
You can tranlate images from an unseen domain to another unseen domain. (e.g. Stroke/Anime ➝ Neanderthal) using following command: 

```
python main.py --unseen2unseen          \
               --config celeba.yml      \
               --exp ./runs/test        \
               --t_0 500                \
               --bs_test 4              \
               --n_iter 10              \
               --n_inv_step 40          \
               --n_test_step 40         \
               --img_path imgs/stroke1.png \
               --model_path  checkpoint/neanderthal.pth
```
- `img_path`: Stroke image or source image in the unseen domain e.g. portrait
- `n_iter`: # of iterations of stochastic foward and generative processes to translate an unseen source image into the image in the trained domain. It's required to be larger than 8. 


## Finetuned Models Using DiffuionCLIP

We provide a [Google Drive](https://drive.google.com/drive/folders/1Uwvm_gckanyRzQkVTLB6GLQbkSBoZDZF?usp=sharing) containing several fintuned models using DiffusionCLIP.


## Additional Results

Here, we show more manipulation of real images in the diverse datasets using DiffusionCLIP where the original pretrained models
are trained on [AFHQ-Dog](https://arxiv.org/abs/1912.01865), [LSUN-Bedroom](https://www.yf.io/p/lsun) and [ImageNet](https://image-net.org/index.php), respectively.

[comment]: <> (![]&#40;imgs/more_manipulation1.png&#41;)

[comment]: <> (![]&#40;imgs/more_manipulation2.png&#41;)

[comment]: <> (![]&#40;imgs/more_manipulation3.png&#41;)

[comment]: <> (![]&#40;imgs/more_manipulation4.png&#41;)

<p align="center">

  <img src="https://github.com/submission10095/DiffusionCLIP_temp/blob/master/imgs/more_manipulation1.png" />

  <img src="https://github.com/submission10095/DiffusionCLIP_temp/blob/master/imgs/more_manipulation2.png" />

  <img src="https://github.com/submission10095/DiffusionCLIP_temp/blob/master/imgs/more_manipulation3.png" />

  <img src="https://github.com/submission10095/DiffusionCLIP_temp/blob/master/imgs/more_manipulation4.png" />

</p>
