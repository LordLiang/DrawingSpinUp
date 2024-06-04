# DrawingSpinUp
Official code for DrawingSpinUp

## Environment Setup

Hardware: 
  - All experiments are run on a single RTX 2080Ti GPU.

Setup environment:
  - Python 3.8.0
  - PyTorch 1.13.1
  - Cuda Toolkit 11.6
  - Ubuntu 18.04

Clone this repository:

```sh
git clone https://github.com/LordLiang/DrawingSpinUp.git

```

Install the required packages:

```sh
conda create -n drawingspinup python=3.8
conda activate drawingspinup
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
# tiny-cuda-nn
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# python-mesh-raycast
git clone https://github.com/cprogrammer1994/python-mesh-raycast
cd python-mesh-raycast
python setup.py develop
```


## Step-1: Contour Removal
### Download
You can download our processed character drawings from [preprocessed.zip](https://portland-my.sharepoint.com/:u:/g/personal/jzhou67-c_my_cityu_edu_hk/EWi-CdpGraRMhbqvc7Fq9k0BulcK2or_9fjaEuWVAi97Dw?e=Sj018E) (a tiny subset of [Amateur Drawings Dataset](https://github.com/facebookresearch/AnimatedDrawings)) and pretrained contour removal models from [experiments.zip](https://portland-my.sharepoint.com/:u:/g/personal/jzhou67-c_my_cityu_edu_hk/Ed6BaAAWgIhGqIMjaju_v4kB_K-DIFGu1bQ7zM3CbQMrTw?e=KaltGi).
```sh
cd DrawingSpinUp
mkdir dataset
cd dataset/AnimatedDrawings
# put preprocessed.zip here
unzip preprocessed.zip
cd ..
cd 1_lama_contour_remover
# put experiments.zip here
unzip experiments.zip
cd ..
```
Of course you can prepare your own image: a 512x512 character drawing 'texture.png' with its foreground mask 'mask.png'. 
### Inference
We use [FFC-ResNet](https://github.com/advimman/lama) as backbone to predict the contour region of a given character drawing. 
For model training, you can refer to the original repo.
For training image rendering, see [1_lama_contour_remover/bicar_render_codes](1_lama_contour_remover/bicar_render_codes) which are borrowed from [Wonder3D](https://github.com/xxlong0/Wonder3D/tree/main/render_codes).
Here we focus on inference:
```sh
cd 1_lama_contour_remover
python predict.py
cd ..
```
## Step-2: Textured Character Generation
Here let us take 0dd66be9d0534b93a092d8c4c4dfd30a as an example. You can try your own image.

```sh
cd 2_charactor_reconstructor
# multi-view image generation
python mv.py --uid 0dd66be9d0534b93a092d8c4c4dfd30a
# textured character reconstruction
python recon.py --uid 0dd66be9d0534b93a092d8c4c4dfd30a
# shape thinning (optional)
python thin.py --uid 0dd66be9d0534b93a092d8c4c4dfd30a
cd ..
```

## Step-3: Stylized Contour Restoration
### Keyframe Pair Prepration

### Training
### Inference

