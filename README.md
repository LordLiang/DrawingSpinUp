# DrawingSpinUp
Official code for DrawingSpinUp

## Step-0: Environment Setup

## Step-1: Contour Removal
### Download
You can download our processed character drawings (a tiny subset of [Amateur Drawings Dataset](https://github.com/facebookresearch/AnimatedDrawings)) from [here](https://portland-my.sharepoint.com/:u:/g/personal/jzhou67-c_my_cityu_edu_hk/EXwpR2S7kYlMnFIFFdYGdOYBSNUfu9tA-s2c50XfWaCTuA?e=jSGv5V) and pretrained contour removal models from [here](https://portland-my.sharepoint.com/:u:/g/personal/jzhou67-c_my_cityu_edu_hk/Ed6BaAAWgIhGqIMjaju_v4kB_K-DIFGu1bQ7zM3CbQMrTw?e=KaltGi).
### Inference
We use [FFC-ResNet](https://github.com/advimman/lama) as backbone to predict the contour region of a given character drawing. 
For model training, you can refer to the original repo.
For training image rendering, see [1_lama_contour_remover/bicar_render_codes](1_lama_contour_remover/bicar_render_codes) which are borrowed from [Wonder3D](https://github.com/xxlong0/Wonder3D/tree/main/render_codes).
Here we focus on inference:
```
cd 1_lama_contour_remover/
python predict.py
cd ..
```

## Step-2: Textured Character Generation
### Multi-view Images Generation

### Textured Character Reconstruction
### Shape Thinning (Optional)

## Step-3: Stylized Contour Restoration
### Keyframe Pair Prepration
### Training
### Inference

