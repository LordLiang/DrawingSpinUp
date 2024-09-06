import os
import json
import argparse
import numpy as np
from PIL import Image, ImageOps
from einops import rearrange
from omegaconf import OmegaConf
import onnxruntime as ort

import torch
from diffusers import DiffusionPipeline

from mvdiffusion.data.single_image_dataset import SingleImageDataset


weight_dtype = torch.float16
model_path = 'dis_pretrained/isnet_dis.onnx'
session = ort.InferenceSession(model_path)

    
def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    return conf


def load_wonder3d_pipeline(config):
    pipeline = DiffusionPipeline.from_pretrained(
    config.pretrained_model_name_or_path,
    custom_pipeline='flamehaze1115/wonder3d-pipeline',
    torch_dtype=weight_dtype
    )

    pipeline.unet.enable_xformers_memory_efficient_attention()
    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    return pipeline


def prepare_data(single_image, config):
    dataset = SingleImageDataset(single_image=single_image, **config.validation_dataset)
    return dataset[0]


def tensor2pil(tensor):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(ndarr)


def mv(uid, pipeline, config, write_image=True):
    print(uid)
    config.output_dir = os.path.join(config.data_root, uid, args.save_folder)
    img_fn = os.path.join(config.data_root, uid, args.img_fn)
    if not os.path.exists(img_fn):
        img_fn = os.path.join(config.data_root, uid, 'char/texture.png')
    single_image = Image.open(img_fn)
    if uid in ['0b39d3ae37ee430dbe721cdcc40e270c',
               'b2f0411a69b149088282f262b77970a7',
               '7d64695e10134f4883cf0f646c21ed30']:
        single_image = add_gray(single_image)
    
    batch = prepare_data(single_image, config)
    pipeline.set_progress_bar_config(disable=True)
    seed = int(config.seed)
    generator = torch.Generator(device=pipeline.unet.device).manual_seed(seed)

    # repeat  (2B, Nv, 3, H, W)
    imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0).to(weight_dtype)

    # (2B, Nv, Nce)
    camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0).to(weight_dtype)
    task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0).to(weight_dtype)
    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1).to(weight_dtype)

    # (B*Nv, 3, H, W)
    imgs_in = rearrange(imgs_in, "Nv C H W -> (Nv) C H W")
    out = pipeline(
        imgs_in,
        camera_embeddings,
        generator=generator,
        output_type='pt',
        num_images_per_prompt=1,
        **config.pipe_validation_kwargs,
    ).images

    bsz = out.shape[0] // 2
    normals_pred = out[:bsz]
    images_pred = out[bsz:]
    VIEWS = config.views
    num_views = len(VIEWS)
    res = config.resolution

    if write_image:
        normal_dir = os.path.join(config.output_dir, "normal")
        color_dir = os.path.join(config.output_dir, "color")
        mask_dir = os.path.join(config.output_dir, "mask")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        mask_front = single_image.split()[-1]
        mask_back = ImageOps.mirror(mask_front)

        for j in range(num_views):
            view = VIEWS[j]
            normal = tensor2pil(normals_pred[j]).resize(res, Image.LANCZOS)
            color = tensor2pil(images_pred[j]).resize(res, Image.LANCZOS)

            if view == 'front':
                mask = mask_front.resize(res, Image.NEAREST)
            elif view == 'back':
                mask = mask_back.resize(res, Image.NEAREST)
            else:
                if uid in ['01522711d3b642ddbfb506307a007990', 
                           '1a2fd47487a24c4c84f2c7d0f7d35147',
                           '1f1654afb5aa4f8daa5db9a96351c226',
                           'd77b86a6b2024cffa36f010e72c0a2af'
                           ]:
                    mask = remove_background(session, normal)
                else:
                    mask = remove_background(session, color)

            normal.save(os.path.join(normal_dir, f"{view}.png"))
            color.save(os.path.join(color_dir, f"{view}.png"))
            mask.save(os.path.join(mask_dir, f"{view}.png"))


def normalize(image, mean, std):
    """Normalize a numpy image with mean and standard deviation."""
    return (image / 255.0 - mean) / std


def remove_background(session, image_pil):
    # Normalize the image using NumPy
    im = np.array(image_pil, dtype=np.float32)  # Convert to float
    im = normalize(im, mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
    im = np.transpose(im, (2, 0, 1))  # CHW format
    im = np.expand_dims(im, axis=0)  # Add batch dimension

    # Run inference
    im = im.astype(np.float32)  
    ort_inputs = {session.get_inputs()[0].name: im}
    ort_outs = session.run(None, ort_inputs)
        
    # Process the model output
    result = ort_outs[0][0][0]  # Assuming single output and single batch
    result = np.clip(result, 0, 1)  # Assuming you want to clip the result to [0, 1]
    result = Image.fromarray((result * 255).astype(np.uint8))
    return result


def add_gray(img):
    img = np.array(img, dtype=np.float32)
    rgb = img[:,:,0:3] * 0.8
    mask = img[:,:,3:4]/255.
    img[:,:,0:3] = rgb * mask + 255 * (1-mask)
    return Image.fromarray(img.astype(np.uint8))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='mv generation')
    parser.add_argument('--config', default='./configs/mvdiffusion-joint-ortho-6views.yaml')
    parser.add_argument('--uid', default='0dd66be9d0534b93a092d8c4c4dfd30a', help='image uid')
    parser.add_argument('--img_fn', default='char/ffc_resnet_inpainted.png', help='image filename')
    parser.add_argument('--save_folder', default='mv', help='save folder')
    parser.add_argument('--all', action='store_true', help='process all examples')
    args = parser.parse_args()
    
    config = load_config(args.config)
    pipeline = load_wonder3d_pipeline(config)
    torch.set_grad_enabled(False)
    pipeline.to(f'cuda:0')

    if args.all:
        with open(config.uid_list_file) as f:
            all_uids = json.load(f)
        for uid in all_uids:
            mv(uid, pipeline, config)
    else:
        mv(args.uid, pipeline, config)