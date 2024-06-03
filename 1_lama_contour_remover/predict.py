import os
import cv2
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.modules import make_generator


def load_checkpoint(config, path, map_location='cuda', strict=True):
    model = make_generator(**config.generator)
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state, strict=strict)
    return model

def move_to_device(obj, device):
    if isinstance(obj, str):
        return obj 
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        return {name: move_to_device(val, device) for name, val in obj.items()}
      
    raise ValueError(f'Unexpected type {type(obj)}')


def predict(config_path='configs/prediction/lama-fourier.yaml'):

    with open(config_path, 'r') as f:
        predict_config = OmegaConf.create(yaml.safe_load(f))
    
    device = torch.device(predict_config.device)
    save_name = predict_config.generator.kind
    checkpoint_path = os.path.join(predict_config.pretrained.path, 'models', predict_config.pretrained.generator_checkpoint)
    model = load_checkpoint(predict_config, checkpoint_path, strict=False, map_location='cpu')
    model.eval()
    model.to(device)

    dataset = make_default_val_dataset(predict_config.indir, predict_config.uid_json, **predict_config.dataset)
    for img_i in tqdm.trange(len(dataset)):
        batch = default_collate([dataset[img_i]])

        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch['predicted'] = model(batch['input'])  

        input = batch['input'][0].permute(1, 2, 0).detach().cpu().numpy()
        A = (input[:,:,0:3] * 255).astype('uint8') 
        M = input[:,:,3:4]
        alpha = (M * 255).astype('uint8')    

        CM = batch['predicted'][0][0].detach().cpu().numpy()
        CM = np.clip(CM*255, 0, 255).astype('uint8')  
        CM_path = os.path.join(predict_config.indir, batch['uid'][0], 'char/'+save_name+'_contour.png')
        cv2.imwrite(CM_path, CM)

        inpait_mask = np.maximum(CM, 255-alpha[:,:,0]).astype(np.uint8)
        A = cv2.inpaint(A, inpait_mask, 3, cv2.INPAINT_TELEA)
        inpainted_path = os.path.join(predict_config.indir, batch['uid'][0], 'char/'+save_name+'_inpainted_0.png')
        cv2.imwrite(inpainted_path, cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
        inpainted_path = os.path.join(predict_config.indir, batch['uid'][0], 'char/'+save_name+'_inpainted.png')
        A = A * M + 255 * (1-M)
        A = np.concatenate((A, alpha), 2)
        cv2.imwrite(inpainted_path, cv2.cvtColor(A, cv2.COLOR_BGRA2RGBA))


if __name__ == '__main__':
    # pix2pixhd
    # predict('configs/prediction/lama-regular.yaml')

    # ffc-resnet
    predict('configs/prediction/lama-fourier.yaml')