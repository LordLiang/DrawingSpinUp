import os
import yaml
import argparse
from PIL import Image
import numpy as np
import time

import torch

from training.data import DatasetFullImages
from training.custom_transforms import to_image_space
from training.trainers import build_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--uid', help='uid of sample', default='0dd66be9d0534b93a092d8c4c4dfd30a')
    parser.add_argument('--no_mask', action='store_true', help='disable mask information')
    parser.add_argument('--no_pos', action='store_true', help='disable dense corresponding')
    parser.add_argument('--checkpoint_id', type=int, default=99999)
    args = parser.parse_args()

    config_file = 'configs/config_stage1.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['job']
        config['data_root'] = os.path.join(config['root_dir'], args.uid, 'mesh/blender_render')

    log_name = 'logs_stage1'
    use_mask = not args.no_mask
    use_pos = not args.no_pos
    pre_dir = config['trainer']['pre_dir']

    if use_mask:
        log_name += '_mask'
        config['generator']['args']['input_channels'] += 1
    
    if use_pos:
        log_name += '_pos'
        config['generator']['args']['input_channels'] += 2

    # build generator and discriminator
    device = config.get('device') or 'cpu'
    generator = build_model(config['generator']['type'], config['generator']['args'], device)
    checkpoint_path = os.path.join(config['root_dir'], args.uid, 'mesh', log_name, 'model_%05d.pth'%args.checkpoint_id)
    print(checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    
    testing_name_list = [f for f in os.listdir(config['data_root']) if not f.startswith('.')]
    result_folder = log_name.replace('logs', 'res')

    start = time.time()
    for test_name in testing_name_list:
        data_root = os.path.join(config['data_root'], test_name)
        dataset = DatasetFullImages(data_root, pre_dir, use_mask, use_pos, False)
        imloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, drop_last=False)  # num_workers=4
        
        with torch.no_grad():
            for i, batch in enumerate(imloader):
                batch = {k: batch[k].to(device) if isinstance(batch[k], torch.Tensor) else batch[k]
                        for k in batch.keys()}
                gan_output = generator(batch['pre'])[0]
                
                gt_test_ganoutput_path = os.path.join(data_root, result_folder)
                os.makedirs(gt_test_ganoutput_path, exist_ok=True)

                img = to_image_space(gan_output.cpu().numpy()).transpose(1, 2, 0)
                alpha = (batch['pre_mask'][0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                img = np.concatenate((img, alpha), 2)
                Image.fromarray(img).save(os.path.join(gt_test_ganoutput_path, batch['file_name'][0]))

    end = time.time()
    print(end-start)
    print("Testing finished", flush=True)
