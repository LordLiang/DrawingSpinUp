import os
import yaml
import argparse
from PIL import Image
import numpy as np

import torch

import models as m
from data import DatasetFullImages
from custom_transforms import to_image_space


def build_model(model_type, args, device):
    model = getattr(m, model_type)(**args)
    return model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='traning type', default='ric')
    parser.add_argument('--uid', help='uid of sample', default='0dd66be9d0534b93a092d8c4c4dfd30a')
    parser.add_argument('--pos', action='store_true', help='use dense corresponding')
    parser.add_argument('--edge', action='store_true', help='use edge information')
    parser.add_argument('--checkpoint_id', type=int, help='checkpoint id', default=99999)
    args = parser.parse_args()

    if args.type == 'baseline':
        config_file = 'configs/config_baseline.yaml'
        log_name = 'logs_baseline'
    elif args.type == 'ric':
        config_file = 'configs/config_ric.yaml'
        log_name = 'logs_ric'
    else:
        quit()
    
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['job']
        config['data_root'] = os.path.join(config['root_dir'], args.uid, 'animation/blender_render')

    if args.pos:
        log_name += '_pos'
        config['generator']['args']['input_channels'] += 2
    if args.edge:
        log_name += '_edge'
        config['generator']['args']['input_channels'] += 1

    # build generator and discriminator
    device = config.get('device') or 'cpu'
    config['generator']['args']['batch_size'] = config['trainer']['batch_size']
    generator = build_model(config['generator']['type'], config['generator']['args'], device)
    checkpoint_path = os.path.join(config['root_dir'], args.uid, 'ours', log_folder, 'model_%05d.pth'%args.checkpoint_id)
    print(checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    
    testing_name_list = [f for f in os.listdir(config['data_root']) if not f.startswith('.')]
    result_folder = log_name.replace('logs', 'res')
    save_alpha = False
    for test_name in testing_name_list:
        if test_name in ['rest_pose', 'rest_rotate']:
            continue
        data_root = os.path.join(config['data_root'], test_name)
        dataset = DatasetFullImages(data_root, args.pos, args.edge)
        imloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, drop_last=False)  # num_workers=4
        
        with torch.no_grad():
            for i, batch in enumerate(imloader):
                batch = {k: batch[k].to(device) if isinstance(batch[k], torch.Tensor) else batch[k]
                        for k in batch.keys()}
                gan_output = generator(batch['pre'])[0]
                
                gt_test_ganoutput_path = os.path.join(data_root, result_folder)
                os.makedirs(gt_test_ganoutput_path, exist_ok=True)

                img = to_image_space(gan_output.cpu().numpy()).transpose(1, 2, 0)
                if save_alpha:
                    alpha = (batch['pre_mask'][0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    img = np.concatenate((img, alpha), 2)
                Image.fromarray(img).save(os.path.join(gt_test_ganoutput_path, batch['file_name'][0]))

    print("Testing finished", flush=True)
