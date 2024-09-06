import os
import yaml
import argparse
import time

import torch

from training.trainers import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--uid', help='uid of sample', default='0dd66be9d0534b93a092d8c4c4dfd30a')
    parser.add_argument('--no_mask', action='store_true', help='disable mask information')
    parser.add_argument('--no_pos', action='store_true', help='disable pos information')
    parser.add_argument('--no_edge', action='store_true', help='disable edge information')
    args = parser.parse_args()

    config_file = 'configs/config_stage2.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['job']
        config['data_root'] = os.path.join(config['root_dir'], args.uid, 'mesh', 'blender_render')
    
    log_name = 'logs_stage2'
    use_mask = not args.no_mask
    use_pos = not args.no_pos
    use_edge = not args.no_edge
    
    if use_mask:
        log_name += '_mask'
        config['generator']['args']['input_channels'] += 1

    if use_pos:
        log_name += '_pos'
        config['generator']['args']['input_channels'] += 2

    if use_edge:
        log_name += '_edge'

    log_folder = os.path.join(config['root_dir'], args.uid, 'mesh', log_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    model_logger = ModelLogger(log_folder, torch.save)
    model_logger.copy_file(config_file)
    
    # build generator and discriminator
    device = config.get('device') or 'cpu'
    # config['generator']['args']['batch_size'] = config['trainer']['batch_size']
    generator = build_model(config['generator']['type'], config['generator']['args'], device)

    opt_generator = build_optimizer(config['opt_generator']['type'], generator, config['opt_generator']['args'])
    discriminator = build_model(config['discriminator']['type'], config['discriminator']['args'], device)
    opt_discriminator = build_optimizer(config['opt_discriminator']['type'], discriminator, config['opt_discriminator']['args'])

    # build perception loss model
    perception_loss_model = build_model(config['perception_loss']['perception_model']['type'],
                                        config['perception_loss']['perception_model']['args'],
                                        device)
    perception_loss_weight = config['perception_loss']['weight']


    # now, start to train!
    # build training dataset
    trainer_config = dict(config['trainer'])
    trainer_config['testing_name_list'] = [f for f in os.listdir(config['data_root']) if not f.startswith('.')]
    trainer_config['post_dir'] = os.path.join(config['root_dir'], args.uid, 'char')

    trainer = Trainer(
        data_root=config['data_root'],
        trainer_config=trainer_config,
        opt_generator=opt_generator, 
        opt_discriminator=opt_discriminator,
        model_logger=model_logger,
        perception_loss_model=perception_loss_model,
        perception_loss_weight=perception_loss_weight,
        use_mask=use_mask,
        use_pos=use_pos, 
        use_edge=use_edge,
        device=device
    )

    start = time.time()
    trainer.train(generator, discriminator, int(config['trainer']['epochs']), log_name.replace('logs','res'), 0)
    print("Training finished, cost time: ", time.time() - start)
