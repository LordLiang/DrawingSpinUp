import os
import yaml
import argparse
import numpy as np

import torch
import torch.optim as optim
import shutil

import training.models as m
from training.trainers import Trainer


class ModelLogger(object):
    def __init__(self, log_dir, save_func):
        self.log_dir = log_dir
        self.save_func = save_func

    def save(self, model, epoch, isGenerator):
        if isGenerator:
            new_path = os.path.join(self.log_dir, "model_%05d.pth" % epoch)
        else:
            new_path = os.path.join(self.log_dir, "disc_%05d.pth" % epoch)
        self.save_func(model.state_dict(), new_path)

    def copy_file(self, source):
        shutil.copy(source, self.log_dir)


def build_model(model_type, args, device):
    model = getattr(m, model_type)(**args)
    return model.to(device)


def build_optimizer(opt_type, model, args):
    args['params'] = model.parameters()
    opt_class = getattr(optim, opt_type)
    return opt_class(**args)


def build_loggers(log_folder):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    model_logger = ModelLogger(log_folder, torch.save)
    scalar_logger = Logger(log_folder)
    return scalar_logger, model_logger


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='traning type', default='ric')
    parser.add_argument('--uid', help='uid of sample', default='0dd66be9d0534b93a092d8c4c4dfd30a')
    parser.add_argument('--pos', action='store_true', help='use dense corresponding')
    parser.add_argument('--edge', action='store_true', help='use edge information')
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

    log_folder = os.path.join(config['root_dir'], args.uid, 'animation', log_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    model_logger = ModelLogger(log_folder, torch.save)
    model_logger.copy_file(config_file)
    
    # build generator and discriminator
    device = config.get('device') or 'cpu'
    config['generator']['args']['batch_size'] = config['trainer']['batch_size']
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
    trainer_config['dir_post'] = os.path.join(config['root_dir'], args.uid, 'char')

    trainer = Trainer(
        data_root=config['data_root'],
        trainer_config=trainer_config,
        opt_generator=opt_generator, 
        opt_discriminator=opt_discriminator,
        model_logger=model_logger,
        perception_loss_model=perception_loss_model,
        perception_loss_weight=perception_loss_weight,
        use_pos=args.pos, 
        use_edge=args.edge,
        device=device
    )

    trainer.train(generator, discriminator, int(config['trainer']['epochs']), log_name.replace('logs','res'), 0)
    print("Training finished", flush=True)
