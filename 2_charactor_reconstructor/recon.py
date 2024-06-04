import os
import json
import argparse
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from instant_nsr import datasets
from instant_nsr import systems


# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver('calc_exp_lr_decay_rate', lambda factor, n: factor**(1./n))
OmegaConf.register_new_resolver('add', lambda a, b: a + b)
OmegaConf.register_new_resolver('sub', lambda a, b: a - b)
# ======================================================= #
def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    return conf


def run_pipeline(config):
    pl.seed_everything(config.seed)
    dm = datasets.make(config.dataset.name, config.dataset)
    system = systems.make(config.system.name, config)
    
    trainer = Trainer(
        devices=1,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_false',
        **config.trainer
    )
    trainer.fit(system, datamodule=dm)
    trainer.test(system, datamodule=dm)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='reconstruction')
    parser.add_argument('--config', default='./configs/neuralangelo-ortho-wmask.yaml')
    parser.add_argument('--uid', default='0dd66be9d0534b93a092d8c4c4dfd30a', help='image uid')
    parser.add_argument('--mv_folder', default='mv', help='mv folder')
    parser.add_argument('--save_folder', default='mesh', help='save folder')
    parser.add_argument('--all', action='store_true', help='process all examples')
    parser.add_argument('--no_front_cut', action='store_true', help='disable front cut')
    parser.add_argument('--no_simplify', action='store_true', help='disable mesh simplification')
    parser.add_argument('--no_vertex_coloring', action='store_true', help='disable vertex coloring')
    parser.add_argument('--thin', action='store_true', help='enable thinning')
    parser.add_argument('--no_shear', action='store_true', help='disable shear')
    args = parser.parse_args()

    config = load_config(args.config)
    config.model.geometry.front_cutting = not args.no_front_cut
    config.model.geometry.mesh_simplify = not args.no_simplify
    config.export.vertex_coloring = not args.no_vertex_coloring
    config.export.thin = args.thin
    config.export.shear = not args.no_shear
    
    if args.all:
        with open(config.uid_list_file) as f:
            all_uids = json.load(f)
        for uid in all_uids:
            config.dataset.input_dir = os.path.join(config.data_root, uid, args.mv_folder)
            config.export.output_dir = os.path.join(config.data_root, uid, args.save_folder)
            run_pipeline(config)
    else:
        config.dataset.input_dir = os.path.join(config.data_root, args.uid, args.mv_folder)
        config.export.output_dir = os.path.join(config.data_root, args.uid, args.save_folder)
        run_pipeline(config) 

