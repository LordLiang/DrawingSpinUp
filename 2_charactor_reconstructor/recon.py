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


def recon(uid, config):
    config.dataset.uid = uid
    config.dataset.input_dir = os.path.join(config.dataset.data_root, uid, 'mv')
    config.export.output_dir = os.path.join(config.dataset.data_root, uid, 'mesh')
    config.dataset.load_front_mask = config.model.geometry.front_cutting
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
    parser.add_argument('--all', action='store_true', help='process all examples')
    args = parser.parse_args()
    config = load_config(args.config)

    with open(config.dataset.thinning_uid_list_file) as f:
        thinning_uids = json.load(f)
    
    if args.all:
        with open(config.dataset.uid_list_file) as f:
            all_uids = json.load(f)
        for uid in all_uids:
            if not uid in thinning_uids:
                config.export.thinning = False
            recon(uid, config)
    else:
        if not args.uid in thinning_uids:
            config.export.thinning = False
        recon(args.uid, config) 
