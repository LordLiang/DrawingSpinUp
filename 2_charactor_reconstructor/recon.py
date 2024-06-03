import os
import json
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

    #################################################
    # start!!!
    config_file = "./configs/neuralangelo-ortho-wmask.yaml"
    config = load_config(config_file)

    with open(config.uid_list_file) as f:
        all_uids = json.load(f)

    for uid in all_uids:
        uid = '0a614ae4e45c424880aab5ff92056c08'
        config.dataset.input_dir = os.path.join(config.data_root, uid, 'mv')
        config.dataset.output_dir = os.path.join(config.data_root, uid, 'mesh')
        run_pipeline(config)
        quit()
