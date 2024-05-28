import logging
import os

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from saicinpainting.training.trainers import make_training_model
from saicinpainting.utils import handle_ddp_subprocess

LOGGER = logging.getLogger(__name__)


@handle_ddp_subprocess()
@hydra.main(config_path='configs/training', config_name='lama-fourier.yaml')
def main(config: OmegaConf):
    save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    config.visualizer.outdir = os.path.join(save_dir, config.visualizer.outdir)

    LOGGER.info(OmegaConf.to_yaml(config))
    OmegaConf.save(config, os.path.join(save_dir, 'config.yaml'))

    checkpoints_dir = os.path.join(save_dir, 'models')
    os.makedirs(checkpoints_dir, exist_ok=True)

    training_model = make_training_model(config)
    trainer_kwargs = OmegaConf.to_container(config.trainer.kwargs, resolve=True)
    trainer = Trainer(
        # there is no need to suppress checkpointing in ddp, because it handles rank on its own
        callbacks=ModelCheckpoint(dirpath=checkpoints_dir, **config.trainer.checkpoint_kwargs),
        default_root_dir=os.getcwd(),
        **trainer_kwargs
    )
    trainer.fit(training_model)



if __name__ == '__main__':
    main()
