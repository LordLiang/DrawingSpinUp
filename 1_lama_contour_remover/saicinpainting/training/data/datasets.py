import logging
import os
import json
from PIL import Image

from omegaconf import open_dict, OmegaConf
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset
from saicinpainting.training.data.aug import get_params, get_transform, get_ABC

LOGGER = logging.getLogger(__name__)


class InpaintingBiCarDataset(Dataset):
    def __init__(self, indir, uid_json, mode='val'):
        self.datadir = indir
        with open(uid_json) as f:
            self.uids = json.load(f)
        if mode == 'train':
            self.uids = self.uids[0:1200]
        else:
            self.uids = self.uids[1200:]

    def __len__(self):
        return len(self.uids) * 6


    def __getitem__(self, index):
        uid = self.uids[index//6]
        img_fn = os.path.join(self.datadir, uid, 'rgba.png')
        svg_fn = os.path.join(self.datadir, uid, f"{index%6:03d}_" + 'contour0001.svg')
        A, M, CM = get_ABC(img_fn, svg_fn)

        # apply the same transform to A, B, C
        transform_params = get_params(A.size, crop_size=512, load_size=572)
        rgb_transform = get_transform(transform_params, num_channels=3, crop_size=512, load_size=572, no_flip=False)
        mask_transform = get_transform(transform_params, num_channels=1, crop_size=512, load_size=572, no_flip=False)

        A = rgb_transform(A)
        M = mask_transform(M)
        CM = mask_transform(CM)
        input = torch.cat([A, M], dim=1)
        return dict(input=input, gt=CM)


class InpaintingDrawingsDataset(Dataset):
    def __init__(self, datadir, uid_json):
        self.datadir = datadir
        with open(uid_json) as f:
            self.uids = json.load(f)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        uid = self.uids[index]
        img_fn = os.path.join(self.datadir, uid, 'char/texture.png')
        img = Image.open(img_fn)
        rgba_transform = get_transform(num_channels=4)
        input = rgba_transform(img)
        return dict(input=input, uid=uid)
        

def make_default_train_dataloader(indir, uid_json, kind='default', 
                                  dataloader_kwargs=None, ddp_kwargs=None, **kwargs):
    dataset = InpaintingBiCarDataset(indir, uid_json, 'train')

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    is_dataset_only_iterable = kind in ('default_web',)

    if ddp_kwargs is not None and not is_dataset_only_iterable:
        dataloader_kwargs['shuffle'] = False
        dataloader_kwargs['sampler'] = DistributedSampler(dataset, **ddp_kwargs)

    if is_dataset_only_iterable and 'shuffle' in dataloader_kwargs:
        with open_dict(dataloader_kwargs):
            del dataloader_kwargs['shuffle']

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_default_val_dataset(indir, uid_json, kind='default', **kwargs):
    if OmegaConf.is_list(indir) or isinstance(indir, (tuple, list)):
        return ConcatDataset([
            make_default_val_dataset(idir, kind=kind, **kwargs) for idir in indir 
        ])

    LOGGER.info(f'Make val dataloader {kind} from {indir}')
    if '3DBiCar' in indir:
        dataset = InpaintingBiCarDataset(indir, uid_json, 'val')
    elif 'AnimatedDrawings' in indir:
        dataset = InpaintingDrawingsDataset(indir, uid_json)
    return dataset


def make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs):
    dataset = make_default_val_dataset(*args, **kwargs)

    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader
