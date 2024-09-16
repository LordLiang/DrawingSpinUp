import os
import json
from PIL import Image
from omegaconf import open_dict, OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset
from saicinpainting.training.data.aug import get_transform, get_params, get_data


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
        img, mask, gt = get_data(img_fn, svg_fn)

        # apply the same transform
        transform_params = get_params(img.size, crop_size=512, load_size=572)
        rgb_transform = get_transform(transform_params, num_channels=3, crop_size=512, load_size=572, no_flip=False)
        mask_transform = get_transform(transform_params, num_channels=1, crop_size=512, load_size=572, no_flip=False)

        img = rgb_transform(img)
        mask = mask_transform(mask)
        gt = mask_transform(gt)
        input = torch.cat([img, mask], dim=0)

        return dict(input=input, gt=gt)


class InpaintingDrawingsDataset(Dataset):
    def __init__(self, datadir, uid_json):
        self.datadir = datadir
        with open(uid_json) as f:
            self.uids = json.load(f)
            self.uids.remove('00d9710f5e9d438db188d78b64b4a1f4')
            self.uids.remove('2a8d91dfc5a7422d9f962d3f02e3b4c0')

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        uid = self.uids[index]
        img_fn = os.path.join(self.datadir, uid, 'char/texture.png')
        img = Image.open(img_fn)
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, (0, 0), img)

        if img.mode == 'RGBA':
            mask = img.split()[-1]
        else:
            mask_fn = os.path.join(self.datadir, uid, 'char/mask.png')
            mask = Image.open(mask_fn)
        
        rgb_transform = get_transform(num_channels=3)
        mask_transform = get_transform(num_channels=1)
        rgb_img = rgb_transform(rgb_img)
        mask = mask_transform(mask)
        input = torch.cat([rgb_img, mask], dim=0)
        
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

