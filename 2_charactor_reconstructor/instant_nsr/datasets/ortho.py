import os
import numpy as np
from PIL import Image
from glob import glob
import cv2

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

import pytorch_lightning as pl

from instant_nsr import datasets
from instant_nsr.models.ray_utils import get_ortho_ray_directions_origins
from instant_nsr.utils.misc import get_rank


def camNormal2worldNormal(rot_c2w, camNormal):
    H,W,_ = camNormal.shape
    normal_img = np.matmul(rot_c2w[None, :, :], camNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])
    return normal_img


def img2normal(img):
    return (img/255.)*2-1


def normal2img(normal):
    return np.uint8((normal*0.5+0.5)*255)


def RT_opengl2opencv(RT):
    R = RT[:3, :3]
    t = RT[:3, 3]
    R_bcam2cv = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R_world2cv = R_bcam2cv @ R
    t_world2cv = R_bcam2cv @ t
    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)
    return RT


def normal_opengl2opencv(normal):
    H,W,C = np.shape(normal)
    R_bcam2cv = np.array([1, -1, -1], np.float32)
    normal_cv = normal * R_bcam2cv[None, None, :]
    return normal_cv


def inv_RT(RT):
    RT_h = np.concatenate([RT, np.array([[0,0,0,1]])], axis=0)
    RT_inv = np.linalg.inv(RT_h)
    return RT_inv[:3, :]


def load_a_prediction(mv_dir, imSize, view_types, cam_pose_dir=None, normal_system='front'):
    all_images = []
    all_normals = []
    all_normals_world = []
    all_masks = []
    all_poses = []
    all_w2cs = []
    directions = []
    ray_origins = []

    RT_front = np.loadtxt(glob(os.path.join(cam_pose_dir, '*_%s_RT.txt'%( 'front')))[0])   # world2cam matrix
    RT_front_cv = RT_opengl2opencv(RT_front)   # convert normal from opengl to opencv
    for idx, view in enumerate(view_types):
        normal = np.array(Image.open(os.path.join(mv_dir, 'normal', '%s.png'%(view))))
        normal = img2normal(normal)
        RT = np.loadtxt(os.path.join(cam_pose_dir, '000_%s_RT.txt'%( view)))  # world2cam matrix
        mask = np.array(Image.open(os.path.join(mv_dir, 'mask', '%s.png'%(view))))
        normal[mask==0] = [0,0,0]
        mask = mask > 127
        
        all_masks.append(mask)
        RT_cv = RT_opengl2opencv(RT)   # convert normal from opengl to opencv
        all_poses.append(inv_RT(RT_cv))   # cam2world
        all_w2cs.append(RT_cv)

        normal_cam_cv = normal_opengl2opencv(normal)
        if normal_system == 'front':
            # print("the loaded normals are defined in the system of front view")
            normal_world = camNormal2worldNormal(inv_RT(RT_front_cv)[:3, :3], normal_cam_cv)
        elif normal_system == 'self':
            # print("the loaded normals are in their independent camera systems")
            normal_world = camNormal2worldNormal(inv_RT(RT_cv)[:3, :3], normal_cam_cv)
        all_normals.append(normal_cam_cv)
        all_normals_world.append(normal_world)

        origins, dirs = get_ortho_ray_directions_origins(W=imSize[0], H=imSize[1])
        ray_origins.append(origins)
        directions.append(dirs)

        image = np.array(Image.open(os.path.join(mv_dir, 'color', '%s.png'%(view))))
        all_images.append(image)

    return np.stack(all_images), np.stack(all_masks), np.stack(all_normals), np.stack(all_normals_world), \
        np.stack(all_poses), np.stack(all_w2cs), np.stack(ray_origins), np.stack(directions)


class OrthoDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()
        self.mv_dir = self.config.input_dir
        self.root_dir = self.mv_dir.split('mv')[0]
        self.imSize = self.config.imSize
        self.img_wh = [self.imSize[0], self.imSize[1]]
        self.w = self.img_wh[0]
        self.h = self.img_wh[1]
        self.has_mask = True

        if self.config.uid in ['025dc91b146d4f57bd114e07165ff7bd',
                               'b03fed9c34f64114a62c7a963fa804e5',
                               'e91d8a6d3aa444f9b10f3a14a6e0a287'
                               ]:
            self.view_types = ['front', 'back']
            view_weights = [1.0, 1.0]
        elif self.config.uid in ['b32e37e2f0354f569ea9265d753891f7',
                                 'b718c3fb937a416b9fe49ff984a1504e',
                                 'd12bed5708ed42f2b615b7911c0291fa',
                                 'd2f443e21595431f9f2cd580f291f51b']:
            self.view_types = ['front', 'front_right', 'back', 'front_left']
            view_weights = [1.0, 1.0, 1.0, 1.0]
        else:
            self.view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
            view_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.view_weights = torch.from_numpy(np.array(view_weights)).float().to(self.rank).view(-1)
        self.view_weights = self.view_weights.view(-1,1,1).repeat(1, self.h, self.w)
        self.cam_pose_dir = self.config.cam_pose_dir
            
        self.images_np, self.masks_np, self.normals_cam_np, self.normals_world_np, \
            self.pose_all_np, self.w2c_all_np, self.origins_np, self.directions_np = load_a_prediction(
                self.mv_dir, self.imSize, self.view_types, self.cam_pose_dir)

        self.all_c2w = torch.from_numpy(self.pose_all_np)
        self.all_masks = torch.from_numpy(self.masks_np)
        self.all_normals_world = torch.from_numpy(self.normals_world_np)
        self.origins = torch.from_numpy(self.origins_np)
        self.directions = torch.from_numpy(self.directions_np)
        
        self.directions = self.directions.float().to(self.rank)
        self.origins = self.origins.float().to(self.rank)
        self.all_c2w, self.all_masks, self.all_normals_world = \
            self.all_c2w.float().to(self.rank), \
            self.all_masks.float().to(self.rank), \
            self.all_normals_world.float().to(self.rank)
        
        self.all_images = torch.from_numpy(self.images_np) / 255.
        self.all_images = self.all_images.float().to(self.rank)
    
        if self.config.load_front_mask:
            front_mask_filepath = os.path.join(self.root_dir, 'char/mask.png')
            self.front_mask = cv2.imread(front_mask_filepath, 0)
            self.front_mask = cv2.rotate(self.front_mask, cv2.ROTATE_90_CLOCKWISE)
        else:
            self.front_mask = None
        

class OrthoDataset(Dataset, OrthoDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_masks)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class OrthoIterableDataset(IterableDataset, OrthoDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('ortho')
class OrthoDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = OrthoIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = OrthoDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = OrthoDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = OrthoDataset(self.config, 'train')    

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
