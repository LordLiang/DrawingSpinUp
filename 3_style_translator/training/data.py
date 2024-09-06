import os
from PIL import Image, ImageFilter

import torch.utils.data

from training.custom_transforms import *


################################
# Dataset full-images for inference
################################
class DatasetFullImages(torch.utils.data.Dataset):
    def __init__(self, data_root, pre_dir, use_mask=False, use_pos=False, use_edge=False):
        super(DatasetFullImages, self).__init__()
        self.data_root = data_root
        self.pre_dir = pre_dir
        self.use_mask, self.use_pos, self.use_edge = use_mask, use_pos, use_edge
        self.fnames = sorted(os.listdir(os.path.join(self.data_root, 'color')))
        self.transform = build_transform()
        self.mask_transform = build_mask_transform()


    def __getitem__(self, item):
        # get an image that is NOT stylized and its stylized counterpart
        fileName = self.fnames[item]
        result = {'file_name': self.fnames[item]}
        pre_color = Image.open(os.path.join(self.data_root, self.pre_dir, fileName))
        mask = pre_color.split()[-1]
        if self.use_pos:
            pre_pos = Image.open(os.path.join(self.data_root, 'pos', fileName))
        if self.use_edge:
            # overlap edge on color for stage2
            pre_edge = Image.open(os.path.join(self.data_root, 'edge', fileName))
            pre_color = overlap_edge_on_img(pre_edge, pre_color)
  
        pre_feature = [self.transform(pre_color)]
        if self.use_mask:
            pre_feature.append(self.mask_transform(mask))
        if self.use_pos:
            pre_feature.append(self.transform(pre_pos)[0:2])# just use X&Y

        pre_tensor = torch.cat(pre_feature, 0)

        result['pre'] = pre_tensor
        result['pre_mask'] = self.mask_transform(mask)

        return result


    def __len__(self):
        return int(len(self.fnames))


#####
# Default "patch" dataset, used for training
#####
class DatasetPatches_M(torch.utils.data.Dataset):
    def __init__(self, data_root, pre_dir, post_dir, post_name, patch_size, use_mask=False, use_pos=False, use_edge=False):
        super(DatasetPatches_M, self).__init__()
        self.data_root, self.pre_dir = data_root, pre_dir
        self.post_dir, self.post_name = post_dir, post_name
        self.patch_size = patch_size
        self.use_mask, self.use_pos, self.use_edge = use_mask, use_pos, use_edge

        self.transform = build_transform()
        self.mask_transform = build_mask_transform()

        self.images_pre = None
        self.images_post = None
        self.images_mask = None
        self.valid_indices = None
        self.valid_indices_left = None

        self.load_image()

    def load_image(self, fileName='0001.png'):
        
        self.pre_color = Image.open(os.path.join(self.data_root, self.pre_dir, fileName))
        self.mask = self.pre_color.split()[-1]
        if os.path.exists(os.path.join(self.post_dir, self.post_name + '.png')):
            self.post_name == 'texture_with_bg'
        self.post_color = Image.open(os.path.join(self.post_dir, self.post_name + '.png'))
        self.post_color = replace_alpha(self.post_color, self.mask)
        
        if self.use_pos:
            self.pre_pos = Image.open(os.path.join(self.data_root, 'pos', fileName))
        if self.use_edge:
            # overlap edge on color for stage2
            pre_edge = Image.open(os.path.join(self.data_root, 'edge', fileName))
            self.pre_color = overlap_edge_on_img(pre_edge, self.pre_color)
            
            self.pre_color = cat_img(self.pre_color)
            self.mask = cat_mask(self.mask)
            self.pre_pos = cat_img(self.pre_pos)
            self.post_color = cat_img(self.post_color)

        self.post_color = white_bg(self.post_color)
        self.preprocessing_image()

    def preprocessing_image(self):
        pre_color = self.pre_color
        post_color = self.post_color

        mask_tensor = self.mask_transform(self.mask)
        post_tensor = self.transform(post_color)
        
        pre_feature = [self.transform(pre_color)]
        if self.use_mask:
            pre_feature.append(mask_tensor)
        if self.use_pos:
            pre_feature.append(self.transform(self.pre_pos)[0:2]) # just use X&Y

        self.images_pre = torch.cat(pre_feature, 0)
        self.images_post = post_tensor
        self.images_mask = mask_tensor

        valid_mask = self.mask.filter(ImageFilter.MaxFilter(7))
        valid_mask = self.mask_transform(valid_mask)

        self.valid_indices = valid_mask.squeeze().nonzero(as_tuple=False)
        self.valid_indices_left = list(range(0, len(self.valid_indices)))      


    def cut_patch(self, im, midpoint, size):
        hs = size // 2
        hn = max(0, midpoint[0] - hs)
        hx = min(midpoint[0] + hs, im.size()[1] - 1)
        xn = max(0, midpoint[1] - hs)
        xx = min(midpoint[1] + hs, im.size()[2] - 1)

        p = im[:, hn:hx, xn:xx]
        if p.size()[1] != size or p.size()[2] != size:
            r = torch.zeros((p.size()[0], size, size))
            r[:, 0:p.size()[1], 0:p.size()[2]] = p
            p = r

        return p


    def cut_patches(self, midpoint, midpoint_r, size):
        patch_pre = self.cut_patch(self.images_pre, midpoint, size)
        patch_pre_mask = self.cut_patch(self.images_mask, midpoint, size)
        patch_post = self.cut_patch(self.images_post, midpoint, size)
        patch_random = self.cut_patch(self.images_post, midpoint_r, size)
        patch_random_mask = self.cut_patch(self.images_mask, midpoint_r, size)

        return patch_pre, patch_pre_mask, patch_post, patch_random, patch_random_mask


    def __getitem__(self, item=None):

        midpoint_id = np.random.randint(0, len(self.valid_indices_left))
        midpoint_r_id = np.random.randint(0, len(self.valid_indices))
        midpoint = self.valid_indices[self.valid_indices_left[midpoint_id], :].squeeze()
        midpoint_r = self.valid_indices[midpoint_r_id, :].squeeze()

        del self.valid_indices_left[midpoint_id]
        if len(self.valid_indices_left) < 1:
            self.valid_indices_left = list(range(0, len(self.valid_indices)))

        result = {}
        patch_pre, patch_pre_mask, patch_post, patch_random, patch_random_mask = self.cut_patches(midpoint, midpoint_r, self.patch_size)

        if "pre" not in result:
            result['pre'] = patch_pre
            result['pre_mask'] = patch_pre_mask
        else:
            result['pre'] = torch.cat((result['pre'], patch_pre), dim=0)
            result['pre_mask'] = torch.cat((result['pre_mask'], patch_pre_mask), dim=0)

        result['post'] = patch_post
        result['already'] = patch_random
        result['already_mask'] = patch_random_mask

        return result

    def __len__(self):
        return sum([(n.numel() // 2) for n in self.valid_indices])  # dont need to restart


