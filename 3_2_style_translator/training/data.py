import os
from PIL import Image
import torch.utils.data
from torch.nn import functional as F
from training.custom_transforms import *


################################
# Dataset full-images for inference
################################
class DatasetFullImages(torch.utils.data.Dataset):
    def __init__(self, data_root, use_pos=False, use_edge=False):
        super(DatasetFullImages, self).__init__()
        self.data_root = data_root
        self.use_pos, self.use_edge = use_pos, use_edge
        self.fnames = sorted(os.listdir(os.path.join(self.data_root, 'color')))
        self.transform = build_transform()
        self.mask_transform = build_mask_transform()
        self.gray_transform = build_gray_transform()


    def __getitem__(self, item):
        # get an image that is NOT stylized and its stylized counterpart
        fileName = self.fnames[item]
        result = {'file_name': self.fnames[item]}
        pre_color = Image.open(os.path.join(self.data_root, 'color', fileName))
        mask = pre_color.split()[-1]
        if self.use_pos:
            pre_pos = Image.open(os.path.join(self.data_root, 'pos', fileName))
        if self.use_edge:
            pre_edge = Image.open(os.path.join(self.data_root, 'edge', fileName))
        
        pre_feature = [self.transform(pre_color)]
        pre_feature.append(self.mask_transform(mask))
        if self.use_pos:
            pre_feature.append(self.transform(pre_pos)[0:2])# just use X&Y

        if self.use_edge:
            pre_feature.append(self.gray_transform(pre_edge))

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
    def __init__(self, data_root, dir_post, patch_size, use_pos=False, use_edge=False):
        super(DatasetPatches_M, self).__init__()
        self.data_root, self.dir_post = data_root, dir_post
        self.patch_size = patch_size
        self.use_pos, self.use_edge = use_pos, use_edge
        
        self.transform = build_transform()
        self.mask_transform = build_mask_transform()
        self.gray_transform = build_gray_transform()

        self.images_pre = None
        self.images_post = None
        self.images_mask = None
        self.valid_indices = None
        self.valid_indices_left = None

        self.load_image()

    def load_image(self, fileName='0001.png'):
        self.post_color = Image.open(os.path.join(self.dir_post, 'texture_with_bg.png'))
        self.post_mask = Image.open(os.path.join(self.dir_post, 'mask.png'))
        self.post_color = apply_alpha(self.post_color, self.post_mask)

        self.pre_color = Image.open(os.path.join(self.data_root, 'color', fileName))
        self.pre_color = apply_alpha(self.pre_color, self.post_mask)

        if self.use_pos:
            self.pre_pos = Image.open(os.path.join(self.data_root, 'pos', fileName))
            self.pre_pos = apply_alpha(self.pre_pos, self.post_mask)

        if self.use_edge:
            self.pre_edge = Image.open(os.path.join(self.data_root, 'edge', fileName))

        self.preprocessing_image()

    def preprocessing_image(self):
        pre_color = self.pre_color
        post_color = self.post_color
        if self.use_pos:
            pre_pos = self.pre_pos
        if self.use_edge:
            pre_edge = self.pre_edge

        mask_tensor = self.mask_transform(self.post_mask)
        post_tensor = self.transform(post_color)
        
        pre_feature = [self.transform(pre_color)]
        pre_feature.append(mask_tensor)
        if self.use_pos:
            pre_feature.append(self.transform(pre_pos)[0:2]) # just use X&Y
        if self.use_edge:
            pre_feature.append(self.gray_transform(pre_edge))

        self.images_pre = torch.cat(pre_feature, 0)
        self.images_post = post_tensor
        self.images_mask = mask_tensor

        erosion_weights = torch.ones((1, 1, 7, 7))
        m = mask_tensor
        m[m < 0.4] = 0
        m = F.conv2d(m.unsqueeze(0), erosion_weights, stride=1, padding=3)
        m[m < erosion_weights.numel()] = 0
        m /= erosion_weights.numel()

        self.valid_indices = m.squeeze().nonzero(as_tuple=False)
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
        return sum([(n.numel() // 2) for n in self.valid_indices]) * 5  # dont need to restart


