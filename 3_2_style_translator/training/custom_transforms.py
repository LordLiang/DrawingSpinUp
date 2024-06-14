import numpy as np
from torchvision import transforms
from PIL import Image
import cv2


def apply_alpha(a, b):
    a_array = np.array(a)
    b_array = np.expand_dims(b, axis=2)
    original_alpha = a_array[:, :, 3:4]
    b_array = np.minimum(b_array, original_alpha)
    new_array = np.concatenate((a_array[:, :, :3], b_array), axis=2)
    x = Image.fromarray(new_array)
    return x


def to_image_space(x):
    return ((np.clip(x, -1, 1) + 1) / 2 * 255).astype(np.uint8)


def rgba_to_rgb(x):
    if x.mode == 'RGBA':
        x = np.array(x).astype(np.float32)
        alpha = x[..., 3:4] / 255.
        x = x[..., :3]*alpha + 255*(1-alpha)
        x = Image.fromarray((x).astype(np.uint8))
    return x


def to_l(x):
    if x.mode == 'RGBA':
        x = rgba_to_rgb(x)
    return x.convert('L')


def build_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    t = [rgba_to_rgb,
         transforms.ToTensor(),
         transforms.Normalize(mean, std)]
    return transforms.Compose(t)
    

def build_gray_transform(mean=(0.5), std=(0.5)):
    t = [to_l,
         transforms.ToTensor(),
         transforms.Normalize(mean, std)]
    return transforms.Compose(t)


def build_mask_transform():
    t = [to_l,
         transforms.ToTensor()]
    return transforms.Compose(t)
