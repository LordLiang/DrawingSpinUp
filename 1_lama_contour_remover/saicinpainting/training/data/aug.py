import random
# import io
# import cairosvg
from PIL import Image
import numpy as np

import torchvision.transforms as transforms


def random_color():
    return (random.randint(0, 255), random.randint(0, 255),random.randint(0, 255))


def read_data(img_fn, svg_fn):
    # img
    img = Image.open(img_fn)

    # sketch
    with open(svg_fn) as f:
        sketch_svg = f.read()
    # change color randomly
    color = random_color()
    sketch_svg = sketch_svg.replace('rgb(0, 0, 0)', 'rgb{}'.format(color))
    sketch = cairosvg.svg2png(bytestring=sketch_svg)
    sketch = Image.open(io.BytesIO(sketch))
    return np.array(img).astype(np.float32), np.array(sketch).astype(np.float32)


def get_data(img_fn, svg_fn):
    img, sketch = read_data(img_fn, svg_fn)

    # color offset
    B_np = img[:,:,0:3] + np.random.randint(0, 50, 3)
    B_np = np.clip(B_np, 0, 255)
    # add white bg
    M_np = img[:,:,3:4] / 255
    B_np = B_np * M_np + 255 * (1 - M_np) # no contour

    # get CM_np
    C_np = sketch[:,:,0:3]
    CM_np = sketch[:,:,3:4] / 255
    CM_np = np.minimum(M_np, CM_np)

    # get A_np
    CM_np_soft = CM_np
    if np.random.rand() > 0.5:
        CM_np_soft = (np.random.rand(1) * 0.5 + 0.5) * CM_np_soft
    if np.random.rand() > 0.5:
        CM_np_soft = (np.random.rand(512, 512, 1) * 0.5 + 0.5) * CM_np_soft
        
    A_np = B_np * (1 - CM_np_soft) + C_np * CM_np_soft # add contour on B to get A
    CM_np = CM_np > 0

    A = Image.fromarray(A_np.astype(np.uint8))
    M = Image.fromarray((M_np[:,:,0] * 255).astype(np.uint8))
    CM = Image.fromarray((CM_np[:,:,0] * 255).astype(np.uint8))
    return A, M, CM


def get_params(size, crop_size=512, load_size=512, preprocess='resize_and_crop'):
    w, h = size
    new_h = h
    new_w = w
    if preprocess == 'resize_and_crop':
        new_h = new_w = load_size
    elif preprocess == 'scale_width_and_crop':
        new_w = load_size
        new_h = load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(params=None, num_channels=3, crop_size=512, load_size=512, no_flip=True, preprocess='resize_and_crop', 
                  method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if num_channels==1:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in preprocess:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, load_size, crop_size, method)))

    if 'crop' in preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], crop_size)))

    if preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)



def __transforms2pil_resize(method):
    mapper = {transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
              transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
              transforms.InterpolationMode.NEAREST: Image.NEAREST,
              transforms.InterpolationMode.LANCZOS: Image.LANCZOS,}
    return mapper[method]


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True