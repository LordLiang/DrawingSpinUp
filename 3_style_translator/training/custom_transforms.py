import numpy as np
from torchvision import transforms
from PIL import Image
import cv2


def to_image_space(x):
    return ((np.clip(x, -1, 1) + 1) / 2 * 255).astype(np.uint8)


def rgba_to_rgb(x):
    if x.mode == 'RGBA':
        x = np.array(x)[..., 0:3]
        x = Image.fromarray((x).astype(np.uint8))
    return x


def build_transform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    t = [rgba_to_rgb,
         transforms.ToTensor(),
         transforms.Normalize(mean, std)]
    return transforms.Compose(t)
    

def build_mask_transform():
    t = [transforms.ToTensor()]
    return transforms.Compose(t)


def overlap_edge_on_img(edge, img):
    edge_mask = np.array(edge) < 255
    img = np.array(img)
    img[edge_mask, 0:3] = 0
    img[edge_mask, 3] = 255
    return Image.fromarray(img)

def overlap_img(img1):
    if isinstance(img1, Image.Image):
        img1 = np.array(img1)
    img2 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
    alpha1 = img1[:, :, 3] / 255.0
    alpha2 = img2[:, :, 3] / 255.0
    # 计算合成后的图像
    combined_image = np.zeros_like(img1[:, :, 0:3])
    for c in range(3):  # 对于 R, G, B 通道
        combined_image[:, :, c] = (alpha1 * img1[:, :, c] + alpha2 * img2[:, :, c] * (1 - alpha1))
    # 合成新的 alpha 通道
    combined_alpha = alpha1 + alpha2 * (1 - alpha1)
    # 将结果合并为 RGBA 图像
    final_image = cv2.merge((combined_image, (combined_alpha * 255).astype('uint8')))
    return Image.fromarray(final_image)


def overlap_mask(mask1):
    if isinstance(mask1, Image.Image):
        mask1 = np.array(mask1)
    mask2 = cv2.rotate(mask1, cv2.ROTATE_90_CLOCKWISE)
    mask = np.maximum(mask1, mask2)
    return Image.fromarray(mask)


def cat_img(img1):
    img2 = overlap_img(img1)
    width, height = img1.size
    new_image = Image.new('RGBA', (width * 2, height))
    # 将两张图像粘贴到新图像中
    new_image.paste(img1, (0, 0))         # 粘贴第一张图像
    new_image.paste(img2, (width, 0))     # 粘贴第二张图像
    return new_image


def cat_mask(mask1):
    mask2 = overlap_mask(mask1)
    width, height = mask1.size
    new_mask = Image.new('L', (width * 2, height))
    # 将两张图像粘贴到新图像中
    new_mask.paste(mask1, (0, 0))         # 粘贴第一张图像
    new_mask.paste(mask2, (width, 0))     # 粘贴第二张图像
    return new_mask


def white_bg(img):
    img = np.array(img).astype(np.float32)
    alpha = img[:,:,3:4]/255.
    img = img[:,:,0:3]*alpha+255*(1-alpha)
    return Image.fromarray(img.astype(np.uint8))


def replace_alpha(img, mask):
    img = np.array(img)
    mask = np.array(mask)
    img[:,:,3] = mask
    return Image.fromarray(img)
