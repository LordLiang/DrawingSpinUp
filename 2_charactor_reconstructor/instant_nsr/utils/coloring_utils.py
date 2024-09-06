import os
import cv2
import torch
import numpy as np
from PIL import Image
import trimesh
import mesh_raycast
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
)
from pytorch3d.renderer.cameras import (
    look_at_view_transform,
    OrthographicCameras,
)


class MaskRenderer():
    def __init__(self, res):
        super(MaskRenderer, self).__init__()
        self.device = torch.device("cuda:0")
        depth_raster_settings = RasterizationSettings(image_size=res)
        R, T = look_at_view_transform(1, 0, 0)
        cameras_pytorch3d = OrthographicCameras(device=self.device, R=R, T=T)
        self.depth_renderer_left = MeshRasterizer(
            cameras=cameras_pytorch3d,
            raster_settings=depth_raster_settings
        )

    def render(self, v_np, f_np):
        verts = torch.from_numpy(v_np.astype(np.float32)).to(self.device)
        faces = torch.from_numpy(f_np.astype(np.float32)).to(self.device)
        meshes = Meshes(verts.unsqueeze(0), faces.unsqueeze(0))
        depth_ref = self.depth_renderer_left(meshes)
        mask = depth_ref.zbuf > -1
        mask = (mask[0,:,:, 0].cpu().numpy()*255).astype(np.uint8)
        return mask


def interpolate_rgb(unknown_points, known_positions, known_colors, k=8):
    tree = cKDTree(known_positions[:,0:2])
    distances, indices = tree.query(unknown_points[:,0:2], k)
    if k == 1:
        distances, indices = distances[:,None], indices[:,None]
    
    # 获取最近邻的颜色
    nearest_colors = known_colors[indices]  # 形状为 (N, k, 3)
    # 计算权重，避免除以零
    weights = 1.0 / (distances + 1e-6)  # 形状为 (N, k)
    # 归一化权重
    weights /= weights.sum(axis=1, keepdims=True)
    # 使用向量化计算未知点的颜色
    unknown_colors = np.einsum('ijk,ij->ik', nearest_colors, weights)
    return unknown_colors


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
def load_color(data_dir, view, mask):
    color = Image.open(os.path.join(data_dir, 'color', '%s.png'%(view))).resize((2048, 2048), Image.LANCZOS)
    mask = cv2.erode(mask, kernel, iterations=1)
    rgba = np.concatenate((np.array(color), mask[:,:,None]), 2)
    rgba = rgba.astype(np.float32) / 255.
    return rgba


def get_color_from_image(pc, color_map, back=False):
    res = color_map.shape[0]
    xy = pc[:, 0:2].copy()
    if back:
        xy[:,0] *= -1
    xy[:,1] *= -1
    xy = (xy + 0.5) * (res-1)
    color_values = direct_query(color_map, xy)
    return color_values


def direct_query(image, coordinates):
    height, width, _ = image.shape
    x_int = np.round(coordinates[:, 0]).astype(int)
    y_int = np.round(coordinates[:, 1]).astype(int)
    x_int = np.clip(x_int, 0, width - 1)
    y_int = np.clip(y_int, 0, height - 1)
    value = image[y_int, x_int]
    return value


mask_renderer = MaskRenderer(res=2048)
def color_projection(vertices, faces, data_dir):
    triangles = vertices[faces]
    triangles = np.array(triangles, dtype='f4')

    # x:right y:up z:front
    vert_colors = np.zeros((len(vertices), 4))

    # front
    mask_front = mask_renderer.render(vertices*2, faces)
    mask_front_tmp = Image.open(os.path.join(data_dir, 'mask', 'front.png')).resize((2048, 2048), Image.LANCZOS)
    mask_front = np.minimum(np.array(mask_front_tmp), mask_front)
    color_front = load_color(data_dir, 'front', mask_front)
    valid_color = get_color_from_image(vertices, color_front)
    valid_indices = np.where(valid_color[:,3]>0)[0]
    valid_color = valid_color[valid_indices]

    for i in range(len(valid_indices)):
        ind = valid_indices[i]
        result = mesh_raycast.raycast(vertices[ind], (0,0,1), mesh=triangles)
        if len(result) > 0:
            farthest_result = max(result, key=lambda x: x['distance'])
            if farthest_result['distance'] == 0:
                vert_colors[ind] = valid_color[i]

    # back
    mask_back = cv2.flip(mask_front, 1)
    valid_indices = np.where(vert_colors[:,3]==0)[0]
    color_back = load_color(data_dir, 'back', mask_back)
    valid_color = get_color_from_image(vertices[valid_indices], color_back, True)
    valid_indices2 = np.where(valid_color[:,3]>0)[0]
    valid_indices = valid_indices[valid_indices2]
    valid_color = valid_color[valid_indices2]

    for i in range(len(valid_indices)):
        ind = valid_indices[i]
        result = mesh_raycast.raycast(vertices[ind], (0,0,-1), mesh=triangles)
        if len(result) > 0:
            farthest_result = max(result, key=lambda x: x['distance'])
            if farthest_result['distance'] == 0:
                vert_colors[ind] = valid_color[i]

    # handle the remaining uncolored vertices
    unknown_points = vertices[vert_colors[:,3]==0]
    known_positions = vertices[vert_colors[:,3]>0]
    known_colors = vert_colors[vert_colors[:,3]>0, 0:3]
    vert_colors[vert_colors[:,3]==0, 0:3] = interpolate_rgb(unknown_points, known_positions, known_colors)
    return vert_colors[:,0:3]


def compute_interpolation_map(tcoords, values, method='linear', shape=(1024, 1024)):
    points = (tcoords * np.asarray(shape)[None, :])
    x = np.arange(shape[0])
    y = np.flip(np.arange(shape[1]))
    X, Y = np.meshgrid(x, y)
    res = griddata(points, values, (X, Y), method=method)
    res[np.isnan(res)] = 0
    res = Image.fromarray(np.clip(res*255, 0, 255).astype(np.uint8))
    return res


def uv_mapping(v_np, faces, vert_colors, save_name):
    import xatlas
    vmapping, indices, uvs = xatlas.parametrize(v_np, faces)
    v_np, f_np = v_np[vmapping], indices
    albedo = compute_interpolation_map(uvs, vert_colors[vmapping])
    mesh = trimesh.Trimesh()
    mesh.vertices = v_np
    mesh.faces = f_np
    mesh.visual = trimesh.visual.TextureVisuals()
    mesh.visual.uv = uvs
    mesh.visual.material = trimesh.visual.material.SimpleMaterial(image=albedo,
                                                                diffuse=[255, 255, 255, 255],
                                                                ambient=[255, 255, 255, 255],
                                                                specular=[255, 255, 255, 255],
                                                                glossiness=0.0)
    mesh.visual.material.name = save_name
    return mesh
