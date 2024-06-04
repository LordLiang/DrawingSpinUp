import os
import numpy as np
from PIL import Image
import trimesh
import mesh_raycast
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import cv2


def rotate_point_cloud(point_cloud, axis, angle):
    rotation_matrix = rotation_matrix_3d(axis, angle)
    rotated_points = np.dot(point_cloud, rotation_matrix.T)[:, :3]
    return rotated_points


def rotation_matrix_3d(axis, angle):
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    rotation_matrix = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                                [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                                [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])
    return rotation_matrix



def direct_query(image, coordinates):
    height, width, _ = image.shape
    x_int = np.round(coordinates[:, 0]).astype(int)
    y_int = np.round(coordinates[:, 1]).astype(int)
    x_int = np.clip(x_int, 0, width - 1)
    y_int = np.clip(y_int, 0, height - 1)
    value = image[y_int, x_int]
    return value


def bilinear_interpolation(image, coordinates):
    # 获取图像尺寸
    height, width, channels = image.shape

    # 将坐标拆分为整数和小数部分
    x_int = np.floor(coordinates[:, 0]).astype(int)
    y_int = np.floor(coordinates[:, 1]).astype(int)
    x_frac = coordinates[:, 0] - x_int
    y_frac = coordinates[:, 1] - y_int

    # 确保坐标不超出图像边界
    x_int = np.clip(x_int, 0, width - 2)
    y_int = np.clip(y_int, 0, height - 2)

    # 获取四个最近像素的颜色值
    value1 = image[y_int, x_int]
    value2 = image[y_int, x_int + 1]
    value3 = image[y_int + 1, x_int]
    value4 = image[y_int + 1, x_int + 1]

    x_frac, y_frac = x_frac[:, None], y_frac[:, None]

    # 对每个颜色通道进行插值
    interpolated_value = (1 - x_frac) * (1 - y_frac) * value1 + x_frac * (1 - y_frac) * value2 + \
                         (1 - x_frac) * y_frac * value3 + x_frac * y_frac * value4

    return interpolated_value


def get_color_from_image(pc, color_map, angle, axis=[0,1,0]):
    res = color_map.shape[0]
    pc = rotate_point_cloud(pc, axis, np.radians(angle))
    xy = pc[:, 0:2].copy()
    xy[:,1] *= -1
    xy = (xy + 0.5) * (res-1)
    # color_values = bilinear_interpolation(color_map, xy)
    color_values = direct_query(color_map, xy)
    return color_values


def interpolate_rgb(unknown_points, known_positions, known_colors, k=10):
    tree = cKDTree(known_positions)
    distances, indices = tree.query(unknown_points, k)
    unknown_colors = np.zeros((len(unknown_points), 3))

    for i in range(len(unknown_points)):
        nearest_index = indices[i]
        nearest_distance = distances[i]
        nearest_color = known_colors[nearest_index]
        weight = 1.0 / (nearest_distance + 1e-6)  # 距离倒数作为权重
        weight = weight / weight.sum()
        unknown_colors[i] = (nearest_color * weight[:, np.newaxis]).sum(0)
    return unknown_colors


kernel = np.ones((3, 3), np.uint8)
def load_color(data_dir, view):
    color_filepath = os.path.join(data_dir, 'color', '%s.png'%(view))
    color = np.array(Image.open(color_filepath), dtype=np.float32) / 255.
    mask_filepath = os.path.join(data_dir, 'mask', '%s.png'%(view))
    mask = np.array(Image.open(mask_filepath))
    mask = cv2.erode(mask, kernel, iterations=4)
    mask = mask.astype(np.float32) / 255.
    rgba = np.concatenate((color, mask[:,:,None]), 2)
    return rgba


ray_directions = {'front':(0,0,1), 'back':(0,0,-1), 'left':(-1,0,0), 'right':(1,0,0), 'front_left': (-1,0,1), 'front_right':(1,0,1)}
rotation_angles = {'front':0, 'back':180, 'left':-90,'right':90, 'front_left':-45, 'front_right':45}
def color_projection(vertices, faces, data_dir):

    triangles = vertices[faces]
    triangles = np.array(triangles, dtype='f4')

    # x:right y:up z:front
    views = ['front', 'back', 'front_left', 'front_right']

    flags = np.ones(len(vertices))
    vert_colors = np.zeros((len(vertices), 4))
    for view in views:
        valid_indices = np.where(flags==1)[0]
        if len(valid_indices) > 0:
            color = load_color(data_dir, view)
            valid_color = get_color_from_image(vertices[valid_indices], color, rotation_angles[view])
            valid_indices2 = np.where(valid_color[:,3]>0.5)[0]
            valid_indices = valid_indices[valid_indices2]
            valid_color = valid_color[valid_indices2]

            for i in range(len(valid_indices)):
                ind = valid_indices[i]
                result = mesh_raycast.raycast(vertices[ind], ray_directions[view], mesh=triangles)
                if len(result) > 0:
                    farthest_result = max(result, key=lambda x: x['distance'])
                    if farthest_result['distance'] == 0:
                        flags[ind] = 0 # 0 means get color
                        vert_colors[ind] = valid_color[i]

    # handle the remaining uncolored vertices
    unknown_points = vertices[flags==1, 0:2]
    known_positions = vertices[flags==0, 0:2]
    known_colors = vert_colors[flags==0, 0:3]
    vert_colors[flags==1, 0:3] = interpolate_rgb(unknown_points, known_positions, known_colors)

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


def PCA(data):
    # normalize 归一化
    mean_data = np.mean(data, axis=0)
    normal_data = data - mean_data
    # 计算对称的协方差矩阵
    H = np.dot(normal_data.T, normal_data)
    # SVD奇异值分解，得到H矩阵的特征值和特征向量
    eigenvectors, eigenvalues, _ = np.linalg.svd(H)
    sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:, sort]
    return eigenvectors


def shear_transformation(v_np):
    v = PCA(v_np[:,1:3]) # PCA方法得到对应的特征向量
    a = -v[1,0] / v[0,0]
    v_np[:,2] += a * v_np[:,1]
    return v_np
