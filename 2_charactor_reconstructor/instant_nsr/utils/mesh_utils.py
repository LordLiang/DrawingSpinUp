import os
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors

from instant_nsr.utils.coloring_utils import color_projection, uv_mapping
from instant_nsr.utils.thinning_utils import thinning_processing


def remesh(verts, faces, face_count):
    mesh = trimesh.Trimesh()
    mesh.vertices = verts
    mesh.faces = faces
    # mesh = max(mesh.split(), key=lambda x: len(x.faces))
    mesh = mesh.simplify_quadratic_decimation(face_count=face_count)
    new_verts = mesh.vertices
    new_faces = mesh.faces

    print(
        f"[INFO] mesh cleaning: {verts.shape} --> {new_verts.shape}, {faces.shape} --> {new_faces.shape}"
    )
    return new_verts, new_faces


def save_mesh(config, verts, faces, vert_colors):
    os.makedirs(config.output_dir, exist_ok=True)
    print('Create the output directory: ', config.output_dir)

    verts = verts * 0.5
    # change to front-facing
    # before x:right y:back z:up
    # after x:right y:up z:front
    v_np_old =  np.zeros_like(verts)  
    v_np_old[:, 0] = verts[:, 0]
    v_np_old[:, 1] = verts[:, 2]
    v_np_old[:, 2] = verts[:, 1] * -1

    if config.thinning:
        v_np_old = thinning_processing(v_np_old, faces, config)

    # smooth
    if config.smoothing:
        mesh = trimesh.Trimesh(vertices=v_np_old, faces=faces)
        trimesh.smoothing.filter_laplacian(mesh, lamb=2, iterations=5, implicit_time_integration=True)
        v_np, faces = mesh.vertices, mesh.faces
    else:
        v_np = v_np_old
    
    # color back-projection
    if config.color_back_projection:
        vert_colors = color_projection(v_np, faces, config.input_dir)
    else:
        nbrs = NearestNeighbors(n_neighbors=1).fit(v_np_old)
        _, indices = nbrs.kneighbors(v_np)
        vert_colors = vert_colors[indices.flatten()]
    
    # shear transformation
    if config.shearing:
        v_np = shear_transformation(v_np)

    # scale
    print("ortho scale is: ", config.ortho_scale)
    v_np = v_np * config.ortho_scale

    # uv mapping
    if config.export_uv:
        mesh = uv_mapping(v_np, faces, vert_colors, config.save_name)
    else: # export vertex colors directly
        mesh = trimesh.Trimesh(vertices=v_np, faces=faces, vertex_colors=vert_colors)

    file_name = os.path.join(config.output_dir, config.save_name + '.obj')
    mesh.export(file_name)
    print(f"[INFO] mesh is saved in : {file_name}")


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