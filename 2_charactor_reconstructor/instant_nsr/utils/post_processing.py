import os
import numpy as np
import trimesh
from instant_nsr.utils.mesh_utils import color_projection, shear_transformation
from instant_nsr.utils.thin import thining_processing


def clean_mesh(verts, faces, face_count):
    mesh = trimesh.Trimesh()
    mesh.vertices = verts
    mesh.faces = faces
    mesh = max(mesh.split(), key=lambda x: len(x.faces))
    mesh = mesh.simplify_quadratic_decimation(face_count=face_count)
    mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=10)
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
    v_np =  np.zeros_like(verts)  
    v_np[:, 0] = verts[:, 0]
    v_np[:, 1] = verts[:, 2]
    v_np[:, 2] = verts[:, 1] * -1

    if config.thin:
        v_np = thining_processing(v_np, faces, config)
        mesh = trimesh.Trimesh(vertices=v_np, faces=faces)
        mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=10)
        v_np, faces = mesh.vertices, mesh.faces
        config.save_name += '_thin'

    # color projection
    if config.vertex_coloring:
        vert_colors = color_projection(v_np, faces, config.input_dir)

    # shear transformation (optional)
    if config.shear:
        v_np = shear_transformation(v_np)
        config.save_name += '_shear'

    # scale
    print("ortho scale is: ", config.ortho_scale)
    v_np = v_np * config.ortho_scale

    # uv mapping
    if config.export_uv:
        from instant_nsr.utils.mesh_utils import uv_mapping
        mesh = uv_mapping(v_np, faces, vert_colors, config.save_name)
    else: # export vertex colors directly
        mesh = trimesh.Trimesh(vertices=v_np, faces=faces, vertex_colors=vert_colors)

    file_name = os.path.join(config.output_dir, config.save_name + '.obj')
    mesh.export(file_name)
    print(f"[INFO] mesh is saved in : {file_name}")

