import os
import json
import numpy as np
import trimesh
import mesh_raycast
import igl
import cv2
from skimage import morphology

res = 512

def get_end_points(skeleton):
    foreground_pixels = np.argwhere(skeleton > 0)
    # detect end points
    end_points = []
    # iterate over the foreground pixels
    for pixel in foreground_pixels:
        row, col = pixel
        # define a 3x3 neighborhood around the pixel
        neighborhood = skeleton[row-1:row+2, col-1:col+2]
        # count the number of foreground pixels in the neighborhood
        foreground_count = np.sum(neighborhood) // 255
        # if the count is equal to 2, it is an end point
        if foreground_count == 2:
            end_points.append((col, row))
    return end_points


def remove_intersection(thin_mask, skeleton, r, color=0):
    ep1 = get_end_points(thin_mask)
    ep2 = get_end_points(skeleton)
    for point in ep1:
        if not point in ep2:
            cv2.circle(thin_mask, point, r, color, -1)
    return thin_mask


def get_thin_coords(thin_mask):
    # get coordnates
    coords = np.argwhere(thin_mask == 255)
    coords = coords.astype(np.float32) / (res-1) - 0.5
    tmp = np.zeros(coords.shape)
    tmp[:, 0] = coords[:, 1]
    tmp[:, 1] = -coords[:, 0]
    return tmp
    # return coords


def get_coord_dist(xy, dist_map, res):
    tmp = xy.copy()
    tmp[:,1] *= -1
    tmp = (tmp + 0.5) * (res-1)
    dist_values = bilinear_interpolation(dist_map, tmp)
    return dist_values


def bilinear_interpolation(image, xy):
    # 获取图像尺寸
    height, width = image.shape[0:2]

    # 将坐标拆分为整数和小数部分
    x_int = np.floor(xy[:,0]).astype(int)
    y_int = np.floor(xy[:,1]).astype(int)
    x_frac = xy[:,0] - x_int
    y_frac = xy[:,1] - y_int

    # 确保坐标不超出图像边界
    x_int = np.clip(x_int, 0, width - 2)
    y_int = np.clip(y_int, 0, height - 2)

    # 获取四个最近像素的颜色值
    value1 = image[y_int, x_int]
    value2 = image[y_int, x_int + 1]
    value3 = image[y_int + 1, x_int]
    value4 = image[y_int + 1, x_int + 1]

    if len(image.shape) == 3:
        x_frac, y_frac = x_frac[:, None], y_frac[:, None]

    # 对每个颜色通道进行插值
    interpolated_value = (1 - x_frac) * (1 - y_frac) * value1 + x_frac * (1 - y_frac) * value2 + \
                         (1 - x_frac) * y_frac * value3 + x_frac * y_frac * value4

    return interpolated_value


def get_offset_mask_double(vertices, faces, thin_coords, coord_dists, min_thickness):
    triangles = vertices[faces]
    triangles = np.array(triangles, dtype='f4')
    offset_mask = np.zeros(vertices.shape[0])
    offset_values = np.zeros(vertices.shape)
    for i in range(len(thin_coords)):
        source = (thin_coords[i, 0], thin_coords[i, 1], 1)
        rs0 = mesh_raycast.raycast(source, (0,0,-1), mesh=triangles)
        if len(rs0) > 0:
            # front view offset  -->
            r1 = min(rs0, key=lambda x: x['distance'])
            f_index_1 = r1['face']
            coords = triangles[f_index_1]
            coord_indexs = faces[f_index_1]
            for j in range(3):
                coord = coords[j]
                coord_index = coord_indexs[j]
                rs = mesh_raycast.raycast(coord, (0,0,-1), mesh=triangles)
                if len(rs) > 0:
                    r = max(rs, key=lambda x: x['distance'])
                    dist = coord[2] - r['point'][2]
                    target_dist = np.maximum(min_thickness, coord_dists[i]*2)
                    if offset_mask[coord_index] == 0 and dist > target_dist:
                        offset_values[coord_index, 2] -= (dist-target_dist)/2
                        offset_mask[coord_index] = 1

            # back view offset  <--
            r2 = max(rs0, key=lambda x: x['distance'])
            f_index_2 = r2['face']
            coords = triangles[f_index_2]
            coord_indexs = faces[f_index_2]
            for j in range(3):
                coord = coords[j]
                coord_index = coord_indexs[j]
                rs = mesh_raycast.raycast(coord, (0,0,1), mesh=triangles)
                if len(rs) > 0:
                    r = max(rs, key=lambda x: x['distance'])
                    dist = r['point'][2] - coord[2]
                    target_dist = np.maximum(min_thickness, coord_dists[i]*2)
                    if offset_mask[coord_index] == 0 and dist > target_dist:
                        offset_values[coord_index, 2] += (dist - target_dist)/2
                        offset_mask[coord_index] = 1

    return offset_values, offset_mask==1


def get_offset_mask_front(vertices, faces, thin_coords, coord_dists, min_thickness):
    triangles = vertices[faces]
    triangles = np.array(triangles, dtype='f4')
    offset_mask = np.zeros(vertices.shape[0])
    offset_values = np.zeros(vertices.shape)
    for i in range(len(thin_coords)):
        source = (thin_coords[i, 0], thin_coords[i, 1], 1)
        rs0 = mesh_raycast.raycast(source, (0,0,-1), mesh=triangles)
        if len(rs0) > 0:
            # front view offset  -->
            r1 = min(rs0, key=lambda x: x['distance'])
            f_index_1 = r1['face']
            coords = triangles[f_index_1]
            coord_indexs = faces[f_index_1]
            for j in range(3):
                coord = coords[j]
                coord_index = coord_indexs[j]
                rs = mesh_raycast.raycast(coord, (0,0,-1), mesh=triangles)
                if len(rs) > 0:
                    r = max(rs, key=lambda x: x['distance'])
                    dist = coord[2] - r['point'][2]
                    target_dist = np.maximum(min_thickness, coord_dists[i]*2)
                    if offset_mask[coord_index] == 0 and dist > target_dist:
                        offset_values[coord_index, 2] -= dist-target_dist
                        offset_mask[coord_index] = 1

            # back view offset  <--
            r2 = max(rs0, key=lambda x: x['distance'])
            f_index_2 = r2['face']
            coords = triangles[f_index_2]
            coord_indexs = faces[f_index_2]
            for j in range(3):
                coord_index = coord_indexs[j]
                offset_mask[coord_index] = 1 # fix back

    return offset_values, offset_mask==1


def get_offset_mask_back(vertices, faces, thin_coords, coord_dists, min_thickness):
    triangles = vertices[faces]
    triangles = np.array(triangles, dtype='f4')
    offset_mask = np.zeros(vertices.shape[0])
    offset_values = np.zeros(vertices.shape)
    for i in range(len(thin_coords)):
        source = (thin_coords[i, 0], thin_coords[i, 1], 1)
        rs0 = mesh_raycast.raycast(source, (0,0,-1), mesh=triangles)
        if len(rs0) > 0:
            # front view offset  -->
            r1 = min(rs0, key=lambda x: x['distance'])
            f_index_1 = r1['face']
            coords = triangles[f_index_1]
            coord_indexs = faces[f_index_1]
            for j in range(3):
                coord_index = coord_indexs[j]
                offset_mask[coord_index] = 1 # fix front

            # back view offset  <--
            r2 = max(rs0, key=lambda x: x['distance'])
            f_index_2 = r2['face']
            coords = triangles[f_index_2]
            coord_indexs = faces[f_index_2]
            for j in range(3):
                coord = coords[j]
                coord_index = coord_indexs[j]
                rs = mesh_raycast.raycast(coord, (0,0,1), mesh=triangles)
                if len(rs) > 0:
                    r = max(rs, key=lambda x: x['distance'])
                    dist = r['point'][2] - coord[2]
                    target_dist = np.maximum(min_thickness, coord_dists[i]*2)
                    if offset_mask[coord_index] == 0 and dist > target_dist:
                        offset_values[coord_index, 2] += dist - target_dist
                        offset_mask[coord_index] = 1

    return offset_values, offset_mask==1


def thining_processing(v, f, mask, save_dir, thin_type, res, min_thickness):
    # compute distance
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    tmp = distance.astype(np.float32)
    tmp = (tmp / tmp.max() * 255.).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, 'distance.png'), tmp)

    tmp = ((distance > 11) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, 'fixed_mask.png'), tmp)

    fixed_mask = get_coord_dist(v[:, 0:2], distance, res) > 11
   
    # compute skeleton
    skeleton = (morphology.skeletonize(mask, method='lee')*255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, 'skeleton.png'), skeleton)

    # get thin mask
    thin_mask = skeleton * (distance <= 6.5)
    cv2.imwrite(os.path.join(save_dir, 'thin_mask.png'), thin_mask)
    # remove end points
    tmp = thin_mask.copy()
    tmp = remove_intersection(tmp, skeleton, 11, 100)
    cv2.imwrite(os.path.join(save_dir, 'thin_mask_rm_ep_gray.png'), tmp)

    thin_mask = remove_intersection(thin_mask, skeleton, 11)
    cv2.imwrite(os.path.join(save_dir, 'thin_mask_rm_ep.png'), thin_mask)
    thin_coords = get_thin_coords(thin_mask)
    coord_dists = get_coord_dist(thin_coords[:, 0:2], distance, res) / res
    

    if thin_type == 'double':
        offset_values, offset_mask = get_offset_mask_double(v, f, thin_coords, coord_dists, min_thickness)
    elif thin_type == 'front':
        offset_values, offset_mask = get_offset_mask_front(v, f, thin_coords, coord_dists, min_thickness)
    elif thin_type == 'back':
        offset_values, offset_mask = get_offset_mask_back(v, f, thin_coords, coord_dists, min_thickness)
    else:
        quit()
    
    s = fixed_mask | offset_mask
    b = np.array([[t[0] for t in [(i, s[i]) for i in range(0, v.shape[0])] if t[1] == 1]]).T
    d_bc = offset_values[s==1]
    d = igl.harmonic(v, f, b, d_bc, 2)
    v = v + d

    return v


if __name__ == '__main__':

    root_dir = '../dataset/AnimatedDrawings/preprocessed'
    uid_list_file = '../dataset/AnimatedDrawings/drawings_uids_thinning.json'
    ortho_scale = 1.35
    thin_type = 'double'
    
    with open(uid_list_file) as f:
        all_uids = json.load(f)

    for uid in all_uids:
        uid = 'ff7ab74a67a443e3bda61e69577f4e80'
        print(uid)
        mesh_dir = os.path.join(root_dir, uid, 'mesh')
        # load mask
        mask_fn = os.path.join(root_dir, uid, 'char/mask.png')
        mask = cv2.imread(mask_fn, -1)
        mask = ((mask>127)*255).astype(np.uint8)
        mesh_fn = os.path.join(mesh_dir, 'it3000-mc512-50000_cut_simpl_shear.obj')
        mesh = trimesh.load_mesh(mesh_fn)

        v, f = mesh.vertices, mesh.faces
        v = v / ortho_scale
        thin_cache_dir = os.path.join(mesh_dir, 'thin')
        os.makedirs(thin_cache_dir, exist_ok=True)

        new_v = thining_processing(v, f, mask, thin_cache_dir, thin_type, 512, min_thickness=1/1024)
        mesh.vertices = new_v * ortho_scale

        file_name_thinned = mesh_fn.replace('.obj', '_thin.obj')
        mesh.export(file_name_thinned)
        print(f"[INFO] thinned mesh is saved in : {file_name_thinned}")
        smoothed_mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=20)
        file_name_smoothed = file_name_thinned.replace('.obj', '_smooth.obj')
        smoothed_mesh.export(file_name_smoothed)
        quit()
