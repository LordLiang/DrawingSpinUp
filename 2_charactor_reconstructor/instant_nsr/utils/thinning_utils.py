import os
import numpy as np
import mesh_raycast
import igl
import cv2
from skimage import morphology

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


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


def remove_short_lines(thin_mask, min_length):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(thin_mask)
    result = np.zeros_like(thin_mask)
    for i in range(1, labels.max()+1):
        print(stats[i])
        if stats[i, 4] >= min_length:
            result[labels == i] = 255
    return result


def get_thin_coords(thin_mask, res):
    # get coordnates
    thin_mask = cv2.dilate(thin_mask, kernel, iterations=1)
    coords = np.argwhere(thin_mask>0)
    coords = coords.astype(np.float32) / (res-1) - 0.5
    tmp = np.zeros(coords.shape)
    tmp[:, 0] = coords[:, 1]
    tmp[:, 1] = -coords[:, 0]
    return tmp


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


def get_offset_mask(vertices, faces, thin_coords, coord_dists, min_thickness, type='double'):
    triangles = vertices[faces]
    triangles = np.array(triangles, dtype='f4')
    offset_mask = np.zeros(vertices.shape[0])
    offset_values = np.zeros(vertices.shape)
    for i in range(len(thin_coords)):
        source = (thin_coords[i, 0], thin_coords[i, 1], 1)
        rs0 = mesh_raycast.raycast(source, (0,0,-1), mesh=triangles)

        if len(rs0) > 0:
            if type == 'double':
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
                        if offset_mask[coord_index] == 0 and dist > target_dist and dist < 0.06:
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
                        if offset_mask[coord_index] == 0 and dist > target_dist and dist < 0.06:
                            offset_values[coord_index, 2] += (dist - target_dist)/2
                            offset_mask[coord_index] = 1

            elif type == 'front':
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
                        if offset_mask[coord_index] == 0 and dist > target_dist and dist < 0.06:
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

            elif type == 'back':
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
                        if offset_mask[coord_index] == 0 and dist > target_dist and dist < 0.06:
                            offset_values[coord_index, 2] += dist - target_dist
                            offset_mask[coord_index] = 1

            else:
                quit()

    return offset_values, offset_mask==1


def thinning_processing(v, f, config, save_cache=True, theta_1=11, theta_2=6, r=11):
    if save_cache:
        cache_dir = os.path.join(config.output_dir, 'thinning_cache')
        os.makedirs(cache_dir, exist_ok=True)

    mask_fn = os.path.join(config.input_dir, '../char/mask.png')
    mask = cv2.imread(mask_fn, -1)
    res = mask.shape[0]
    min_thickness = 1 / res

    # compute distance and sketleton
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    skeleton = morphology.skeletonize(mask, method='lee')

    # get fixed mask
    fix_mask = get_coord_dist(v[:, 0:2], distance, res) >= theta_1

    # get move-needed mask
    mov_mask = skeleton * (distance <= theta_2)

    # remove intersections
    mov_mask_new = remove_intersection(mov_mask.copy(), skeleton, r)
    
    
    if save_cache:
        tmp = distance.astype(np.float32)
        tmp = (tmp / tmp.max() * 255.).astype(np.uint8)
        cv2.imwrite(os.path.join(cache_dir, 'distance.png'), tmp)
        cv2.imwrite(os.path.join(cache_dir, 'skeleton.png'), skeleton)
        cv2.imwrite(os.path.join(cache_dir, 'mov_mask.png'), mov_mask)
        tmp = ((distance >= theta_1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(cache_dir, 'fix_mask.png'), tmp)  
        tmp = remove_intersection(mov_mask.copy(), skeleton, r, 100)
        cv2.imwrite(os.path.join(cache_dir, 'mov_mask_rm_inter_gray.png'), tmp)
        cv2.imwrite(os.path.join(cache_dir, 'mov_mask_rm_inter.png'), mov_mask_new)

    thin_coords = get_thin_coords(mov_mask_new, res)
    coord_dists = get_coord_dist(thin_coords[:, 0:2], distance, res) / res
    offset_values, offset_mask = get_offset_mask(v, f, thin_coords, coord_dists, min_thickness, config.thinning_type)
    
    s = fix_mask | offset_mask
    b = np.array([[t[0] for t in [(i, s[i]) for i in range(0, v.shape[0])] if t[1] == 1]]).T
    d_bc = offset_values[s==1]
    d = igl.harmonic(v, f, b, d_bc, 2)
    v = v + d

    return v
