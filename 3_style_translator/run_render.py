import subprocess
import os
import numpy as np
import argparse
import time
import glob
import cv2
import OpenEXR
import Imath
import array


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
def depth2edge(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    data_window = exr_file.header()['dataWindow']
    w = data_window.max.x - data_window.min.x + 1
    h = data_window.max.y - data_window.min.y + 1

    depth = array.array('f', exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth).reshape((h,w))
    bg = depth>1000
    depth_min, depth_max = depth[~bg].min(), depth[~bg].max()
    depth = (depth-depth_min) / (depth_max-depth_min)*200
    depth[bg] = 255
    edge = cv2.Canny(np.uint8(depth), threshold1=50, threshold2=150)
    edge = cv2.dilate(edge, kernel, iterations=1)
    return edge


if __name__ == '__main__':

    # run 'export DISPLAY=:1' if terminal rendering

    parser = argparse.ArgumentParser(description='frame rendering')
    parser.add_argument('--data_dir', default='../dataset/AnimatedDrawings/preprocessed', help='data root')
    parser.add_argument('--uid', default='0dd66be9d0534b93a092d8c4c4dfd30a', help='image uid')
    parser.add_argument('--blender_install_path', default='../blender-3.3.1-linux-x64/blender', help='blender path')
    parser.add_argument('--test', action='store_true', help='test')
    args = parser.parse_args()

    input_dir = os.path.join(args.data_dir, args.uid, 'mesh/fbx_files')
    mesh_file = glob.glob(os.path.join(args.data_dir, args.uid, 'mesh/*.obj'))[0]
    script_file = './blender_animation.py'
    
    if not args.test:
        action_types = ['rest_pose']
    else:
        action_types = [item.replace('.fbx', '') for item in os.listdir(input_dir)]
        action_types.remove('rest_pose')
        if len(action_types) == 0:
            action_types.append('rest_rotate')

    for action_type in action_types:

        output_dir = os.path.join(args.data_dir, args.uid, 'mesh/blender_render', action_type)

        start = time.time()

        if not os.path.exists(os.path.join(output_dir, 'depth')):
            # render color and pos
            if action_type == 'rest_pose':
                fbx_file = os.path.join(input_dir, 'rest_pose.fbx')
                config_file = 'configs/blender/config_ortho.blend'
            if action_type == 'rest_rotate':
                fbx_file = os.path.join(input_dir, 'rest_pose.fbx')
                config_file = 'configs/blender/config_ortho_rotate.blend'
            else:
                fbx_file = os.path.join(input_dir, '%s.fbx'%(action_type))
                config_file = 'configs/blender/config_ortho.blend'

            subprocess.run(f'{args.blender_install_path} -b {config_file} -E BLENDER_EEVEE --python {script_file} \
                                                        -- --fbx_file {fbx_file} \
                                                            --output_dir {output_dir} \
                                                            --mesh_file {mesh_file}', shell=True)

        # compute edge  
        if os.path.exists(os.path.join(output_dir, 'depth')) and os.path.exists(os.path.join(args.data_dir, args.uid, 'char/ffc_resnet_inpainted.png')):             
            os.makedirs(os.path.join(output_dir, 'edge'), exist_ok=True)
            files = os.listdir(os.path.join(output_dir, 'pos'))
            
            for file in files:
                # z-depth
                depth_fn = os.path.join(output_dir, 'depth', file.replace('png', 'exr'))
                edge = depth2edge(depth_fn)
                cv2.imwrite(os.path.join(output_dir, 'edge', file), 255-edge)

        end = time.time()
        num_frame = len(glob.glob(os.path.join(output_dir, 'color', '*.png')))
        print((end-start)/num_frame, num_frame)
        
