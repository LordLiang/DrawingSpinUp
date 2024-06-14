import subprocess
import os
import cv2
import numpy as np
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='frame rendering')
    parser.add_argument('--data_dir', default='../dataset/AnimatedDrawings/preprocessed', help='data root')
    parser.add_argument('--uid', default='0dd66be9d0534b93a092d8c4c4dfd30a', help='image uid')
    parser.add_argument('--blender_install_path', default='../blender-3.3.1-linux-x64/blender', help='blender path')
    parser.add_argument('--keyframe', action='store_true', help='render keyframe pair for training')

    args = parser.parse_args()

    input_dir = os.path.join(args.data_dir, args.uid, 'animation/fbx_files')
    mesh_file = os.path.join(args.data_dir, args.uid, 'mesh/it3000-mc512-50000_cut_simpl_thin_shear.obj')
    if not os.path.exists(mesh_file):
        mesh_file = os.path.join(args.data_dir, args.uid, 'mesh/it3000-mc512-50000_cut_simpl_shear.obj')

    render_types = ['color', 'pos', 'edge']
    script_file = 'blender_animation.py'

    if args.keyframe:
        action_types = ['keyframe']
    else:
        action_types = []
        for item in os.listdir(input_dir):
            item = item.replace('.fbx', '')
            if item != 'rest_pose':
                action_types.append(item)

    for action_type in action_types:
        output_dir = os.path.join(args.data_dir, args.uid, 'animation/blender_render', action_type)
        if action_type == 'keyframe':
            input_file = os.path.join(input_dir, 'rest_pose.fbx')
            num_frame = 1
            config_file = 'config/config.blend'

        elif action_type == 'rest_rotate':
            input_file = os.path.join(input_dir, 'rest_pose.fbx')
            num_frame = 60
            config_file = 'config/config_rotate.blend'

        else:
            input_file = os.path.join(input_dir, '%s.fbx'%(action_type))
            num_frame = -1
            config_file = 'config/config.blend'

        for render_type in render_types:
            if render_type in ['color', 'pos']:
                subprocess.run(f'{args.blender_install_path} --background {config_file} \
                            	                             --python {script_file} \
                            	                             -- --fbx_file {input_file} \
                            	                                --output_dir {output_dir} \
                            	                                --mesh_file {mesh_file} \
                                                                --num_frame {num_frame} \
                            	                                --type {render_type}', shell=True)

            elif render_type == 'edge':
                # canny
                os.makedirs(os.path.join(output_dir, 'edge'), exist_ok=True)
                files = os.listdir(os.path.join(output_dir, 'pos'))
                for file in files:
                    pos = cv2.imread(os.path.join(output_dir, 'pos', file), -1).astype(np.float32) / 255.0
                    b, g, r, alpha = cv2.split(pos)
                    b[alpha<1] = 2
                    g[alpha<1] = 2
                    r[alpha<1] = 2
                    
                    grad_x_b = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y_b = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)
                    grad_x_g = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y_g = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
                    grad_x_r = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y_r = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=3)

                    val_b = np.sqrt(np.square(grad_x_b) + np.square(grad_y_b))
                    val_g = np.sqrt(np.square(grad_x_g) + np.square(grad_y_g))
                    val_r = np.sqrt(np.square(grad_x_r) + np.square(grad_y_r))

                    edge = np.maximum(np.maximum(val_b, val_g), val_r)
                    edge = ((edge>0.3)*255).astype(np.uint8)
                    cv2.imwrite(os.path.join(output_dir, 'edge', file), edge)
            else:
                quit()