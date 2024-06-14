from PIL import Image
import os
import glob
import argparse


parser = argparse.ArgumentParser(description='generate GIF file')
parser.add_argument('--data_dir', default='../dataset/AnimatedDrawings/preprocessed', help='data root')
parser.add_argument('--uid', default='0dd66be9d0534b93a092d8c4c4dfd30a', help='image uid')
args = parser.parse_args()

data_dir = os.path.join(args.data_dir, args.uid, 'animation/blender_render')
action_types = [f for f in os.listdir(data_dir) if not f.startswith('.')]
action_types.remove('keyframe')
render_types = [f for f in os.listdir(os.path.join(data_dir, 'keyframe')) if f.startswith('res_')]

for action_type in action_types:
    for render_type in render_types:
        print(action_type, render_type)
        path = os.path.join(data_dir, action_type, render_type)
        os.makedirs(os.path.join(data_dir, '../gif'), exist_ok=True)
        save_fn = os.path.join(data_dir, '../gif', action_type+'_'+render_type+'.gif')
        frame_fns = glob.glob(path+'/*.png')
        frame_fns = sorted(frame_fns)
        frames = []
        for frame_fn in frame_fns:
            frame = Image.open(frame_fn)
            frames.append(frame)
        frames[0].save(save_fn, save_all=True, append_images=frames[1:], duration=30, disposal=1, loop=0)
