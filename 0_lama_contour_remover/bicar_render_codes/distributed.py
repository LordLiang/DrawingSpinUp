import json
import subprocess
import os
import argparse

parser = argparse.ArgumentParser(description='bicar rendering')

parser.add_argument('--input_models_path', type=str, default='../../dataset/3DBiCar/bicar_uids.json',
                    help='Path to a json file containing a list of 3D object files.')

parser.add_argument('--start_i', type=int, default=0,
                    help='the index of first object to be rendered.')

parser.add_argument('--end_i', type=int, default=1500,
                    help='the index of the last object to be rendered.')

parser.add_argument('--bicar_root', type=str, default='../../dataset/3DBiCar/raw',
                    help='Path to a json file containing a list of 3D object files.')

parser.add_argument('--save_folder', type=str, default='../../dataset/3DBiCar/front_contour_render',
                    help='Path to a json file containing a list of 3D object files.')

parser.add_argument('--blender_install_path', type=str, default='../../blender-3.3.1-linux-x64/blender',
                    help='blender path.')

parser.add_argument('--ortho_scale', type=float, default=1.35,
                    help='ortho rendering usage; how large the object is')

parser.add_argument('--random_pose', action='store_true',
                    help='whether randomly rotate the poses to be rendered')

args = parser.parse_args()


if __name__ == "__main__":

    if args.input_models_path is not None:
        with open(args.input_models_path, "r") as f:
            model_paths = json.load(f)

    args.end_i = len(model_paths) if args.end_i > len(model_paths) else args.end_i

    for item in model_paths[args.start_i:args.end_i]:
        obj_path = os.path.join(args.bicar_root, item)
        print(args.blender_install_path)
        command = (
            f" CUDA_VISIBLE_DEVICES=0 "
            f" {args.blender_install_path} --background -E CYCLES --python blenderProc_ortho.py --"
            f" --object_path {obj_path}"
            f" --output_folder {args.save_folder}"
            f" --ortho_scale {args.ortho_scale} "
        )

        if args.random_pose:
            print("random pose to render")
            command += f" --random_pose"

        subprocess.run(command, shell=True)


