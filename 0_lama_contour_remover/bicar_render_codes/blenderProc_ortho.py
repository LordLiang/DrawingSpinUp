import argparse, os, sys, math
import bpy
bpy.ops.preferences.addon_enable(module='render_freestyle_svg')
from mathutils import Vector, Matrix
import time
import numpy as np


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def get_a_camera_location(loc):
    location = Vector([loc[0],loc[1],loc[2]])
    direction = - location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    rotation_euler = rot_quat.to_euler()
    return location, rotation_euler

def build_transformation_mat(translation, rotation):
    translation = np.array(translation)
    rotation = np.array(rotation)
    mat = np.eye(4)
    mat[:3, 3] = translation
    mat[:3, :3] = rotation
    return mat

def load_object(object_path):
    """Loads two obj models into the scene."""
    m_mtl = os.path.join(object_path, 'tpose', 'm.bmp')
    if not os.path.exists(m_mtl):
        os.rename(m_mtl.replace('bmp', 'BMP'), m_mtl)
    e_mtl = os.path.join(object_path, 'tpose', 'e.bmp')
    if not os.path.exists(e_mtl):
        os.rename(e_mtl.replace('bmp', 'BMP'), e_mtl)
    m_obj = os.path.join(object_path, 'tpose', 'm.obj')
    e_obj = os.path.join(object_path, 'tpose', 'e.obj')
    bpy.ops.wm.obj_import(filepath=m_obj)
    bpy.ops.wm.obj_import(filepath=e_obj)

def reset_scene():
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']
    env_light = 1.0 #0.4
    back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
    back_node.inputs['Strength'].default_value = 1.0 #0.5

def save_images(args):

    # Place camera
    bpy.data.cameras[0].type = "ORTHO"
    bpy.data.cameras[0].ortho_scale = args.ortho_scale   
    print("ortho scale ", args.ortho_scale)

    reset_scene()
    object_file = args.object_path
    object_uid = os.path.basename(object_file).split(".")[0]
    args.output_folder = os.path.join(args.output_folder)
    os.makedirs(os.path.join(args.output_folder, object_uid), exist_ok=True)

    # load the object
    load_object(object_file)
    normalize_scene()

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    
    camera_location = np.array([0, -2.0, 0])  # camera_front
    _location,_rotation = get_a_camera_location(camera_location)
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=_location, rotation=_rotation,scale=(1, 1, 1))
    _camera = bpy.context.selected_objects[0]
    _constraint = _camera.constraints.new(type='TRACK_TO')
    _constraint.track_axis = 'TRACK_NEGATIVE_Z'
    _constraint.up_axis = 'UP_Y'
    _camera.parent = cam_empty
    _constraint.target = cam_empty
    _constraint.owner_space = 'LOCAL'

    bpy.context.view_layer.update()

    bpy.ops.object.select_all(action='DESELECT')
    cam_empty.select_set(True)
    
    if args.random_pose :
        print("random poses")
        delta_z = np.random.uniform(-45, 45, 1)  # left right rotate
        delta_x = np.random.uniform(-15, 15, 1)  # up and down rotate
        delta_y = 0
    else:
        print("fix poses")
        delta_z = 0
        delta_x = 0
        delta_y = 0 
        
    bpy.ops.transform.rotate(value=math.radians(delta_z),orient_axis='Z',orient_type='VIEW')
    bpy.ops.transform.rotate(value=math.radians(delta_y),orient_axis='Y',orient_type='VIEW')
    bpy.ops.transform.rotate(value=math.radians(delta_x),orient_axis='X',orient_type='VIEW')
    bpy.ops.object.select_all(action='DESELECT')

    # set camera
    cam = bpy.data.objects[f'Camera.001']
    location, rotation = cam.matrix_world.decompose()[0:2]
    cam_pose = build_transformation_mat(location, rotation.to_matrix())
    bpy.context.scene.render.resolution_x = args.resolution
    bpy.context.scene.render.resolution_y = args.resolution
    bpy.context.scene.camera.matrix_world = Matrix(cam_pose)

    # render rgba
    bpy.data.scenes['Scene'].render.film_transparent = True
    bpy.context.scene.render.image_settings.color_mode='RGBA'
    bpy.context.scene.render.image_settings.file_format='PNG'
    bpy.data.scenes['Scene'].render.filepath = os.path.join(args.output_folder, object_uid, 'rgba')
    bpy.ops.render.render(write_still=True)

    # render contour
    bpy.ops.scene.freestyle_linestyle_new()
    bpy.data.scenes['Scene'].render.use_freestyle = True
    bpy.context.scene.svg_export.use_svg_export = True
    freestyle_settings = bpy.data.scenes['Scene'].view_layers['ViewLayer'].freestyle_settings
    # select external contour only
    linesets = freestyle_settings.linesets['LineSet']
    linesets.select_silhouette = False
    linesets.select_crease = False
    linesets.select_border = False
    linesets.select_external_contour = True
    # line style setting
    linestyle = freestyle_settings.linesets['LineSet'].linestyle
    linestyle.thickness_position = 'INSIDE'
    linestyle.caps = 'ROUND'
    linestyle.chaining = 'SKETCHY'
    for i in range(6):
        linestyle.thickness = i*5+1+np.random.randint(5)
        bpy.data.scenes['Scene'].render.filepath = os.path.join(args.output_folder, object_uid, f"{i:03d}_" + 'contour')
        bpy.ops.render.render(write_still=False)


if __name__ == "__main__":
    try:
        idx = sys.argv.index("--")
        script_args = sys.argv[idx + 1:]
    except ValueError as e:  # '--' not in the list:
        script_args = []

    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument("--object_path", type=str, required=True, help="Path to the object file")
    parser.add_argument('--output_folder', type=str, default='output',
                        help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Resolution of the images.')
    parser.add_argument('--ortho_scale', type=float, default=1.25,
                        help='ortho rendering usage; how large the object is')
    parser.add_argument('--random_pose', action='store_true',
                        help='whether randomly rotate the poses to be rendered')     
    args = parser.parse_args(script_args)

    start_i = time.time() 
    save_images(args)
    end_i = time.time()
    print("Finished", args.object_path, "in", end_i - start_i, "seconds")
