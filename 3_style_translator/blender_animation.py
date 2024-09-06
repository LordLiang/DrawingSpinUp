import bpy
import os
import sys
import argparse
import numpy as np
import trimesh
from mathutils import Vector


def render_color_and_pos(fbx_file, mesh_file, output_dir):
    # load fbx
    bpy.ops.import_scene.fbx(filepath=fbx_file)
    armature = bpy.context.object
    armature.scale = (1, 1, 1)

    # you can rotate the character to change the viewpoint if you want
    if fbx_file.split('/')[-1] in ['jumping.fbx', 'zombie.fbx']:
        armature.delta_rotation_euler[2] = np.radians(30)

    obj = bpy.context.selected_objects[1]
    mesh = obj.data
    faces = mesh.polygons
    indices = np.array([face.vertices for face in faces])
    trimesh_obj = trimesh.load_mesh(mesh_file)

    vert_colors = trimesh_obj.visual.vertex_colors
    vert_colors = vert_colors[:,0:3] / 255.
    vert_colors = vert_colors[indices].reshape(-1, 3)

    vertices = trimesh_obj.vertices
    v_min, v_max = vertices.min(0), vertices.max(0)
    vert_pos = (vertices - v_min) / (v_max - v_min)
    vert_pos = vert_pos[indices].reshape(-1, 3)

    animation_data = bpy.context.object.animation_data
    if animation_data:

        # repaint weight automatically
        armature.data.pose_position = 'REST'
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        bpy.ops.paint.weight_from_bones(type='AUTOMATIC')
        bpy.ops.object.mode_set(mode='OBJECT')
        armature.data.pose_position = 'POSE'

        # adjust the view window
        bpy.context.scene.frame_start = int(animation_data.action.frame_range[0])
        bpy.context.scene.frame_end = int(animation_data.action.frame_range[1])
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')
        for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
            bpy.context.scene.frame_set(frame)
            bbox = bpy.context.selected_objects[1].bound_box
            matrix_world =  bpy.context.selected_objects[1].matrix_world
            for point in bbox:
                world_point = matrix_world @ Vector(point)
                min_x = min(min_x, world_point[0])
                max_x = max(max_x, world_point[0])
                min_y = min(min_y, world_point[1])
                max_y = max(max_y, world_point[1])
                min_z = min(min_z, world_point[2])
                max_z = max(max_z, world_point[2])

        # translate mesh to the center position
        armature.delta_location[0] = -(max_x + min_x)/2
        armature.delta_location[1] = max_y - min_y
        armature.delta_location[2] = -(max_z + min_z)/2
        # zoom the camera window to a suitable size    
        ratio = max(max_x-min_x, max_z-min_z)
        if ratio > 1.35:
            size = int(512/1.35*ratio)
            if size%4 > 0:
                size = size+4-size%4 # need to be a multiple of 4
            bpy.data.scenes["Scene"].render.resolution_x = size
            bpy.data.scenes["Scene"].render.resolution_y = size
            bpy.data.cameras["Camera"].ortho_scale = 1.35*(size/512)

    material = bpy.data.materials.new(name='VertexColorMaterial')
    mesh.materials.append(material)
    mesh.vertex_colors.new(name='VertexColors')
    vertex_colors = mesh.vertex_colors["VertexColors"]
    material.use_nodes = True
    nodes = material.node_tree.nodes

    vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
    output_node = nodes.get('Material Output')
    links = material.node_tree.links
    links.new(vertex_color_node.outputs[0], output_node.inputs[0])

    # color
    save_path = os.path.join(output_dir, 'color')
    os.makedirs(save_path, exist_ok=True)
    bpy.data.scenes['Scene'].render.filepath = save_path + '/'
    for i, color in enumerate(vert_colors):
        vertex_colors.data[i].color = (color[0], color[1], color[2], 1)
    bpy.ops.render.render(animation=True)

    # pos and depth
    save_path = os.path.join(output_dir, 'pos')
    os.makedirs(save_path, exist_ok=True)
    bpy.data.scenes['Scene'].render.filepath = save_path + '/'
    for i, color in enumerate(vert_pos):
        vertex_colors.data[i].color = (color[0], color[1], color[2], 1)

    # depth
    depth_save_path = os.path.join(output_dir, 'depth')
    os.makedirs(depth_save_path, exist_ok=True)
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)
    bpy.data.scenes['Scene'].view_layers["ViewLayer"].use_pass_z = True
    render_node = tree.nodes.new(type='CompositorNodeRLayers')
    depth_node = tree.nodes.new(type='CompositorNodeOutputFile')
    depth_node.format.file_format = 'OPEN_EXR'
    depth_node.base_path = ''
    depth_node.file_slots[0].path = depth_save_path + '/'
    tree.links.new(render_node.outputs['Depth'], depth_node.inputs[0])
    bpy.ops.render.render(animation=True)


if __name__ == '__main__':
    try:
        idx = sys.argv.index("--")
        script_args = sys.argv[idx + 1:]
    except ValueError as e:  # '--' not in the list:
        script_args = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--fbx_file', type=str, help='path to fbx file', required=True)
    parser.add_argument('--mesh_file', type=str, help='path to load obj', required=True)
    parser.add_argument('--output_dir', type=str, help='path to save renderings', required=True)
    
    args = parser.parse_args(script_args)
    render_color_and_pos(args.fbx_file, args.mesh_file, args.output_dir)