import bpy
import os
import sys
import argparse
import numpy as np
import trimesh
from mathutils import Vector


def render(fbx_file, output_dir, mesh_file, frame_end=-1, type='color'):
    if fbx_file.split('.')[-1] == 'fbx':  
        # load fbx
        bpy.ops.import_scene.fbx(filepath=fbx_file)
        armature = bpy.context.object
        armature.scale = (1, 1, 1)

        # you can rotate the character to change the viewpoint if you want
        if fbx_file.split('/')[-1] == 'jumping.fbx':
            armature.delta_rotation_euler[2] = 45

        mesh = bpy.context.selected_objects[1].data
        faces = mesh.polygons
        indices = np.array([face.vertices for face in faces])
        obj = trimesh.load_mesh(mesh_file)

        if frame_end == -1:
            animation_data = bpy.context.object.animation_data
            bpy.context.scene.frame_start = int(animation_data.action.frame_range[0])-1
            bpy.context.scene.frame_end = int(animation_data.action.frame_range[1])-1

            min_x, max_x = float('inf'), float('-inf')
            min_z, max_z = float('inf'), float('-inf')
            for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
                bpy.context.scene.frame_set(frame)
                bbox = bpy.context.selected_objects[1].bound_box
                matrix_world =  bpy.context.selected_objects[1].matrix_world
                for point in bbox:
                    world_point = matrix_world @ Vector(point)
                    min_x = min(min_x, world_point[0])
                    max_x = max(max_x, world_point[0])
                    min_z = min(min_z, world_point[2])
                    max_z = max(max_z, world_point[2])

            # translate mesh to the center position
            armature.delta_location[0] = -(max_x + min_x)/2
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

        else:
            bpy.context.scene.frame_start = 1
            bpy.context.scene.frame_end = frame_end
    else:
        quit()
    
    if type == 'color':
        # color
        save_path = os.path.join(output_dir, 'color')
        
        vert_colors = obj.visual.vertex_colors
        vert_colors = vert_colors[:,0:3] / 255.
        vert_colors = vert_colors[indices].reshape(-1, 3)

        mesh.materials.clear()
        material = bpy.data.materials.new(name='PointColorMaterial')
        mesh.materials.append(material)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        for node in nodes:
            if node.type == 'TEX_IMAGE':
                nodes.remove(node)

        color_layer = mesh.vertex_colors.new(name='VertexColors')
        vertex_colors = mesh.vertex_colors["VertexColors"]

        for i, color in enumerate(vert_colors):
            vertex_colors.data[i].color = (color[0], color[1], color[2], 1)

        vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
        vertex_color_node.layer_name = 'VertexColors'
        output_node = nodes.get('Material Output')
        links = material.node_tree.links
        links.new(vertex_color_node.outputs[0], output_node.inputs[0])
        bpy.context.scene.view_settings.view_transform = 'Standard'

    elif type == 'pos':
        save_path = os.path.join(output_dir, 'pos')
        
        # replace color with correspondence
        vertices = obj.vertices
        mesh.materials.clear()
        v_min, v_max = vertices.min(0), vertices.max(0)
        v_corr = (vertices - v_min) / (v_max - v_min)
        v_corr = v_corr[indices].reshape(-1, 3)

        material = bpy.data.materials.new(name='PointColorMaterial')
        mesh.materials.append(material)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        for node in nodes:
            if node.type == 'TEX_IMAGE':
                nodes.remove(node)

        color_layer = mesh.vertex_colors.new(name='VertexColors')
        vertex_colors = mesh.vertex_colors["VertexColors"]

        for i, color in enumerate(v_corr):
            vertex_colors.data[i].color = (color[0], color[1], color[2], 1)

        vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
        vertex_color_node.layer_name = 'VertexColors'
        output_node = nodes.get('Material Output')
        links = material.node_tree.links
        links.new(vertex_color_node.outputs[0], output_node.inputs[0])
        bpy.context.scene.view_settings.view_transform = 'Standard'

    else:
        print('unsupported rendering type!')
        quit()

    os.makedirs(save_path, exist_ok=True)
    bpy.data.scenes['Scene'].render.filepath = save_path + '/'
    bpy.ops.render.render(animation=True)


if __name__ == '__main__':
    try:
        idx = sys.argv.index("--")
        script_args = sys.argv[idx + 1:]
    except ValueError as e:  # '--' not in the list:
        script_args = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--fbx_file', type=str, help='path to fbx file', required=True)
    parser.add_argument('--output_dir', type=str, help='path to save renderings', required=True)
    parser.add_argument('--mesh_file', type=str, help='path to load obj', required=True)
    parser.add_argument('--num_frame', type=int, help='number of frames', default=-1)
    parser.add_argument('--type', type=str, help='render type', default='color')
    args = parser.parse_args(script_args)
    render(args.fbx_file, args.output_dir, args.mesh_file, args.num_frame, args.type)