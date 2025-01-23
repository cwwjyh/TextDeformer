import bpy

def import_and_merge_glb(file_path, output_path):
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Import the GLB file
    bpy.ops.import_scene.gltf(filepath=file_path)

    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Select all mesh objects
    meshes = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
            meshes.append(obj)

    if meshes:
        # Join selected meshes
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()

    # Deselect all again
    bpy.ops.object.select_all(action='DESELECT')

    # Select only the combined mesh
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)

    # Export the combined mesh
    bpy.ops.export_scene.gltf(filepath=output_path, use_selection=True)

# Example usage

file_path = "/data/caiweiwei/TextDeformer-main/meshes/0002c6eafa154e8bb08ebafb715a8d46.glb"
output_path = "/data/caiweiwei/TextDeformer-main/meshes/merged_mesh/0002c6eafa154e8bb08ebafb715a8d46_merge.glb"
import_and_merge_glb(file_path, output_path)