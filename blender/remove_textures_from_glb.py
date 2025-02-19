#虚拟环境；base
# 因为objaverse是通过blender渲染的，所以去除纹理和r颜色和材质的code也要用blender的python文件，如果用open3d or trimesh 都会导致去除纹理的ojaverse 的坐标变错

import bpy
import os

def remove_texture_and_materials(input_dir, output_dir):
    """
    去除 3D 网格的纹理、颜色和材质，只保留几何信息。

    参数:
    - input_dir: 输入目录，包含多个 .glb 文件。
    - output_dir: 输出目录，保存处理后的 .glb 文件。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有 .glb 文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".glb"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 清空当前场景
            bpy.ops.wm.read_factory_settings(use_empty=True)

            # 导入 .glb 文件
            bpy.ops.import_scene.gltf(filepath=input_path)

            # 遍历所有对象，去除材质和纹理
            for obj in bpy.data.objects:
                if obj.type == 'MESH':
                    # 去除材质
                    obj.data.materials.clear()

            # 导出处理后的网格为 .glb 文件
            bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')

            print(f"已处理: {input_path} -> {output_path}")

if __name__ == "__main__":
    # 输入目录和输出目录
    input_directory = "/data/caiweiwei/TextDeformer-main/meshes/animals" # 替换为你的输入目录
    output_directory = "/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/animals1" 
     # 调用函数处理
    remove_texture_and_materials(input_directory, output_directory)