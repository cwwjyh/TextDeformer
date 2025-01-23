import trimesh


def glb_to_obj(input_path, output_path):
    # 加载GLB文件
    scene = trimesh.load(input_path, file_type='glb')

    # 获取场景中的所有网格
    meshes = [mesh for mesh in scene.geometry.values()]

    # 合并所有网格
    combined_mesh = trimesh.util.concatenate(meshes)

    # 计算顶点和顶点法线的数量
    num_vertices = len(combined_mesh.vertices)
    num_vertex_normals = len(combined_mesh.vertex_normals)

    # 导出为OBJ文件，包含顶点法线信息
    with open(output_path, 'w') as f:
        # 写入顶点法线信息
        for vn in combined_mesh.vertex_normals:
            f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")

        # 写入顶点信息
        for v in combined_mesh.vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # 写入顶点数量和顶点法线数量的注释
        f.write(f"# {num_vertices} vertices, {num_vertex_normals} vertex normals\n")

        # 写入面信息
        for face in combined_mesh.faces:
            f.write(f"f {face[0] + 1}//{face[0] + 1} {face[1] + 1}//{face[1] + 1} {face[2] + 1}//{face[2] + 1}\n")


# 指定输入GLB文件路径和输出OBJ文件路径
input_glb = '/data/caiweiwei/TextDeformer-main/meshes/06417230a4c74e4a93b9d659b25970a1.glb'
output_obj = '/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/laptop.obj'

# 执行转换
glb_to_obj(input_glb, output_obj)
