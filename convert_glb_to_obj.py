import open3d as o3d
import numpy as np
# import trimesh

def read_and_save_mesh(input_path, output_path):
    # 读取三角网格
    mesh = o3d.io.read_triangle_mesh(input_path, enable_post_processing=True, print_progress=True)
    vertices = np.asarray(mesh.vertices)
    # 打印网格信息
    print(f"原始网格包含 {len(vertices)} 个顶点和 {len(mesh.triangles)} 个面")
        # # 预处理网格
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()

    vertices = np.asarray(mesh.vertices)
    # 打印网格信息
    print(f"预处理之后的网格包含 {len(vertices)} 个顶点和 {len(mesh.triangles)} 个面")

    # 检查网格是否成功加载
    if mesh.is_empty():
        print("Failed to load mesh.")
        return

    # 保存为 OBJ 格式
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Mesh saved to {output_path}")

# 使用示例
# input_path = '/data/caiweiwei/Open3D-main/data_mesh/objaverse/41c1a302435f42ccb220fe48d491bb92.glb' #cow
# output_path = '/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/cow.obj'
# input_path = '/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/2b0c154662f74aa083e9561bde0b7981.glb'  # 
# output_path = '/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/frog.obj'  #

# input_path = '/data/caiweiwei/TextDeformer-main/meshes/06417230a4c74e4a93b9d659b25970a1.glb'
# output_path = '/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/desktop_re.obj'
# input_path = '/data/caiweiwei/Open3D-main/data_mesh/objaverse/32a399bfa6ad4de7b0328e92a389d732.glb' #可以正常在TextDeformer-orignal中跑通
# output_path = '/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/pig.obj'

# input_path = '/data/caiweiwei/TextDeformer-main/laptop_convertbyblender.obj' #blender工具直接转换的obj都不能加载
# output_path = "/data/caiweiwei/TextDeformer-main/laptop_convertbyblender_clear.obj"


# input_path = '/data/caiweiwei/TextDeformer-main/meshes/merged_mesh/3cb1e468672748e0bb8642a41fbc358e_merge.glb'
# output_path = "/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/flower.obj"
input_path = "/data/caiweiwei/TextDeformer-main/frog_bymeshlabz-novt.obj"
output_path = "/data/caiweiwei/TextDeformer-main/frog_bymeshlabz-novt_clear.obj"
read_and_save_mesh(input_path, output_path)




# import trimesh
# import numpy as np

# def remove_unused_vertices(mesh):
#     # 获取所有面展平后的顶点索引
#     used_vertices = set()
#     for face in mesh.faces:
#         for vertex_index in face:
#             used_vertices.add(vertex_index)
#     # 获取未使用的顶点索引
#     all_vertices = set(range(len(mesh.vertices)))
#     unused_vertices = all_vertices - used_vertices
#     # 删除未使用的顶点
#     # mesh.remove_vertices(list(unused_vertices))
#     mesh.remove_unreferenced_vertices()
#       # 重新索引面
#     mask = np.isin(mesh.faces, list(used_vertices)).any(axis=1)  # 创建掩码
#     mesh.update_faces(mask)  # 更新面索引
#     return mesh

# def save_obj_without_vt(mesh, filename):
#     with open(filename, 'w') as f:
#         # 写入顶点
#         for vertex in mesh.vertices:
#             f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')

#         # 写入法线
#         if mesh.vertex_normals is not None:
#             for normal in mesh.vertex_normals:
#                 f.write(f'vn {normal[0]} {normal[1]} {normal[2]}\n')
#         # 写入面，不写入纹理坐标
#         for face in mesh.faces:
#             # 转换为OBJ的1-based索引
#             f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')

# def main():
#     # 读取GLB文件
#     input_glb = '/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/2b0c154662f74aa083e9561bde0b7981.glb'  # 替换为你的GLB文件路径
#     output_obj = '/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/frog_open3d.obj'  # 输出的OBJ文件路径

#     scene = trimesh.load(input_glb)
#     # breakpoint()
#      # 提取第一个网格
#     mesh = list(scene.geometry.values())[0]  # 将odict_values转换为列表并获取第一个网格对象
#     # 去除没有边相连的点ß
#     mesh = remove_unused_vertices(mesh)
#  # 去除纹理坐标
#     if hasattr(mesh.visual, 'texcoords'):
#         del mesh.visual.texcoords
#     # 保存为OBJ文件，不包含vt字段
#     save_obj_without_vt(mesh, output_obj)
#     print(f'OBJ文件已保存为 {output_obj}')

# if __name__ == '__main__':
#     main()
