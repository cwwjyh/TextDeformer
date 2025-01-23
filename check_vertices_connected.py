#虚拟环境为base
import trimesh

def check_vertices_connected(obj_file):
    # 加载OBJ文件
    mesh = trimesh.load(obj_file)

    # 获取所有面
    faces = mesh.faces

    # 创建一个集合来存储所有连接的顶点
    connected_vertices = set()

    # 遍历每个面，记录连接的顶点
    for face in faces:
        connected_vertices.update(face)

    # 检查每个顶点是否都在连接的顶点集合中
    all_vertices = set(range(len(mesh.vertices)))
    disconnected_vertices = all_vertices - connected_vertices

    if not disconnected_vertices:
        print("obj_file: ", obj_file)
        print("所有顶点都有边相连。")
    else:
        print(f"以下顶点没有边相连: {disconnected_vertices}")

# 使用示例
# check_vertices_connected('/data/caiweiwei/TextDeformer-main/meshes/pig.obj') #所有顶点都有边相连。
# check_vertices_connected('/data/caiweiwei/TextDeformer-main/meshes/dog.obj') #所有顶点都有边相连。
# check_vertices_connected("/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/cow.obj") #所有顶点都有边相连。
check_vertices_connected('/data/caiweiwei/TextDeformer-main/laptop_convertbyblender.obj')