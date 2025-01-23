import trimesh

def check_mesh_integrity(mesh):
    # 检查是否有退化的面或顶点
    if mesh.is_volume:
        print("网格是封闭的，没有退化的面或顶点。")
    else:
        print("网格不是封闭的，可能存在退化的面或顶点。")
    
    # 检查是否有洞
    if mesh.fill_holes():
        print("网格有洞，已尝试填充。")
    else:
        print("网格没有洞。")
    
    # 检查是否有不连贯的部分
    if mesh.is_watertight:
        print("网格是水密的，没有不连贯的部分。")
    else:
        print("网格不是水密的，可能存在不连贯的部分。")

# 加载OBJ文件
# obj_path ='/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/cow.obj' # 替换为你的OBJ文件路径
obj_path = '/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/pig.obj'
mesh = trimesh.load(obj_path)

# 调用检查函数
check_mesh_integrity(mesh)