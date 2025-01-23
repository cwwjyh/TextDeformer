import trimesh

def check_single_mesh(obj_path):
    # 加载OBJ文件
    scene = trimesh.load(obj_path)
    
    # 检查是否为场景
    if isinstance(scene, trimesh.Scene):
        # 获取场景中的所有网格
        meshes = list(scene.geometry.values())
        
        # 检查是否只有一个网格
        if len(meshes) == 1:
            print("该OBJ文件中只有一个网格。")
        else:
            print(f"该OBJ文件中有{len(meshes)}个网格。")
    else:
        print(obj_path)
        print("该OBJ文件中只有一个网格。")


# # 设置OBJ文件路径
# obj_path = "a.obj"  # 替换为你的OBJ文件路径



# 使用示例
check_single_mesh('/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/cow.obj') #该OBJ文件中只有一个网格。
# check_single_mesh('/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/pig.obj')
# check_single_mesh('/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/desktop.obj')