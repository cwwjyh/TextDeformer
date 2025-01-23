import pymeshlab

def convert_glb_to_obj(input_glb, output_obj):
    # 创建一个新的MeshSet
    ms = pymeshlab.MeshSet()
    
    # 加载GLB文件
    ms.load_new_mesh(input_glb)

    #   # 打印可用的过滤器列表
    # print("Available filters:")
    # ms.print_filter_script()
    
    # # 去除没有边相连的顶点
    ms.apply_filter('remove_unreferenced_vertices')
    # # 去除没有边相连的顶点
    # ms.apply_filter('meshing_remove_unreferenced_vertices')
    
    # 保存为OBJ文件
    ms.save_current_mesh(output_obj, save_vertex_color=False, save_face_color=False, save_wedge_texcoord=False, save_wedge_normal=False, save_polygonal=False)

# 设置输入和输出文件路径
# input_glb = "/data/caiweiwei/TextDeformer-main/meshes/06417230a4c74e4a93b9d659b25970a1.glb"  # 替换为你的GLB文件路径
# output_obj = "/data/caiweiwei/TextDeformer-main/meshes/no_textures_mesh/laptop_with_meshlab2.obj"  # 替换为你的OBJ文件路径

input_glb = '/data/caiweiwei/TextDeformer-main/frog_bymeshlabz-novt.obj'
output_obj = '/data/caiweiwei/TextDeformer-main/frog_bymeshlabz-novt_clear.obj'

# 调用转换函数
convert_glb_to_obj(input_glb, output_obj)