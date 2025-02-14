# import pymeshlab

# def laplacian_smooth_with_pymeshlab(input_mesh_path, output_mesh_path, iterations=10, lambda_factor=0.5):
#     """
#     使用 pymeshlab 对网格进行拉普拉斯平滑处理
#     :param input_mesh_path: 输入网格文件路径
#     :param output_mesh_path: 输出网格文件路径
#     :param iterations: 平滑迭代次数（默认10次）
#     :param lambda_factor: 平滑系数（默认0.5）
#     """
#     # 创建 MeshLab 实例
#     ms = pymeshlab.MeshSet()

#     # 打印可用的过滤器
#     print("可用的过滤器列表:")
#     ms.print_filter_list()

#     # 加载网格
#     ms.load_new_mesh(input_mesh_path)

#     # 应用拉普拉斯平滑
#     ms.apply_filter("laplacian_smooth", 
#                     iterations=iterations, 
#                     lambda_value=lambda_factor)

#     # 保存平滑后的网格
#     ms.save_current_mesh(output_mesh_path)

# # 使用示例
# if __name__ == "__main__":
#     # 输入和输出文件路径
#     input_mesh_path = "/data/caiweiwei/TextDeformer-main/outputs/donkey2donkey_gt_4_ref_1000epo_l2_10_25bs/mesh_final/mesh.obj"  # 替换为你的输入文件路径
#     output_mesh_path = "/data/caiweiwei/TextDeformer-main/outputs/donkey2donkey_gt_4_ref_1000epo_l2_10_25bs/mesh_final/smoothed_output.obj"  # 替换为你的输出文件路径

#     # 执行平滑处理
#     laplacian_smooth_with_pymeshlab(input_mesh_path, 
#                                     output_mesh_path, 
#                                     iterations=15,  # 迭代次数
#                                     lambda_factor=0.6)  # 平滑系数

import pymeshlab

def laplacian_smooth_with_pymeshlab(input_mesh_path, output_mesh_path, stepsmoothnum=10):
    """
    使用 pymeshlab 对网格进行拉普拉斯平滑处理
    :param input_mesh_path: 输入网格文件路径
    :param output_mesh_path: 输出网格文件路径
    :param stepsmoothnum: 平滑迭代次数（默认10次）
    """
    # 创建 MeshLab 实例
    ms = pymeshlab.MeshSet()

    # 加载网格
    ms.load_new_mesh(input_mesh_path)

    # 应用坐标拉普拉斯平滑
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=stepsmoothnum)

    # 这里可以添加其他过滤器
    # ms.apply_filter("some_other_filter")

    # 保存平滑后的网格
    ms.save_current_mesh(output_mesh_path)

    # 保存过滤器脚本
    ms.save_filter_script('my_script.mlx')

# 使用示例
if __name__ == "__main__":
    # input_mesh_path = "/data/caiweiwei/TextDeformer-main/outputs/spot2cow_gt_mv_ref_2000epo_l2_10_25bs/mesh_best_total/mesh.obj"  # 替换为你的输入文件路径
    # output_mesh_path = "/data/caiweiwei/TextDeformer-main/outputs/spot2cow_gt_mv_ref_2000epo_l2_10_25bs/mesh_best_total/smoothed_output.obj"  # 替换为你的输出文件路径

    # input_mesh_path = "/data/caiweiwei/TextDeformer-main/outputs/donkey2donkey_gt_4_ref_1000epo_l2_10_25bs/mesh_final/mesh.obj"
    # output_mesh_path = "/data/caiweiwei/TextDeformer-main/outputs/donkey2donkey_gt_4_ref_1000epo_l2_10_25bs/mesh_final/smoothed_output.obj"

    input_mesh_path = "/data/caiweiwei/TextDeformer-main/outputs/donkey2donkey_gt_mv_ref_1000epo_l2_10_25bs/mesh_final/mesh.obj"
    output_mesh_path = "/data/caiweiwei/TextDeformer-main/outputs/donkey2donkey_gt_mv_ref_1000epo_l2_10_25bs/mesh_final/smoothed_output.obj"

    # 执行平滑处理
    laplacian_smooth_with_pymeshlab(input_mesh_path, output_mesh_path, stepsmoothnum=10)