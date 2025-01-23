#这个方式处理3D mesh之后可以背textdeformer跑起来，虚拟环境：base
#修改textdeformer处理雅可比矩阵的方式：使用 Cholesky 分解换成使用稀疏LU分解时。但是使用LU分解也需要进行3D mesh处理才能跑起来

import numpy as np
import pymeshlab as ml
from typing import Tuple
from numpy.typing import NDArray

def cleanup_mesh(
    input_glb_path: str,
    output_obj_path: str,
) -> None:
    """
    Applies a series of filters to the input mesh from a GLB file and saves the output as an OBJ file.

    For instance,
    - Duplicate vertex removal
    - Unreference vertex removal
    - Remove isolated pieces
    """

    # Load the mesh from the GLB file
    meshset = ml.MeshSet()
    meshset.load_new_mesh(input_glb_path)

    # 删除重复的面
    meshset.meshing_remove_duplicate_vertices()

    # 删除未使用的顶点
    meshset.meshing_remove_unreferenced_vertices()

     
    # meshset.meshing_remove_isolated_vertices()

    meshset.meshing_merge_close_vertices()

    #  删除孤立顶点
    meshset.meshing_remove_connected_component_by_diameter()

    # meshset.meshing_remove_connected_component_by_face_number() #这个对与处理car.obj比较重要，如果没有这步，car.ob跑不同textdeformer

    # 删除零面积的面
    meshset.meshing_remove_null_faces()

    meshset.meshing_remove_folded_faces()

    # Save the processed mesh to an OBJ file
    meshset.save_current_mesh(output_obj_path)

# 使用示例
# cleanup_mesh("/data/caiweiwei/TextDeformer-main/meshes/clean_mesh/bluebird_animated131.obj", "/data/caiweiwei/TextDeformer-main/meshes/clean_mesh/bird.obj")
cleanup_mesh("/data/caiweiwei/TextDeformer-main/meshes/clean_mesh/Brown_Horse105.obj", '/data/caiweiwei/TextDeformer-main/meshes/clean_mesh/horse.obj')