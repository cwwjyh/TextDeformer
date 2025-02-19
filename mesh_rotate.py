import trimesh
import numpy as np
import argparse



def rotate_mesh(mesh_path, output_path, angle_deg, axis):
    """
    旋转 3D 网格，并保存为新的文件。

    :param mesh_path: 输入网格文件路径
    :param output_path: 输出网格文件路径
    :param angle_deg: 旋转角度（度数）
    :param axis: 旋转轴，'x'，'y'，'z' 之一
    """
    # 加载网格
    mesh = trimesh.load(mesh_path)

    # 将角度从度数转换为弧度
    angle_rad = np.deg2rad(angle_deg)

    # 选择旋转轴
    if axis == 'x':
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [1, 0, 0])  # 绕 X 轴旋转
    elif axis == 'y':
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])  # 绕 Y 轴旋转
    elif axis == 'z':
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])  # 绕 Z 轴旋转
    else:
        raise ValueError("旋转轴必须是 'x', 'y', 或 'z'")

    # 应用旋转变换
    mesh.apply_transform(rotation_matrix)

    # 保存旋转后的网格
    mesh.export(output_path)

    print(f"网格旋转 {angle_deg} 度，并保存到 {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--name", type=str)
    parser.add_argument("--angle", type=int, default=-90)
    parser.add_argument("--axis", type=str, default='z')
    args = parser.parse_args()
    angle = args.angle
    asix = args.axis
    name = args.name
    # 示例用法
    mesh_path = f'meshes/{name}.obj'  # 输入网格文件路径
    output_path = f'meshes/rotated/{name}-rotate.obj'  # 输出网格文件路径
    rotate_mesh(mesh_path, output_path, angle, asix)  # 绕 Z 轴旋转 45 度并保存

