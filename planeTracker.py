import os
import argparse
import numpy as np
from scipy.spatial.distance import cosine, euclidean


class PlaneTracker:
    def __init__(self, normal_thresh=0.1, center_thresh=0.1, max_history=None):
        self.global_id_counter = 1
        self.history_planes = {}  # global_id -> {normal, center}
        self.normal_thresh = normal_thresh  # 法向量余弦相似度阈值（越小越严格）
        self.center_thresh = center_thresh  # 中心点欧氏距离阈值（越小越严格）
        self.max_history = max_history  # 限制历史平面数量

    def match_plane(self, plane):
        """匹配当前平面与历史平面（基于世界坐标系特征）"""
        for gid, feat in self.history_planes.items():
            # 计算法向量余弦相似度（范围[-1,1]，越接近1越相似）
            normal_sim = cosine(plane['normal'], feat['normal'])
            # 计算中心点欧氏距离
            center_dist = euclidean(plane['center'], feat['center'])

            # 检查是否满足相似性阈值（可根据需求调整阈值）
            # print(f"Normal sim: {normal_sim}, Center dist: {center_dist}")
            if normal_sim < self.normal_thresh and center_dist < self.center_thresh:
                return gid
        return None

    def process_frame(self, planes):
        """处理单帧平面，返回带全局ID的平面列表"""
        result = []

        for plane in planes:
            matched_id = self.match_plane(plane)
            if matched_id is not None:
                plane['global_id'] = matched_id  # 复用历史ID
            else:
                plane['global_id'] = self.global_id_counter  # 分配新ID
                self.global_id_counter += 1

            # 保存当前平面的世界坐标特征到历史
            self.history_planes[plane['global_id']] = {
                'normal': plane['normal'].copy(),
                'center': plane['center'].copy()
            }

            result.append(plane)

        # 可选：限制历史平面数量防止内存溢出
        if self.max_history and len(self.history_planes) > self.max_history:
            excess = len(self.history_planes) - self.max_history
            for gid in sorted(self.history_planes.keys())[:excess]:
                del self.history_planes[gid]

        return result


def parse_plane_file(file_path):
    """解析平面文件（格式：每行包含平面参数，前3个为法向量，接下来3个为中心点）"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    planes = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 11:  # 至少需要法向量(3) + 中心点(3) + 其他字段
            continue

        plane = {
            'original_tokens': tokens.copy(),  # 保留原始数据用于输出
            'normal': np.array(list(map(float, tokens[5:8]))),  # 法向量（假设第6-8列）
            'center': np.array(list(map(float, tokens[8:11])))   # 中心点（假设第9-11列）
        }
        planes.append(plane)

    return planes, lines[0] if lines and lines[0].startswith('#') else "# no header"


def write_plane_file(file_path, header, planes):
    with open(file_path, 'w') as f:
        f.write(header.strip() + "\n")
        for p in planes:
            tokens = p['original_tokens']
            tokens[0] = str(p['global_id'])  # replace plane_index with global_id
            f.write(" ".join(tokens) + "\n")

def parse_trajectory_file(traj_path):
    """解析轨迹文件（每行是4x4齐次变换矩阵，共16个数值）"""
    traj_data = {}
    with open(traj_path, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # 将行分割为16个数值，转换为4x4矩阵
            try:
                matrix = np.array(list(map(float, line.split()))).reshape(4, 4)
                # 轨迹文件行号作为帧标识（可根据实际需求调整，如使用文件名）
                frame_id = f"frame_{line_idx:06d}"  # 示例格式：frame_000000
                traj_data[frame_id] = matrix
            except Exception as e:
                print(f"警告：解析轨迹行 {line_idx} 失败，错误：{str(e)}")
                continue
    return traj_data


def transform_plane_to_world(plane, transform_matrix):
    """将平面从相机坐标系转换到世界坐标系"""
    # 相机坐标系下的齐次坐标（添加w=1）
    normal_cam_hom = np.append(plane['normal'], 0)  # 法向量是方向向量，w=0
    center_cam_hom = np.append(plane['center'], 1)   # 点坐标，w=1

    # 转换到世界坐标系（齐次变换）
    normal_world_hom = transform_matrix @ normal_cam_hom
    center_world_hom = transform_matrix @ center_cam_hom

    # 转换回3D坐标（去除齐次分量）
    normal_world = normal_world_hom[:3]
    center_world = center_world_hom[:3]

    # 法向量归一化（避免尺度影响）
    normal_world = normal_world / np.linalg.norm(normal_world)

    return {
        'original_tokens': plane['original_tokens'],
        'normal': normal_world,
        'center': center_world
    }

import open3d as o3d
import numpy as np
import random

def create_plane_mesh(normal, center, size=0.5, color=None):
    """
    根据法向量和中心点构造一个矩形平面 mesh
    """
    # 默认颜色
    if color is None:
        color = [random.random(), random.random(), random.random()]

    # 构造局部坐标轴：给定 normal，计算局部 x、y 方向
    z = normal / np.linalg.norm(normal)
    # 构造任意一个与 z 不平行的向量
    tmp = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    # 构造四个角点（在平面上的小矩形）
    half = size / 2
    corners = [
        center + x * half + y * half,
        center - x * half + y * half,
        center - x * half - y * half,
        center + x * half - y * half,
    ]

    # 创建三角面片
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [2, 3, 0]])
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def visualize_planes(planes):
    """
    planes: 列表，每个元素包含 'normal' 和 'center'
    """
    geometries = []
    for plane in planes:
        mesh = create_plane_mesh(plane['normal'], plane['center'])
        geometries.append(mesh)

    o3d.visualization.draw_geometries(geometries)




def main(input_dir, output_dir, traj_file):
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载轨迹数据（行号 -> 4x4变换矩阵）
    traj_data = parse_trajectory_file(traj_file)
    tracker = PlaneTracker(normal_thresh=0.01, center_thresh=0.5, max_history=1000)

    # 按顺序处理每一帧
    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_data.txt")])
    for frame_idx, frame_file in enumerate(frame_files):
        if frame_idx >30:
            break
        input_path = os.path.join(input_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file)

        # 跳过无轨迹数据的帧（根据行号匹配）
        if frame_idx >= len(traj_data):
            print(f"警告：{frame_file} 无对应轨迹数据（轨迹行数不足），跳过")
            continue

        # 获取当前帧的外参（相机->世界的变换矩阵）
        transform_matrix = traj_data[f"frame_{frame_idx:06d}"]  # 匹配行号格式
        # transform_matrix = np.linalg.inv(transform_matrix) 

        # 1. 解析原始平面数据（相机坐标系下的平面）
        planes, header = parse_plane_file(input_path)

        # 2. 将平面从相机坐标系转换到世界坐标系
        world_planes = []
  
        for plane in planes:
            # 执行坐标变换
            world_plane = transform_plane_to_world(plane, transform_matrix)
            world_planes.append(world_plane)

        # 3. 处理世界坐标系下的平面（跟踪并分配全局ID）
        tracked_planes = tracker.process_frame(world_planes)

        # 4. 写入输出文件（替换原始索引为全局ID）
        write_plane_file(output_path, header, tracked_planes)
        print(f"处理完成：{frame_file} -> {output_path}（使用轨迹行 {frame_idx}）")

# 示例使用
# planes 是你原来的列表，每个 plane 包含 normal 和 center
    # visualize_planes(world_planes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="平面跟踪（世界坐标系）")
    parser.add_argument('-i', required=True, help="输入目录（包含帧平面文件）")
    parser.add_argument('-o', required=True, help="输出目录（保存带全局ID的平面文件）")
    parser.add_argument('-t', required=True, help="轨迹文件路径（traj.txt）")
    args = parser.parse_args()

    main(args.i, args.o, args.t)