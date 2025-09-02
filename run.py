import subprocess

# 场景名称列表
#  "room2", "office0", "office1", "office2", "office3", "office4","room1" ,"room0"
# scenes = ["office4"]

#  "fr1_desk",  "fr1_desk2", "fr1_room","fr2_xyz", "fr3_office"'
scenes = ["fr1_desk_2"]
# CUDA 设备 ID
cuda_device = "0"

for scene in scenes:
    # config_path = f"configs/rgbd/replica/{scene}.yaml"
    config_path = f"configs/rgbd/tum/{scene}.yaml"
    cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python slam.py --config {config_path}"
    print(f"正在处理: {scene}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"完成: {scene}\n")
    except subprocess.CalledProcessError as e:
        print(f"出错: {scene}，错误信息如下：\n{e}\n")


# # Re-load the original depth image as grayscale
# depth_img = Image.open(depth_path).convert("I")  # Keep 16/32-bit depth
# depth_np_raw = np.array(depth_img).astype(np.float32)

# # Apply depth scale
# depth_scale = 6553.5
# depth_meters = depth_np_raw / depth_scale  # Convert to meters

# # Camera intrinsics
# fx, fy = 600.0, 600.0
# cx, cy = 599.5, 339.5
# H, W = depth_meters.shape

# # Create pixel coordinate grid
# u, v = np.meshgrid(np.arange(W), np.arange(H))
# z = depth_meters
# x = (u - cx) * z / fx
# y = (v - cy) * z / fy

# # Stack to create 3D point cloud
# points = np.stack((x, y, z), axis=-1)

# # Estimate normals from point cloud using cross product of neighboring vectors
# normals = np.zeros_like(points)
# for i in range(1, H - 1):
#     for j in range(1, W - 1):
#         dzdx = points[i, j + 1] - points[i, j - 1]
#         dzdy = points[i + 1, j] - points[i - 1, j]
#         n = np.cross(dzdx, dzdy)
#         norm = np.linalg.norm(n)
#         if norm > 1e-5:
#             normals[i, j] = n / norm
#         else:
#             normals[i, j] = [0, 0, 0]

# # Convert to [0, 255] for visualization
# normal_map_corrected = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)

# # Save the corrected normal map
# corrected_normal_path = "/mnt/data/normal_map_corrected_from_depth.png"
# Image.fromarray(normal_map_corrected).save(corrected_normal_path)

# corrected_normal_path
