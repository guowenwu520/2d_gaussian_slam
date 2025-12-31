import os

def count_planes_in_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # 跳过注释行（如以 '#' 开头的）
        data_lines = [line for line in lines if not line.strip().startswith('#') and line.strip()]
        return len(data_lines)

def process_all_txt_files(folder_path):
    total_planes = 0
    file_plane_counts = []
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('_label.txt')]
    max_planes = 0
    min_planes = 0
    for txt_file in txt_files:
        full_path = os.path.join(folder_path, txt_file)
        plane_count = count_planes_in_txt(full_path)
        file_plane_counts.append((txt_file, plane_count))
        total_planes += plane_count
        if plane_count > max_planes:
            max_planes = plane_count
        if plane_count < min_planes or min_planes == 0:
            min_planes = plane_count

    avg_planes = total_planes / len(txt_files) if txt_files else 0

    print(f"📄 总共 {len(txt_files)} 个 .txt 文件")
    print(f"🧱 所有文件中总共 {total_planes} 个平面")
    print(f"📊 每个文件平均平面数：{avg_planes:.2f}")
    print(f"📊 最大平面数：{max_planes}")
    print(f"📊 最小平面数：{min_planes}")
    print("\n📋 各文件平面数量如下：")
    # for name, count in file_plane_counts:
    #     print(f"  {name}: {count} 个平面")

# 使用方法：替换成你自己的路径
# scenes = ["rgbd_dataset_freiburg1_desk","rgbd_dataset_freiburg1_desk2","rgbd_dataset_freiburg1_room","rgbd_dataset_freiburg2_xyz","rgbd_dataset_freiburg3_long_office_household"]
# for scene in scenes:
#    print(f"📂 正在处理 {scene} 数据集...")
#    your_folder = f'/home/guowenwu/workspace/indoor_GS_SLAM/RGBD_GS_SLAM/datasets/tum/{scene}/plane'  # 替换为你的实际路径
#    process_all_txt_files(your_folder)

#scenes = ["office0","office1","office2","office3","office4","room0","room1","room2"]
scenes = ["office0"]
for scene in scenes:
   print(f"📂 正在处理 {scene} 数据集...")
   your_folder = f'/home/wuxiangyu/Desktop/data/2d_gaussian_slam/datasets/replica/{scene}/label'  # 替换为你的实际路径
   process_all_txt_files(your_folder)