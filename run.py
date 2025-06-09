import subprocess

# 场景名称列表
#  "room2", "office0", "office1", "office2", "office3", "office4","room1" ,"room0"
scenes = ["room1"]

#  "fr1_desk",  "fr1_desk2", "fr1_room","fr2_xyz", "fr3_office"'
# scenes = ["fr1_room"]
# CUDA 设备 ID
cuda_device = "0"

for scene in scenes:
    config_path = f"configs/rgbd/replica/{scene}.yaml"
    # config_path = f"configs/rgbd/tum/{scene}.yaml"
    cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python slam.py --config {config_path}"
    print(f"正在处理: {scene}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"完成: {scene}\n")
    except subprocess.CalledProcessError as e:
        print(f"出错: {scene}，错误信息如下：\n{e}\n")
