import os
import logging
from argparse import ArgumentParser
import shutil

parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="SIMPLE_PINHOLE", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--include_folders", nargs="*", default=[], help="Subfolders to include inside input/")
args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

# ---- Flatten images from subfolders into a new folder ----
input_root = os.path.join(args.source_path, "input")
flatten_input = os.path.join(args.source_path, "input_flattened")
os.makedirs(flatten_input, exist_ok=True)

all_subfolders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]
subfolders = args.include_folders if args.include_folders else all_subfolders

has_subfolders = any(os.path.isdir(os.path.join(input_root, f)) for f in os.listdir(input_root))

if not has_subfolders:
    print(f"No subfolders found in '{input_root}', using it directly.")
    image_path_arg = input_root
    camera_flag = "--ImageReader.single_camera 1"
else:
    # ---- Flatten images from selected subfolders into a new folder ----
    all_subfolders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]
    subfolders = args.include_folders if args.include_folders else all_subfolders

    if not subfolders:
        logging.error("No valid subfolders found in 'input/' to include.")
        exit(1)

    print(f"Flattening subfolders {subfolders} into: {flatten_input}")
    os.makedirs(flatten_input, exist_ok=True)

    for folder in subfolders:
        full_folder_path = os.path.join(input_root, folder)
        if not os.path.isdir(full_folder_path):
            continue
        for file in os.listdir(full_folder_path):
            src = os.path.join(full_folder_path, file)
            if os.path.isfile(src):
                dst = os.path.join(flatten_input, f"{folder}_{file}")
                shutil.copy2(src, dst)

    image_path_arg = flatten_input
    camera_flag = "--ImageReader.single_camera 1"
# ---- COLMAP Pipeline ----
if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    # Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor " \
        "--database_path " + args.source_path + "/distorted/database.db " \
        "--image_path " + image_path_arg + " " \
        + camera_flag + " " \
        "--ImageReader.camera_model " + args.camera + " " \
        "--SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher " \
        "--database_path " + args.source_path + "/distorted/database.db " \
        "--SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # Mapping
    mapper_cmd = colmap_command + " mapper " \
        "--database_path " + args.source_path + "/distorted/database.db " \
        "--image_path " + image_path_arg + " " \
        "--output_path " + args.source_path + "/distorted/sparse " \
        "--Mapper.ba_global_function_tolerance=0.000001"
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

# Image undistortion
img_undist_cmd = colmap_command + " image_undistorter " \
    "--image_path " + image_path_arg + " " \
    "--input_path " + args.source_path + "/distorted/sparse/0 " \
    "--output_path " + args.source_path + " " \
    "--output_type COLMAP"
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Image undistorter failed with code {exit_code}. Exiting.")
    exit(exit_code)

# Move sparse files
files = os.listdir(os.path.join(args.source_path, "sparse"))
os.makedirs(os.path.join(args.source_path, "sparse/0"), exist_ok=True)
for file in files:
    if file == '0':
        continue
    src = os.path.join(args.source_path, "sparse", file)
    dst = os.path.join(args.source_path, "sparse/0", file)
    shutil.move(src, dst)

# Resize images if requested
if args.resize:
    print("Copying and resizing...")

    for scale, percent in [("2", 50), ("4", 25), ("8", 12.5)]:
        resized_dir = os.path.join(args.source_path, f"images_{scale}")
        os.makedirs(resized_dir, exist_ok=True)

    image_dir = os.path.join(args.source_path, "images")
    for file in os.listdir(image_dir):
        src = os.path.join(image_dir, file)
        for scale, percent in [("2", 50), ("4", 25), ("8", 12.5)]:
            dst = os.path.join(args.source_path, f"images_{scale}", file)
            shutil.copy2(src, dst)
            cmd = f'{magick_command} mogrify -resize {percent}% "{dst}"'
            exit_code = os.system(cmd)
            if exit_code != 0:
                logging.error(f"{percent}% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

print("Done.")
