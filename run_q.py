import os
import json
import argparse
import numpy as np
import imageio
from tqdm import tqdm


def convert_depth_bin_to_png(scene_dir, output_dir=None):
    depth_bin_path = os.path.join(scene_dir, "iphone/depth.bin")
    metadata_path = os.path.join(scene_dir, "dslr/nerfstudio/transforms_undistorted.json")

    if not os.path.exists(depth_bin_path):
        print(f"❌ No depth.bin found in: {depth_bin_path}")
        return

    if not os.path.exists(metadata_path):
        print(f"❌ No metadata found in: {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    frames = metadata["frames"]
    h, w = metadata["h"], metadata["w"]
    num_frames = len(frames)

    print(f"📦 Processing scene: {scene_dir}")
    print(f"↪ Frame resolution: {h}x{w}, Total frames: {num_frames}")

    with open(depth_bin_path, 'rb') as f:
        depth_data = np.frombuffer(f.read(), dtype=np.float32)

    assert depth_data.size == num_frames * h * w, "Depth bin size does not match metadata dimensions!"
    depth_data = depth_data.reshape((num_frames, h, w))

    if output_dir is None:
        output_dir = os.path.join(scene_dir, "dslr/undistorted_depths")

    os.makedirs(output_dir, exist_ok=True)

    for i, frame in enumerate(tqdm(frames, desc="Saving depth PNGs")):
        image_rel_path = frame["file_path"]  # e.g., '00000.JPG'
        filename = os.path.basename(image_rel_path).replace(".JPG", ".png")
        depth_mm = (depth_data[i] * 1000).astype(np.uint16)
        imageio.imwrite(os.path.join(output_dir, filename), depth_mm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ScanNet++ depth.bin to PNGs")
    parser.add_argument("--root", required=True, help="Root directory containing ScanNet++ scenes")
    parser.add_argument("--scenes", nargs="*", help="List of scene folder names (default: all subdirs in root)")
    args = parser.parse_args()

    root_dir = args.root
    scene_list = args.scenes or [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for scene in scene_list:
        scene_path = os.path.join(root_dir, scene)
        convert_depth_bin_to_png(scene_path)
