import numpy as np
from PIL import Image
import os
import shutil
import open3d as o3d
GROUPS = {
    'chess': 6,
    'heads': 2,
    'fire': 4,
    'office': 10,
    'redkitchen': 14,
    'pumpkin': 8,
    'stairs': 6
}

raw_dataset_path_root = './data/7Scene_Raw/'
new_dataset_path_root = './data/7Scene_ICP/'
for scenes in GROUPS:
    print(f'start process group: {scenes}')
    for vidx in range(GROUPS[scenes]):
        print(f'start process video: {(vidx + 1):02}')
        from_path = os.path.join(raw_dataset_path_root, scenes, f"seq-{(vidx + 1):02}", f"seq-{(vidx + 1):02}")
        
        image_directory = os.path.join(new_dataset_path_root, scenes, f"seq-{(vidx + 1):02}", 'images')
        depth_directory = os.path.join(new_dataset_path_root, scenes, f"seq-{(vidx + 1):02}", 'depth_images')
        traj_out_path = os.path.join(new_dataset_path_root, scenes, f"seq-{(vidx + 1):02}", 'traj.txt')
        print(image_directory)
        # Save RGBD Images
        os.makedirs(image_directory, exist_ok=True)
        os.makedirs(depth_directory, exist_ok=True)
        info = []
        index = 0
        while True:
            rgb_image_path = os.path.join(from_path, f'frame-{index:06}.color.png')
            depth_image_path = os.path.join(from_path, f'frame-{index:06}.depth.png')
            pose_path = os.path.join(from_path, f'frame-{index:06}.pose.txt')
            if not os.path.isfile(pose_path):
                break
            rgb_out = os.path.join(image_directory, f'frame{index:06}.jpg')
            depth_out = os.path.join(depth_directory, f'depth{index:06}.png')
            img = Image.open(rgb_image_path)  
            depth = Image.open(depth_image_path)
            img.save(rgb_out)
            depth.save(depth_out)
            info.append(np.loadtxt(pose_path))
            index += 1
        # Saving the flattened matrices to a text file
        with open(traj_out_path, 'w') as file:
            for matrix in info:
                flattened_matrix = matrix.flatten()  # Ensuring the matrix is flattened to 16 elements
                file.write(' '.join(f"{item:.6f}" for item in flattened_matrix) + '\n')
    print('----------------')
    
    
