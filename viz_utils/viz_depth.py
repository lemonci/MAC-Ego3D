import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# Load the images
idx = 250
img1 = np.array(o3d.io.read_image(f'/media/hdd/xiaohao/data/7Scene_Raw/heads/seq-02/seq-02/frame-{idx-1:06}.depth.png'))  # GT
img2 = np.array(o3d.io.read_image(f'/home/jyuzhu/CP-SLAM/cp-slam/experiments/heads/viz_2/render_depth_{idx:05}.png')) #cp
img3 = np.array(o3d.io.read_image(f'/media/hdd/xiaohao/experiments_ma_7scene/results_quality_nv/heads/depth/agent_0_depth_{idx-1}.png')) #ma
# img1 = np.array(o3d.io.read_image('/media/hdd/xiaohao/data/cpslam_data_icp/cp-slam/Apart-0/apart_0_part1/depth_images/depth000300.png'))  # GT
# img2 = np.array(o3d.io.read_image('/home/jyuzhu/CP-SLAM/cp-slam/experiments/apart_0/viz_1/render_depth_00300.png')) #cp
# img3 = np.array(o3d.io.read_image('/media/hdd/xiaohao/experiments_ma_ulim/results_quality/Apart-0/depth/agent_0_depth_299.png')) #ma

# Determine the global min and max across all images for a consistent color scale
img1[img1 > 10000] = 0
img1 = img1 / 1000.0
img2 = img2 / 1000.0
img3 = img3 / 1000.0
# img1[img1 >= 65535] = 0
# img1 = img1 / 6553.5
# img2 = img2 / 6553.5
# img3 = img3 / 1000.0
print(np.max(img1),np.max(img2),np.max(img3))
vmin = min(np.min(img1), np.min(img2), np.min(img3))
vmax = max(np.max(img1), np.max(img2), np.max(img3))

# Create a figure to hold the plots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot each image
cmap = 'jet'  # Choose a colormap that represents depth well, 'jet' is another good option for depth
axs[0].imshow(img1, cmap=cmap,vmin=vmin,vmax=vmax)
axs[0].set_title('Ground Truth')
axs[0].axis('off')  # Turn off axis for cleaner look

axs[1].imshow(img2, cmap=cmap,vmin=vmin,vmax=vmax)
axs[1].set_title('CP-SLAM')
axs[1].axis('off')

axs[2].imshow(img3, cmap=cmap,vmin=vmin,vmax=vmax)
axs[2].set_title('MAGIC-Recon')
axs[2].axis('off')

# Display the plot
plt.tight_layout()
# plt.show()
plt.savefig("viz_depth.png", bbox_inches='tight',dpi=600)