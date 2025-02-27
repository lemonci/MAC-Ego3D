import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# Load the images
for idx in range(50,1000,50):
    img1 = np.array(o3d.io.read_image(f'/media/hdd/xiaohao/experiments_ma_7scene/results_quality_nv/stairs_1/rgb/agent_1_rgb_{idx-1}.png')) # RGB
    img2 = np.array(o3d.io.read_image(f'/media/hdd/xiaohao/experiments_ma_7scene/results_quality_nv/stairs_1/depth/agent_1_depth_{idx-1}.png')) # Depth
    # img1 = np.array(o3d.io.read_image(f'/media/hdd/xiaohao/experiments_ma_ulim/results_quality_nv/Office-0/rgb/agent_1_rgb_{idx-1}.png')) # RGB
    # img2 = np.array(o3d.io.read_image(f'/media/hdd/xiaohao/experiments_ma_ulim/results_quality_nv/Office-0/depth/agent_1_depth_{idx-1}.png')) # Depth

    # Determine the global min and max across all images for a consistent color scale
    img2[img2 > 10000] = 0
    img2 = img2 / 1000.0
    # img1[img1 >= 65535] = 0
    # img1 = img1 / 6553.5
    # img2 = img2 / 6553.5
    # img3 = img3 / 1000.0
    vmin = np.min(img2)
    vmax = np.max(img2)

    # Create a figure to hold the plots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot each image
    cmap = 'jet'  # Choose a colormap that represents depth well, 'jet' is another good option for depth
    axs[0].imshow(img1)
    axs[0].set_title('Render RGB')
    axs[0].axis('off')  # Turn off axis for cleaner look

    cax = axs[1].imshow(img2, cmap=cmap,vmin=vmin,vmax=vmax)
    axs[1].set_title('Render Depth')
    axs[1].axis('off')

    cbar = fig.colorbar(cax)
    cbar.set_label('Depth scale')
    # Display the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"viz/viz_{int(idx / 50)}.png", bbox_inches='tight',dpi=600)
    plt.close()