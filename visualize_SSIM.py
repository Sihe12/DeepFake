import numpy as np
import matplotlib.pyplot as plt

# Load the SSIM mask
ssim_mask = np.load("test_ssim/0/axntxmycwd_0.npy")  # Change the path accordingly

# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(ssim_mask, cmap="gray")  # Show as grayscale
plt.colorbar()  # Add color scale for reference
plt.title("SSIM Mask")
plt.axis("off")  # Hide axes
plt.show()

# Load the SSIM mask
ssim_mask = np.load("train_ssim/1/abqwwspghj_0.npy")  # Change the path accordingly

# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(ssim_mask, cmap="gray")  # Show as grayscale
plt.colorbar()  # Add color scale for reference
plt.title("SSIM Mask")
plt.axis("off")  # Hide axes
plt.show()
