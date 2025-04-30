import numpy as np
import matplotlib.pyplot as plt

ssim_mask = np.load("test_ssim/0/axntxmycwd_0.npy")  

plt.figure(figsize=(6, 6))
plt.imshow(ssim_mask, cmap="gray")  
plt.colorbar()  
plt.title("SSIM Mask")
plt.axis("off") 
plt.show()

ssim_mask = np.load("train_ssim/1/abqwwspghj_0.npy")  

plt.figure(figsize=(6, 6))
plt.imshow(ssim_mask, cmap="gray") 
plt.colorbar()  
plt.title("SSIM Mask")
plt.axis("off") 
plt.show()
