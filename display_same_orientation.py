from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

path1 = "same_orientation\\a_0.945667028427124_0.186184361577034_-0.24610111117362976_-0.10238886624574661_.jpeg"
img1 = Image.open(path1).convert('L')
img_data1 = np.asarray(img1) 
print(img_data1.shape)
img_data1 = img_data1 * 255 / np.max(img_data1)

path2 = "steelData/a_0.9159111976623535_-0.0681811273097992_0.3638837933540344_-0.15506961941719055_.jpeg"
img2 = Image.open(path2).convert('L')
img_data2 = np.asarray(img2) 
img_data2 = img_data2 * 255 / np.max(img_data2)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img_data1)
ax2.imshow(img_data2)
plt.show()