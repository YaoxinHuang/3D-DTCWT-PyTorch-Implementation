# %%
# Example 1: 2D Image for DTCWT
import torch
import numpy as np

from PIL import Image as Image
from torchvision.transforms.functional import pil_to_tensor
from matplotlib import pyplot as plt

from dtcwt import DTCWTForward
from dtcwt.misc import normalize_01

img = Image.open(r'../assets/musk.png').convert('RGB') # [W, H]
img_l = img.convert('L')
img_t = torch.tensor(pil_to_tensor(img)).unsqueeze(0).to(torch.float32) / 255. # [N, C, H, W]

# 2D DTCWT

# Low-Frequency Components
plt.figure(figsize=(10, 10))
for i in range(0, 6):
    dtcwt = DTCWTForward(input_dim=2, J=i, skip_hps=True)
    low, high = dtcwt(img_t)
    plt.subplot(2, 3, i+1)
    plt.title("Low Level: {}".format(i+1))
    plt.imshow(normalize_01(low[0].permute(1,2,0).numpy(), torch.max(low).numpy(), torch.min(low).numpy()), cmap='gray')
plt.show()
plt.close()  # Close the previous figure

# High-Frequency Components
plt.figure(figsize=(10, 10))
dtcwt = DTCWTForward(input_dim=2, J=3, skip_hps=False)
low, high = dtcwt(img_t)
high = high[0]
for i in range(high.shape[2]):
    img = high[0, :, i,..., 1]
    plt.subplot(2, 3, i+1)
    plt.title("Orientation: {}".format(i+1))
    plt.imshow(normalize_01(img, torch.max(img), torch.min(img)).permute(1, 2, 0).numpy(), cmap='gray')
plt.show()  # Display the new figure
print(low.shape, high.shape)

# %%
# Example 2: 3D Example for DTCWT
import torch
from dtcwt import DTCWTForward
from dtcwt.misc import normalize_01

input = torch.randn(1, 1, 100, 100, 100) # [N, C, D, H, W]
dtcwt = DTCWTForward(input_dim=3, J=3, skip_hps=False)
low, high = dtcwt(input)
print(low.shape, high[-1].shape)