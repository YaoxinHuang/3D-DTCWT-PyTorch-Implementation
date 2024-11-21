
# Example 1: using pywt

import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_wavelets import DTCWTForward
from yaoxin_tools.tools import usual_reader

# 创建一个示例信号（图像）
reader = usual_reader()
image = reader(r'/Users/Yaoxin/Downloads/黄英博证件照.png', 'torch').to(torch.float32)

# 使用 DTCWTForward 进行双树复小波变换
dtcwt = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b', include_scale=False)
yl, highpasses = dtcwt(image)

# 绘制低频分量
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(yl.squeeze().numpy(), cmap='gray')
plt.title('Lowpass Component')
plt.subplot(1, 2, 2)
plt.imshow(image.squeeze().numpy(), cmap='gray')

# 绘制每一级的高频分量
for i in range(len(highpasses)):
    for j in range(highpasses[i].shape[2]):
        # plt.subplot(len(highpasses), highpasses[i].shape[2], i * highpasses[i].shape[2] + j + 1)
        # plt.imshow(np.abs(highpasses[i].squeeze().numpy()[j, ..., 1]), cmap='gray')
        plt.title(f'Highpass Level {i+1} Direction {j+1}')
        plt.axis('off')

plt.show()

# %%
from torch.nn.functional import conv2d
import torch

X = torch.tensor([1,2,3,4,5]).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
X = X.repeat(1,2,1,1) # repaert rows
print(X)
X = torch.cat([X,X], dim=0) # batch = 2
weights = torch.tensor([-1, -1, -1]).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()

# using group = 1, default
Y1 = conv2d(X, weights, padding=0, groups=1)
print(Y1)


# %%
