# -*- coding: utf-8 -*-
# Importing necessary packages
import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter
#%% 
# Preprocessing
#Image enhancement
#Haar wavelet Quantized histogram (HawQ) 

def CLFAHE(my_img):
    
    image_bw = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY)
    
    #Declaration of the clahe
    #ClipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit = 0.5)
    final_img = clahe.apply(image_bw) + 4
    return final_img

# Reading the image from the present directory

# img = cv2.imread("Dataset/IS-T-LG/['IMG_01_1'] frame 7038.jpg")
# img = cv2.resize(img,[300,224])
# cv2.imshow("original",img)
# En_Im = CLFAHE(img)
# cv2.imshow("Enhanced Image",En_Im)


#%%
# Noise removal
def rolling_guidance_filter(image, sigma_s, sigma_r, num_iterations=3):
    height, width = image.shape[:2]
    I = np.copy(image)
    P = np.copy(image)

    for _ in range(num_iterations):
        I = gaussian_filter(I, sigma_s)
        P = gaussian_filter(P, sigma_s)
        delta = image - P
        guidance = sigma_r * delta
        I += guidance

    return I

# Convert image to float32
# resized_image = En_Im.astype(np.float32)

# Normalize image to the range [0, 1]
# resized_image /= 255.0

# # Apply the rolling guidance filter
# filtered_image = rolling_guidance_filter(resized_image, sigma_s=0.3, sigma_r=0.3, num_iterations=1)
# # Convert the filtered image to the range [0, 255]
# filtered_image *= 255.0
# # Convert the filtered image to uint8
# filtered_image = filtered_image.astype(np.uint8)
# cv2.imshow('Filtered Image', filtered_image)
# color_img = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2RGB)
#%%
# Feature Extraction
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class ConvPatch(nn.Module):
#   def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#     super(ConvPatch, self).__init__()
#     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#     self.norm = nn.LayerNorm([out_channels, 1, 1])
#   def forward(self, x):
#     x = self.conv(x)
#     x = self.norm(x)
#     return x
# class PatchMerging(nn.Module):
#   def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0):
#     super(PatchMerging, self).__init__()
#     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#     self.norm = nn.LayerNorm([out_channels, 1, 1])
#   def forward(self, x):
#     x = self.conv(x)
#     x = self.norm(x)
#     x = F.gelu(x)
#     return x
# class BasicLayer(nn.Module):
#   def __init__(self, in_channels, out_channels, num_blocks, kernel_size, stride=1, padding=0):
#     super(BasicLayer, self).__init__()
#     self.blocks = nn.ModuleList([
#       ConvPatch(in_channels, out_channels, kernel_size, stride, padding)
#       for _ in range(num_blocks)
#     ])
#   def forward(self, x):
#     for block in self.blocks:
#       x = block(x)
#     return x
# class CP_SwinT(nn.Module):
#   def __init__(self, in_channels=3, hidden_dim=96, num_classes=1000):
#     super(CP_SwinT, self).__init__()
#     self.patch_embed = ConvPatch(in_channels, hidden_dim, kernel_size=3, stride=3)
#     self.stage1 = BasicLayer(hidden_dim, hidden_dim, num_blocks=2, kernel_size=3, padding=1)
#     self.patch_merge = PatchMerging(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1) # Updated kernel_size
#     self.stage2 = BasicLayer(hidden_dim * 2, hidden_dim * 2, num_blocks=2, kernel_size=3, padding=1)
#     self.stage3 = BasicLayer(hidden_dim * 2, hidden_dim * 2, num_blocks=2, kernel_size=3, padding=1)
#     self.classifier = nn.Linear(hidden_dim * 2, num_classes)
#   def forward(self, x):
#     x = self.patch_embed(x)
#     x = self.stage1(x)
#     x = self.patch_merge(x)
#     x = self.stage2(x)
#     x = self.stage3(x)
#     x = x.mean(dim=[2, 3]) # Global average pooling
#     x = self.classifier(x)
#     return x
# def load_and_process_image(color_img):
#   image = torch.tensor([color_img], dtype=torch.float32)
#   return image

# color_img = cv2.resize(color_img,(3,3))
# image = load_and_process_image(color_img)
# model = CP_SwinT(in_channels=3, hidden_dim=96, num_classes=100)
# features = model(image).detach().numpy()
# print(features)

