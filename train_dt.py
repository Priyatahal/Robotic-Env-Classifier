import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter
import Preprocessing
import feature_extraction
import feature_extractor
import EmRo_optimizer
import warnings
warnings.filterwarnings('ignore')
#%%
model = feature_extractor.Conv_patct_swin_transformer()
# For reading dataset
path =  "Dataset"    
imgs = []; fils = [];label = []; img_features = []
for folders in os.listdir(path):
    for files in os.listdir(path+"/"+folders):
        fils.append(files)
        print(files)
        label.append(folders)
        if files.endswith(".jpg"):
            img = cv2.imread(path+"/"+folders+"/"+files)
            img = cv2.resize(img,[300,224])
            En_Im = Preprocessing.CLFAHE(img)
            # Convert image to float32
            resized_image = En_Im.astype(np.float32)
            # Normalize image to the range [0, 1]
            resized_image /= 255.0
            filtered_image = Preprocessing.rolling_guidance_filter(resized_image, sigma_s=0.3, sigma_r=0.3, num_iterations=1)
            # Convert the filtered image to the range [0, 255]
            filtered_image *= 255.0
            # Convert the filtered image to uint8
            filtered_image = filtered_image.astype(np.uint8)
            color_img = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2RGB)
            
            color_img = cv2.resize(color_img,(224,224))
            color_img = np.expand_dims(color_img,axis=0)
            features = model.predict(color_img)
            img_features.append(features)
            np.save("Files/img_features",img_features)
            # print(features)
    
# img_features = np.concatenate([arr for arr in img_features])            
labels = label[1:]    
#%%
# Feature selection
# Set the parameters
nPop = 208
Dim = 100
Max_iter = 15
lb = -1
ub = 1
# pim = lb + sequence*(np.subtract(ub , lb))
X = np.random.uniform(lb, ub, size=(20,nPop, nPop,3))
y = np.load("Files/Labels.npy")
data = np.load("Img_features.npy")
GbestPosition, Curve = EmRo_optimizer.MRFO(X, y, nPop, Dim, Max_iter, lb, ub)
selected_features = GbestPosition > 0  
selected_data = X[:, np.load("Files/selected_features.npy"),:,:]
# np.save("Files/selected_features1",selected_features)
# selected_data = np.load("Files/selected_data.npy")
selected_data = selected_data[:,:,np.load("Files/selected_features.npy"),:]




