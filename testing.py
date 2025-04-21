# -*- coding: utf-8 -*-
import cv2
import numpy as np
import Preprocessing
import feature_extractor
import EmRo_optimizer
import warnings
warnings.filterwarnings('ignore')
from tkinter import filedialog
import tensorflow as tf
from __pycache__.utils import *

#%%
def testing():
    # Preprocessing
    read_img = filedialog.askopenfilename()
    img = cv2.imread(read_img)
    img = cv2.resize(img,[300,224])
    cv2.imshow("Original_img",img)
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
    image = cv2.cvtColor(En_Im, cv2.COLOR_GRAY2RGB)
    cv2.imshow("Pre processed img",image)
    #%%
    # Feature Extraction
    color_img = cv2.resize(image,(224,224))
    color_img = np.expand_dims(color_img,axis=0)
    model = feature_extractor.Conv_patct_swin_transformer()
    features = model.predict(color_img)
    
     
    #%%
    # Feature selection
    y = np.load("Files/Labels.npy")
    selected_data = features[:, np.load("Files/selected_features1.npy"),:,:]
    selected_data = selected_data[:,:,np.load("Files/selected_features1.npy"),:]
    
    #%%
    # Classification
    clas = ["DS-T-LG" ,"DW-S","DW-T-O","IS-S","IS-T-DW","IS-T-LG","LG-S","LG-T-DS","LG-T-DW","LG-T-IS","LG-T-O","LG-T-SE"]
    
    @predict
    def predition(y,selected_data,read_img):
        num_classes = len(np.unique(y))
        model_1 = tf.saved_model.load("EffNet_1-20230615T121459Z-001/EffNet_1")
        model1_pred = model_1(selected_data)
        
        model_2 = tf.saved_model.load("EffNet_2-20230615T121535Z-001/EffNet_2")
        model2_pred = model_2(selected_data)
         
        model_3 = tf.saved_model.load("EffNet3-20230615T121703Z-001/EffNet3")
        model3_pred = model_3(selected_data)
        
        final_preds = np.argmax(model1_pred + model2_pred + model3_pred, axis=1)
        clss = clas[int(final_preds)]
        return clss
    clss = predition(y,selected_data,read_img)
    propose_pred = clas[clss]
    
    #%%
    # Existing
    num_class = len(np.unique(y))
    import Classifier
    ResNet_mdl = Classifier.ResNet(selected_data,num_class)
    ResNet_pred = ResNet_mdl.predict(selected_data)
    DCNN_mdl = Classifier.DCNN(selected_data,num_class)
    DCNN_pred = DCNN_mdl.predict(selected_data)
    
    EffNet1_mdl = Classifier.EffNet_1(selected_data,num_class)
    EffNet1_pred = EffNet1_mdl.predict(selected_data)
    
    EffNet2_mdl = Classifier.EffNet_2(selected_data,num_class)
    EffNet2_pred = EffNet2_mdl.predict(selected_data)
    
    EffNet3_mdl = Classifier.EffNet_3(selected_data,num_class)
    EffNet3_pred = EffNet3_mdl.predict(selected_data)
    print()
    print("Your Selected input's class is:",propose_pred)

    import performance
    performance.results(propose_pred,DCNN_pred,ResNet_pred,EffNet1_pred,EffNet2_pred,EffNet3_pred)
    
testing()