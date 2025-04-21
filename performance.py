import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as font_manager
from sklearn.preprocessing import LabelBinarizer
import validation as V
from __pycache__.utils import *
import itertools
import math 
from sklearn.metrics import matthews_corrcoef
#%%
predition = np.load("Files/predition.npy")
def results(proposed_pred,Dcnn,Resnet,effnet_1,effnet_2,effnet_3):
    def calculation(y_tst,y_pred):
        cnf_matrix = confusion_matrix(y_tst,y_pred)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float);FN = FN.astype(float);
        TP = TP.astype(float);TN = TN.astype(float);  
        TPR = TP/(TP+FN);TNR = TN/(TN+FP) 
        PPV = TP/(TP+FP);NPV = TN/(TN+FN)
        FPR1 = FP/(FP+TN);FNR1 = FN/(TP+FN)
        FDR = FP/(TP+FP)
        ACC = (TP+TN)/(TP+FP+FN+TN)  
        Accuracy1=sum(ACC)/len(ACC)
        acc1 = np.mean(Accuracy1)
        recall1=TP/(TP+FN)
        Rel = np.max(recall1)
        recall1 = round(Rel)
        precision1=TP/(TP+FP)
        prscon = np.max(precision1)
        precision1 = round(prscon)
        fs=((2*precision1*recall1)/(precision1+recall1))
        f1_score1 = round(fs)
        print("Accuracy : ",(Accuracy1));
        print("Precision : ",(precision1));
        print("Recall : ",(recall1));print("F_Measure : ",(f1_score1))
        return Accuracy1,precision1,TPR,FPR1,recall1,f1_score1,cnf_matrix
    
    def calculation1(y_tst,y_pred):
        cnf_matrix = confusion_matrix(y_tst,y_pred)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float);FN = FN.astype(float);
        TP = TP.astype(float);TN = TN.astype(float);  
        TPR = TP/(TP+FN);TNR = TN/(TN+FP) 
        PPV = TP/(TP+FP);NPV = TN/(TN+FN)
        FPR1 = FP/(FP+TN);FNR1 = FN/(TP+FN)
        FDR = FP/(TP+FP)
        ACC = (TP+TN)/(TP+FP+FN+TN)  
        Accuracy1=sum(ACC)/len(ACC)
        acc1 = np.mean(Accuracy1)
        recall1=TP/(TP+FN)
        Rel1 = np.max(recall1)
        recall1 = round(Rel1)
        precision1=TP/(TP+FP)
        prscon1 = np.max(precision1)
        precision1 = round(prscon1)
        fs1=((2*precision1*recall1)/(precision1+recall1))
        f1_score1 = round(fs1)
        print("Accuracy : ",(Accuracy1));
        print("Precision : ",(precision1));
        print("Recall : ",(recall1));print("F_Measure : ",(f1_score1))
        return Accuracy1,precision1,TPR,FPR1,recall1,f1_score1,cnf_matrix
    
    def calculation2(y_tst,y_pred):
        cnf_matrix = confusion_matrix(y_tst,y_pred)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float);FN = FN.astype(float);
        TP = TP.astype(float);TN = TN.astype(float);  
        TPR = TP/(TP+FN);TNR = TN/(TN+FP) 
        PPV = TP/(TP+FP);NPV = TN/(TN+FN)
        FPR1 = FP/(FP+TN);FNR1 = FN/(TP+FN)
        FDR = FP/(TP+FP)
        ACC = (TP+TN)/(TP+FP+FN+TN)  
        Accuracy1=sum(ACC)/len(ACC)
        acc1 = np.mean(Accuracy1)
        recall1=TP/(TP+FN)
        Rel2 = np.max(recall1)
        recall1 = round(Rel2)
        # print("recall1",recall1)
        precision1=TP/(TP+FP)
        prscon2 = np.max(precision1)
        precision1 = round(prscon2)
        fs2=((2*precision1*recall1)/(precision1+recall1))
        f1_score1 = round(fs2)
        print("Accuracy : ",(Accuracy1));
        print("Precision : ",(precision1));
        print("Recall : ",(recall1));print("F_Measure : ",(f1_score1))
        return Accuracy1,precision1,TPR,FPR1,recall1,f1_score1,cnf_matrix
    
    def calculation3(y_tst,y_pred):
        cnf_matrix = confusion_matrix(y_tst,y_pred)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float);FN = FN.astype(float);
        TP = TP.astype(float);TN = TN.astype(float);  
        TPR = TP/(TP+FN);TNR = TN/(TN+FP) 
        PPV = TP/(TP+FP);NPV = TN/(TN+FN)
        FPR1 = FP/(FP+TN);FNR1 = FN/(TP+FN)
        FDR = FP/(TP+FP)
        ACC = (TP+TN)/(TP+FP+FN+TN)  
        Accuracy1=sum(ACC)/len(ACC)
        acc1 = np.mean(Accuracy1)
        recall1=TP/(TP+FN)
        Rel3 = np.max(recall1)
        recall1 = round(Rel3)
        precision1=TP/(TP+FP)
        prscon3 = np.max(precision1)
        precision1 = round(prscon3)
        fs3=((2*precision1*recall1)/(precision1+recall1))
        f1_score1 = round(fs3)
        print("Accuracy : ",(Accuracy1));
        print("Precision : ",(precision1));
        print("Recall : ",(recall1));print("F_Measure : ",(f1_score1))
        return Accuracy1,precision1,TPR,FPR1,recall1,f1_score1,cnf_matrix
    
    
    def calculation4(y_tst,y_pred):
        cnf_matrix = confusion_matrix(y_tst,y_pred)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float);FN = FN.astype(float);
        TP = TP.astype(float);TN = TN.astype(float);  
        TPR = TP/(TP+FN);TNR = TN/(TN+FP) 
        PPV = TP/(TP+FP);NPV = TN/(TN+FN)
        FPR1 = FP/(FP+TN);FNR1 = FN/(TP+FN)
        FDR = FP/(TP+FP)
        ACC = (TP+TN)/(TP+FP+FN+TN)  
        Accuracy1=sum(ACC)/len(ACC)
        acc1 = np.mean(Accuracy1)
        recall1=TP/(TP+FN)
        Rel4 = np.max(recall1)
        recall1 = round(Rel4)
        precision1=TP/(TP+FP)
        prscon4 = np.max(precision1)
        precision1 = round(prscon4)
        fs4=((2*precision1*recall1)/(precision1+recall1))
        f1_score1 = round(fs4)
        print("Accuracy : ",(Accuracy1));
        print("Precision : ",(precision1));
        print("Recall : ",(recall1));print("F_Measure : ",(f1_score1))
        return Accuracy1,precision1,TPR,FPR1,recall1,f1_score1,cnf_matrix
    
    def calculation5(y_tst,y_pred):
        cnf_matrix = confusion_matrix(y_tst,y_pred)
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float);FN = FN.astype(float);
        TP = TP.astype(float);TN = TN.astype(float);  
        TPR = TP/(TP+FN);TNR = TN/(TN+FP) 
        PPV = TP/(TP+FP);NPV = TN/(TN+FN)
        FPR1 = FP/(FP+TN);FNR1 = FN/(TP+FN)
        FDR = FP/(TP+FP)
        ACC = (TP+TN)/(TP+FP+FN+TN)  
        Accuracy1=sum(ACC)/len(ACC)
        acc1 = np.mean(Accuracy1)
        recall1=TP/(TP+FN)
        Rel5 = np.max(recall1)
        recall1 = round(Rel5)
        precision1=TP/(TP+FP)
        prscon5 = np.max(precision1)
        precision1 = round(prscon5)
        fs5=((2*precision1*recall1)/(precision1+recall1))
        f1_score1 = round(fs5)
        print("Accuracy : ",(Accuracy1));
        print("Precision : ",(precision1));
        print("Recall : ",(recall1));print("F_Measure : ",(f1_score1))
        return Accuracy1,precision1,TPR,FPR1,recall1,f1_score1,cnf_matrix
    #%%
    print("\noverAll Performance for Proposed:\n*******************\n")
    y_tst=predition[:,0];y_pred=predition[:,1]
    MCC1 = matthews_corrcoef(y_tst, y_pred)
    MCC1 = round(MCC1)
    Accuracy1,precision1,TPR,FPR,recall1,F_measure1,cnf_matrix1 = calculation(y_tst,y_pred)
    print("MCC:",MCC1)
    
    print("\noverAll Performance for EfficientNet B7:\n*******************\n")
    y_tst=predition[:,0];y_pred=predition[:,2]
    MCC2 = matthews_corrcoef(y_tst, y_pred)
    MCC2 = round(MCC2)
    Accuracy2,precision2,TPR,FPR,recall2,F_measure2,cnf_matrix2 = calculation1(y_tst,y_pred)
    print("MCC:",MCC2)
    
    print("\noverAll Performance for EfficientNet B5:\n*******************\n")
    y_tst=predition[:,0];y_pred=predition[:,3]
    MCC3 = matthews_corrcoef(y_tst, y_pred)
    MCC3 = round(MCC3)
    Accuracy3,precision3,TPR,FPR,recall3,F_measure3,cnf_matrix3 = calculation2(y_tst,y_pred)
    print("MCC:",MCC3)
    
    
    print("\noverAll Performance for EfficientNet B0:\n*******************\n")
    y_tst=predition[:,0];y_pred=predition[:,4]
    MCC4 = matthews_corrcoef(y_tst, y_pred)
    MCC4 = round(MCC4)
    Accuracy4,precision4,TPR,FPR,recall4,F_measure4,cnf_matrix4 = calculation3(y_tst,y_pred)
    print("MCC:",MCC4)
    
    
    print("\noverAll Performance for ResNet :\n*******************\n")
    y_tst=predition[:,0];y_pred=predition[:,5]
    MCC5 = matthews_corrcoef(y_tst, y_pred)
    MCC5 = round(MCC5)
    Accuracy5,precision5,TPR,FPR,recall5,F_measure5,cnf_matrix5 = calculation4(y_tst,y_pred)
    print("MCC:",MCC5)
    
    
    print("\noverAll Performance for DCNN :\n*******************\n")
    y_tst=predition[:,0];y_pred=predition[:,6]
    MCC6 = matthews_corrcoef(y_tst, y_pred)
    MCC6 = round(MCC6)
    Accuracy6,precision6,TPR,FPR,recall6,F_measure6,cnf_matrix6 = calculation5(y_tst,y_pred)
    print("MCC:",MCC6)
    
    
    #%%
    def plot_confusion_matrix(cm, classes,
        
                              normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
        classes=["DS-T-LG","DW-S","DW-T-O","IS-S","IS-T-DW","IS-T-LG","LG-S","LG-T-DS","LG-T-DW","LG-T-IS","LG-T-O","LG-T-SE"]
    
        plt.figure(figsize=(8,6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title("Confusion matrix",y=1,fontweight='bold',fontsize=20)
        tick_marks = np.arange(len(classes))
        
        plt.xticks(tick_marks,classes,fontname = "Times New Roman",fontsize=12,fontweight='bold',rotation=90)
        plt.yticks(tick_marks, classes,fontname = "Times New Roman",fontsize=12,fontweight='bold')
        plt.ylabel("Predicted  Classes",fontname = "Times New Roman",fontsize=12,fontweight='bold')
        plt.xlabel("Actual Classes",fontname = "Times New Roman",fontsize=12,fontweight='bold')
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
            color="white" if cm[i, j] > thresh else "black",                  
            horizontalalignment="center")
        plt.tight_layout()
        plt.show()
        plt.savefig("Result/Confson_matrx.png")
        #%%
        Class = np.unique(y_tst)
        plot_confusion_matrix(cnf_matrix1,Class )
        
    #%%
    colours = [[0.85,0.33,0.1] ,[0.93,0.69,0.13] ,[0.49,0.18,0.56] ,
               [0.29,0.58,0.26] ,"#800000",[0,0.4471,0.7412] ]
    
    
    Acc = [Accuracy6*100,Accuracy5*100,Accuracy4*100,Accuracy3*100,Accuracy2*100,Accuracy1*100]
    barlen = list(range(len(Acc))) ; col = colours[:len(Acc)]
    plt.figure();#plt.ylim([80,100])
    plt.bar(barlen,Acc, 0.2, color = col,edgecolor='black')
    plt.xticks(barlen,["DCNN","ResNet",'EN B0','EN B5',"EN B7","Proposed"],fontweight='bold',family='Times New Roman',fontsize=14)
    plt.ylabel("Accuracy (%)",fontweight='bold',fontsize=12)
    plt.savefig("Result/Accuracy.png")
    plt.show()
    
    
    
    
    Acc = [precision6*100,precision5*100,precision4*100,precision3*100,precision2*100,precision1*100]
    barlen = list(range(len(Acc))) ; col = colours[:len(Acc)]
    plt.figure();#plt.ylim([80,100])
    plt.bar(barlen,Acc, 0.2, color = col,edgecolor='black')
    plt.xticks(barlen,["DCNN","ResNet",'EN B0','EN B5',"EN B7","Proposed"],fontweight='bold',family='Times New Roman',fontsize=14)
    plt.ylabel("Precision (%)",fontweight='bold',fontsize=12)
    plt.savefig("Result/Precision.png")
    plt.show()
    
    Acc = [F_measure6*100,F_measure5*100,F_measure4*100,F_measure3*100,F_measure2*100,F_measure1*100]
    barlen = list(range(len(Acc))) ; col = colours[:len(Acc)]
    plt.figure();#plt.ylim([80,100])
    plt.bar(barlen,Acc, 0.2, color = col,edgecolor='black')
    plt.xticks(barlen,["DCNN","ResNet",'EN B0','EN B5',"EN B7","Proposed"],fontweight='bold',family='Times New Roman',fontsize=14)
    plt.ylabel("F Measure (%)",fontweight='bold',fontsize=12)
    plt.savefig("Result/F_Measure.png")
    plt.show()
    
    Acc = [recall6*100,recall5*100,recall4*100,recall3*100,recall2*100,recall1*100]
    barlen = list(range(len(Acc))) ; col = colours[:len(Acc)]
    plt.figure();#plt.ylim([80,100])
    plt.bar(barlen,Acc, 0.2, color = col,edgecolor='black')
    plt.xticks(barlen,["DCNN","ResNet",'EN B0','EN B5',"EN B7","Proposed"],fontweight='bold',family='Times New Roman',fontsize=14)
    plt.ylabel("Recall (%)",fontweight='bold',fontsize=12)
    plt.savefig("Result/Recall.png")
    plt.show()
    
        #%%
    # Define the font properties
    Train_Loss,Test_Loss,Train_Accuracy,Test_Accuracy=V.model_acc_loss(100)
    Train_Loss,val_Loss,Train_Accuracy,val_Accuracy=V.model_acc_loss(80)
    
    font = font_manager.FontProperties( weight='bold',style='normal',size=14)
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 14}
    # Plot the data
    plt.figure()
    plt.plot(Train_Accuracy, '-', label='Training')
    # plt.plot(Test_Accuracy, '-', label='Testing') 
    plt.plot(val_Accuracy, '-', label='Validation') 
    plt.grid(linewidth=0.5, axis='both', alpha=0.9)
    plt.grid(which='minor', linestyle=':', linewidth=0.5)
    # Set the x-label and y-label font properties
    plt.xlabel('Epoch', fontdict=font)
    plt.ylabel('Accuracy', fontdict=font)
    # Set the legend font properties
    plt.legend(prop=font)
    # Show the plot
    plt.show()
    plt.savefig("Result/TA.png")
    # Plot the data
    plt.figure()
    plt.plot(Train_Loss,'-', label='Training')
    plt.plot(val_Loss, '-', label='Validation') 
    plt.grid(linewidth=0.5, axis='both', alpha=0.9)
    plt.grid(which='minor', linestyle=':', linewidth=0.5)
    # Set the x-label and y-label font properties
    plt.xlabel('Epoch', fontdict=font)
    plt.ylabel('Loss', fontdict=font)
    # Set the legend font properties
    plt.legend(prop=font)
    # Show the plot
    plt.show()
    plt.savefig("Result/TL.png")
