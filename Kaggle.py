import PIL as pil
import os, os.path
import pandas as pd
import numpy as np
import cv2




def folder(path_test, path_train, path_val):
    test_normal = []
    test_pneu = []
    train_normal = []
    train_pneu = []
    val_normal = []
    val_pneu = []

    list_path = [path_test, path_train, path_val]
    for path in list_path:
        if os.path.exists(os.path.join(path, "NORMAL")):
            new_path = os.path.join(path, "NORMAL")
            for file in os.listdir(new_path):
                if str(new_path) == os.path.join(path_test, "NORMAL"):
                    test_normal.append(file)						
                elif str(new_path) == os.path.join(path_train, "NORMAL"):
                    train_normal.append(file)
                elif str(new_path) == os.path.join(path_val, "NORMAL"):
                    val_normal.append(file)
        if os.path.exists(os.path.join(path, "PNEUMONIA")):
            new_path = os.path.join(path, "PNEUMONIA")
            for file in os.listdir(new_path):
                if str(new_path) == os.path.join(path_test, "PNEUMONIA"):
                    test_pneu.append(file)		
                elif str(new_path) == os.path.join(path_train, "PNEUMONIA"):
                    train_pneu.append(file)
                elif str(new_path) == os.path.join(path_val, "PNEUMONIA"):
                    val_pneu.append(file)
    #print(val_normal, val_pneu)
    return test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu



def as_df(test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu) :
    print(len(test_normal), len(test_pneu), len(val_normal))

    #liste = [test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu]
    
    

    test = test_normal + test_pneu
    train = train_normal + train_pneu
    val = val_normal + val_pneu
    
    
    #print(test)

    #liste = [test, train, val]
    liste = [[test_normal, test_pneu], [train_normal, train_pneu], [val_normal, val_pneu]]
    list_df = []
    
    for normal, pneu in liste :
        cls = []
        #print(list)
        #joining_normal = list.join('_normal')
        #joining_pneu = list.join('_pneu')
        for index in range(len(normal)) :
            cls.append("normal")
        for index in range(len(pneu)) :
            cls.append("pneumonia")

        list_df.append(pd.DataFrame({'class' : cls, 'image' : normal+pneu}))
        
        print(list_df)
        return list_df
        #df_.join(list) = df_

    



    '''
    cls = []
    for i in range(len(test_normal)) :
        cls.append("normal")
    for i in range(len(test_pneu)) :
        cls.append("pneumonia")

    print("length of class {}".format(len(cls)))

    df_test = pd.DataFrame({'class' : cls, 'image' : test})
    print(df_test)
    #print("this is the test list : {}".format(test))

    #df_test = pd.DataFrame({'normal' : test_normal, 'pneumonial' : test_pneu})
    #print(df_test)
    '''
'''
    liste = [test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu]
    for l in liste :
        l = pd.Dataframe(l)
    print(test_normal)

    return test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu
'''

'''
import cv2
import os
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
'''
#join(path, s) 
#exists(path) 
#samefile(f1, f2)

if __name__ == "__main__":
    #path_test = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/test"
    #path_train = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/train"
    #path_val = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/val"
    path_test = "/home/sdv/m2bi/rgoulancourt/Kaggle/data/archive/chest_xray/test"
    path_train = "/home/sdv/m2bi/rgoulancourt/Kaggle/data/archive/chest_xray/train"
    path_val = "/home/sdv/m2bi/rgoulancourt/Kaggle/data/archive/chest_xray/val"

    test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu = folder(path_test, path_train, path_val)
    all_df = as_df(test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu)
    #listing = list(folder(path_test, path_train, path_val))
    #print(listing)



