import PIL as pil
import os, os.path
import cv2


test_normal = []
test_pneu = []
train_normal = []
train_pneu = []
val_normal = []
val_pneu = []


def folder(path_test, path_train, path_val):
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
		elif os.path.exists(os.path.join(path, "PNEUMONIA")):
			new_path = os.path.join(path, "PNEUMONIA")
			for file in os.listdir(new_path):
				if str(new_path) == os.path.join(path_test, "PNEUMONIA"):
					test_pneu.append(file)		
				elif str(new_path) == os.path.join(path_train, "PNEUMONIA"):
					print(new_path)
					train_pneu.append(file)
				elif str(new_path) == os.path.join(path_val, "PNEUMONIA"):
					print(new_path)
					val_pneu.append(file)
	#print(val_normal, val_pneu)
	#return test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu


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
	path_test = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/test"
	path_train = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/train"
	path_val = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/val"

	listing = folder(path_test, path_train, path_val)
	#listing = list(folder(path_test, path_train, path_val))
	print(listing)







