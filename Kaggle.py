import PIL as pil
import os, os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split




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
	return test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu



def as_df(test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu) :

	test = test_normal + test_pneu
	train = train_normal + train_pneu
	val = val_normal + val_pneu
	
	liste = [[test_normal, test_pneu], [train_normal, train_pneu], [val_normal, val_pneu]]
	list_df = []
	print(liste[2])
	for normal, pneu in liste :
		cls = []
		for index in range(len(normal)) :
			cls.append("normal")
		for index in range(len(pneu)) :
			cls.append("pneumonia")

		list_df.append(pd.DataFrame({'class' : cls, 'image' : normal+pneu}))
		
		print(list_df)
		# print(len(list_df))
	return list_df
	

def data_visualisation(dataframes) :
	hist = []
	# camembert = []
	# labels = 'normal', 'pneumonia'
	#print(labels)
	for df in dataframes :
		hist.append(df['class'].hist())
		#camembert.append(df['class'].pie())
		#plt.show()

	# for plot in hist :
	# 	#plot_hist = plt.plot(plot)
	# 	plt.show()

	return hist







if __name__ == "__main__":
	#path_test = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/test"
	#path_train = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/train"
	#path_val = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/val"
	path_test = "/home/sdv/m2bi/rgoulancourt/Kaggle/data/archive/chest_xray/test"
	path_train = "/home/sdv/m2bi/rgoulancourt/Kaggle/data/archive/chest_xray/train"
	path_val = "/home/sdv/m2bi/rgoulancourt/Kaggle/data/archive/chest_xray/val"

	test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu = folder(path_test, path_train, path_val)
	all_df = as_df(test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu)
	graph = data_visualisation(all_df)
