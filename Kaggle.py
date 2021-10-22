import os, os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator




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


def data_splitting(dataframes) :
	df_train = dataframes[1]
	train_len = 0.80*len(df_train)

	shuffled_train =  df_train.sample(frac = 1).reset_index(drop = True)
	print("SHUFFELD DATA ----------------------", shuffled_train)
	train_data = shuffled_train.iloc[1:int(train_len), :]
	print("TRAIN DATA ----------------------------")
	print(train_data)
	val_data = shuffled_train.iloc[int(train_len):len(shuffled_train), :]
	print("VAL DATA ----------------------------")
	print(val_data)

	return train_data, val_data


def Data_Augmentation(train_df, val_df, list_df) :
	print("--------------------DATA AUGMENTATION ------------------------")
	test_df = list_df[2]
	# print(test_df)
	train_datagen = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.2, zoom_range = 0.2,
			horizontal_flip = True, rotation_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1)
	# print(type(train_datagen))
	# print(train_datagen)

	val_datagen = ImageDataGenerator(rescale = 1/ 255.)
	# print(type(val_datagen))
	# print(val_datagen)

	calibration_train = train_datagen.flow_from_dataframe(dataframe = train_df, x_col = 'image', y_col = 'class', class_mode = 'binary',
			batch_size = BATCH_SIZE, seed = SEED_SET)
	print(type(calibration_train), calibration_train)
	print(train_df)

	# calibration_train = train_datagen.flow_from_dataframe(x_col = 'image', y_col = 'class', class_mode = 'binary',
	# 		batch_size = BATCH_SIZE, seed = SEED_SET)


def CNN() :
	pass




if __name__ == "__main__":
	#path_test = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/test"
	#path_train = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/train"
	#path_val = "/Users/rgoulanc/Desktop/Rebecca/FAC/M2BI/KAGGLE/chest_xray/val"
	path_test = "/home/sdv/m2bi/rgoulancourt/Kaggle/data/archive/chest_xray/test"
	path_train = "/home/sdv/m2bi/rgoulancourt/Kaggle/data/archive/chest_xray/train"
	path_val = "/home/sdv/m2bi/rgoulancourt/Kaggle/data/archive/chest_xray/val"

	# Important parameters
	IMG_SIZE = 224
	BATCH_SIZE = 32
	SEED_SET = 42


	test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu = folder(path_test, path_train, path_val)
	all_df = as_df(test_normal, test_pneu, train_normal, train_pneu, val_normal, val_pneu)
	graph = data_visualisation(all_df)
	train_data, val_data = data_splitting(all_df)
	augmentation = Data_Augmentation(train_data, val_data, all_df)


